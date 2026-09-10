ARG GIGAAM_MODEL=gigaam-v3-e2e-rnnt
ARG GIGAAM_PRECISION=fp32

# Этап 1: компилируем Go-сервис. В итоговый образ переносится только бинарник;
# исходники, компилятор и кэш Go остаются на этом этапе.
FROM golang:1.26.1-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=1 go build -trimpath -ldflags='-s -w' -o /out/gigaam ./cmd/gigaam

# Этап 2: скачиваем ONNX Runtime 1.29.0 для целевой архитектуры и проверяем SHA-256.
# В итоговый образ переносим библиотеки и лицензии, архив остаётся здесь.
# Alpine используется только для скачивания и распаковки; ONNX здесь не запускается.
FROM alpine:3.24.1 AS runtime-library
ARG TARGETARCH
RUN apk add --no-cache ca-certificates curl
RUN set -eu; \
    case "$TARGETARCH" in \
      amd64) ort_arch=x64; ort_sha=c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba ;; \
      arm64) ort_arch=aarch64; ort_sha=e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f ;; \
      *) echo 'Supported platforms: linux/amd64 and linux/arm64' >&2; exit 1 ;; \
    esac; \
    curl --fail --location --retry 3 --max-time 600 \
      "https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-${ort_arch}-1.29.0.tgz" -o /tmp/ort.tgz; \
    echo "$ort_sha  /tmp/ort.tgz" | sha256sum -c -; \
    mkdir /ort; tar -xzf /tmp/ort.tgz --strip-components=1 -C /ort

# Этап 3: скачиваем выбранную GigaAM и Silero VAD, проверяем размеры и SHA-256.
# Модели войдут в образ готовыми: при запуске сервис ничего не скачивает.
# Этот этап кэшируется независимо от изменений Go-кода.
FROM alpine:3.24.1 AS model-files
ARG GIGAAM_MODEL
ARG GIGAAM_PRECISION
RUN apk add --no-cache ca-certificates curl jq
COPY internal/models/manifest.json /tmp/manifest.json
COPY scripts/download-models.sh /usr/local/bin/download-models
RUN sh /usr/local/bin/download-models /tmp/manifest.json /models "$GIGAAM_MODEL" "$GIGAAM_PRECISION"

# Этап 4: собираем рабочий образ из бинарника, ONNX Runtime, моделей и FFmpeg.
# Сервис запускается от пользователя без root-прав; /work хранит временные файлы.
# Только этот итоговый образ используется при запуске контейнера.
# Debian предоставляет glibc для скачанной сборки ONNX Runtime и Go-бинарника с CGo.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg curl libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 gigaam \
    && useradd --uid 10001 --gid gigaam --no-create-home --shell /usr/sbin/nologin gigaam \
    && mkdir -p /models /work /usr/share/licenses/onnxruntime \
    && chown gigaam:gigaam /models /work && chmod 700 /models /work
COPY --from=runtime-library /ort/lib/ /usr/local/lib/
COPY --from=runtime-library /ort/LICENSE /ort/ThirdPartyNotices.txt /usr/share/licenses/onnxruntime/
RUN ldconfig
COPY --from=model-files --chown=10001:10001 /models/ /models/
COPY --from=build /out/gigaam /usr/local/bin/gigaam
ARG GIGAAM_MODEL
ARG GIGAAM_PRECISION
ENV GIGAAM_MODEL=${GIGAAM_MODEL} GIGAAM_PRECISION=${GIGAAM_PRECISION} GIGAAM_PORT=8080 GIGAAM_MODEL_DIR=/models GIGAAM_WORK_DIR=/work \
    GIGAAM_ORT_LIBRARY=/usr/local/lib/libonnxruntime.so
USER 10001:10001
WORKDIR /work
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail --silent --show-error "http://127.0.0.1:${GIGAAM_PORT}/healthz" || exit 1
ENTRYPOINT ["/usr/local/bin/gigaam"]

# Повторение тестов контейнеров

Сравнение расхода памяти, размеров и скорости: [повторные замеры после перезагрузки](../benchmarks/alpine-vs-debian-reboot-20260910-101540/report.md) и [исходная серия Debian / Alpine](../benchmarks/alpine-vs-debian/report.md). Измерение отдельного образа можно повторить так из корня репозитория, остановив другие нагрузки Docker:

Перед первой сборкой Alpine подготовьте библиотеку: `sh scripts/prepare-onnx-alpine.sh` ([подробнее](alpine-build.md)).

```bash
docker build -t gigaam-service:debian .
docker build -f Dockerfile.alpine -t gigaam-service:alpine .
python3 scripts/benchmark-container.py --image gigaam-service:alpine \
  --video /absolute/path/recording.mp4 --output bin/benchmarks/alpine
```

Скрипт создаёт временный контейнер на порту `18080` со случайным ключом, ждёт готовности модели, измеряет простой и четыре запроса (первый отдельно, затем три повторных), сохраняет метрики и удаляет только свой контейнер. `--port` и `--runs` меняют порт и число повторных запросов. Результаты включают текст распознавания, поэтому каталог с ними следует считать пользовательскими данными.

## Эксперимент с mimalloc

Для эксперимента с аллокатором есть `Dockerfile.alpine.mimalloc`. Он собирает mimalloc 3.5.1 под musl и добавляет его через `LD_PRELOAD` поверх уже собранного Alpine-сервиса. ONNX повторно компилировать не нужно. Замена действует на нативные выделения памяти сервиса и дочернего FFmpeg; Go продолжает использовать собственную кучу.

```bash
docker build -f Dockerfile.alpine \
  -t gigaam-service:alpine .
docker build -f Dockerfile.alpine.mimalloc --build-arg MIMALLOC_BUILD_JOBS=6 \
  -t gigaam-service:alpine-mimalloc .
```

По умолчанию mimalloc собирается в 6 параллельных задач; его тесты выполняются при сборке. Базовый образ можно задать через `ALPINE_SERVICE_IMAGE`. [Результаты сравнения mallocng и mimalloc](../benchmarks/musl-vs-mimalloc/report.md).

Повторить парное сравнение на macOS с Docker Desktop можно так:

```bash
python3 scripts/benchmark-pair.py \
  --baseline gigaam-service:alpine --candidate gigaam-service:alpine-mimalloc \
  --wav /absolute/path/reference.wav --video /absolute/path/recording.mp4 \
  --output bin/benchmarks/allocators
```

Каталог результата должен быть новым. Скрипт по очереди запускает три пары контейнеров на каждый вход, меняет порядок образов, записывает своп и давление памяти macOS. Каждый контейнер выполняет полный прогревочный и один измеряемый запрос. Запущенный `gigaam-docker-asr-1` временно останавливается и восстанавливается по завершении; другие работающие контейнеры приводят к отмене теста.


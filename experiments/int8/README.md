# Выбор FP32 / INT8

Сервис поддерживает готовые FP32- и INT8-графы RNNT и CTC из [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx/tree/322c3b29492673eb7d0b434bfa9dfb8653e34d02), ревизия `322c3b29492673eb7d0b434bfa9dfb8653e34d02`. Самостоятельная квантизация не выполняется. VAD, YAML и словари прежние, VAD всегда FP32.

Основной [манифест](../../internal/models/manifest.json) содержит оба варианта с отдельными размерами и SHA-256. Графы лежат в разных каталогах `fp32` и `int8` под стандартными именами, ожидаемыми движком. Go-бинарник общий; выбранный вариант определяется `GIGAAM_PRECISION` при старте. Неизвестное значение отклоняется, отсутствие файлов выбранной модели даёт `model_unavailable`; автоматического перехода на другой precision нет.

## Docker Compose

Укажите в `.env`:

```dotenv
GIGAAM_MODEL=gigaam-v3-e2e-rnnt
GIGAAM_PRECISION=int8
```

Затем выполните из корня репозитория команду нужного варианта.

Alpine (сначала один раз [подготовьте ONNX](../../docs/alpine-build.md)):

```bash
docker compose -f compose.yaml -f compose.alpine.yaml up -d --build
```

Debian:

```bash
docker compose -f compose.yaml up -d --build
```

Для возврата к FP32 установите `GIGAAM_PRECISION=fp32` и повторите команду. Образ содержит только выбранный комплект, поэтому переключение обновляет слой моделей и настройку контейнера. Go и ONNX повторно компилировать не нужно после первоначальной сборки версии сервиса, поддерживающей оба precision.

При ручной сборке доступны те же аргументы:

```bash
docker build -f Dockerfile.alpine \
  --build-arg GIGAAM_PRECISION=int8 -t gigaam-service:alpine-int8 .
```

API не меняется: поле `model` — `gigaam-v3-e2e-rnnt` или `gigaam-v3-e2e-ctc`. Precision виден в логе `model ready`.

## Локальный запуск или внешний каталог моделей

Если нужны оба комплекта без пересборки образа при переключении, скачайте их заранее в отдельные подкаталоги одного корня (нужны curl, jq, sha256sum):

```bash
sh scripts/download-models.sh internal/models/manifest.json /path/to/models gigaam-v3-e2e-rnnt fp32
sh scripts/download-models.sh internal/models/manifest.json /path/to/models gigaam-v3-e2e-rnnt int8
```

Второй вызов не перезаписывает FP32-графы; общий VAD скачивается повторно с тем же SHA-256. Укажите этот корень в `GIGAAM_MODEL_DIR`, выберите `GIGAAM_PRECISION` и перезапустите процесс. Для контейнера каталог можно смонтировать в `/models` только для чтения, с доступом UID/GID 10001:10001. Без установленных файлов сменить precision одной переменной не получится.

## Результаты и ограничения

[Сравнение RNNT FP32 / INT8](../../benchmarks/fp32-vs-int8/report.md): в тесте на Apple M4 INT8 примерно в 1,6 раза быстрее, ОЗУ после загрузки — 263 против 895 МиБ. Результаты относятся к RNNT; на CTC эти показатели не переносились.

На одной лекции INT8 отличается от FP32 на 20–22 словесные правки из 1309 слов. Есть транслитерация терминов, но также изменения чисел (`4 095` → `495`, `510` → `50`/`500` в MP4) и слов. Это не только формат записи. Без ручного эталона нельзя определить абсолютную точность или процент её падения; FP32 тоже допускает ошибки. Поэтому FP32 остаётся по умолчанию.

Первый эксперимент использовал отдельный бинарник с [экспериментальным манифестом](manifest.json). Этот манифест и [снимок Dockerfile](../../benchmarks/fp32-vs-int8/Dockerfile.experiment) сохранены для истории замеров на исходном коде ревизии `b011ba5`. Для текущего запуска они не нужны: используйте штатные Dockerfile и `GIGAAM_PRECISION`.

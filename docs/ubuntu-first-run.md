# Первый запуск на Ubuntu: Pentium Gold 6405U

Инструкция для HP 260 G4 Desktop Mini PC с Pentium Gold 6405U, 8 ГБ ОЗУ и SSD 128 ГБ. При подключении проверены Ubuntu 24.04.2, два физических ядра / четыре потока и раздел `/` размером около 55 ГиБ. Ранее предполагался G6405, но `lscpu` показал 6405U. Пароли и закрытые ключи в чат передавать не нужно.

Текущий экземпляр впоследствии переключён на Alpine: [результаты и команды просмотра логов](../benchmarks/hp-6405u-alpine/report.md). Для дальнейших пересборок используйте [инструкцию Alpine](alpine-build.md). Прямой доступ из домашней сети теперь доступен по [192.168.0.36:8080](hp-lan.md). Ниже сохранена инструкция первоначального запуска Debian.

## Проверить мини-ПК

Выполнить именно на Ubuntu:

```bash
cat /etc/os-release
uname -m
lscpu
free -h
df -h / /var/lib
id -un
hostname -I
command -v docker
```

Если Docker установлен, дополнительно:

```bash
sudo docker version
sudo docker compose version
sudo docker info --format '{{.DockerRootDir}}'
sudo docker system df
```

Проверить свободное место на файловой системе DockerRootDir. Исходные файлы, сборочный кэш, образы и временные файлы сервиса занимают место отдельно. При стандартных лимитах перед **каждой транскрибацией** требуется около **20,5 ГиБ свободного места** в `/work`: максимальный файл 20 ГиБ + резерв 512 МиБ; для WAV дополнительно нужно около 115,2 МБ на час записи. Проверка не зависит от фактического размера отправляемого файла. Это требование к свободному месту после сборки, а не к общему размеру SSD.

Если места меньше, для первого небольшого файла можно отдельно уменьшить `GIGAAM_MAX_FILE_BYTES` в `.env`, например до `1073741824` (1 ГиБ); при остальных стандартных настройках проверка потребует около 1,5 ГиБ. Общий лимит проекта остаётся 20 ГиБ.

## Docker и исходники

Если Docker отсутствует, после проверки версии Ubuntu установить Docker Engine, Buildx и Compose plugin из [официального apt-репозитория Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). Docker Desktop для этого запуска не требуется. Существующую установку Docker сначала проверить, автоматически удалять пакеты или данные не нужно.

Для первого запуска выбран стандартный Debian Dockerfile: он скачивает готовый ONNX Runtime для amd64 и не компилирует C++ на мини-ПК. FP32 RNNT остаётся начальным вариантом; INT8 можно сравнить после успешного запуска.

6405U поддерживает SSE4.1/4.2, но не AVX/AVX2/VNNI — [характеристики Intel](https://www.intel.com/content/www/us/en/products/sku/197888/intel-pentium-gold-6405u-processor-2m-cache-2-40-ghz/specifications.html). В [MLAS ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/blob/v1.29.0/onnxruntime/core/mlas/lib/platform.cpp) AVX-ядра выбираются после проверки возможностей CPU. На этом мини-ПК загрузка модели и два реальных запроса уже прошли: [результаты запуска](../benchmarks/hp-6405u-first-run/report.md). Это подтверждает работу данной CPU-сборки с FP32 RNNT. Замеры M4 сюда не переносятся.

На момент подготовки выбор FP32/INT8 присутствует в локальных незакоммиченных изменениях. Обычный `git clone` не перенесёт эти изменения. Для запуска именно текущего кода подготовлен локальный архив `bin/ubuntu-prep/gigaam-source.tar.gz`: Dockerfile, Compose, Go-код, скрипты и пример конфигурации. В нём нет `.env`, ключей, моделей, записей и Git-истории. После получения SSH-доступа перенести архив и распаковать в новую отдельную папку проекта на мини-ПК. Архив — снимок на момент подготовки; после изменения кода его нужно обновить.

## Запуск из папки проекта на Ubuntu

Подготовить новый `.env` из `.env.example`, если его ещё нет. Сгенерировать отдельный ключ локально через `openssl rand -hex 32` и записать его в `GIGAAM_API_KEY`. Права файла — `chmod 600 .env`. Начальные настройки:

```dotenv
GIGAAM_MODEL=gigaam-v3-e2e-rnnt
GIGAAM_PRECISION=fp32
GIGAAM_CONCURRENCY=1
GIGAAM_ONNX_THREADS=2
```

Два потока — начальное значение для двух физических ядер, а не измеренный оптимум. Остальные настройки взять из примера.

```bash
sudo docker compose -f compose.yaml up -d --build
sudo docker compose -f compose.yaml logs --tail=80 asr
sudo docker compose -f compose.yaml exec asr df -h /work
```

Ожидается лог `model ready`. `GET /healthz` проверяет только работу HTTP, поэтому дополнительно нужен авторизованный `GET /v1/models`. Команды ниже выполняются в Bash на мини-ПК из папки проекта; ключ читается из созданного нами `.env` и передаётся curl через стандартный ввод:

```bash
gigaam_key=$(sed -n 's/^GIGAAM_API_KEY=//p' .env)
printf 'header = "Authorization: Bearer %s"\n' "$gigaam_key" |
  curl --silent --show-error --fail-with-body --config - \
    http://127.0.0.1:8080/v1/models
```

Первый запрос — небольшая запись с речью, примерно 30–60 секунд. Заменить путь на реальный файл:

```bash
printf 'header = "Authorization: Bearer %s"\n' "$gigaam_key" |
  curl --silent --show-error --fail-with-body --max-time 600 --config - \
    http://127.0.0.1:8080/v1/audio/transcriptions \
    -F 'file=@/path/to/sample.mp4' \
    -F 'model=gigaam-v3-e2e-rnnt' \
    -F 'response_format=text' -o transcript.txt
unset gigaam_key
sudo docker compose -f compose.yaml logs --tail=40 asr
sudo docker stats --no-stream
```

Успех: `/v1/models` возвращает 200, транскрибация возвращает текст, итоговый лог имеет `stage=complete`, `code=ok`. Затем можно отправлять обычную лекцию и измерять время. После ошибки curl проверить HTTP-ответ и логи: файл `transcript.txt` может содержать описание ошибки.

Порт в Compose привязан к loopback мини-ПК. Для обращения с Mac использовать SSH-туннель на свободный локальный порт, например 18080 → 127.0.0.1:8080 на мини-ПК; конкретную команду составить после получения SSH-адреса. Публичный порт для первого запуска открывать не требуется.

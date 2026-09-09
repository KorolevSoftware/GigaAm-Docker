# Модели и воспроизводимость

Рабочий сервис использует готовые экспорты [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx/tree/322c3b29492673eb7d0b434bfa9dfb8653e34d02). Ревизия `322c3b29492673eb7d0b434bfa9dfb8653e34d02` зафиксирована; используются только FP32-файлы. Их размеры и SHA-256 из LFS-метаданных включены в [manifest.json](../internal/models/manifest.json). Для небольших YAML/словарей SHA-256 рассчитан по скачанным байтам.

Silero VAD зафиксирован на `867c2aa692646a1f1de3e94a15c9dd9f614c0acb`, файл `src/silero_vad/data/silero_vad.onnx`. Его SHA-256 также в манифесте.

Go — 1.26.1. Нативная ONNX Runtime — [1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0). Go-обвязка `yalue/onnxruntime_go` остаётся на v1.22.0 и использует C API v22: её номер версии не обозначает версию динамически загруженного рантайма. Dockerfile проверяет SHA-256 официальных Linux-архивов amd64/arm64. ONNX Runtime и её уведомления о лицензиях включаются в образ. Фактическая версия библиотеки выводится при инициализации; телеметрия отключается до и после создания ORT environment.

Архивы релиза v1.29.0 и SHA-256 из GitHub Release API:

| Архив | SHA-256 |
|---|---|
| `onnxruntime-linux-x64-1.29.0.tgz` | `c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba` |
| `onnxruntime-linux-aarch64-1.29.0.tgz` | `e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f` |
| `onnxruntime-osx-arm64-1.29.0.tgz` | `d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92` |

Реализация признаков повторяет `FeatureExtractor`: 16 kHz, 64 HTK mel-фильтра без нормализации, periodic Hann, окно/FFT 320, шаг 160, `center=false`, power=2, `log(clamp(x, 1e-9, 1e9))`. Окно и mel-коэффициенты в `feature_coefficients.json` сгенерированы `scripts/feature_coefficients.py` на закреплённом PyTorch/Torchaudio 2.7.1, чтобы сохранить округление FP32. CTC удаляет blank и последовательные повторы. RNNT использует predictor LSTM 1×320, greedy argmax и максимум 3 токена на кадр; состояние predictor меняется только при эмиссии. Токены SentencePiece восстанавливаются по экспортированному словарю; `▁` превращается в пробел, начальный dummy prefix удаляется.

Эталонные функции взяты из официального [GigaAM, ревизия 7447938d791c4f3e643386ee22c33777004293a5](https://github.com/salute-developers/GigaAM/tree/7447938d791c4f3e643386ee22c33777004293a5). В выбранном готовом экспорте CTC один выход `log_probs`; длина рассчитывается по официальному subsampling factor 4. RNNT имеет отдельные encoder, decoder, joint. `scripts/reference.py` вызывает официальные функции подготовки признаков и CTC/RNNT-декодирования и проверяет словарь против настоящего SentencePiece-токенизатора. Изменяемые default-ветки при старте сервиса не запрашиваются.

Модели скачиваются скриптом `scripts/download-models.sh` на этапе Docker `model-files`. Сборка проверяет размер и SHA-256 каждого файла и включает выбранный ASR-комплект вместе с Silero VAD в образ. Рабочий Go-сервис только находит установленные файлы; автоматической загрузки и восстановления повреждённого кэша при старте нет.

## Подготовка проверки релиза

Пользователю сервиса экспорт не нужен. Следующий процесс предназначен разработчику релиза, на Python 3.12; он создаёт эталонные файлы с текстом только из публичных или разрешённых тестовых записей.

```bash
python3.12 -m venv .venv-reference
. .venv-reference/bin/activate
pip install -r scripts/requirements-reference.txt

git clone https://github.com/salute-developers/GigaAM.git /tmp/gigaam-official
git -C /tmp/gigaam-official checkout 7447938d791c4f3e643386ee22c33777004293a5
mkdir -p /tmp/gigaam-tokenizers
for model in v3_e2e_ctc v3_e2e_rnnt; do
  curl --fail --location \
    "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/${model}_tokenizer.model" \
    -o "/tmp/gigaam-tokenizers/${model}_tokenizer.model"
done
# Скрипт сверяет tokenizer с хешем именно выбранного экспорта.
python scripts/reference.py --official /tmp/gigaam-official \
  --models /path/to/model-cache --tokenizers /tmp/gigaam-tokenizers \
  --output testdata/golden public-short-16k-mono.wav

GIGAAM_TEST_MODELS=/path/to/model-cache \
GIGAAM_TEST_ORT=/path/to/libonnxruntime.so \
GIGAAM_TEST_GOLDEN="$PWD/testdata/golden" make integration
```

Признаки сравниваются с абсолютным допуском 0,002 в log-mel; текст — строго, включая пунктуацию. Для почти нулевых mel-энергий (< 1e-6) допустим log-разрыв до 0,02 только при абсолютной разнице линейной энергии ≤ 1e-9. Это учитывает потерю относительной точности float32 FFT при взаимном сокращении отсчётов: на синтетическом chirp наблюдались log-разница 0,00916 и всего 1,65e-10 в линейной энергии. На речевых эталонах применяется основной допуск. В CI также включён небольшой независимый Torch-эталон `testdata/features.json`. Нельзя менять golden-файлы по выводу Go, чтобы скрыть расхождения. Нативный VAD можно отдельно проверить переменными `GIGAAM_TEST_ORT` и `GIGAAM_TEST_VAD` через `go test -v ./internal/inference -run TestNativeVAD`. При заданных `GIGAAM_TEST_MODELS` и `GIGAAM_TEST_ORT` тест `TestNativeCancellation` измеряет задержку завершения нативного encoder Run после отмены.

## Обновление комплектов

1. Выбрать конкретную ревизию исходной модели и exporter-кода. Готовый использованный репозиторий не публикует точную версию окружения первоначального экспортёра; это ограничение происхождения артефактов, а не обещание воспроизвести их байт-в-байт.
2. При собственном экспорте закрепить checkout GigaAM, версии Python/PyTorch/ONNX/ORT, вызвать `model.to_onnx(dir_path=...)` для обеих E2E-моделей; выгрузить `id_to_str(i)` для всех токенов и последним `<blk>`. Не подменять FP32 графы INT8 или PyTorch checkpoint.
3. Проверить имена/формы входов и выходов, короткую речь, тишину, максимальное окно и золотые эталоны. При смене API экспорта обновить адаптер вместе с манифестом.
4. Опубликовать графы, словари, YAML и лицензии по неизменяемым HTTPS-адресам; рассчитать размеры и SHA-256 по окончательным байтам. Обновить ревизию и файлы `manifest.json`, закрепить exporter в документации релиза.
5. Пройти [приёмку](acceptance.md), затем выпускать образ. Ошибки сравнения запрещают считать новый комплект проверенным.

Удаление старых ревизий выполняет оператор. Сервис самостоятельно их не удаляет. Исходные пользовательские записи никогда не используются загрузчиком моделей и не передаются авторам моделей.

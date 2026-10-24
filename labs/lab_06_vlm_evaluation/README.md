# Лабораторная работа №6: Построение и оценка VLM-пайплайна

## Материалы

- [Студенческий ноутбук](student.ipynb) — теория, постановки задач и ячейки для самостоятельной реализации.

## Вычислительный профиль

По умолчанию SmolVLM2-2.2B-Instruct: на CUDA — NF4 inference, без CUDA полный режим честно останавливается. Сокращённый режим проверяет код, но CPU-инференс этой модели может быть медленным.

Для запуска не нужны файлы пользователя: `open_eval_data.py` загружает 20 опубликованных примеров (по 5 captioning, open VQA, document QA/OCR и structured count JSON) из PixMo-Cap, PixMo-AskModelAnything, PixMo-Docs и PixMo-Count. Метаданные датасетов: [Cap](https://huggingface.co/datasets/allenai/pixmo-cap), [Ask](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything), [Docs](https://huggingface.co/datasets/allenai/pixmo-docs), [Count](https://huggingface.co/datasets/allenai/pixmo-count); в карточках указана **ODC-BY-1.0**, но права исходных фотографий из сторонних URL могут отличаться. PixMo-Docs — сгенерированные документы с опубликованными вопросами/ответами, это document QA, а не эталон посимвольной транскрипции. Скачивание требует интернета и Pillow; полноценный inference требует PyTorch, Transformers, Accelerate, bitsandbytes и GPU с CUDA. При недоступности исходных URL загрузчик выдаёт явную ошибку, а не подставляет изображения.

Начните с `FAST_DEV_RUN=True`: режим использует по одной записи каждого источника только для проверки кода. Для отчёта выполните `False`. `artifacts/manual_review.jsonl` создаётся ноутбуком и заполняется человеком после просмотра изображений; неполная форма не прерывает Run All, но `error_report.json` отмечает `complete=false` и не является итоговой оценкой. Творческий бонус запускается отдельно через `RUN_BONUS=True`. Результаты и контрольные точки сохраняйте в `artifacts/` и переносите в постоянное хранилище до завершения сессии Colab.

## Навигация

- [Все лабораторные работы](../README.md)
- [Программа курса и требования к сдаче](../../README.md)

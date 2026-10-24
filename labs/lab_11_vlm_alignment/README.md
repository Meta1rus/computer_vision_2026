# Лабораторная работа 11. Alignment VLM по предпочтениям

[Студенческий notebook](student.ipynb)

## Входные данные и вычисления

Предоставьте data/lab11/preferences.jsonl: id, split (train или blind_test), image, prompt, chosen, rejected, error_type. Используется адаптер lab 9, CUDA и мультимодальный TRL. Слепой аудит заполняется человеком.

`FAST_DEV_RUN=True` — только smoke; для сдачи установите `False`, сохраните `artifacts/` в постоянном хранилище, зафиксируйте версии модели и библиотек. Данные и метрики не подменяются фиктивными примерами. CPU не является заменой GPU для VLM-обучения.

## Проверяемые артефакты

`artifacts/`: preferences.jsonl, pair_audit.json, aligned_adapter/, blind_predictions.jsonl, blind_judgements.jsonl, blind_report.json. Приложите сведения об источнике данных, лицензии, конфигурации, времени и памяти, оценку на независимом test и разбор ошибок. Если обязательные входные данные отсутствуют, notebook завершается явной ошибкой.

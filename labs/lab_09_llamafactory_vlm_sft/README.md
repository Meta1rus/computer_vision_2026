# Лабораторная работа 9. Fine-tuning VLM с помощью LLaMA-Factory

[Студенческий notebook](student.ipynb)

## Входные данные и вычисления

Предоставьте data/lab9/manifest.jsonl: id, split (train, validation, id_test, ood_test, general_test), image, question, answer, label_type; проверьте лицензию и отсутствие пересечений. CUDA и установленный LLaMA-Factory обязательны.

`FAST_DEV_RUN=True` — только smoke; для сдачи установите `False`, сохраните `artifacts/` в постоянном хранилище, зафиксируйте версии модели и библиотек. Данные и метрики не подменяются фиктивными примерами. CPU не является заменой GPU для VLM-обучения.

## Проверяемые артефакты

`artifacts/`: sft.yaml, train.log, lab9_adapter/, predictions.jsonl, evaluation.json. Приложите сведения об источнике данных, лицензии, конфигурации, времени и памяти, оценку на независимом test и разбор ошибок. Если обязательные входные данные отсутствуют, notebook завершается явной ошибкой.

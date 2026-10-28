# Лабораторная работа №10. Эффективный fine-tuning VLM с помощью Unsloth

**Дата:** 28.10.2026
**Ноутбук:** [student.ipynb](student.ipynb) — выполнен полностью, `FAST_DEV_RUN = False`, ядро окружения `.venv-unsloth`

## Постановка

Повторить SFT Qwen2.5-VL-3B из работы 9 в Unsloth при том же бюджете: 100 шагов × accumulation 16 на тех же данных. Сравнить LoRA ранга 8 и 16 по времени, памяти и качеству, а затем — с LLaMA-Factory при одинаковых условиях.

## Данные и модель

- Та же подвыборка ScienceQA, что в работе 9: ID и хеши совпадают, по 64 вопроса в ID, OOD и general.
- Unsloth 2026.10.8 на Apple Silicon работает через MLX 0.32: 4-bit квантование при загрузке, LoRA на всех линейных слоях LLM, vision tower заморожен; lr 1e-4 cosine, seed 42.
- Loss считается только по ответу ассистента (`train_on_responses_only`), как в LLaMA-Factory. В MLX-бэкенде опция `completion_only_loss` для VLM не действует, поэтому маска ставится явно.

## Результаты

| | время обучения, с | пик памяти, ГБ | ID | OOD | general |
|---|---:|---:|---:|---:|---:|
| LLaMA-Factory, rank 8 (работа 9) | — | — | 0,891 | **0,719** | 0,875 |
| Unsloth, rank 8 | 648 | 5,89 | 0,906 | 0,703 | 0,891 |
| Unsloth, rank 16 | 659 | 6,08 | **0,953** | 0,703 | **0,906** |

**Скорость при равных условиях** (`speed_benchmark.json`). Свободный GPU, одни и те же 96 примеров (6 шагов × 16), одинаковые LoRA:

| стек | шаг, с | пик памяти, ГБ |
|---|---:|---:|
| LLaMA-Factory (PyTorch MPS, NF4) | 34,0 | 3,3 |
| Unsloth (MLX) | **7,0** | 5,9 |

Unsloth быстрее в 4,8 раза, но берёт больше памяти: квантование MLX оставляет часть модулей в 16 бит.

**Бонус** (`bonus.md`): accumulation 8 при 200 шагах против 16 при 100 шагах (те же 1 600 примеров) — EM на validation 0,812 против 0,797, время ×1,02. Гипотеза о равноценности подтверждена.

## Анализ ошибок

- **ID, rank 16** — 3 ошибки: магниты и два вопроса «погода или климат» без текста отрывка (загрузчик курса не передаёт поле hint ScienceQA).
- **OOD** — генетика и планирование эксперимента, 13–14 из 19 ошибок общие с LLaMA-Factory.
- **General** — карты и обмен. Все ошибки общие с LLaMA-Factory, нужных знаний в train нет.

## Выводы

1. При равных данных, бюджете и обучаемых модулях Unsloth на M3 Max учит в 4,8 раза быстрее LLaMA-Factory с тем же качеством.
2. Rank 16 почти не стоит дополнительного времени и памяти (+2 %, +0,2 ГБ), даёт +3 вопроса на ID и ничего на OOD.
3. На Apple Silicon маску ответа для VLM в Unsloth нужно задавать явно, а пиковую память — мерить по прогону: общий пик процесса включает модели предыдущих ячеек.

## Артефакты

`artifacts/`:
- `protocol.json`, `benchmark.json`, `loss_comparison.png`;
- `predictions_r8_a16.jsonl`, `predictions_r16_a16.jsonl`, `quality.json`;
- `speed_benchmark.json`, `speed/` (конфигурация, лог и отметки времени замера LLaMA-Factory);
- `comparison.json`, `bonus.md`, `bonus_results.json`;
- `unsloth_*/training.json`.

Веса адаптеров и копии токенизатора хранятся локально.

## Воспроизведение

```bash
cd labs/lab_10_unsloth_efficiency
../../.venv-unsloth/bin/jupyter nbconvert --to notebook --execute --inplace student.ipynb
```

Нужны данные и артефакты работы 9, для замера скорости — окружение `.venv-lf`. Около 45 минут на M3 Max при свободном GPU.

**Вычислительные ресурсы:** MacBook Pro (Apple M3 Max, 40-core GPU, 48 ГБ); Unsloth 2026.10.8, MLX 0.32.4, mlx-vlm 0.7.9, transformers 5.17; для сравнения — LLaMA-Factory 0.9.5 на PyTorch 2.14.1 MPS.

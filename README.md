# Современное компьютерное зрение — лабораторные работы

Решения 18 лабораторных работ курса «Современное компьютерное зрение» (магистратура ПМИ, 2026).
Шаблоны заданий взяты из [fmi-ml-courses/computer_vision_2026_students](https://github.com/fmi-ml-courses/computer_vision_2026_students).

Все эксперименты выполняются локально на MacBook Pro (Apple M3 Max, 40-core GPU, 48 ГБ unified memory) через PyTorch MPS, без Colab.

## Работы

| № | Тема | Статус |
|---:|---|---|
| 1 | [Transfer learning и fine-tuning визуального backbone](labs/lab_01_transfer_learning) | шаблон |
| 2 | [Детекция и сегментация объектов](labs/lab_02_detection_segmentation) | шаблон |
| 3 | [Foundation models для сегментации: SAM](labs/lab_03_sam) | шаблон |
| 4 | [Self-supervised learning визуальных представлений](labs/lab_04_self_supervised) | шаблон |
| 5 | [CLIP: zero-shot-классификация и мультимодальный поиск](labs/lab_05_clip_retrieval) | шаблон |
| 6 | [Построение и оценка VLM-пайплайна](labs/lab_06_vlm_evaluation) | шаблон |
| 7 | [Автоматическая разметка изображений с помощью VLM](labs/lab_07_vlm_annotation) | шаблон |
| 8 | [Active learning и human-in-the-loop](labs/lab_08_active_learning) | шаблон |
| 9 | [Fine-tuning VLM с помощью LLaMA-Factory](labs/lab_09_llamafactory_vlm_sft) | шаблон |
| 10 | [Эффективный fine-tuning VLM с помощью Unsloth](labs/lab_10_unsloth_efficiency) | шаблон |
| 11 | [Alignment VLM по предпочтениям](labs/lab_11_vlm_alignment) | шаблон |
| 12 | [Генерация синтетического датасета для VLM](labs/lab_12_synthetic_vlm_data) | шаблон |
| 13 | [Fine-tuning diffusion model](labs/lab_13_diffusion_lora) | шаблон |
| 14 | [Управляемая генерация изображений](labs/lab_14_controlled_generation) | шаблон |
| 15 | [Flow matching и rectified flow](labs/lab_15_flow_matching) | шаблон |
| 16 | [Визуальные токенизаторы и autoregressive generation](labs/lab_16_visual_tokenization) | шаблон |
| 17 | [Diffusion model с нуля и classifier-free guidance](labs/lab_17_ddpm_cfg) | шаблон |
| 18 | [Быстрые samplers и distillation](labs/lab_18_fast_samplers_distillation) | шаблон |

## Окружение

- macOS 26.6, Python 3.11, PyTorch 2.14.1 с бэкендом MPS;
- зафиксированные версии — в [`requirements.txt`](requirements.txt).

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Запуск

Ноутбук каждой работы выполняется из её каталога:

```bash
cd labs/lab_01_transfer_learning
jupyter nbconvert --to notebook --execute --inplace student.ipynb
```

Датасеты скачиваются в общий каталог `data/` в корне репозитория (в git не попадает), метрики и рисунки сохраняются в `labs/<работа>/artifacts/`. Веса моделей и чекпоинты остаются локально и в репозиторий не добавляются.

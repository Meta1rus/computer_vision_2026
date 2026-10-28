# Современное компьютерное зрение — лабораторные работы

Решения 18 лабораторных работ курса «Современное компьютерное зрение» (магистратура ПМИ, 2026).
Шаблоны заданий взяты из [fmi-ml-courses/computer_vision_2026_students](https://github.com/fmi-ml-courses/computer_vision_2026_students).

Все эксперименты выполняются локально на MacBook Pro (Apple M3 Max, 40-core GPU, 48 ГБ unified memory) через PyTorch MPS, без Colab.

## Работы

| № | Тема | Статус |
|---:|---|---|
| 1 | [Transfer learning и fine-tuning визуального backbone](labs/lab_01_transfer_learning) | выполнена: EfficientNet-B0, test acc 0,944 |
| 2 | [Детекция и сегментация объектов](labs/lab_02_detection_segmentation) | выполнена: LRASPP, test mIoU 0,804 |
| 3 | [Foundation models для сегментации: SAM](labs/lab_03_sam) | выполнена: SAM с рамкой — IoU 0,967 (LRASPP 0,974), точка — 0,815 |
| 4 | [Self-supervised learning визуальных представлений](labs/lab_04_self_supervised) | выполнена: linear probe DINOv2, test acc 0,959 |
| 5 | [CLIP: zero-shot-классификация и мультимодальный поиск](labs/lab_05_clip_retrieval) | выполнена: zero-shot 0,937 → 0,957, mAP поиска 0,969 |
| 6 | [Построение и оценка VLM-пайплайна](labs/lab_06_vlm_evaluation) | выполнена: SmolVLM2 NF4 — документы 4/5 верно, подсчёт 1/5; абляция 8 вариантов |
| 7 | [Автоматическая разметка изображений с помощью VLM](labs/lab_07_vlm_annotation) | выполнена: Qwen2.5-VL-3B — accuracy 0,935, подписи верны в 80 %, downstream −3,5 п. п. |
| 8 | [Active learning и human-in-the-loop](labs/lab_08_active_learning) | выполнена: k-center AUC 0,899 против 0,850 у random |
| 9 | [Fine-tuning VLM с помощью LLaMA-Factory](labs/lab_09_llamafactory_vlm_sft) | выполнена: QLoRA SFT — ID 0,828 → 0,891, OOD 0,578 → 0,719, general без изменений |
| 10 | [Эффективный fine-tuning VLM с помощью Unsloth](labs/lab_10_unsloth_efficiency) | выполнена: Unsloth (MLX) в 4,8 раза быстрее LLaMA-Factory при том же качестве; rank 16 — ID 0,953 |
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

Работа 9 вызывает LLaMA-Factory из отдельного окружения — у него жёстче закреплены версии transformers/peft/trl: [`requirements-llamafactory.txt`](requirements-llamafactory.txt), CLI — `.venv-lf/bin/llamafactory-cli` (можно переопределить переменной `LLAMAFACTORY_CLI`).

Работа 10 выполняется в отдельном окружении Unsloth, который на Apple Silicon работает через MLX: [`requirements-unsloth.txt`](requirements-unsloth.txt); ноутбук запускается ядром этого окружения (`.venv-unsloth`).

## Запуск

Ноутбук каждой работы выполняется из её каталога:

```bash
cd labs/lab_01_transfer_learning
jupyter nbconvert --to notebook --execute --inplace student.ipynb
```

Датасеты скачиваются в общий каталог `data/` в корне репозитория (в git не попадает), метрики и рисунки сохраняются в `labs/<работа>/artifacts/`. Веса моделей и чекпоинты остаются локально и в репозиторий не добавляются.

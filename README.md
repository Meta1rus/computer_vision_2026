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
| 11 | [Alignment VLM по предпочтениям](labs/lab_11_vlm_alignment) | выполнена: DPO поверх SFT — EM 0,906 (SFT 0,891), слепой win rate против SFT 0,51 |
| 12 | [Генерация синтетического датасета для VLM](labs/lab_12_synthetic_vlm_data) | выполнена: VLM подтвердила 99 % синтетики, перенос счёта на реальные фото не обнаружен |
| 13 | [Fine-tuning diffusion model](labs/lab_13_diffusion_lora) | выполнена: LoRA r16 — сходство с train 0,602 → 0,659 без копирования, CLIP-текст 0,354 |
| 14 | [Управляемая генерация изображений](labs/lab_14_controlled_generation) | выполнена: ControlNet Canny — edge F1 0,10 → 0,78; синтетика без реальных данных 0,716 |
| 15 | [Flow matching и rectified flow](labs/lab_15_flow_matching) | выполнена: угловое сопоставление выпрямляет траектории в 8,5 раз |
| 16 | [Визуальные токенизаторы и autoregressive generation](labs/lab_16_visual_tokenization) | выполнена: VQ-VAE K=16/64, top-k даёт 0,778 согласования класса |
| 17 | [Diffusion model с нуля и classifier-free guidance](labs/lab_17_ddpm_cfg) | выполнена: DDPM + CFG, при w≥2 класс соблюдается в 99,5 % |
| 18 | [Быстрые samplers и distillation](labs/lab_18_fast_samplers_distillation) | выполнена: DDIM/DPM-Solver++ в 100 раз быстрее DDPM, student-10 сохраняет разнообразие teacher |

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

## Итоги

Все 18 работ выполнены полностью (`FAST_DEV_RUN = False`): у каждой есть выполненный ноутбук, артефакты и отчёт с анализом ошибок и ограничениями.

**Основные результаты**

- *Представления и сегментация (1–5).* Fine-tuning EfficientNet-B0 — 0,944 на test, LRASPP — mIoU 0,804. SAM с рамкой почти не уступает обученной модели (IoU 0,967 против 0,974), с точкой заметно хуже (0,815). Linear probe поверх DINOv2 — 0,959. CLIP zero-shot — 0,937, после линейной адаптации — 0,957, mAP поиска 0,857 → 0,969.
- *VLM: оценка, разметка, данные (6–8, 12).* SmolVLM2 хорошо читает документы и плохо считает объекты. Классы авторазметки Qwen2.5-VL верны в 93,5 % случаев, но классификатор на этих метках на 3,5 п. п. хуже, чем на ручных. Active learning с k-center даёт площадь под кривой обучения 0,899 против 0,850 у случайного отбора. Синтетические сцены VLM подтверждает в 99 % случаев, но перенос счёта на реальные фото не обнаружен.
- *Дообучение VLM (9–11).* QLoRA SFT Qwen2.5-VL-3B в LLaMA-Factory поднимает точность на OOD с 0,578 до 0,719. Unsloth на MLX учит в 4,8 раза быстрее при том же качестве. DPO поверх SFT почти ничего не меняет (EM 0,906 против 0,891, слепой win rate 0,51): пары оказались слишком лёгкими для уже дообученной модели.
- *Генерация (13–18).* LoRA для SD 1.5 переносит облик объекта без копирования train. ControlNet Canny поднимает edge F1 с 0,10 до 0,78. Угловое сопоставление в flow matching выпрямляет траектории в 8,5 раза. DDPM с CFG при w ≥ 2 генерирует нужный класс в 99,5 % случаев. DDIM и DPM-Solver++ в 100 раз быстрее DDPM, а дистиллированный student сохраняет разнообразие teacher.

**Общие выводы**

1. Метрики и промежуточные данные нужно проверять вручную, начиная с худших случаев. Так нашлись ошибки в самой постановке: рамка SAM по неверной маске (работа 3, IoU 0,936 → 0,967) и почти пустые контуры Canny (работа 14, классификатор на синтетике 0,109 → 0,716). В работе 10 проверка показала, что в Unsloth на MLX маска ответа для VLM не применяется и её нужно задавать явно.
2. Автоматические метрики для ответов VLM искажают картину: в работе 6 форма ответа влияла на EM сильнее, чем правильность. Поэтому в работах 6, 7 и 11 метрики дополнены ручной разметкой, слепым сравнением и разбором ошибок.
3. Синтетика полезна, только если добавляет то, чего нет в реальных данных. Повтор тех же форм (работа 14) и упрощённые сцены (работа 12) прироста не дали.
4. Весь курс выполним локально на Apple Silicon: PyTorch через MPS, Unsloth через MLX. Ограничивает в основном объём unified memory: VLM и диффузионные модели занимают по 7–10 ГБ.

**Ограничения.** Большинство экспериментов проведено с одним seed и на небольших оценочных наборах. Ручную оценку в работах 6, 7 и 11 выполнял один разметчик. Часть замеров времени сделана при параллельной нагрузке на GPU. Подробности — в отчётах работ.

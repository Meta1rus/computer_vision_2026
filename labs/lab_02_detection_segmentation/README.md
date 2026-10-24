# Лабораторная работа №2. Детекция и сегментация объектов

**Дата:** 24.10.2026
**Ноутбук:** [student.ipynb](student.ipynb) — выполнен полностью, `FAST_DEV_RUN = False`

## Постановка

Семантическая сегментация Oxford-IIIT Pet (3 класса trimap: животное, фон, граница). Проверка разметки, дообучение компактного сегментатора, сравнение с аугментацией отражением, оценка на малых объектах и разбор FP/FN.

## Данные

- Oxford-IIIT Pet (Parkhi et al., 2012; CC BY-SA 4.0), загрузка через torchvision.
- trainval 3 680 → train 2 944 / validation 736 (стратификация по 37 породам, проверка отсутствия одинаковых файлов по md5), test — официальный, 3 669.
- Изображение и маска масштабируются к 320×320 совместно; маска — nearest, метки 1–3 перекодированы в 0–2.
- Статистика train: животное — 30 % пикселей, фон — 58 %, граница — 11 %; медианная площадь животного — 28 % кадра; пустых масок 7 (`label_stats.json`, `label_check.png`).

## Модель и обучение

LRASPP-MobileNetV3-Large (torchvision, веса COCO-VOC), обе головы (`low_classifier`, `high_classifier`) заменены на 3-классовые. AdamW, OneCycle (max lr 5e-4), batch 16, 12 эпох. Пиксельная cross-entropy, чекпоинт после каждой эпохи с возможностью продолжения. Аугментированная модель отличается только синхронным горизонтальным отражением пары «изображение — маска».

## Результаты

| модель | val mIoU | test mIoU | test IoU: pet / background / border |
|---|---:|---:|---|
| baseline | 0,7986 | 0,8016 | 0,883 / 0,939 / 0,583 |
| + отражение | **0,8021** | **0,8041** | 0,885 / 0,940 / 0,588 |

Поклассовая динамика на validation (`class_deltas.json`): pet +0,25 п. п., background +0,16 п. п., border +0,61 п. п.

**Малые объекты** (животное занимает меньше 15,8 % кадра — 20-й перцентиль train; 150 изображений validation): foreground IoU 0,801 → 0,806 с отражением, против 0,887 → 0,889 на всей validation.

## Анализ ошибок

У 20 худших изображений validation FP преобладает над FN в 75 % случаев (`error_examples.png`):

- люди, обнимающие собаку, попадают в маску животного (Boxer, IoU 0,29–0,53);
- статуя кошки рядом с живой кошкой сегментируется как животное;
- мелкие животные (1–2 % кадра) сегментируются частично;
- спорная разметка: у собаки в свитере trimap отмечает только голову;
- граница — самый слабый класс (IoU ≈ 0,57), это узкая полоса вокруг контура.

Сложнейшее изображение validation — Boxer с девушкой в очках (`bonus_hard_case.png`, pet IoU 0,285): маска «растекается» на человека.

## Выводы

1. LRASPP-MobileNetV3 даёт test mIoU 0,80 (IoU животного 0,88) за ~4,5 минуты обучения на M3 Max.
2. Отражение добавляет 0,3 п. п. mIoU на validation и test, сильнее всего на границе. Эффект мал, и для уверенного вывода нужны повторы по seed.
3. Основные ошибки — ложные срабатывания на людях и животноподобных объектах и недосегментация мелких животных.

## Артефакты

`artifacts/`: `split.json`, `label_stats.json`, `val_metrics.json`, `augmentation_comparison.json`, `small_objects.json`, `test_metrics.json`, `error_analysis.json`, `class_deltas.json`, `bonus_hard_case.json`, рисунки `label_check.png`, `error_examples.png`, `bonus_hard_case.png`. Веса `segmentation.pt` и `segmentation_aug.pt` хранятся локально.

## Воспроизведение

```bash
cd labs/lab_02_detection_segmentation
jupyter nbconvert --to notebook --execute --inplace student.ipynb
```

Около 10 минут на M3 Max (плюс однократная загрузка Oxford-IIIT Pet, ~800 МБ). Seed 42, torchvision 0.29.1, PyTorch 2.14.1.

**Вычислительные ресурсы:** MacBook Pro (Apple M3 Max, 40-core GPU, 48 ГБ), PyTorch MPS.

"""Download a small, provenance-bearing PixMo evaluation subset; no user files required.

Metadata: https://huggingface.co/datasets/allenai/{pixmo-cap,pixmo-ask-model-anything,pixmo-docs,pixmo-count}
Dataset cards declare ODC-BY-1.0; original web photographs may have separate rights.
"""
from pathlib import Path
import hashlib
import json
import urllib.request

API = 'https://datasets-server.huggingface.co/rows'
DATASETS = (
    ('captioning', 'pixmo-cap', 'default', 'train', 5),
    ('vqa', 'pixmo-ask-model-anything', 'default', 'train', 5),
    ('ocr', 'pixmo-docs', 'other', 'validation', 5),
    ('structured', 'pixmo-count', 'default', 'validation', 5),
)


def _download(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (educational dataset evaluation)'})
    with urllib.request.urlopen(req, timeout=35) as response:
        return response.read()


def _rows(dataset, config, split, offset, length=100):
    from urllib.parse import urlencode
    url = API + '?' + urlencode(dict(dataset='allenai/' + dataset, config=config, split=split, offset=offset, length=length))
    return json.loads(_download(url))['rows']


def load_open_eval(root='data/lab06', per_task=None):
    """Return real examples with published references; downloads are cached and checked.

    Structured task extracts existing PixMo-Count label/count fields, not invented labels.
    OCR uses published document QA answer, not a verbatim transcription benchmark.
    """
    from PIL import Image
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    examples = []
    for task, dataset, config, split, default_limit in DATASETS:
        limit = per_task or default_limit
        got = 0
        for offset in range(0, 400, 100):
            for item in _rows(dataset, config, split, offset):
                row = item['row']
                idx = item['row_idx']
                image_url = row['image']['src'] if dataset == 'pixmo-docs' else row['image_url']
                path = root / f'{dataset}-{config}-{split}-{idx}.jpg'
                try:
                    if not path.is_file():
                        data = _download(image_url)
                        if row.get('image_sha256') and hashlib.sha256(data).hexdigest() != row['image_sha256']:
                            raise ValueError('SHA256 mismatch')
                        path.write_bytes(data)
                    with Image.open(path) as im:
                        im.verify()
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                if task == 'captioning':
                    question, reference, category = 'Describe the image.', row['caption'], 'caption'
                elif task == 'vqa':
                    question, reference, category = row['question'].strip(), row['answer'], 'open_vqa'
                elif task == 'ocr':
                    q = row['questions']
                    question, reference, category = q['question'][0], q['answer'][0], 'document_qa'
                else:
                    question = f"How many {row['label']} are visible? Return only JSON with keys label and count, using label '{row['label']}'."
                    reference = json.dumps({'label': row['label'], 'count': row['count']}, ensure_ascii=False)
                    category = 'object_count_json'
                examples.append(dict(image=str(path.resolve()), question=question, reference=reference,
                                     category=category, difficulty='published', license='ODC-BY-1.0 (dataset metadata; image rights may differ)',
                                     task=task, source='https://huggingface.co/datasets/allenai/' + dataset,
                                     source_split=split, source_row=idx, image_url=image_url if dataset != 'pixmo-docs' else None,
                                     image_sha256=row.get('image_sha256')))
                got += 1
                if got == limit:
                    break
            if got == limit:
                break
        if got != limit:
            raise RuntimeError(f'{dataset}: downloaded {got}/{limit}; retry network or inspect source URLs')
    return examples

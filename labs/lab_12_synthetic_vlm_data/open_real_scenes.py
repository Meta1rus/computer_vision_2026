"""Independent real-photograph count benchmark from PixMo-Count (ODC-BY-1.0 metadata).

https://huggingface.co/datasets/allenai/pixmo-count — val/test human verified.
Photos depict real objects, NOT geometric-shape instances. This is a cross-domain
counting experiment, not evidence of in-domain shape-counting generalization.
Image URLs are hosted by original third parties; their rights can differ from the
annotation dataset's ODC-BY license. No images are redistributed in the repository.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlencode

API = 'https://datasets-server.huggingface.co/rows'


def load_real_scenes(root='data/lab12', train_size=24, test_size=12):
    from PIL import Image
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for split, source_split, limit in [('train', 'validation', train_size), ('test', 'test', test_size)]:
        chosen = []
        for offset in range(0, 540, 100):
            query = urlencode(dict(dataset='allenai/pixmo-count', config='default', split=source_split, offset=offset, length=min(100, 540-offset)))
            with urllib.request.urlopen(API + '?' + query, timeout=35) as response:
                rows = json.load(response)['rows']
            for entry in rows:
                r = entry['row']
                # Common class support with the synthetic 1..3 renderer; test is human-verified.
                if r['count'] not in (2, 3) or sum(v['answer'] == str(r['count']) for v in chosen) >= (limit + 1)//2:
                    continue
                idx = entry['row_idx']
                path = root / f'pixmo-count-{split}-{idx}.jpg'
                try:
                    if not path.exists():
                        req = urllib.request.Request(r['image_url'], headers={'User-Agent': 'Mozilla/5.0 (educational research)'})
                        with urllib.request.urlopen(req, timeout=30) as response:
                            data = response.read()
                        path.write_bytes(data)
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                chosen.append(dict(id=f'pixmo-count-{split}-{idx}', split=split,
                                   image=str(path.resolve()), answer=str(r['count']), label=r['label'],
                                   question=f"How many {r['label']} are visible? Answer with one integer.",
                                   source='https://huggingface.co/datasets/allenai/pixmo-count',
                                   source_row=idx, source_split=source_split, image_url=r['image_url'],
                                   image_sha256=r['image_sha256'], downloaded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   checksum_matches=hashlib.sha256(path.read_bytes()).hexdigest() == r['image_sha256'],
                                   dataset_license='ODC-BY-1.0; image rights may differ'))
                if len(chosen) >= limit:
                    break
            if len(chosen) >= limit:
                break
        if len(chosen) < limit or {r['answer'] for r in chosen} != {'2', '3'}:
            raise RuntimeError(f'{split}: {len(chosen)}/{limit} available real images or missing class; retry/inspect original URLs')
        out.extend(chosen)
    if {r['image_sha256'] for r in out[:train_size]} & {r['image_sha256'] for r in out[train_size:]}:
        raise ValueError('Real train/test overlap')
    manifest = root / 'real_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    return out

"""CPU-only helpers for a bounded event-weighting preflight."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OPENPI = PROJECT.parents[1] / "projects/openpi"
DATA = Path('/home/zhsu/jyluo/dataset_root/physical-intelligence/libero')
SEED = 20261022


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def events_and_mask(actions, radius=5):
    signs = np.asarray(actions)[:, 6] > 0
    changes = np.flatnonzero(signs[1:] != signs[:-1]) + 1
    events = [int(k) for k in changes if k >= 2 and k+1 < len(signs)
              and signs[k-2] == signs[k-1] and signs[k] == signs[k+1]]
    mask = np.zeros(len(signs), dtype=bool)
    for k in events:
        mask[max(0, k-radius):min(len(signs), k+radius+1)] = True
    return events, mask


def image_value(value):
    if value.get('bytes') is not None:
        return np.asarray(Image.open(io.BytesIO(value['bytes'])).convert('RGB'))
    return np.asarray(Image.open(DATA/value['path']).convert('RGB'))

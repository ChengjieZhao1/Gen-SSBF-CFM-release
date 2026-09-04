import json
import random
from pathlib import Path

import numpy as np
import torch


def resolve_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float_dict(payload):
    out = {}
    for key, value in payload.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            out[key] = float(value.item())
        else:
            out[key] = value
    return out


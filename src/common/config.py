"""Central config + path resolution shared by Windows, WSL2 and Colab.

Dataset root is controlled by ONE variable: env VR_DATA_ROOT (falls back to
configs/default.json -> data_root, relative to the project root).
Artifacts dir: env VR_ARTIFACTS_DIR (same fallback logic).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    p = Path(path) if path else ROOT / "configs" / "default.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(env_name, default_value):
    value = os.environ.get(env_name) or default_value
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def data_root(cfg):
    return _resolve("VR_DATA_ROOT", cfg["data_root"])


def artifacts_dir(cfg):
    return _resolve("VR_ARTIFACTS_DIR", cfg["artifacts_dir"])

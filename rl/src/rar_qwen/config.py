from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_base_dir"] = str(config_path.parent.parent)
    return cfg


def project_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_base_dir"]) / path


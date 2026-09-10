"""Loader dung chung cho cac file config TOML cua project.

Toan bo config (base_scene, object_roi, gof_meshing, simulation) deu dung
TOML thay vi YAML. Ham nay thay the pattern `open(...) + yaml.safe_load(...)`
lap lai o nhieu file bang mot diem load duy nhat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def load_toml_config(path: Union[str, Path]) -> dict:
    """Doc mot file `.toml` va tra ve noi dung duoi dang dict.

    Dung `cfg.get(key, default)` de truy cap tung tham so, giu nguyen quy
    uoc da co tu thoi con dung YAML.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return tomllib.load(f)

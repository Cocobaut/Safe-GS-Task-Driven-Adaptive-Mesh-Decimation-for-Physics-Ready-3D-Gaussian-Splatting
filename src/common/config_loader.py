"""Loader dung chung cho cac file config TOML cua project.

Toan bo config (base_scene, object_roi, gof_meshing, simulation) deu dung
TOML thay vi YAML. Ham nay thay the pattern `open(...) + yaml.safe_load(...)`
lap lai o nhieu file bang mot diem load duy nhat.
"""

from __future__ import annotations

import re
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


def resolve_raw_image_dir(cfg: dict, default: str = "data/raw") -> str:
    """Tra ve duong dan thu muc anh raw tu 1 config TOML da load.

    Uu tien theo thu tu:
      1. `raw_image_dir` neu co trong toml (override thu cong, ghi de tat ca).
      2. Bang `[dataset]` (vd `name = "replica"`, `scene = "office0"`) -> tra
         cuu qua `Dataset Configs/<name>.yaml` (xem `dataset_config.py`) va
         lay `images_dir`.
      3. `default`.
    """
    if cfg.get("raw_image_dir"):
        return cfg["raw_image_dir"]

    dataset_cfg = cfg.get("dataset")
    if dataset_cfg and dataset_cfg.get("name"):
        from src.common.dataset_config import load_dataset_paths

        overrides = {k: v for k, v in dataset_cfg.items() if k != "name"}
        paths = load_dataset_paths(dataset_cfg["name"], **overrides)
        return paths["images_dir"]

    return default


def _slugify(text: str) -> str:
    """Chuan hoa 1 chuoi thanh ten thu muc an toan: 'Replica 8 Scene' -> 'replica_8_scene'."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "run"


def resolve_run_tag(cfg: dict, default: str = "default") -> str:
    """Sinh 1 ten rieng cho tung lan chay (dataset + scene), dung de dat ten
    thu muc output (vd sfm) khong bi ghi de giua cac dataset/scene khac nhau.

    Uu tien theo thu tu:
      1. `[dataset] tag = "..."` neu co (dat ten thu cong).
      2. Ghep `[dataset] name` + `scene` (vd name="replica", scene="office0"
         -> "replica_office0").
      3. `default`.
    """
    dataset_cfg = cfg.get("dataset") or {}
    if dataset_cfg.get("tag"):
        return _slugify(str(dataset_cfg["tag"]))

    parts = [str(dataset_cfg[k]) for k in ("name", "scene") if dataset_cfg.get(k)]
    if parts:
        return _slugify("_".join(parts))

    return default

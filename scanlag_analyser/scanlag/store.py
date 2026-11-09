"""Results storage. The only module that writes to disk, and it never writes to raw data.

Every output path is resolved and checked: it must be inside the results directory and must
not be inside any registered raw-data directory. A results directory remembers which raw
directories it belongs to, so two experiments with the same folder name (e.g. two ``AMP7``
folders) can't silently share one results directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
REGISTRY_FILE = ".scanlag_results.json"
LABEL_CHUNK = 512


class RawDataProtectionError(PermissionError):
    pass


def default_results_dir(raw_dir: str | Path, root: str | Path = RESULTS_ROOT) -> Path:
    """``<root>/<raw data folder name>``, with ``root`` defaulting to ``<project root>/results``."""
    return Path(root) / Path(raw_dir).name


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


class ResultsStore:
    def __init__(self, results_dir: str | Path, raw_dirs: list[str | Path]):
        self.root = Path(results_dir).resolve()
        self.raw_dirs = sorted({Path(d).resolve() for d in raw_dirs})
        for raw in self.raw_dirs:
            if _is_within(self.root, raw) or _is_within(raw, self.root):
                raise RawDataProtectionError(
                    f"results directory {self.root} must be separate from raw data {raw}"
                )
        self._register()

    @classmethod
    def for_raw_dir(cls, raw_dir: str | Path, results_dir: str | Path | None = None) -> ResultsStore:
        return cls(results_dir or default_results_dir(raw_dir), [raw_dir])

    def _register(self) -> None:
        registry = self.root / REGISTRY_FILE
        mine = [str(d) for d in self.raw_dirs]
        if registry.exists():
            known = json.loads(registry.read_text())["raw_dirs"]
            if sorted(known) != mine:
                raise FileExistsError(
                    f"{self.root} already holds results for {known}; "
                    f"choose another results directory for {mine}"
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps({"raw_dirs": mine}, indent=2))

    def path(self, *parts: str) -> Path:
        """Resolve an output path, refusing anything outside the results dir or inside raw data."""
        target = self.root.joinpath(*parts).resolve()
        if not _is_within(target, self.root):
            raise RawDataProtectionError(f"{target} is outside the results directory {self.root}")
        for raw in self.raw_dirs:
            if _is_within(target, raw):
                raise RawDataProtectionError(f"refusing to write inside raw data: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, obj, *parts: str) -> Path:
        target = self.path(*parts)
        target.write_text(json.dumps(obj, indent=2, default=_json_default))
        return target

    def write_table(self, df: pd.DataFrame, *parts: str) -> Path:
        target = self.path(*parts)
        if target.suffix == ".csv":
            df.to_csv(target, index=False)
        else:
            df.to_parquet(target, index=False)
        return target

    def write_mask(self, mask: np.ndarray, *parts: str) -> Path:
        target = self.path(*parts)
        Image.fromarray(mask.astype(bool)).save(target)  # 1-bit PNG
        return target

    def write_image(self, image: np.ndarray, *parts: str) -> Path:
        target = self.path(*parts)
        Image.fromarray(image).save(target)
        return target

    def create_labels(self, shape: tuple[int, int, int], *parts: str) -> zarr.Array:
        """A [T, H, W] uint16 label stack, chunked per frame and compressed."""
        target = self.path(*parts)
        return zarr.create_array(
            store=str(target),
            shape=shape,
            chunks=(1, min(LABEL_CHUNK, shape[1]), min(LABEL_CHUNK, shape[2])),
            dtype=np.uint16,
            fill_value=0,
            overwrite=True,
        )


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, os.PathLike):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")

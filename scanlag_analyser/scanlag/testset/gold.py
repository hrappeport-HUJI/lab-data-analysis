"""The hand-corrected gold set: 256x256 tiles in ``<testset>/manual/tile_XX_<category>/``.

Each tile has the RGB frame, the background (first) scan, an earlier frame, the MATLAB labels it
started from (``labels_init.png``) and the corrected labels (``labels_corrected.png``). The
evaluation ignore region is everything outside the plate's working-area mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scanlag.detect.classical import stretch_limits, subtract_background
from scanlag.testset.tiles import OTHER_ID_START, TILE


@dataclass
class GoldTile:
    name: str
    category: str
    plate_id: str
    time_min: float
    image: np.ndarray  # uint8 [H, W, 3]
    background: np.ndarray
    previous: np.ndarray
    labels: np.ndarray  # uint16 corrected colonies, 0 = background
    matlab: np.ndarray  # uint16 MATLAB labels: kept colonies < OTHER_ID_START <= other blobs
    ignore: np.ndarray  # bool, outside the working area

    @property
    def matlab_kept(self) -> np.ndarray:
        return np.where(self.matlab < OTHER_ID_START, self.matlab, 0)

    @property
    def difference(self) -> np.ndarray:
        return difference_image(self.image, self.background)


def _load(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def difference_image(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Gray(image - background), contrast-stretched to [0, 1] on this tile."""
    gray = subtract_background(image, background)
    lo, hi = stretch_limits(gray, 0.01, 0.999) * 255
    return np.clip((gray - lo) / max(hi - lo, 1), 0, 1)


def load_gold_set(testset: Path) -> list[GoldTile]:
    tiles = []
    for folder in sorted((testset / "manual").glob("tile_*")):
        corrected = folder / "labels_corrected.png"
        if not corrected.exists():
            raise FileNotFoundError(f"{folder.name} has not been corrected yet")
        meta = json.loads((folder / "tile.json").read_text())
        mask = _load(testset / "plates" / meta["plate_id"] / "mask.png").astype(bool)
        y, x = meta["y"], meta["x"]
        tiles.append(
            GoldTile(
                name=folder.name,
                category=meta["category"],
                plate_id=meta["plate_id"],
                time_min=meta["time_min"],
                image=_load(folder / "image.png"),
                background=_load(folder / "background.png"),
                previous=_load(folder / "previous.png"),
                labels=_load(corrected).astype(np.uint16),
                matlab=_load(folder / "labels_init.png").astype(np.uint16),
                ignore=~mask[y : y + TILE, x : x + TILE],
            )
        )
    return tiles

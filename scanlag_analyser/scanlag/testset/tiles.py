"""Select small tiles from the MATLAB test set for manual correction.

Three categories target the cases where threshold detection is weakest:

* ``early``    - frames where many colonies have just appeared (small, low contrast)
* ``merged``   - frames where touching colonies form one blob in MATLAB
* ``excluded`` - neighbourhoods of colonies the user excluded by hand (debris, writing, ...)

Each tile folder holds the RGB crop from the raw scan at that frame, the background (first)
scan, an earlier frame, and ``labels_init.png``: MATLAB's segmentation with kept colonies under
their MATLAB ID and every other blob (merged, excluded, rejected) under its own ID >= 10000.
The annotator edits that into ``labels_corrected.png``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage as ndi
from PIL import Image

from scanlag.io.raw import read_window
from scanlag.detect.channels import PREVIOUS_GAP  # noqa: F401 (also imported from here)
from scanlag.store import ResultsStore
from scanlag.testset.build import IGNORE

TILE = 256
OTHER_ID_START = 10000
MIN_MASK_COVERAGE = 0.6


@dataclass
class TileSpec:
    plate_id: str
    category: str
    frame: int
    x: int  # top-left in plate-crop coordinates
    y: int
    score: int


def _best_window(points: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
    """Tile position containing the most points, with enough of the tile inside the mask."""
    h, w = mask.shape
    best = None
    for px, py in points:
        x = int(np.clip(px - TILE // 2, 0, w - TILE))
        y = int(np.clip(py - TILE // 2, 0, h - TILE))
        if mask[y : y + TILE, x : x + TILE].mean() < MIN_MASK_COVERAGE:
            continue
        inside = (points[:, 0] >= x) & (points[:, 0] < x + TILE) & (points[:, 1] >= y) & (points[:, 1] < y + TILE)
        score = int(inside.sum())
        if best is None or score > best[2]:
            best = (x, y, score)
    return best


def candidate_tiles(plate_dir: Path) -> list[TileSpec]:
    pid = plate_dir.name
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    colonies = pd.read_parquet(plate_dir / "colonies.parquet")
    mask = np.array(Image.open(plate_dir / "mask.png"))
    valid = colonies[colonies.status == 0]
    specs = []

    if len(valid) >= 5:
        frame = int(np.percentile(valid.appearance_frame, 30))
        fresh = valid[(valid.appearance_frame > frame - 4) & (valid.appearance_frame <= frame)]
        pts = tracks[(tracks.frame == frame) & tracks.colony_id.isin(fresh.colony_id)][["x", "y"]].to_numpy()
        if (best := _best_window(pts, mask)) and best[2] >= 3:
            specs.append(TileSpec(pid, "early", frame, *best))

    merged = tracks[tracks.merged]
    if len(merged):
        # Shortly after merging starts: blobs are touching but still recognisable.
        frame = int(np.percentile(merged.frame, 25))
        pts = merged[merged.frame == frame][["x", "y"]].to_numpy()
        if (best := _best_window(pts, mask)) and best[2] >= 4:
            specs.append(TileSpec(pid, "merged", frame, *best))

    excluded = colonies[colonies.status == 2]
    if len(excluded):
        frame = int(tracks.frame.max() * 0.6)
        pts = excluded[["seed_x", "seed_y"]].to_numpy()
        if (best := _best_window(pts, mask)) and best[2] >= 1:
            specs.append(TileSpec(pid, "excluded", frame, *best))
    return specs


def select_tiles(testset: Path, per_category: int, seed: int = 0) -> list[TileSpec]:
    """Up to ``per_category`` tiles per category, spread over experiments, one tile per plate."""
    rng = np.random.default_rng(seed)
    index = pd.read_csv(testset / "index.csv")
    if "error" in index:
        index = index[index["error"].isna()]
    plates = [testset / "plates" / pid for pid in sorted(index.plate_id)]
    candidates = [s for p in plates for s in candidate_tiles(p)]
    chosen, used_plates = [], set()
    for category in ("early", "merged", "excluded"):
        pool = [s for s in candidates if s.category == category and s.plate_id not in used_plates]
        by_experiment: dict[str, list[TileSpec]] = {}
        for s in sorted(pool, key=lambda s: -s.score):
            by_experiment.setdefault(s.plate_id.split("__")[0], []).append(s)
        for group in by_experiment.values():  # top-scoring plates first, random among them
            top = group[: max(3, len(group) // 3)]
            rng.shuffle(top)
            group[: len(top)] = top
        picked = []
        while len(picked) < per_category and any(by_experiment.values()):
            for group in by_experiment.values():  # round-robin across experiments
                while group and group[0].plate_id in used_plates:
                    group.pop(0)
                if group and len(picked) < per_category:
                    spec = group.pop(0)
                    picked.append(spec)
                    used_plates.add(spec.plate_id)
        chosen += picked
    return chosen


def init_labels(test_labels: np.ndarray) -> np.ndarray:
    """MATLAB IDs for kept colonies, fresh IDs >= OTHER_ID_START for each IGNORE blob."""
    out = np.where(test_labels == IGNORE, 0, test_labels).astype(np.uint16)
    others, n = ndi.label(test_labels == IGNORE, structure=np.ones((3, 3)))
    out[others > 0] = (others[others > 0] + OTHER_ID_START - 1).astype(np.uint16)
    return out


def plate_window(info: dict, frame: int, x: int, y: int, size: int | tuple[int, int] = TILE) -> np.ndarray:
    """RGB window of a raw scan, at (x, y) in plate-crop coordinates (``info`` is plate.json).

    ``size`` is the side of a square window or (height, width).
    """
    h, w = (size, size) if isinstance(size, int) else size
    ox, oy = info["raw_offset_xy"][frame]
    rows = slice(int(oy) + y, int(oy) + y + h)
    cols = slice(int(ox) + x, int(ox) + x + w)
    return read_window(Path(info["raw_dir"]) / info["raw_scans"][frame], rows, cols)


def export_tile(testset: Path, spec: TileSpec, index: int, store: ResultsStore) -> Path:
    plate_dir = testset / "plates" / spec.plate_id
    info = json.loads((plate_dir / "plate.json").read_text())
    raw_dir = Path(info["raw_dir"])
    offsets = np.asarray(info["raw_offset_xy"], dtype=int)
    name = f"tile_{index:02d}_{spec.category}"

    def crop(frame: int) -> np.ndarray:
        return plate_window(info, frame, spec.x, spec.y)

    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    tile_labels = labels[spec.frame, spec.y : spec.y + TILE, spec.x : spec.x + TILE]
    store.write_image(crop(spec.frame), "manual", name, "image.png")
    store.write_image(crop(0), "manual", name, "background.png")
    store.write_image(crop(max(spec.frame - PREVIOUS_GAP, 0)), "manual", name, "previous.png")
    store.write_image(init_labels(tile_labels), "manual", name, "labels_init.png")
    store.write_json(
        {
            **vars(spec),
            "time_min": info["times_min"][spec.frame],
            "raw_scan": info["raw_scans"][spec.frame],
            "raw_dir": raw_dir,
            "raw_x": int(offsets[spec.frame][0] + spec.x),
            "raw_y": int(offsets[spec.frame][1] + spec.y),
            "tile_size": TILE,
        },
        "manual",
        name,
        "tile.json",
    )
    return store.path("manual", name)

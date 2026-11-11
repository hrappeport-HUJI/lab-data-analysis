"""Training tiles for colony segmentation, from MATLAB pseudo-labels and the gold set.

Every tile is 256x256 px at native scan resolution and holds:

* ``x``      uint8 [H, W, 3]: the model channels (see ``model_channels``)
* ``labels`` uint16 [H, W]: one ID per colony, 0 = background
* ``ignore`` bool [H, W]: pixels that must not contribute to the loss

MATLAB tiles (PLAN §5.2) use only colonies kept after review. IGNORE covers everything else
MATLAB segmented (merged blobs, excluded or rejected blobs) and everything outside the plate's
working-area mask. Gold tiles use the hand-corrected labels. In merged tiles, MATLAB merged blobs
the annotator left untouched are also ignored, since they were never checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

from scanlag.detect.classical import rgb2gray_uint8, subtract_background
from scanlag.store import ResultsStore
from scanlag.testset.build import IGNORE
from scanlag.testset.gold import load_gold_set
from scanlag.testset.tiles import OTHER_ID_START, PREVIOUS_GAP, TILE, plate_window

CHANNELS = ("diff", "gray", "gray_previous")
FRAME_QUANTILES = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
MIN_MASK_COVERAGE = 0.5


def model_channels(image: np.ndarray, background: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """[gray(image - background), gray(image), gray(previous)] as uint8."""
    return np.stack([subtract_background(image, background), rgb2gray_uint8(image), rgb2gray_uint8(previous)], -1)


def _frames(colonies: pd.DataFrame, n_frames: int) -> list[int]:
    valid = colonies[colonies.status == 0]
    if valid.empty:
        return []
    appear = valid.appearance_frame.to_numpy()
    frames = {int(np.quantile(appear, q)) for q in FRAME_QUANTILES}
    frames.add(max(int(appear.min()) - 3, 0))  # mostly empty plate: negative examples
    frames.add(n_frames - 1)
    return sorted(frames)


def _positions(tracks: pd.DataFrame, frame: int, mask: np.ndarray, rng, per_frame: int) -> list[tuple[int, int]]:
    """Tile corners centred on random colonies present at ``frame``, plus one random spot."""
    h, w = mask.shape
    pts = tracks.loc[tracks.frame == frame, ["x", "y"]].to_numpy()
    pts = pts[rng.permutation(len(pts))]
    ys, xs = np.nonzero(mask)
    random_pt = np.array([[xs[i], ys[i]] for i in rng.integers(0, len(xs), 5)])
    out = []
    for candidates, limit in ((pts[: per_frame * 3], per_frame), (random_pt, per_frame + 1)):
        for px, py in candidates:
            if len(out) >= limit:
                break
            x = int(np.clip(px - TILE // 2, 0, w - TILE))
            y = int(np.clip(py - TILE // 2, 0, h - TILE))
            if mask[y : y + TILE, x : x + TILE].mean() < MIN_MASK_COVERAGE:
                continue
            if any(abs(x - a) < TILE // 2 and abs(y - b) < TILE // 2 for a, b in out):
                continue  # avoid near-duplicates
            out.append((x, y))
    return out


def plate_tiles(plate_dir: Path, rng, per_frame: int = 3, avoid: list[tuple[int, int]] = ()) -> dict[str, list]:
    info = json.loads((plate_dir / "plate.json").read_text())
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    colonies = pd.read_parquet(plate_dir / "colonies.parquet")
    mask = np.array(Image.open(plate_dir / "mask.png")).astype(bool)
    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    out = {"x": [], "labels": [], "ignore": [], "meta": []}
    background = {}
    for frame in _frames(colonies, labels.shape[0]):
        for x, y in _positions(tracks, frame, mask, rng, per_frame):
            if any(abs(x - a) < TILE and abs(y - b) < TILE for a, b in avoid):
                continue  # overlaps a gold tile; the gold labels take precedence
            if (x, y) not in background:
                background[x, y] = plate_window(info, 0, x, y)
            image = plate_window(info, frame, x, y)
            previous = plate_window(info, max(frame - PREVIOUS_GAP, 0), x, y)
            lab = labels[frame, y : y + TILE, x : x + TILE]
            out["x"].append(model_channels(image, background[x, y], previous))
            out["labels"].append(np.where(lab == IGNORE, 0, lab).astype(np.uint16))
            out["ignore"].append((lab == IGNORE) | ~mask[y : y + TILE, x : x + TILE])
            out["meta"].append({"source": "matlab", "plate_id": plate_dir.name, "frame": frame, "x": x, "y": y})
    return out


def gold_tiles(testset: Path) -> dict[str, list]:
    out = {"x": [], "labels": [], "ignore": [], "meta": []}
    for t in load_gold_set(testset):
        ignore = t.ignore.copy()
        labels = t.labels.copy()
        if t.category == "merged":
            for i in np.unique(labels[labels >= OTHER_ID_START]):
                blob = labels == i
                if np.array_equal(blob, t.matlab == i):  # never edited
                    ignore |= blob
                    labels[blob] = 0
        meta = json.loads((testset / "manual" / t.name / "tile.json").read_text())
        out["x"].append(model_channels(t.image, t.background, t.previous))
        out["labels"].append(labels)
        out["ignore"].append(ignore)
        out["meta"].append({"source": "gold", "plate_id": t.plate_id, "frame": meta["frame"], "x": meta["x"], "y": meta["y"], "tile": t.name})
    return out


def build_training_set(testset: Path, store: ResultsStore, per_frame: int = 3, seed: int = 0, log=print) -> Path:
    rng = np.random.default_rng(seed)
    gold = gold_tiles(testset)
    avoid: dict[str, list] = {}
    for m in gold["meta"]:
        avoid.setdefault(m["plate_id"], []).append((m["x"], m["y"]))

    index = pd.read_csv(testset / "index.csv")
    if "error" in index:
        index = index[index["error"].isna()]
    parts = [gold]
    for i, pid in enumerate(sorted(index.plate_id)):
        part = plate_tiles(testset / "plates" / pid, rng, per_frame, avoid.get(pid, []))
        parts.append(part)
        log(f"[{i + 1}/{len(index)}] {pid}: {len(part['x'])} tiles")

    data = {k: [v for p in parts for v in p[k]] for k in ("x", "labels", "ignore", "meta")}
    path = store.path("train", "tiles.npz")
    np.savez_compressed(path, x=np.stack(data["x"]), labels=np.stack(data["labels"]), ignore=np.stack(data["ignore"]))
    store.write_table(pd.DataFrame(data["meta"]), "train", "tiles.csv")
    return path


def load_training_set(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    with np.load(path) as f:
        x, labels, ignore = f["x"], f["labels"], f["ignore"]
    return x, labels, ignore, pd.read_csv(path.with_suffix(".csv"))

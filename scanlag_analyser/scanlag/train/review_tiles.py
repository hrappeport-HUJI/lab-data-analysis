"""Training tiles from a reviewed run: the places where the review found the detector wrong.

A colony the reviewer excluded by hand is a place where detection or tracking went wrong, and its flags
say how. Each such colony gives one tile, in the category of its first matching flag:

* ``doublet`` ("possible doublet"): two colonies that touched when first detected, so the detector
  never separated them; at the appearance frame + 4, when both are visible and still small
* ``split`` ("irregular split"): the merged blob was divided badly; at the last frame
* ``fragment`` ("large at appearance"): a piece split off a neighbour; at the appearance frame
* ``track`` ("jump" or "flicker"): the segmentation switched between frames; at the frame of the
  largest area change
* ``excluded`` (no such flag: debris, writing, reflections, a speck): at its last frame

Colonies the reviewer included are fine and give no tile. With ``unreviewed=True``, included flagged
colonies not reviewed yet also give tiles, by the same categories (their flags may or may not be errors). Each tile is TILE x TILE
px around the colony, written in the layout of the MATLAB gold set (``scanlag.testset.tiles``):

    <out>/manual/tile_<scanner>_p<plate>_c<colony>_<category>/
        image.png, background.png, previous.png   RGB crops of that frame, frame 0, frame - PREVIOUS_GAP
        labels_init.png                           the pipeline's colonies (colony IDs) and other detected
                                                  objects (IDs >= OTHER_ID_START)
        tile.json                                 where it comes from, and why
    <out>/plates/<scanner>__plate_<k>/mask.png    the working area, for the loss mask

so ``annotate_tiles.py --testset <out>`` opens them for correction (into ``labels_corrected.png``) and
``scanlag.testset.gold.load_gold_set(<out>)`` reads the corrected ones for training.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from scanlag.io.matlab_import import ColonyStatus
from scanlag.io.raw import read_window
from scanlag.pipeline.flags import kinds_of
from scanlag.testset.tiles import OTHER_ID_START, PREVIOUS_GAP, TILE

CATEGORY_KINDS = {"doublet": ["possible doublet"], "split": ["irregular split"],
                  "fragment": ["large at appearance"], "track": ["jump", "flicker"]}
CATEGORIES = ["excluded", *CATEGORY_KINDS]


@dataclass
class ReviewTile:
    scanner: str
    plate: int
    colony_id: int
    category: str
    frame: int
    x: int  # top-left in plate-crop coordinates
    y: int


def tile_frame(category: str, colony: pd.Series, track: pd.DataFrame, n_frames: int) -> int:
    appear = int(colony.appearance_frame)
    if category in ("excluded", "split") or track.empty:
        return int(track.frame.max()) if len(track) else n_frames - 1
    if category == "doublet":
        later = track[track.frame >= appear + 4].frame
        return int(later.iloc[0]) if len(later) else appear
    if category == "fragment":
        return appear
    a = track.sort_values("frame").area_px.to_numpy(float)  # track: largest change between frames
    if len(a) < 2:
        return appear
    change = np.abs(np.log(np.maximum(a[1:], 1) / np.maximum(a[:-1], 1)))
    return int(track.sort_values("frame").frame.iloc[int(np.argmax(change)) + 1])


def category_of(kinds) -> str:
    """The tile category of a colony with these flag kinds (``excluded`` if none matches)."""
    for cat, names in CATEGORY_KINDS.items():
        if set(kinds) & set(names):
            return cat
    return "excluded"


def plate_review_tiles(plate_dir: Path, categories=CATEGORIES, per_plate: int = 20,
                       unreviewed: bool = False) -> list[ReviewTile]:
    """The tiles of one reviewed plate (module docstring), at most ``per_plate`` (spread over the
    categories), without two tiles of the same frame overlapping by more than half."""
    from scanlag.pipeline.flags import is_unreviewed, reviewed_column

    plate_dir = Path(plate_dir)
    info = json.loads((plate_dir / "plate.json").read_text())
    col = pd.read_parquet(plate_dir / "colonies.parquet")
    reviewed = reviewed_column(col)
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    n_frames, size = len(info["times_min"]), info["crop_xy_size"][2]
    wanted: list[tuple[str, pd.Series]] = []
    for i, c in col.iterrows():
        if c.status == ColonyStatus.MANUALLY_EXCLUDED and c.auto_status == ColonyStatus.VALID:
            cat = category_of(kinds_of(c["flags"]))
        elif unreviewed and c.status == ColonyStatus.VALID and is_unreviewed(c["flags"], reviewed.at[i]):
            cat = category_of(set(kinds_of(c["flags"])) - set(kinds_of(reviewed.at[i])))
            if cat == "excluded":  # an unreviewed flag that is no segmentation error (e.g. gap, shape)
                continue
        else:
            continue
        if cat in categories:
            wanted.append((cat, c))
    # round robin over the categories, so one kind doesn't take every slot
    by_cat = {cat: [w for w in wanted if w[0] == cat] for cat in categories}
    order = [by_cat[cat][j] for j in range(max(map(len, by_cat.values()), default=0)) for cat in categories
             if j < len(by_cat[cat])]
    out: list[ReviewTile] = []
    for cat, c in order:
        if len(out) >= per_plate:
            break
        track = tracks[tracks.colony_id == c.colony_id]
        k = tile_frame(cat, c, track, n_frames)
        row = track[track.frame == k]
        cx, cy = (float(row.x.iloc[0]), float(row.y.iloc[0])) if len(row) else (float(c.seed_x), float(c.seed_y))
        x = max(0, int(np.clip(round(cx) - TILE // 2, 0, size - TILE)))
        y = max(0, int(np.clip(round(cy) - TILE // 2, 0, size - TILE)))
        if any(t.frame == k and abs(t.x - x) < TILE // 2 and abs(t.y - y) < TILE // 2 for t in out):
            continue
        out.append(ReviewTile(info["scanner"], int(info["plate"]), int(c.colony_id), cat, int(k), x, y))
    return out


def init_labels(own: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Colony IDs where the tracker assigned pixels; every other detected object gets its own ID from
    OTHER_ID_START (as the MATLAB tiles mark objects that are no kept colony)."""
    from scipy import ndimage as ndi

    out = own.astype(np.uint16)
    others, _ = ndi.label((labels > 0) & (own == 0), structure=np.ones((3, 3)))
    out[others > 0] = (others[others > 0] + OTHER_ID_START - 1).astype(np.uint16)
    return out


def write_tile(plate_dir: Path, tile: ReviewTile, store) -> Path:
    """Write one tile (see the module docstring) through ``store`` (rooted at the output folder)."""
    plate_dir = Path(plate_dir)
    info = json.loads((plate_dir / "plate.json").read_text())
    x0, y0, _ = info["crop_xy_size"]
    raw = Path(info["raw_dir"])
    name = f"tile_{tile.scanner}_p{tile.plate}_c{tile.colony_id}_{tile.category}"

    def crop(k: int) -> np.ndarray:
        dx, dy = info["shift_xy"][k]
        oy, ox = y0 + dy + tile.y, x0 + dx + tile.x
        return read_window(raw / info["raw_scans"][k], slice(oy, oy + TILE), slice(ox, ox + TILE))

    window = (tile.frame, slice(tile.y, tile.y + TILE), slice(tile.x, tile.x + TILE))

    def read(name: str) -> np.ndarray:  # padded to TILE x TILE at the crop's edge
        a = np.asarray(zarr.open_array(str(plate_dir / name), mode="r")[window])
        return np.pad(a, ((0, TILE - a.shape[0]), (0, TILE - a.shape[1])))

    labels = read("labels.zarr")
    own = read("colonies.zarr") if (plate_dir / "colonies.zarr").exists() else labels
    store.write_image(crop(tile.frame), "manual", name, "image.png")
    store.write_image(crop(0), "manual", name, "background.png")
    store.write_image(crop(max(tile.frame - PREVIOUS_GAP, 0)), "manual", name, "previous.png")
    store.write_image(init_labels(own, labels), "manual", name, "labels_init.png")
    plate_id = f"{tile.scanner}__plate_{tile.plate}"
    col = pd.read_parquet(plate_dir / "colonies.parquet").set_index("colony_id").loc[tile.colony_id]
    store.write_json({
        "plate_id": plate_id, "category": tile.category, "scanner": tile.scanner, "plate": tile.plate,
        "colony_id": tile.colony_id, "frame": tile.frame, "x": tile.x, "y": tile.y,
        "time_min": info["times_min"][tile.frame], "raw_scan": info["raw_scans"][tile.frame], "raw_dir": str(raw),
        "raw_x": int(x0 + info["shift_xy"][tile.frame][0] + tile.x),
        "raw_y": int(y0 + info["shift_xy"][tile.frame][1] + tile.y),
        "tile_size": TILE, "flags": col["flags"], "status": int(col.status), "results": str(plate_dir.parent.parent),
    }, "manual", name, "tile.json")
    mask = store.path("plates", plate_id, "mask.png")
    if not mask.exists():
        shutil.copyfile(plate_dir / "mask.png", mask)
    return store.path("manual", name)

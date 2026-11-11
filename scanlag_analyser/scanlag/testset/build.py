"""Build the reference test set from legacy MATLAB results.

For every analyzed plate, MATLAB detection + tracking is replayed (bit-exact, see
``regenerate.py``) to recover per-frame label images. Only colonies the user kept
(``IgnoredColonies == 0``) are labelled. Everything else MATLAB segmented is marked IGNORE, so
an evaluation neither rewards nor penalizes it:

* colonies excluded manually, near the border, or with zero final area,
* blobs holding more than one colony (merged), since MATLAB can't say which pixel is whose,
* blobs MATLAB never accepted as a colony (below the 10 px minimum, no seed inside).

Label images are in the coordinates of the MATLAB plate crop. ``plate.json`` records where that
crop sits in each raw scan, so the labels can be compared with results computed from raw data.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from scanlag.io.matlab_import import ColonyStatus, MatlabPlate, load_plate
from scanlag.store import ResultsStore
from scanlag.testset.regenerate import LegacyReplay, check_parity, read_rgb

IGNORE = np.iinfo(np.uint16).max
ROI_SUFFIX = re.compile(r"_\d+\.tiff?$")


def plate_id(plate_dir: Path, data_root: Path) -> str:
    return "__".join(Path(plate_dir).relative_to(data_root).parts).replace("__ROI__", "__")


def raw_scan_path(plate: MatlabPlate, frame: int) -> Path:
    """The raw scan a ROI image was cut from (``<scan>_<plate>.tif`` -> ``<raw dir>/<scan>.tiff``).

    Usually the raw dir is ``../..`` of the plate dir. Some prep scripts wrote the ROI folder
    into a sibling condition folder, so siblings are searched too.
    """
    stem = ROI_SUFFIX.sub("", plate.files[frame])
    home = plate.plate_dir.parent.parent
    siblings = sorted(p for p in home.parent.iterdir() if p.is_dir() and p != home)
    for raw_dir in [home, *siblings]:
        for ext in (".tiff", ".tif"):
            candidate = raw_dir / (stem + ext)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"raw scan for {plate.files[frame]} not found in {home} or its siblings")


def round_half_away(x: np.ndarray) -> np.ndarray:
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def raw_offsets(plate: MatlabPlate, roi0: np.ndarray) -> tuple[np.ndarray, bool]:
    """(T, 2) per-frame (x, y) position of the crop's top-left pixel in the raw scan.

    The frame-0 position is found by template matching. Later frames add MATLAB's cumulative
    registration shift (CropROI.m shifts by round(cumsum(motions))).
    """
    raw0 = tifffile.imread(raw_scan_path(plate, 0))[..., :3]
    gray = lambda a: cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)
    scores = cv2.matchTemplate(gray(raw0), gray(roi0), cv2.TM_SQDIFF)
    x0, y0 = cv2.minMaxLoc(scores)[2]
    h, w = roi0.shape[:2]
    exact = bool(np.array_equal(raw0[y0 : y0 + h, x0 : x0 + w], roi0))
    shift = np.zeros((plate.n_frames, 2))
    if plate.motions is not None and len(plate.motions) >= plate.n_frames:
        shift = round_half_away(np.cumsum(plate.motions[: plate.n_frames], axis=0))
    return np.array([x0, y0]) + shift, exact


def build_plate(plate_dir: Path, data_root: Path, store: ResultsStore) -> dict:
    plate = load_plate(plate_dir)
    pid = plate_id(plate_dir, data_root)
    replay = LegacyReplay(plate)
    keep = plate.status == ColonyStatus.VALID
    times = plate.times_min

    labels_out = store.create_labels((plate.n_frames, *replay.background.shape[:2]), "plates", pid, "labels.zarr")
    merged_frames = []
    for k, _, labels, blob_to_colony in replay.frames():
        lut = np.full(len(blob_to_colony), IGNORE, np.uint16)
        lut[0] = 0
        single = blob_to_colony >= 0
        kept = single.copy()
        kept[single] = keep[blob_to_colony[single]]
        lut[kept] = blob_to_colony[kept] + 1
        labels_out[k] = lut[labels]
        merged_frames.append(blob_to_colony)

    area, centroid, blob = replay.tracker.tracks.as_arrays()
    parity = check_parity(plate, area)
    offsets, origin_exact = raw_offsets(plate, replay.background)

    # A colony is merged in frame k if its blob is labelled MERGED in that frame.
    merged = np.zeros_like(area, dtype=bool)
    for k, blob_to_colony in enumerate(merged_frames):
        present = blob[k] > 0
        merged[k, present] = blob_to_colony[blob[k, present]] == -2

    rows = []
    t_idx, c_idx = np.nonzero(area)
    for k, c in zip(t_idx, c_idx):
        cx, cy = centroid[k, c] - 1  # to 0-based pixel coordinates
        rows.append((k, times[k], c + 1, area[k, c], cx, cy, cx + offsets[k, 0], cy + offsets[k, 1], merged[k, c]))
    tracks = pd.DataFrame(rows, columns=["frame", "time_min", "colony_id", "area_px", "x", "y", "x_raw", "y_raw", "merged"])
    store.write_table(tracks, "plates", pid, "tracks.parquet")

    appear = np.argmax(area > 0, axis=0)
    ever_merged = merged.any(axis=0)
    seeds = np.array(replay.tracker.tracks.seeds).reshape(-1, 2)
    colonies = pd.DataFrame(
        {
            "colony_id": np.arange(1, plate.n_colonies + 1),
            "status": plate.status,
            "status_name": [ColonyStatus(s).name for s in plate.status],
            "appearance_frame": appear,
            "appearance_min": times[appear],
            "first_merged_frame": np.where(ever_merged, np.argmax(merged, axis=0), -1),
            "final_area_px": area[-1],
            "seed_x": seeds[:, 1],
            "seed_y": seeds[:, 0],
        }
    )
    store.write_table(colonies, "plates", pid, "colonies.parquet")
    store.write_mask(replay.mask, "plates", pid, "mask.png")
    info = {
        "plate_id": pid,
        "matlab_plate_dir": plate_dir,
        "description": plate.description,
        "raw_scans": [raw_scan_path(plate, k).name for k in range(plate.n_frames)],
        "raw_dir": raw_scan_path(plate, 0).parent,
        "raw_offset_xy": offsets,
        "raw_origin_exact": origin_exact,
        "times_min": times,
        "threshold": plate.threshold,
        "limits": plate.limits,
        "mask_kind": "full" if plate.full_mask is not None else "circle",
        "label_values": {"background": 0, "colony": "MATLAB colony index (1-based)", "ignore": int(IGNORE)},
        "parity": vars(parity),
    }
    store.write_json(info, "plates", pid, "plate.json")
    return {
        "plate_id": pid,
        "experiment": pid.split("__")[0],
        "n_frames": plate.n_frames,
        "n_colonies": plate.n_colonies,
        "n_kept": int(keep.sum()),
        "n_manually_excluded": int((plate.status == ColonyStatus.MANUALLY_EXCLUDED).sum()),
        "parity_exact": parity.exact,
        "parity_frac_equal": parity.frac_cells_equal,
        "raw_origin_exact": origin_exact,
    }

"""Read results produced by the legacy MATLAB ScanLag pipeline (read-only).

The MATLAB code stores everything for one plate in ``<plate_dir>/<plate_dir_name>_data.mat``.
Coordinates in that file are MATLAB-style (1-based, x = column, y = row).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import scipy.io as sio

MATLAB_EPOCH_OFFSET = 719529  # datenum of 1970-01-01
MINUTES_PER_DAY = 24 * 60


class ColonyStatus(IntEnum):
    """Codes of the MATLAB ``IgnoredColonies`` vector."""

    VALID = 0
    NEAR_BORDER = 1
    MANUALLY_EXCLUDED = 2
    ZERO_FINAL_AREA = 3


@dataclass
class MatlabPlate:
    plate_dir: Path
    files: list[str]  # ROI image names, in processing order
    file_datenums: np.ndarray  # (T,) MATLAB datenums
    starting_datenum: float | None
    area: np.ndarray  # (T, N) pixels; 0 = colony not present in that frame
    centroid: np.ndarray  # (T, N, 2) 1-based (x, y); 0 = absent
    bbox: np.ndarray  # (T, N, 4) MATLAB BoundingBox
    status: np.ndarray  # (N,) ColonyStatus codes
    threshold: float
    limits: np.ndarray  # (2,) contrast-stretch limits in [0, 1]
    full_mask: np.ndarray | None  # (H, W) bool, if a polygon mask was drawn
    circle: tuple[float, float, float] | None  # (x, y, r), 1-based, if no full mask
    description: str
    motions: np.ndarray | None  # (T, 2) per-frame (u, v) registration from PrevCut

    @property
    def n_frames(self) -> int:
        return len(self.files)

    @property
    def n_colonies(self) -> int:
        return self.area.shape[1]

    @property
    def times_min(self) -> np.ndarray:
        """Frame times in minutes since the experiment start (MATLAB GetTimes semantics)."""
        start = self.starting_datenum or 0.0
        if not start:
            return self.file_datenums.copy()
        return np.round((self.file_datenums - start) * MINUTES_PER_DAY)

    def image_path(self, frame: int) -> Path:
        return self.plate_dir / self.files[frame]

    def working_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Port of GetMask.m."""
        if self.full_mask is not None:
            return self.full_mask
        x, y, r = self.circle
        rows, cols = np.mgrid[1 : shape[0] + 1, 1 : shape[1] + 1]
        return np.hypot(cols - x, rows - y) <= r


def data_file(plate_dir: Path) -> Path:
    """Port of GetDataName.m."""
    plate_dir = Path(plate_dir)
    return plate_dir / f"{plate_dir.name}_data.mat"


def _struct_field(struct, name: str):
    return getattr(struct, name) if struct is not None and name in struct._fieldnames else None


def _pad_frames(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """MATLAB grows Area/Centroid/BBox only up to the last frame in which any colony was
    seen, so trailing colony-free frames are missing. Restore them as zeros."""
    missing = n_frames - arr.shape[0]
    if missing < 0:
        raise ValueError(f"tracking has {arr.shape[0]} frames but only {n_frames} files")
    return np.pad(arr, [(0, missing)] + [(0, 0)] * (arr.ndim - 1))


def load_plate(plate_dir: str | Path) -> MatlabPlate:
    plate_dir = Path(plate_dir)
    path = data_file(plate_dir)
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    # Unsqueezed copies keep the (T, N[, k]) layout when T or N is 1.
    tracks = sio.loadmat(path, variable_names=["Area", "Centroid", "BBox"])

    files = np.atleast_1d(m["FilesName"]).tolist()
    n_frames = len(files)
    area = np.asarray(tracks["Area"], dtype=np.int64)
    n_colonies = area.shape[1]
    area = _pad_frames(area, n_frames)
    centroid = _pad_frames(np.asarray(tracks["Centroid"], dtype=float).reshape(-1, n_colonies, 2), n_frames)
    bbox = _pad_frames(np.asarray(tracks["BBox"], dtype=float).reshape(-1, n_colonies, 4), n_frames)

    circle = None
    full_mask = m.get("FullMask")
    if full_mask is not None:
        full_mask = np.asarray(full_mask) != 0
    else:
        c = m.get("CircleMask", m.get("PlateCirc"))
        circle = (float(c.X), float(c.Y), float(c.R))

    motions = None
    prev_cut = m.get("PrevCut")
    raw_motions = _struct_field(prev_cut, "Motions")
    if raw_motions is not None:
        motions = np.asarray(raw_motions[:, 1:], dtype=float)

    starting = m.get("StartingTime")
    return MatlabPlate(
        plate_dir=plate_dir,
        files=files,
        file_datenums=np.atleast_1d(np.asarray(m["FilesDateTime"], dtype=float)),
        starting_datenum=float(starting) if starting is not None else None,
        area=area,
        centroid=centroid,
        bbox=bbox,
        status=np.atleast_1d(np.asarray(m["IgnoredColonies"], dtype=np.int64)),
        threshold=float(m["TH"]),
        limits=np.asarray(m["Limits"], dtype=float),
        full_mask=full_mask,
        circle=circle,
        description=str(m.get("Description", "")),
        motions=motions,
    )


def find_analyzed_plates(root: str | Path) -> list[Path]:
    """All plate directories under ``root`` whose data file contains tracking results."""
    plates = []
    for mat in sorted(Path(root).rglob("*_data.mat")):
        if mat.name != f"{mat.parent.name}_data.mat":
            continue
        if "Area" in {name for name, *_ in sio.whosmat(mat)}:
            plates.append(mat.parent)
    return plates

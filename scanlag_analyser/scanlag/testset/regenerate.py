"""Re-run the MATLAB detection + tracking on a legacy plate, from the ROI images MATLAB used.

This recovers the per-frame label images MATLAB never saved, and verifies the Python port
by comparing with the stored ``Area`` matrix.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import tifffile

from scanlag.detect.classical import binarize, label_column_major
from scanlag.io.matlab_import import MatlabPlate
from scanlag.track.seed_tracker import SeedTracker


@dataclass
class ParityReport:
    n_colonies_matlab: int
    n_colonies_python: int
    n_frames_compared: int
    exact: bool
    frac_cells_equal: float  # fraction of (frame, colony) area entries that agree


def read_rgb(path) -> np.ndarray:
    return tifffile.imread(path)[..., :3]


NOT_A_COLONY = -1
MERGED = -2


class LegacyReplay:
    """Replays MATLAB detection + tracking on one plate, frame by frame."""

    def __init__(self, plate: MatlabPlate):
        self.plate = plate
        self.background = read_rgb(plate.image_path(0))
        self.mask = plate.working_mask(self.background.shape[:2])
        self.tracker = SeedTracker()

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (frame, image, labels, blob_to_colony).

        ``blob_to_colony[b]`` is the 0-based MATLAB colony index of blob ``b``, or
        NOT_A_COLONY (no seed inside), or MERGED (several seeds inside).
        """
        plate = self.plate
        for k in range(plate.n_frames):
            image = self.background if k == 0 else read_rgb(plate.image_path(k))
            binary = binarize(image, self.background, plate.limits, plate.threshold, self.mask)
            labels, n = label_column_major(binary)
            self.tracker.update(labels, n)
            blob_to_colony = np.full(n + 1, NOT_A_COLONY, np.int64)
            for colony, blob in enumerate(self.tracker.tracks.blob[k]):
                if blob:
                    blob_to_colony[blob] = colony if blob_to_colony[blob] == NOT_A_COLONY else MERGED
            yield k, image, labels, blob_to_colony


def check_parity(plate: MatlabPlate, area_python: np.ndarray) -> ParityReport:
    t = min(area_python.shape[0], plate.area.shape[0])
    n = min(area_python.shape[1], plate.area.shape[1])
    a, b = area_python[:t, :n], plate.area[:t, :n]
    same_shape = area_python.shape == plate.area.shape
    return ParityReport(
        n_colonies_matlab=plate.n_colonies,
        n_colonies_python=area_python.shape[1],
        n_frames_compared=t,
        exact=same_shape and bool(np.array_equal(a, b)),
        frac_cells_equal=float((a == b).mean()) if a.size else 1.0,
    )

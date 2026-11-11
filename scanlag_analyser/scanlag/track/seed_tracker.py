"""Seed-point colony tracker, a faithful port of Images2Colonies.m.

A colony is identified by the (rounded) centroid of its first detection. In every later frame,
the blob covering that seed pixel is assigned to the colony. Blobs not covering any seed and
with at least ``min_area`` pixels become new colonies. Several seeds in one blob means the
colonies merged, and they all report the merged blob's properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

MIN_AREA = 10
NEAR_BORDER_PX = 15


@dataclass
class SeedTracks:
    seeds: list[tuple[int, int]] = field(default_factory=list)  # 0-based (row, col)
    area: list[np.ndarray] = field(default_factory=list)  # per frame, (N,) at that time
    centroid: list[np.ndarray] = field(default_factory=list)  # per frame, (N, 2) 1-based (x, y)
    blob: list[np.ndarray] = field(default_factory=list)  # per frame, (N,) blob label or 0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(area[T, N], centroid[T, N, 2], blob[T, N]), zero-padded for late colonies."""
        n = len(self.seeds)
        t = len(self.area)
        area = np.zeros((t, n), np.int64)
        centroid = np.zeros((t, n, 2))
        blob = np.zeros((t, n), np.int64)
        for k in range(t):
            m = len(self.area[k])
            area[k, :m], centroid[k, :m], blob[k, :m] = self.area[k], self.centroid[k], self.blob[k]
        return area, centroid, blob


def matlab_round(x: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(x) + 0.5).astype(np.int64)  # values here are positive


def blob_properties(labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Area and 1-based (x, y) centroid per label 1..n (regionprops 'basic'); NaN centroid for
    absent labels. Integer coordinate sums are exact in float64, so this equals ndimage.mean."""
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    area = np.bincount(ids, minlength=n + 1)[1 : n + 1]
    sx = np.bincount(ids, weights=cols, minlength=n + 1)[1 : n + 1]
    sy = np.bincount(ids, weights=rows, minlength=n + 1)[1 : n + 1]
    with np.errstate(invalid="ignore"):
        centroid = np.column_stack([sx / area + 1, sy / area + 1])
    return area.astype(np.int64), centroid.reshape(n, 2)


class SeedTracker:
    def __init__(self, min_area: int = MIN_AREA):
        self.min_area = min_area
        self.tracks = SeedTracks()

    def update(self, labels: np.ndarray, n: int) -> None:
        """Add one frame. ``labels`` must be numbered as bwlabel does (column-major)."""
        area, centroid = blob_properties(labels, n)
        seeds = self.tracks.seeds
        blob_of_seed = np.array([labels[r, c] for r, c in seeds], dtype=np.int64)

        is_new = np.ones(n + 1, bool)
        is_new[blob_of_seed] = False
        new_blobs = [b for b in range(1, n + 1) if is_new[b] and area[b - 1] >= self.min_area]
        for b in new_blobs:
            x, y = matlab_round(centroid[b - 1])
            seeds.append((int(y) - 1, int(x) - 1))

        blobs = np.concatenate([blob_of_seed, new_blobs]).astype(np.int64)
        # Row 0 stands for "no blob", so absent colonies get zeros (also when n == 0).
        self.tracks.area.append(np.concatenate([[0], area])[blobs])
        self.tracks.centroid.append(np.vstack([np.zeros((1, 2)), centroid])[blobs])
        self.tracks.blob.append(blobs)


def border_status(mask: np.ndarray, seeds: list[tuple[int, int]], radius: int = NEAR_BORDER_PX) -> np.ndarray:
    """Port of FindColoniesInWorkingArea.m: True if the disk around the seed is fully inside the mask."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx**2 + yy**2) <= radius**2
    inside = ndi.binary_erosion(mask, structure=disk, border_value=0)
    return np.array([inside[r, c] for r, c in seeds], dtype=bool)

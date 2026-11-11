"""Footprint colony tracker: each colony keeps its own pixels, also after it merges.

Colonies don't move and only grow, so a colony is followed by its footprint (its pixels in the
previous frame) rather than by a single seed pixel. Per frame, each detected blob is matched to
the colonies whose previous footprint it overlaps:

* none: a new colony, if the blob has at least ``min_area`` pixels (seed = its rounded centroid,
  as in the seed tracker)
* one: the whole blob belongs to that colony. Several blobs may belong to one colony (the
  detector broke it into fragments); they are all its pixels
* several (a merge): the blob is split between them. Each colony keeps the blob pixels it owned
  in the previous frame, and every new pixel goes to the nearest colony front (watershed on the
  distance to the previous footprints, inside the blob). Two neighbours grow by about the same
  step between scans, so this boundary follows the line where their fronts met, and it is
  carried forward frame by frame.

A colony owns a blob if the blob covers its seed pixel or at least ``min_overlap`` of the
colony's previous footprint. A one-pixel overlap from alignment jitter thus doesn't count as a
merge. A blob that overlaps colonies only below that fraction goes to the one it overlaps most,
so a fragment never starts a new colony.

A colony that is already touching a neighbour in the first frame it is detected can't be
separated: it is absorbed into the neighbour (the neighbour's track usually gets a "jump" flag).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from scanlag.track.seed_tracker import MIN_AREA, SeedTracks, blob_properties, matlab_round

MIN_OVERLAP = 0.1  # fraction of the colony's previous footprint


def colony_properties(own: np.ndarray, n_col: int) -> tuple[np.ndarray, np.ndarray]:
    """Area and 1-based (x, y) centroid per colony 1..n_col; zeros for colonies absent here."""
    rows, cols = np.nonzero(own)
    ids = own[rows, cols]
    area = np.bincount(ids, minlength=n_col + 1)[1:]
    sx = np.bincount(ids, weights=cols, minlength=n_col + 1)[1:]
    sy = np.bincount(ids, weights=rows, minlength=n_col + 1)[1:]
    present = area > 0
    centroid = np.zeros((n_col, 2))
    centroid[present] = np.column_stack([sx, sy])[present] / area[present, None] + 1
    return area.astype(np.int64), centroid


class FootprintTracker:
    def __init__(self, min_area: int = MIN_AREA, min_overlap: float = MIN_OVERLAP):
        self.min_area = min_area
        self.min_overlap = min_overlap
        self.tracks = SeedTracks()  # blob: the detected label holding most of the colony's pixels
        self.own: np.ndarray | None = None  # colony label image of the last frame (0 = none)

    def update(self, labels: np.ndarray, n: int) -> np.ndarray:
        """Add one frame of detected ``labels`` (1..n). Returns its colony label image."""
        labels = np.asarray(labels, np.int64)
        seeds = self.tracks.seeds
        prev = np.zeros(labels.shape, np.int64) if self.own is None else self.own
        owners = self._owners(labels, n, prev)

        lut = np.zeros(n + 1, np.int64)
        for b, cs in owners.items():
            if len(cs) == 1:
                lut[b] = cs[0]
        # New colonies go in the same lookup table: their blobs are disjoint from the split ones
        blob_area, blob_centroid = blob_properties(labels, n)
        for b in range(1, n + 1):
            if b not in owners and blob_area[b - 1] >= self.min_area:
                x, y = matlab_round(blob_centroid[b - 1])
                seeds.append((int(y) - 1, int(x) - 1))
                lut[b] = len(seeds)
        own = lut[labels]

        objects = ndi.find_objects(labels, max_label=n)
        for b, cs in owners.items():
            if len(cs) > 1:
                self._split(labels, prev, b, cs, objects[b - 1], own)

        n_col = len(seeds)
        area, centroid = colony_properties(own, n_col)
        self.tracks.area.append(area)
        self.tracks.centroid.append(centroid)
        self.tracks.blob.append(self._main_blob(labels, own, n_col))
        self.own = own
        return own

    def _owners(self, labels: np.ndarray, n: int, prev: np.ndarray) -> dict[int, list[int]]:
        """blob -> colonies (1-based) it belongs to, for blobs that overlap any colony."""
        n_col = len(self.tracks.seeds)
        if n == 0 or n_col == 0:
            return {}
        both = (labels > 0) & (prev > 0)
        pairs, counts = np.unique(labels[both] * (n_col + 1) + prev[both], return_counts=True)
        blob, col = pairs // (n_col + 1), pairs % (n_col + 1)
        prev_area = np.bincount(prev.ravel(), minlength=n_col + 1)
        strong = counts >= self.min_overlap * prev_area[col]

        owners: dict[int, set[int]] = {}
        for b, c in zip(blob[strong], col[strong]):
            owners.setdefault(int(b), set()).add(int(c))
        for c, (r, cc) in enumerate(self.tracks.seeds, start=1):
            b = int(labels[r, cc])
            if b:
                owners.setdefault(b, set()).add(c)
        for b in np.unique(blob):  # weak overlaps only: give the fragment to its main colony
            if int(b) not in owners:
                here = blob == b
                owners[int(b)] = {int(col[here][counts[here].argmax()])}
        return {b: sorted(cs) for b, cs in owners.items()}

    def _split(self, labels, prev, b: int, cs: list[int], sl, own) -> None:
        region = labels[sl] == b
        markers = np.where(region & np.isin(prev[sl], cs), prev[sl], 0)
        for c in cs:  # an owner by seed only (absent last frame) grows from its seed pixel
            r, cc = self.tracks.seeds[c - 1]
            r, cc = r - sl[0].start, cc - sl[1].start
            if not (markers == c).any() and 0 <= r < region.shape[0] and 0 <= cc < region.shape[1]:
                markers[r, cc] = c
        distance, (ri, ci) = ndi.distance_transform_edt(markers == 0, return_indices=True)
        split = watershed(distance, markers, mask=region)
        split = np.where(split > 0, split, markers[ri, ci])  # pieces of the blob without a marker
        own[sl][region] = split[region]

    @staticmethod
    def _main_blob(labels: np.ndarray, own: np.ndarray, n_col: int) -> np.ndarray:
        main = np.zeros(n_col, np.int64)
        both = own > 0
        if both.any():
            base = labels.max() + 1
            pairs, counts = np.unique(own[both].astype(np.int64) * base + labels[both], return_counts=True)
            col, blob = pairs // base, pairs % base
            order = np.lexsort((counts, col))  # per colony, the largest count last
            main[col[order] - 1] = blob[order]
        return main

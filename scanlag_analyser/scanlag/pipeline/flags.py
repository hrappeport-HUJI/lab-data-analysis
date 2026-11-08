"""Flag colonies whose track or shape looks suspicious, so the reviewer checks them first.

Track checks use the colony's own area. With the seed tracker, frames where it is merged with
another colony report the blob's area, so they are skipped; the footprint tracker gives a merged
colony its own share of the blob, so those frames are checked too (a bad split shows as a
shrink or a jump):

* gap: the colony disappears for some frames after it appeared
* transient merge: it shares a blob with another colony in some frames and is separate again
  later (the segmentation flickers; with the seed tracker the area jumps to the blob's and back)
* shrinks: the area drops below SHRINK x the largest area seen so far (colonies only grow)
* jump: the area grows by more than JUMP x between consecutive frames (merge with debris,
  condensation, a segmentation switch)
* stays small: still below SMALL px at the end (dust, a speck, a bubble)
* large at appearance: already above APPEAR_MAX px in its first frame. New colonies start at
  about 10-15 px; a bigger one is usually a piece split off a neighbouring colony for one frame
  (the two "merge" in the next frame and share the blob from then on)

Shape checks use the colony's mask in the last frame it is not merged (from colonies.zarr with
the footprint tracker, else its blob in labels.zarr):

* irregular: solidity (area / convex-hull area) below SOLIDITY. Colonies are round and convex
* elongated: eccentricity above ECCENTRICITY (scratches, writing, rim reflections)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops

SHRINK = 0.75
JUMP = 2.0
JUMP_MIN_AREA = 15  # px; jumps of tiny blobs are noise
SHRINK_MIN_AREA = 20
SMALL = 20
APPEAR_MAX = 20  # px
SOLIDITY = 0.85
ECCENTRICITY = 0.85
SHAPE_MIN_AREA = 30


def track_flags(a: np.ndarray, merged: np.ndarray, merged_area_is_own: bool = False) -> list[str]:
    """Flags for one colony's area series (all frames)."""
    present = np.nonzero(a > 0)[0]
    if not len(present):
        return []
    a, merged = a[present[0] :].astype(float), merged[present[0] :]
    flags = []
    if a[0] > APPEAR_MAX:
        flags.append(f"large at appearance ({int(a[0])} px)")
    gaps = int(((a == 0) & ~merged).sum())
    if gaps:
        flags.append(f"gap ({gaps} frames)")
    splits = int((merged[:-1] & ~merged[1:] & (a[1:] > 0)).sum())
    if splits:  # merged, then separate again: the segmentation flickers between frames
        flags.append(f"transient merge ({int(merged.sum())} frames)")
    own = np.where((merged & ~merged_area_is_own) | (a == 0), np.nan, a)
    if np.isfinite(own).sum() >= 2:
        peak = np.fmax.accumulate(np.nan_to_num(own))
        drop = own < SHRINK * peak
        drop &= peak >= SHRINK_MIN_AREA
        if drop.any():
            worst = np.nanmin(own[drop] / peak[drop])
            flags.append(f"shrinks to {worst:.0%}")
        ratio = own[1:] / own[:-1]
        jump = (ratio > JUMP) & (own[:-1] >= JUMP_MIN_AREA)
        if jump.any():
            flags.append(f"jump x{np.nanmax(ratio[jump]):.1f}")
    if a[-1] > 0 and a[-1] < SMALL and not merged[-1]:
        flags.append(f"stays small ({int(a[-1])} px)")
    return flags


def shape_flags(region) -> list[str]:
    if region is None or region.area < SHAPE_MIN_AREA:
        return []
    flags = []
    if region.solidity < SOLIDITY:
        flags.append(f"irregular shape (solidity {region.solidity:.2f})")
    if region.eccentricity > ECCENTRICITY:
        flags.append(f"elongated (eccentricity {region.eccentricity:.2f})")
    return flags


def colony_flags(area: np.ndarray, merged: np.ndarray, blob: np.ndarray, labels, own=None) -> pd.DataFrame:
    """One row per colony: ``flags`` (text, '' if none) and ``n_flags``.

    ``area``, ``merged``, ``blob`` are [T, N] tracker arrays; ``labels`` is the [T, H, W] label stack.
    ``own`` is the [T, H, W] colony ID stack of the footprint tracker (None for the seed tracker).
    """
    n_frames, n = area.shape
    shape_frame = np.full(n, -1)
    for c in range(n):
        alone = np.nonzero((area[:, c] > 0) & ~merged[:, c])[0]
        if len(alone):
            shape_frame[c] = alone[-1]
    regions: dict[int, object] = {}
    for k in np.unique(shape_frame[shape_frame >= 0]):
        lab = np.asarray((labels if own is None else own)[int(k)])
        props = {r.label: r for r in regionprops(lab)}
        for c in np.nonzero(shape_frame == k)[0]:
            regions[c] = props.get(int(blob[k, c]) if own is None else c + 1)
    rows = []
    for c in range(n):
        flags = track_flags(area[:, c], merged[:, c], own is not None) + shape_flags(regions.get(c))
        rows.append({"colony_id": c + 1, "flags": "; ".join(flags), "n_flags": len(flags)})
    return pd.DataFrame(rows, columns=["colony_id", "flags", "n_flags"])

"""Flag colonies whose track or shape looks suspicious, so the reviewer checks them first.

Track checks use the colony's own area. With the seed tracker, frames where it is merged with
another colony report the blob's area, so they are skipped; the footprint tracker gives a merged
colony its own share of the blob, so those frames are checked too (a bad split shows as a
shrink or a jump):

* gap: the colony disappears for some frames after it appeared
* transient merge (seed tracker only): it shares a blob with another colony in some frames and is
  separate again later, so its area jumps to the blob's and back. With the footprint tracker a
  merged colony keeps its own area, and touching neighbours (most merges on a dense plate) are
  harmless; a merge that does damage shows as a shrink, a jump or flicker instead
* shrinks: the area drops below SHRINK x the largest area seen so far (colonies only grow)
* jump: the area grows by more than JUMP x between consecutive frames (merge with debris,
  condensation, a segmentation switch)
* flicker: the area changes by more than FLICKER_STEP between consecutive frames at least
  FLICKER_MIN times (from FLICKER_MIN_AREA px): the colony keeps gaining and losing a neighbour's
  pixels, typically next to a colony outside the working area. Its area track is unusable
* stays small: still below SMALL px at the end (dust, a speck, a bubble)
* large at appearance: already above APPEAR_MAX px in its first frame. New colonies start at
  about 10-15 px; a bigger one is usually a piece split off a neighbouring colony for one frame
  (the two "merge" in the next frame and share the blob from then on)

Plate-level checks (``colony_flags`` with ``times``):

* possible doublet: FIXED_AGE_H after its appearance (and not merged until then), the colony is at
  least DOUBLET_RATIO x the median area of the colonies that appeared within DOUBLET_WINDOW_H of it,
  at the same age. Two colonies that already touch when they are first detected are tracked as one
  (the tracker cannot separate them), so the plate has one colony fewer and the growth track is
  the pair's

Bad frames (``bad_frames``): a scan in which many established colonies are suddenly smaller or
larger than in the frames before and after (a blurred or badly lit scan, condensation). Their
areas are ignored by the shrink, jump, flicker and gap checks, and the frames are listed in
plate.json (``bad_frames``) and marked in the tracks table (``bad_frame``).

Resolving: after checking a flagged colony, the reviewer can mark its flags as resolved (review window:
the check box in the flagged list, or Ctrl+O). ``colonies.parquet``'s ``flags_resolved`` keeps the flag
kinds resolved (``; ``-separated). A colony stays resolved while all its current flag kinds are among
them, so a new kind raised by ``--reflag`` brings it back (``is_unresolved``).

Shape checks use the colony's mask in the last frame it is not merged (from colonies.zarr with
the footprint tracker, else its blob in labels.zarr):

* irregular: solidity (area / convex-hull area) below SOLIDITY. Colonies are round and convex
* elongated: eccentricity above ECCENTRICITY (scratches, writing, rim reflections)
* irregular split (footprint tracker): the colony is merged in the last frame and its own share of
  the blob has solidity below SPLIT_SOLIDITY. A share bounded by the straight lines where
  neighbouring fronts met stays convex; an L-shaped or ragged share means the split went wrong
"""

from __future__ import annotations

import re

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
SPLIT_SOLIDITY = 0.8
FLICKER_STEP = 0.4  # relative change between consecutive frames
FLICKER_MIN = 5  # changes
FLICKER_MIN_AREA = 100  # px
FIXED_AGE_H = 4.0
DOUBLET_RATIO = 1.6
DOUBLET_WINDOW_H = 1.0
DOUBLET_MIN_PEERS = 10
BAD_FRAME_STEP = 0.15  # an established colony's area off by more than this from its neighbours' mean ...
BAD_FRAME_FRACTION = 0.05  # ... in at least this fraction of established colonies
BAD_FRAME_MIN_AREA = 50  # px: established colonies


def kinds_of(flags: str | None) -> list[str]:
    """'gap (3 frames); jump x2.1' -> ['gap', 'jump']: the flag names without their numbers."""
    return [re.split(r" \(| x\d| to \d", f)[0] for f in (flags or "").split("; ") if f]


def is_unresolved(flags: str | None, resolved: str | None) -> bool:
    """True if the colony has a flag kind that is not among the resolved ones."""
    done = set(kinds_of(resolved))
    return any(k not in done for k in kinds_of(flags))


def unresolved_mask(colonies: pd.DataFrame) -> np.ndarray:
    """Per colony row: has unresolved flags (``flags_resolved`` missing: nothing resolved)."""
    resolved = colonies["flags_resolved"] if "flags_resolved" in colonies else pd.Series("", index=colonies.index)
    return np.array([is_unresolved(f, r) for f, r in zip(colonies["flags"], resolved.fillna(""))], bool)


def bad_frames(area: np.ndarray) -> list[int]:
    """Frames (not the first or last) in which at least BAD_FRAME_FRACTION of the established colonies
    (at least BAD_FRAME_MIN_AREA px in both neighbouring frames) differ by more than BAD_FRAME_STEP from
    the mean of their areas in the frames before and after. ``area`` is the [T, N] tracker array."""
    a = np.array(area, float)
    if a.shape[0] < 3 or a.shape[1] == 0:
        return []
    bad = []
    while True:  # the worst frame first: a bad frame also skews its neighbours' comparison
        prev, cur, nxt = a[:-2], a[1:-1], a[2:]
        est = (prev >= BAD_FRAME_MIN_AREA) & (nxt >= BAD_FRAME_MIN_AREA)
        ref = (prev + nxt) / 2
        off = est & (np.abs(cur - ref) > BAD_FRAME_STEP * ref)
        n_est = est.sum(axis=1)
        frac = np.divide(off.sum(axis=1), n_est, out=np.zeros(len(n_est)), where=n_est >= 10)
        if not (frac >= BAD_FRAME_FRACTION).any():
            return sorted(bad)
        cand = np.nonzero(frac >= BAD_FRAME_FRACTION)[0]
        dev = [np.median(np.abs(cur[j] - ref[j])[est[j]] / ref[j][est[j]]) for j in cand]
        k = int(cand[int(np.argmax(dev))])
        bad.append(k + 1)
        a[k + 1] = ref[k]  # replaced by its neighbours' mean, then look again


def track_flags(a: np.ndarray, merged: np.ndarray, merged_area_is_own: bool = False, bad=None) -> list[str]:
    """Flags for one colony's area series (all frames). ``bad``: frame indices whose areas are ignored."""
    present = np.nonzero(a > 0)[0]
    if not len(present):
        return []
    skip = np.zeros(len(a), bool)
    if bad is not None and len(bad):
        skip[np.asarray(bad, int)] = True
    a, merged, skip = a[present[0] :].astype(float), merged[present[0] :], skip[present[0] :]
    flags = []
    if a[0] > APPEAR_MAX:
        flags.append(f"large at appearance ({int(a[0])} px)")
    gaps = int(((a == 0) & ~merged & ~skip).sum())
    if gaps:
        flags.append(f"gap ({gaps} frames)")
    splits = int((merged[:-1] & ~merged[1:] & (a[1:] > 0)).sum())
    if splits and not merged_area_is_own:  # seed tracker: the area jumps to the blob's and back
        flags.append(f"transient merge ({int(merged.sum())} frames)")
    own = np.where((merged & (not merged_area_is_own)) | (a == 0), np.nan, a)
    own = own[~skip]  # bad frames dropped: neighbouring good frames are compared directly
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
        steps = (np.abs(ratio - 1) > FLICKER_STEP) & (own[:-1] >= FLICKER_MIN_AREA)
        if steps.sum() >= FLICKER_MIN:
            flags.append(f"flicker ({int(steps.sum())} changes)")
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


def doublet_ratios(area: np.ndarray, merged: np.ndarray, times_min: np.ndarray) -> np.ndarray:
    """Per colony: its area FIXED_AGE_H after appearance over the median of its peers' (colonies that
    appeared within DOUBLET_WINDOW_H) at the same age; NaN if it merged or the run ended before then,
    or it has fewer than DOUBLET_MIN_PEERS peers."""
    t = np.asarray(times_min, float) / 60
    n_frames, n = area.shape
    out = np.full(n, np.nan)
    if not n:
        return out
    appear = np.argmax(area > 0, axis=0)
    at_age = np.full(n, np.nan)
    for c in range(n):
        if not area[appear[c], c]:
            continue
        k = int(np.searchsorted(t, t[appear[c]] + FIXED_AGE_H))
        if k < n_frames and not merged[appear[c] : k + 1, c].any() and area[k, c] > 0:
            at_age[c] = area[k, c]
    t_app = t[appear]
    for c in np.nonzero(np.isfinite(at_age))[0]:
        peers = np.isfinite(at_age) & (np.abs(t_app - t_app[c]) <= DOUBLET_WINDOW_H)
        peers[c] = False
        if peers.sum() >= DOUBLET_MIN_PEERS:
            out[c] = at_age[c] / np.median(at_age[peers])
    return out


def colony_flags(area: np.ndarray, merged: np.ndarray, blob: np.ndarray, labels, own=None, times_min=None,
                 bad=None) -> pd.DataFrame:
    """One row per colony: ``flags`` (text, '' if none) and ``n_flags``.

    ``area``, ``merged``, ``blob`` are [T, N] tracker arrays; ``labels`` is the [T, H, W] label stack.
    ``own`` is the [T, H, W] colony ID stack of the footprint tracker (None for the seed tracker).
    ``times_min`` (frame times) enables the doublet check; ``bad``: frames to ignore (default: found
    with ``bad_frames``).
    """
    n_frames, n = area.shape
    if bad is None:
        bad = bad_frames(area)
    doublet = doublet_ratios(area, merged, times_min) if times_min is not None else np.full(n, np.nan)
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
    split_solidity = {}
    if own is not None and n:
        last_merged = merged[-1] & (area[-1] >= SHAPE_MIN_AREA)
        if last_merged.any():
            props = {r.label: r for r in regionprops(np.asarray(own[n_frames - 1]))}
            for c in np.nonzero(last_merged)[0]:
                if (r := props.get(c + 1)) is not None:
                    split_solidity[c] = r.solidity
    rows = []
    for c in range(n):
        flags = track_flags(area[:, c], merged[:, c], own is not None, bad) + shape_flags(regions.get(c))
        if split_solidity.get(c, 1.0) < SPLIT_SOLIDITY:
            flags.append(f"irregular split (solidity {split_solidity[c]:.2f})")
        if doublet[c] >= DOUBLET_RATIO:
            flags.append(f"possible doublet (x{doublet[c]:.1f})")
        rows.append({"colony_id": c + 1, "flags": "; ".join(flags), "n_flags": len(flags)})
    return pd.DataFrame(rows, columns=["colony_id", "flags", "n_flags"])

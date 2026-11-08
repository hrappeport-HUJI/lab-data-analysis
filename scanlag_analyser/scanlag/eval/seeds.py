"""Score detections against the colonies a user kept in MATLAB, by seed pixel.

The reference is the set of colonies with status VALID (kept after review). Each colony is
identified by its MATLAB seed (the pixel where it first appeared, which it covers in every later
frame). Outlines are not compared.

* A detection is a true positive if it covers the seed of at least one kept colony, including a
  colony MATLAB only accepts at a later frame (counted separately as ``tp_before_matlab``).
* Recall is over kept colonies that exist at the frame. A colony is found if a detection covers
  its seed and that detection is not already credited to another colony (one-to-one, so a blob
  over two colonies finds only one).
* Every other detection is a false positive. It is broken down by what lies under it: the seed
  of a colony MATLAB excluded (near border / manually / zero area), or no MATLAB colony at all.

Detections whose centroid lies outside the working-area mask are dropped first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scanlag.io.matlab_import import ColonyStatus

FP_KINDS = {int(s): f"fp_{s.name.lower()}" for s in ColonyStatus if s != ColonyStatus.VALID}


def drop_outside(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)[1:]
    if not len(ids):
        return labels
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, ids)).round().astype(int).T
    outside = ids[~mask[np.clip(cy, 0, mask.shape[0] - 1), np.clip(cx, 0, mask.shape[1] - 1)]]
    return np.where(np.isin(labels, outside), 0, labels)


def score_frame(pred: np.ndarray, colonies: pd.DataFrame, frame: int, mask: np.ndarray) -> dict:
    """``colonies`` is a plate's colonies.parquet; only colonies that exist at ``frame`` count."""
    pred = drop_outside(pred, mask)
    sx, sy = colonies.seed_x.to_numpy(int), colonies.seed_y.to_numpy(int)
    under = pred[sy, sx]  # detection covering each colony's seed (0 = none)
    status = colonies.status.to_numpy()
    kept = status == ColonyStatus.VALID
    present = colonies.appearance_frame.to_numpy() <= frame

    det_ids = np.unique(pred)[1:]
    # A detection over a kept colony that MATLAB only picks up later is still a real colony.
    det_kept = np.unique(under[kept & (under > 0)])
    det_present = np.unique(under[kept & present & (under > 0)])
    row = {
        "n_kept": int((kept & present).sum()),
        "n_det": len(det_ids),
        "tp": len(det_kept),
        "tp_before_matlab": len(np.setdiff1d(det_kept, det_present)),
        "kept_found": len(det_present),  # one colony per detection
        "kept_seed_covered": int((under[kept & present] > 0).sum()),
    }
    fp = np.setdiff1d(det_ids, det_kept)
    by_status = {}
    for d, s in zip(under, status):
        if d in fp and d not in by_status:
            by_status[d] = int(s)
    for s, name in FP_KINDS.items():
        row[name] = sum(v == s for v in by_status.values())
    row["fp_no_matlab_colony"] = len(fp) - len(by_status)
    row["fp"] = len(fp)
    return row


def summarize(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, np.integer))}
    s["precision"] = s["tp"] / s["n_det"] if s["n_det"] else float("nan")
    s["recall"] = s["kept_found"] / s["n_kept"] if s["n_kept"] else float("nan")
    s["f1"] = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"]) if s["tp"] else 0.0
    return s

"""Instance-level detection metrics: match predicted to true colonies by IoU.

With a threshold of 0.5 or more, each object can match at most one object on the other side, so
the matching is unique without any assignment step. Below 0.5 the Hungarian algorithm picks the
matching with the largest total IoU.

Objects that lie mostly in the ignore region (e.g. outside the plate's working area) are removed
from both sides before matching, so they count neither as hits nor as errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

SMALL_AREA = 30  # px; "small" colonies in the gold-set metrics (PLAN §5.5)


@dataclass
class Matching:
    true_ids: np.ndarray
    true_areas: np.ndarray
    pred_ids: np.ndarray
    pred_areas: np.ndarray
    pairs: np.ndarray  # [k, 2] indices into (true_ids, pred_ids)
    pair_iou: np.ndarray  # [k]

    @property
    def true_matched(self) -> np.ndarray:
        hit = np.zeros(len(self.true_ids), bool)
        hit[self.pairs[:, 0]] = True
        return hit

    @property
    def pred_matched(self) -> np.ndarray:
        hit = np.zeros(len(self.pred_ids), bool)
        hit[self.pairs[:, 1]] = True
        return hit


def drop_ignored(labels: np.ndarray, ignore: np.ndarray | None, max_fraction: float = 0.5) -> np.ndarray:
    """Remove objects with more than ``max_fraction`` of their pixels in ``ignore``."""
    if ignore is None or not ignore.any():
        return labels
    ids, areas = np.unique(labels, return_counts=True)
    ignored_ids, ignored_px = np.unique(labels[ignore], return_counts=True)
    frac = dict(zip(ignored_ids, ignored_px / areas[np.searchsorted(ids, ignored_ids)]))
    drop = [i for i, f in frac.items() if i and f > max_fraction]
    return np.where(np.isin(labels, drop), 0, labels)


def match(true: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5, ignore: np.ndarray | None = None) -> Matching:
    true, pred = drop_ignored(true, ignore), drop_ignored(pred, ignore)
    t_ids, t_inv = np.unique(true, return_inverse=True)
    p_ids, p_inv = np.unique(pred, return_inverse=True)
    # contingency table of pixel counts, background at index 0 if present
    table = np.zeros((len(t_ids), len(p_ids)), np.int64)
    np.add.at(table, (t_inv.ravel(), p_inv.ravel()), 1)
    t_fg, p_fg = t_ids != 0, p_ids != 0
    t_areas, p_areas = table.sum(1)[t_fg], table.sum(0)[p_fg]
    inter = table[np.ix_(t_fg, p_fg)]
    iou = inter / (t_areas[:, None] + p_areas[None, :] - inter) if inter.size else inter.astype(float)

    if iou_threshold >= 0.5:
        rows, cols = np.nonzero(iou > iou_threshold)
    else:
        rows, cols = linear_sum_assignment(-iou) if iou.size else (np.array([], int), np.array([], int))
        keep = iou[rows, cols] > iou_threshold
        rows, cols = rows[keep], cols[keep]
    return Matching(t_ids[t_fg], t_areas, p_ids[p_fg], p_areas, np.stack([rows, cols], 1).reshape(-1, 2), iou[rows, cols])


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def counts(m: Matching, small_area: int = SMALL_AREA) -> dict:
    """TP/FP/FN overall and for small colonies.

    Small recall uses true colonies below ``small_area``; small precision uses predictions below it.
    A small prediction matched to a large colony (or the reverse) is a hit on its own side.
    """
    t_small, p_small = m.true_areas < small_area, m.pred_areas < small_area
    t_hit, p_hit = m.true_matched, m.pred_matched
    return {
        "n_true": len(m.true_ids),
        "n_pred": len(m.pred_ids),
        "tp": len(m.pairs),
        "fp": int((~p_hit).sum()),
        "fn": int((~t_hit).sum()),
        "iou_sum": float(m.pair_iou.sum()),
        "small_true": int(t_small.sum()),
        "small_true_hit": int((t_small & t_hit).sum()),
        "small_pred": int(p_small.sum()),
        "small_pred_hit": int((p_small & p_hit).sum()),
    }


def summarize(rows: list[dict]) -> dict:
    """Pool per-tile counts (micro-average) into F1, precision, recall and small-colony F1."""
    s = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    prec = s["tp"] / s["n_pred"] if s["n_pred"] else 1.0
    rec = s["tp"] / s["n_true"] if s["n_true"] else 1.0
    sp = s["small_pred_hit"] / s["small_pred"] if s["small_pred"] else 1.0
    sr = s["small_true_hit"] / s["small_true"] if s["small_true"] else 1.0
    return {
        "f1": f1(s["tp"], s["fp"], s["fn"]),
        "precision": prec,
        "recall": rec,
        "mean_iou_matched": s["iou_sum"] / s["tp"] if s["tp"] else float("nan"),
        "small_f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0,
        "small_precision": sp,
        "small_recall": sr,
        **{k: s[k] for k in ("n_true", "n_pred", "tp", "fp", "fn", "small_true", "small_pred")},
    }

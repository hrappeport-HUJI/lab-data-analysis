"""Vectorized, physically grounded founder-event tracker.

Tracks individual bacterial founders across phase-contrast (and optional
propidium iodide / YFP fluorescence) time-lapse stacks without per-frame
colony-wide object graphs. Key mechanisms:

1. **Stage-drift estimation** via 2x-binned central ``phase_cross_correlation``,
   gated by correlation error and maximum jump to reject corrupted frames.
2. **Phase-darkness label filter** removing Omnipose background-ripple
   hallucinations (labels whose median phase / frame background median exceeds
   ``dark_ratio_max``, default ``0.85``, i.e. refractility ``r > -0.15``).
3. **Focus-QC pause** based on contrast-normalized Laplacian variance so
   transiently defocused frames pause tracking without burning ``max_gap``.
4. **Dynamic-bbox Voronoi-constrained tracking** (``slide=4``, ``through=True``,
   ``max_gap=5``, ``growth=1.4``, ``frac=0.2``, ``persist=2``) with
   single-frame transient contact merge splitting via prior-footprint EDT.
5. **Physically grounded fate calling**:
   - ``division``: confirmed ``>= 2`` daughters for ``persist`` frames after
     reaching ``growth * area0``.
   - ``pi_positive_death``: in-track PI surge above background noise floor
     sustained for ``pi_streak`` frames (safe against healthy cells with
     decaying elevated frame-0 PI), plus persistent high-level baseline dead
     cells that never decay or grow.
   - ``ghost_lysis``: abrupt phase-neutral/bright transition (``delta_r >= 0.25``
     and ``r_post >= -0.08``) on loss where no dark/segmented neighbouring cell
     occupies or lies within ``ghost_clear_radius_px`` (~10 px) of the shifted
     last footprint (preventing neighbour-shove false lysis calls).
   - ``no_division``: intact single-cell lineage followed to the end of the
     recording, annotated with ``growth_ratio`` and ``elongation_ratio``.
   - ``lost``: ``contact``, ``edge``, ``missing``, or ``defocus_truncated``.
6. **Frame-0 doublet & debris scoring**: ``notch_ratio`` along major axis,
   ``solidity0``, ``touching_neighbour``, ``length_z``, ``area_z``, and optional
   ``baseline_yfp_net``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
from skimage.registration import phase_cross_correlation


@dataclass
class TrackerConfig:
    """Configuration for :class:`FieldTracker` and helper utilities."""

    # Voronoi-constrained lineage tracking
    slide: int = 4
    growth: float = 1.4
    frac: float = 0.2
    persist: int = 2
    max_gap: int = 5
    through: bool = True
    b_contact: float = 0.25
    split_transient_contact: bool = True
    split_ignore_invader_area_frac: float = 0.85
    pad: int = 48
    max_lost_dist_px: int = 6

    # Phase-darkness ripple filter (cell_median / bg_median <= 0.85 <=> r <= -0.15)
    dark_ratio_max: float = 0.85

    # Stage-drift estimation (2x-binned central phase_cross_correlation)
    drift_bin: int = 2
    drift_upsample: int = 4
    drift_max_err: float = 0.08
    drift_max_jump: float = 30.0

    # Focus quality control
    focus_drop_ratio: float = 0.60
    focus_baseline_frames: int = 3

    # In-track PI-positive death calling
    pi_abs_min: float = 100.0
    pi_noise_mult: float = 6.0
    pi_surge_min: float = 100.0
    pi_streak: int = 2
    pi_high_baseline_min: float = 1000.0
    pi_high_baseline_streak: int = 5
    pi_high_baseline_decay: float = 0.85
    pi_high_baseline_growth: float = 1.2

    # Anti-shove phase-ghost lysis calling
    ghost_delta_r: float = 0.25
    ghost_r_post_min: float = -0.08
    ghost_clear_radius_px: int = 10
    ring_inner_px: int = 3
    ring_outer_px: int = 10
    min_ring_pixels: int = 20

    # Frame-0 doublet & debris scoring
    doublet_notch_max: float = 0.78
    doublet_solidity_max: float = 0.89
    doublet_len_z_min: float = 2.0
    doublet_area_z_min: float = 2.0
    debris_area_min: int = 40
    debris_area_z_min: float = -3.0


@dataclass
class FounderResult:
    """Outcome summary for one tracked founder."""

    founder_id: str
    decision: str
    event_frame: int | None
    lower_frame: int
    upper_frame: int
    last_identity_frame: int
    tracking_end_reason: str
    label0: int = 0
    area0: int = 0
    max_area: int = 0
    last_area: int = 0
    growth_ratio: float = 1.0
    elongation_ratio: float = 1.0
    pi_0: float | None = None
    pi_max: float | None = None
    pi_last: float | None = None
    pi_surge_max: float | None = None
    r_0: float | None = None
    r_last: float | None = None
    r_post: float | None = None
    delta_r: float | None = None
    contains_gap: bool = False
    contains_defocus_pause: bool = False
    used_transient_contact_split: bool = False
    flags: list[str] = field(default_factory=list)
    frame0_scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "founder_id": self.founder_id,
            "decision": self.decision,
            "event_frame": self.event_frame,
            "lower_frame": self.lower_frame,
            "upper_frame": self.upper_frame,
            "last_identity_frame": self.last_identity_frame,
            "tracking_end_reason": self.tracking_end_reason,
            "label0": self.label0,
            "area0": self.area0,
            "max_area": self.max_area,
            "last_area": self.last_area,
            "growth_ratio": self.growth_ratio,
            "elongation_ratio": self.elongation_ratio,
            "pi_0": self.pi_0,
            "pi_max": self.pi_max,
            "pi_last": self.pi_last,
            "pi_surge_max": self.pi_surge_max,
            "r_0": self.r_0,
            "r_last": self.r_last,
            "r_post": self.r_post,
            "delta_r": self.delta_r,
            "contains_gap": self.contains_gap,
            "contains_defocus_pause": self.contains_defocus_pause,
            "used_transient_contact_split": self.used_transient_contact_split,
            "flags": list(self.flags),
            **{f"f0_{k}": v for k, v in self.frame0_scores.items() if k != "flags"},
        }


def _central_binned(image: np.ndarray, bin_factor: int = 2) -> tuple[np.ndarray, int]:
    """Return central crop (optionally 2x-binned) for fast drift/focus checks."""
    h, w = image.shape[:2]
    if min(h, w) >= 1024:
        sub = image[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4].astype(np.float32)
        if bin_factor > 1:
            bh = (sub.shape[0] // bin_factor) * bin_factor
            bw = (sub.shape[1] // bin_factor) * bin_factor
            sub = sub[:bh, :bw].reshape(bh // bin_factor, bin_factor, bw // bin_factor, bin_factor).mean((1, 3))
            return sub, bin_factor
        return sub, 1
    if min(h, w) >= 64:
        sub = image[h // 8 : (7 * h) // 8, w // 8 : (7 * w) // 8].astype(np.float32)
        return sub, 1
    return image.astype(np.float32), 1


def estimate_frame_shift(
    prev_phase: np.ndarray,
    curr_phase: np.ndarray,
    cfg: TrackerConfig | None = None,
) -> tuple[float, float, float, bool]:
    """Estimate stage shift ``(dy, dx)`` from ``prev_phase`` to ``curr_phase``.

    Uses 2x-binned central ``phase_cross_correlation`` and gates on both
    ``err <= cfg.drift_max_err`` and ``hypot(dy, dx) <= cfg.drift_max_jump``.
    Note the sign convention: if an object moves by ``(oy, ox)`` in image
    coordinates from ``prev_phase`` to ``curr_phase``, ``dy = -oy, dx = -ox``,
    so a footprint at ``(y, x)`` in ``prev`` is at ``(y - dy, x - dx)`` in ``curr``.

    Returns:
        ``(dy, dx, err, accepted)``. When ``accepted`` is ``False``,
        ``dy`` and ``dx`` are reset to ``0.0``.
    """
    cfg = cfg or TrackerConfig()
    a, scale = _central_binned(prev_phase, cfg.drift_bin)
    b, _ = _central_binned(curr_phase, cfg.drift_bin)
    if a.shape != b.shape or a.size == 0 or float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return 0.0, 0.0, 1.0, False
    shift, err, _ = phase_cross_correlation(a, b, upsample_factor=cfg.drift_upsample, normalization=None)
    dy = float(scale * shift[0])
    dx = float(scale * shift[1])
    err_f = float(err) if np.isfinite(err) else 1.0
    if err_f <= cfg.drift_max_err and float(np.hypot(dy, dx)) <= cfg.drift_max_jump:
        return dy, dx, err_f, True
    return 0.0, 0.0, err_f, False


def estimate_field_shifts(
    phase_frames: Sequence[np.ndarray],
    cfg: TrackerConfig | None = None,
) -> dict[int, tuple[float, float]]:
    """Estimate per-frame stage shifts ``{frame_idx: (dy, dx)}`` for a stack."""
    cfg = cfg or TrackerConfig()
    shifts: dict[int, tuple[float, float]] = {}
    if len(phase_frames) < 2:
        return shifts
    prev_bin, scale = _central_binned(phase_frames[0], cfg.drift_bin)
    for f in range(1, len(phase_frames)):
        curr_bin, _ = _central_binned(phase_frames[f], cfg.drift_bin)
        if curr_bin.shape != prev_bin.shape or float(np.std(prev_bin)) < 1e-6 or float(np.std(curr_bin)) < 1e-6:
            shifts[f] = (0.0, 0.0)
        else:
            shift, err, _ = phase_cross_correlation(
                prev_bin, curr_bin, upsample_factor=cfg.drift_upsample, normalization=None
            )
            dy = float(scale * shift[0])
            dx = float(scale * shift[1])
            err_f = float(err) if np.isfinite(err) else 1.0
            shifts[f] = (
                (dy, dx)
                if (err_f <= cfg.drift_max_err and float(np.hypot(dy, dx)) <= cfg.drift_max_jump)
                else (0.0, 0.0)
            )
        prev_bin = curr_bin
    return shifts


def compute_focus_score(phase_image: np.ndarray) -> float:
    """Compute robust background-normalized Laplacian median sharpness."""
    h, w = phase_image.shape[:2]
    if h >= 1024 and w >= 1024:
        sub = phase_image[h // 4 : (3 * h) // 4 : 2, w // 4 : (3 * w) // 4 : 2].astype(np.float32)
    else:
        sub = phase_image.astype(np.float32)
    bg_med = float(np.median(sub))
    if bg_med <= 1e-6:
        lo, hi = np.percentile(sub, [1.0, 99.0])
        bg_med = max(float(hi - lo), 1.0)
    return float(np.median(np.abs(ndi.laplace(sub / bg_med))))


def fast_bg_median(image: np.ndarray, step: int = 4) -> float:
    """Fast background median via regular strided subsampling."""
    h, w = image.shape[:2]
    if min(h, w) >= step * 16:
        return float(np.median(image[::step, ::step]))
    return float(np.median(image))


def filter_phase_dark_labels(
    labels: np.ndarray,
    phase_image: np.ndarray,
    bg_median: float | None = None,
    dark_ratio_max: float = 0.85,
) -> np.ndarray:
    """Zero out non-phase-dark Omnipose labels (e.g. background ripple hallucinations).

    Uses vectorized ``np.bincount`` mean screening followed by exact per-object
    median refinement via ``ndimage.find_objects`` for borderline labels within
    ``+-0.08`` of ``dark_ratio_max``.

    Args:
        labels: 2D integer label image (``0`` = background).
        phase_image: Co-registered 2D phase-contrast intensity image.
        bg_median: Optional precomputed frame background median.
        dark_ratio_max: Maximum allowed ``cell_median / bg_median`` ratio
            (default ``0.85``, equivalent to refractility ``r <= -0.15``).

    Returns:
        Filtered integer label array of the same shape/dtype as ``labels``.
    """
    max_lbl = int(labels.max())
    if max_lbl == 0:
        return labels.copy()
    if bg_median is None:
        bg_median = fast_bg_median(phase_image)
    if bg_median <= 0.0:
        return labels.copy()

    flat_lbl = labels.ravel()
    flat_ph = phase_image.ravel()
    cnts = np.bincount(flat_lbl, minlength=max_lbl + 1)
    sums = np.bincount(flat_lbl, weights=flat_ph, minlength=max_lbl + 1)
    mean_ratios = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0) / bg_median

    lo_band = max(0.0, dark_ratio_max - 0.08)
    hi_band = dark_ratio_max + 0.08

    keep = np.zeros(max_lbl + 1, dtype=bool)
    keep[1:] = (cnts[1:] > 0) & (mean_ratios[1:] <= lo_band)

    borderline = np.flatnonzero((cnts > 0) & (mean_ratios > lo_band) & (mean_ratios <= hi_band))
    borderline = borderline[borderline > 0]
    if borderline.size > 0:
        boxes = ndi.find_objects(labels)
        for lid in borderline:
            box = boxes[int(lid) - 1]
            if box is None:
                continue
            med_ratio = float(np.median(phase_image[box][labels[box] == int(lid)])) / bg_median
            keep[int(lid)] = med_ratio <= dark_ratio_max

    out = labels.copy()
    out[~keep[out]] = 0
    return out


def _major_axis_length(mask: np.ndarray) -> float:
    """Fast major-axis length (4 * sqrt(max PCA eigenvalue)) matching regionprops."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n <= 1:
        return float(n)
    yc = ys.astype(np.float64) - float(ys.mean())
    xc = xs.astype(np.float64) - float(xs.mean())
    yy = float(np.dot(yc, yc)) / n
    xx = float(np.dot(xc, xc)) / n
    yx = float(np.dot(yc, xc)) / n
    common = float(np.sqrt((yy - xx) ** 2 + 4.0 * yx * yx))
    lam_max = 0.5 * (yy + xx + common)
    return float(4.0 * np.sqrt(max(lam_max, 0.0)))


def _solidity(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0:
        return 1.0
    if area <= 4:
        return 1.0
    return float(area / max(1, int(convex_hull_image(mask).sum())))


def compute_notch_ratio(mask: np.ndarray) -> float:
    """Measure mid-body width constriction along a cell mask's major axis.

    Returns ``min_middle_width / min(left_peak_width, right_peak_width)`` in
    ``(0, 1]``. A value well below ``1.0`` (e.g. ``< 0.78``) indicates a
    deep septum constriction or merged doublet.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 12:
        return 1.0
    pts = np.column_stack([ys, xs]).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    minor = eigvecs[:, 0]
    proj_maj = pts @ major
    proj_min = pts @ minor
    t_min, t_max = float(proj_maj.min()), float(proj_maj.max())
    length = t_max - t_min
    if length < 8.0:
        return 1.0
    nbins = max(9, int(round(length)))
    edges = np.linspace(t_min, t_max, nbins + 1)
    widths = np.zeros(nbins, dtype=np.float64)
    for b in range(nbins):
        hi_cond = proj_maj <= edges[b + 1] if b == nbins - 1 else proj_maj < edges[b + 1]
        in_b = (proj_maj >= edges[b]) & hi_cond
        cnt = int(in_b.sum())
        if cnt >= 2:
            widths[b] = float(proj_min[in_b].max() - proj_min[in_b].min() + 1.0)
        elif cnt == 1:
            widths[b] = 1.0
    left_slice = widths[max(1, int(0.15 * nbins)) : max(2, int(0.40 * nbins))]
    right_slice = widths[min(nbins - 2, int(0.60 * nbins)) : min(nbins - 1, int(0.85 * nbins))]
    mid_slice = widths[int(0.35 * nbins) : max(int(0.35 * nbins) + 1, int(0.65 * nbins))]
    if not left_slice.size or not right_slice.size or not mid_slice.size:
        return 1.0
    denom = min(float(left_slice.max()), float(right_slice.max()))
    if denom <= 1e-6:
        return 1.0
    return float(min(1.0, float(mid_slice.min()) / denom))


def compute_frame0_population_stats(labels0: np.ndarray) -> dict[str, Any]:
    """Compute field-wide robust area and length statistics at frame 0."""
    boxes = ndi.find_objects(labels0)
    props = regionprops(labels0)
    if not props:
        return {
            "boxes": boxes,
            "props_by_label": {},
            "med_area": 0.0,
            "mad_area": 1.0,
            "med_len": 0.0,
            "mad_len": 1.0,
        }
    areas = np.array([p.area for p in props], dtype=np.float64)
    lens = np.array([p.axis_major_length for p in props], dtype=np.float64)
    med_a = float(np.median(areas))
    mad_a = float(1.4826 * np.median(np.abs(areas - med_a)))
    med_l = float(np.median(lens))
    mad_l = float(1.4826 * np.median(np.abs(lens - med_l)))
    return {
        "boxes": boxes,
        "props_by_label": {int(p.label): p for p in props},
        "med_area": med_a,
        "mad_area": max(mad_a, 1.0),
        "med_len": med_l,
        "mad_len": max(mad_l, 1.0),
    }


def measure_footprint_channels(
    mask: np.ndarray,
    phase_crop: np.ndarray,
    pi_crop: np.ndarray | None = None,
    dark_labels_crop: np.ndarray | None = None,
    bg_phase: float | None = None,
    bg_pi_fallback: float | None = None,
    ring_inner_px: int = 3,
    ring_outer_px: int = 10,
    min_ring_pixels: int = 20,
) -> tuple[float, float | None, float | None, float | None]:
    """Measure phase refractility and net PI intensity on a footprint mask.

    Background PI is estimated strictly on a local annulus
    ``(dilate(mask, ring_outer) & ~dilate(mask, ring_inner)) & (dark_labels == 0)``
    so bright neighbouring cells never contaminate the background reference.

    Returns:
        ``(r_median, pi_net, pi_bg_mad, pi_bg_se)``. PI fields are ``None`` if
        ``pi_crop`` is omitted or ``mask`` is empty.
    """
    n_px = int(mask.sum())
    if n_px == 0:
        return 0.0, None, None, None
    bg_p = float(bg_phase) if bg_phase and bg_phase > 0 else float(np.median(phase_crop))
    r_med = (float(np.median(phase_crop[mask])) - bg_p) / max(bg_p, 1e-6)

    if pi_crop is None:
        return r_med, None, None, None

    inner = ndi.binary_dilation(mask, iterations=ring_inner_px)
    outer = ndi.binary_dilation(mask, iterations=ring_outer_px)
    ring = outer & ~inner
    if dark_labels_crop is not None:
        ring &= dark_labels_crop == 0
    ring_vals = pi_crop[ring]
    if ring_vals.size < min_ring_pixels:
        if dark_labels_crop is not None:
            clean_patch = pi_crop[dark_labels_crop == 0]
            if clean_patch.size >= min_ring_pixels:
                ring_vals = clean_patch
        if ring_vals.size < min_ring_pixels and bg_pi_fallback is not None:
            bg_pi = float(bg_pi_fallback)
            bg_mad = 0.0
            pi_net = float(np.mean(pi_crop[mask])) - bg_pi
            return r_med, pi_net, bg_mad, 0.0
        if ring_vals.size == 0:
            ring_vals = pi_crop.ravel()

    bg_pi = float(np.median(ring_vals))
    bg_mad = float(1.4826 * np.median(np.abs(ring_vals.astype(np.float64) - bg_pi)))
    n_eff = max(1, min(n_px, int(ring_vals.size)))
    bg_se = float(bg_mad / np.sqrt(n_eff))
    pi_net = float(np.mean(pi_crop[mask])) - bg_pi
    return r_med, pi_net, bg_mad, bg_se


def score_frame0_founder(
    labels0: np.ndarray,
    phase0: np.ndarray,
    label_id: int,
    pop_stats: Mapping[str, Any] | None = None,
    yfp0: np.ndarray | None = None,
    bg_phase0: float | None = None,
    cfg: TrackerConfig | None = None,
) -> dict[str, Any]:
    """Compute frame-0 morphology, doublet, debris, and baseline YFP metrics."""
    cfg = cfg or TrackerConfig()
    if pop_stats is None:
        pop_stats = compute_frame0_population_stats(labels0)
    boxes = pop_stats["boxes"]
    prop = pop_stats["props_by_label"].get(int(label_id))
    box = boxes[int(label_id) - 1] if 0 < int(label_id) <= len(boxes) else None
    if prop is None or box is None:
        return {
            "label_id": int(label_id),
            "area0": 0,
            "length_px": 0.0,
            "solidity0": 1.0,
            "notch_ratio": 1.0,
            "touching_neighbour": False,
            "touching_count": 0,
            "length_z": 0.0,
            "area_z": 0.0,
            "refractility0": 0.0,
            "baseline_yfp_net": None,
            "doublet_score": 0.0,
            "is_doublet": False,
            "is_debris": True,
            "flags": ["debris"],
        }

    h, w = labels0.shape
    mask_box = labels0[box] == int(label_id)
    area0 = int(prop.area)
    length_px = float(prop.axis_major_length)
    sol0 = float(prop.solidity)
    notch = compute_notch_ratio(mask_box)

    pad = 12
    y0, y1 = max(0, box[0].start - pad), min(h, box[0].stop + pad)
    x0, x1 = max(0, box[1].start - pad), min(w, box[1].stop + pad)
    sub_lbl = labels0[y0:y1, x0:x1]
    sub_mask = sub_lbl == int(label_id)
    dil2 = ndi.binary_dilation(sub_mask, iterations=2)
    touch_ids = np.unique(sub_lbl[dil2 & (sub_lbl > 0) & (sub_lbl != int(label_id))])
    touch_count = int(touch_ids.size)
    touching = touch_count > 0

    area_z = (area0 - float(pop_stats["med_area"])) / float(pop_stats["mad_area"])
    length_z = (length_px - float(pop_stats["med_len"])) / float(pop_stats["mad_len"])

    if bg_phase0 is None:
        bg_phase0 = fast_bg_median(phase0)
    r0, yfp_net, _, _ = measure_footprint_channels(
        sub_mask,
        phase0[y0:y1, x0:x1],
        pi_crop=yfp0[y0:y1, x0:x1] if yfp0 is not None else None,
        dark_labels_crop=sub_lbl,
        bg_phase=bg_phase0,
        ring_inner_px=cfg.ring_inner_px,
        ring_outer_px=cfg.ring_outer_px,
        min_ring_pixels=cfg.min_ring_pixels,
    )

    notch_bad = notch < cfg.doublet_notch_max
    sol_bad = sol0 < cfg.doublet_solidity_max
    size_bad = (length_z >= cfg.doublet_len_z_min) and (area_z >= cfg.doublet_area_z_min)

    doublet_score = 0.0
    if notch_bad:
        doublet_score += 0.45
    if sol_bad:
        doublet_score += 0.35
    if touching:
        doublet_score += 0.35
    if size_bad:
        doublet_score += 0.25
    doublet_score = min(1.0, doublet_score)

    is_doublet = bool(touching or (notch_bad and sol_bad) or ((notch_bad or sol_bad) and size_bad))
    is_debris = bool(
        (area0 < cfg.debris_area_min)
        or (area_z < cfg.debris_area_z_min)
        or (r0 > cfg.dark_ratio_max - 1.0)
    )

    flags: list[str] = []
    if touching:
        flags.append("touching_neighbour")
    if is_doublet:
        flags.append("possible_doublet")
    if size_bad and not is_doublet:
        flags.append("elongated_f0")
    if is_debris:
        flags.append("possible_debris")

    return {
        "label_id": int(label_id),
        "area0": area0,
        "length_px": round(length_px, 2),
        "solidity0": round(sol0, 4),
        "notch_ratio": round(notch, 4),
        "touching_neighbour": touching,
        "touching_count": touch_count,
        "length_z": round(float(length_z), 3),
        "area_z": round(float(area_z), 3),
        "refractility0": round(float(r0), 4),
        "baseline_yfp_net": round(float(yfp_net), 2) if yfp_net is not None else None,
        "doublet_score": round(float(doublet_score), 3),
        "is_doublet": is_doublet,
        "is_debris": is_debris,
        "flags": flags,
    }


def _first_division(counts: list[tuple[int, int]], persist: int) -> int | None:
    if len(counts) < persist:
        return None
    for i in range(len(counts) - persist + 1):
        if all(n >= 2 for _, n in counts[i : i + persist]):
            return counts[i][0]
    return None


@dataclass
class _FounderState:
    founder_id: str
    active: bool
    label0: int
    area0: int
    length0: float
    by0: int
    by1: int
    bx0: int
    bx1: int
    fam_crop: np.ndarray
    prev_f: int
    last_seen: int
    max_area: int
    last_area: int
    max_length: float
    last_length: float
    counts: list[tuple[int, int]]
    missing: int = 0
    first_missing: int | None = None
    contact_streak: int = 0
    used_transient_contact_split: bool = False
    contains_gap: bool = False
    contains_defocus_pause: bool = False
    started_at_edge: bool = False
    # Channel history
    r_0: float | None = None
    r_last: float | None = None
    r_post: float | None = None
    delta_r: float | None = None
    pi_0: float | None = None
    pi_running_min: float | None = None
    pi_max: float | None = None
    pi_last: float | None = None
    pi_surge_max: float = 0.0
    pi_surge_streak: int = 0
    pi_surge_first_frame: int | None = None
    pi_surge_prev_frame: int | None = None
    pi_high_streak: int = 0
    pi_high_first_frame: int | None = None
    pi_high_prev_frame: int | None = None
    # Terminal call
    result: FounderResult | None = None
    frame0_scores: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


class FieldTracker:
    """Stateful online/offline founder tracker for one microscopy field.

    Can be driven either frame-by-frame online (e.g. by ``fast_omnipose.py``)
    or in batch via :func:`track_field`.
    """

    def __init__(
        self,
        labels0: np.ndarray,
        founders: Mapping[str, tuple[float, float]],
        phase0: np.ndarray | None = None,
        pi0: np.ndarray | None = None,
        yfp0: np.ndarray | None = None,
        dark_filter_applied: bool = False,
        cfg: TrackerConfig | None = None,
    ) -> None:
        self.cfg = cfg or TrackerConfig()
        self.H, self.W = labels0.shape[:2]
        self.cum_dy = 0.0
        self.cum_dx = 0.0
        self.cum_shifts: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
        self.last_frame = 0
        self.last_usable_frame = 0
        self.defocused_after_last_use = False

        bg_p0 = fast_bg_median(phase0) if phase0 is not None else None
        if phase0 is not None and not dark_filter_applied:
            l0 = filter_phase_dark_labels(labels0, phase0, bg_median=bg_p0, dark_ratio_max=self.cfg.dark_ratio_max)
        else:
            l0 = labels0

        self.prev_usable_labels = l0
        self.prev_phase_bin = _central_binned(phase0, self.cfg.drift_bin)[0] if phase0 is not None else None
        self.focus_history: list[float] = []
        self.baseline_focus: float | None = None
        if phase0 is not None:
            f0_score = compute_focus_score(phase0)
            self.focus_history.append(f0_score)
            self.baseline_focus = f0_score

        pop_stats = compute_frame0_population_stats(l0)
        bg_pi0 = fast_bg_median(pi0) if pi0 is not None else None
        self.states: dict[str, _FounderState] = {}

        for fid, (fy, fx) in founders.items():
            cy = int(round(float(fy)))
            cx = int(round(float(fx)))
            cy_clamped = min(max(cy, 0), self.H - 1)
            cx_clamped = min(max(cx, 0), self.W - 1)
            id0 = int(l0[cy_clamped, cx_clamped])
            if id0 == 0:
                rad = self.cfg.max_lost_dist_px + 4
                y_lo, y_hi = max(0, cy_clamped - rad), min(self.H, cy_clamped + rad + 1)
                x_lo, x_hi = max(0, cx_clamped - rad), min(self.W, cx_clamped + rad + 1)
                ys, xs = np.nonzero(l0[y_lo:y_hi, x_lo:x_hi])
                if len(ys):
                    ys_g = ys + y_lo
                    xs_g = xs + x_lo
                    k = int(np.argmin((ys_g - cy_clamped) ** 2 + (xs_g - cx_clamped) ** 2))
                    if float(np.hypot(ys_g[k] - cy_clamped, xs_g[k] - cx_clamped)) <= self.cfg.max_lost_dist_px:
                        id0 = int(l0[ys_g[k], xs_g[k]])

            if id0 == 0:
                res = FounderResult(
                    founder_id=fid,
                    decision="not_segmented",
                    event_frame=None,
                    lower_frame=0,
                    upper_frame=0,
                    last_identity_frame=-1,
                    tracking_end_reason="not_segmented",
                )
                self.states[fid] = _FounderState(
                    founder_id=fid,
                    active=False,
                    label0=0,
                    area0=0,
                    length0=0.0,
                    by0=0,
                    by1=0,
                    bx0=0,
                    bx1=0,
                    fam_crop=np.zeros((0, 0), dtype=bool),
                    prev_f=0,
                    last_seen=-1,
                    max_area=0,
                    last_area=0,
                    max_length=0.0,
                    last_length=0.0,
                    counts=[],
                    result=res,
                )
                continue

            box = pop_stats["boxes"][id0 - 1]
            by0, by1 = int(box[0].start), int(box[0].stop)
            bx0, bx1 = int(box[1].start), int(box[1].stop)
            fam_crop = l0[by0:by1, bx0:bx1] == id0
            area0 = int(fam_crop.sum())
            length0 = _major_axis_length(fam_crop)

            scores = (
                score_frame0_founder(
                    l0,
                    phase0,
                    id0,
                    pop_stats=pop_stats,
                    yfp0=yfp0,
                    bg_phase0=bg_p0,
                    cfg=self.cfg,
                )
                if phase0 is not None
                else {"label_id": id0, "area0": area0, "length_px": round(length0, 2), "flags": []}
            )

            r0 = scores.get("refractility0") if phase0 is not None else None
            pi0_net = None
            if pi0 is not None:
                pad_pi = self.cfg.ring_outer_px + 2
                py0, py1 = max(0, by0 - pad_pi), min(self.H, by1 + pad_pi)
                px0, px1 = max(0, bx0 - pad_pi), min(self.W, bx1 + pad_pi)
                sub_m = l0[py0:py1, px0:px1] == id0
                r0_val, pi0_net, _, _ = measure_footprint_channels(
                    sub_m,
                    phase0[py0:py1, px0:px1] if phase0 is not None else np.ones_like(sub_m, dtype=np.float32),
                    pi_crop=pi0[py0:py1, px0:px1],
                    dark_labels_crop=l0[py0:py1, px0:px1],
                    bg_phase=bg_p0,
                    bg_pi_fallback=bg_pi0,
                    ring_inner_px=self.cfg.ring_inner_px,
                    ring_outer_px=self.cfg.ring_outer_px,
                    min_ring_pixels=self.cfg.min_ring_pixels,
                )
                if r0 is None and phase0 is not None:
                    r0 = r0_val

            st = _FounderState(
                founder_id=fid,
                active=True,
                label0=id0,
                area0=area0,
                length0=max(length0, 1.0),
                by0=by0,
                by1=by1,
                bx0=bx0,
                bx1=bx1,
                fam_crop=fam_crop,
                prev_f=0,
                last_seen=0,
                max_area=area0,
                last_area=area0,
                max_length=max(length0, 1.0),
                last_length=max(length0, 1.0),
                counts=[(0, 1)],
                started_at_edge=(by0 == 0 or bx0 == 0 or by1 == self.H or bx1 == self.W),
                r_0=r0,
                r_last=r0,
                pi_0=pi0_net,
                pi_running_min=pi0_net,
                pi_max=pi0_net,
                pi_last=pi0_net,
                frame0_scores=scores,
                flags=list(scores.get("flags", [])),
            )
            if pi0_net is not None and pi0_net >= self.cfg.pi_high_baseline_min:
                st.pi_high_streak = 1
                st.pi_high_first_frame = 0
                st.pi_high_prev_frame = 0
            self.states[fid] = st

    def active_ids(self) -> list[str]:
        """Return IDs of still-active founders."""
        return [fid for fid, st in self.states.items() if st.active]

    def active_bboxes(self, next_dy: float = 0.0, next_dx: float = 0.0, pad: int | None = None) -> dict[str, tuple[int, int, int, int]]:
        """Return predicted ``(y0, y1, x0, x1)`` ROIs on the next frame for active founders."""
        pad = self.cfg.pad if pad is None else int(pad)
        out: dict[str, tuple[int, int, int, int]] = {}
        for fid, st in self.states.items():
            if not st.active:
                continue
            prev_cdy, prev_cdx = self.cum_shifts[st.prev_f]
            iy = int(round((self.cum_dy + next_dy) - prev_cdy))
            ix = int(round((self.cum_dx + next_dx) - prev_cdx))
            sby0, sby1 = st.by0 - iy, st.by1 - iy
            sbx0, sbx1 = st.bx0 - ix, st.bx1 - ix
            y0 = max(0, sby0 - pad)
            y1 = min(self.H, sby1 + pad)
            x0 = max(0, sbx0 - pad)
            x1 = min(self.W, sbx1 + pad)
            out[fid] = (y0, y1, x0, x1)
        return out

    def check_focus_and_shift(
        self,
        frame_idx: int,
        phase_image: np.ndarray | None,
        shift_override: tuple[float, float] | None = None,
    ) -> tuple[bool, float, float]:
        """Evaluate focus QC and stage shift on ``phase_image`` before segmentation."""
        dy, dx = 0.0, 0.0
        if shift_override is not None:
            dy, dx = float(shift_override[0]), float(shift_override[1])
        elif phase_image is not None and self.prev_phase_bin is not None:
            curr_bin, scale = _central_binned(phase_image, self.cfg.drift_bin)
            if curr_bin.shape == self.prev_phase_bin.shape and float(np.std(self.prev_phase_bin)) >= 1e-6 and float(np.std(curr_bin)) >= 1e-6:
                s, err, _ = phase_cross_correlation(
                    self.prev_phase_bin, curr_bin, upsample_factor=self.cfg.drift_upsample, normalization=None
                )
                cand_dy = float(scale * s[0])
                cand_dx = float(scale * s[1])
                err_f = float(err) if np.isfinite(err) else 1.0
                if err_f <= self.cfg.drift_max_err and float(np.hypot(cand_dy, cand_dx)) <= self.cfg.drift_max_jump:
                    dy, dx = cand_dy, cand_dx

        is_focused = True
        if phase_image is not None:
            score = compute_focus_score(phase_image)
            if self.baseline_focus is not None and self.baseline_focus > 0.0:
                if score < self.cfg.focus_drop_ratio * self.baseline_focus:
                    is_focused = False
            if is_focused:
                self.focus_history.append(score)
                if len(self.focus_history) <= self.cfg.focus_baseline_frames:
                    self.baseline_focus = float(np.median(self.focus_history))

        return is_focused, dy, dx

    def _finalize(
        self,
        st: _FounderState,
        decision: str,
        event_frame: int | None,
        lower_frame: int,
        upper_frame: int,
        reason: str,
    ) -> FounderResult:
        st.active = False
        res = FounderResult(
            founder_id=st.founder_id,
            decision=decision,
            event_frame=event_frame,
            lower_frame=int(lower_frame),
            upper_frame=int(upper_frame),
            last_identity_frame=int(st.last_seen),
            tracking_end_reason=reason,
            label0=st.label0,
            area0=st.area0,
            max_area=st.max_area,
            last_area=st.last_area,
            growth_ratio=round(st.last_area / max(st.area0, 1), 4),
            elongation_ratio=round(st.last_length / max(st.length0, 1.0), 4),
            pi_0=round(st.pi_0, 2) if st.pi_0 is not None else None,
            pi_max=round(st.pi_max, 2) if st.pi_max is not None else None,
            pi_last=round(st.pi_last, 2) if st.pi_last is not None else None,
            pi_surge_max=round(st.pi_surge_max, 2) if st.pi_0 is not None else None,
            r_0=round(st.r_0, 4) if st.r_0 is not None else None,
            r_last=round(st.r_last, 4) if st.r_last is not None else None,
            r_post=round(st.r_post, 4) if st.r_post is not None else None,
            delta_r=round(st.delta_r, 4) if st.delta_r is not None else None,
            contains_gap=st.contains_gap,
            contains_defocus_pause=st.contains_defocus_pause,
            used_transient_contact_split=st.used_transient_contact_split,
            flags=sorted(set(st.flags)),
            frame0_scores=dict(st.frame0_scores),
        )
        st.result = res
        return res

    def pause_defocused(self, frame_idx: int) -> None:
        """Pause all active founders on a defocused frame without incrementing ``missing``."""
        self.last_frame = max(self.last_frame, frame_idx)
        self.defocused_after_last_use = True
        for st in self.states.values():
            if st.active:
                st.contains_defocus_pause = True

    def step(
        self,
        frame_idx: int,
        labels: np.ndarray | None,
        phase_image: np.ndarray | None = None,
        pi_image: np.ndarray | None = None,
        shift_override: tuple[float, float] | None = None,
        dark_filter_applied: bool = False,
        force_focused: bool | None = None,
    ) -> dict[str, FounderResult]:
        """Advance tracking by one frame. Returns newly finalized founders on this frame."""
        self.last_frame = max(self.last_frame, frame_idx)
        cfg = self.cfg

        is_focused, dy, dx = self.check_focus_and_shift(frame_idx, phase_image, shift_override=shift_override)
        if force_focused is not None:
            is_focused = bool(force_focused)

        if not is_focused or labels is None:
            self.pause_defocused(frame_idx)
            return {}

        self.defocused_after_last_use = False
        self.cum_dy += dy
        self.cum_dx += dx
        self.cum_shifts[frame_idx] = (self.cum_dy, self.cum_dx)
        if phase_image is not None:
            self.prev_phase_bin = _central_binned(phase_image, cfg.drift_bin)[0]

        bg_p = fast_bg_median(phase_image) if phase_image is not None else None
        bg_pi = fast_bg_median(pi_image) if pi_image is not None else None
        if phase_image is not None and not dark_filter_applied:
            labs_f = filter_phase_dark_labels(labels, phase_image, bg_median=bg_p, dark_ratio_max=cfg.dark_ratio_max)
        else:
            labs_f = labels

        newly_finished: dict[str, FounderResult] = {}

        for fid, st in self.states.items():
            if not st.active:
                continue
            prev_cdy, prev_cdx = self.cum_shifts[st.prev_f]
            iy = int(round(self.cum_dy - prev_cdy))
            ix = int(round(self.cum_dx - prev_cdx))
            sby0, sby1 = st.by0 - iy, st.by1 - iy
            sbx0, sbx1 = st.bx0 - ix, st.bx1 - ix

            if sby1 <= 0 or sby0 >= self.H or sbx1 <= 0 or sbx0 >= self.W:
                newly_finished[fid] = self._finalize(st, "lost", frame_idx, st.last_seen, frame_idx, "edge")
                continue

            pad = cfg.pad
            y0, y1 = max(0, sby0 - pad), min(self.H, sby1 + pad)
            x0, x1 = max(0, sbx0 - pad), min(self.W, sbx1 + pad)

            fm = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            cy0, cy1 = max(0, y0 - sby0), min(st.by1 - st.by0, y1 - sby0)
            cx0, cx1 = max(0, x0 - sbx0), min(st.bx1 - st.bx0, x1 - sbx0)
            if cy1 > cy0 and cx1 > cx0:
                fm[sby0 + cy0 - y0 : sby0 + cy1 - y0, sbx0 + cx0 - x0 : sbx0 + cx1 - x0] = st.fam_crop[cy0:cy1, cx0:cx1]

            if not fm.any():
                res = self._handle_missing(st, frame_idx, fm, y0, y1, x0, x1, labs_f, phase_image, bg_p)
                if res is not None:
                    newly_finished[fid] = res
                continue

            oy0, oy1 = max(0, y0 + iy), min(self.H, y1 + iy)
            ox0, ox1 = max(0, x0 + ix), min(self.W, x1 + ix)
            om = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            if oy1 > oy0 and ox1 > ox0:
                om[oy0 - iy - y0 : oy1 - iy - y0, ox0 - ix - x0 : ox1 - ix - x0] = (
                    self.prev_usable_labels[oy0:oy1, ox0:ox1] > 0
                )
            om &= ~fm

            lab = labs_f[y0:y1, x0:x1]
            ids, inv = np.unique(lab, return_inverse=True)
            inv = inv.reshape(lab.shape)
            area = np.bincount(inv.ravel())

            d_fam = ndi.distance_transform_edt(~fm)
            d_oth = ndi.distance_transform_edt(~om) if om.any() else np.full_like(d_fam, np.inf)
            grown = (d_fam <= cfg.slide) & (d_fam < d_oth)
            on_fam = np.bincount(inv.ravel(), weights=grown.ravel())
            on_oth = np.bincount(inv.ravel(), weights=om.ravel())

            members: list[int] = []
            merged: list[int] = []
            for idx, lid in enumerate(ids):
                if lid == 0:
                    continue
                a = float(on_fam[idx] / area[idx])
                b = float(on_oth[idx] / area[idx])
                if a >= 0.5 and b < cfg.b_contact:
                    members.append(idx)
                elif b >= 0.5 and a < 0.5:
                    continue
                elif a >= 0.25 and b >= cfg.b_contact:
                    merged.append(idx)
                elif not cfg.through and a >= 0.5:
                    members.append(idx)

            split_mask: np.ndarray | None = None
            split_applied = False
            area_members = int(sum(area[i] for i in members))
            area_prev = int(fm.sum())
            need_split = bool(merged) and (area_members < cfg.split_ignore_invader_area_frac * area_prev)

            if need_split:
                if cfg.split_transient_contact and st.contact_streak == 0:
                    prior = np.where(fm, 1, np.where(om, 2, 0))
                    _, (piy, pix) = ndi.distance_transform_edt(prior == 0, return_indices=True)
                    nearest = prior[piy, pix]
                    m_merged = np.isin(inv, merged) & (nearest == 1)
                    if m_merged.any():
                        st.contact_streak = 1
                        st.used_transient_contact_split = True
                        split_mask = m_merged
                        split_applied = True
                    else:
                        newly_finished[fid] = self._finalize(st, "lost", frame_idx, st.last_seen, frame_idx, "contact")
                        continue
                else:
                    newly_finished[fid] = self._finalize(st, "lost", frame_idx, st.last_seen, frame_idx, "contact")
                    continue
            else:
                st.contact_streak = 0

            if not members and split_mask is None:
                res = self._handle_missing(st, frame_idx, fm, y0, y1, x0, x1, labs_f, phase_image, bg_p)
                if res is not None:
                    newly_finished[fid] = res
                continue

            # Track recovered cleanly on frame_idx
            st.missing = 0
            st.first_missing = None
            m_clean = np.isin(inv, members) if members else np.zeros_like(fm)
            m = (m_clean | split_mask) if split_mask is not None else m_clean
            mys, mxs = np.nonzero(m)

            expanded_to_full = False
            if (
                (int(mys.min()) == 0 and y0 > 0)
                or (int(mxs.min()) == 0 and x0 > 0)
                or (int(mys.max()) == y1 - y0 - 1 and y1 < self.H)
                or (int(mxs.max()) == x1 - x0 - 1 and x1 < self.W)
            ):
                if members and split_mask is None:
                    member_lids = ids[members]
                    full_m = np.isin(labs_f, member_lids)
                    fys, fxs = np.nonzero(full_m)
                    if not st.started_at_edge and (
                        int(fys.min()) == 0 or int(fxs.min()) == 0 or int(fys.max()) == self.H - 1 or int(fxs.max()) == self.W - 1
                    ):
                        newly_finished[fid] = self._finalize(st, "lost", frame_idx, st.last_seen, frame_idx, "edge")
                        continue
                    by0, by1 = int(fys.min()), int(fys.max()) + 1
                    bx0, bx1 = int(fxs.min()), int(fxs.max()) + 1
                    st.fam_crop = full_m[by0:by1, bx0:bx1]
                    st.by0, st.by1, st.bx0, st.bx1 = by0, by1, bx0, bx1
                    m_sum = int(st.fam_crop.sum())
                    sub_lab = labs_f[by0:by1, bx0:bx1]
                    big_count = sum(1 for lid in member_lids if int((sub_lab == lid).sum()) >= cfg.frac * m_sum)
                    expanded_to_full = True

            if not expanded_to_full:
                if not st.started_at_edge and (
                    int(mys.min()) + y0 == 0
                    or int(mxs.min()) + x0 == 0
                    or int(mys.max()) + y0 == self.H - 1
                    or int(mxs.max()) + x0 == self.W - 1
                ):
                    newly_finished[fid] = self._finalize(st, "lost", frame_idx, st.last_seen, frame_idx, "edge")
                    continue
                by0, by1 = int(mys.min()) + y0, int(mys.max()) + y0 + 1
                bx0, bx1 = int(mxs.min()) + x0, int(mxs.max()) + x0 + 1
                st.fam_crop = m[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0]
                st.by0, st.by1, st.bx0, st.bx1 = by0, by1, bx0, bx1
                m_sum = int(st.fam_crop.sum())
                if split_applied and not members:
                    # Single merged blob split by EDT alone must not fabricate a 2-daughter split
                    big_count = 1
                else:
                    big_areas = [int(area[i]) for i in members]
                    if split_mask is not None:
                        big_areas.append(int(split_mask.sum()))
                    big_count = sum(1 for a_px in big_areas if a_px >= cfg.frac * m_sum)

            st.last_area = m_sum
            st.max_area = max(st.max_area, m_sum)
            cur_len = _major_axis_length(st.fam_crop)
            st.last_length = cur_len
            st.max_length = max(st.max_length, cur_len)

            # Channel measurements on tracked mask
            if phase_image is not None or pi_image is not None:
                p_ring = cfg.ring_outer_px + 2
                ry0, ry1 = max(0, st.by0 - p_ring), min(self.H, st.by1 + p_ring)
                rx0, rx1 = max(0, st.bx0 - p_ring), min(self.W, st.bx1 + p_ring)
                sub_f_mask = np.zeros((ry1 - ry0, rx1 - rx0), dtype=bool)
                sub_f_mask[st.by0 - ry0 : st.by1 - ry0, st.bx0 - rx0 : st.bx1 - rx0] = st.fam_crop
                r_cur, pi_cur, _, pi_se = measure_footprint_channels(
                    sub_f_mask,
                    phase_image[ry0:ry1, rx0:rx1] if phase_image is not None else np.ones_like(sub_f_mask, dtype=np.float32),
                    pi_crop=pi_image[ry0:ry1, rx0:rx1] if pi_image is not None else None,
                    dark_labels_crop=labs_f[ry0:ry1, rx0:rx1],
                    bg_phase=bg_p,
                    bg_pi_fallback=bg_pi,
                    ring_inner_px=cfg.ring_inner_px,
                    ring_outer_px=cfg.ring_outer_px,
                    min_ring_pixels=cfg.min_ring_pixels,
                )
                if phase_image is not None:
                    st.r_last = r_cur
                if pi_cur is not None:
                    st.pi_last = pi_cur
                    st.pi_max = pi_cur if st.pi_max is None else max(st.pi_max, pi_cur)
                    rmin = pi_cur if st.pi_running_min is None else st.pi_running_min
                    surge = pi_cur - rmin
                    if surge > st.pi_surge_max:
                        st.pi_surge_max = surge
                    thr = max(cfg.pi_abs_min, cfg.pi_noise_mult * (pi_se or 0.0))
                    if pi_cur >= thr and surge >= cfg.pi_surge_min:
                        if st.pi_surge_streak == 0:
                            st.pi_surge_first_frame = frame_idx
                            st.pi_surge_prev_frame = st.last_seen
                        st.pi_surge_streak += 1
                    else:
                        st.pi_surge_streak = 0
                        st.pi_surge_first_frame = None
                        st.pi_surge_prev_frame = None
                    st.pi_running_min = min(rmin, pi_cur)

                    if (
                        st.pi_0 is not None
                        and pi_cur >= cfg.pi_high_baseline_min
                        and pi_cur >= cfg.pi_high_baseline_decay * st.pi_0
                        and m_sum <= cfg.pi_high_baseline_growth * st.area0
                    ):
                        if st.pi_high_streak == 0:
                            st.pi_high_first_frame = frame_idx
                            st.pi_high_prev_frame = st.last_seen
                        st.pi_high_streak += 1
                    else:
                        st.pi_high_streak = 0
                        st.pi_high_first_frame = None
                        st.pi_high_prev_frame = None

                    if st.pi_surge_streak >= cfg.pi_streak and m_sum < cfg.growth * st.area0:
                        ev_f = int(st.pi_surge_first_frame if st.pi_surge_first_frame is not None else frame_idx)
                        lo_f = int(st.pi_surge_prev_frame if st.pi_surge_prev_frame is not None else st.last_seen)
                        st.last_seen = frame_idx
                        newly_finished[fid] = self._finalize(
                            st, "pi_positive_death", ev_f, lo_f, ev_f, "pi_positive_death"
                        )
                        continue

                    if st.pi_high_streak >= cfg.pi_high_baseline_streak:
                        ev_f = int(st.pi_high_first_frame if st.pi_high_first_frame is not None else frame_idx)
                        lo_f = int(st.pi_high_prev_frame if st.pi_high_prev_frame is not None else st.last_seen)
                        st.last_seen = frame_idx
                        newly_finished[fid] = self._finalize(
                            st, "pi_positive_death", ev_f, lo_f, ev_f, "pi_positive_death"
                        )
                        continue

            n_daughters = big_count if m_sum >= cfg.growth * st.area0 else 1
            st.counts.append((frame_idx, n_daughters))
            st.last_seen = frame_idx
            st.prev_f = frame_idx

            div_frame = _first_division(st.counts, cfg.persist)
            if div_frame is not None:
                div_idx = next(i for i, (fr, _) in enumerate(st.counts) if fr == div_frame)
                lo_frame = st.counts[div_idx - 1][0] if div_idx > 0 else 0
                newly_finished[fid] = self._finalize(
                    st, "division", div_frame, lo_frame, div_frame, "division"
                )
                continue

            # If a single non-growing cell surges in PI even slightly after reaching >= growth (edge case: already caught if < growth)
            if st.pi_surge_streak >= cfg.pi_streak:
                ev_f = int(st.pi_surge_first_frame if st.pi_surge_first_frame is not None else frame_idx)
                lo_f = int(st.pi_surge_prev_frame if st.pi_surge_prev_frame is not None else st.last_seen)
                newly_finished[fid] = self._finalize(
                    st, "pi_positive_death", ev_f, lo_f, ev_f, "pi_positive_death"
                )

        self.prev_usable_labels = labs_f
        self.last_usable_frame = frame_idx
        return newly_finished

    def _handle_missing(
        self,
        st: _FounderState,
        frame_idx: int,
        fm_roi: np.ndarray,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        labs_f: np.ndarray,
        phase_image: np.ndarray | None,
        bg_p: float | None,
    ) -> FounderResult | None:
        cfg = self.cfg
        st.missing += 1
        if st.missing == 1:
            st.first_missing = frame_idx
            # Anti-shove ghost lysis test on the FIRST missing frame
            if phase_image is not None and fm_roi.any() and st.r_last is not None:
                bg_phase_val = bg_p if (bg_p is not None and bg_p > 0.0) else fast_bg_median(phase_image)
                r_post = (float(np.median(phase_image[y0:y1, x0:x1][fm_roi])) - bg_phase_val) / max(bg_phase_val, 1e-6)
                delta_r = r_post - st.r_last
                st.r_post = r_post
                st.delta_r = delta_r
                if delta_r >= cfg.ghost_delta_r and r_post >= cfg.ghost_r_post_min:
                    dil_clear = ndi.binary_dilation(fm_roi, iterations=cfg.ghost_clear_radius_px)
                    occupied_or_near = bool((labs_f[y0:y1, x0:x1][dil_clear] > 0).any())
                    if not occupied_or_near:
                        return self._finalize(
                            st,
                            "ghost_lysis",
                            frame_idx,
                            st.last_seen,
                            frame_idx,
                            "ghost_lysis",
                        )
        if st.missing > 0:
            st.contains_gap = True
        if st.missing > cfg.max_gap:
            miss_start = st.first_missing if st.first_missing is not None else frame_idx
            return self._finalize(st, "lost", miss_start, st.last_seen, miss_start, "missing")
        return None

    def finish(self, recording_last_frame: int | None = None) -> dict[str, FounderResult]:
        """Finalize any remaining active founders at end of recording."""
        if recording_last_frame is None:
            recording_last_frame = self.last_frame
        results: dict[str, FounderResult] = {}
        for fid, st in self.states.items():
            if not st.active:
                assert st.result is not None
                results[fid] = st.result
                continue
            if self.defocused_after_last_use or st.last_seen < self.last_usable_frame:
                miss_start = st.first_missing if st.first_missing is not None else min(recording_last_frame, st.last_seen + 1)
                reason = "defocus_truncated" if self.defocused_after_last_use and st.first_missing is None else "missing"
                results[fid] = self._finalize(st, "lost", miss_start, st.last_seen, miss_start, reason)
            elif st.last_seen >= recording_last_frame or st.last_seen >= self.last_usable_frame:
                results[fid] = self._finalize(
                    st, "no_division", None, recording_last_frame, recording_last_frame, "recording_end"
                )
            else:
                results[fid] = self._finalize(st, "lost", st.last_seen, st.last_seen, st.last_seen, "missing")
        return results


def track_field(
    labels_seq: Sequence[np.ndarray],
    founders: Mapping[str, tuple[float, float]],
    phase_seq: Sequence[np.ndarray] | None = None,
    pi_seq: Sequence[np.ndarray] | None = None,
    yfp0: np.ndarray | None = None,
    shifts: Mapping[int, tuple[float, float]] | None = None,
    dark_filter_applied: bool = False,
    cfg: TrackerConfig | None = None,
) -> dict[str, FounderResult]:
    """Track all founders in a single field across ``labels_seq``."""
    if not labels_seq:
        return {}
    cfg = cfg or TrackerConfig()
    tracker = FieldTracker(
        labels_seq[0],
        founders,
        phase0=phase_seq[0] if phase_seq is not None else None,
        pi0=pi_seq[0] if pi_seq is not None else None,
        yfp0=yfp0,
        dark_filter_applied=dark_filter_applied,
        cfg=cfg,
    )
    for f in range(1, len(labels_seq)):
        if not tracker.active_ids():
            break
        tracker.step(
            f,
            labels_seq[f],
            phase_image=phase_seq[f] if phase_seq is not None else None,
            pi_image=pi_seq[f] if pi_seq is not None else None,
            shift_override=shifts.get(f) if shifts is not None else None,
            dark_filter_applied=dark_filter_applied,
        )
    return tracker.finish(recording_last_frame=len(labels_seq) - 1)


def track_founder(
    labels_seq: Sequence[np.ndarray],
    cy: float,
    cx: float,
    phase_seq: Sequence[np.ndarray] | None = None,
    pi_seq: Sequence[np.ndarray] | None = None,
    yfp0: np.ndarray | None = None,
    shifts: Mapping[int, tuple[float, float]] | None = None,
    dark_filter_applied: bool = False,
    cfg: TrackerConfig | None = None,
    founder_id: str = "founder",
) -> FounderResult:
    """Convenience single-founder wrapper around :func:`track_field`."""
    return track_field(
        labels_seq,
        {founder_id: (cy, cx)},
        phase_seq=phase_seq,
        pi_seq=pi_seq,
        yfp0=yfp0,
        shifts=shifts,
        dark_filter_applied=dark_filter_applied,
        cfg=cfg,
    )[founder_id]

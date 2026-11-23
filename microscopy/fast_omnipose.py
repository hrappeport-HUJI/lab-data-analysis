"""Stop-on-Event Packed Active-ROI Mosaic Omnipose GPU Runner (FP32).

Runs ``bact_phase_omni`` in FP32 without FP16 autocast (which overflows in
``res_down_0``), using fast ``[::4, ::4]`` quantile normalization, compact
shelf-packed active-founder ROI canvases padded with normalized frame background
median, online ``FieldTracker`` feedback to retire completed founders immediately
upon ``division``, ``pi_positive_death``, ``ghost_lysis``, or ``lost``, and
asynchronous ``zlib`` level-1 ``.npz`` writing on a background thread.

Pure-CPU helpers (:func:`normalize_frame_quantiles`, :func:`pack_active_rois`,
:func:`unpack_mosaic_labels`, :func:`save_npz_zlib1`, :class:`AsyncNpzWriter`)
import only standard-library / NumPy / SciPy packages so unit tests can run
in either CPU or GPU Python environments.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from microscopy.founder_tracker import (
    FieldTracker,
    FounderResult,
    TrackerConfig,
    fast_bg_median,
    filter_phase_dark_labels,
)


@dataclass(frozen=True)
class TilePlacement:
    """Mapping between a global frame ROI and its packed mosaic canvas slot."""

    founder_id: str
    # Desired global crop bounds clipped to image shape
    sy0: int
    sy1: int
    sx0: int
    sx1: int
    # Top-left coordinate of the active image crop inside the mosaic canvas
    ty0: int
    tx0: int
    # Center in global coordinates (used to resolve overlapping ROI write regions)
    center_y: float
    center_x: float

    @property
    def height(self) -> int:
        return self.sy1 - self.sy0

    @property
    def width(self) -> int:
        return self.sx1 - self.sx0


def normalize_frame_quantiles(
    image: np.ndarray,
    step: int = 4,
    q_lo: float = 0.0001,
    q_hi: float = 0.9999,
) -> tuple[float, float, float]:
    """Compute fast strided quantile bounds and background median.

    Matches Omnipose's ``0.01% - 99.99%`` ``normalize99`` scaling in ~1-2 ms
    on a ``2048x2048`` image instead of ~230 ms for full-array percentiles.
    """
    h, w = image.shape[:2]
    sub = image[::step, ::step] if min(h, w) >= step * 16 else image
    lo, hi = np.quantile(sub, [q_lo, q_hi])
    med = float(np.median(sub))
    return float(lo), float(hi), med


def normalize_array(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize an array to FP32 ``[0, 1]`` using precomputed bounds."""
    scale = max(float(hi - lo), 1e-6)
    out = (image.astype(np.float32, copy=False) - float(lo)) / scale
    return np.clip(out, 0.0, 1.0, out=out if out.dtype == np.float32 and image is not out else None)


def pack_active_rois(
    phase_image: np.ndarray,
    active_bboxes: Mapping[str, tuple[int, int, int, int]],
    lo: float | None = None,
    hi: float | None = None,
    bg_median: float | None = None,
    min_roi_size: int = 160,
    context_pad: int = 16,
    align_multiple: int = 16,
) -> tuple[np.ndarray, list[TilePlacement], tuple[float, float, float]]:
    """Pack active-founder bounding boxes into a compact FP32 normalized canvas.

    Uses shelf bin-packing sorted by descending tile height so large filaments
    do not inflate the slot height of compact single cells, while keeping at
    least ``context_pad`` pixels of normalized frame background median around
    every tile so UNet boundary effects never leak across adjacent ROIs.

    Returns:
        ``(canvas, placements, (lo, hi, bg_median))``
    """
    if lo is None or hi is None or bg_median is None:
        q_lo, q_hi, q_med = normalize_frame_quantiles(phase_image)
        lo = q_lo if lo is None else float(lo)
        hi = q_hi if hi is None else float(hi)
        bg_median = q_med if bg_median is None else float(bg_median)

    scale = max(float(hi - lo), 1e-6)
    norm_med = float(np.clip((float(bg_median) - float(lo)) / scale, 0.0, 1.0))

    if not active_bboxes:
        sz = max(align_multiple, 64)
        return np.full((sz, sz), norm_med, dtype=np.float32), [], (float(lo), float(hi), float(bg_median))

    H, W = phase_image.shape[:2]
    items: list[tuple[str, int, int, int, int, int, int]] = []
    total_slot_area = 0
    max_slot_w = 0

    for fid, (y0, y1, x0, x1) in active_bboxes.items():
        cy = 0.5 * (y0 + y1)
        cx = 0.5 * (x0 + x1)
        req_h = max(min_roi_size, int(y1 - y0))
        req_w = max(min_roi_size, int(x1 - x0))
        ry0 = max(0, int(round(cy - 0.5 * req_h)))
        rx0 = max(0, int(round(cx - 0.5 * req_w)))
        ry1 = min(H, ry0 + req_h)
        rx1 = min(W, rx0 + req_w)
        ry0 = max(0, ry1 - req_h)
        rx0 = max(0, rx1 - req_w)
        ch = ry1 - ry0
        cw = rx1 - rx0
        slot_h = ch + 2 * context_pad
        slot_w = cw + 2 * context_pad
        total_slot_area += slot_h * slot_w
        max_slot_w = max(max_slot_w, slot_w)
        items.append((fid, ry0, ry1, rx0, rx1, slot_h, slot_w))

    # Sort tallest first for tight shelf packing
    items.sort(key=lambda t: (-t[5], -t[6], t[0]))
    target_w = max(max_slot_w, int(math.ceil(math.sqrt(total_slot_area * 1.08))))
    target_w = int(math.ceil(target_w / align_multiple) * align_multiple)

    # First pass: assign shelf coordinates
    coords: list[tuple[str, int, int, int, int, int, int]] = []
    cur_y = 0
    cur_x = 0
    shelf_h = 0
    used_w = 0
    for fid, sy0, sy1, sx0, sx1, slot_h, slot_w in items:
        if cur_x > 0 and cur_x + slot_w > target_w:
            cur_y += shelf_h
            cur_x = 0
            shelf_h = 0
        ty0 = cur_y + context_pad
        tx0 = cur_x + context_pad
        coords.append((fid, sy0, sy1, sx0, sx1, ty0, tx0))
        cur_x += slot_w
        used_w = max(used_w, cur_x)
        shelf_h = max(shelf_h, slot_h)

    used_h = cur_y + shelf_h
    canvas_h = max(align_multiple, int(math.ceil(used_h / align_multiple) * align_multiple))
    canvas_w = max(align_multiple, int(math.ceil(used_w / align_multiple) * align_multiple))

    canvas = np.full((canvas_h, canvas_w), norm_med, dtype=np.float32)
    placements: list[TilePlacement] = []
    inv_scale = np.float32(1.0 / scale)
    lo_f = np.float32(lo)

    for fid, sy0, sy1, sx0, sx1, ty0, tx0 in coords:
        h_v = sy1 - sy0
        w_v = sx1 - sx0
        if h_v > 0 and w_v > 0:
            crop = (phase_image[sy0:sy1, sx0:sx1].astype(np.float32) - lo_f) * inv_scale
            np.clip(crop, 0.0, 1.0, out=crop)
            canvas[ty0 : ty0 + h_v, tx0 : tx0 + w_v] = crop
        placements.append(
            TilePlacement(
                founder_id=fid,
                sy0=sy0,
                sy1=sy1,
                sx0=sx0,
                sx1=sx1,
                ty0=ty0,
                tx0=tx0,
                center_y=0.5 * (sy0 + sy1),
                center_x=0.5 * (sx0 + sx1),
            )
        )

    return canvas, placements, (float(lo), float(hi), float(bg_median))


def unpack_mosaic_labels(
    canvas_labels: np.ndarray,
    placements: Sequence[TilePlacement],
    frame_shape: tuple[int, int],
    drop_tile_border_px: int = 2,
    phase_image: np.ndarray | None = None,
    bg_median: float | None = None,
    dark_ratio_max: float = 0.85,
) -> np.ndarray:
    """Unpack mosaic label tiles back into a global ``(H, W)`` integer array.

    If two active founders' padded ROIs overlap in global frame coordinates,
    pixels inside the intersection are assigned to whichever tile has its
    interior closer to the pixel (distance-to-edge priority), avoiding
    truncating objects near an adjacent tile's margin.

    When ``phase_image`` is supplied, phase-darkness ripple filtering runs
    per tile before copying into the full ``(H, W)`` array, which is
    mathematically equivalent to whole-frame filtering while avoiding a full
    ``H x W`` scan on sparse frames.
    """
    H, W = int(frame_shape[0]), int(frame_shape[1])
    out = np.zeros((H, W), dtype=np.int32)
    if not placements or int(canvas_labels.max()) == 0:
        return out

    # Only allocate an HxW margin map when at least two tiles overlap globally
    has_overlap = False
    n_pl = len(placements)
    for i in range(n_pl):
        a = placements[i]
        for j in range(i + 1, n_pl):
            b = placements[j]
            if a.sy0 < b.sy1 and b.sy0 < a.sy1 and a.sx0 < b.sx1 and b.sx0 < a.sx1:
                has_overlap = True
                break
        if has_overlap:
            break

    best_margin = np.full((H, W), -1.0, dtype=np.float32) if has_overlap else None
    if phase_image is not None and bg_median is None:
        bg_median = fast_bg_median(phase_image)
    next_lbl = 1

    for pl in placements:
        h_v = pl.height
        w_v = pl.width
        if h_v <= 0 or w_v <= 0:
            continue
        sub = canvas_labels[pl.ty0 : pl.ty0 + h_v, pl.tx0 : pl.tx0 + w_v]
        if drop_tile_border_px > 0 and h_v > 2 * drop_tile_border_px and w_v > 2 * drop_tile_border_px:
            # Drop labels touching the outer margin of the unpacked crop ONLY on sides
            # that do not touch the actual image frame boundary.
            bmask = np.zeros((h_v, w_v), dtype=bool)
            k = drop_tile_border_px
            if pl.sy0 > 0:
                bmask[:k, :] = True
            if pl.sy1 < H:
                bmask[-k:, :] = True
            if pl.sx0 > 0:
                bmask[:, :k] = True
            if pl.sx1 < W:
                bmask[:, -k:] = True
            if bmask.any():
                b_ids = np.unique(sub[bmask])
                b_ids = b_ids[b_ids > 0]
                if b_ids.size > 0:
                    sub = np.where(np.isin(sub, b_ids), 0, sub)

        if phase_image is not None:
            sub = filter_phase_dark_labels(
                sub,
                phase_image[pl.sy0 : pl.sy1, pl.sx0 : pl.sx1],
                bg_median=bg_median,
                dark_ratio_max=dark_ratio_max,
            )

        fg = sub > 0
        if not fg.any():
            continue

        # Relabel consecutively inside tile to keep global label IDs compact
        uniq_ids, inv = np.unique(sub[fg], return_inverse=True)
        mapped = np.zeros_like(sub, dtype=np.int32)
        mapped[fg] = (inv + next_lbl).astype(np.int32)
        next_lbl += len(uniq_ids)

        dst_lbl = out[pl.sy0 : pl.sy1, pl.sx0 : pl.sx1]
        if best_margin is not None:
            yy = np.minimum(np.arange(h_v, dtype=np.float32), (h_v - 1) - np.arange(h_v, dtype=np.float32))[:, None]
            xx = np.minimum(np.arange(w_v, dtype=np.float32), (w_v - 1) - np.arange(w_v, dtype=np.float32))[None, :]
            margin = np.minimum(yy, xx)
            dst_mgn = best_margin[pl.sy0 : pl.sy1, pl.sx0 : pl.sx1]
            take = fg & ((dst_lbl == 0) | (margin > dst_mgn))
            dst_lbl[take] = mapped[take]
            dst_mgn[take] = margin[take]
        else:
            dst_lbl[fg] = mapped[fg]

    return out


def save_npz_zlib1(path: str | Path, labels: np.ndarray) -> None:
    """Write ``labels`` to an ``.npz`` archive with fast zlib level-1 compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_val = int(labels.max()) if labels.size else 0
    arr = labels.astype(np.uint16 if max_val < 65535 else np.uint32, copy=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        with zf.open("labels.npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, arr, allow_pickle=False)


class AsyncNpzWriter:
    """Background-thread writer for fast zlib level-1 ``.npz`` label planes."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[concurrent.futures.Future[None]] = []

    def submit(self, path: str | Path, labels: np.ndarray) -> None:
        self._futures.append(self._executor.submit(save_npz_zlib1, path, labels.copy()))

    def wait_all(self) -> float:
        t0 = time.perf_counter()
        for fut in self._futures:
            fut.result()
        self._futures.clear()
        return time.perf_counter() - t0

    def close(self) -> None:
        try:
            self.wait_all()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncNpzWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@dataclass
class FrameProfile:
    frame: int
    n_active_in: int
    n_active_out: int
    focused: bool
    canvas_shape: tuple[int, int]
    area_ratio: float
    t_read_ms: float
    t_shift_focus_ms: float
    t_pack_ms: float
    t_net_ms: float
    t_post_ms: float
    t_unpack_ms: float
    t_track_ms: float
    t_total_ms: float


class FastOmniposeRunner:
    """GPU Stop-on-Event Packed Active-ROI Omnipose + Online FounderTracker."""

    def __init__(
        self,
        model_type: str = "bact_phase_omni",
        gpu: bool = True,
        min_roi_size: int = 160,
        context_pad: int = 16,
        cfg: TrackerConfig | None = None,
    ) -> None:
        self.model_type = model_type
        self.gpu = gpu
        self.min_roi_size = int(min_roi_size)
        self.context_pad = int(context_pad)
        self.cfg = cfg or TrackerConfig()
        self._model = None
        self._torch = None
        self._transforms = None
        self._omnipose_core = None

    def load_model(self) -> float:
        """Lazily load ``bact_phase_omni`` onto GPU (FP32)."""
        if self._model is not None:
            return 0.0
        t0 = time.perf_counter()
        import torch
        from cellpose_omni import models, transforms
        import omnipose.core

        self._torch = torch
        self._transforms = transforms
        self._omnipose_core = omnipose.core
        self._model = models.CellposeModel(gpu=self.gpu, model_type=self.model_type)
        return time.perf_counter() - t0

    def segment_normalized_fp32(self, norm_img: np.ndarray, normalize_inside_net: bool = False) -> tuple[np.ndarray, float, float]:
        """Run FP32 ``bact_phase_omni`` + ``compute_masks`` on a pre-normalized ``[0, 1]`` 2D image."""
        self.load_model()
        assert self._model is not None
        assert self._transforms is not None
        assert self._omnipose_core is not None
        assert self._torch is not None

        xc = self._transforms.convert_image(
            norm_img.astype(np.float32, copy=False), [0, 0], None, None, False, False, nchan=self._model.nchan
        )
        if self.gpu and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        yf, _ = self._model._run_nets(
            xc,
            net_avg=True,
            augment=False,
            tile=False,
            normalize=normalize_inside_net,
        )
        if self.gpu and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        t_net = time.perf_counter() - t0

        yf = yf.astype(np.float32, copy=False)
        dP = yf[..., :2].transpose(2, 0, 1).copy()
        cp = yf[..., 2].copy()
        bd = yf[..., 3].copy() if self._model.nclasses >= 4 else None

        t1 = time.perf_counter()
        masks = self._omnipose_core.compute_masks(
            dP,
            cp,
            bd=bd,
            mask_threshold=0,
            flow_threshold=0,
            cluster=False,
            omni=True,
            use_gpu=self._model.gpu,
            device=self._model.device,
            nclasses=self._model.nclasses,
            dim=self._model.dim,
        )[0]
        if self.gpu and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
        t_post = time.perf_counter() - t1
        return np.asarray(masks, dtype=np.int32), t_net, t_post

    def warmup(self, sample_image: np.ndarray | None = None) -> None:
        """Run one small dummy pass to warm up cuDNN & CUDA kernels."""
        self.load_model()
        if sample_image is not None:
            h, w = sample_image.shape[:2]
            crop = sample_image[: min(256, h), : min(256, w)].astype(np.float32)
            lo, hi, _ = normalize_frame_quantiles(crop)
            norm = normalize_array(crop, lo, hi)
        else:
            norm = np.zeros((256, 256), dtype=np.float32)
        _ = self.segment_normalized_fp32(norm)

    def run_stack(
        self,
        num_frames: int,
        phase_reader: Callable[[int], np.ndarray],
        pi_reader: Callable[[int], np.ndarray | None] | None = None,
        yfp0_reader: Callable[[], np.ndarray | None] | None = None,
        founders: Mapping[str, tuple[float, float]] | None = None,
        save_labels_dir: str | Path | None = None,
        edge_margin_px: int = 12,
        normalize_inside_net: bool = False,
    ) -> tuple[dict[str, FounderResult], list[FrameProfile], float]:
        """Run frame-0 full-frame + online stop-on-event packed-ROI segmentation & tracking.

        Args:
            num_frames: Number of frames ``0 .. num_frames-1``.
            phase_reader: Callable ``(frame_idx) -> 2D phase image``.
            pi_reader: Optional callable ``(frame_idx) -> 2D PI image or None``.
            yfp0_reader: Optional callable ``() -> frame-0 YFP image or None``.
            founders: Optional explicit mapping ``{founder_id: (y, x)}``. If omitted,
                all phase-dark segmented objects on frame 0 away from the border
                are initialized as founders.
            save_labels_dir: If given, asynchronously saves per-frame ``<frame:09d>.npz``
                using ``zlib`` level 1.
            edge_margin_px: Border margin when auto-detecting founders on frame 0.
            normalize_inside_net: Passed to ``_run_nets(normalize=...)``. Default
                ``False`` because inputs are already ``[::4, ::4]`` quantile-normalized.

        Returns:
            ``(results_by_founder, frame_profiles, writer_wait_s)``
        """
        self.load_model()
        writer = AsyncNpzWriter(max_workers=2) if save_labels_dir is not None else None
        out_dir = Path(save_labels_dir) if save_labels_dir is not None else None
        profiles: list[FrameProfile] = []

        try:
            # Frame 0: Full-frame FP32 Omnipose
            t_fr0 = time.perf_counter()
            t0 = time.perf_counter()
            ph0 = phase_reader(0)
            pi0 = pi_reader(0) if pi_reader is not None else None
            yfp0 = yfp0_reader() if yfp0_reader is not None else None
            t_read0 = time.perf_counter() - t0

            t0 = time.perf_counter()
            lo0, hi0, bg0 = normalize_frame_quantiles(ph0)
            norm0 = normalize_array(ph0, lo0, hi0)
            t_norm0 = time.perf_counter() - t0

            raw_l0, t_net0, t_post0 = self.segment_normalized_fp32(
                norm0, normalize_inside_net=normalize_inside_net
            )

            t0 = time.perf_counter()
            l0 = filter_phase_dark_labels(raw_l0, ph0, bg_median=bg0, dark_ratio_max=self.cfg.dark_ratio_max)

            if founders is None:
                from skimage.measure import regionprops

                H0, W0 = l0.shape
                auto_founders: dict[str, tuple[float, float]] = {}
                for p in regionprops(l0):
                    cy, cx = float(p.centroid[0]), float(p.centroid[1])
                    if (
                        edge_margin_px <= cy < H0 - edge_margin_px
                        and edge_margin_px <= cx < W0 - edge_margin_px
                    ):
                        auto_founders[f"f0-o{int(p.label)}"] = (cy, cx)
                founders_map: Mapping[str, tuple[float, float]] = auto_founders
            else:
                founders_map = founders

            tracker = FieldTracker(
                l0,
                founders_map,
                phase0=ph0,
                pi0=pi0,
                yfp0=yfp0,
                dark_filter_applied=True,
                cfg=self.cfg,
            )
            t_track0 = time.perf_counter() - t0
            if writer is not None and out_dir is not None:
                writer.submit(out_dir / "000000000.npz", l0)

            n_act = len(tracker.active_ids())
            profiles.append(
                FrameProfile(
                    frame=0,
                    n_active_in=len(founders_map),
                    n_active_out=n_act,
                    focused=True,
                    canvas_shape=(ph0.shape[0], ph0.shape[1]),
                    area_ratio=1.0,
                    t_read_ms=t_read0 * 1000.0,
                    t_shift_focus_ms=0.0,
                    t_pack_ms=t_norm0 * 1000.0,
                    t_net_ms=t_net0 * 1000.0,
                    t_post_ms=t_post0 * 1000.0,
                    t_unpack_ms=0.0,
                    t_track_ms=t_track0 * 1000.0,
                    t_total_ms=(time.perf_counter() - t_fr0) * 1000.0,
                )
            )

            # Frames 1 .. num_frames - 1: Stop-on-Event Packed Active-ROI Mosaic
            H, W = ph0.shape[:2]
            for f in range(1, num_frames):
                n_in = len(tracker.active_ids())
                if n_in == 0:
                    break

                t_fr = time.perf_counter()
                t0 = time.perf_counter()
                ph_f = phase_reader(f)
                pi_f = pi_reader(f) if pi_reader is not None else None
                t_read = time.perf_counter() - t0

                t0 = time.perf_counter()
                is_focused, dy, dx = tracker.check_focus_and_shift(f, ph_f)
                t_sf = time.perf_counter() - t0

                if not is_focused:
                    t0 = time.perf_counter()
                    tracker.pause_defocused(f)
                    t_tr = time.perf_counter() - t0
                    profiles.append(
                        FrameProfile(
                            frame=f,
                            n_active_in=n_in,
                            n_active_out=len(tracker.active_ids()),
                            focused=False,
                            canvas_shape=(0, 0),
                            area_ratio=0.0,
                            t_read_ms=t_read * 1000.0,
                            t_shift_focus_ms=t_sf * 1000.0,
                            t_pack_ms=0.0,
                            t_net_ms=0.0,
                            t_post_ms=0.0,
                            t_unpack_ms=0.0,
                            t_track_ms=t_tr * 1000.0,
                            t_total_ms=(time.perf_counter() - t_fr) * 1000.0,
                        )
                    )
                    continue

                t0 = time.perf_counter()
                bboxes = tracker.active_bboxes(next_dy=dy, next_dx=dx, pad=self.cfg.pad)
                lo_f, hi_f, bg_f = normalize_frame_quantiles(ph_f)
                canvas, placements, _ = pack_active_rois(
                    ph_f,
                    bboxes,
                    lo=lo_f,
                    hi=hi_f,
                    bg_median=bg_f,
                    min_roi_size=self.min_roi_size,
                    context_pad=self.context_pad,
                )
                t_pack = time.perf_counter() - t0

                m_canvas, t_net, t_post = self.segment_normalized_fp32(
                    canvas, normalize_inside_net=normalize_inside_net
                )

                t0 = time.perf_counter()
                filtered = unpack_mosaic_labels(
                    m_canvas,
                    placements,
                    (H, W),
                    phase_image=ph_f,
                    bg_median=bg_f,
                    dark_ratio_max=self.cfg.dark_ratio_max,
                )
                t_unpack = time.perf_counter() - t0

                t0 = time.perf_counter()
                tracker.step(
                    f,
                    filtered,
                    phase_image=ph_f,
                    pi_image=pi_f,
                    shift_override=(dy, dx),
                    dark_filter_applied=True,
                    force_focused=True,
                )
                t_track = time.perf_counter() - t0

                if writer is not None and out_dir is not None:
                    writer.submit(out_dir / f"{f:09d}.npz", filtered)

                ch, cw = canvas.shape[:2]
                profiles.append(
                    FrameProfile(
                        frame=f,
                        n_active_in=n_in,
                        n_active_out=len(tracker.active_ids()),
                        focused=True,
                        canvas_shape=(ch, cw),
                        area_ratio=float(ch * cw) / float(H * W),
                        t_read_ms=t_read * 1000.0,
                        t_shift_focus_ms=t_sf * 1000.0,
                        t_pack_ms=t_pack * 1000.0,
                        t_net_ms=t_net * 1000.0,
                        t_post_ms=t_post * 1000.0,
                        t_unpack_ms=t_unpack * 1000.0,
                        t_track_ms=t_track * 1000.0,
                        t_total_ms=(time.perf_counter() - t_fr) * 1000.0,
                    )
                )

            results = tracker.finish(recording_last_frame=num_frames - 1)
            writer_wait_s = writer.wait_all() if writer is not None else 0.0
            return results, profiles, writer_wait_s
        finally:
            if writer is not None:
                writer.close()


def _build_cli_parser():
    import argparse

    p = argparse.ArgumentParser(description="Stop-on-event packed active-ROI FP32 Omnipose runner.")
    p.add_argument("--pos-dir", required=True, help="Directory with img_<f:09d>_PhaseDAPI_000.tif frames")
    p.add_argument("--num-frames", type=int, required=True, help="Number of frames to process starting at 0")
    p.add_argument("--out-json", required=True, help="Output JSON path for founder results & timing")
    p.add_argument("--save-labels-dir", default=None, help="Optional directory to save zlib1 .npz labels")
    p.add_argument("--founders-json", default=None, help="Optional JSON mapping {founder_id: [y, x]}")
    p.add_argument("--min-roi-size", type=int, default=160)
    p.add_argument("--context-pad", type=int, default=16)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import tifffile

    args = _build_cli_parser().parse_args(argv)
    pos_dir = Path(args.pos_dir)
    founders = json.loads(Path(args.founders_json).read_text()) if args.founders_json else None
    if founders is not None:
        founders = {k: (float(v[0]), float(v[1])) for k, v in founders.items()}

    def read_ph(f: int) -> np.ndarray:
        return tifffile.imread(pos_dir / f"img_{f:09d}_PhaseDAPI_000.tif")

    def read_pi(f: int) -> np.ndarray | None:
        p = pos_dir / f"img_{f:09d}_mCherryDAPI_000.tif"
        return tifffile.imread(p) if p.exists() else None

    def read_yfp0() -> np.ndarray | None:
        p = pos_dir / "img_000000000_YFPDAPImirrEXP_000.tif"
        return tifffile.imread(p) if p.exists() else None

    runner = FastOmniposeRunner(min_roi_size=args.min_roi_size, context_pad=args.context_pad)
    t_load = runner.load_model()
    runner.warmup(read_ph(0))
    t0 = time.perf_counter()
    results, profiles, wait_s = runner.run_stack(
        args.num_frames,
        read_ph,
        pi_reader=read_pi,
        yfp0_reader=read_yfp0,
        founders=founders,
        save_labels_dir=args.save_labels_dir,
    )
    elapsed_s = time.perf_counter() - t0
    payload = {
        "model_load_s": round(t_load, 3),
        "elapsed_s": round(elapsed_s, 3),
        "writer_wait_s": round(wait_s, 3),
        "n_founders": len(results),
        "results": {fid: res.to_dict() for fid, res in results.items()},
        "frames": [p.__dict__ for p in profiles],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for microscopy.founder_tracker and CPU helpers in microscopy.fast_omnipose."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from microscopy.fast_omnipose import (
    AsyncNpzWriter,
    normalize_frame_quantiles,
    pack_active_rois,
    save_npz_zlib1,
    unpack_mosaic_labels,
)
from microscopy.founder_tracker import (
    FieldTracker,
    TrackerConfig,
    compute_focus_score,
    compute_notch_ratio,
    estimate_frame_shift,
    filter_phase_dark_labels,
    measure_footprint_channels,
    score_frame0_founder,
    track_field,
    track_founder,
)


def _make_bg(shape=(128, 160), val=2000.0, noise=15.0, seed=0):
    rng = np.random.default_rng(seed)
    return (val + rng.normal(0.0, noise, shape)).astype(np.float32)


def _draw_rod(labels, phase, lid, y, x0, x1, half_w=4, cell_phase=1050.0, halo=2250.0):
    labels[y - half_w : y + half_w + 1, x0 : x1 + 1] = lid
    if halo is not None:
        mask = labels == lid
        ring = ndi.binary_dilation(mask, iterations=2) & ~mask & (labels == 0)
        phase[ring] = halo
    phase[y - half_w : y + half_w + 1, x0 : x1 + 1] = cell_phase


def test_estimate_frame_shift_sign_and_gating():
    shape = (160, 160)
    im0 = _make_bg(shape, val=2000.0, noise=2.0, seed=1)
    lbl0 = np.zeros(shape, dtype=np.int32)
    _draw_rod(lbl0, im0, 1, 50, 40, 70)
    _draw_rod(lbl0, im0, 2, 100, 60, 90)
    _draw_rod(lbl0, im0, 3, 75, 95, 120)

    # Move all features by (+4, +6) in image coordinates
    im1 = np.roll(im0, shift=(4, 6), axis=(0, 1))
    dy, dx, err, ok = estimate_frame_shift(im0, im1)
    assert ok
    assert err <= 0.08
    # phase_cross_correlation(prev, curr) returns (-oy, -ox) so (y - dy, x - dx) maps prev -> curr
    assert round(dy) == -4 and round(dx) == -6

    # Uncorrelated blank frame fails error gating and resets to (0, 0)
    blank = np.full(shape, 2000.0, dtype=np.float32)
    dy_b, dx_b, _, ok_b = estimate_frame_shift(im0, blank)
    assert not ok_b
    assert (dy_b, dx_b) == (0.0, 0.0)


def test_phase_dark_filter_removes_ripples_keeps_dark_cells():
    shape = (128, 128)
    ph = _make_bg(shape, val=2000.0, noise=10.0, seed=2)
    lbl = np.zeros(shape, dtype=np.int32)

    # Real bacterium (median ~ 1050 -> ratio ~ 0.525 <= 0.85)
    lbl[30:40, 25:55] = 1
    ph[30:40, 25:55] = 1050.0

    # Background ripple hallucination (median ~ 1960 -> ratio ~ 0.98 > 0.85)
    lbl[70:85, 60:95] = 2
    ph[70:85, 60:95] = 1960.0

    # Borderline ripple whose mean is <0.85 due to a few dark pixels, but median is 0.90 > 0.85
    lbl[20:30, 80:100] = 3
    ph[20:30, 80:100] = 1800.0  # ratio 0.90
    ph[20:24, 80:88] = 600.0  # minority dark artifact pulling mean down to ~0.80

    filtered = filter_phase_dark_labels(lbl, ph, dark_ratio_max=0.85)
    assert (filtered == 1).sum() == (lbl == 1).sum()
    assert (filtered == 2).sum() == 0
    assert (filtered == 3).sum() == 0


def test_dividing_rod_with_ripple_suppression():
    T = 10
    shape = (128, 160)
    labs, phs = [], []
    for f in range(T):
        ph = _make_bg(shape, val=2000.0, noise=8.0, seed=10 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 4:
            # Grow from width 20 (f0) -> 32 (f3)
            _draw_rod(lbl, ph, 1, 60, 40, 60 + f * 4)
        else:
            # Divide at f=4 into two daughters with >=1.4x total area
            _draw_rod(lbl, ph, 1, 60, 38, 56)
            _draw_rod(lbl, ph, 2, 60, 58, 76)
        # Inject bright background ripple adjacent to dividing cell at f=2..5
        lbl[35:50, 45:65] = 9
        ph[35:50, 45:65] = 2050.0
        labs.append(lbl)
        phs.append(ph)

    res = track_founder(labs, 60.0, 50.0, phase_seq=phs)
    assert res.decision == "division"
    assert (res.lower_frame, res.upper_frame, res.event_frame) == (3, 4, 4)


def test_grazing_neighbor_and_transient_1frame_contact_vs_persistent_fusion():
    shape = (128, 160)

    # Case A: Grazing non-merged neighbor + 1-frame transient merge then separation & division
    labs_a, phs_a = [], []
    for f in range(9):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=20 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 2:
            _draw_rod(lbl, ph, 1, 60, 40, 60 + f * 3)
            _draw_rod(lbl, ph, 2, 74, 40, 60)  # nearby separate neighbor
        elif f == 2:
            # Transient 1-frame contact merge overlapping both founder and neighbor priors
            lbl[56:79, 40:64] = 1
            ph[56:79, 40:64] = 1050.0
        elif f < 5:
            # Separated again
            _draw_rod(lbl, ph, 1, 60, 40, 68)
            _draw_rod(lbl, ph, 2, 75, 40, 60)
        else:
            # Divided cleanly at f=5
            _draw_rod(lbl, ph, 1, 60, 38, 55)
            _draw_rod(lbl, ph, 3, 60, 57, 74)
            _draw_rod(lbl, ph, 2, 75, 40, 60)
        labs_a.append(lbl)
        phs_a.append(ph)

    res_a = track_founder(labs_a, 60.0, 50.0, phase_seq=phs_a)
    assert res_a.decision == "division"
    assert res_a.event_frame == 5
    assert res_a.used_transient_contact_split

    # Case B: Persistent contact merge (>=2 consecutive frames) -> lost (contact)
    labs_b, phs_b = [], []
    for f in range(6):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=30 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 2:
            _draw_rod(lbl, ph, 1, 60, 40, 60)
            _draw_rod(lbl, ph, 2, 73, 40, 60)
        else:
            lbl[56:78, 40:62] = 1
            ph[56:78, 40:62] = 1050.0
        labs_b.append(lbl)
        phs_b.append(ph)

    res_b = track_founder(labs_b, 60.0, 50.0, phase_seq=phs_b)
    assert res_b.decision == "lost"
    assert res_b.tracking_end_reason == "contact"
    assert res_b.event_frame == 3


def test_focus_qc_pause_does_not_burn_max_gap():
    shape = (128, 160)
    labs, phs = [], []
    for f in range(12):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=40 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        # Put high-contrast sharp anchors on focused frames
        _draw_rod(lbl, ph, 1, 60, 40, 60)
        _draw_rod(lbl, ph, 2, 30, 80, 110)
        _draw_rod(lbl, ph, 3, 95, 25, 55)
        if 3 <= f <= 9:
            # 7 consecutive defocused frames (> max_gap=5!) blurred smoothly
            ph = ndi.gaussian_filter(ph, sigma=4.0)
            lbl[:] = 0  # Omnipose fails during defocus
        labs.append(lbl)
        phs.append(ph)

    assert compute_focus_score(phs[5]) < 0.60 * compute_focus_score(phs[0])
    res = track_founder(labs, 60.0, 50.0, phase_seq=phs)
    assert res.contains_defocus_pause
    assert res.decision == "no_division"
    assert res.last_identity_frame == 11


def test_pi_positive_death_in_track_vs_healthy_high_baseline_decay():
    shape = (128, 160)
    # 1. Non-dividing founder turns bright PI-positive in-track at f=4,5 while still dark & segmented
    labs1, phs1, pis1 = [], [], []
    for f in range(10):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=50 + f)
        pi = _make_bg(shape, val=500.0, noise=15.0, seed=60 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        _draw_rod(lbl, ph, 1, 60, 40, 60)
        if f >= 4:
            pi[56:65, 40:61] += 450.0
        labs1.append(lbl)
        phs1.append(ph)
        pis1.append(pi)

    res_dead = track_founder(labs1, 60.0, 50.0, phase_seq=phs1, pi_seq=pis1)
    assert res_dead.decision == "pi_positive_death"
    assert res_dead.event_frame == 4

    # 2. Healthy founder starts with elevated +650 ADU PI that decays as it grows and divides
    labs2, phs2, pis2 = [], [], []
    for f in range(9):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=70 + f)
        pi = _make_bg(shape, val=500.0, noise=15.0, seed=80 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 4:
            _draw_rod(lbl, ph, 1, 60, 40, 60 + f * 4)
            pi[56:65, 40 : 61 + f * 4] += max(50.0, 650.0 - f * 150.0)
        else:
            _draw_rod(lbl, ph, 1, 60, 38, 56)
            _draw_rod(lbl, ph, 2, 60, 58, 76)
            pi[56:65, 38:57] += 80.0
            pi[56:65, 58:77] += 80.0
        labs2.append(lbl)
        phs2.append(ph)
        pis2.append(pi)

    res_healthy = track_founder(labs2, 60.0, 50.0, phase_seq=phs2, pi_seq=pis2)
    assert res_healthy.decision == "division"
    assert res_healthy.event_frame == 4


def test_ghost_lysis_anti_shove_distinction():
    shape = (128, 160)

    # Case A: True empty-footprint phase-ghost lysis at f=4
    labs_a, phs_a = [], []
    for f in range(8):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=90 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 4:
            _draw_rod(lbl, ph, 1, 60, 40, 60)
        else:
            # Cell vanishes, leaving phase-neutral/bright footprint (~1980, delta_r > +0.25) with empty surroundings
            ph[56:65, 40:61] = 1980.0
        labs_a.append(lbl)
        phs_a.append(ph)

    res_lysis = track_founder(labs_a, 60.0, 50.0, phase_seq=phs_a)
    assert res_lysis.decision == "ghost_lysis"
    assert res_lysis.event_frame == 4
    assert res_lysis.delta_r is not None and res_lysis.delta_r >= 0.25

    # Case B: Neighbor shoves cell out and sits within 5 px of old footprint -> NOT ghost_lysis!
    labs_b, phs_b = [], []
    for f in range(12):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=100 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        _draw_rod(lbl, ph, 3, 20, 20, 45)
        _draw_rod(lbl, ph, 4, 100, 100, 130)
        if f < 4:
            _draw_rod(lbl, ph, 1, 60, 40, 60)
            _draw_rod(lbl, ph, 2, 71, 40, 60)
        else:
            # Neighbor (clearly owned by om) encroaches within 10 px while footprint phase rises
            ph[56:65, 40:61] = 2050.0
            _draw_rod(lbl, ph, 2, 69, 40, 65)
        labs_b.append(lbl)
        phs_b.append(ph)

    res_shove = track_founder(labs_b, 60.0, 50.0, phase_seq=phs_b)
    assert res_shove.decision != "ghost_lysis"
    assert res_shove.decision == "lost"

    # Case C: Unsegmented cell shifts ~9 px (no label in labs_f, but raw phase-dark pixels sit inside halo)
    labs_c, phs_c = [], []
    for f in range(12):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=105 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        if f < 4:
            _draw_rod(lbl, ph, 1, 60, 40, 60)
        else:
            # Footprint is neutral background, but intact unsegmented dark rod sits 9 px above
            ph[56:65, 40:61] = 2000.0
            ph[47:54, 40:61] = 1050.0
        labs_c.append(lbl)
        phs_c.append(ph)

    res_shift = track_founder(labs_c, 60.0, 50.0, phase_seq=phs_c)
    assert res_shift.decision != "ghost_lysis"
    assert res_shift.decision == "lost"


def test_ghost_lysis_blocked_by_bright_colony_halo():
    """Bright phase-contrast halo from an encroaching colony vetoes ghost_lysis."""
    shape = (128, 160)
    labs, phs = [], []
    for f in range(8):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=140 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        # Anchor rods so phase_cross_correlation measures zero stage drift
        _draw_rod(lbl, ph, 10, 20, 20, 50)
        _draw_rod(lbl, ph, 11, 105, 100, 135)
        if f < 4:
            _draw_rod(lbl, ph, 1, 60, 40, 60)
        else:
            # No segmented cell or dark core, but bright colony phase halo (>1.15x bg)
            ph[55:65, 39:62] = 2450.0
        labs.append(lbl)
        phs.append(ph)

    res = track_founder(labs, 60.0, 50.0, phase_seq=phs)
    assert res.decision != "ghost_lysis"
    assert res.decision == "lost"


def test_frame0_doublet_debris_and_yfp_scoring():
    shape = (128, 160)
    ph = _make_bg(shape, val=2000.0, noise=5.0, seed=110)
    yfp = _make_bg(shape, val=400.0, noise=5.0, seed=111)
    lbl = np.zeros(shape, dtype=np.int32)

    # Normal single rods (population reference)
    for idx, y in enumerate([20, 40, 60, 80], start=1):
        _draw_rod(lbl, ph, idx, y, 20, 45, half_w=4)

    # Pinched dumbbell doublet (deep waist notch in middle)
    lbl[18:27, 90:106] = 5
    lbl[20:25, 106:112] = 5  # deep constriction at septum
    lbl[18:27, 112:128] = 5
    ph[lbl == 5] = 1050.0
    yfp[lbl == 5] += 250.0

    # Tiny debris particle
    lbl[105:108, 100:103] = 6
    ph[105:108, 100:103] = 1750.0

    s_norm = score_frame0_founder(lbl, ph, 1)
    s_dbl = score_frame0_founder(lbl, ph, 5, yfp0=yfp)
    s_deb = score_frame0_founder(lbl, ph, 6)

    assert not s_norm["is_doublet"]
    assert s_norm["notch_ratio"] > 0.85
    assert s_dbl["notch_ratio"] < 0.78
    assert s_dbl["is_doublet"]
    assert s_dbl["baseline_yfp_net"] == pytest.approx(250.0, abs=15.0)
    assert s_deb["is_debris"]


def test_no_division_elongation_and_growth_metrics():
    shape = (128, 160)
    labs, phs = [], []
    for f in range(6):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=120 + f)
        lbl = np.zeros(shape, dtype=np.int32)
        # Filamentous growth without septation: width 21 -> 46 (~2.2x)
        _draw_rod(lbl, ph, 1, 60, 40, 60 + f * 5)
        labs.append(lbl)
        phs.append(ph)

    res = track_founder(labs, 60.0, 50.0, phase_seq=phs)
    assert res.decision == "no_division"
    assert res.growth_ratio > 1.8
    assert res.elongation_ratio > 1.8


def test_fast_omnipose_roi_pack_unpack_and_zlib1(tmp_path):
    H, W = 512, 512
    ph = _make_bg((H, W), val=2000.0, noise=10.0, seed=130)
    bboxes = {
        "a": (50, 120, 60, 130),
        "b": (80, 150, 100, 170),  # deliberately overlaps 'a' padded region
        "c": (300, 420, 310, 450),
    }
    lo, hi, bg_med = normalize_frame_quantiles(ph)
    canvas, placements, _ = pack_active_rois(
        ph, bboxes, lo=lo, hi=hi, bg_median=bg_med, min_roi_size=128, context_pad=16
    )
    assert len(placements) == 3
    assert canvas.dtype == np.float32
    assert canvas.shape[0] % 16 == 0 and canvas.shape[1] % 16 == 0
    assert 0.0 <= float(canvas.min()) and float(canvas.max()) <= 1.0

    # Paint one distinct object at center of each packed slot
    m_canvas = np.zeros(canvas.shape, dtype=np.int32)
    for k, pl in enumerate(placements, start=1):
        cy = pl.ty0 + pl.height // 2
        cx = pl.tx0 + pl.width // 2
        m_canvas[cy - 6 : cy + 7, cx - 10 : cx + 11] = k

    unpacked = unpack_mosaic_labels(m_canvas, placements, (H, W))
    assert unpacked.shape == (H, W)
    assert len(np.unique(unpacked[unpacked > 0])) == 3

    # Verify zlib level-1 sync and async .npz round-trip
    p_sync = tmp_path / "sync.npz"
    p_async = tmp_path / "async.npz"
    save_npz_zlib1(p_sync, unpacked)
    with AsyncNpzWriter(max_workers=1) as writer:
        writer.submit(p_async, unpacked)
    with np.load(p_sync) as z1, np.load(p_async) as z2:
        assert np.array_equal(z1["labels"], unpacked)
        assert np.array_equal(z2["labels"], unpacked)


def test_pipeline_founder_v2_end_to_end(tmp_path):
    """End-to-end pipeline.run with tracking.events='founder_v2' + summarize + workbench."""
    import json
    import tifffile
    from microscopy.events import summarize
    from microscopy.importing import initialize
    from microscopy.pipeline import run
    from microscopy.review import materialize
    from microscopy.workbench.adapter import RunReviewModel

    raw = tmp_path / "raw"
    pos_dir = raw / "Pos1"
    pos_dir.mkdir(parents=True)
    lab_dir = tmp_path / "labels" / "Pos1"
    lab_dir.mkdir(parents=True)

    shape = (128, 160)
    for f in range(8):
        ph = _make_bg(shape, val=2000.0, noise=5.0, seed=200 + f)
        pi = _make_bg(shape, val=200.0, noise=5.0, seed=300 + f)
        lbl = np.zeros(shape, dtype=np.uint16)

        # Founder 1 (top-left): grows and divides at f=4
        if f < 4:
            x1 = 44 + 4 * f
            _draw_rod(lbl, ph, 1, 28, 24, x1)
        else:
            _draw_rod(lbl, ph, 1, 28, 22, 41)
            _draw_rod(lbl, ph, 4, 28, 44, 63)

        # Founder 2 (middle): stays single, surges in PI at f=3..7
        _draw_rod(lbl, ph, 2, 68, 24, 44)
        if f >= 3:
            pi[lbl == 2] += 450.0

        # Founder 3 (bottom-right): stays single, healthy, no division
        _draw_rod(lbl, ph, 3, 102, 95, 118)

        tifffile.imwrite(pos_dir / f"img_{f:09d}_PhaseDAPI_000.tif", ph.astype(np.uint16))
        tifffile.imwrite(pos_dir / f"img_{f:09d}_mCherryDAPI_000.tif", pi.astype(np.uint16))
        np.savez_compressed(lab_dir / f"{f:09d}.npz", labels=lbl)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "file_pattern": r"Pos(?P<field>\d+)/img_(?P<frame>\d+)_(?P<channel>[^/]+)_000\.tif",
                "structure_channel": "PhaseDAPI",
                "fluorescence_channels": ["mCherryDAPI"],
                "frame_interval_s": 300.0,
                "segmentation": {
                    "adapter": "precomputed",
                    "labels_dir": str(tmp_path / "labels"),
                    "min_area_px": 10,
                },
                "tracking": {
                    "adapter": "overlap",
                    "events": "founder_v2",
                    "founder_v2": {"persist": 2},
                },
                "founders": {"edge_margin_px": 3, "isolation_px": 0},
            }
        )
    )
    p = initialize(raw, cfg_path, tmp_path / "proj")

    res = run(p, fields=["1"])
    assert res["graph_scope"] == "founder_v2"
    assert len(res["founder_events"]) == 3

    data = materialize(p, res["run_id"], [])
    rows = summarize(p, data)
    by_dec = {r["decision"]: r for r in rows}
    assert set(by_dec.keys()) == {"division", "pi_positive_death", "no_division"}
    assert by_dec["division"]["upper_frame"] == 4
    assert by_dec["pi_positive_death"]["upper_frame"] == 3
    assert len(by_dec["no_division"]["observation_ids"]) == 8

    # Workbench adapter must load founder_v2 runs (including death fates) without error
    wb = RunReviewModel(p.root, res["run_id"], read_only=True)
    st = wb.state()
    assert len(st["records"]) == 3
    assert {r["tracking_end_reason"] for r in st["records"]} == {
        "division",
        "pi_positive_death",
        "recording_end",
    }


"""Bad frames, the flags added after the 2026_10_15 pilot, and the QC report."""

import json

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

from scanlag.pipeline.flags import bad_frames, colony_flags, doublet_ratios, track_flags
from scanlag.pipeline.qc import flag_kinds, gallery_frames, qc_notes, write_qc
from scanlag.store import ResultsStore
from scanlag.testset.tiles import TILE


def growing(n_frames=20, n_col=30, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(60, 400, n_frames)[:, None] * rng.uniform(0.8, 1.2, n_col)[None]
    return base.round().astype(np.int64)


def test_bad_frame_found_and_ignored_by_the_flags():
    area = growing()
    assert bad_frames(area) == []
    area[10] = (area[10] * 0.6).astype(np.int64)  # a blurred scan: every colony smaller for one frame
    assert bad_frames(area) == [10]
    merged = np.zeros(area.shape[0], bool)
    assert any(f.startswith("shrinks") for f in track_flags(area[:, 0], merged))
    assert not any(f.startswith("shrinks") for f in track_flags(area[:, 0], merged, bad=[10]))
    one = growing()
    one[10, 0] = one[10, 0] // 2  # one colony alone: a real problem, not a bad frame
    assert bad_frames(one) == []


def test_flicker():
    a = np.array([0, 20, 60, 120, 200, 400, 210, 420, 215, 430, 220, 440, 230])
    flags = track_flags(a, np.zeros(len(a), bool), merged_area_is_own=True)
    assert any(f.startswith("flicker") for f in flags)
    assert not any(f.startswith("flicker") for f in track_flags(np.linspace(20, 440, 13), np.zeros(13, bool)))


def test_transient_merge_only_for_the_seed_tracker():
    smooth = np.array([0, 0, 10, 14, 20, 28, 40, 55, 70])
    touch = np.zeros(9, bool)
    touch[5:7] = True
    assert any(f.startswith("transient merge") for f in track_flags(smooth, touch))
    assert track_flags(smooth, touch, merged_area_is_own=True) == []  # footprint: own area, harmless touch


def test_possible_doublet():
    times = np.arange(30) * 15.0  # min
    n = 15
    area = np.zeros((30, n), np.int64)
    for c in range(n):  # all appear at frame 2 and grow alike ...
        area[2:, c] = np.linspace(12, 300, 28).round()
    area[2:, 0] *= 2  # ... but colony 1 is two colonies tracked as one
    merged = np.zeros_like(area, bool)
    ratios = doublet_ratios(area, merged, times)
    assert ratios[0] > 1.9 and np.allclose(ratios[1:], 1)
    labels = np.zeros((30, 4, 4), np.uint16)
    flags = colony_flags(area, merged, np.ones_like(area), labels, times_min=times)["flags"]
    assert "possible doublet" in flags.iloc[0] and not flags.iloc[1:].str.contains("doublet").any()
    merged[5, 0] = True  # merged before the measuring age: no ratio
    assert np.isnan(doublet_ratios(area, merged, times)[0])


def test_irregular_split():
    own = np.zeros((2, 30, 30), np.uint16)
    own[1, 5:25, 5:10] = 1  # an L-shaped share of a merged blob
    own[1, 20:25, 5:25] = 1
    own[1, 5:20, 12:25] = 2
    area = np.array([[0, 0], [int((own[1] == 1).sum()), int((own[1] == 2).sum())]])
    merged = np.array([[False, False], [True, True]])
    flags = colony_flags(area, merged, np.ones((2, 2), int), own.copy(), own)["flags"]
    assert "irregular split" in flags.iloc[0] and "irregular split" not in flags.iloc[1]


def test_flag_kinds_and_gallery_frames():
    kinds = flag_kinds(pd.Series(["shrinks to 46%; jump x2.1; flicker (48 changes)", "", "possible doublet (x1.7)"]))
    assert kinds.tolist() == ["shrinks", "jump", "flicker", "possible doublet"]
    track = pd.DataFrame({"frame": range(5, 50), "merged": [False] * 10 + [True] * 5 + [False] * 30})
    assert gallery_frames(5, 15, 50, track) == [5, 9, 14, 15, 20, 49]
    assert gallery_frames(5, -1, 50, track) == [5, 9, 17, 29, 49, 49]


def test_qc_notes():
    row = dict(scanner="A", plate=1, est_colonies=100.0, n_colonies=200, tracked_vs_est=1.0, n_included=180,
               n_final_full=150, n_final_full_est=180, merge_excess=0.2, n_flagged_included=50, flagged_share=0.28,
               n_possible_doublets=9, bad_frames="162", flags="shrinks 30")
    notes = qc_notes(pd.DataFrame([row]))
    assert len(notes) == 5 and all(n.startswith("A plate 1") for n in notes)
    quiet = row | dict(tracked_vs_est=0.1, merge_excess=0.02, flagged_share=0.05, n_flagged_included=9,
                       n_possible_doublets=0, bad_frames="")
    assert qc_notes(pd.DataFrame([quiet])) == []
    resolved = row | dict(tracked_vs_est=0.1, merge_excess=0.02, n_possible_doublets=0, bad_frames="",
                          n_flags_unresolved=5)  # 50 flagged, 45 of them checked in the review
    assert qc_notes(pd.DataFrame([resolved])) == []


def tiny_plate(tmp_path, n_col=1):
    """A results folder with one 200 px plate (S, plate 1) of ``n_col`` growing discs on raw TIFF scans.
    Colony 1 is flagged "jump x2.1"; the others are unflagged. Returns (store, plate folder)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    n, size = 6, 200
    names = [f"_S_20261015_{1300 + k:04d}.tiff" for k in range(n)]
    centres = [(100, 100), (40, 40), (160, 150)][:n_col]
    own = np.zeros((n, size, size), np.uint16)
    yy, xx = np.mgrid[:size, :size]
    for k in range(n):
        scan = np.full((size, size, 3), 40, np.uint8)
        for i, (cy, cx) in enumerate(centres):
            disc = np.hypot(yy - cy, xx - cx) <= 4 + 3 * k
            scan[disc] = 150
            own[k][disc] = i + 1
        tifffile.imwrite(raw / names[k], scan)
    store = ResultsStore(tmp_path / "res", [raw])
    pdir = ("S", "plate_1")
    store.write_json({"scanner": "S", "plate": 1, "raw_dir": str(raw), "raw_scans": names,
                      "times_min": [15.0 * k for k in range(n)], "crop_xy_size": [0, 0, size],
                      "shift_xy": [[0, 0]] * n, "tracker": "footprint", "bad_frames": [3],
                      "final_count": {"n_rim": 0, "n_full": n_col, "n_full_est": n_col}}, *pdir, "plate.json")
    zarr.save_array(str(store.path(*pdir, "colonies.zarr")), own)
    zarr.save_array(str(store.path(*pdir, "labels.zarr")), own)
    Image.fromarray(np.full((size, size), 255, np.uint8)).save(store.path(*pdir, "mask.png"))
    ids = list(range(1, n_col + 1))
    pd.DataFrame({"colony_id": ids, "status": 0, "auto_status": 0, "appearance_frame": 0, "appearance_min": 0.0,
                  "first_merged_frame": -1, "final_area_px": [int((own[-1] == i).sum()) for i in ids],
                  "seed_x": [c[1] for c in centres], "seed_y": [c[0] for c in centres],
                  "n_flags": [1] + [0] * (n_col - 1), "flags": ["jump x2.1"] + [""] * (n_col - 1)}
                 ).to_parquet(store.path(*pdir, "colonies.parquet"))
    pd.DataFrame([{"frame": k, "time_min": 15.0 * k, "colony_id": i, "area_px": int((own[k] == i).sum()),
                   "x": float(centres[i - 1][1]), "y": float(centres[i - 1][0]), "blob": i, "merged": False}
                  for k in range(n) for i in ids]).to_parquet(store.path(*pdir, "tracks.parquet"))
    return store, store.root / "S" / "plate_1"


def test_write_qc_on_a_tiny_plate(tmp_path):
    store, plate = tiny_plate(tmp_path)
    pd.DataFrame({"scanner": ["S"], "plate": [1], "est_colonies": [1.0]}).to_csv(store.path("density", "density.csv"))
    index = write_qc(store.root, [plate], store, galleries=2, log=lambda m: None)
    page = index.read_text()
    assert "S_p1_outlines.png" in page and "S_p1_jump.png" in page
    assert (store.root / "qc" / "S_p1_jump.png").stat().st_size > 1000
    table = pd.read_csv(store.root / "qc" / "qc.csv")
    assert table.n_flagged_included.tolist() == [1] and table.tracked_vs_est.tolist() == [0.0]
    assert table.n_flags_unresolved.tolist() == [1]
    assert "bad frame(s) 3" in (store.root / "qc" / "notes.txt").read_text()
    assert json.loads((plate / "plate.json").read_text())["bad_frames"] == [3]


def test_resolving_flags():
    from scanlag.pipeline.flags import is_unresolved, kinds_of, unresolved_mask

    assert kinds_of("shrinks to 46%; possible doublet (x1.7)") == ["shrinks", "possible doublet"]
    assert is_unresolved("jump x2.1; gap (3 frames)", "jump")
    assert not is_unresolved("jump x3.0", "jump")  # the numbers may change with --reflag, the kind doesn't
    assert not is_unresolved("", "")
    col = pd.DataFrame({"flags": ["jump x2.1", "gap (2 frames)", ""], "flags_resolved": ["jump", "", ""]})
    assert unresolved_mask(col).tolist() == [False, True, False]
    assert unresolved_mask(col.drop(columns="flags_resolved")).tolist() == [True, True, False]


def test_review_window_resolves_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets
    from scanlag.gui.review import ReviewWindow

    store, plate = tiny_plate(tmp_path, n_col=3)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = ReviewWindow([plate])
    assert w.checks["Show excluded"].isChecked()
    assert "Exclude all" not in [b.text() for b in w.findChildren(QtWidgets.QPushButton)]
    assert w._flag_ids == [1] and w.flag_count.text().startswith("1 of 1")
    w.flag_list.item(0).setCheckState(QtCore.Qt.Checked)
    for _ in range(3):
        app.processEvents()
    col = pd.read_parquet(plate / "colonies.parquet")
    assert col.flags_resolved.tolist() == ["jump", "", ""]
    assert w.flag_count.text().startswith("0 of 1")
    w.select(1)
    assert len(w.ring) == 2  # the close and the wide ring
    w.flag_list.clearSelection()
    w.toggle_resolved()  # Ctrl+O on a resolved colony unresolves it
    assert pd.read_parquet(plate / "colonies.parquet").flags_resolved.tolist() == ["", "", ""]
    log = json.loads((plate / "review.json").read_text())["log"]
    assert [e["action"] for e in log] == ["resolve flags", "unresolve flags"]
    w.close()


def test_export_review_tiles(tmp_path):
    from scanlag.testset.gold import load_gold_set
    from scanlag.train.review_tiles import plate_review_tiles, write_tile

    store, plate = tiny_plate(tmp_path, n_col=3)
    col = pd.read_parquet(plate / "colonies.parquet")
    col.loc[col.colony_id == 2, "status"] = 2  # excluded by the reviewer
    col.loc[col.colony_id == 3, ["flags", "n_flags", "flags_resolved"]] = ["possible doublet (x1.8)", 1, "possible doublet"]
    col["flags_resolved"] = col["flags_resolved"].fillna("")
    col.to_parquet(plate / "colonies.parquet")
    tiles = plate_review_tiles(plate)
    assert sorted((t.colony_id, t.category) for t in tiles) == [(1, "track"), (2, "excluded")]  # 3 is resolved
    out = ResultsStore(store.root / "training_tiles", store.raw_dirs)
    for t in tiles:
        folder = write_tile(plate, t, out)
        assert {p.name for p in folder.iterdir()} == {"image.png", "background.png", "previous.png", "labels_init.png",
                                                      "tile.json"}
        Image.open(folder / "labels_init.png").save(folder / "labels_corrected.png")  # "corrected" unchanged
    gold = load_gold_set(out.root)  # readable as a gold set for training
    assert len(gold) == 2 and gold[0].image.shape == (TILE, TILE, 3) and gold[0].labels.shape == (TILE, TILE)
    assert (gold[0].labels > 0).any()

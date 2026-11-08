import numpy as np

from scanlag.pipeline.flags import shape_flags, track_flags
from scanlag.pipeline.plates import Plate, circle_mask, crop
from scanlag.pipeline.scans import SCAN_NAME


def test_scan_name_parsing():
    m = SCAN_NAME.match("epec-amp7_1-5.1u_20260224_2025.tiff")
    assert m["prefix"] == "epec-amp7" and m["scanner"] == "1-5.1u"
    m = SCAN_NAME.match("vapc_scan_1-1.2_20250807_1511.tif")
    assert m["prefix"] == "vapc_scan" and m["scanner"] == "1-1.2"
    assert SCAN_NAME.match("vapc_scan_1-3_20250806_1726_motions.mat") is None
    m = SCAN_NAME.match("_1-4.4.1u_20261016_1047.tiff")  # empty prefix
    assert m["prefix"] == "" and m["scanner"] == "1-4.4.1u" and m["date"] == "20261016"
    m = SCAN_NAME.match("1-4.1u_20261015_1307.TIF")  # no prefix at all
    assert m["prefix"] is None and m["scanner"] == "1-4.1u"
    assert SCAN_NAME.match("run_20261015_1307.log") is None


def test_track_flags():
    smooth = np.array([0, 0, 10, 14, 20, 28, 40, 55, 70])
    assert track_flags(smooth, np.zeros(9, bool)) == []
    gap = smooth.copy()
    gap[5] = 0
    assert any(f.startswith("gap") for f in track_flags(gap, np.zeros(9, bool)))
    shrink = smooth.copy()
    shrink[6] = 15
    assert any(f.startswith("shrinks") for f in track_flags(shrink, np.zeros(9, bool)))
    jump = smooth.copy()
    jump[7:] = [200, 210]
    assert any(f.startswith("jump") for f in track_flags(jump, np.zeros(9, bool)))
    merged = np.zeros(9, bool)
    merged[7:] = True
    assert track_flags(jump, merged) == []  # the jump is a merge that lasts, already marked as such
    flicker = np.zeros(9, bool)
    flicker[6] = True
    assert any(f.startswith("transient merge") for f in track_flags(smooth, flicker))
    fragment = np.array([0, 0, 29, 133, 144, 157, 170, 180, 190])  # split off a neighbour, merged from then on
    merged = np.zeros(9, bool)
    merged[3:] = True
    assert track_flags(fragment, merged) == ["large at appearance (29 px)"]
    assert not any(f.startswith("large") for f in track_flags(np.array([0, 20, 30, 40]), np.zeros(4, bool)))


def test_track_flags_use_split_areas():
    from scanlag.pipeline.flags import colony_flags

    a = np.array([15, 30, 60, 20, 25, 30])  # a bad split takes most of the colony in frame 3
    merged = np.array([False, False, True, True, True, True])
    assert not any(f.startswith("shrinks") for f in track_flags(a, merged))  # seed tracker: merged frames skipped
    assert any(f.startswith("shrinks") for f in track_flags(a, merged, merged_area_is_own=True))

    own = np.zeros((2, 20, 20), np.uint16)  # shape flags read the colony's own mask
    own[:, 2:18, 2:5] = 1
    labels = np.zeros_like(own)
    labels[:, 2:18, 2:18] = 1  # the detected blob is a square that the colony shares with a neighbour
    area = np.full((2, 1), 48)
    args = np.zeros((2, 1), bool), np.ones((2, 1), int), labels
    assert "elongated" in colony_flags(area, *args, own)["flags"].iloc[0]
    assert "elongated" not in colony_flags(area, *args)["flags"].iloc[0]


def test_flag_kinds():
    from scanlag.gui.review import ReviewWindow

    kinds = ReviewWindow.flag_kinds("gap (3 frames); jump x2.1; shrinks to 50%; large at appearance (29 px)")
    assert kinds == ["gap", "jump", "shrinks", "large at appearance"]
    assert ReviewWindow.flag_kinds("") == []


def test_shape_flags():
    from skimage.measure import regionprops

    disk = np.zeros((40, 40), int)
    yy, xx = np.mgrid[:40, :40]
    disk[(yy - 20) ** 2 + (xx - 20) ** 2 <= 64] = 1
    assert shape_flags(regionprops(disk)[0]) == []
    line = np.zeros((40, 40), int)
    line[18:22, 2:38] = 1
    assert any(f.startswith("elongated") for f in shape_flags(regionprops(line)[0]))


def test_crop_and_mask():
    img = np.arange(100).reshape(10, 10)
    c = crop(img, -2, 8, 4)
    assert c.shape == (4, 4) and c[0, 2] == img[8, 0] and c[0, 0] == 0
    m = circle_mask(100, Plate(1, 50.0, 50.0, 50.0), 0.8)
    assert m[50, 50] and not m[50, 5] and abs(m.sum() - np.pi * 40**2) < 200


def test_find_scanners_groups_and_names(tmp_path):
    from scanlag.pipeline.scans import find_scanners

    for d, scanner, stamps in (("T2", "1-3", ["20250806_1726", "20250806_1926"]), ("T24", "1-1.2", ["20250807_1511"])):
        (tmp_path / d / "ROI" / "sep_1").mkdir(parents=True)
        for s in stamps:
            (tmp_path / d / f"vapc_scan_{scanner}_{s}.tif").write_bytes(b"")
        (tmp_path / d / "ROI" / "sep_1" / f"vapc_scan_{scanner}_{stamps[0]}_1.tif").write_bytes(b"")  # MATLAB crop
    (tmp_path / "T2" / "run_vapc_scan20250806_1726.log").write_text("2025/08/06 17:26:30 - Starting.\n")
    scanners = {s.name: s for s in find_scanners(tmp_path)}
    assert set(scanners) == {"1-3", "1-1.2"}
    assert len(scanners["1-3"].scans) == 2 and scanners["1-3"].times_min == [0.0, 120.0]
    assert scanners["1-3"].time_zero == "first scan"  # the log starts after the first scan
    (tmp_path / "T2" / "run_vapc_scan20250806_1726.log").write_text("2025/08/06 17:16:00 - Starting.\n")
    scanners = {s.name: s for s in find_scanners(tmp_path)}
    assert scanners["1-3"].time_zero == "run_vapc_scan20250806_1726.log" and scanners["1-3"].times_min == [10.0, 130.0]


def test_select_scanners(tmp_path):
    import pytest

    from scanlag.pipeline.scans import find_scanners, select_scanners

    for name in ("epec-amp7_1-4.1u_20260224_2025.tiff", "epec-ctrl_1-5.1u_20260224_2025.tiff", "_1-6.1u_20260224_2025.tiff"):
        (tmp_path / name).write_bytes(b"")
    scanners = find_scanners(tmp_path)
    names = lambda chosen: sorted(s.name for s in chosen)
    assert names(select_scanners(scanners)) == ["1-4.1u", "1-5.1u", "1-6.1u"]
    assert names(select_scanners(scanners, prefixes=["EPEC-amp"])) == ["1-4.1u"]
    assert names(select_scanners(scanners, prefixes=["epec"])) == ["1-4.1u", "1-5.1u"]
    assert names(select_scanners(scanners, names=["1-6.1u", "1-5.1u"])) == ["1-5.1u", "1-6.1u"]
    assert names(select_scanners(scanners, prefixes=["epec"], names=["1-6.1u", "1-5.1u"])) == ["1-5.1u"]
    with pytest.raises(ValueError, match="no scanner matches"):
        select_scanners(scanners, names=["9-9"])


def test_retrack_plate(tmp_path):
    import json

    import pandas as pd
    from scipy import ndimage as ndi

    from scanlag.pipeline.run import load_plate_dirs, retrack_plate
    from scanlag.store import ResultsStore

    raw = tmp_path / "raw"
    raw.mkdir()
    store = ResultsStore(tmp_path / "res", [raw])
    yy, xx = np.indices((60, 80))
    labels = store.create_labels((6, 60, 80), "A", "plate_1", "labels.zarr")
    for k, r in enumerate(range(5, 17, 2)):  # two colonies that merge in the last frames
        lab, _ = ndi.label(((yy - 30) ** 2 + (xx - 25) ** 2 <= r * r) | ((yy - 30) ** 2 + (xx - 52) ** 2 <= r * r))
        labels[k] = lab
    store.write_mask(np.ones((60, 80), bool), "A", "plate_1", "mask.png")
    store.write_json({"scanner": "A", "plate": 1, "times_min": [0, 30, 60, 90, 120, 150], "min_area": 10},
                     "A", "plate_1", "plate.json")
    pd.DataFrame({"colony_id": [1, 2, 3]}).to_parquet(store.path("A", "plate_1", "colonies.parquet"))
    store.path("A", "plate_1", "review.json").write_text('{"visited": true, "log": []}')

    d = load_plate_dirs(store.root)[0]
    r = retrack_plate(store, d)
    assert (r["old_tracker"], r["old_colonies"], r["colonies"]) == ("seed", 3, 2)
    assert json.loads((d / "plate.json").read_text())["tracker"] == "footprint"
    assert (d / "colonies.seed.bak.parquet").exists() and (d / "review.seed.bak.json").exists()
    assert not (d / "review.json").exists() and (d / "colonies.zarr").exists()
    tracks = pd.read_parquet(d / "tracks.parquet")
    assert tracks.merged.any() and (tracks.groupby("colony_id").area_px.diff().dropna() > 0).all()

    retrack_plate(store, d, "seed")  # back to MATLAB's tracker: merged colonies report the blob
    assert not (d / "colonies.zarr").exists()
    tracks = pd.read_parquet(d / "tracks.parquet")
    last = tracks[tracks.frame == 5]
    assert last.merged.all() and last.area_px.nunique() == 1

import numpy as np
import pandas as pd
import pytest

from scanlag.analysis.conditions import FitConfig, doubling_time, parse_plates, radial_growth, read_layout


def test_parse_plates():
    assert parse_plates("1,2,4-6") == [1, 2, 4, 5, 6]
    assert parse_plates(3) == [3]
    with pytest.raises(ValueError):
        parse_plates("1,a")


def test_read_layout(tmp_path):
    path = tmp_path / "conditions.csv"
    path.write_text('condition,scanner,plates\nCTRL,A,"1,2"\nCTRL,B,1-3\nDRUG,A,4-6\n,B,4\n')
    layout = read_layout(path)
    assert len(layout) == 8  # the row without a condition is ignored
    assert list(layout[layout.scanner == "B"].plate) == [1, 2, 3]
    path.write_text("condition,scanner,plates\nCTRL,A,1-3\nDRUG,A,3\n")
    with pytest.raises(ValueError, match="A plate 3"):
        read_layout(path)


def test_radial_growth_fits_sqrt_area_until_merge():
    t = np.arange(12) * 0.25
    area = (4 + 2.0 * t) ** 2  # sqrt(area) grows 2 px^0.5 / h
    merged = np.arange(12) >= 9
    tracks = pd.DataFrame({"scanner": "A", "plate": 1, "colony_id": 1, "frame": np.arange(12), "time_h": t,
                           "area_px": np.where(merged, area * 3, area), "merged": merged})
    g = radial_growth(tracks, FitConfig(fit_first=1, fit_last=12))
    assert g.slope.iloc[0] == pytest.approx(2.0)
    assert g.n.iloc[0] == 9  # frames before the merge
    assert radial_growth(tracks, FitConfig(fit_first=2, fit_last=6)).n.iloc[0] == 5
    assert np.isnan(radial_growth(tracks, FitConfig(fit_first=8, fit_last=12)).slope.iloc[0])  # 2 points left


def test_doubling_time():
    t = np.arange(10) * 0.5
    tracks = pd.DataFrame({"colony_id": 1, "frame": np.arange(10), "time_min": t * 60,
                           "area_px": [12, 18, 25, 40, 60, 85, 120, 150, 200, 260], "merged": False})
    assert doubling_time(tracks, 20, 80).loc[1] == pytest.approx(1.5)  # 1.0 h -> 2.5 h
    assert doubling_time(tracks, 40, 150).loc[1] == pytest.approx(2.0)
    assert doubling_time(tracks.assign(merged=tracks.frame >= 4), 20, 80).isna().all()
    # with the footprint tracker's split areas the merged frames can be used
    assert doubling_time(tracks.assign(merged=tracks.frame >= 4), 20, 80, use_merged=True).loc[1] == pytest.approx(1.5)


def test_radial_growth_with_split_areas():
    t = np.arange(12) * 0.25
    area = (4 + 2.0 * t) ** 2
    tracks = pd.DataFrame({"scanner": "A", "plate": 1, "colony_id": 1, "frame": np.arange(12), "time_h": t,
                           "area_px": area, "merged": np.arange(12) >= 3})
    assert radial_growth(tracks, FitConfig(), use_merged=False).n.iloc[0] == 3
    g = radial_growth(tracks, FitConfig(), use_merged=True)
    assert g.n.iloc[0] == 12 and g.slope.iloc[0] == pytest.approx(2.0)


def _plate_dir(tmp_path, tracker):
    import json

    d = tmp_path / str(len(list(tmp_path.iterdir())))
    d.mkdir()
    info = {"scanner": "A", "plate": 1, "times_min": [0, 30, 60, 90, 120]}
    if tracker != "legacy":
        info["tracker"] = tracker
    (d / "plate.json").write_text(json.dumps(info))
    pd.DataFrame({"colony_id": [1], "status": [0], "auto_status": [0], "appearance_frame": [0],
                  "appearance_min": [0.0], "final_area_px": [100], "seed_x": [5], "seed_y": [5], "n_flags": [0],
                  "flags": [""]}).to_parquet(d / "colonies.parquet")
    pd.DataFrame({"frame": range(5), "time_min": [0.0, 30, 60, 90, 120], "colony_id": 1,
                  "area_px": [15, 25, 50, 85, 100], "x": 5.0, "y": 5.0, "blob": 1,
                  "merged": [False, False, True, True, True]}).to_parquet(d / "tracks.parquet")
    return d


def test_plate_tables_merged_growth(tmp_path):
    from scanlag.pipeline.export import plate_tables

    growth = "growth_20_80px_min"
    assert np.isnan(plate_tables(_plate_dir(tmp_path, "footprint"))[0][growth].iloc[0])  # default: exclude
    assert plate_tables(_plate_dir(tmp_path, "footprint"), "split")[0][growth].iloc[0] == 60
    with pytest.warns(UserWarning, match="seed tracker"):  # old results: the merged area is the blob's
        assert np.isnan(plate_tables(_plate_dir(tmp_path, "legacy"), "split")[0][growth].iloc[0])
    with pytest.raises(ValueError):
        plate_tables(_plate_dir(tmp_path, "seed"), "all")

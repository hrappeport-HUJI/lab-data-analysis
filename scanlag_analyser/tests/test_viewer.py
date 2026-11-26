"""scripts/scanlag_viewer.py: the interactive page built from an analysis folder."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DV = ROOT.parent / "skills" / "data-viewer"
pytestmark = pytest.mark.skipif(not (DV / "build_viewer.py").is_file(), reason="data-viewer not next to scanlag_analyser")


def load_script():
    spec = importlib.util.spec_from_file_location("scanlag_viewer", ROOT / "scripts" / "scanlag_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def results(tmp_path, extra=True):
    rng = np.random.default_rng(1)
    rows = []
    for scanner in ("s1", "s2"):
        for plate in range(1, 7):
            for c in range(10):
                rows.append({"condition": "A" if plate <= 3 else "B", "scanner": scanner, "plate": plate, "colony_id": c,
                             "appearance_h": rng.normal(7, 1), "radial_growth_um_per_h": np.nan if c == 0 else rng.normal(90, 10),
                             "doubling_time_h": rng.normal(2.5, .3), "radial_fit_r2": .98, "first_merged_min": 300.0 if c == 1 else np.nan,
                             "final_area_px": 400, "n_flags": int(c == 2), "flags": "doublet" if c == 2 else ""})
    (tmp_path / "analysis").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path / "analysis" / "all_colonies.csv", index=False)
    head = "condition,scanner,plates" + (",strain" if extra else "")
    body = ["B,s1,4-6", "B,s2,4-6", "A,s1,1-3", "A,s2,1-3"]
    strains = [",wt", ",wt", ",mut", ",mut"] if extra else [""] * 4
    (tmp_path / "conditions.csv").write_text("\n".join([head] + [b + s for b, s in zip(body, strains)]) + "\n")
    return tmp_path


def test_viewer_has_groups_views_and_cdf(tmp_path):
    mod = load_script()
    out = mod.build(results(tmp_path), tmp_path / "v.html", "cdn", None)
    sys.path.insert(0, str(DV))
    from build_viewer import read_viewer_data

    df, spec = read_viewer_data(out)
    assert len(df) == 120 and df["_id"].is_unique and df["_id"][0] == "s1 · plate 1 · colony 0"
    assert spec["filters"][:2] == ["condition", "strain"]
    assert spec["columns"]["condition"]["order"] == ["B", "A"]           # layout order
    assert set(df.loc[df.condition == "A", "strain"]) == {"mut"}
    assert [v["name"] for v in spec["views"]][:2] == ["Lag histograms (all colonies)", "Lag vs growth rate"]
    assert spec["features"]["cdf"]["column"] == "appearance_h"
    assert df["radial_growth_um_per_h"].isna().sum() == 12               # kept, shown in the band


def test_viewer_without_extra_groups_and_bad_colour(tmp_path):
    mod = load_script()
    res = results(tmp_path, extra=False)
    assert mod.build(res, tmp_path / "v.html", "cdn", "plate_id").is_file()
    with pytest.raises(SystemExit):
        mod.build(res, tmp_path / "w.html", "cdn", "strain")


def test_several_folders_in_one_page(tmp_path):
    mod = load_script()
    day1, day2 = results(tmp_path / "day1"), results(tmp_path / "day2")   # same scanner names in both
    import json
    for d in (day1, day2):
        (d / "analysis" / "settings.json").write_text(json.dumps({"fit": {"fit_first": 1, "fit_last": 12}}))
    sys.path.insert(0, str(DV))
    from build_viewer import read_viewer_data

    out = mod.build([day1, day2], tmp_path / "results.html", "cdn", "condition", "strain", "style")
    df, spec = read_viewer_data(out)
    assert len(df) == 240 and df["_id"].is_unique
    assert set(spec["columns"]["scanner"]["order"]) == {"day1/s1", "day1/s2", "day2/s1", "day2/s2"}
    assert spec["split_by"] == "strain" and spec["split_style"] == "style"
    assert spec["views"][3]["split_by"] is None                          # the by-plate view is not split
    (day2 / "analysis" / "settings.json").write_text(json.dumps({"fit": {"fit_first": 2, "fit_last": 12}}))
    with pytest.raises(SystemExit, match="different fit settings"):
        mod.build([day1, day2], tmp_path / "r2.html", "cdn", None)
    with pytest.raises(SystemExit):
        mod.build([day1], tmp_path / "r3.html", "cdn", "condition", "condition")

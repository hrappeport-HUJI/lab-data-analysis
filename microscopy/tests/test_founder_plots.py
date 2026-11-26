import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from microscopy import founder_plots as fp


@pytest.fixture
def result(tmp_path):
    rng = np.random.default_rng(0)
    founders, rows = [], []
    for c in ["A", "B", "NOYFP"]:
        for i in range(12):
            fid = f"{c}-{i}"
            divides = i < 9
            lower = 5 + i % 4
            founders.append({"id": fid, "field": 0, "condition": c, "decision": "division" if divides else "no_division",
                             "lower_frame": lower if divides else np.nan,
                             "division_midpoint_s": (lower + .5) * 420 if divides else np.nan,
                             "excluded_reason": "", "flags": "[]"})
            pi0 = 50 + 10 * i
            rows.append({"founder_id": fid, "condition": c, "measurement": "baseline_PI", "frame": 0, "time_s": 0,
                         "status": "ok", "value": pi0})
            rows.append({"founder_id": fid, "condition": c, "measurement": "baseline_YFP", "frame": 0, "time_s": 0,
                         "status": "ok", "value": 0.3 if c == "NOYFP" else 30 + i})
            for frame in range(12):
                # PI falls by 1 ADU/min for the first frames; daughters (after lower_frame) are much brighter
                value = pi0 - 7 * frame + (500 if divides and frame > lower else 0) + rng.normal(0, .01)
                rows.append({"founder_id": fid, "condition": c, "measurement": "PI_series", "frame": frame,
                             "time_s": 420 * frame, "status": "ok", "value": value})
                if frame % 4 == 0:
                    rows.append({"founder_id": fid, "condition": c, "measurement": "YFP_series", "frame": frame,
                                 "time_s": 420 * frame + 2, "status": "ok", "value": 30 + i + frame})
    pd.DataFrame(founders).to_csv(tmp_path / "founders.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp_path / "measurements.csv", index=False)
    return tmp_path


def test_metrics_follow_definitions(result):
    d = fp.load(result, rate_window_min=30, relative_rate_min_PI=60)
    f = d.founders.set_index("id")
    # 7 ADU per 7-min frame = 1 ADU/min falling, and daughters never enter the window
    assert np.allclose(f.PI_rate.dropna(), -1, atol=1e-3)
    assert np.isnan(f.loc["A-0", "relative_rate"])  # initial PI 50 is below the 60 floor
    assert f.loc["A-2", "relative_rate"] == pytest.approx(-1 / 70 * 100, rel=1e-3)
    assert np.isnan(f.loc["A-10", "lag_h"]) and f.loc["A-1", "lag_h"] == pytest.approx(6.5 * 420 / 3600)


def test_every_figure_draws_and_settings_override(result, tmp_path):
    d = fp.load(result)
    common = dict(show=False, no_yfp=["NOYFP"], exclude_outliers=True)
    for figure in fp.FIGURES:
        stats = fp.plot(d, figure, common, dict(x_scale="log"), save_to=tmp_path / figure)
        assert (tmp_path / f"{figure}.png").exists()
        if "initial_YFP" in fp.FIGURES[figure]:
            assert "NOYFP" not in set(stats.condition)
    with_non = fp.plot(d, "initial_PI_vs_lag", common, dict(include_non_dividers=True))
    assert (with_non.n_not_shown_in_stats > 0).all()
    assert fp.yfp_histograms(d, common, times_h=[0, 1.5, 40]) [-1] is None
    tables = [fp.plot(d, f, common) for f in ("initial_PI_vs_lag", "PI_rate_vs_lag")]
    pd.concat(tables)  # combining tables must work
    assert set(fp.removed_outliers(d, tables).columns) >= {"id", "figure"}
    with pytest.raises(ValueError, match="Unknown setting"):
        fp.plot(d, "initial_PI_vs_lag", common, dict(typo=1))


def test_strain_differences_and_lane_marginals(result, tmp_path):
    d = fp.load(result)
    s = dict(show=False, panels=[["A", "B", "NOYFP"]])
    t = fp.strain_differences(d, "initial_PI_vs_lag", s)
    summary = t[(t.kind == "summary") & (t.variable == "initial_PI")].set_index("condition")
    assert summary.loc["A", "n"] == 9 and summary.loc["A", "median"] == pytest.approx(90)
    # identical distributions in every condition: no difference, Holm p never below the raw p
    pairs = t[t.kind == "Mann-Whitney"]
    assert len(pairs) == 6 and (pairs.p_holm >= pairs.p - 1e-12).all() and (pairs.p > .5).all()
    assert len(t[t.kind == "Kruskal-Wallis"]) == 2
    for style in ["lanes", "overlay"]:
        fp.plot(d, "initial_PI_vs_lag", s, marginal_style=style, save_to=tmp_path / style)
        assert (tmp_path / f"{style}.png").exists()
    fp.plot(d, "initial_PI_vs_lag", s, strain_summary=False)

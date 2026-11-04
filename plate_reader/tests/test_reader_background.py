import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reader_background as rb


class FakeExperiment:
    """The parts of ReaderExperiment that reader_background uses."""

    def __init__(self, traces, traits=None, time=None):
        self.measurements = {}
        self.data = {}
        for channel, wells in traces.items():
            self.measurements[channel] = channel
            n = len(next(iter(wells.values())))
            self.data[(channel, "time")] = np.arange(n, dtype=float) / 6 if time is None else time
            for well, y in wells.items():
                self.data[(channel, well)] = np.asarray(y, dtype=float)
        first = next(iter(traces.values()))
        self.wells = np.array(list(first))
        self.wells_2d = np.array([list(first)])
        self.plate_map_dict = {"well": self.wells_2d}
        self.traits = traits or {}
        for trait, values in self.traits.items():
            self.plate_map_dict[trait] = np.array([[values.get(w) for w in first]], dtype=object)

    def well_to_trait_value(self, well, trait):
        return self.traits[trait].get(well, np.nan)


def flat(value, n=60, noise=0.0002, seed=0):
    rng = np.random.default_rng(seed)
    return value + rng.normal(0, noise, n)


def growing(start=0.09, n=60, onset=25, seed=1):
    t = np.arange(n, dtype=float)
    return flat(start, n, seed=seed) + 0.8 / (1 + np.exp(-(t - onset - 12) / 3))


@pytest.mark.parametrize("first,expected", [(0.1113, 1), (0.0851, 1), (0.0880, 0)])
def test_jump_found_in_either_direction(first, expected):
    y = flat(0.0880)
    y[0] = first
    assert rb.detect_jump(y) == expected


def test_later_spike_is_not_a_jump():
    y = np.array([0.0887, 0.0884, 0.0885, 0.0885, 0.1008, 0.1063, 0.0890, 0.0888, 0.0889, 0.0887])
    assert rb.detect_jump(y) == 0
    assert rb.well_start(y)["start"] == pytest.approx(0.0885)
    assert not rb.well_start(y)["flat"]


def test_steady_growth_at_the_start_is_not_a_jump():
    y = 0.09 + 0.002 * np.arange(20.0) ** 2
    assert rb.detect_jump(y) == 0


def test_jump_is_capped_at_three_reads():
    y = np.array([0.20, 0.15, 0.12, 0.10] + [0.09] * 20)
    assert rb.detect_jump(y) <= 3


def test_start_uses_the_median_of_five_reads_after_the_jump():
    y = np.array([0.11, 0.0878, 0.0876, 0.0878, 0.0877, 0.0880, 0.09, 0.2, 0.4, 0.5])
    start = rb.well_start(y)
    assert start["jump_reads"] == 1
    assert start["start"] == pytest.approx(0.0878)


def test_contaminated_control_is_dropped_and_clean_one_sets_the_constant():
    clean = flat(0.0890)
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": clean, "A3": growing(0.088)}})
    result = rb.compute_background(exp, control_wells=["A2", "A3"])
    table = result.controls.set_index("well")
    assert table.loc["A2", "status"] == "clean" and bool(table.loc["A2", "used"])
    assert table.loc["A3", "status"] == "contaminated" and not bool(table.loc["A3", "used"])
    assert result.constants["background"].iloc[0] == pytest.approx(clean.mean())

    corrected = rb.apply_background(exp, result)
    assert corrected == {"OD": "OD_bg"}
    assert exp.data[("OD_bg", "A1")] == pytest.approx(exp.data[("OD", "A1")] - clean.mean())
    assert np.isnan(exp.data[("OD_bg", "A3")]).all()


def test_control_constant_skips_each_controls_jump():
    control = flat(0.0890)
    control[0] = 0.12
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": control}})
    result = rb.compute_background(exp, control_wells=["A2"])
    assert result.constants["background"].iloc[0] == pytest.approx(control[1:].mean())


def test_all_controls_contaminated_falls_back_to_the_wells_start():
    sample = growing(0.0905)
    exp = FakeExperiment({"OD": {"A1": sample, "A2": growing(0.088), "A3": growing(0.087)}})
    result = rb.compute_background(exp, control_wells=["A2", "A3"])
    row = result.wells.set_index("well").loc["A1"]
    assert row["method"] == "well_start"
    assert row["background"] == pytest.approx(np.median(sample[:5]))


def test_no_control_and_cells_at_start_stops():
    exp = FakeExperiment({"OD": {"A1": growing(0.6), "A2": growing(0.3)}})
    with pytest.raises(rb.BackgroundUnresolved, match="may not be medium alone"):
        rb.compute_background(exp)


def test_manual_background_resolves_the_stop_for_the_named_wells_only():
    high, normal = growing(0.6), growing(0.09)
    exp = FakeExperiment({"OD": {"A1": high, "A2": normal}})
    result = rb.compute_background(exp, manual_background={"OD": {"A1": 0.09}})
    table = result.wells.set_index("well")
    assert table.loc["A1", "method"] == "manual" and table.loc["A1", "background"] == 0.09
    assert table.loc["A2", "method"] == "well_start"


def test_suspect_control_stops_until_decided():
    suspect = flat(0.089)
    suspect[30:] += 0.01
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089, seed=3), "A3": suspect}})
    with pytest.raises(rb.BackgroundUnresolved, match="A3"):
        rb.compute_background(exp, control_wells=["A2", "A3"])
    kept = rb.compute_background(exp, control_wells=["A2", "A3"], control_decisions={"A3": "keep"})
    dropped = rb.compute_background(exp, control_wells=["A2", "A3"], control_decisions={"A3": "drop"})
    assert kept.constants["n_controls"].iloc[0] == 2
    assert dropped.constants["n_controls"].iloc[0] == 1


def test_controls_match_by_medium_and_unmatched_medium_uses_its_start():
    lb, m9 = flat(0.0890, seed=4), flat(0.0830, seed=5)
    other = growing(0.0950)
    exp = FakeExperiment(
        {"OD": {"A1": growing(0.09), "A2": growing(0.084), "A3": other, "A4": lb, "A5": m9}},
        traits={"medium": {"A1": "LB", "A2": "M9", "A3": "TSB", "A4": "LB", "A5": "M9"}},
    )
    result = rb.compute_background(exp, control_wells=["A4", "A5"], match_traits=["medium"])
    table = result.wells.set_index("well")
    assert table.loc["A1", "background"] == pytest.approx(lb.mean())
    assert table.loc["A2", "background"] == pytest.approx(m9.mean())
    assert table.loc["A3", "method"] == "well_start"


def test_control_without_a_medium_value_stops():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089)}}, traits={"medium": {"A1": "LB"}})
    with pytest.raises(rb.BackgroundUnresolved, match="no value"):
        rb.compute_background(exp, control_wells=["A2"], match_traits=["medium"])


def test_each_channel_gets_its_own_constant_and_od_decides_contamination():
    od = {"A1": growing(), "A2": flat(0.089), "A3": growing(0.088)}
    yfp = {"A1": flat(500, noise=3) + 5 * np.arange(60.0), "A2": flat(100, noise=1), "A3": flat(100, noise=1, seed=8)}
    exp = FakeExperiment({"OD": od, "YFP": yfp})
    result = rb.compute_background(exp, control_wells=["A2", "A3"])
    constants = result.constants.set_index("measurement")["background"]
    assert constants["OD"] == pytest.approx(od["A2"].mean())
    assert constants["YFP"] == pytest.approx(yfp["A2"].mean())


def test_offset_is_per_channel_and_defaults_to_zero():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089)}})
    result = rb.compute_background(exp, control_wells=["A2"])
    rb.apply_background(exp, result, offset={"OD": 0.01})
    constant = result.constants["background"].iloc[0]
    assert exp.data[("OD_bg", "A1")] == pytest.approx(exp.data[("OD", "A1")] - constant + 0.01)

    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089)}})
    result = rb.compute_background(exp, control_wells=["A2"])
    rb.apply_background(exp, result)
    assert exp.data[("OD_bg", "A1")] == pytest.approx(exp.data[("OD", "A1")] - constant)


def test_strongly_drifting_blank_stops_until_accepted():
    drifting = 100 + 8 * np.arange(60.0)
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089)}, "LUX": {"A1": flat(5000, noise=5), "A2": drifting}})
    with pytest.raises(rb.BackgroundUnresolved, match="one constant does not describe"):
        rb.compute_background(exp, control_wells=["A2"])
    result = rb.compute_background(exp, control_wells=["A2"], accept_blank_drift=["LUX"])
    assert "FLAG blank_drift in LUX" in rb.summarize(result)


def test_well_that_falls_below_its_start_is_flagged():
    y = np.concatenate([flat(0.098, 8), flat(0.089, 40, seed=2)])
    exp = FakeExperiment({"OD": {"A1": y, "A2": growing(0.09, n=48)}})
    with pytest.raises(rb.BackgroundUnresolved, match="fall more than 0.005 OD below"):
        rb.compute_background(exp)
    result = rb.compute_background(exp, accept_own_start=["A1"])
    rb.apply_background(exp, result)
    assert "dips_below_background" in result.wells.set_index("well").loc["A1", "flags"]


def test_trailing_reads_without_time_are_dropped():
    time = np.array([0.0, 0.1, 0.2, np.nan, np.nan])
    exp = FakeExperiment({"OD": {"A1": [0.09, 0.09, 0.09, np.nan, np.nan]}}, time=time)
    assert rb.trim_trailing_empty_reads(exp) == {"OD": 2}
    assert len(exp.data[("OD", "time")]) == 3 and len(exp.data[("OD", "A1")]) == 3


def test_verify_loaded_rejects_backwards_time():
    exp = FakeExperiment({"OD": {"A1": [0.09, 0.09, 0.09]}}, time=np.array([0.0, 1.0, 0.5]))
    with pytest.raises(ValueError, match="does not increase"):
        rb.verify_loaded(exp)


def test_channel_ratio_uses_the_shorter_channel_and_blanks_low_denominators():
    exp = FakeExperiment({"OD": {"A1": [0.01, 0.1, 0.2, np.nan]}, "YFP": {"A1": [10.0, 20.0, 40.0, 80.0]}})
    rb.channel_ratio(exp, "YFP", "OD", "YFP_per_OD", min_denominator=0.05)
    out = exp.data[("YFP_per_OD", "A1")]
    assert np.isnan(out[0]) and out[1] == pytest.approx(200) and out[2] == pytest.approx(200) and np.isnan(out[3])


def test_manual_key_that_matches_nothing_raises():
    exp = FakeExperiment({"OD": {"A1": growing(0.6), "A2": growing(0.09)}}, traits={"medium": {"A1": "lb", "A2": "lb"}})
    with pytest.raises(ValueError, match="neither a well nor a medium"):
        rb.compute_background(exp, match_traits=["medium"], manual_background={"OD": {"M9": 0.09}})
    with pytest.raises(ValueError, match="Loaded channels"):
        rb.compute_background(exp, match_traits=["medium"], manual_background={"YFP": 100})
    for key in ("lb", "LB", " lb "):
        result = rb.compute_background(exp, match_traits=["medium"], manual_background={"OD": {key: 0.09}})
        assert set(result.wells["method"]) == {"manual"}


def test_per_well_manual_value_on_a_well_with_a_clean_control_raises():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.089)}})
    with pytest.raises(ValueError, match="has a clean control"):
        rb.compute_background(exp, control_wells=["A2"], manual_background={"OD": {"A1": 0.09}})


def test_start_above_others_is_flagged_when_using_the_wells_start():
    wells = {f"A{i}": growing(0.089 + i * 0.0002, seed=i) for i in range(1, 6)}
    wells["A6"] = growing(0.106, seed=9)
    exp = FakeExperiment({"OD": wells})
    with pytest.raises(rb.BackgroundUnresolved, match="start more than 0.01 OD above"):
        rb.compute_background(exp)
    table = rb.compute_background(exp, accept_own_start=True).wells.set_index("well")
    assert "start_above_others" in table.loc["A6", "flags"]
    assert "start_above_others" not in table.loc["A1", "flags"]


def test_blank_time_windows_blanks_all_wells_inside_the_window():
    exp = FakeExperiment({"OD": {"A1": flat(0.09), "A2": flat(0.09, seed=3)}})
    time = exp.data[("OD", "time")]
    assert rb.blank_time_windows(exp, [(1.0, 2.0)]) == {"OD": int(((time >= 1) & (time <= 2)).sum())}
    assert np.isnan(exp.data[("OD", "A2")][(time >= 1) & (time <= 2)]).all()
    assert np.isfinite(exp.data[("OD", "A2")][time < 1]).all()
    assert np.isfinite(time).all()


def test_slow_rise_from_the_first_read_is_not_a_jump():
    # MGR, well B9: the rise had already begun, so the first read belongs to the curve.
    y = np.array([0.0932, 0.0930, 0.0932, 0.0939, 0.0953, 0.0970, 0.0986, 0.1010, 0.1040, 0.1080])
    assert rb.detect_jump(y) == 0


def test_two_controls_that_disagree_stop_the_run():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.088), "A3": flat(0.096, seed=4)}})
    result = rb.compute_background(exp, control_wells=["A2", "A3"], raise_if_unresolved=False)
    assert result.unresolved


def test_excluded_control_does_not_set_the_constant():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.088), "A3": flat(0.096, seed=4)}})
    result = rb.compute_background(exp, control_wells=["A2", "A3"], exclude_wells=["A3"])
    assert result.wells.set_index("well").loc["A1", "background"] == pytest.approx(0.088, abs=0.0005)


def test_control_key_ignores_case_and_number_format():
    traits = {"medium": {"A1": "LB ", "A2": "lb"}, "glucose": {"A1": 1, "A2": 1.0}}
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.088)}}, traits=traits)
    result = rb.compute_background(exp, control_wells=["A2"], match_traits=["medium", "glucose"])
    assert result.wells.set_index("well").loc["A1", "method"] == "medium_control"


def test_sample_without_a_match_value_stops():
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.088)}}, traits={"medium": {"A2": "lb"}})
    result = rb.compute_background(exp, control_wells=["A2"], match_traits=["medium"], raise_if_unresolved=False)
    assert result.unresolved


def test_control_matching_no_sample_stops():
    traits = {"medium": {"A1": "lb", "A2": "M9"}}
    exp = FakeExperiment({"OD": {"A1": growing(), "A2": flat(0.088)}}, traits=traits)
    result = rb.compute_background(exp, control_wells=["A2"], match_traits=["medium"], raise_if_unresolved=False)
    assert result.unresolved


def test_control_with_too_few_reads_cannot_be_verified():
    exp = FakeExperiment({"OD": {"A1": flat(0.09, n=5, seed=2), "A2": flat(0.088, n=5)}})
    result = rb.compute_background(exp, control_wells=["A2"], raise_if_unresolved=False)
    assert result.unresolved

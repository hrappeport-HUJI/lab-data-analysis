import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "scripts"))
sys.dont_write_bytecode = True

import make_plate_map
import readthrough as rt

DESIGN = dict(red="R", green="G", reporter_trait="plasmid", in_frame="intact", autofluorescence="delta", test=["tga"],
              condition_trait="dose", replicate_trait="rep")


def table(rows):
    return pd.DataFrame(rows, columns=["well", "reporter", "role", "condition", "replicate", "red", "green", "od", "reads_in_window"])


def row(well, reporter, condition, replicate, red, green, od=0.5):
    role = {"intact": rt.ROLE_IN_FRAME, "delta": rt.ROLE_AUTOFLUORESCENCE, "tga": rt.ROLE_TEST}[reporter]
    return [well, reporter, role, condition, replicate, red, green, od, 4]


def test_readthrough_follows_the_papers_formula():
    design = rt.ReadthroughDesign(**DESIGN)
    data = table([row("A1", "delta", 0, 1, 8, 1000), row("A2", "delta", 0, 2, 12, 1000),
                  row("A3", "intact", 0, 1, 510, 1000), row("A4", "intact", 0, 2, 610, 1000),
                  row("A5", "tga", 0, 1, 15, 1000), row("A6", "tga", 0, 2, 20, 2000)])
    summary, replicates = rt.readthrough(data, design)
    # autofluorescence 10; intact R:G 0.5 and 0.6, mean 0.55; TGA R:G 0.005 and 0.005, mean 0.005
    assert summary.loc[0, "readthrough_percent"] == pytest.approx(100 * 0.005 / 0.55)
    assert replicates["readthrough_percent"].tolist() == pytest.approx([100 * 0.005 / 0.55] * 2)
    assert summary.loc[0, "n_test"] == 2 and summary.loc[0, "n_in_frame"] == 2 and summary.loc[0, "n_autofluorescence"] == 2
    # the test replicates agree, so its error comes from the intact replicates and the autofluorescence
    in_frame_se = np.std([0.5, 0.6], ddof=1) / np.sqrt(2)
    blank_se = np.std([8, 12], ddof=1) / np.sqrt(2)
    expected = np.sqrt((in_frame_se / 0.55) ** 2 + (blank_se / 1500 / 0.005) ** 2 + (blank_se / 1000 / 0.55) ** 2)
    assert summary.loc[0, "se_percent"] == pytest.approx(summary.loc[0, "readthrough_percent"] * expected)


def test_autofluorescence_is_taken_per_condition():
    design = rt.ReadthroughDesign(**DESIGN)
    data = table([row("A1", "delta", 0, 1, 10, 1000), row("B1", "delta", 5, 1, 50, 1000),
                  row("A2", "intact", 0, 1, 510, 1000), row("B2", "intact", 5, 1, 550, 1000),
                  row("A3", "tga", 0, 1, 20, 1000), row("B3", "tga", 5, 1, 60, 1000)])
    summary, _ = rt.readthrough(data, design)
    assert summary["readthrough_percent"].tolist() == pytest.approx([2.0, 2.0])


def test_one_replicate_gives_a_value_but_no_error():
    design = rt.ReadthroughDesign(**DESIGN)
    data = table([row("A1", "delta", 0, 1, 10, 1000), row("A2", "intact", 0, 1, 510, 1000), row("A3", "tga", 0, 1, 20, 1000)])
    summary, _ = rt.readthrough(data, design)
    assert summary.loc[0, "readthrough_percent"] == pytest.approx(2.0)
    assert np.isnan(summary.loc[0, "se_percent"]) and np.isnan(summary.loc[0, "sd_replicates_percent"])


@pytest.mark.parametrize("change", [dict(green=0), dict(od=0.05)])
def test_wells_without_green_or_growth_get_no_ratio(change):
    design = rt.ReadthroughDesign(**DESIGN)
    bad = row("A3", "tga", 0, 1, 20, change.get("green", 1000), change.get("od", 0.5))
    data = table([row("A1", "delta", 0, 1, 10, 1000), row("A2", "intact", 0, 1, 510, 1000), bad])
    summary, replicates = rt.readthrough(data, design)
    assert np.isnan(summary.loc[0, "readthrough_percent"]) and replicates.empty


def test_window_mean_ignores_missing_reads_and_channels_of_different_length():
    assert rt.window_mean([0, 1, 2, 3], [1, np.nan, 3], 1.5, 1) == (3.0, 1)
    assert rt.window_mean([0, 1], [1, 2], 9, 1) == (pytest.approx(np.nan, nan_ok=True), 0)


def write_plate(path, percent=1.0, reader_hours=24):
    """A Tecan rows export with known values: three reporters, two replicates, three doses, medium controls."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    sheet.append(["Application: Tecan i-control"])
    sheet.append(["Device: infinite 200Pro", None, None, None, "Serial number: 2503006744"])
    sheet.append(["Plate", None, None, None, "Greiner 96 Flat Bottom"])
    sheet.append(["Start Time:", "11/2/2026 5:16:10 PM"])
    sheet.append([])
    columns = {1: "tga", 2: "tga", 3: "intact", 4: "intact", 5: "delta", 6: "delta", 7: "medium"}
    doses = {"A": 1.0, "B": 2.0, "C": 3.0}
    wells = [f"{r}{c}" for r in doses for c in columns]
    hours = np.arange(0, reader_hours + 0.01, 0.25)
    blanks = {"OD": 0.09, "GFP": 1500.0, "mScarlet": 70.0}

    def signal(channel, well):
        kind, dose = columns[int(well[1:])], doses[well[0]]
        growth = 1 / (1 + np.exp(-(hours - 6)))
        if kind == "medium":
            value = 0 * hours
        elif channel == "OD":
            value = 0.6 * growth
        elif channel == "GFP":
            value = 2000 * dose * growth
        elif kind == "delta":
            value = 8 * growth
        else:
            ratio = 0.3 * (percent / 100 if kind == "tga" else 1)
            value = 8 * growth + ratio * 2000 * dose * growth
        return value + blanks[channel]

    for channel in blanks:
        sheet.append([channel])
        sheet.append(["Cycle Nr.", *range(1, len(hours) + 1)])
        sheet.append(["Time [s]", *(hours * 3600).tolist()])
        sheet.append(["Temp. [°C]", *[37.0] * len(hours)])
        for well in wells:
            sheet.append([well, *signal(channel, well).tolist()])
        sheet.append([])
    sheet.append(["End Time:", "11/3/2026 5:16:10 PM"])
    workbook.create_sheet("Sheet1")
    workbook.save(path)
    layout = {"plate": {"rows": 3, "cols": 7}, "well_status": {"medium_control": ["A7:C7"]},
              "traits": {"plasmid": {"tga": ["A1:C2"], "intact": ["A3:C4"], "delta": ["A5:C6"]},
                         "rep": {"1": ["A1:C1", "A3:C3", "A5:C5"], "2": ["A2:C2", "A4:C4", "A6:C6"]},
                         "dose": {"1": ["A1:A7"], "2": ["B1:B7"], "3": ["C1:C7"]}},
              "match_traits": ["dose"]}
    rows, cols, blocks = make_plate_map.build_blocks(layout)
    make_plate_map.write_plate_map(path.parent / "map.xlsx", rows, cols, blocks)
    return ["A7", "B7", "C7"]


def load(tmp_path, percent):
    import reader_background as rb
    import reader_functions as rf

    controls = write_plate(tmp_path / "run.xlsx", percent)
    exp = rf.ReaderExperiment(str(tmp_path / "run.xlsx"), str(tmp_path / "map.xlsx"), {"OD": "OD", "GFP": "GFP", "mScarlet": "mScarlet"},
                              reader="plex", sheet_name="Sheet2")
    background = rb.compute_background(exp, control_wells=controls, match_traits=["dose"], od_measurement="OD")
    corrected = rb.apply_background(exp, background)
    design = rt.ReadthroughDesign(red="mScarlet", green="GFP", od="OD", reporter_trait="plasmid", in_frame="intact", autofluorescence="delta",
                                  test=["tga"], condition_trait="dose", replicate_trait="rep")
    samples = [w for w in exp.wells if w not in controls]
    return exp, background, corrected, design, samples, controls


def test_known_readthrough_comes_back_through_the_whole_pipeline(tmp_path):
    exp, _, corrected, design, samples, _ = load(tmp_path, 1.0)
    summary, _ = rt.readthrough(rt.reporter_table(exp, corrected, design, samples, 20, 0.25), design)
    assert summary["readthrough_percent"].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def run_checks(tmp_path, percent):
    exp, background, corrected, design, samples, controls = load(tmp_path, percent)
    return rt.sanity_checks(exp, corrected, background, design, samples, 20, control_wells=controls,
                            measurements={"OD": "OD", "GFP": "GFP", "mScarlet": "mScarlet"},
                            reader_file=str(tmp_path / "run.xlsx"), sheet="Sheet2", stability_times=[16, 20, 23]).set_index("check")


def test_a_healthy_plate_passes_the_sanity_tests(tmp_path):
    checks = run_checks(tmp_path, 1.0)
    assert "FAIL" not in checks["status"].values, checks.to_string()
    assert checks.loc["Pipeline matches the raw sheet", "status"] == "PASS"
    assert checks.loc["Readthrough in the expected range", "status"] == "PASS"
    assert checks.loc["Design complete", "status"] == "WARN"  # two replicates


def test_a_test_reporter_as_red_as_the_intact_one_fails_the_sanity_tests(tmp_path):
    checks = run_checks(tmp_path, 100.0)
    assert checks.loc["Readthrough in the expected range", "status"] == "FAIL"
    assert "nearly as red" in checks.loc["Readthrough in the expected range", "detail"]


def test_raw_reader_handles_tecan_rows(tmp_path):
    write_plate(tmp_path / "run.xlsx")
    time, values = rt.raw_well_series(tmp_path / "run.xlsx", "Sheet2", "GFP", "A7")
    assert time[1] == pytest.approx(0.25) and values[0] == pytest.approx(1500)
    assert rt.raw_well_series(tmp_path / "run.xlsx", "Sheet2", "CFP", "A7") is None


def config_for(**changes):
    analysis = {"type": "readthrough", "name": "rt", "red": "mScarlet", "green": "GFP", "od": "OD", "reporter_trait": "plasmid", "in_frame": "intact",
                "autofluorescence": "delta", "test": ["tga"], "condition_trait": "dose", "replicate_trait": "rep", "time_h": 20}
    analysis.update(changes)
    return {"title": "T", "purpose": "P", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
            "measurements": {"OD": "OD", "GFP": "GFP", "mScarlet": "mScarlet"}, "layout_description": "L",
            "background": {"control_wells": ["A7", "B7", "C7"], "match_traits": ["dose"]},
            "plots": {"group_traits": ["plasmid", "dose"], "bio_rep_trait": "rep", "color_trait": "dose"},
            "analyses": [{"type": "curves", "name": "od", "channel": "OD", "facet_trait": "plasmid"}, analysis]}


def run_build(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.run([sys.executable, "-B", str(CODE_DIR / "scripts" / "build_notebook.py"), str(path)], capture_output=True, text=True)


@pytest.mark.parametrize("changes,message", [
    ({"test": ["taa"]}, "no wells have"),
    ({"in_frame": "wt"}, "no wells have"),
    ({"red": "CFP"}, "not a loaded channel"),
    ({"replicate_trait": "clone"}, "not in the plate map"),
    ({"time_h": 99}, "outside the"),
    ({"tim_h": 20}, "unknown option"),
    ({"expected_percent": [5, 1]}, "[low, high]"),
])
def test_build_refuses_a_bad_readthrough_analysis(tmp_path, changes, message):
    write_plate(tmp_path / "run.xlsx")
    result = run_build(tmp_path / "experiment_config.json", config_for(**changes))
    assert result.returncode != 0 and message in result.stdout, result.stdout + result.stderr


def test_generated_readthrough_notebook_runs_and_gets_the_known_answer(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    write_plate(tmp_path / "run.xlsx", 2.0)
    result = run_build(tmp_path / "experiment_config.json", config_for())
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = nbformat.read(tmp_path / f"{tmp_path.name}_analysis.ipynb", as_version=4)
    nbclient.NotebookClient(notebook, timeout=170, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    summary = pd.read_csv(tmp_path / "rt.csv")
    assert summary["readthrough_percent"].tolist() == pytest.approx([2.0, 2.0, 2.0], abs=1e-6)
    checks = pd.read_csv(tmp_path / "rt_sanity.csv")
    assert "FAIL" not in checks["status"].values, checks.to_string()
    assert (tmp_path / "rt_time_course.csv").exists() and (tmp_path / "figures" / "rt.png").exists()

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = CODE_DIR / "scripts"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True

import build_notebook
import inspect_reader
import make_plate_map
import reader_functions as rf


def tecan_header(sheet, serial, labels):
    sheet.append(["Application: Tecan i-control", None, None, None, "Tecan i-control , 2.0.10.0"])
    sheet.append(["Device: infinite 200Pro", None, None, None, f"Serial number: {serial}", None, None, None, "Serial number: "])
    sheet.append(["Plate", None, None, None, "Greiner 96 Flat Bottom"])
    sheet.append(["List of actions in this measurement script:"])
    sheet.append(["Kinetic"])
    sheet.append(["Absorbance"])
    sheet.append([])
    if len(labels) == 1:
        sheet.append([f"Label: {labels[0]}"])
    else:
        sheet.append([f"{len(labels)} Labels"])
        sheet.append(["Mode", None, None, None, "Absorbance"])
        sheet.append(["Measurement Wavelength", None, None, None, 630, "nm"])
        sheet.append(["Mode", None, None, None, "Fluorescence Top Reading"])
        sheet.append(["Excitation Wavelength", None, None, None, 485, "nm"])
    sheet.append(["Start Time:", "8/25/2026 2:06:47 PM"])
    sheet.append([])


def make_tecan(path, parser, wells=("A1", "A2", "B1", "B2")):
    serial = "1506006237" if parser == "old" else "2503006744"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    labels = ["OD", "YFP"]
    tecan_header(sheet, serial, labels)
    for label, base in zip(labels, (0.1, 500)):
        sheet.append([label])
        if parser == "old":
            sheet.append(["Cycle Nr.", "Time [s]", "Temp. [°C]", *wells])
            for k in range(8):
                sheet.append([k + 1, 600.0 * k, 37.0, *[base * (1 + 0.01 * k + 0.1 * j) for j in range(len(wells))]])
        else:
            sheet.append(["Cycle Nr.", *range(1, 9)])
            sheet.append(["Time [s]", *[600.0 * k for k in range(8)]])
            sheet.append(["Temp. [°C]", *[37.0] * 8])
            for j, well in enumerate(wells):
                sheet.append([well, *[base * (1 + 0.01 * k + 0.1 * j) for k in range(8)]])
        sheet.append([])
    sheet.append(["End Time:", "8/25/2026 4:00:00 PM"])
    workbook.create_sheet("Sheet1")
    workbook.save(path)


LAYOUT = {
    "plate": {"rows": 2, "cols": 2},
    "well_status": {"medium_control": ["B2"]},
    "traits": {"strain": {"KLY": ["A1", "A2", "B1"]}, "medium": {"LB": ["A1:B2"]}, "dilution": {"1000": ["A1", "B1"], "0.5": ["A2"]}},
    "match_traits": ["medium"],
}


@pytest.mark.parametrize("parser,name", [("old", "Monochromator"), ("plex", "Plex")])
def test_inspect_identifies_reader_labels_and_settings(tmp_path, parser, name):
    path = tmp_path / "run.xlsx"
    make_tecan(path, parser)
    report = inspect_reader.inspect_file(path)
    sheet = report["sheets"][0]
    assert sheet["reader_name"] == name and sheet["parser"] == parser
    assert [b["library_label"] for b in sheet["blocks"]] == ["OD", "YFP"]
    assert sheet["blocks"][0]["suggested_channel"] == "OD"
    assert sheet["blocks"][0]["mode"] == "Absorbance"
    assert sheet["blocks"][0]["reads"] == 8
    assert sheet["blocks"][0]["median_interval_min"] == 10
    assert report["sheets"][1]["format"] == "empty"


def test_inspect_reports_unknown_serial_and_overflow(tmp_path):
    path = tmp_path / "run.xlsx"
    make_tecan(path, "old")
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["Sheet2"]
    for row in sheet.iter_rows():
        if row[0].value == "Device: infinite 200Pro":
            row[4].value = "Serial number: 999"
        if row[0].value == 3:
            row[3].value = "OVER"
    workbook.save(path)
    info = inspect_reader.inspect_file(path)["sheets"][0]
    assert info["reader_name"] == "UNKNOWN READER" and "not in the known readers" in info["warning"]
    assert info["blocks"][0]["non_numeric_values"] == {"OVER": 1}


def test_plate_map_round_trips_through_the_library(tmp_path):
    rows, cols, blocks = make_plate_map.build_blocks(LAYOUT)
    path = tmp_path / "map.xlsx"
    make_plate_map.write_plate_map(path, rows, cols, blocks)
    read = rf.ReaderExperiment.get_plate_map_dict(path)
    assert list(read) == ["well", "well_status", "strain", "medium", "dilution"]
    assert read["well"].tolist() == [["A1", "A2"], ["B1", "B2"]]
    assert read["well_status"][1, 1] == "medium_control"
    assert read["dilution"][0, 0] == 1000 and read["dilution"][0, 1] == 0.5
    assert read["medium"][1, 1] == "LB"


@pytest.mark.parametrize("change,message", [
    (lambda d: d["traits"]["strain"].update(TBL8=["A1"]), "assigned both"),
    (lambda d: d["traits"]["strain"].update(KLY=["A1", "A2"]), "without a value"),
    (lambda d: d["traits"]["medium"].update(LB=["A1:B1", "Z9"]), "outside"),
    (lambda d: d["well_status"].update(empty=["A1"]), "marked empty"),
    (lambda d: d["traits"]["medium"].update(LB=["A1:B1"]), "without a value"),
])
def test_plate_map_rejects_contradictions(change, message):
    layout = json.loads(json.dumps(LAYOUT))
    change(layout)
    with pytest.raises(make_plate_map.LayoutError, match=message):
        make_plate_map.build_blocks(layout)


def build_config(tmp_path, **changes):
    make_tecan(tmp_path / "run.xlsx", "old")
    rows, cols, blocks = make_plate_map.build_blocks(LAYOUT)
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    config = {
        "title": "Test", "purpose": "Test.", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "Test.",
        "background": {"control_wells": ["B2"], "match_traits": ["medium"]},
        "plots": {"group_traits": ["strain"]},
        "analyses": [{"type": "threshold_time", "name": "t", "channel": "OD", "threshold": 0.05}],
    }
    config.update(changes)
    path = tmp_path / "experiment_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def run_build(path):
    return subprocess.run([sys.executable, "-B", str(SCRIPTS / "build_notebook.py"), str(path)], capture_output=True, text=True)


def test_build_writes_a_valid_notebook(tmp_path):
    result = run_build(build_config(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    notebook = json.loads((tmp_path / f"{tmp_path.name}_analysis.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "READER_PARSER = 'old'" in source and "CONTROL_WELLS = ['B2']" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")


@pytest.mark.parametrize("changes,message", [
    ({"sheet": "Sheet9"}, "not in the reader file"),
    ({"sheet": "Sheet1"}, "holds no reader data"),
    ({"measurements": {"OD": "OD", "YFP": "GFP"}}, "not in sheet"),
    ({"measurements": {"YFP": "YFP"}}, "does not load"),
    ({"reader_parser": "plex"}, "needs 'old'"),
    ({"background": {"control_wells": ["A1"], "match_traits": ["medium"]}}, "plate map marks medium controls"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["media"]}}, "not in the plate map"),
    ({"plots": {"group_traits": ["strains"]}}, "not in the plate map"),
    ({"analyses": [{"type": "threshold_time", "name": "t", "channel": "OD"}]}, "needs a threshold"),
    ({"analyses": [{"type": "unknown", "name": "t"}]}, "Unknown analysis"),
    ({"analyses": [{"type": "curves", "name": "bad name", "channel": "OD"}]}, "letters, digits"),
    ({"backgroud": {}}, "unknown option"),
    ({"background": {"match_traits": ["medium"]}}, "must state control_wells"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["medium"], "ofset": {}}}, "unknown option"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["medium"], "offset": {"GFP": 0.1}}}, "names channel"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["medium"], "exclude_wells": ["Z9"]}}, "not a loaded well"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["medium"], "control_decisions": {"B2": "toss"}}}, "'keep' or 'drop'"),
    ({"background": {"control_wells": ["B2"], "match_traits": ["medium"], "accept_own_start": ["Z9"]}}, "not a loaded well"),
    ({"plots": {"group_trait": ["strain"]}}, "unknown option"),
    ({"plots": {}}, "needs group_traits"),
    ({"plots": {"group_traits": ["strain"], "annotate_traits": ["nope"]}}, "not in the plate map"),
    ({"analyses": [{"type": "threshold_time", "name": "t", "channel": "OD", "threshold": 0.05, "direction": "sideways"}]}, "'up' or 'down'"),
    ({"analyses": [{"type": "threshold_time", "name": "t", "channel": "OD", "threshold": 0.05, "treshold": 1}]}, "unknown option"),
    ({"analyses": [{"type": "curves", "name": "c", "channel": "OD", "filter": {"strain": "TBL8"}}]}, "no wells with"),
    ({"analyses": [{"type": "curves", "name": "c", "channel": "OD", "yscale": "symlog"}]}, "'linear' or 'log'"),
    ({"analyses": [{"type": "max_growth_rate", "name": "m", "channel": "OD", "t_start": 1, "t_span": [0, 5]}]}, "not both"),
    ({"analyses": [{"type": "curves", "name": "c", "channel": "r"}, {"type": "ratio", "name": "r", "numerator": "YFP", "denominator": "OD", "min_denominator": 0.1}]}, "must come before"),
    ({"analyses": [{"type": "ratio", "name": "OD_bg", "numerator": "YFP", "denominator": "OD", "min_denominator": 0.1}]}, "would overwrite"),
    ({"analyses": [{"type": "ratio", "name": "r", "numerator": "GFP", "denominator": "OD", "min_denominator": 0.1}]}, "numerator"),
    ({"analyses": [{"type": "ratio", "name": "r", "numerator": "YFP", "denominator": "OD", "min_denominator": 0.1}, {"type": "fraction_of_start", "name": "f", "channel": "r", "fraction": 0.5}]}, "must be a loaded channel"),
    ({"background": {"control_wells": []}, "analyses": [{"type": "fraction_of_start", "name": "f", "channel": "OD", "fraction": 0.5}]}, "plate map marks medium controls"),
])
def test_build_refuses_bad_configs(tmp_path, changes, message):
    result = run_build(build_config(tmp_path, **changes))
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not list(tmp_path.glob("*.ipynb"))


def test_build_refuses_a_plate_map_the_library_cannot_read(tmp_path):
    path = build_config(tmp_path)
    workbook = openpyxl.load_workbook(tmp_path / "map.xlsx")
    sheet = workbook["Sheet1"]
    sheet.cell(row=sheet.max_row + 1, column=1, value="X")
    workbook.save(tmp_path / "map.xlsx")
    result = run_build(path)
    assert result.returncode != 0 and "stray" in result.stdout


def test_build_reads_plate_map_blocks_that_start_with_a_blank_row(tmp_path):
    layout = {"plate": {"rows": 3, "cols": 2},
              "well_status": {"empty": ["A1:A2"], "medium_control": ["C2"]},
              "traits": {"strain": {"KLY": ["B1:C1", "B2"]}, "medium": {"LB": ["B1:C2"]}},
              "match_traits": ["medium"]}
    rows, cols, blocks = make_plate_map.build_blocks(layout)
    path = tmp_path / "map.xlsx"
    make_plate_map.write_plate_map(path, rows, cols, blocks)
    read, error = build_notebook.read_plate_map(path)
    assert error is None and list(read) == ["well", "well_status", "strain", "medium"]
    assert build_notebook.map_well_values(read, "well_status")["C2"] == "medium_control"
    assert build_notebook.map_well_values(read, "strain")["B2"] == "KLY"


def test_build_refuses_to_overwrite_a_notebook(tmp_path):
    path = build_config(tmp_path)
    assert run_build(path).returncode == 0
    second = run_build(path)
    assert second.returncode != 0 and "already exists" in second.stdout + second.stderr


def test_inspect_reports_a_gap_and_a_partial_last_read(tmp_path):
    path = tmp_path / "run.xlsx"
    make_tecan(path, "old")
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["Sheet2"]
    rows = [row for row in sheet.iter_rows() if row[0].value == 8][:1]
    rows[0][3].value = None
    for row in sheet.iter_rows():
        if row[0].value == 5 and row[1].value == 2400.0:
            row[1].value = 3000.0
    workbook.save(path)
    report = inspect_reader.inspect_file(path)
    block = report["sheets"][0]["blocks"][0]
    assert [g["minutes"] for g in block["interval_gaps"]]
    assert block["partial_reads"][0]["read"] == 8
    text = inspect_reader.format_report(report, grids=True)
    assert "Was the plate taken out" in text and "stopped during this read" in text and "Start (median" in text


def test_preview_reports_a_decision_and_saves_the_table(tmp_path):
    path = build_config(tmp_path, background={"control_wells": ["B2"], "match_traits": ["medium"]})
    env = dict(__import__("os").environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "preview_background.py"), str(path)],
                            capture_output=True, text=True, env=env)
    assert (tmp_path / "background_table.csv").exists(), result.stdout + result.stderr
    assert result.returncode in (0, 2), result.stdout + result.stderr
    assert "Medium controls" in result.stdout


def test_fraction_of_start_needs_a_control_or_a_manual_blank(tmp_path):
    layout = {**LAYOUT, "well_status": {}, "match_traits": []}
    layout["traits"] = {"strain": {"KLY": ["A1:B2"]}}
    make_tecan(tmp_path / "run.xlsx", "old")
    rows, cols, blocks = make_plate_map.build_blocks(layout)
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    config = {
        "title": "T", "purpose": "P", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "L", "background": {"control_wells": []},
        "plots": {"group_traits": ["strain"]},
        "analyses": [{"type": "fraction_of_start", "name": "f", "channel": "OD", "fraction": 0.5}],
    }
    path = tmp_path / "experiment_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_build(path)
    assert result.returncode != 0 and "needs medium controls or a manual blank" in result.stdout


def test_build_refuses_an_unknown_serial(tmp_path):
    path = build_config(tmp_path)
    workbook = openpyxl.load_workbook(tmp_path / "run.xlsx")
    workbook["Sheet2"]["E2"] = "Serial number: 999"
    workbook.save(tmp_path / "run.xlsx")
    result = run_build(path)
    assert result.returncode != 0 and "unknown serial" in result.stdout


def test_build_accepts_a_label_written_with_its_prefix(tmp_path):
    path = build_config(tmp_path, measurements={"OD": "Label: OD", "YFP": "YFP"})
    result = run_build(path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_build_refuses_cells_to_the_right_of_the_plate(tmp_path):
    path = build_config(tmp_path)
    workbook = openpyxl.load_workbook(tmp_path / "map.xlsx")
    sheet = workbook["Sheet1"]
    strain_row = next(r for r in range(1, sheet.max_row + 1) if sheet.cell(r, 1).value == "strain")
    sheet.cell(strain_row + 1, 4).value = "stray"
    workbook.save(tmp_path / "map.xlsx")
    result = run_build(path)
    assert result.returncode != 0 and "right of the plate" in result.stdout


def test_build_warns_when_the_map_hints_at_controls_but_none_are_listed(tmp_path):
    layout = {"plate": {"rows": 2, "cols": 2}, "traits": {"strain": {"KLY": ["A1", "A2", "B1"], "blank": ["B2"]}}}
    make_tecan(tmp_path / "run.xlsx", "old")
    rows, cols, blocks = make_plate_map.build_blocks(layout)
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    config = {
        "title": "T", "purpose": "P", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "L", "background": {"control_wells": []},
        "plots": {"group_traits": ["strain"]}, "analyses": [],
    }
    problems, facts = build_notebook.check_config(config, tmp_path)
    assert not problems
    assert any("lists no medium controls" in warning for warning in facts["warnings"])


def test_inspect_keeps_the_action_list_past_shaking_rows(tmp_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    sheet.append(["Application: Tecan i-control"])
    sheet.append(["Device: infinite 200Pro", None, None, None, "Serial number: 1506006237"])
    sheet.append(["List of actions in this measurement script:"])
    sheet.append(["Kinetic"])
    sheet.append(["Shaking (Orbital)"])
    sheet.append(["Wait (Time)"])
    sheet.append(["Absorbance"])
    sheet.append([])
    sheet.append(["Label: OD"])
    sheet.append(["Measurement Wavelength", None, None, None, 630, "nm"])
    sheet.append(["Start Time:", "8/25/2026 2:06:47 PM"])
    sheet.append([])
    sheet.append(["OD"])
    sheet.append(["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1"])
    for k in range(6):
        sheet.append([k + 1, 600.0 * k, 37.0, 0.09])
    workbook.save(tmp_path / "run.xlsx")
    info = inspect_reader.inspect_file(tmp_path / "run.xlsx")["sheets"][0]
    assert "Absorbance" in info["actions"]
    assert info["blocks"][0]["mode"] == "Absorbance"


def docstring_layout():
    text = make_plate_map.__doc__
    return json.loads(text[text.index("{"):text.index("Number-like")])


def test_the_documented_layout_example_is_valid(tmp_path):
    rows, cols, blocks = make_plate_map.build_blocks(docstring_layout())
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    read = rf.ReaderExperiment.get_plate_map_dict(tmp_path / "map.xlsx")
    assert read["well_status"][1, 10] == "medium_control"


def test_preview_stops_when_the_library_loses_wells(tmp_path):
    make_tecan(tmp_path / "run.xlsx", "plex")
    rows, cols, blocks = make_plate_map.build_blocks(LAYOUT)
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    workbook = openpyxl.load_workbook(tmp_path / "run.xlsx")
    sheet = workbook["Sheet2"]
    for row in sheet.iter_rows():
        if row[0].value == "A2":
            row[2].value = None
    workbook.save(tmp_path / "run.xlsx")
    config = {
        "title": "T", "purpose": "P", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "L",
        "background": {"control_wells": ["B2"], "match_traits": ["medium"]},
    }
    path = tmp_path / "experiment_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    env = dict(__import__("os").environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-B", str(SCRIPTS / "preview_background.py"), str(path)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "the library loaded" in result.stdout


def make_full_plate(path):
    """Tecan columns export of the documented layout: LB wells grow, M9 wells lyse, and every OD first read jumps."""
    wells = [f"{row}{col}" for row in "BCDEFG" for col in range(2, 12)]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    tecan_header(sheet, "1506006237", ["OD", "YFP"])
    hours = np.arange(72) / 6
    rng = np.random.default_rng(0)

    def od(well):
        blank = 0.088 if well[0] in "BCD" else 0.0895
        if well[1:] == "11":
            y = blank + 0 * hours
        elif well[0] in "BCD":
            y = blank + 0.8 / (1 + np.exp(-(hours - 5) / 0.5))
        else:
            y = blank + 0.5 * np.exp(-hours / 4)
        y = y + rng.normal(0, 0.0001, len(y))
        y[0] += 0.02
        return y

    def yfp(well):
        if well[1:] == "11":
            return 100 + rng.normal(0, 0.5, len(hours))
        return 100 + 2000 * (od(well) - 0.088) + rng.normal(0, 0.5, len(hours))

    for label, signal in (("OD", od), ("YFP", yfp)):
        sheet.append([label])
        sheet.append(["Cycle Nr.", "Time [s]", "Temp. [°C]", *wells])
        columns = [signal(well) for well in wells]
        for k, hour in enumerate(hours):
            sheet.append([k + 1, 3600.0 * hour, 37.0, *[float(column[k]) for column in columns]])
        sheet.append([])
    sheet.append(["End Time:", "8/25/2026 4:00:00 PM"])
    workbook.create_sheet("Sheet1")
    workbook.save(path)


def test_generated_notebook_runs_and_gets_the_known_answers(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("seaborn")
    make_full_plate(tmp_path / "run.xlsx")
    rows, cols, blocks = make_plate_map.build_blocks(docstring_layout())
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    grow = {"medium": "LB"}
    lyse = {"medium": "M9"}
    config = {
        "title": "Full plate", "purpose": "Test.", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "See the documented example.",
        "background": {"control_wells": [f"{row}11" for row in "BCDEFG"], "match_traits": ["medium"]},
        "plots": {"color_trait": "dilution", "group_traits": ["medium", "dilution"], "tech_rep_trait": "tech_rep"},
        "analyses": [
            {"type": "curves", "name": "od_curves", "channel": "OD", "facet_trait": "medium"},
            {"type": "threshold_time", "name": "t_02", "channel": "OD", "threshold": 0.2, "filter": grow},
            {"type": "fraction_of_start", "name": "half", "channel": "OD", "fraction": 0.5, "filter": lyse},
            {"type": "max_growth_rate", "name": "mgr", "channel": "OD", "filter": grow},
            {"type": "ratio", "name": "yfp_per_od", "numerator": "YFP", "denominator": "OD", "min_denominator": 0.05},
            {"type": "curves", "name": "ratio_curves", "channel": "yfp_per_od", "filter": grow},
            {"type": "custom", "name": "extra", "code": "print('custom cell ran')"},
        ],
    }
    path = tmp_path / "experiment_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_build(path)
    assert result.returncode == 0, result.stdout + result.stderr

    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook_path = tmp_path / f"{tmp_path.name}_analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbclient.NotebookClient(notebook, timeout=170, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()

    import pandas as pd
    table = pd.read_csv(tmp_path / "background_table.csv")
    od_blanks = table[table["measurement"] == "OD"]
    values = {str(row.medium if "medium" in table else row.match).casefold(): row.background for row in od_blanks.itertuples()}
    assert values["lb"] == pytest.approx(0.088, abs=0.0002)
    assert values["m9"] == pytest.approx(0.0895, abs=0.0002)
    crossing = pd.read_csv(tmp_path / "t_02.csv")["t_02_h"]
    assert len(crossing) == 27
    assert crossing.dropna().between(4.45 - 0.05, 4.45 + 0.05).all() and crossing.notna().all()
    half = pd.read_csv(tmp_path / "half.csv")["half_h"]
    assert len(half) == 27 and half.notna().all()
    assert half.between(4 * np.log(2) - 0.15, 4 * np.log(2) + 0.15).all()
    assert pd.read_csv(tmp_path / "mgr.csv").iloc[:, -1].notna().all()
    for figure in ("od_curves_LB.png", "od_curves_M9.png", "t_02.png", "ratio_curves.png"):
        assert (tmp_path / "figures" / figure).exists(), figure


def test_build_refuses_a_figure_with_too_many_colors(tmp_path):
    wells = [f"{row}{col}" for row in "ABC" for col in range(1, 5)]
    make_tecan(tmp_path / "run.xlsx", "old", wells=wells)
    layout = {"plate": {"rows": 3, "cols": 4},
              "traits": {"strain": {f"s{i}": [well] for i, well in enumerate(wells)},
                         "medium": {"LB": ["A1:A4", "B1:B2"], "M9": ["B3:B4", "C1:C4"]}}}
    rows, cols, blocks = make_plate_map.build_blocks(layout)
    make_plate_map.write_plate_map(tmp_path / "map.xlsx", rows, cols, blocks)
    config = {
        "title": "T", "purpose": "P", "reader_file": "run.xlsx", "sheet": "Sheet2", "plate_map": "map.xlsx",
        "measurements": {"OD": "OD", "YFP": "YFP"}, "layout_description": "L", "background": {"control_wells": []},
        "plots": {"group_traits": ["strain"]},
        "analyses": [{"type": "curves", "name": "c", "channel": "OD", "color_trait": "strain"}],
    }
    problems, _ = build_notebook.check_config(config, tmp_path)
    assert any("too many to tell apart" in problem for problem in problems)
    config["analyses"][0]["facet_trait"] = "medium"
    assert not build_notebook.check_config(config, tmp_path)[0]
    config["analyses"][0]["filter"] = {"medium": "LB"}
    del config["analyses"][0]["facet_trait"]
    assert not build_notebook.check_config(config, tmp_path)[0]

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reader_functions as rf


def _old_reader_sheet(start_time, columns, rows):
    width = max(len(columns), 2)
    raw = [
        ["Start Time:", start_time] + [np.nan] * (width - 2),
        ["Label: OD"] + [np.nan] * (width - 1),
        columns + [np.nan] * (width - len(columns)),
    ]
    for row in rows:
        raw.append(row + [np.nan] * (width - len(row)))
    raw.append(["End Time:", np.nan] + [np.nan] * (width - 2))
    return pd.DataFrame(raw)


def test_concatenate_reader_sheets_old_offsets_time_and_fills_missing_wells(tmp_path):
    workbook = tmp_path / "reader.xlsx"
    sheet2 = _old_reader_sheet(
        "06/24/2026 10:00:00 AM",
        ["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1"],
        [[1, 0, 37, 0.10], [2, 60, 37, 0.20]],
    )
    sheet3 = _old_reader_sheet(
        "06/24/2026 10:10:00 AM",
        ["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1", "B1"],
        [[1, 0, 37, 0.30, 0.40], [2, 60, 37, 0.50, 0.60]],
    )
    with pd.ExcelWriter(workbook) as writer:
        sheet2.to_excel(writer, sheet_name="Sheet2", header=False, index=False)
        sheet3.to_excel(writer, sheet_name="Sheet3", header=False, index=False)

    out = rf.ReaderExperiment.concatenate_reader_sheets_old(
        workbook,
        "Label: OD",
        ["Sheet2", "Sheet3"],
    )

    assert out["Time [s]"].tolist() == [0, 60, 600, 660]
    assert out["A1"].tolist() == [0.10, 0.20, 0.30, 0.50]
    assert pd.isna(out["B1"].iloc[:2]).all()
    assert out["B1"].iloc[2:].tolist() == [0.40, 0.60]

    zero_filled = rf.ReaderExperiment.concatenate_reader_sheets_old(
        workbook,
        "Label: OD",
        ["Sheet2", "Sheet3"],
        missing_well_value=0,
    )
    assert zero_filled["B1"].iloc[:2].tolist() == [0, 0]


def _old_multi_label_sheet():
    width = 4
    raw = [
        ["Start Time:", "06/24/2026 10:00:00 AM"] + [np.nan] * (width - 2),
        ["OD"] + [np.nan] * (width - 1),
        ["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1"],
        [1, 0, 37, 0.10],
        [2, 60, 37, 0.20],
        [np.nan] * width,
        ["YFP"] + [np.nan] * (width - 1),
        ["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1"],
        [1, 1, 37, 500],
        [2, 61, 37, 600],
        [np.nan] * width,
        ["End Time:", np.nan] + [np.nan] * (width - 2),
    ]
    return pd.DataFrame(raw)


def _plex_multi_label_sheet():
    width = 4
    raw = [
        ["Start Time:", "06/24/2026 10:00:00 AM"] + [np.nan] * (width - 2),
        ["OD"] + [np.nan] * (width - 1),
        ["Cycle Nr.", 1, 2, np.nan],
        ["Time [s]", 0, 60, np.nan],
        ["Temp. [°C]", 37, 37, np.nan],
        ["A1", 0.10, 0.20, np.nan],
        [np.nan] * width,
        ["YFP"] + [np.nan] * (width - 1),
        ["Cycle Nr.", 1, 2, np.nan],
        ["Time [s]", 1, 61, np.nan],
        ["Temp. [°C]", 37, 37, np.nan],
        ["A1", 500, 600, np.nan],
        [np.nan] * width,
        ["End Time:", np.nan] + [np.nan] * (width - 2),
    ]
    return pd.DataFrame(raw)


def _write_sheet(path, frame, sheet="Sheet2"):
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=sheet, header=False, index=False)


def test_old_parser_selects_named_block_and_rejects_unknown_label(tmp_path):
    workbook = tmp_path / "multi.xlsx"
    _write_sheet(workbook, _old_multi_label_sheet())

    yfp = rf.ReaderExperiment.get_measurement_df_from_reader_file_old(workbook, "YFP", sheet_name="Sheet2")
    assert yfp["A1"].tolist() == [500, 600]

    for bad_label in ["GFP", "Label: GFP"]:
        try:
            rf.ReaderExperiment.get_measurement_df_from_reader_file_old(workbook, bad_label, sheet_name="Sheet2")
        except ValueError as err:
            assert "OD" in str(err) and "YFP" in str(err)
        else:
            raise AssertionError(f"{bad_label!r} should not return another block's data")


def test_plex_parser_selects_named_block_and_rejects_unknown_label(tmp_path):
    workbook = tmp_path / "multi.xlsx"
    _write_sheet(workbook, _plex_multi_label_sheet())

    yfp = rf.ReaderExperiment.get_measurement_df_from_reader_file_plex(workbook, "YFP", sheet_name="Sheet2")
    assert yfp["A1"].tolist()[:2] == [500, 600]

    try:
        rf.ReaderExperiment.get_measurement_df_from_reader_file_plex(workbook, "GFP", sheet_name="Sheet2")
    except ValueError as err:
        assert "OD" in str(err) and "YFP" in str(err)
    else:
        raise AssertionError("an unknown label should not return another block's data")


def test_single_label_sheet_accepts_both_spellings_and_rejects_other_names(tmp_path):
    workbook = tmp_path / "single.xlsx"
    sheet = _old_reader_sheet(
        "06/24/2026 10:00:00 AM",
        ["Cycle Nr.", "Time [s]", "Temp. [°C]", "A1"],
        [[1, 0, 37, 0.10], [2, 60, 37, 0.20]],
    )
    _write_sheet(workbook, sheet)

    for label in ["OD", "Label: OD"]:
        out = rf.ReaderExperiment.get_measurement_df_from_reader_file_old(workbook, label, sheet_name="Sheet2")
        assert out["A1"].tolist() == [0.10, 0.20]

    try:
        rf.ReaderExperiment.get_measurement_df_from_reader_file_old(workbook, "YFP", sheet_name="Sheet2")
    except ValueError:
        pass
    else:
        raise AssertionError("YFP should not return the OD block")


def test_synergy_parser_reads_runs_longer_than_24_hours(tmp_path):
    import datetime as dt

    workbook = tmp_path / "synergy.xlsx"
    raw = [
        [630, None, None, None],
        [None, None, None, None],
        [None, "Time", "T° 630", "A1"],
        [None, dt.time(0, 0, 3), 37.0, 0.10],
        [None, dt.time(23, 59, 0), 37.0, 0.20],
        [None, dt.timedelta(days=1, minutes=3), 37.0, 0.30],
        [None, dt.time(0, 0, 0), None, None],
    ]
    frame = pd.DataFrame(raw, dtype=object)
    frame.iloc[5, 1] = "24:03:00"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Plate 1 - Sheet1", header=False, index=False)
        sheet = writer.sheets["Plate 1 - Sheet1"]
        sheet.cell(row=6, column=2).value = dt.timedelta(days=1, minutes=3)
        sheet.cell(row=6, column=2).number_format = "[h]:mm:ss"

    out = rf.ReaderExperiment.get_measurement_df_from_reader_file_synergy(workbook, 630, sheet_name="Plate 1 - Sheet1")
    time = pd.to_numeric(out["time"]).to_numpy()
    assert np.all(np.diff(time) > 0)
    assert np.isclose(time[-1], 24.05)

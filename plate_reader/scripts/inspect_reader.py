#!/usr/bin/env python3
"""Report which plate reader produced an Excel file and what it measured.

Usage:
    python inspect_reader.py READER_FILE.xlsx [--json OUT.json] [--grids]

Prints a readable report and optionally writes the same facts as JSON. Reads the file only.
--grids adds, per channel, a plate-shaped table of each well's start and peak. It helps when
checking a layout and looking for wells that stay flat.
Needs openpyxl and nothing else, so any agent can run it before the analysis environment exists.
"""
import argparse
import datetime as dt
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    import openpyxl
except ImportError:
    sys.exit("inspect_reader.py needs openpyxl: pip install openpyxl")

# Serial numbers decide this, not the device name: all three Tecans call themselves "infinite 200Pro".
KNOWN_READERS = {
    "1506006237": {"name": "Monochromator", "parser": "old"},
    "1105003777": {"name": "Infinite", "parser": "old"},
    "2503006744": {"name": "Plex", "parser": "plex"},
    "2401110F": {"name": "Synergy H1", "parser": "synergy"},
}
WELL = re.compile(r"^[A-Z]{1,2}\d{1,2}$")


def _text(value):
    return "" if value is None else str(value).strip()


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace("*", ""))
    except ValueError:
        return None


def _hours(value):
    if isinstance(value, dt.timedelta):
        return value.total_seconds() / 3600
    if isinstance(value, dt.datetime):
        return value.hour + value.minute / 60 + value.second / 3600
    if isinstance(value, dt.time):
        return value.hour + value.minute / 60 + value.second / 3600
    return None


def _cell(row, index):
    return row[index] if index < len(row) else None


def _cell_has_value(value):
    return value is not None and str(value).strip() != ""


def _well_sort_key(well):
    return (well[0], int(well[1:]))


def compress_wells(wells):
    """'B2-G11 (60 wells)' when the wells fill a rectangle, otherwise a list."""
    if not wells:
        return "none"
    rows = sorted({w[0] for w in wells})
    cols = sorted({int(w[1:]) for w in wells})
    if len(wells) == len(rows) * len(cols):
        return f"{rows[0]}{cols[0]}-{rows[-1]}{cols[-1]} ({len(wells)} wells)"
    return f"{','.join(sorted(wells, key=_well_sort_key))} ({len(wells)} wells)"


def _block_stats(times_h, temps, well_values):
    finite = [t for t in times_h if t is not None]
    intervals = [b - a for a, b in zip(finite, finite[1:])]
    tokens = Counter()
    wells_with_data, wells_without_data = [], []
    starts, peaks = {}, {}
    for well, values in well_values.items():
        numbers = []
        for value in values:
            if value is None:
                continue
            number = _number(value)
            if number is None:
                tokens[str(value).strip()] += 1
            else:
                numbers.append(number)
        (wells_with_data if numbers else wells_without_data).append(well)
        if numbers:
            starts[well] = statistics.median(numbers[1:6] or numbers)
            peaks[well] = max(numbers)

    median_interval = statistics.median(intervals) if intervals else None
    gaps = []
    if median_interval:
        for t0, t1 in zip(finite, finite[1:]):
            if t1 - t0 > 1.5 * median_interval:
                gaps.append({"from_h": round(t0, 3), "to_h": round(t1, 3), "minutes": round((t1 - t0) * 60, 1)})

    # A read with a timestamp can still lack values, if the run was stopped within it.
    partial_reads = []
    n_measured = len(wells_with_data)
    for index, t in enumerate(times_h):
        if t is None or not n_measured:
            continue
        count = sum(1 for w in wells_with_data if index < len(well_values[w]) and _cell_has_value(well_values[w][index]))
        if count < n_measured:
            partial_reads.append({"read": index + 1, "time_h": round(t, 3), "wells_with_values": count})

    return {
        "reads": len(finite),
        "first_h": round(finite[0], 4) if finite else None,
        "last_h": round(finite[-1], 4) if finite else None,
        "median_interval_min": round(statistics.median(intervals) * 60, 2) if intervals else None,
        "time_increases": all(i > 0 for i in intervals),
        "first_temperatures_C": [t for t in temps if t is not None][:4],
        "wells_measured": compress_wells(wells_with_data),
        "well_list": sorted(wells_with_data, key=_well_sort_key),
        "wells_without_data": sorted(wells_without_data, key=_well_sort_key),
        "non_numeric_values": dict(tokens),
        "interval_gaps": gaps,
        "partial_reads": partial_reads,
        "start_by_well": starts,
        "peak_by_well": peaks,
    }


def _channel_for(mode, name):
    mode_text = (mode or "").lower()
    if "absorbance" in mode_text:
        return "OD"
    if "lum" in mode_text:
        return name
    return name


def inspect_tecan_sheet(rows):
    info = {"format": "tecan", "settings": {}, "actions": [], "blocks": []}
    device_row = next(r for r in rows if _text(_cell(r, 0)).startswith("Device:"))
    info["device"] = _text(device_row[0]).replace("Device:", "").strip()
    for value in device_row[1:]:
        if "Serial number" in _text(value):
            info["serial"] = _text(value).split(":")[1].strip()
            break

    labels, current, in_actions = [], None, False
    first_cycle = next((i for i, r in enumerate(rows) if _text(_cell(r, 0)).startswith("Cycle Nr.")), None)
    meta_rows = rows if first_cycle is None else rows[:first_cycle]
    for row in meta_rows:
        key = _text(_cell(row, 0))
        value = _text(_cell(row, 4))
        unit = _text(_cell(row, 5))
        if key == "Start Time:":
            info["start_time"] = _text(_cell(row, 1))
        elif key == "Plate":
            info["plate"] = value
        elif key == "Part of Plate":
            info["part_of_plate"] = value
        elif key == "Wait (Plate)":
            info["settings"]["Target temperature"] = unit.replace("Target Temperature:", "").strip()
        elif key.startswith("List of actions"):
            in_actions = True
            continue
        elif key.startswith("Shaking") or key == "Wait (Time)":
            info["settings"][key.rstrip(":")] = f"{value} {unit}".strip()
        if in_actions:
            if not key:
                in_actions = False
            else:
                if not key.startswith(("Shaking", "Wait")):
                    info["actions"].append(key)
                continue
        if key.startswith("Label:"):
            current = {"label": key.replace("Label:", "").strip()}
            labels.append(current)
        elif key == "Mode":
            current = {"Mode": value}
            labels.append(current)
        elif current is not None and key and key not in ("Part of Plate", "Start Time:") and value:
            current[key] = f"{value} {unit}".strip()

    names_above = []
    layout = None
    for i, row in enumerate(rows):
        if not _text(_cell(row, 0)).startswith("Cycle Nr."):
            continue
        above = _text(_cell(rows[i - 1], 0)) if i else ""
        names_above.append(above)
        if _text(_cell(row, 1)).startswith("Time"):
            layout = "columns"
            header = [_text(c) for c in row]
            data = []
            for r in rows[i + 1:]:
                if _cell(r, 1) is None or _text(_cell(r, 0)).startswith("End Time"):
                    break
                data.append(r)
            times = [_number(_cell(r, 1)) for r in data]
            times = [t / 3600 if t is not None else None for t in times]
            temps = [_number(_cell(r, 2)) for r in data]
            well_values = {h: [_cell(r, j) for r in data] for j, h in enumerate(header) if WELL.match(h)}
        else:
            layout = "rows"
            end = i + 1
            while end < len(rows) and _text(_cell(rows[end], 0)):
                end += 1
            block = {_text(_cell(r, 0)): list(r[1:]) for r in rows[i:end]}
            times = [(_number(v) / 3600 if _number(v) is not None else None) for v in block.get("Time [s]", [])]
            temps = [_number(v) for v in block.get("Temp. [°C]", [])]
            n = sum(1 for t in times if t is not None)
            well_values = {k: v[:n] for k, v in block.items() if WELL.match(k)}
        info["blocks"].append(_block_stats(times, temps, well_values))

    info["layout"] = layout
    single_label = len(info["blocks"]) == 1 and labels and "label" in labels[0]
    for index, block in enumerate(info["blocks"]):
        setting = labels[index] if index < len(labels) else {}
        name = names_above[index] or setting.get("label", "")
        if single_label and "Mode" not in setting:
            modes = [a for a in info["actions"] if a in ("Absorbance", "Fluorescence", "Luminescence")]
            has_wavelength = "Measurement Wavelength" in setting or "Wavelength" in setting
            setting = dict(setting, Mode=modes[0] if modes else ("Absorbance" if has_wavelength else ""))
        block["library_label"] = name
        block["mode"] = setting.get("Mode")
        block["settings"] = {k: v for k, v in setting.items() if k not in ("label", "Mode")}
        block["suggested_channel"] = _channel_for(block["mode"], name)
    info["run_finished"] = any(_text(_cell(r, 0)).startswith("End Time") for r in rows)
    return info


def inspect_synergy_sheet(rows):
    info = {"format": "synergy", "layout": "synergy", "blocks": [], "procedure": []}
    for row in rows:
        key = _text(_cell(row, 0))
        if key == "Reader Type:":
            info["device"] = _text(_cell(row, 1))
        elif key == "Reader Serial Number:":
            info["serial"] = _text(_cell(row, 1))
        elif key == "Plate Type":
            info["plate"] = _text(_cell(row, 1))
        elif key == "Date":
            info["date"] = _text(_cell(row, 1))[:10]
        elif key == "Time" and "time" not in info:
            info["time"] = _text(_cell(row, 1))
    info["start_time"] = f"{info.get('date', '?')} {info.get('time', '?')}"

    starts = [i for i, r in enumerate(rows)
              if _cell(r, 0) is not None and i + 2 < len(rows) and _text(_cell(rows[i + 2], 1)) == "Time"]
    procedure_start = next((i for i, r in enumerate(rows) if _text(_cell(r, 0)) == "Procedure Details"), None)
    if procedure_start is not None:
        for row in rows[procedure_start + 1:(starts[0] if starts else len(rows))]:
            line = " ".join(x for x in (_text(_cell(row, 0)), _text(_cell(row, 1))) if x)
            if line:
                info["procedure"].append(line)

    for start in starts:
        key = _cell(rows[start], 0)
        header = [_text(c) for c in rows[start + 2]]
        data = []
        for r in rows[start + 3:]:
            if all(c is None for c in r[2:]):
                break
            data.append(r)
        times = [_hours(_cell(r, 1)) for r in data]
        temps = [_number(_cell(r, 2)) for r in data]
        well_values = {h: [_cell(r, j) for r in data] for j, h in enumerate(header) if WELL.match(h)}
        block = _block_stats(times, temps, well_values)
        library_label = int(key) if isinstance(key, float) and key.is_integer() else key
        block["library_label"] = library_label
        is_absorbance = _number(key) is not None
        block["mode"] = "Absorbance" if is_absorbance else ("Luminescence" if ":Lum" in str(key) else "Fluorescence")
        block["settings"] = {"Wavelength": f"{library_label} nm"} if is_absorbance else {"Channel": str(key)}
        block["suggested_channel"] = "OD" if is_absorbance else re.split(r"[:]", str(key))[0]
        info["blocks"].append(block)
    info["run_finished"] = None
    return info


def inspect_file(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    report = {"file": str(path), "sheets": []}
    for sheet_name in workbook.sheetnames:
        rows = [tuple(row) for row in workbook[sheet_name].iter_rows(values_only=True)]
        first_cells = [_text(_cell(r, 0)) for r in rows[:60]]
        if any(c.startswith("Device:") for c in first_cells):
            info = inspect_tecan_sheet(rows)
        elif any(c == "Reader Type:" for c in first_cells):
            info = inspect_synergy_sheet(rows)
        else:
            has_content = any(any(c is not None for c in r) for r in rows)
            info = {"format": "not a reader sheet" if has_content else "empty"}
        info["sheet"] = sheet_name
        serial = info.get("serial")
        known = KNOWN_READERS.get(serial)
        if info["format"] in ("tecan", "synergy"):
            info["reader_name"] = known["name"] if known else "UNKNOWN READER"
            layout_parser = {"columns": "old", "rows": "plex", "synergy": "synergy"}.get(info.get("layout"))
            info["parser"] = layout_parser
            if known and layout_parser and known["parser"] != layout_parser:
                info["warning"] = (f"Serial {serial} normally exports for the {known['parser']!r} parser, but this "
                                   f"sheet's layout needs {layout_parser!r}.")
            elif not known:
                info["warning"] = f"Serial {serial} is not in the known readers. Ask which reader this is."
        report["sheets"].append(info)
    workbook.close()
    return report


def _grid(values, digits):
    rows = sorted({w[0] for w in values})
    cols = sorted({int(w[1:]) for w in values})
    lines = ["      " + " ".join(str(c).rjust(digits + 3) for c in cols)]
    for row in rows:
        cells = [f"{values[row + str(c)]:.{digits}f}".rjust(digits + 3) if row + str(c) in values else ".".rjust(digits + 3)
                 for c in cols]
        lines.append(f"    {row} " + " ".join(cells))
    return lines


def _start_datetime(text):
    for pattern in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(str(text), pattern)
        except ValueError:
            pass
    return None


def overlapping_sheets(report):
    """Pairs of sheets from one reader whose time spans overlap, so one may be a copy of part of the other."""
    spans = []
    for info in report["sheets"]:
        start = _start_datetime(info.get("start_time"))
        last = [b["last_h"] for b in info.get("blocks", []) if b.get("last_h") is not None]
        if start and last:
            spans.append((info.get("serial"), info["sheet"], start, start + dt.timedelta(hours=max(last))))
    pairs = []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if a[0] == b[0] and a[2] < b[3] and b[2] < a[3]:
                pairs.append((a[1], b[1]))
    return pairs


def format_report(report, grids=False):
    lines = [f"File: {report['file']}"]
    data_sheets = [s["sheet"] for s in report["sheets"] if s["format"] in ("tecan", "synergy")]
    if len(data_sheets) > 1:
        lines.append(f"WARNING: {len(data_sheets)} sheets hold data: {data_sheets}. They may be separate experiments. "
                     "Ask which one or ones to analyze.")
    for first, second in overlapping_sheets(report):
        lines.append(f"WARNING: sheets {first!r} and {second!r} overlap in time on one reader, so one may be a copy "
                     "of part of the other. Compare them before choosing.")
    for info in report["sheets"]:
        lines.append("")
        if info["format"] not in ("tecan", "synergy"):
            lines.append(f"Sheet {info['sheet']!r}: {info['format']}")
            continue
        lines.append(f"Sheet {info['sheet']!r}: {info['reader_name']} ({info.get('device')}, serial {info.get('serial')})")
        lines.append(f"  Library parser: reader={info.get('parser')!r}, sheet_name={info['sheet']!r}")
        lines.append(f"  Started: {info.get('start_time')}   Plate: {info.get('plate')}")
        if info.get("part_of_plate"):
            lines.append(f"  Part of plate read: {info['part_of_plate']}")
        for key, value in info.get("settings", {}).items():
            lines.append(f"  {key}: {value}")
        if info.get("procedure"):
            lines.append("  Procedure:")
            lines.extend(f"    {line}" for line in info["procedure"])
        if info.get("run_finished") is False:
            lines.append("  WARNING: no 'End Time:' row, so the run or the export may be incomplete")
        if info.get("warning"):
            lines.append(f"  WARNING: {info['warning']}")
        if not info["blocks"]:
            lines.append("  WARNING: no data blocks found")
        for block in info["blocks"]:
            settings = ", ".join(f"{k}={v}" for k, v in block["settings"].items())
            lines.append(f"  Channel {block['suggested_channel']!r} (library label {block['library_label']!r}): "
                         f"{block['mode']}. {settings}")
            lines.append(f"    {block['reads']} reads, {block['first_h']}-{block['last_h']} h, every "
                         f"{block['median_interval_min']} min; first temperatures {block['first_temperatures_C']} °C")
            lines.append(f"    Wells: {block['wells_measured']}")
            missing = block["wells_without_data"]
            if missing:
                shown = ",".join(missing) if len(missing) <= 12 else f"{len(missing)} wells"
                lines.append(f"    Columns present but no numeric data: {shown}")
            if block["non_numeric_values"]:
                lines.append(f"    WARNING: non-numeric values read as missing: {block['non_numeric_values']}")
            if not block["time_increases"]:
                lines.append("    WARNING: time does not increase")
            for gap in block["interval_gaps"]:
                lines.append(f"    WARNING: {gap['minutes']} min between reads at {gap['from_h']}-{gap['to_h']} h. "
                             "Was the plate taken out? Ask what happened.")
            for partial in block["partial_reads"]:
                lines.append(f"    WARNING: read {partial['read']} at {partial['time_h']} h has values in only "
                             f"{partial['wells_with_values']} wells. The run was probably stopped during this read.")
            if grids and block["start_by_well"]:
                digits = 3 if max(block["peak_by_well"].values()) < 10 else 0
                lines.append("    Start (median of reads 2-6):")
                lines.extend("  " + line for line in _grid(block["start_by_well"], digits))
                lines.append("    Peak:")
                lines.extend("  " + line for line in _grid(block["peak_by_well"], digits))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reader_file", type=Path)
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    parser.add_argument("--grids", action="store_true", help="add a plate-shaped table of each well's start and peak")
    args = parser.parse_args()
    report = inspect_file(args.reader_file)
    print(format_report(report, grids=args.grids))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

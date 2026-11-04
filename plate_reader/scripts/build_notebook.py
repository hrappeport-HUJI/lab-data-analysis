#!/usr/bin/env python3
"""Check an experiment config against its files and write the analysis notebook.

Usage:
    python build_notebook.py EXPERIMENT_FOLDER/experiment_config.json

The config format is in the reader-experiment skill, references/config.md. All paths in the config are relative to the
experiment folder, and the notebook is written there. Standard library plus openpyxl only.
Every problem found is reported together, and no notebook is written until they are fixed.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from pprint import pformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_reader import inspect_file  # noqa: E402

try:
    import openpyxl
except ImportError:
    sys.exit("build_notebook.py needs openpyxl: pip install openpyxl")


def py(value):
    """A Python literal, for pasting into a code cell."""
    return pformat(value, width=100, sort_dicts=False)


# ---------------------------------------------------------------- reading inputs
def read_plate_map(path):
    """Read the blocks by position, as the library does, because a block may contain blank rows."""
    workbook = openpyxl.load_workbook(path, data_only=True)
    rows = [list(r) for r in workbook["Sheet1"].iter_rows(values_only=True)]
    while rows and all(v is None for v in rows[-1]):
        rows.pop()
    if rows and any(v is not None for v in rows[0]):
        return None, f"Plate map {path.name}: the first row must be blank, or the library misreads it"
    body = rows[1:]
    blanks = [i for i, row in enumerate(body[1:]) if row[0] is None]
    n_rows = blanks[0] if blanks else len(body) - 1
    blocks = {}
    for start in range(0, len(body), n_rows + 2):
        name = body[start][0]
        blocks[name] = body[start + 1:start + 1 + n_rows]
    return blocks, None


def library_geometry_problem(path):
    """Apply the geometry rule of ReaderExperiment.get_plate_map_dict and report where it would fail.

    The library assumes equally sized blocks separated by one blank row, and that the sheet
    ends with the last block. A stray cell below the map breaks that, or shifts the blocks.
    """
    workbook = openpyxl.load_workbook(path, data_only=True)
    rows = [list(r) for r in workbook["Sheet1"].iter_rows(values_only=True)]
    while rows and all(v is None for v in rows[-1]):
        rows.pop()
    body = rows[1:]
    first_column = [r[0] for r in body]
    blanks = [i for i, v in enumerate(first_column[1:]) if v is None]
    n_rows = blanks[0] if blanks else len(body) - 1
    expected = len(body) // (n_rows + 2) + 1
    names = []
    for i in range(expected):
        position = i * (n_rows + 2)
        if position >= len(body):
            return (f"Plate map {path.name}: the library can't read it, because {len(body) + 1} rows don't fit blocks of "
                    f"{n_rows} rows. Look for stray cells below the last block.")
        names.append(first_column[position])
    if any(not isinstance(name, str) or re.fullmatch(r"[A-Z]\d+", name) for name in names):
        return (f"Plate map {path.name}: the library would read block names {names}. Blocks must be {n_rows} rows "
                "high and separated by one blank row, with nothing below the last block.")
    return None


def map_well_values(blocks, trait):
    wells, values = blocks["well"], blocks[trait]
    out = {}
    for r, row in enumerate(wells):
        for c, well in enumerate(row):
            if well is not None:
                value = values[r][c] if r < len(values) and c < len(values[r]) else None
                out[well] = value
    return out


def selected_sheets(config):
    sheet = config["sheet"]
    return sheet if isinstance(sheet, list) else [sheet]


ALLOWED_TOP = {"title", "purpose", "reader_file", "sheet", "plate_map", "measurements", "od_channel", "reader_parser",
               "exclude_time_windows", "layout_description", "background", "plots", "analyses", "notebook",
               "overwrite_notebook", "zettelkasten_note"}
ALLOWED_BACKGROUND = {"control_wells", "match_traits", "control_decisions", "manual", "exclude_wells",
                      "accept_blank_drift", "accept_own_start", "offset"}
ALLOWED_PLOTS = {"color_trait", "group_traits", "bio_rep_trait", "tech_rep_trait", "annotate_traits"}
COMMON = {"type", "name", "title", "note"}
ALLOWED_ANALYSIS = {
    "curves": COMMON | {"channel", "filter", "color_trait", "facet_trait", "yscale", "ylim", "t_span", "ylabel"},
    "threshold_time": COMMON | {"channel", "threshold", "direction", "t_start", "filter", "group_traits"},
    "fraction_of_start": COMMON | {"channel", "fraction", "direction", "start_reads", "start_skip", "min_start",
                                   "filter", "group_traits"},
    "max_growth_rate": COMMON | {"channel", "t_start", "t_span", "options", "filter", "group_traits"},
    "ratio": COMMON | {"numerator", "denominator", "min_denominator"},
    "custom": COMMON | {"code"},
    "readthrough": COMMON | {"red", "green", "od", "reporter_trait", "in_frame", "autofluorescence", "test", "condition_trait",
                             "replicate_trait", "time_h", "window_h", "expected_percent", "time_course_step_h",
                             "stability_times_h"},
}
CONTROL_HINTS = {"ref", "ref_well", "blank", "control", "ctrl", "medium", "medium_control", "no_cells"}


def _unknown_keys(found, allowed, where):
    return [f"{where}: unknown option {key!r}. Allowed: {sorted(allowed)}" for key in found if key not in allowed]


def _strip_label(label):
    return re.sub(r"^Label:\s*", "", str(label).strip())


def find_block(info, label):
    """The data block the library would read for this label."""
    for block in info["blocks"]:
        if info["format"] == "tecan":
            if _strip_label(block["library_label"]) == _strip_label(label):
                return block
        elif block["library_label"] == label:
            return block
    return None


def readthrough_problems(analysis, name, config, traits, blocks, loaded, info):
    """A readthrough analysis needs all its parts on the plate and a time inside the run."""
    problems = []
    for key in ("red", "green", "reporter_trait", "in_frame", "autofluorescence", "test", "condition_trait", "replicate_trait", "time_h"):
        if key not in analysis:
            problems.append(f"Analysis {name!r} needs {key}")
    if problems:
        return problems
    measurements = config["measurements"]
    for key in ("red", "green", "od"):
        if key in analysis and analysis[key] not in measurements:
            problems.append(f"Analysis {name!r}: {key} {analysis[key]!r} is not a loaded channel: {list(measurements)}")
    if not isinstance(analysis["test"], list) or not analysis["test"]:
        problems.append(f"Analysis {name!r}: test must be a list of plate-map values")
        return problems
    if analysis["reporter_trait"] in traits:
        present = {map_well_values(blocks, analysis["reporter_trait"]).get(w) for w in loaded} - {None}
        for value in [analysis["in_frame"], analysis["autofluorescence"], *analysis["test"]]:
            if value not in present:
                problems.append(f"Analysis {name!r}: no wells have {analysis['reporter_trait']!r} = {value!r}. Values there: {sorted(present, key=str)}")
    expected = analysis.get("expected_percent")
    if expected is not None and not (isinstance(expected, list) and len(expected) == 2 and expected[0] < expected[1]):
        problems.append(f"Analysis {name!r}: expected_percent must be [low, high]")
    if not config.get("background", {}).get("control_wells") and not config.get("background", {}).get("manual"):
        problems.append(f"Analysis {name!r} needs medium controls or a manual blank: a well's own start would remove the signal to be measured.")
    for key in ("red", "green"):
        block = find_block(info, measurements.get(analysis[key])) if analysis[key] in measurements else None
        if block is not None and not (block["first_h"] <= analysis["time_h"] <= block["last_h"]):
            problems.append(f"Analysis {name!r}: time_h {analysis['time_h']} is outside the {analysis[key]} reads, {block['first_h']}–{block['last_h']} h")
    return problems


def check_config(config, folder):
    """Return (problems, facts). Facts are the reader sheet and the plate-map traits."""
    problems = _unknown_keys(config, ALLOWED_TOP, "Config")
    for key in ("title", "purpose", "reader_file", "sheet", "plate_map", "measurements", "layout_description", "background"):
        if key not in config:
            problems.append(f"Config is missing {key!r}")
    if "background" in config and "control_wells" not in config["background"]:
        problems.append("background must state control_wells, as an empty list if the plate has none")
    if problems:
        return problems, {}
    problems += _unknown_keys(config["background"], ALLOWED_BACKGROUND, "background")
    problems += _unknown_keys(config.get("plots", {}), ALLOWED_PLOTS, "plots")

    reader_path = folder / config["reader_file"]
    map_path = folder / config["plate_map"]
    facts = {"warnings": []}
    if not reader_path.exists():
        problems.append(f"Reader file not found in the experiment folder: {config['reader_file']}")
    if not map_path.exists():
        problems.append(f"Plate map not found in the experiment folder: {config['plate_map']}")
    if problems:
        return problems, facts

    report = inspect_file(reader_path)
    sheets = {s["sheet"]: s for s in report["sheets"]}
    chosen = []
    for name in selected_sheets(config):
        if name not in sheets:
            problems.append(f"Sheet {name!r} is not in the reader file. Sheets: {list(sheets)}")
        elif sheets[name]["format"] not in ("tecan", "synergy"):
            problems.append(f"Sheet {name!r} holds no reader data ({sheets[name]['format']})")
        else:
            chosen.append(sheets[name])
    if problems:
        return problems, facts
    info = chosen[0]
    facts["info"] = info
    for sheet in chosen:
        if sheet["reader_name"] == "UNKNOWN READER":
            problems.append(f"Sheet {sheet['sheet']!r} has unknown serial {sheet.get('serial')}. Ask which reader this "
                            "is, then add it to KNOWN_READERS in inspect_reader.py.")
    if len({sheet.get("serial") for sheet in chosen}) > 1:
        problems.append("The chosen sheets come from different readers: "
                        + ", ".join(f"{sheet['sheet']} ({sheet.get('serial')})" for sheet in chosen))
    if isinstance(config["sheet"], list) and info["parser"] != "old":
        problems.append("Joining several sheets is only supported by the library's 'old' parser")
    if config.get("reader_parser") not in (None, info["parser"]):
        problems.append(f"Config says reader_parser={config['reader_parser']!r}, but this sheet needs {info['parser']!r}")

    for channel, label in config["measurements"].items():
        if find_block(info, label) is None:
            labels = [b["library_label"] for b in info["blocks"]]
            problems.append(f"Channel {channel!r}: label {label!r} is not in sheet {info['sheet']!r}. Labels: {labels}")
    od_labels = [b["library_label"] for b in info["blocks"] if (b.get("mode") or "").lower().startswith("absorbance")]
    od_channel = config.get("od_channel", "OD" if "OD" in config["measurements"] else None)
    if od_channel is None and od_labels:
        problems.append(f"The sheet has an absorbance channel ({od_labels}) that the config does not load. "
                        "Load it and set od_channel: contamination checks depend on it.")
    if od_channel is not None and od_channel not in config["measurements"]:
        problems.append(f"od_channel {od_channel!r} is not in measurements")
    facts["od_channel"] = od_channel

    library_error = library_geometry_problem(map_path)
    if library_error:
        problems.append(library_error)
        return problems, facts
    blocks, error = read_plate_map(map_path)
    if error:
        problems.append(error)
        return problems, facts
    if "well" not in blocks:
        problems.append("The plate map has no 'well' block")
        return problems, facts
    n_cols = max((c + 1 for row in blocks["well"] for c, well in enumerate(row) if well is not None), default=0)
    for name, grid in blocks.items():
        stray = [value for row in grid for value in row[n_cols:] if value is not None]
        if stray:
            problems.append(f"Plate map block {name!r} has cells to the right of the plate ({stray[:3]}). "
                            "The library would read them as wells. Clear them in the copy.")
    traits = [t for t in blocks if t != "well"]
    facts["traits"] = traits
    facts["warnings"] += [f"Plate-map trait {t!r} has stray spaces. Use that exact name in the config."
                          for t in traits if t != str(t).strip()]

    status = map_well_values(blocks, "well_status") if "well_status" in blocks else {}
    map_wells = [w for row in blocks["well"] for w in row if w is not None]
    loaded = [w for w in map_wells if status.get(w) != "empty"]
    facts["loaded_wells"] = loaded

    measured = set()
    for sheet in chosen:
        for block in sheet["blocks"]:
            measured |= set(block["well_list"])
    described = {w: False for w in map_wells}
    for trait in traits:
        if trait == "well_status":
            continue
        for well, value in map_well_values(blocks, trait).items():
            if value is not None and str(value).strip().lower() != "empty":
                described[well] = True
    unmeasured = [w for w in loaded if w not in measured and described.get(w)]
    outside_map = [w for w in measured if w not in set(map_wells)]
    ignored = sorted(w for w in measured if status.get(w) == "empty")
    if unmeasured:
        problems.append(f"Plate map lists wells the reader has no data for: {','.join(unmeasured)}. "
                        "Mark them empty or check the sheet.")
    if outside_map:
        problems.append(f"The reader measured wells outside the plate map: {','.join(sorted(outside_map))}.")
    if ignored:
        facts["warnings"].append(f"The reader has data for {len(ignored)} wells the plate map marks empty, and they won't "
                                 f"be analyzed: {','.join(ignored[:12])}{' ...' if len(ignored) > 12 else ''}. "
                                 "Check this is the right map for the sheet.")

    if not isinstance(config["sheet"], list):
        facts["expected"] = {}
        for channel, label in config["measurements"].items():
            block = find_block(info, label)
            if block is not None:
                facts["expected"][channel] = [block["reads"], len(set(block["well_list"]) & set(loaded))]

    background = config["background"]
    control_wells = background["control_wells"]
    loaded_set = set(loaded)
    for well in control_wells:
        if well not in loaded_set:
            problems.append(f"Control well {well} is not a loaded well of the plate map")
    map_controls = [w for w, v in status.items() if v == "medium_control"]
    if map_controls and sorted(map_controls) != sorted(control_wells):
        problems.append(f"The plate map marks medium controls {sorted(map_controls)} but the config lists "
                        f"{sorted(control_wells)}")
    if not control_wells:
        hints = sorted({str(value) for trait in traits for row in blocks[trait] for value in row
                        if value is not None and str(value).strip().casefold() in CONTROL_HINTS})
        if hints:
            facts["warnings"].append(f"The config lists no medium controls, but the plate map holds the values {hints}. "
                                     "Confirm with the experimenter that none of these wells is medium alone.")
    for trait in background.get("match_traits", []):
        if trait not in traits:
            problems.append(f"match_traits names {trait!r}, not in the plate map. Traits: {traits}")
    for name in ("offset", "manual"):
        for channel in background.get(name, {}):
            if channel not in config["measurements"]:
                problems.append(f"background.{name} names channel {channel!r}. Loaded channels: {list(config['measurements'])}")
    for channel in background.get("accept_blank_drift", []):
        if channel not in config["measurements"]:
            problems.append(f"accept_blank_drift names {channel!r}, not a loaded channel")
    for well in background.get("exclude_wells", []):
        if well not in loaded_set:
            problems.append(f"exclude_wells names {well}, not a loaded well of the plate map")
    for well, decision in background.get("control_decisions", {}).items():
        if well not in control_wells:
            problems.append(f"control_decisions names {well}, which is not in control_wells")
        if decision not in ("keep", "drop"):
            problems.append(f"control_decisions[{well!r}] must be 'keep' or 'drop', got {decision!r}")
    accepted = background.get("accept_own_start", [])
    if accepted is not True:
        for well in accepted:
            if well not in loaded_set:
                problems.append(f"accept_own_start names {well}, not a loaded well of the plate map")

    plots = config.get("plots", {})
    named = list(plots.get("group_traits", [])) + list(plots.get("annotate_traits") or [])
    named += [plots.get("color_trait"), plots.get("bio_rep_trait"), plots.get("tech_rep_trait")]
    for analysis in config.get("analyses", []):
        named += [analysis.get("color_trait"), analysis.get("facet_trait")]
        named += list(analysis.get("group_traits", []))
        named += list(analysis.get("filter", {}))
        named += [analysis.get(key) for key in ("reporter_trait", "condition_trait", "replicate_trait")]
    for trait in dict.fromkeys(named):
        if trait is not None and trait not in traits:
            problems.append(f"Trait {trait!r} is used by the config but is not in the plate map. Traits: {traits}")

    for window in config.get("exclude_time_windows", []):
        if not (isinstance(window, list) and len(window) == 2 and window[0] < window[1]):
            problems.append(f"exclude_time_windows entries must be [start_h, end_h], got {window!r}")

    loaded_channels = list(config["measurements"])
    reserved = set(loaded_channels) | {f"{channel}_bg" for channel in loaded_channels}
    available = list(loaded_channels)
    names = []
    for analysis in config.get("analyses", []):
        kind = analysis.get("type")
        if kind not in ALLOWED_ANALYSIS:
            problems.append(f"Unknown analysis type {kind!r}. Known: {sorted(ALLOWED_ANALYSIS)}")
            continue
        name = analysis.get("name")
        if not name or not re.fullmatch(r"[A-Za-z0-9_]+", str(name)):
            problems.append(f"Analysis {kind}: 'name' must use only letters, digits and underscores, got {name!r}")
        elif name in names:
            problems.append(f"Two analyses share the name {name!r}")
        names.append(name)
        problems += _unknown_keys(analysis, ALLOWED_ANALYSIS[kind], f"Analysis {name!r}")
        if kind in ("curves", "threshold_time", "max_growth_rate") and analysis.get("channel") not in available:
            problems.append(f"Analysis {name!r}: channel {analysis.get('channel')!r} is not among {available}. "
                            "A ratio must come before the analyses that use it.")
        if kind == "fraction_of_start":
            if analysis.get("channel") not in loaded_channels:
                problems.append(f"Analysis {name!r}: channel {analysis.get('channel')!r} must be a loaded channel, "
                                f"one of {loaded_channels}")
            if "fraction" not in analysis:
                problems.append(f"Analysis {name!r} needs a fraction")
            if not control_wells and analysis.get("channel") not in background.get("manual", {}):
                problems.append(f"Analysis {name!r}: a well's start is its background when there is no control, so its "
                                "corrected start is about 0 and a fraction of it means nothing. It needs medium "
                                "controls or a manual blank for this channel.")
        if kind == "threshold_time" and "threshold" not in analysis:
            problems.append(f"Analysis {name!r} needs a threshold")
        if kind in ("threshold_time", "fraction_of_start") and analysis.get("direction", "up") not in ("up", "down"):
            problems.append(f"Analysis {name!r}: direction must be 'up' or 'down', got {analysis.get('direction')!r}")
        if kind in ("threshold_time", "fraction_of_start", "max_growth_rate"):
            if not analysis.get("group_traits", plots.get("group_traits")):
                problems.append(f"Analysis {name!r} needs group_traits, here or in plots")
        if kind == "max_growth_rate" and "t_start" in analysis and "t_span" in analysis:
            problems.append(f"Analysis {name!r}: use t_start or t_span, not both")
        if kind == "curves" and analysis.get("yscale", "linear") not in ("linear", "log"):
            problems.append(f"Analysis {name!r}: yscale must be 'linear' or 'log'")
        if kind == "ratio":
            for key in ("numerator", "denominator", "min_denominator"):
                if key not in analysis:
                    problems.append(f"Analysis {name!r} needs {key}")
            for key in ("numerator", "denominator"):
                if key in analysis and analysis[key] not in available:
                    problems.append(f"Analysis {name!r}: {key} {analysis[key]!r} is not among {available}")
            if name in reserved or name in available:
                problems.append(f"Analysis {name!r}: this ratio's name would overwrite an existing channel")
            available.append(name)
        if kind == "custom" and "code" not in analysis:
            problems.append(f"Analysis {name!r} needs code")
        if kind == "readthrough":
            problems += readthrough_problems(analysis, name, config, traits, blocks, loaded, info)
        for trait, wanted in analysis.get("filter", {}).items():
            if trait not in traits:
                continue
            present = {map_well_values(blocks, trait).get(w) for w in loaded} - {None}
            for value in (wanted if isinstance(wanted, list) else [wanted]):
                if value not in present:
                    problems.append(f"Analysis {name!r}: filter {trait!r} has no wells with {value!r}. "
                                    f"Values there: {sorted(present, key=str)}")
        color = analysis.get("color_trait", plots.get("color_trait"))
        if kind == "curves" and color in traits:
            left_out = set(control_wells) | set(background.get("exclude_wells", []))
            shown = [w for w in loaded if w in measured and w not in left_out]
            for trait, wanted in analysis.get("filter", {}).items():
                if trait in traits:
                    allowed = wanted if isinstance(wanted, list) else [wanted]
                    values = map_well_values(blocks, trait)
                    shown = [w for w in shown if values.get(w) in allowed]
            facet = analysis.get("facet_trait")
            facet_values = map_well_values(blocks, facet) if facet in traits else {}
            colors = map_well_values(blocks, color)
            figures = {}
            for well in shown:
                figures.setdefault(facet_values.get(well), set()).add(colors.get(well))
            most = max((len(found - {None}) for found in figures.values()), default=0)
            if most > 10:
                problems.append(f"Analysis {name!r}: {color!r} has {most} values in one figure, too many to tell apart "
                                "by color. Filter or facet the figure, or color by another trait.")
    return problems, facts


# ---------------------------------------------------------------- cells
class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, text):
        self.cells.append(("markdown", text.strip()))

    def code(self, text):
        self.cells.append(("code", text.strip()))

    def to_json(self):
        cells = []
        for i, (kind, text) in enumerate(self.cells):
            cell = {"cell_type": kind, "id": f"cell-{i:03d}", "metadata": {}, "source": text.splitlines(keepends=True)}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}


def reader_markdown(config, info):
    lines = [f"**Reader.** {info['reader_name']} ({info.get('device')}, serial {info.get('serial')}). "
             f"Started {info.get('start_time')}. File `{config['reader_file']}`, sheet `{config['sheet']}`, "
             f"parser `{info['parser']}`. Plate: {info.get('plate')}."]
    if info.get("part_of_plate"):
        lines.append(f"Part of plate read: {info['part_of_plate']}.")
    lines += ["", "| Channel | Library label | Mode | Settings | Reads | Span (h) | Interval (min) |",
              "|---|---|---|---|---:|---:|---:|"]
    wanted = {id(find_block(info, v)): k for k, v in config["measurements"].items()}
    for block in info["blocks"]:
        channel = wanted.get(id(block), "(not loaded)")
        settings = ", ".join(f"{k} {v}" for k, v in block["settings"].items())
        lines.append(f"| {channel} | `{block['library_label']}` | {block['mode']} | {settings} | {block['reads']} | "
                     f"{block['first_h']}–{block['last_h']} | {block['median_interval_min']} |")
    warnings = [f"`{b['suggested_channel']}`: non-numeric values read as missing: {b['non_numeric_values']}"
                for b in info["blocks"] if b["non_numeric_values"]]
    if warnings:
        lines += ["", "Reader values that could not be used: " + "; ".join(warnings) + "."]
    return "\n".join(lines)


def background_markdown(config):
    background = config.get("background", {})
    wells = background.get("control_wells", [])
    match = background.get("match_traits", [])
    offset = background.get("offset", {})
    parts = []
    if wells:
        scope = f", matched to each well on {', '.join(match)}" if match else ""
        parts.append(f"- Medium controls: {', '.join(wells)}{scope}. Each is checked for contamination on OD. "
                     "The clean controls give one constant per medium and channel: their mean over the run, after each control's initial jump.")
        parts.append("- A well with no clean control in its medium uses its own start: the median of 5 reads after the initial jump, valid only if the start is medium alone.")
    else:
        parts.append("- No medium controls on this plate. Each well uses its own start: the median of 5 reads after the initial jump, valid only if the start is medium alone.")
    parts.append("- The initial jump is detected per well and channel, in up to the first 3 reads, in either direction.")
    if background.get("manual"):
        parts.append(f"- Blank values given by the experimenter: `{background['manual']}`.")
    if background.get("control_decisions"):
        parts.append(f"- Experimenter's calls on flagged controls: `{background['control_decisions']}`.")
    if config.get("exclude_time_windows"):
        parts.append(f"- Reads left out of all wells, in hours: `{config['exclude_time_windows']}`.")
    if background.get("exclude_wells"):
        parts.append(f"- Wells left out: {', '.join(background['exclude_wells'])}.")
    if background.get("accept_blank_drift"):
        parts.append(f"- One constant accepted despite a drifting blank in: {', '.join(background['accept_blank_drift'])}.")
    if background.get("accept_own_start"):
        accepted = background["accept_own_start"]
        parts.append(f"- Own start accepted despite doubts for: {'all such wells' if accepted is True else ', '.join(accepted)}.")
    nonzero = {k: v for k, v in offset.items() if v}
    parts.append(f"- Offset added after subtraction: `{nonzero}`." if nonzero else "- No offset is added after subtraction (set `BACKGROUND_OFFSET` to change this).")
    return "**Background.**\n\n" + "\n".join(parts)


SETUP = r'''
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

NOTEBOOK_DIR = Path.cwd()
CODE_DIR = Path(os.environ.get("LAB_PLATE_READER_CODE", @@CODE_DIR@@))  # the plate_reader folder of lab-data-analysis
if not (CODE_DIR / "reader_functions.py").exists():
    raise FileNotFoundError(f"No plate-reader code in {CODE_DIR}. Set LAB_PLATE_READER_CODE to the plate_reader folder of your lab-data-analysis clone.")
sys.path.insert(0, str(CODE_DIR))

import reader_functions as rf
import reader_background as rb
import readthrough as rt
import util

if Path(rf.__file__).resolve().parent != CODE_DIR.resolve():
    raise ImportError(f"Loaded reader_functions from {rf.__file__}, not from {CODE_DIR}. Restart the kernel.")
print(f"Code commit {util.code_version(CODE_DIR)}. The notebook was built from commit @@BUILT_FROM@@.")

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
FIG_DIR = NOTEBOOK_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
'''

LOAD = r'''
exp = rf.ReaderExperiment(READER_FILE, PLATE_MAP, MEASUREMENTS, reader=READER_PARSER, sheet_name=SHEET)
dropped_reads = rb.trim_trailing_empty_reads(exp)

# The library's reading must agree with the separate reading made when this notebook was built.
loaded_check = rb.verify_loaded(exp).set_index("measurement")
for channel, (reads, wells) in EXPECTED_FROM_FILE.items():
    found = (int(loaded_check.loc[channel, "reads"]), int(loaded_check.loc[channel, "wells_with_data"]))
    if found != (reads, wells):
        raise ValueError(f"{channel}: the library loaded {found[0]} reads and {found[1]} wells with data, "
                         f"but the file holds {reads} reads and {wells} wells with data.")

if EXCLUDE_TIME_WINDOWS:
    print("Reads blanked in the excluded time windows:", rb.blank_time_windows(exp, EXCLUDE_TIME_WINDOWS))

display(rb.verify_loaded(exp))
print("Run started:", exp.start_time)
print("Plate-map traits:", [t for t in exp.plate_map_dict if t != "well"])
print("Wells loaded:", len(exp.wells), "| reads dropped after the last timestamp:", dropped_reads)

def _has_data(well):
    return any(np.isfinite(exp.data[(channel, well)]).any() for channel in MEASUREMENTS)


SAMPLE_WELLS = [w for w in exp.wells if w not in set(CONTROL_WELLS) | set(EXCLUDE_WELLS) and _has_data(w)]
print("Sample wells:", len(SAMPLE_WELLS), "| medium controls:", len(CONTROL_WELLS), "| left out:", len(EXCLUDE_WELLS))
'''

RAW_GRID = r'''
for channel in MEASUREMENTS:
    fig, axes = exp.plot_plate_grid(
        channel, color_by_trait=COLOR_TRAIT, annotate_well_by_trait=ANNOTATE_TRAITS, show_well_id=True,
        title=f"{channel}, raw", save_fig_path=FIG_DIR / f"{channel}_raw_plate.png",
    )
    plt.show()
'''

BACKGROUND = r'''
background = rb.compute_background(
    exp, control_wells=CONTROL_WELLS, match_traits=MATCH_TRAITS, od_measurement=OD_CHANNEL,
    measurements=list(MEASUREMENTS), control_decisions=CONTROL_DECISIONS,
    manual_background=MANUAL_BACKGROUND, exclude_wells=EXCLUDE_WELLS, accept_blank_drift=ACCEPT_BLANK_DRIFT,
    accept_own_start=ACCEPT_OWN_START,
)
CORRECTED = rb.apply_background(exp, background, offset=BACKGROUND_OFFSET)

print(rb.summarize(background))
if len(background.controls):
    display(background.controls)
if len(background.constants):
    display(background.constants)
rb.background_table(background).to_csv("background_table.csv", index=False)

for channel in MEASUREMENTS:
    rb.plot_background_qc(exp, background, channel, save_path=FIG_DIR / f"{channel}_background_qc.png")
    plt.show()
'''

CORRECTED_GRID = r'''
for channel, corrected in CORRECTED.items():
    fig, axes = exp.plot_plate_grid(
        corrected, color_by_trait=COLOR_TRAIT, annotate_well_by_trait=ANNOTATE_TRAITS, show_well_id=True,
        title=f"{channel}, background subtracted", save_fig_path=FIG_DIR / f"{channel}_corrected_plate.png",
    )
    plt.show()
'''

HELPERS = r'''
def select_wells(filters=None, wells=None):
    """Sample wells whose traits match, e.g. {"medium": "lb", "glucose": [0, 1]}."""
    out = []
    for well in (SAMPLE_WELLS if wells is None else wells):
        keep = True
        for trait, wanted in (filters or {}).items():
            wanted = wanted if isinstance(wanted, (list, tuple)) else [wanted]
            keep = keep and exp.well_to_trait_value(well, trait) in wanted
        if keep:
            out.append(well)
    return out


def trait_colors(trait, wells):
    """Fixed colors per trait value, so a value keeps its color in every figure."""
    values = sorted({exp.well_to_trait_value(w, trait) for w in SAMPLE_WELLS}, key=str)
    if len(values) > 10:  # too many for the whole plate, so color only what this figure shows
        values = sorted({exp.well_to_trait_value(w, trait) for w in wells}, key=str)
    if len(values) > 10:
        raise ValueError(f"{trait!r} has {len(values)} values in this figure, too many to tell apart by color. "
                         "Filter or facet the figure, or color by another trait.")
    return dict(zip(values, sns.color_palette("colorblind", 10)))


def channel_arrays(channel, well):
    return np.asarray(exp.data[(channel, "time")], dtype=float), np.asarray(exp.data[(channel, well)], dtype=float)


def time_to_threshold(time, y, threshold, direction="up", t_start=None):
    """Interpolated time of the first crossing at or after t_start. NaN if it never crosses."""
    valid = np.isfinite(time) & np.isfinite(y)
    if t_start is not None:
        valid &= time >= t_start
    t, v = time[valid], y[valid]
    if len(t) == 0:
        return np.nan
    crossed = v >= threshold if direction == "up" else v <= threshold
    if not crossed.any():
        return np.nan
    i = int(np.argmax(crossed))
    if i == 0 or v[i] == v[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (threshold - v[i - 1]) / (v[i] - v[i - 1]) * (t[i] - t[i - 1]))


def first_read_time(source, well):
    """Time of the first read after a well's initial jump, so that the jump can't count as a crossing."""
    time, y = channel_arrays(CORRECTED.get(source, source), well)
    times = time[np.isfinite(time) & np.isfinite(y)]
    row = background.wells[(background.wells["measurement"] == source) & (background.wells["well"] == well)]
    first = int(row["jump_reads"].iloc[0]) if len(row) and np.isfinite(row["jump_reads"].iloc[0]) else 0
    return float(times[first]) if len(times) > first else np.nan


def well_table(wells=None):
    """One row per well with its plate-map traits."""
    rows = []
    for well in (SAMPLE_WELLS if wells is None else wells):
        row = {"well": well}
        for trait in exp.plate_map_dict:
            if trait not in ("well", "well_status"):
                row[trait] = exp.well_to_trait_value(well, trait)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric(table, metric, group_traits):
    """Mean, SD and SE per group. Technical replicates are averaged within each biological replicate first."""
    data = table.dropna(subset=[metric])
    unit = "wells"
    if BIO_REP_TRAIT:
        data = data.groupby(group_traits + [BIO_REP_TRAIT], dropna=False, sort=False)[metric].mean().reset_index()
        unit = "biological replicates"
    elif TECH_REP_TRAIT:
        unit = "technical replicates"
    stats = data.groupby(group_traits, dropna=False, sort=False)[metric].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats = stats.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_sd", "se": f"{metric}_se", "count": f"n_{unit.replace(' ', '_')}"})
    missing = table[table[metric].isna()]
    if len(missing):
        print(f"{metric}: no value for {len(missing)} wells:", ", ".join(missing["well"]))
    return stats


def plot_metric(table, stats, metric, group_traits, ylabel, name):
    """Every well as a dot, with the group mean and SE in black."""
    labels = [" ".join(str(v) for v in row) for row in stats[group_traits].itertuples(index=False)]
    positions = {tuple(map(str, row)): i for i, row in enumerate(stats[group_traits].itertuples(index=False))}
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels) + 2), 4.5))
    rng = np.random.default_rng(0)
    for row in table.dropna(subset=[metric]).to_dict("records"):
        key = tuple(str(row[t]) for t in group_traits)
        ax.scatter(positions[key] + rng.uniform(-0.12, 0.12), row[metric], s=26, color="#7fa8c9", edgecolor="white", linewidth=0.4, zorder=2)
    ax.errorbar(range(len(labels)), stats[f"{metric}_mean"], yerr=stats[f"{metric}_se"], fmt="o", color="black", markersize=6, capsize=4, linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def well_axis(axes, well):
    r, c = np.argwhere(exp.wells_2d == well)[0]
    return axes[r, c]


METRICS = {}
'''


def channel_expression(channel, measurements):
    """Corrected channels are looked up from CORRECTED; a derived channel is used as it is."""
    return f"CORRECTED[{channel!r}]" if channel in measurements else repr(channel)


def analysis_cells(nb, analysis, config):
    kind, name = analysis["type"], analysis["name"]
    measurements = config["measurements"]
    title = analysis.get("title", name.replace("_", " "))
    plots = config.get("plots", {})
    default_groups = plots.get("group_traits", [])
    groups = analysis.get("group_traits", default_groups)
    note = analysis.get("note", "")

    if kind == "custom":
        nb.md(f"## {title}\n\n{note}")
        nb.code(analysis["code"])
        return

    if kind == "ratio":
        nb.md(f"## {title}\n\n`{analysis['numerator']}` / `{analysis['denominator']}`, both background subtracted. "
              f"Reads with `{analysis['denominator']}` below {analysis['min_denominator']} are left blank, because the ratio "
              f"is unstable there. {note}")
        numerator = channel_expression(analysis["numerator"], measurements)
        denominator = channel_expression(analysis["denominator"], measurements)
        nb.code(f"rb.channel_ratio(exp, {numerator}, {denominator}, {name!r}, min_denominator={analysis['min_denominator']!r})")
        return

    if kind == "readthrough":
        readthrough_cells(nb, analysis, config, title, note)
        return

    channel = channel_expression(analysis["channel"], measurements)
    if kind == "curves":
        color = analysis.get("color_trait", plots.get("color_trait"))
        facet = analysis.get("facet_trait")
        filters = analysis.get("filter", {})
        scale = analysis.get("yscale", "linear")
        reps = [t for t in (plots.get("bio_rep_trait"), plots.get("tech_rep_trait")) if t]
        reading = ("Lines are means over replicate wells, and bands are ±1 SD across them as `plot_data_static` computes it "
                   "(population SD)." if reps else "Each line is one well.")
        nb.md(f"## {title}\n\n{reading} {note}")
        extra = ""
        if "ylim" in analysis:
            extra += f", ylim={tuple(analysis['ylim'])!r}"
        if "t_span" in analysis:
            extra += f", t_span={tuple(analysis['t_span'])!r}"
        color_arg = f"({color!r}, trait_colors({color!r}, wells))" if color else "None"
        suffix = ", background subtracted" if analysis["channel"] in measurements else ""
        ylabel = analysis.get("ylabel", f"{analysis['channel']}{suffix}")
        rep_arg = py(reps) if reps else "None"
        if facet:
            nb.code(f'''
ylabel = {ylabel!r}
for value in sorted({{exp.well_to_trait_value(w, {facet!r}) for w in select_wells({filters!r})}}, key=str):
    wells = select_wells({{**{filters!r}, {facet!r}: value}})
    fig, ax = exp.plot_data_static(
        measurement={channel}, wells=wells, trait_color_map={color_arg}, rep_trait={rep_arg},
        yscale={scale!r}, title={title!r} + ", " + {facet!r} + " = " + str(value){extra},
    )
    ax.set_ylabel(ylabel, fontsize=12)
    fig.savefig(FIG_DIR / ({name!r} + "_" + str(value).replace("/", "_") + ".png"), dpi=200, bbox_inches="tight")
    plt.show()
''')
        else:
            nb.code(f'''
wells = select_wells({filters!r})
fig, ax = exp.plot_data_static(
    measurement={channel}, wells=wells, trait_color_map={color_arg}, rep_trait={rep_arg},
    yscale={scale!r}, title={title!r}{extra},
)
ax.set_ylabel({ylabel!r}, fontsize=12)
fig.savefig(FIG_DIR / {name + ".png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
        return

    filters = analysis.get("filter", {})
    t_start = analysis.get("t_start", 0.0)
    if kind == "threshold_time":
        threshold, direction = analysis["threshold"], analysis.get("direction", "up")
        relative = bool(t_start)
        measure = f"time from {t_start} h" if relative else "time"
        nb.md(f"## {title}\n\nInterpolated {measure} until `{analysis['channel']}` first goes {direction} through {threshold}, on background-subtracted data, looking only after each well's initial jump. {note}")
        metric = f"{name}_h"
        nb.code(f'''
table = well_table(select_wells({filters!r}))
channel = {channel}
table[{metric!r}] = [time_to_threshold(*channel_arrays(channel, w), {threshold!r}, direction={direction!r},
                                       t_start=max({t_start!r}, first_read_time({analysis["channel"]!r}, w))) - {t_start!r}
                     for w in table["well"]]
table.to_csv({name + ".csv"!r}, index=False)
stats = summarize_metric(table, {metric!r}, {py(groups)})
stats.to_csv({name + "_summary.csv"!r}, index=False)
METRICS[{name!r}] = table
display(stats)
plot_metric(table, stats, {metric!r}, {py(groups)}, {("Time [h]")!r}, {name!r})
''')
        nb.code(f'''
fig, axes = exp.plot_plate_grid(channel, color_by_trait=COLOR_TRAIT, show_well_id=True, title={title!r})
for well, value in zip(table["well"], table[{metric!r}]):
    if np.isfinite(value):
        well_axis(axes, well).scatter([value + {t_start!r}], [{threshold!r}], s=14, color="black", zorder=5)
for ax in axes.flat:
    ax.axhline({threshold!r}, color="0.6", linestyle=":", linewidth=0.6)
fig.savefig(FIG_DIR / {name + "_plate.png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
        return

    if kind == "fraction_of_start":
        fraction = analysis["fraction"]
        source = analysis["channel"]
        direction = analysis.get("direction", "down")
        start_reads = analysis.get("start_reads", 2)
        start_skip = analysis.get("start_skip")
        min_start = analysis.get("min_start", 0.02 if source == config.get("od_channel", "OD") else 0.0)
        rule = f"the first {start_skip} reads dropped" if start_skip is not None else "the initial jump dropped where it can be detected"
        nb.md(f"## {title}\n\nInterpolated time from the start of the run until `{source}` reaches {fraction} × its starting value. "
              f"The start is the mean of {start_reads} reads, background subtracted, with {rule}. In wells already changing fast the "
              f"reads after a jump don't agree, so the jump can't be detected and the first read stays in. The second table shows how much the choice matters. "
              f"Wells get no value if their start is below {min_start}, or if their background is their own start, which makes the corrected start about 0. {note}")
        metric = f"{name}_h"
        nb.code(f'''
source_wells = background.wells[background.wells["measurement"] == {source!r}].set_index("well")
jumps, methods = source_wells["jump_reads"], source_wells["method"]
table = well_table(select_wells({filters!r}))
channel = {channel}
START_READS, START_SKIP, MIN_START = {start_reads!r}, {start_skip!r}, {min_start!r}


def reads_to_skip(well, skip):
    if skip is not None:
        return int(skip)
    return 0 if pd.isna(jumps[well]) else int(jumps[well])


def corrected_start(well, skip=START_SKIP):
    y = channel_arrays(channel, well)[1]
    y = y[np.isfinite(y)]
    first = reads_to_skip(well, skip)
    if len(y) < first + START_READS:
        return np.nan
    return float(np.mean(y[first:first + START_READS]))


def usable_start(well, skip=START_SKIP):
    start = corrected_start(well, skip)
    return methods[well] != "well_start" and np.isfinite(start) and start >= MIN_START


def crossing_time(well, skip=START_SKIP):
    if not usable_start(well, skip):
        return np.nan
    time, y = channel_arrays(channel, well)
    times = time[np.isfinite(time) & np.isfinite(y)]
    return time_to_threshold(time, y, {fraction!r} * corrected_start(well, skip), direction={direction!r},
                             t_start=times[reads_to_skip(well, skip)])


table["start_corrected"] = [corrected_start(w) for w in table["well"]]
table[{metric!r}] = [crossing_time(w) for w in table["well"]]
no_start = [w for w in table["well"] if not usable_start(w)]
if no_start:
    print(f"No value for {{len(no_start)}} wells, because their start can't be used:", ", ".join(no_start))
table.to_csv({name + ".csv"!r}, index=False)
stats = summarize_metric(table, {metric!r}, {py(groups)})
stats.to_csv({name + "_summary.csv"!r}, index=False)
METRICS[{name!r}] = table
display(stats)
plot_metric(table, stats, {metric!r}, {py(groups)}, {("Time [h]")!r}, {name!r})

too_fast = sum(not rb.jump_assessable(exp.data[({source!r}, w)], {source!r} == OD_CHANNEL) for w in table["well"])
print(f"A jump was found in {{int((jumps[table['well']] > 0).sum())}} of {{len(table)}} wells. In {{too_fast}} wells the reads after "
      "the first were changing too fast to tell, so finding no jump there doesn't mean the first read is sound.")
sensitivity = table[{py(groups)} + ["well"]].copy()
for skip in (0, 1, 2):
    sensitivity[f"drop_{{skip}}_reads_h"] = [crossing_time(w, skip) for w in table["well"]]
sensitivity = sensitivity.groupby({py(groups)}, dropna=False, sort=False)[[f"drop_{{k}}_reads_h" for k in (0, 1, 2)]].mean().reset_index()
sensitivity.to_csv({name + "_start_sensitivity.csv"!r}, index=False)
display(sensitivity)
''')
        nb.code(f'''
fig, axes = exp.plot_plate_grid(channel, color_by_trait=COLOR_TRAIT, show_well_id=True, title={title!r})
for well, value, start in zip(table["well"], table[{metric!r}], table["start_corrected"]):
    if np.isfinite(value):
        well_axis(axes, well).scatter([value], [{fraction!r} * start], s=14, color="black", zorder=5)
fig.savefig(FIG_DIR / {name + "_plate.png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
        return

    if kind == "max_growth_rate":
        options = analysis.get("options", {})
        span = tuple(analysis.get("t_span", (t_start or None, None)))
        nb.md(f"## {title}\n\nMaximum slope of log(`{analysis['channel']}`), from `ReaderExperiment.estimate_max_log_growth` on background-subtracted data. Options: `{options or 'library defaults'}`. {note}")
        metric = f"{name}_per_h"
        nb.code(f'''
channel = {channel}
table = well_table(select_wells({filters!r}))
fits = {{w: rf.ReaderExperiment.estimate_max_log_growth(*channel_arrays(channel, w), t_span={span!r}, **{options!r}) for w in table["well"]}}
table[{metric!r}] = [fits[w]["mu"] for w in table["well"]]
table["doubling_time_min"] = np.log(2) / table[{metric!r}] * 60
table["fit_r2"] = [fits[w]["r2"] for w in table["well"]]
table["fit_time_h"] = [fits[w]["t0"] for w in table["well"]]
table.to_csv({name + ".csv"!r}, index=False)
stats = summarize_metric(table, {metric!r}, {py(groups)})
stats.to_csv({name + "_summary.csv"!r}, index=False)
METRICS[{name!r}] = table
display(stats)
plot_metric(table, stats, {metric!r}, {py(groups)}, "Max growth rate [1/h]", {name!r})
''')
        nb.code(f'''
fig, axes = exp.plot_plate_grid(channel, yscale="log", ylim=(1e-3, None), color_by_trait=COLOR_TRAIT, show_well_id=True, title={title!r})
for well, fit in fits.items():
    if np.isfinite(fit["mu"]):
        t = np.array([fit["t0"] - 0.75, fit["t0"] + 0.75])
        well_axis(axes, well).plot(t, fit["y0"] * np.exp(fit["mu"] * (t - fit["t0"])), color="black", linewidth=0.9)
fig.savefig(FIG_DIR / {name + "_plate.png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
        return


def readthrough_cells(nb, analysis, config, title, note):
    name = analysis["name"]
    time_h, window_h = analysis["time_h"], analysis.get("window_h", 0.25)
    expected = analysis.get("expected_percent", [0.1, 10.0])
    step = analysis.get("time_course_step_h", 1.0)
    stability = analysis.get("stability_times_h")
    condition = analysis["condition_trait"]
    design = {key: analysis[key] for key in ("red", "green", "reporter_trait", "in_frame", "autofluorescence", "test", "condition_trait", "replicate_trait")}
    design["od"] = analysis.get("od", config.get("od_channel", "OD" if "OD" in config["measurements"] else None))
    nb.md(f"## {title}\n\nMethod of [Hinnu et al. 2024](https://doi.org/10.1038/s41598-024-57597-8), on medium-subtracted signals, at {time_h} h "
          f"(mean of the reads within {window_h} h). In each condition, the mean red of the `{analysis['autofluorescence']}` wells is subtracted from the red of "
          f"every reporter well, and each well's red:green ratio (R:G) is taken. Then\n\n$$\\text{{readthrough (\\%)}} = 100 \\times \\frac{{\\text{{mean R:G of test reporter}}}}{{\\text{{mean R:G of intact reporter}}}}$$\n\n"
          f"The paper doesn't normalize to OD or subtract green autofluorescence, so neither does this. Each biological replicate is a separate culture, so the standard error propagates the "
          f"between-replicate errors of both means and of the autofluorescence. {note}")
    nb.code(f'''
RT_DESIGN = rt.ReadthroughDesign(**{py(design)})
RT_TIME_H, RT_WINDOW_H, RT_EXPECTED = {time_h!r}, {window_h!r}, {tuple(expected)!r}
rt_wells = rt.reporter_table(exp, CORRECTED, RT_DESIGN, SAMPLE_WELLS, RT_TIME_H, RT_WINDOW_H)
rt_summary, rt_replicates = rt.readthrough(rt_wells, RT_DESIGN)
rt.add_ratios(rt_wells, RT_DESIGN).to_csv({name + "_wells.csv"!r}, index=False)
rt_summary.to_csv({name + ".csv"!r}, index=False)
rt_replicates.to_csv({name + "_replicates.csv"!r}, index=False)
display(rt_summary[["reporter", "condition", "readthrough_percent", "se_percent", "sd_replicates_percent", "n_test", "n_in_frame", "n_autofluorescence", "test_ratio", "in_frame_ratio", "red_autofluorescence"]].round(4))
display(rt_replicates.round(4))
''')
    nb.code(f'''
fig, ax = rt.plot_readthrough(rt_summary, rt_replicates, {condition!r}, expected_percent=RT_EXPECTED, title={title!r} + f", {{RT_TIME_H}} h")
fig.savefig(FIG_DIR / {name + ".png"!r}, dpi=200, bbox_inches="tight")
plt.show()
fig, axes = rt.plot_ratio_panels(rt_wells, RT_DESIGN, {condition!r}, title=f"Signals behind the ratio, {{RT_TIME_H}} h")
fig.savefig(FIG_DIR / {name + "_signals.png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
    nb.md("### Over the run\n\nThe same calculation at each time. Wells below OD 0.1 are left out, because their signals are too close to the medium.")
    nb.code(f'''
last_h = min(np.nanmax(exp.data[(CORRECTED[c], "time")]) for c in (RT_DESIGN.red, RT_DESIGN.green))
rt_course = rt.time_course(exp, CORRECTED, RT_DESIGN, SAMPLE_WELLS, np.arange({step!r}, last_h, {step!r}), RT_WINDOW_H)
rt_course.to_csv({name + "_time_course.csv"!r}, index=False)
fig, axes = rt.plot_time_course(rt_course, {condition!r}, chosen_time=RT_TIME_H, title="Readthrough over the run")
fig.savefig(FIG_DIR / {name + "_time_course.png"!r}, dpi=200, bbox_inches="tight")
plt.show()
''')
    nb.md("### Sanity tests\n\n`FAIL` means don't interpret the number until it is explained. `WARN` means look at it and report it.")
    stability_code = py(stability) if stability else f"[t for t in (RT_TIME_H - 4, RT_TIME_H, RT_TIME_H + 6) if t < last_h]"
    nb.code(f'''
rt_sanity = rt.sanity_checks(
    exp, CORRECTED, background, RT_DESIGN, SAMPLE_WELLS, RT_TIME_H, RT_WINDOW_H, expected_percent=RT_EXPECTED,
    control_wells=[w for w in CONTROL_WELLS if w not in EXCLUDE_WELLS], measurements=MEASUREMENTS, reader_file=READER_FILE, sheet=SHEET,
    stability_times={stability_code},
)
rt_sanity.to_csv({name + "_sanity.csv"!r}, index=False)
with pd.option_context("display.max_colwidth", None):
    display(rt_sanity)
print("Sanity tests:", rt_sanity["status"].value_counts().to_dict())
for row in rt_sanity[rt_sanity["status"] != "PASS"].itertuples():
    print(f"{{row.status}}: {{row.check}}. {{row.detail}}")
''')


def built_from(code_dir):
    def git(*args):
        return subprocess.run(["git", "-C", str(code_dir), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        return commit + (" with uncommitted changes" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def build(config, facts):
    info = facts["info"]
    background = config.get("background", {})
    plots = config.get("plots", {})
    nb = Notebook()

    nb.md(f"# {config['title']}\n\n**Purpose.** {config['purpose']}\n\n{reader_markdown(config, info)}\n\n"
          f"**Plate layout.** Map: `{config['plate_map']}`. {config['layout_description']}\n\n{background_markdown(config)}"
          + (f"\n\n**Lab notebook.** [[{config['zettelkasten_note']}]]" if config.get("zettelkasten_note") else ""))
    code_dir = Path(__file__).resolve().parents[1]
    recorded = facts.get("record_code_dir") or str(code_dir)
    nb.code(SETUP.replace("@@CODE_DIR@@", repr(recorded)).replace("@@BUILT_FROM@@", built_from(code_dir)))
    nb.md("## Settings")
    offsets = {channel: background.get("offset", {}).get(channel, 0.0) for channel in config["measurements"]}
    annotate = plots.get("annotate_traits")
    nb.code("\n".join([
        f"READER_FILE = {config['reader_file']!r}",
        f"SHEET = {config['sheet']!r}",
        f"PLATE_MAP = {config['plate_map']!r}",
        f"READER_PARSER = {info['parser']!r}  # {info['reader_name']}, serial {info.get('serial')}",
        f"MEASUREMENTS = {py(config['measurements'])}",
        f"OD_CHANNEL = {facts['od_channel']!r}",
        f"EXPECTED_FROM_FILE = {py(facts.get('expected', {}))}  # channel: [reads, wells with data], read separately from the library",
        f"EXCLUDE_TIME_WINDOWS = {py(config.get('exclude_time_windows', []))}  # hours; reads set to NaN in all wells",
        "",
        "# Background",
        f"CONTROL_WELLS = {py(background.get('control_wells', []))}",
        f"MATCH_TRAITS = {py(background.get('match_traits', []))}",
        f"CONTROL_DECISIONS = {py(background.get('control_decisions', {}))}  # the experimenter's call on flagged controls",
        f"MANUAL_BACKGROUND = {py(background.get('manual', {}))}  # blank values given by the experimenter",
        f"EXCLUDE_WELLS = {py(background.get('exclude_wells', []))}",
        f"ACCEPT_BLANK_DRIFT = {py(background.get('accept_blank_drift', []))}  # channels where one constant is accepted despite a drifting blank",
        f"ACCEPT_OWN_START = {py(background.get('accept_own_start', []))}  # wells whose own start is accepted despite doubts",
        f"BACKGROUND_OFFSET = {py(offsets)}  # added after subtraction, per channel; 0 means plain subtraction",
        "",
        "# Plots and statistics",
        f"COLOR_TRAIT = {plots.get('color_trait')!r}",
        f"ANNOTATE_TRAITS = {tuple(annotate) if annotate else None!r}",
        f"BIO_REP_TRAIT = {plots.get('bio_rep_trait')!r}",
        f"TECH_REP_TRAIT = {plots.get('tech_rep_trait')!r}",
    ]))
    nb.md("## Load and verify")
    nb.code(LOAD)
    nb.md("## Raw traces on the plate")
    nb.code(RAW_GRID)
    nb.md("## Medium background\n\nThe panels show the medium controls with the constant subtracted, the first reads of each "
          "sample well after subtraction (the jump shows at read 0), and the value subtracted in each well. The text lists any flagged wells.")
    nb.code(BACKGROUND)
    nb.md("## Background-subtracted traces on the plate")
    nb.code(CORRECTED_GRID)
    if config.get("analyses"):
        nb.md("## Analysis helpers")
        nb.code(HELPERS)
        for analysis in config["analyses"]:
            analysis_cells(nb, analysis, config)
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--record-code-dir", default=None,
                        help="The plate_reader path to write into the notebook, when this script runs somewhere other than where "
                             "the notebook will later be run, such as a VM mounting the Mac's folders.")
    args = parser.parse_args()
    folder = args.config.resolve().parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    problems, facts = check_config(config, folder)
    facts["record_code_dir"] = args.record_code_dir
    if problems:
        print("Config problems. No notebook written:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    notebook_name = config.get("notebook", f"{folder.name}_analysis.ipynb")
    output = folder / notebook_name
    if output.exists() and not config.get("overwrite_notebook", False):
        sys.exit(f"{output} already exists. Set overwrite_notebook to true only if it can be replaced.")
    output.write_text(json.dumps(build(config, facts).to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Reader: {facts['info']['reader_name']}. Channels: {list(config['measurements'])}. Loaded wells: {len(facts['loaded_wells'])}.")
    for warning in facts.get("warnings", []):
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()

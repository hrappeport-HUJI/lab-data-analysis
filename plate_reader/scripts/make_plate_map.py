#!/usr/bin/env python3
"""Write a plate map in the format reader_functions.ReaderExperiment reads, from a layout JSON.

Usage:
    python make_plate_map.py LAYOUT.json OUT_plate_map.xlsx [--markdown OUT.md]
    python make_plate_map.py --from-map EXISTING_plate_map.xlsx [--markdown OUT.md]

The second form writes nothing to the map: it only prints the grid of a map someone supplied.

Layout JSON (ranges are rectangles like "B2:D4", single wells like "E10", or lists of these):

    {
      "plate": {"rows": 8, "cols": 12},
      "well_status": {
        "empty": ["A1:A12", "H1:H12", "B1:G1", "B12:G12"],
        "medium_control": ["B11:G11"]
      },
      "traits": {
        "strain":   {"KLY": ["B2:G10"]},
        "medium":   {"LB": ["B2:D11"], "M9": ["E2:G11"]},
        "dilution": {"1000": ["B2:G5"], "10000": ["B6:G10"]},
        "tech_rep": {"1": ["B2:B10", "E2:E10"], "2": ["C2:C10", "F2:F10"], "3": ["D2:D10", "G2:G10"]}
      },
      "match_traits": ["medium"]
    }

Number-like values become numbers, as in the existing plate maps. "empty" covers wells with no
culture, including NaCl moat wells; the library doesn't load them. Sample wells must have a value
for every trait. Medium-control wells need one only for "match_traits". A well assigned twice
within a trait is an error, because that means the description contradicts itself.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import openpyxl
except ImportError:
    sys.exit("make_plate_map.py needs openpyxl: pip install openpyxl")

WELL = re.compile(r"^([A-Z])(\d{1,2})$")
STATUS_VALUES = ("empty", "medium_control")


class LayoutError(ValueError):
    pass


def _coerce(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    try:
        if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text):
            return float(text)
    except ValueError:
        pass
    return text


def expand(spec, n_rows, n_cols):
    """Wells named by a range, a single well, or a list of these."""
    if isinstance(spec, (list, tuple)):
        wells = []
        for item in spec:
            wells.extend(expand(item, n_rows, n_cols))
        return wells
    parts = str(spec).strip().upper().split(":")
    if len(parts) > 2:
        raise LayoutError(f"Bad well range: {spec!r}")
    corners = []
    for part in parts:
        match = WELL.match(part)
        if not match:
            raise LayoutError(f"Bad well name in {spec!r}")
        row, col = ord(match.group(1)) - ord("A"), int(match.group(2)) - 1
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise LayoutError(f"{part} is outside the {n_rows} x {n_cols} plate")
        corners.append((row, col))
    (r0, c0), (r1, c1) = corners[0], corners[-1]
    if r1 < r0 or c1 < c0:
        raise LayoutError(f"Range {spec!r} runs backwards")
    return [f"{chr(ord('A') + r)}{c + 1}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def build_blocks(layout):
    """Validate the layout and return (n_rows, n_cols, {block name: {well: value}})."""
    n_rows, n_cols = layout["plate"]["rows"], layout["plate"]["cols"]
    all_wells = [f"{chr(ord('A') + r)}{c + 1}" for r in range(n_rows) for c in range(n_cols)]

    status = {}
    for value, spec in layout.get("well_status", {}).items():
        if value not in STATUS_VALUES:
            raise LayoutError(f"well_status must be one of {STATUS_VALUES}, got {value!r}")
        for well in expand(spec, n_rows, n_cols):
            if well in status:
                raise LayoutError(f"{well} has two well_status values")
            status[well] = value

    traits = {}
    for raw_name, assignments in layout.get("traits", {}).items():
        name = raw_name.strip()
        if name in ("well", "well_status"):
            raise LayoutError(f"{name!r} is reserved")
        if name in traits:
            raise LayoutError(f"Trait {name!r} appears twice")
        cells = {}
        for raw_value, spec in assignments.items():
            value = _coerce(raw_value)
            for well in expand(spec, n_rows, n_cols):
                if well in cells:
                    raise LayoutError(f"{well} is assigned both {cells[well]!r} and {value!r} in trait {name!r}")
                if status.get(well) == "empty":
                    raise LayoutError(f"{well} is marked empty but has {name}={value!r}")
                cells[well] = value
        traits[name] = cells

    match_traits = layout.get("match_traits", [])
    for trait in match_traits:
        if trait not in traits:
            raise LayoutError(f"match_traits names {trait!r}, which is not a trait")
    problems = []
    for name, cells in traits.items():
        for well in all_wells:
            state = status.get(well)
            if state == "empty" or well in cells:
                continue
            if state == "medium_control" and name not in match_traits:
                continue
            problems.append((name, well))
    if problems:
        by_trait = {}
        for name, well in problems:
            by_trait.setdefault(name, []).append(well)
        details = "; ".join(f"{name}: {','.join(wells)}" for name, wells in by_trait.items())
        raise LayoutError(f"Wells without a value (mark them empty or assign them). {details}")

    blocks = {"well": {w: w for w in all_wells}}
    if status:
        blocks["well_status"] = status
    blocks.update(traits)
    return n_rows, n_cols, blocks


def write_plate_map(path, n_rows, n_cols, blocks):
    """The library reads Sheet1 with its first row as a header, so that row stays blank."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    row = 2
    for index, (name, cells) in enumerate(blocks.items()):
        sheet.cell(row=row, column=1, value=name)
        for r in range(n_rows):
            for c in range(n_cols):
                value = cells.get(f"{chr(ord('A') + r)}{c + 1}")
                if value is not None:
                    sheet.cell(row=row + 1 + r, column=c + 1, value=value)
        row += n_rows + 2
    workbook.save(path)


def read_back(path):
    """Read the file the way get_plate_map_dict does, to prove it round-trips."""
    workbook = openpyxl.load_workbook(path, data_only=True)
    rows = [list(r) for r in workbook["Sheet1"].iter_rows(values_only=True)]
    if any(v is not None for v in rows[0]):
        raise LayoutError("The first row must be blank")
    return rows


def blocks_from_map(path):
    """Read an existing plate map into the same structure that build_blocks returns."""
    from build_notebook import library_geometry_problem, read_plate_map

    problem = library_geometry_problem(path)
    if problem:
        raise LayoutError(problem)
    grids, error = read_plate_map(path)
    if error:
        raise LayoutError(error)
    wells = grids["well"]
    n_rows = len(wells)
    n_cols = max(c + 1 for row in wells for c, well in enumerate(row) if well is not None)
    blocks = {}
    for name, grid in grids.items():
        cells = {}
        for r in range(n_rows):
            for c in range(n_cols):
                value = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
                if value is not None:
                    cells[f"{chr(ord('A') + r)}{c + 1}"] = value
        blocks[str(name).strip() if name != "well_status" else name] = cells
    return n_rows, n_cols, blocks


def preview(n_rows, n_cols, blocks):
    width = {}
    lines = []
    for name, cells in blocks.items():
        if name == "well":
            continue
        grid = [[str(cells.get(f"{chr(ord('A') + r)}{c + 1}", ".")) for c in range(n_cols)] for r in range(n_rows)]
        width = max(len(v) for row in grid for v in row)
        lines.append(f"{name}:")
        lines.append("    " + " ".join(str(c + 1).rjust(width) for c in range(n_cols)))
        for r in range(n_rows):
            lines.append(f"  {chr(ord('A') + r)} " + " ".join(v.rjust(width) for v in grid[r]))
        lines.append("")
    return "\n".join(lines)


def markdown_table(n_rows, n_cols, blocks):
    """A plate-map table in the shape the Zettelkasten's experiment notes use."""
    traits = [name for name in blocks if name != "well"]
    header = "| Row / Column | " + " | ".join(str(c + 1) for c in range(n_cols)) + " |"
    rule = "| --- | " + " | ".join("---" for _ in range(n_cols)) + " |"
    lines = [f"Plate map ({n_rows} × {n_cols}):", "", header, rule]
    for r in range(n_rows):
        cells = []
        for c in range(n_cols):
            well = f"{chr(ord('A') + r)}{c + 1}"
            state = blocks.get("well_status", {}).get(well)
            values = [str(blocks[name][well]) for name in traits if name != "well_status" and well in blocks[name]]
            if state == "medium_control":
                values = ["Blank"] + values
            if state == "empty":
                values = []
            cells.append(" ".join(values))
        lines.append(f"| {chr(ord('A') + r)} | " + " | ".join(cells) + " |")
    trait_names = [t for t in traits if t != "well_status"]
    lines += ["", "Cells list: " + ", ".join(trait_names) + ". `Blank`: medium control. An empty cell holds no culture."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layout", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--from-map", type=Path, help="print the grid of an existing plate map instead of writing one")
    parser.add_argument("--markdown", type=Path, help="also write a Markdown plate-map table")
    args = parser.parse_args()
    if bool(args.from_map) == bool(args.layout and args.output):
        parser.error("give either LAYOUT.json and OUT_plate_map.xlsx, or --from-map")
    try:
        if args.from_map:
            n_rows, n_cols, blocks = blocks_from_map(args.from_map)
        else:
            layout = json.loads(args.layout.read_text(encoding="utf-8"))
            n_rows, n_cols, blocks = build_blocks(layout)
    except LayoutError as error:
        sys.exit(f"Layout problem: {error}")
    if args.output:
        write_plate_map(args.output, n_rows, n_cols, blocks)
        read_back(args.output)
    print(preview(n_rows, n_cols, blocks))
    if args.output:
        print(f"Wrote {args.output}")
    if args.markdown:
        args.markdown.write_text(markdown_table(n_rows, n_cols, blocks) + "\n", encoding="utf-8")
        print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()

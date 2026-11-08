"""Group reviewed ScanLag results into conditions, write per-condition tables and histograms.

    python scripts/scanlag_analysis.py <results folder or experiment name> [--layout conditions.csv]

The results folder is the one written by scanlag_pipeline.py. A bare experiment name (e.g. EPEC260224)
is looked up in <project root>/results when no such folder exists in the working directory.

The layout says which plates belong to which condition, one row per (condition, scanner):

    condition,scanner,plates
    CTRL,1-4.1u,"1,2,3"
    CTRL,1-5.1u,1-3
    AMP5,1-4.1u,"4,5,6"
    AMP5,1-5.1u,4-6

By default it is <results folder>/conditions.csv. If that file doesn't exist, a template listing
every scanner and its plates is written there to fill in. The layout can also be given on the
command line instead, one --condition per condition:

    --condition "CTRL=1-4.1u:1-3;1-5.1u:1-3" --condition "AMP5=1-4.1u:4-6;1-5.1u:4-6"

Only colonies included in the review are used. Output (<results folder>/analysis/ by default):

  <condition>/colonies.csv     one row per colony: appearance, radial growth rate, doubling time
  <condition>/area_tracks.csv  area per colony and frame (long format, with time)
  <condition>/area_matrix.csv  one row per colony, one column per frame
  all_colonies.csv, summary.csv (per condition: counts, mean/median/std of each metric)
  histograms.html              interactive histograms (zoom, pan, hover), open in a browser

Growth (--growth):
  radial    slope of a linear fit of sqrt(area) against time over the colony's frames --fit-frames
            (counted from its appearance, 1 = appearance frame; default 1-12), stopping before it
            merges with another colony
  doubling  time for the area to grow from 20 to 80 px (radius doubles; MATLAB's growth time),
            range set with --doubling-range
  both      both histograms (the tables always hold both)

Merged colonies (--merged-growth):
  exclude   (default, as MATLAB) both growth metrics stop at the colony's first merge
  split     also use the merged frames, where the footprint tracker gives each colony its share
            of the merged blob (plates tracked with --tracker seed fall back to exclude)

Bins: appearance and doubling-time histograms use the scan interval as bin width (--bin to change,
in minutes); the radial growth histogram uses 0.05 px^0.5/h (--rate-bin). The x axes run from 0 to 1.1 x the
largest value. The page opens in the default browser (--no-open to skip).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from scanlag.analysis.conditions import (FitConfig, area_matrix, available_plates, collect, expand_layout,
                                         layout_template, read_layout, settings, summary)
from scanlag.analysis.plots import histograms, write_html
from scanlag.cli import split_prompt_args
from scanlag.store import REGISTRY_FILE, RESULTS_ROOT, ResultsStore


def layout_from_args(specs: list[str]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        name, _, rest = spec.partition("=")
        if not name.strip() or not rest:
            raise ValueError(f"--condition {spec!r}: expected NAME=scanner:plates[;scanner:plates]")
        for part in rest.split(";"):
            scanner, _, plates = part.rpartition(":")
            if not scanner:
                raise ValueError(f"--condition {spec!r}: expected scanner:plates, got {part!r}")
            rows.append({"condition": name.strip(), "scanner": scanner.strip(), "plates": plates})
    return pd.DataFrame(rows)


def folder_name(condition: str) -> str:
    """The output folder of a condition: its name with characters not allowed in paths replaced by _."""
    return re.sub(r'[<>:"/\\|?*\s]+', "_", condition).strip("_") or "condition"


def check_folder_names(conditions) -> None:
    """Raise ValueError if two conditions would write to the same folder (e.g. "A B" and "A/B")."""
    by_folder: dict[str, list[str]] = {}
    for c in dict.fromkeys(conditions):
        by_folder.setdefault(folder_name(c), []).append(c)
    if clashes := [names for names in by_folder.values() if len(names) > 1]:
        raise ValueError("conditions that would share an output folder, rename one of each: "
                         + "; ".join(" / ".join(repr(n) for n in names) for names in clashes))


def find_results(path: Path) -> Path:
    """``path`` if it is a folder, else <project root>/results/<path> when ``path`` is a bare name."""
    if not path.is_dir() and len(path.parts) == 1 and (RESULTS_ROOT / path).is_dir():
        return RESULTS_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path,
                        help="results folder written by scanlag_pipeline.py, or an experiment name in <project root>/results")
    parser.add_argument("--layout", type=Path, help="condition layout CSV (default <results>/conditions.csv)")
    parser.add_argument("--condition", action="append", default=[], help="NAME=scanner:plates[;scanner:plates]")
    parser.add_argument("--out", type=Path, help="output folder (default <results>/analysis)")
    parser.add_argument("--growth", choices=["radial", "doubling", "both"], default="radial",
                        help="growth histogram(s) to draw, see above (default radial)")
    parser.add_argument("--fit-frames", default="1-12",
                        help="frames of each colony used for the sqrt(area) fit, from appearance (default 1-12)")
    parser.add_argument("--doubling-range", type=float, nargs=2, default=[FitConfig.lb, FitConfig.ub],
                        metavar=("LB", "UB"), help="area range for the doubling time, px (default 20 80)")
    parser.add_argument("--bin", type=float, help="bin width of the time histograms, min (default: scan interval)")
    parser.add_argument("--rate-bin", type=float, default=0.05,
                        help="bin width of the radial growth histogram, px^0.5/h (default 0.05)")
    parser.add_argument("--merged-growth", choices=["exclude", "split"], default="exclude",
                        help="use the frames where a colony is merged with another for growth (default exclude)")
    parser.add_argument("--no-open", action="store_true", help="don't open the histograms in the browser")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))

    results = find_results(args.results).resolve()
    if not results.is_dir():
        sys.exit(f"{results} is not a folder")
    registry = results / REGISTRY_FILE
    raw_dirs = json.loads(registry.read_text())["raw_dirs"] if registry.exists() else []
    print("plates found: " + "; ".join(f"{s}: {','.join(map(str, p))}" for s, p in available_plates(results).items()))

    if args.condition:
        try:
            layout = expand_layout(layout_from_args(args.condition))
        except ValueError as e:
            sys.exit(str(e))
    else:
        path = args.layout or results / "conditions.csv"
        if not path.exists():
            layout_template(results).to_csv(path, index=False)
            sys.exit(f"no condition layout yet: wrote a template to {path}\n"
                     "fill in the condition column (split a scanner's plates over several rows if needed) "
                     "and run again")
        try:
            layout = read_layout(path)
        except ValueError as e:
            sys.exit(str(e))
        print(f"layout: {path}")
    try:
        check_folder_names(layout.condition)
    except ValueError as e:
        sys.exit(str(e))

    first, _, last = args.fit_frames.partition("-")
    lb, ub = args.doubling_range
    if not (first.strip().isdigit() and last.strip().isdigit()) or not 1 <= int(first) < int(last) or not 0 < lb < ub:
        sys.exit("--fit-frames takes FIRST-LAST (e.g. 1-12) and --doubling-range LB UB with 0 < LB < UB")
    cfg = FitConfig(fit_first=int(first), fit_last=int(last), lb=lb, ub=ub, merged_growth=args.merged_growth)
    data = collect(results, layout, cfg)
    for m in data.missing:
        print(f"warning: no results for {m} (empty slot or not processed), skipped")

    out = ResultsStore(args.out or results / "analysis", raw_dirs)
    for cond in data.conditions:
        col = data.colonies[data.colonies.condition == cond]
        tr = data.tracks[data.tracks.condition == cond]
        sub = folder_name(cond)
        out.write_table(col, sub, "colonies.csv")
        out.write_table(tr, sub, "area_tracks.csv")
        out.write_table(area_matrix(tr) if len(tr) else tr, sub, "area_matrix.csv")
        plates = col[["scanner", "plate"]].drop_duplicates()
        print(f"{cond}: {len(col)} colonies from {len(plates)} plates "
              f"({', '.join(f'{s}/{p}' for s, p in plates.itertuples(index=False))})")
    out.write_table(data.colonies, "all_colonies.csv")
    stats = summary(data.colonies, data.conditions)
    out.write_table(stats, "summary.csv")
    out.write_json(settings(cfg, layout), "settings.json")  # with the layout used

    frame_step = data.tracks.groupby("scanner").time_min.apply(lambda t: pd.Series(t.unique()).sort_values().diff().median())
    step = float(frame_step.median()) if frame_step.notna().any() else None
    time_bin = args.bin or step
    print(f"time bin width {time_bin:g} min" if time_bin else "time bin width auto")
    figures = [histograms(data.colonies, data.conditions, "appearance_min", "Appearance time",
                          "appearance time (min)", "min", width=time_bin)]
    if args.growth in ("radial", "both"):
        figures.append(histograms(data.colonies, data.conditions, "radial_growth_sqrtpx_per_h",
                                  f"Radial growth rate (sqrt(area) fit over frames {cfg.fit_first}-{cfg.fit_last} "
                                  "from appearance)", "slope of sqrt(area) (px<sup>1/2</sup>/h)", "px^0.5/h",
                                  width=args.rate_bin))
    if args.growth in ("doubling", "both"):
        figures.append(histograms(data.colonies, data.conditions, "doubling_time_min",
                                  f"Radial doubling time ({lb:g} to {ub:g} px)", f"time from {lb:g} to {ub:g} px (min)",
                                  "min", width=time_bin))
    page = out.path("histograms.html")
    write_html(figures, page, f"ScanLag {results.name}")
    if not args.no_open:
        webbrowser.open(page.as_uri())
    print(stats[["condition", "n_plates", "n_colonies", "appearance_min_median", "radial_growth_sqrtpx_per_h_median",
                 "doubling_time_min_median"]].round(3).to_string(index=False))
    print(f"written to {out.root}")


if __name__ == "__main__":
    main()

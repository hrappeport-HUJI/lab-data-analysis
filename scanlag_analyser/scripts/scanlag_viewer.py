"""Interactive viewer of a ScanLag analysis: one HTML page per experiment, built with the data-viewer.

    python scripts/scanlag_viewer.py <results folder> [--out FILE] [--plotly cdn|inline] [--color-by COLUMN]

Reads <results>/analysis/all_colonies.csv (run scanlag_analysis.py first) and <results>/conditions.csv, and
writes <results>/analysis/viewer.html: one point per included colony, with

* filters and colours by condition, scanner, plate, and any extra column of conditions.csv (e.g. strain,
  medium, replicate: one value per row of the layout, so every plate of that row gets it),
* preset views: lag histograms of all colonies, lag vs growth rate (Spearman per group, live), lag vs
  doubling time,
* histograms per group along both axes, and 1 - CDF curves of lag (or growth) per group under the plot,
* rank-based comparisons between the groups shown (Mann-Whitney, Kruskal-Wallis, Holm-corrected).

"Lag" here is the appearance time: the first scan in which the colony is detected. The page says so.

The data-viewer lives next to this folder in lab-data-analysis (skills/data-viewer). Elsewhere, point the
environment variable DATA_VIEWER_DIR at a copy of it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_COLS = {"condition", "scanner", "plates"}


def find_data_viewer() -> Path:
    for d in (os.environ.get("DATA_VIEWER_DIR"), HERE.parents[1] / "skills" / "data-viewer"):
        if d and (Path(d) / "build_viewer.py").is_file():
            return Path(d)
    sys.exit("data-viewer not found: set DATA_VIEWER_DIR to the folder holding build_viewer.py "
             "(skills/data-viewer in https://github.com/hrappeport-HUJI/lab-data-analysis)")


def group_columns(layout_csv: Path, colonies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the extra columns of conditions.csv to the colonies, by scanner and plate."""
    sys.path.insert(0, str(HERE.parent))
    from scanlag.analysis.conditions import parse_plates

    if not layout_csv.is_file():
        return colonies, []
    table = pd.read_csv(layout_csv, dtype=str).dropna(how="all")
    table.columns = [c.strip() for c in table.columns]
    lower = {c.lower(): c for c in table.columns}
    extra = [c for c in table.columns if c.lower() not in BASE_COLS]
    if not extra:
        return colonies, []
    table = table.dropna(subset=[lower["condition"]])
    rows = [{"scanner": str(r[lower["scanner"]]).strip(), "plate": p, **{c: (str(r[c]).strip() if pd.notna(r[c]) else "") for c in extra}}
            for _, r in table.iterrows() for p in parse_plates(r[lower["plates"]])]
    groups = pd.DataFrame(rows)
    out = colonies.merge(groups, on=["scanner", "plate"], how="left")
    for c in extra:
        out[c] = out[c].fillna("").replace("", "(none)")
    return out, extra


def ordered(values) -> list[str]:
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def build(results: Path, out: Path, plotly: str, color_by: str | None) -> Path:
    table = results / "analysis" / "all_colonies.csv"
    if not table.is_file():
        sys.exit(f"{table} not found: run scripts/scanlag_analysis.py {results} first")
    df = pd.read_csv(table)
    if df.empty:
        sys.exit(f"{table} has no colonies")
    df["plate"] = df["plate"].astype(int)
    df, extra = group_columns(results / "conditions.csv", df)
    layout_order = ordered(pd.read_csv(results / "conditions.csv", dtype=str).iloc[:, 0].dropna().str.strip()) \
        if (results / "conditions.csv").is_file() else []

    df["condition"] = df["condition"].astype(str)
    df["scanner"] = df["scanner"].astype(str)
    df["plate_id"] = df["scanner"] + " · plate " + df["plate"].astype(str)
    df["key"] = df["plate_id"] + " · colony " + df["colony_id"].astype(str)
    df["merged"] = df["first_merged_min"].notna().map({True: "merged with a neighbour", False: "never merged"})
    df["flagged"] = (df["n_flags"].fillna(0) > 0).map({True: "flagged, kept in review", False: "not flagged"})

    cond_order = [c for c in layout_order if c in set(df["condition"])] + sorted(set(df["condition"]) - set(layout_order))
    cats = {
        "condition": {"label": "Condition", "kind": "category", "order": cond_order},
        **{c: {"label": c, "kind": "category", "order": ordered(df[c])} for c in extra},
        "scanner": {"label": "Scanner", "kind": "category", "order": sorted(df["scanner"].unique())},
        "plate_id": {"label": "Plate", "kind": "category",
                     "order": sorted(df["plate_id"].unique(), key=lambda s: (s.split(" · ")[0], int(s.rsplit(" ", 1)[1])))},
        "merged": {"label": "Merged", "kind": "category", "order": ["never merged", "merged with a neighbour"]},
        "flagged": {"label": "Review flags", "kind": "category", "order": ["not flagged", "flagged, kept in review"]},
    }
    nums = {
        "appearance_h": {"label": "Lag: appearance time (h)", "short": "Lag", "never_outlier": True,
                         "description": "Time from time zero to the first scan in which the colony is detected (a blob of about 10 px)."},
        "radial_growth_um_per_h": {"label": "Radial growth rate (µm/h)", "short": "Growth rate",
                                   "missing": "no growth rate (merged early or too few frames)",
                                   "description": "Slope of sqrt(area) against time over the first frames after appearance, before any merge, as the radius of a disc of equal area."},
        "doubling_time_h": {"label": "Doubling time, 20 → 80 px (h)", "short": "Doubling time",
                            "missing": "no doubling time (merged first, never reached 80 px, or both sizes between two scans)"},
        "final_area_px": {"label": "Final area (px)", "short": "Final area", "scale": "log", "never_outlier": True},
        "radial_fit_r2": {"label": "Growth fit R²", "short": "Fit R²"},
    }
    settings_file = results / "analysis" / "settings.json"
    fit = json.loads(settings_file.read_text()).get("fit", {}) if settings_file.is_file() else {}
    fit_txt = (f" Growth rate fit: frames {fit.get('fit_first')}-{fit.get('fit_last')} after appearance, merged frames "
               f"{'split' if fit.get('merged_growth') == 'split' else 'excluded'}; doubling time {fit.get('lb')} → {fit.get('ub')} px.") if fit else ""
    groups_txt = f" Groups from conditions.csv: {', '.join(extra)}." if extra else ""
    notes = ("One point per colony kept after the review (status included). Lag is the appearance time: the first scan in which "
             "the colony is detected, measured from the run's time zero. It is a detection time, not the end of the lag, and a fast "
             "grower is detected earlier, so check the lag vs growth correlation before reading a lag difference as a lag effect." + fit_txt
             + " Colonies without a growth value sit in the band at the top of the lag vs growth views; the lag histograms and the "
             "1 - CDF curves use every colony." + groups_txt)
    first = color_by or "condition"
    if first not in cats:
        sys.exit(f"--color-by {first!r}: choose one of {', '.join(cats)}")
    common = {"outliers": False, "color_by": first, "bins": 80}  # lag is in scan steps; 22 bins hide its shape
    spec = {
        "title": f"ScanLag · {results.resolve().parent.name if results.name.startswith('scanlag_results') else results.name}",
        "subtitle": f"{len(df)} colonies · {df['condition'].nunique()} conditions · {df['plate_id'].nunique()} plates",
        "row_meaning": "one colony, kept after the review",
        "notes": notes, "id": "key",
        "label": "{condition} · {id}",
        "color_by": first,
        "filters": ["condition", *extra, "scanner", "plate_id", "merged", "flagged"],
        "hover": ["condition", *extra, "plate_id"],
        "views": [
            {"name": "Lag histograms (all colonies)", "x": "appearance_h", "y": "final_area_px", "y_scale": "log", "band": False, "trend": False, **common},
            {"name": "Lag vs growth rate", "x": "appearance_h", "y": "radial_growth_um_per_h", "band": True, **common},
            {"name": "Lag vs doubling time", "x": "appearance_h", "y": "doubling_time_h", "band": True, **common},
            {"name": "Lag vs growth rate, by plate", "x": "appearance_h", "y": "radial_growth_um_per_h", "band": True, **common, "color_by": "plate_id"},
        ],
        "features": {"cdf": {"column": "appearance_h", "columns": ["appearance_h", "radial_growth_um_per_h", "doubling_time_h"],
                             "scale": "log", "title": "1 − CDF: fraction of colonies above x (for lag: not yet appeared)"}},
        "columns": {**cats, **nums},
    }
    keep = ["key", *spec["columns"]]
    sys.path.insert(0, str(find_data_viewer()))
    from build_viewer import write_viewer

    write_viewer(df[keep], spec, out, plotly=plotly)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="results folder (with analysis/all_colonies.csv)")
    parser.add_argument("--out", type=Path, help="page to write (default <results>/analysis/viewer.html)")
    parser.add_argument("--plotly", default="cdn", help="cdn (needs internet, default) or inline (works offline, +3.5 MB)")
    parser.add_argument("--color-by", help="first colour grouping: condition (default), an extra conditions.csv column, scanner or plate_id")
    a = parser.parse_args(argv)
    out = build(a.results, a.out or a.results / "analysis" / "viewer.html", a.plotly, a.color_by)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

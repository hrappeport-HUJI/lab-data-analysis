"""Group reviewed plates into experimental conditions and compute per-colony growth metrics.

The layout table maps conditions to plates, one row per (condition, scanner) pair:

    condition,scanner,plates
    CTRL,1-4.1u,"1,2,3"
    CTRL,1-5.1u,1-3
    AMP5,1-4.1u,"4,5,6"

``plates`` takes a comma-separated list and/or ranges (``1-3``). Plates that were skipped by the
pipeline (empty slots) are ignored with a warning.

Per colony (included colonies only):

* ``appearance_min`` (and ``_h``): first frame the colony is detected
* ``radial_growth_sqrtpx_per_h``: slope of a least-squares line through sqrt(area) against time,
  over the colony's frames ``fit_frames`` (counted from appearance, 1 = the appearance frame; default
  1-12), up to the frame before it first merges (see ``merged_growth``)
  (also ``radial_growth_um_per_h``: the same slope as the radius of a disc, sqrt(area / pi), in um)
* ``doubling_time_min`` (and ``_h``): time from the first frame >= lb px to the first frame >= ub px (default 20 and
  80: the radius doubles; MATLAB getAppearanceGrowth); NaN if the colony merged before reaching ub
  (unless ``merged_growth`` is "split"), never reached ub, or passed both thresholds between two scans
  (a time <= 0 is blanked)

``merged_growth`` (FitConfig): "exclude" (default) stops both growth metrics at the colony's first
merge, as MATLAB. "split" also uses the merged frames, where the footprint tracker gives each
colony its share of the merged blob (seed-tracked plates fall back to "exclude").
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scanlag.pipeline.export import GROWTH_LB, GROWTH_UB, first_time, plate_tables, use_merged_frames
from scanlag.pipeline.run import plate_info

UM_PER_PX = 25400 / 300  # scans are 300 dpi
KEY = ["scanner", "plate", "colony_id"]


@dataclass
class FitConfig:
    fit_first: int = 1  # frames counted from appearance (1 = appearance frame)
    fit_last: int = 12
    min_points: int = 4
    lb: float = GROWTH_LB  # px, doubling time from lb to ub
    ub: float = GROWTH_UB
    merged_growth: str = "exclude"  # or "split": also use merged frames (footprint tracker)


def parse_plates(text) -> list[int]:
    """'1,2,4-6' -> [1, 2, 4, 5, 6]."""
    plates = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            plates += range(int(m[1]), int(m[2]) + 1)
        elif part.isdigit():
            plates.append(int(part))
        else:
            raise ValueError(f"cannot read plate list {text!r}")
    return plates


def read_layout(path: Path) -> pd.DataFrame:
    """Layout CSV -> one row per (condition, scanner, plate), in file order."""
    table = pd.read_csv(path, dtype=str, skip_blank_lines=True).dropna(how="all")
    table.columns = [c.strip().lower() for c in table.columns]
    missing = {"condition", "scanner", "plates"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} needs the columns condition, scanner, plates (missing {sorted(missing)})")
    return expand_layout(table.dropna(subset=["condition"]))


def expand_layout(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of (condition, scanner, plates) -> one row per (condition, scanner, plate). Raises
    ValueError if a plate is in more than one condition."""
    rows = [{"condition": r.condition.strip(), "scanner": r.scanner.strip(), "plate": p}
            for r in table.itertuples() for p in parse_plates(r.plates)]
    layout = pd.DataFrame(rows, columns=["condition", "scanner", "plate"])
    dup = layout[layout.duplicated(["scanner", "plate"], keep=False)]
    if not dup.empty:
        raise ValueError("plates assigned to more than one condition: "
                         + ", ".join(sorted({f"{s} plate {p}" for s, p in zip(dup.scanner, dup.plate)})))
    return layout


def available_plates(results: Path) -> dict[str, list[int]]:
    """scanner -> processed plate numbers in a results folder."""
    from scanlag.pipeline.run import load_plate_dirs

    found: dict[str, list[int]] = {}
    for d in load_plate_dirs(results):
        found.setdefault(d.parent.name, []).append(int(d.name.split("_")[1]))
    return {s: sorted(p) for s, p in sorted(found.items())}


def layout_template(results: Path) -> pd.DataFrame:
    """One row per scanner with its plates and an empty condition, for the user to fill in."""
    return pd.DataFrame([{"condition": "", "scanner": s, "plates": ",".join(map(str, p))}
                         for s, p in available_plates(results).items()])


def doubling_time(tracks: pd.DataFrame, lb: float, ub: float, use_merged: bool = False) -> pd.Series:
    """colony_id -> hours from the first frame >= lb px to the first >= ub px (one plate). NaN if the
    colony merged before reaching ub (unless ``use_merged``), never reached it, or passed both
    between two scans."""
    t_lb, t_ub = first_time(tracks, lb, own_only=not use_merged), first_time(tracks, ub, own_only=not use_merged)
    growth = t_ub - t_lb.reindex(t_ub.index)
    if not use_merged:
        first_merge = tracks[tracks.merged].groupby("colony_id").time_min.min().reindex(t_ub.index)
        growth = growth.where(~(first_merge <= t_ub))
    return growth.where(growth > 0) / 60


def radial_growth(tracks: pd.DataFrame, cfg: FitConfig, use_merged: bool = False) -> pd.DataFrame:
    """Per colony: slope of sqrt(area) vs time (px^0.5 / h) over its frames fit_first..fit_last
    (counted from appearance), stopping before the colony first merges (unless ``use_merged``)."""
    t = tracks[tracks.area_px > 0]
    first_merge = t[t.merged & (not use_merged)].groupby(KEY).frame.min().rename("first_merge")
    appear = t.groupby(KEY).frame.min().rename("appear")
    t = t.join(first_merge, on=KEY).join(appear, on=KEY)
    k = t.frame - t.appear + 1
    t = t[(t.frame < t.first_merge.fillna(np.inf)) & (k >= cfg.fit_first) & (k <= cfg.fit_last)]
    t = t.assign(y=np.sqrt(t.area_px))

    def fit(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n < cfg.min_points or g.time_h.nunique() < 2:
            return pd.Series({"slope": np.nan, "r2": np.nan, "n": n})
        slope, intercept = np.polyfit(g.time_h, g.y, 1)
        resid = g.y - (slope * g.time_h + intercept)
        r2 = 1 - resid.var() / g.y.var() if g.y.var() > 0 else np.nan
        return pd.Series({"slope": slope, "r2": r2, "n": n})

    if t.empty:
        return pd.DataFrame(columns=KEY + ["slope", "r2", "n"])
    return t.groupby(KEY)[["time_h", "y"]].apply(fit).reset_index()


@dataclass
class ConditionData:
    colonies: pd.DataFrame  # included colonies with condition and growth metrics
    tracks: pd.DataFrame  # area tracks of the included colonies
    conditions: list[str]  # in layout order
    missing: list[str]  # layout plates without results


def collect(results: Path, layout: pd.DataFrame, cfg: FitConfig = FitConfig()) -> ConditionData:
    """The included colonies and their tracks of every layout plate, with growth metrics and a
    condition column; layout plates without results are listed in ``missing``."""
    colonies, tracks, missing = [], [], []
    for r in layout.itertuples():
        plate_dir = results / r.scanner / f"plate_{r.plate}"
        if not (plate_dir / "colonies.parquet").exists():
            missing.append(f"{r.scanner} plate {r.plate}")
            continue
        with warnings.catch_warnings():  # use_merged_frames below warns once per plate
            warnings.simplefilter("ignore")
            col, tr, _ = plate_tables(plate_dir, cfg.merged_growth)
        use_merged = use_merged_frames(plate_info(plate_dir), cfg.merged_growth)
        col, tr = col[col.included], tr[tr.included]
        tr = tr.drop(columns="included")
        growth = radial_growth(tr, cfg, use_merged).set_index("colony_id")
        slope = col.colony_id.map(growth.slope)
        col = col.assign(
            condition=r.condition,
            radial_growth_sqrtpx_per_h=slope.to_numpy(),
            radial_growth_um_per_h=(slope / np.sqrt(np.pi) * UM_PER_PX).to_numpy(),
            radial_fit_r2=col.colony_id.map(growth.r2).to_numpy(),
            radial_fit_points=col.colony_id.map(growth.n).fillna(0).astype(int).to_numpy(),
            doubling_time_h=col.colony_id.map(doubling_time(tr, cfg.lb, cfg.ub, use_merged)).to_numpy(),
        )
        col = col.assign(
            doubling_time_min=col.doubling_time_h * 60,
        )
        colonies.append(col)
        tracks.append(tr.assign(condition=r.condition))
    if not colonies:
        raise FileNotFoundError(f"none of the layout plates were found in {results}")
    colonies = pd.concat(colonies, ignore_index=True)
    colonies = colonies[["condition", "scanner", "plate", "colony_id", "appearance_frame", "appearance_min",
                         "appearance_h", "radial_growth_sqrtpx_per_h", "radial_growth_um_per_h", "radial_fit_r2",
                         "radial_fit_points", "doubling_time_min", "doubling_time_h", "first_merged_min", "final_area_px", "seed_x",
                         "seed_y", "n_flags", "flags"]]
    tracks = pd.concat(tracks, ignore_index=True)
    tracks = tracks[["condition", "scanner", "plate", "colony_id", "frame", "time_min", "time_h", "area_px", "x",
                     "y", "merged"]]
    return ConditionData(colonies, tracks, list(dict.fromkeys(layout.condition)), missing)


def area_matrix(tracks: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per colony (condition, scanner, plate, colony_id), one column per frame.

    Scanners are imaged at their own times, so the frame columns are only comparable within a
    scanner; the times are in the long tracks table.
    """
    wide = tracks.pivot_table(index=["condition"] + KEY, columns="frame", values="area_px", fill_value=0)
    wide.columns = [f"frame_{c}" for c in wide.columns]
    return wide.reset_index()


def summary(colonies: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """One row per condition: plate and colony counts, and mean/median/std of each metric."""
    rows = []
    for c in conditions:
        d = colonies[colonies.condition == c]
        row = {"condition": c, "n_scanners": d.scanner.nunique(),
               "n_plates": d[["scanner", "plate"]].drop_duplicates().shape[0], "n_colonies": len(d)}
        for m in ["appearance_min", "radial_growth_sqrtpx_per_h", "radial_growth_um_per_h", "doubling_time_min"]:
            v = d[m].dropna()
            row |= {f"{m}_n": len(v), f"{m}_mean": v.mean(), f"{m}_median": v.median(), f"{m}_std": v.std()}
        rows.append(row)
    return pd.DataFrame(rows)


def settings(cfg: FitConfig, layout: pd.DataFrame) -> dict:
    return {"fit": vars(cfg), "layout": layout.to_dict("records")}

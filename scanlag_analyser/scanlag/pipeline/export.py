"""Write the reviewed results as plain tables.

Per plate (``<results>/<scanner>/plate_<k>/``):

* ``tracks.csv``: one row per colony and frame: frame, time_min, time_h, colony_id, area_px, x, y,
  merged (the colony shares a blob with another colony: with the footprint tracker the area is its
  share of the blob, with the seed tracker the whole blob's), included
* ``colonies.csv``: one row per colony: status (0 included, 1 near border, 2 excluded by the
  reviewer, 3 zero final area), appearance time, time from LB to UB px (MATLAB
  getAppearanceGrowth, NaN if the colony merged before reaching UB), final area, flags
* ``area_matrix.csv``: time x colony area table of the included colonies (MATLAB's ``Area``)

Experiment level (``<results>/``): ``all_colonies.csv``, ``all_tracks.csv`` (with scanner and plate
columns) and ``summary.csv`` (colony counts per plate).

``export`` always computes the growth time with ``merged_growth="exclude"`` (as MATLAB); see
``plate_tables`` for the "split" alternative.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scanlag.io.matlab_import import ColonyStatus
from scanlag.store import ResultsStore

GROWTH_LB, GROWTH_UB = 20, 80  # px, MATLAB getAppearanceGrowth defaults
MERGED_GROWTH = ("exclude", "split")


def use_merged_frames(info: dict, merged_growth: str) -> bool:
    """Whether growth metrics may use the frames where a colony is merged (see ``merged_growth``)."""
    from scanlag.pipeline.run import merged_area_is_own

    if merged_growth not in MERGED_GROWTH:
        raise ValueError(f"merged_growth must be one of {MERGED_GROWTH}, not {merged_growth!r}")
    if merged_growth == "exclude":
        return False
    if not merged_area_is_own(info):
        warnings.warn(f"{info['scanner']} plate {info['plate']} was tracked with the seed tracker: merged colonies "
                      "report the whole blob, so their merged frames are excluded from growth")
        return False
    return True


def first_time(tracks: pd.DataFrame, area: float, own_only: bool) -> pd.Series:
    t = tracks[(tracks.area_px >= area) & (~tracks.merged if own_only else True)]
    return t.groupby("colony_id").time_min.min()


def plate_tables(plate_dir: Path, merged_growth: str = "exclude") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(colonies, tracks, area matrix) tables of one processed plate, as written by ``export``.

    ``merged_growth``: "exclude" (default, as MATLAB) ignores the frames where a colony is merged, so
    a colony that merges before reaching UB has no growth time. "split" uses its share of the merged
    blob instead; it needs plates tracked by the footprint tracker (seed-tracked plates fall back to
    "exclude", with a warning).
    """
    info = json.loads((plate_dir / "plate.json").read_text())
    use_merged = use_merged_frames(info, merged_growth)
    colonies = pd.read_parquet(plate_dir / "colonies.parquet")
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    included = colonies.set_index("colony_id").status == ColonyStatus.VALID
    tracks = tracks.assign(time_h=tracks.time_min / 60,
                           included=tracks.colony_id.map(included).fillna(False).astype(bool))

    lb = first_time(tracks, GROWTH_LB, own_only=not use_merged)
    ub = first_time(tracks, GROWTH_UB, own_only=not use_merged)
    first_merge = tracks[tracks.merged].groupby("colony_id").time_min.min()
    ids = colonies.colony_id
    reached_ub = ub.reindex(ids).notna()
    if not use_merged:
        reached_ub &= ~(first_merge.reindex(ids) <= ub.reindex(ids)).fillna(False)
    growth = (ub.reindex(ids) - lb.reindex(ids)).where(reached_ub.to_numpy())
    colonies = colonies.assign(
        status_name=[ColonyStatus(s).name for s in colonies.status],
        included=colonies.status == ColonyStatus.VALID,
        scanner=info["scanner"],
        plate=info["plate"],
        appearance_h=colonies.appearance_min / 60,
        **{f"growth_{GROWTH_LB}_{GROWTH_UB}px_min": growth.to_numpy()},
        first_merged_min=first_merge.reindex(ids).to_numpy(),
    )
    cols = ["scanner", "plate", "colony_id", "status", "status_name", "auto_status", "included", "appearance_frame",
            "appearance_min", "appearance_h", f"growth_{GROWTH_LB}_{GROWTH_UB}px_min", "first_merged_min",
            "final_area_px", "seed_x", "seed_y", "n_flags", "flags"]
    colonies = colonies[cols]

    tracks = tracks.assign(scanner=info["scanner"], plate=info["plate"])[
        ["scanner", "plate", "frame", "time_min", "time_h", "colony_id", "area_px", "x", "y", "merged", "included"]]
    inc = tracks[tracks.included]
    times = pd.Series(info["times_min"], name="time_min")
    if inc.empty:
        matrix = pd.DataFrame(index=range(len(times)))
    else:
        matrix = inc.pivot_table(index="frame", columns="colony_id", values="area_px", fill_value=0)
        matrix = matrix.reindex(range(len(times)), fill_value=0)
    matrix.columns = [f"colony_{c}" for c in matrix.columns]
    matrix.insert(0, "time_min", times.to_numpy())
    return colonies, tracks, matrix


def export(results: Path, store: ResultsStore, log=print, plate_dirs: list[Path] | None = None) -> None:
    """Write the tables of ``plate_dirs`` (default: every processed plate under ``results``), see the
    module docstring. The experiment-level tables then hold only those plates."""
    from scanlag.pipeline.run import load_plate_dirs

    all_col, all_tr, summary = [], [], []
    for plate_dir in load_plate_dirs(results) if plate_dirs is None else plate_dirs:
        colonies, tracks, matrix = plate_tables(plate_dir)
        rel = plate_dir.relative_to(store.root).parts
        store.write_table(colonies, *rel, "colonies.csv")
        store.write_table(tracks, *rel, "tracks.csv")
        store.write_table(matrix, *rel, "area_matrix.csv")
        review = plate_dir / "review.json"
        reviewed = json.loads(review.read_text()).get("visited", False) if review.exists() else False
        all_col.append(colonies)
        all_tr.append(tracks)
        counts = colonies.status_name.value_counts()
        summary.append({"scanner": rel[0], "plate": rel[1], "reviewed": reviewed, "n_colonies": len(colonies),
                        "n_included": int(colonies.included.sum()),
                        **{f"n_{s.name.lower()}": int(counts.get(s.name, 0)) for s in ColonyStatus},
                        "n_flagged_included": int(((colonies.n_flags > 0) & colonies.included).sum())})
    if not summary:
        raise FileNotFoundError(f"no processed plates in {results}")
    store.write_table(pd.concat(all_col, ignore_index=True), "all_colonies.csv")
    store.write_table(pd.concat(all_tr, ignore_index=True), "all_tracks.csv")
    store.write_table(pd.DataFrame(summary), "summary.csv")
    log(f"exported {len(summary)} plates to {store.root}")
    if unreviewed := [f"{s['scanner']}/{s['plate']}" for s in summary if not s["reviewed"]]:
        log(f"{len(unreviewed)} of them were never opened in the review window (automatic statuses): "
            + ", ".join(unreviewed))

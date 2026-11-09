"""Compare pipeline results with a MATLAB ScanLag analysis of the same scans.

    python scripts/compare_with_matlab.py <results folder> <MATLAB ROI folder> [--scanner 1-4.1u]

The MATLAB plates are the ``sep_<scanner no.>_<plate>`` folders under the ROI folder. Each MATLAB ROI
image is located in the raw scan (template matching), so MATLAB centroids map exactly onto the
pipeline's plate crop. Colonies are matched one to one by their position at appearance (Hungarian,
at most --max-dist px apart). Both are timed with the same scan times.

Output (<results>/matlab_comparison/): matches.csv (every colony of either, with its partner and
both statuses), plates.csv (counts per plate), pairs.csv (included colonies kept by both, with
appearance, 20-80 px time and radial growth of each), and figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scanlag.analysis.conditions import FitConfig, doubling_time, radial_growth
from scanlag.cli import split_prompt_args
from scanlag.io.matlab_import import ColonyStatus, load_plate
from scanlag.io.raw import read_scan
from scanlag.pipeline.export import plate_tables
from scanlag.store import REGISTRY_FILE, ResultsStore

MATLAB_COLOR, NEW_COLOR = "#2a78d6", "#eb6834"


def gray(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a


def roi_offset(matlab, raw_dir: Path) -> tuple[int, int]:
    """(x, y) of the MATLAB ROI's top-left corner in the raw scan."""
    roi = read_scan(matlab.image_path(0))
    stem = Path(matlab.files[0]).stem.rsplit("_", 1)[0]
    raw = next(raw_dir / f"{stem}{ext}" for ext in (".tif", ".tiff") if (raw_dir / f"{stem}{ext}").exists())
    score = cv2.matchTemplate(gray(read_scan(raw)), gray(roi), cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(score)
    if best < 0.95:
        raise RuntimeError(f"{matlab.plate_dir.name}: ROI not found in {raw.name} (correlation {best:.2f})")
    return x, y


def matlab_tables(matlab, times_min: np.ndarray, dx: float, dy: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MATLAB tracks and colonies in pipeline crop coordinates (0-based)."""
    area, cen = matlab.area, matlab.centroid
    t_idx, c_idx = np.nonzero(area > 0)
    x, y = cen[t_idx, c_idx, 0] - 1 + dx, cen[t_idx, c_idx, 1] - 1 + dy
    tracks = pd.DataFrame({"scanner": "", "plate": 0, "frame": t_idx, "time_min": times_min[t_idx],
                           "time_h": times_min[t_idx] / 60, "colony_id": c_idx + 1, "area_px": area[t_idx, c_idx],
                           "x": x, "y": y})
    # getMergedColonies: colonies sharing a centroid in a frame are one blob
    tracks["merged"] = tracks.duplicated(["frame", "x", "y"], keep=False)
    first = tracks.sort_values("frame").groupby("colony_id").first()
    colonies = pd.DataFrame({"colony_id": np.arange(1, matlab.n_colonies + 1), "status": matlab.status})
    colonies = colonies.join(first[["frame", "time_min", "x", "y"]].rename(
        columns={"frame": "appearance_frame", "time_min": "appearance_min", "x": "seed_x", "y": "seed_y"}),
        on="colony_id")
    colonies["final_area_px"] = area[-1]
    return colonies, tracks


def growth_metrics(colonies: pd.DataFrame, tracks: pd.DataFrame, cfg: FitConfig) -> pd.DataFrame:
    slope = radial_growth(tracks.assign(scanner="", plate=0), cfg).set_index("colony_id").slope
    return colonies.assign(appearance_h=colonies.appearance_min / 60,
                           doubling_time_h=colonies.colony_id.map(doubling_time(tracks, cfg.lb, cfg.ub)).to_numpy(),
                           radial_growth=colonies.colony_id.map(slope).to_numpy())


def match(a: pd.DataFrame, b: pd.DataFrame, max_dist: float) -> list[tuple[int, int, float]]:
    """One-to-one matches (row in a, row in b, distance) by seed position."""
    if a.empty or b.empty:
        return []
    d = np.hypot(a.seed_x.to_numpy()[:, None] - b.seed_x.to_numpy()[None],
                 a.seed_y.to_numpy()[:, None] - b.seed_y.to_numpy()[None])
    rows, cols = linear_sum_assignment(np.where(d <= max_dist, d, 1e6))
    return [(r, c, d[r, c]) for r, c in zip(rows, cols) if d[r, c] <= max_dist]


def draw_tracks(ax, tracks: pd.DataFrame, ids, colors, title: str) -> None:
    for cid, color in zip(ids, colors):
        t = tracks[tracks.colony_id == cid].sort_values("frame")
        if t.empty:
            continue
        a, m = t.area_px.to_numpy().astype(float), t.merged.to_numpy()
        ax.plot(t.time_h, np.where(m, np.nan, a), color=color, lw=0.9)
        ax.plot(t.time_h, np.where(m, a, np.nan), color=color, lw=0.9, alpha=0.25)
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("time (h)")
    ax.grid(alpha=0.3, which="both", lw=0.3)


def scatter(ax, x, y, label: str) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.scatter(x, y, s=14, color=NEW_COLOR, alpha=0.7, edgecolor="white", linewidth=0.5)
    if len(x):
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#8a8a85", lw=1, ls="--")
        diff = y - x
        ax.text(0.03, 0.97, f"n = {len(x)}\nnew - MATLAB: median {np.median(diff):+.2f}, "
                f"mean {diff.mean():+.2f}\n|diff| median {np.median(np.abs(diff)):.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set_xlabel(f"MATLAB {label}")
    ax.set_ylabel(f"new {label}")
    ax.grid(alpha=0.3, lw=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="results folder written by scanlag_pipeline.py")
    parser.add_argument("roi", type=Path, help="MATLAB ROI folder (holds sep_<n>_<plate>)")
    parser.add_argument("--scanner", help="pipeline scanner name (default: the only one)")
    parser.add_argument("--matlab-scanner", default="1", help="the <n> in MATLAB's sep_<n>_<plate> (default 1)")
    parser.add_argument("--max-dist", type=float, default=8, help="max seed distance for a match, px")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))

    results = args.results.resolve()
    raw_dirs = json.loads((results / REGISTRY_FILE).read_text())["raw_dirs"]
    scanners = sorted(p.parent.name for p in results.glob("*/scanner.json"))
    scanner = args.scanner or (scanners[0] if len(scanners) == 1 else sys.exit(f"choose --scanner from {scanners}"))
    store = ResultsStore(results, raw_dirs)
    out = "matlab_comparison"
    cfg = FitConfig()

    all_matches, plate_rows, pairs, per_plate = [], [], [], []
    for plate_dir in sorted((results / scanner).glob("plate_*"), key=lambda p: int(p.name.split("_")[1])):
        k = int(plate_dir.name.split("_")[1])
        mdir = args.roi / f"sep_{args.matlab_scanner}_{k}"
        if not mdir.exists():
            print(f"plate {k}: no MATLAB folder {mdir.name}, skipped")
            continue
        info = json.loads((plate_dir / "plate.json").read_text())
        matlab = load_plate(mdir)
        stems = [Path(f).stem.rsplit("_", 1)[0] for f in matlab.files]
        if stems != [Path(f).stem for f in info["raw_scans"]]:
            sys.exit(f"plate {k}: MATLAB and pipeline used different scans")
        x0, y0, _ = info["crop_xy_size"]
        rx, ry = roi_offset(matlab, Path(info["raw_dir"]))
        times = np.asarray(info["times_min"], dtype=float)
        m_col, m_tr = matlab_tables(matlab, times, rx - x0, ry - y0)
        m_col = growth_metrics(m_col, m_tr, cfg)
        n_col, n_tr, _ = plate_tables(plate_dir)
        n_col = growth_metrics(n_col[["colony_id", "status", "appearance_frame", "appearance_min", "seed_x", "seed_y",
                                      "final_area_px"]], n_tr, cfg)

        # match the colonies MATLAB saw growing (final area > 0) against all of the pipeline's
        m_use = m_col[m_col.appearance_frame.notna()].reset_index(drop=True)
        found = match(m_use, n_col, args.max_dist)
        m_idx = {r: (c, d) for r, c, d in found}
        n_matched = {c for _, c, _ in found}
        rows = []
        for r, mc in m_use.iterrows():
            c, d = m_idx.get(r, (None, np.nan))
            nc = n_col.iloc[c] if c is not None else None
            rows.append({"plate": k, "matlab_id": mc.colony_id, "matlab_status": ColonyStatus(mc.status).name,
                         "new_id": nc.colony_id if nc is not None else np.nan,
                         "new_status": ColonyStatus(nc.status).name if nc is not None else "",
                         "distance_px": d, "seed_x": mc.seed_x, "seed_y": mc.seed_y})
        for c, nc in n_col.iterrows():
            if c not in n_matched:
                rows.append({"plate": k, "matlab_id": np.nan, "matlab_status": "", "new_id": nc.colony_id,
                             "new_status": ColonyStatus(nc.status).name, "distance_px": np.nan,
                             "seed_x": nc.seed_x, "seed_y": nc.seed_y})
        matches = pd.DataFrame(rows)
        all_matches.append(matches)

        m_kept = matches.matlab_status == "VALID"
        n_inc = matches.new_status == "VALID"
        plate_rows.append({"plate": k, "matlab_colonies": len(m_col), "matlab_growing": len(m_use),
                           **{f"matlab_{s.name.lower()}": int((m_col.status == s).sum()) for s in ColonyStatus},
                           "new_colonies": len(n_col), "new_included": int(n_inc.sum()),
                           "both_kept": int((m_kept & n_inc).sum()),
                           "matlab_kept_only": int((m_kept & ~n_inc).sum()),
                           "new_included_only": int((n_inc & ~m_kept).sum())})
        both = matches[m_kept & n_inc]
        mm, nn = m_col.set_index("colony_id"), n_col.set_index("colony_id")
        p = pd.DataFrame({"plate": k, "matlab_id": both.matlab_id.astype(int).to_numpy(),
                          "new_id": both.new_id.astype(int).to_numpy()})
        for col in ["appearance_h", "doubling_time_h", "radial_growth", "final_area_px"]:
            p[f"matlab_{col}"] = mm.loc[p.matlab_id, col].to_numpy()
            p[f"new_{col}"] = nn.loc[p.new_id, col].to_numpy()
        pairs.append(p)
        per_plate.append((k, m_tr, n_tr, matches, m_col, n_col))

    if not per_plate:
        sys.exit("no plates to compare")
    matches, plates = pd.concat(all_matches, ignore_index=True), pd.DataFrame(plate_rows)
    pairs = pd.concat(pairs, ignore_index=True)
    store.write_table(matches, out, "matches.csv")
    store.write_table(plates, out, "plates.csv")
    store.write_table(pairs, out, "pairs.csv")
    status_table = pd.crosstab(matches.matlab_status.replace("", "(not found)"),
                               matches.new_status.replace("", "(not found)"))
    store.write_table(status_table.reset_index(), out, "status_crosstab.csv")

    # area tracks: MATLAB kept (left) vs new included (right); matched colonies share a colour
    n = len(per_plate)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.3 * n), squeeze=False, sharey=True)
    for (k, m_tr, n_tr, mt, m_col, n_col), (a0, a1) in zip(per_plate, axes):
        m_ids = mt.loc[mt.matlab_status == "VALID", "matlab_id"].astype(int).tolist()
        n_ids = mt.loc[mt.new_status == "VALID", "new_id"].astype(int).tolist()
        cmap = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(mt), 1)))[np.random.default_rng(0).permutation(len(mt))]
        row_color = dict(zip(range(len(mt)), cmap))
        m_color = {int(r.matlab_id): row_color[i] for i, r in enumerate(mt.itertuples()) if r.matlab_id == r.matlab_id}
        n_color = {int(r.new_id): row_color[i] for i, r in enumerate(mt.itertuples()) if r.new_id == r.new_id}
        draw_tracks(a0, m_tr, m_ids, [m_color[i] for i in m_ids], f"plate {k}: MATLAB kept ({len(m_ids)})")
        draw_tracks(a1, n_tr, n_ids, [n_color[i] for i in n_ids], f"plate {k}: new, included ({len(n_ids)})")
        a0.set_ylabel("area (px)")
    fig.tight_layout()
    fig.savefig(store.path(out, "area_tracks.png"), dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    scatter(axes[0], pairs.matlab_appearance_h.to_numpy(), pairs.new_appearance_h.to_numpy(), "appearance (h)")
    scatter(axes[1], pairs.matlab_doubling_time_h.to_numpy(), pairs.new_doubling_time_h.to_numpy(),
            "20-80 px time (h)")
    scatter(axes[2], pairs.matlab_radial_growth.to_numpy(), pairs.new_radial_growth.to_numpy(),
            "sqrt(area) slope (px$^{1/2}$/h)")
    fig.suptitle("Colonies kept in MATLAB and included in the new run, matched by position", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(store.path(out, "paired_metrics.png"), dpi=120)

    print(plates.to_string(index=False))
    print("\nMATLAB status (rows) vs new status (columns):")
    print(status_table.to_string())
    for col in ["appearance_h", "doubling_time_h", "radial_growth", "final_area_px"]:
        d = (pairs[f"new_{col}"] - pairs[f"matlab_{col}"]).dropna()
        print(f"{col}: n={len(d)}, new - MATLAB median {d.median():+.3f}, mean {d.mean():+.3f}, "
              f"|diff| median {d.abs().median():.3f}, max {d.abs().max():.3f}")
    print(f"written to {store.root / out}")


if __name__ == "__main__":
    main()

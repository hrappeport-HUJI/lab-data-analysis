"""Plot area tracks of the kept (included) colonies: MATLAB vs a model run.

    python scripts/plot_area_tracks.py --plates EPEC260224 [--testset testset_matlab] [--model unet_scanlag_v1]

Needs ``scripts/track_with_model.py`` to have run on the same plates. For each plate: MATLAB (left)
and the model (right), one line per kept colony, same colour for the same colony in both panels.
Merged stretches (the area is that of a blob holding several colonies) are drawn faded. Also writes
``overview.png`` with every plate. Figures go to ``<testset>/model_runs/<model>/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore


def draw(ax, tracks: pd.DataFrame, ids, colors) -> None:
    for cid, color in zip(ids, colors):
        t = tracks[tracks.colony_id == cid].sort_values("frame")
        if t.empty:
            continue
        h, a, m = t.time_min.to_numpy() / 60, t.area_px.to_numpy().astype(float), t.merged.to_numpy()
        ax.plot(h, np.where(m, np.nan, a), color=color, lw=0.8)
        ax.plot(h, np.where(m, a, np.nan), color=color, lw=0.8, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("time (h)")
    ax.grid(alpha=0.3, which="both", lw=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plates", required=True, help="plate-ID prefix, e.g. EPEC260224")
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    parser.add_argument("--model", default="unet_scanlag_v1",
                        help="model name, as in <testset>/model_runs/<model> (default unet_scanlag_v1)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    run = args.testset / "model_runs" / args.model / "plates"
    plates = sorted(p.name for p in run.iterdir() if p.name.startswith(args.plates) and (p / "tracks.parquet").exists())
    n = len(plates)
    ov, ov_ax = plt.subplots((n + 3) // 4, 4, figsize=(18, 3.6 * ((n + 3) // 4)), squeeze=False, sharey=True)
    rows = []
    for k, pid in enumerate(plates):
        colonies = pd.read_parquet(args.testset / "plates" / pid / "colonies.parquet")
        ids = colonies.loc[colonies.status == 0, "colony_id"].to_numpy()
        matlab = pd.read_parquet(args.testset / "plates" / pid / "tracks.parquet")
        model = pd.read_parquet(run / pid / "tracks.parquet")
        matlab, model = matlab[matlab.colony_id.isin(ids)], model[model.colony_id.isin(ids)]
        colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(ids), 1)))[np.random.default_rng(0).permutation(len(ids))]

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
        draw(ax[0], matlab, ids, colors)
        draw(ax[1], model, ids, colors)
        found = model.colony_id.nunique()
        ax[0].set_title(f"MATLAB ({len(ids)} kept colonies)")
        ax[1].set_title(f"{args.model} ({found} of them detected at least once)")
        ax[0].set_ylabel("area (px)")
        fig.suptitle(f"{pid}: area tracks of kept colonies (faded = merged)")
        fig.tight_layout()
        fig.savefig(store.path("model_runs", args.model, "figures", f"{pid}.png"), dpi=110)
        plt.close(fig)

        a = ov_ax.flat[k]
        for tr, color in ((matlab, "0.45"), (model, "tab:red")):
            for _, t in tr[~tr.merged].groupby("colony_id"):
                a.plot(t.time_min / 60, t.area_px, color=color, lw=0.4, alpha=0.5)
        a.set_yscale("log")
        a.set_title(pid.replace("EPEC", ""), fontsize=9)
        a.grid(alpha=0.3, lw=0.3)

        # per-colony comparison: time to reach 20 px, before any merge
        def first(tr, thr):
            t = tr[(tr.area_px >= thr) & ~tr.merged].groupby("colony_id").time_min.min()
            return t.reindex(ids)

        for thr in (20, 80):
            dm, dp = first(matlab, thr), first(model, thr)
            both = dm.notna() & dp.notna()
            diff = (dp - dm)[both]
            rows.append({"plate_id": pid, "threshold_px": thr, "n_kept": len(ids), "n_both": int(both.sum()),
                         "median_diff_min": diff.median(), "mean_abs_diff_min": diff.abs().mean()})
    for a in ov_ax.flat[n:]:
        a.axis("off")
    ov.suptitle(f"Area tracks of kept colonies, not merged (gray: MATLAB, red: {args.model})")
    ov.supxlabel("time (h)")
    ov.supylabel("area (px)")
    ov.tight_layout()
    ov.savefig(store.path("model_runs", args.model, "figures", f"overview_{args.plates}.png"), dpi=100)
    table = pd.DataFrame(rows)
    store.write_table(table, "model_runs", args.model, "figures", f"timing_{args.plates}.csv")
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

"""Frozen result snapshots, founder summaries, descriptive plots, planned tests."""

from pathlib import Path
from itertools import combinations
from math import comb
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .storage import Project, write_json, utc
from .review import materialize
from .events import summarize
from .measurement import measure


def pairwise(values, pairs, unit="field", allow_field=False):
    if unit not in {"field", "biological_replicate"}:
        raise ValueError("Inference unit must be field or biological_replicate")
    rows = []
    for metric, g in values.groupby("metric"):
        for a, b in pairs:
            row = {
                "metric": metric,
                "condition_1": a,
                "condition_2": b,
                "unit": unit,
                "p_raw": None,
                "p_holm": None,
            }
            if unit == "field" and not allow_field:
                row["status"] = "field_exchangeability_not_confirmed"
                rows.append(row)
                continue
            if (
                unit == "biological_replicate"
                and g[g.condition.isin([a, b])][unit].isna().any()
            ):
                row["status"] = "missing_biological_replicate"
                rows.append(row)
                continue
            subset = g[g.condition.isin([a, b])]
            if subset.groupby(unit).condition.nunique().max() > 1:
                row["status"] = "paired_design_requires_another_test"
                rows.append(row)
                continue
            units = subset.groupby(["condition", unit]).value.median().dropna()
            x = (
                units.loc[a].to_numpy()
                if a in units.index.get_level_values(0)
                else np.array([])
            )
            y = (
                units.loc[b].to_numpy()
                if b in units.index.get_level_values(0)
                else np.array([])
            )
            row.update(
                n_units_1=len(x),
                n_units_2=len(y),
                median_1=float(np.median(x)) if len(x) else None,
                median_2=float(np.median(y)) if len(y) else None,
            )
            if min(len(x), len(y)) < 2:
                row["status"] = "insufficient_units"
                rows.append(row)
                continue
            ranks = rankdata(np.r_[x, y])
            n1 = len(x)
            center = n1 * (len(ranks) + 1) / 2
            observed = abs(ranks[:n1].sum() - center)
            total = comb(len(ranks), n1)
            if total > 200000:
                row["status"] = "too_many_exact_allocations_use_a_test_adapter"
                rows.append(row)
                continue
            count = sum(
                abs(ranks[list(ix)].sum() - center) >= observed - 1e-12
                for ix in combinations(range(len(ranks)), n1)
            )
            row.update(
                p_raw=count / total,
                status="ok",
                permutations=total,
                method="two-sided exact unpaired rank permutation of unit medians",
            )
            rows.append(row)
    valid = sorted(
        [r for r in rows if r["p_raw"] is not None], key=lambda r: r["p_raw"]
    )
    previous = 0
    for i, row in enumerate(valid):
        previous = max(previous, min(1, row["p_raw"] * (len(rows) - i)))
        row["p_holm"] = previous
    return rows


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze(project, run_id=None, reviewed_only=False):
    p = project if isinstance(project, Project) else Project(project)
    run_id = run_id or p.latest_run()
    review = p.review(run_id)
    data = materialize(p, run_id, review["operations"])
    founders = summarize(p, data)
    selected = [f for f in founders if not reviewed_only or f["reviewed"]]
    measurements = measure(p, run_id, data, selected)
    result_id = p.new_id("result")
    out = p.root / "results" / result_id
    out.mkdir(parents=True)
    # Both configuration and review state are captured; subsequent edits cannot change this result.
    write_json(out / "review_snapshot.json", {"run_id": run_id, **review})
    write_json(out / "config_snapshot.json", p.config)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "observation_ids"} for r in founders]
    ).to_csv(out / "founders.csv", index=False)
    pd.DataFrame(data["observations"]).to_csv(out / "observations.csv", index=False)
    pd.DataFrame(measurements).to_csv(out / "measurements.csv", index=False)
    values = []
    for f in selected:
        if f["decision"] == "division" and f["division_midpoint_s"] is not None:
            values.append(
                {
                    "founder_id": f["id"],
                    "field": f["field"],
                    "biological_replicate": f["biological_replicate"],
                    "condition": f["condition"],
                    "metric": "first_observed_division_h",
                    "value": f["division_midpoint_s"] / 3600,
                }
            )
    byid = {f["id"]: f for f in selected}
    for m in measurements:
        if m.get("status") == "ok" and m.get("value") is not None:
            f = byid[m["founder_id"]]
            values.append(
                {
                    "founder_id": f["id"],
                    "field": f["field"],
                    "biological_replicate": f["biological_replicate"],
                    "condition": f["condition"],
                    "metric": m["measurement"],
                    "value": m["value"],
                }
            )
    table = pd.DataFrame(
        values,
        columns=[
            "founder_id",
            "field",
            "biological_replicate",
            "condition",
            "metric",
            "value",
        ],
    )
    # Repeated frames / descendants do not inflate founder counts in comparisons.
    collapsed = (
        table.groupby(["founder_id", "field", "condition", "metric"], dropna=False)
        .value.median()
        .reset_index()
    )
    collapsed["biological_replicate"] = collapsed.founder_id.map(
        {f["id"]: f["biological_replicate"] for f in selected}
    )
    policy = p.config.get("analysis", {})
    tests = (
        pairwise(
            collapsed,
            policy.get("comparisons", []),
            policy.get("unit", "field"),
            policy.get("allow_field_inference", False),
        )
        if len(table)
        else []
    )
    collapsed["decision"] = collapsed.founder_id.map(
        {f["id"]: f["decision"] for f in selected}
    )
    collapsed.to_csv(out / "founder_metrics.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "comparisons.csv", index=False)
    stats = []
    plots = []
    conditions = sorted(
        {f["condition"] for f in selected if f["decision"] != "exclude"}
    )
    for index, (metric, g) in enumerate(collapsed.groupby("metric")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for i, c in enumerate(conditions):
            v = g[g.condition == c].value.to_numpy()
            stats.append(
                {
                    "metric": metric,
                    "condition": c,
                    "n_founders": len(v),
                    "mean": float(np.mean(v)) if len(v) else None,
                    "median": float(np.median(v)) if len(v) else None,
                }
            )
            if len(v):
                for ax in axes:
                    if len(v) > 1 and np.ptp(v) > 0:
                        ax.violinplot([v], positions=[i + 1], showmedians=True)
                    else:
                        ax.scatter([i + 1], [v[0]])
        for ax in axes:
            ax.set_xticks(range(1, len(conditions) + 1), conditions)
            ax.set_ylabel(metric)
        axes[0].set_title("Linear")
        axes[1].set_yscale("symlog", linthresh=1)
        axes[1].set_title("Symmetric log (linear within ±1)")
        comparisons = [t for t in tests if t["metric"] == metric]
        caption = "; ".join(
            (
                f"{r['condition_1']}–{r['condition_2']}: adjusted p={r['p_holm']:.3g}"
                if r.get("p_holm") is not None
                else f"{r['condition_1']}–{r['condition_2']}: {r['status']}"
            )
            for r in comparisons
        )
        fig.suptitle(metric)
        fig.text(
            0.5,
            0.02,
            caption or "Descriptive only; no requested statistical tests",
            ha="center",
            fontsize=8,
            wrap=True,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.94))
        name = f"metric_{index:03}.png"
        fig.savefig(out / name, dpi=150)
        fig.savefig(out / f"metric_{index:03}.svg")
        plt.close(fig)
        plots.append({"file": name, "metric": metric})
    extra_plots, strata, correlations = descriptive_outputs(collapsed, out, policy)
    plots.extend(extra_plots)
    pd.DataFrame(strata).to_csv(out / "stratified_summaries.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "correlations.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "summaries.csv", index=False)
    status_counts = {
        s: sum(f["decision"] == s for f in selected)
        for s in ["division", "lost", "no_division", "exclude"]
    }
    summary = {
        "schema_version": 1,
        "result_id": result_id,
        "run_id": run_id,
        "review_revision": review["revision"],
        "created_at": utc(),
        "reviewed_only": reviewed_only,
        "founders": len(selected),
        "counts": status_counts,
        "plots": plots,
        "statistics": stats,
        "stratified_statistics": strata,
        "correlations": correlations,
        "comparisons": tests,
        "warnings": data["needs_review"]
        + [
            "First observed division includes the first cell cycle. Tracking loss is not a negative growth observation.",
            "Fields are not established independent culture replicates. Inference requires appropriate exchangeability.",
            "All-frame or descendant measurements are reduced to a median per founder for condition comparisons. See measurements.csv for individual observations.",
        ],
    }
    write_json(out / "summary.json", clean(summary))
    write_json(
        p.root / "latest_results.json",
        {
            "result_id": result_id,
            "run_id": run_id,
            "review_revision": review["revision"],
        },
    )
    return clean(summary)


def descriptive_outputs(table, out, policy):
    """Common-bin histograms by outcome; correlations are descriptive at cell level."""
    plots = []
    stats = []
    correlations = []
    for i, (metric, g) in enumerate(table.groupby("metric")):
        conditions = sorted(g.condition.unique())
        decisions = ["all", "division", "lost", "no_division"]
        fig, axes = plt.subplots(
            len(conditions),
            len(decisions),
            figsize=(13, max(3, 2.5 * len(conditions))),
            squeeze=False,
        )
        bins = np.histogram_bin_edges(g.value, bins="auto")
        for row, c in enumerate(conditions):
            for col, d in enumerate(decisions):
                values = g[
                    (g.condition == c) & ((g.decision == d) if d != "all" else True)
                ].value.to_numpy()
                stats.append(
                    {
                        "metric": metric,
                        "condition": c,
                        "decision": d,
                        "n_founders": len(values),
                        "mean": float(np.mean(values)) if len(values) else None,
                        "median": float(np.median(values)) if len(values) else None,
                    }
                )
                ax = axes[row, col]
                ax.hist(values, bins=bins, color="#427b63")
                ax.set_title(f"{c}: {d} (n={len(values)})")
                ax.set_ylabel("Founders")
        fig.suptitle(metric)
        fig.tight_layout()
        name = f"histogram_{i:03}.png"
        fig.savefig(out / name, dpi=150)
        fig.savefig(out / f"histogram_{i:03}.svg")
        plt.close(fig)
        plots.append({"file": name, "metric": metric + " by outcome"})
    for i, (xname, yname) in enumerate(policy.get("correlations", [])):
        wide = (
            table.pivot(
                index=["founder_id", "condition"], columns="metric", values="value"
            ).reset_index()
            if len(table)
            else pd.DataFrame()
        )
        if xname not in wide or yname not in wide:
            continue
        groups = [("pooled", wide)] + [(c, g) for c, g in wide.groupby("condition")]
        fig, axes = plt.subplots(
            1, len(groups), figsize=(4 * len(groups), 4), squeeze=False
        )
        for ax, (group, g) in zip(axes.flat, groups):
            paired = g[[xname, yname]].dropna()
            rho = (
                float(spearmanr(paired[xname], paired[yname]).statistic)
                if len(paired) >= 3
                and paired[xname].nunique() > 1
                and paired[yname].nunique() > 1
                else None
            )
            correlations.append(
                {
                    "x": xname,
                    "y": yname,
                    "condition": group,
                    "n_founders": len(paired),
                    "spearman_rho": rho,
                    "p_value": None,
                    "status": "descriptive_cell_level_no_independence_assumed",
                }
            )
            ax.scatter(paired[xname], paired[yname], s=12)
            ax.set(
                xlabel=xname,
                ylabel=yname,
                title=(
                    f"{group}: n={len(paired)}, rho={rho:.2f}"
                    if rho is not None
                    else f"{group}: n={len(paired)}, rho undefined"
                ),
            )
        fig.tight_layout()
        name = f"correlation_{i:03}.png"
        fig.savefig(out / name, dpi=150)
        fig.savefig(out / f"correlation_{i:03}.svg")
        plt.close(fig)
        plots.append({"file": name, "metric": xname + " vs " + yname})
    return plots, stats, correlations

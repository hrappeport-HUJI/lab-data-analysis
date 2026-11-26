"""Founder-level PI / YFP / lag figures, drawn from a result folder of ``lab-microscopy analyze``.

A notebook loads the data once (``load``) and draws each figure with one ``plot`` call. Everything a
person is likely to change is a keyword argument; settings are merged left to right, so a figure's own
settings override the shared ones: ``plot(data, "initial_PI_vs_lag", COMMON, PI_LAG)``.

Definitions (per founder)
  initial_PI     background-corrected mean PI in the founder's baseline outline (``baseline_PI``)
  PI_rate        least-squares slope of the per-frame PI value against time over the first
                 ``rate_window_min`` minutes, stopping before division so daughters never count;
                 needs >= 3 frames spanning >= half the window. ADU/pixel/min
  relative_rate  PI_rate / initial_PI x 100, in % of the initial PI per minute; only where
                 initial_PI >= ``relative_rate_min_PI``
  initial_YFP    background-corrected mean YFP in the baseline outline (``baseline_YFP``)
  lag_h          midpoint of the first-division interval, in hours
Per-frame value: mean inside the outline minus the median of a 3-10 px ring that excludes all cells.
"""
from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.scale import SymmetricalLogTransform
from scipy.stats import kruskal, mannwhitneyu, rankdata, theilslopes

VARIABLES = {
    "initial_PI": "Initial PI (ADU/pixel)",
    "PI_rate": "Early PI-change rate (ADU/pixel/min)",
    "relative_rate": "Early PI change relative to initial PI (%/min)",
    "initial_YFP": "Initial YFP (ADU/pixel)",
    "lag_h": "First division (h)",
}
FIGURES = {
    "initial_PI_vs_lag": ("initial_PI", "lag_h"),
    "PI_rate_vs_lag": ("PI_rate", "lag_h"),
    "PI_rate_vs_initial_PI": ("initial_PI", "PI_rate"),
    "relative_rate_vs_lag": ("relative_rate", "lag_h"),
    "relative_rate_vs_initial_PI": ("initial_PI", "relative_rate"),
    "initial_YFP_vs_lag": ("initial_YFP", "lag_h"),
    "YFP_vs_PI": ("initial_PI", "initial_YFP"),
}
LINEAR_ZONE = {"initial_PI": 10, "PI_rate": 0.5, "relative_rate": 0.5, "initial_YFP": 10, "lag_h": 0.5}
MARKERS = "o^sDvP"
INK, MUTED = "#222222", "#6b6b6b"
DEFAULTS = dict(
    panels=None,                 # list of lists of conditions; default: one panel per condition
    colors=None,                 # {condition: colour}
    x_scale="linear", y_scale="linear",          # "linear" or "log" (symmetric log: zero and negatives allowed)
    x_limits="panel", y_limits="panel",          # "panel", "shared", or (low, high)
    auto_limit_percentiles=(0, 100),
    log_linear_zone=None,        # {variable: half-width of the linear part of a log axis}
    bins=22, normalize_histograms=True,
    marginal_style="lanes",      # "lanes": one row (top) / column (side) per condition; "overlay": all on one axis
    strain_summary=True,         # table under each panel: n, median (mean) per condition and between-condition tests
    exclude_outliers=True, outlier_fence=3.0,
    include_non_dividers=False,
    trend_lines=True,
    no_yfp=(),                   # conditions without a YFP signal; left out of YFP figures
    columns=3, panel_size=(6, 5.2), save_to=None, show=True,
)


# ============================================================ data

@dataclass
class Data:
    founders: pd.DataFrame
    measurements: pd.DataFrame
    recording_end_h: float
    result: Path


def load(result_dir, rate_window_min=30, relative_rate_min_PI=20, condition_names=None):
    """Read founders.csv and measurements.csv of one result folder and derive the founder metrics."""
    result = Path(result_dir)
    f = pd.read_csv(result / "founders.csv")
    m = pd.read_csv(result / "measurements.csv")
    if condition_names:
        f["condition"] = f.condition.replace(condition_names)
        m["condition"] = m.condition.replace(condition_names)
    ok = m[m.status == "ok"]
    for name, column in [("baseline_PI", "initial_PI"), ("baseline_YFP", "initial_YFP")]:
        base = ok[ok.measurement == name].groupby("founder_id").value.median()
        f[column] = f.id.map(base)
    f["lag_h"] = np.where(f.decision == "division", f.division_midpoint_s / 3600, np.nan)
    series = ok[ok.measurement == "PI_series"]
    last = dict(zip(f.id, np.where(f.decision == "division", f.lower_frame, np.inf)))
    window_s, rates = rate_window_min * 60, {}
    for fid, a in series.groupby("founder_id"):
        if fid not in last:
            continue
        t0 = a.time_s.min()
        a = a[(a.time_s - t0 <= window_s) & (a.frame <= last[fid])]
        a = a.groupby("frame", as_index=False).agg(time_s=("time_s", "first"), value=("value", "median"))
        if len(a) >= 3 and a.time_s.max() - a.time_s.min() >= window_s / 2:
            rates[fid] = float(np.polyfit((a.time_s - t0) / 60, a.value, 1)[0])
    f["PI_rate"] = f.id.map(rates)
    f["relative_rate"] = np.where(f.initial_PI >= relative_rate_min_PI, f.PI_rate / f.initial_PI * 100, np.nan)
    end = series.time_s.max() / 3600 if len(series) else np.nan
    return Data(f, m, float(end), result)


# ============================================================ statistics

def _permutation_rho(x, y, seed):
    a, b = rankdata(x).astype(float), rankdata(y).astype(float)
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if len(x) < 3 or den == 0:
        return np.nan, np.nan
    rho, observed, n = float(a @ b / den), abs(a @ b), len(a)
    if n <= 8:
        perms = np.array(list(itertools.permutations(range(n))))
        return rho, float(np.mean(np.abs(b[perms] @ a) >= observed - 1e-12))
    rng, extreme, total = np.random.default_rng(seed), 0, 0
    for start in range(0, 19999, 5000):
        size = min(5000, 19999 - start)
        perms = rng.permuted(np.broadcast_to(b, (size, n)), axis=1)
        extreme += np.count_nonzero(np.abs(perms @ a) >= observed - 1e-12); total += size
    return rho, (extreme + 1) / (total + 1)


def _outlier_mask(values, variable, fence):
    """Tukey far-out fences; PI and YFP judged on asinh(v/10) because they are roughly log-normal."""
    v = np.asarray(values, float)
    t = np.arcsinh(v / 10) if variable in ("initial_PI", "initial_YFP") else v
    finite = np.isfinite(t)
    if finite.sum() < 4:
        return np.zeros(len(v), bool)
    q1, q3 = np.percentile(t[finite], [25, 75])
    return finite & ((t < q1 - fence * (q3 - q1)) | (t > q3 + fence * (q3 - q1)))


def _select(f, condition, x, y, s):
    """(used, extra, removed) founders of one condition. ``used`` enter the statistics."""
    g = f[(f.condition == condition) & (f.decision != "exclude")]
    lag = y == "lag_h"
    if lag:
        used = g[g.decision == "division"].dropna(subset=[x, y])
        extra = g[g.decision == "no_division"].dropna(subset=[x]) if s["include_non_dividers"] else g.iloc[:0]
    else:
        pool = g if s["include_non_dividers"] else g[g.decision == "division"]
        used, extra = pool.dropna(subset=[x, y]), g.iloc[:0]
    removed = used.iloc[:0]
    if s["exclude_outliers"]:
        bad = np.zeros(len(used), bool)
        for var in (x, y):
            if var != "lag_h":  # the outcome is never an outlier
                bad |= _outlier_mask(used[var], var, s["outlier_fence"])
        removed, used = used[bad], used[~bad]
        if len(extra):
            extra = extra[~_outlier_mask(extra[x], x, s["outlier_fence"])]
    return used, extra, removed


def statistics(data, figure, *settings, **overrides):
    """Spearman rho per condition (two-sided permutation p, Holm across the figure) and Theil-Sen fit."""
    s = _settings(settings, overrides)
    x, y = FIGURES[figure]
    rows, removed = [], []
    for condition in _conditions(data, s, figure):
        used, extra, out = _select(data.founders, condition, x, y, s)
        removed.extend(out.id.tolist())
        seed = int(hashlib.sha256(f"{figure}/{condition}".encode()).hexdigest()[:8], 16)
        rho, p = _permutation_rho(used[x].to_numpy(), used[y].to_numpy(), seed)
        fit = theilslopes(used[y], used[x]) if len(used) >= 3 and used[x].nunique() > 1 else None
        rows.append({"figure": figure, "condition": condition, "n": len(used), "n_not_shown_in_stats": len(extra),
                     "n_outliers_removed": len(out), "spearman_rho": rho, "p": p,
                     "theil_sen_slope": fit.slope if fit else np.nan,
                     "theil_sen_intercept": fit.intercept if fit else np.nan})
    valid, prev = sorted([r for r in rows if np.isfinite(r["p"])], key=lambda r: r["p"]), 0
    for i, r in enumerate(valid):
        prev = max(prev, min(1, r["p"] * (len(valid) - i))); r["p_holm"] = prev
    table = pd.DataFrame(rows)
    table["p_holm"] = [r.get("p_holm", np.nan) for r in rows]
    table.attrs["removed_ids"] = removed  # plain list: pandas compares attrs when combining tables
    return table


# ============================================================ helpers

def _settings(settings, overrides):
    s = dict(DEFAULTS)
    for d in settings:
        s.update(d or {})
    s.update(overrides)
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown setting(s): {sorted(unknown)}")
    return s


def _conditions(data, s, figure=None):
    names = [c for group in _panels(data, s) for c in group]
    if figure and "initial_YFP" in FIGURES[figure]:
        names = [c for c in names if c not in s["no_yfp"]]
    return names


def _panels(data, s):
    return s["panels"] or [[c] for c in sorted(data.founders.condition.dropna().unique())]


def _colors(data, s):
    palette = ["#1764c0", "#d52828", "#2a9d3c", "#8a3ffc", "#e07b00", "#00838f", "#6d4c41", "#c2185b"]
    names = sorted(data.founders.condition.dropna().unique())
    return {**{c: palette[i % len(palette)] for i, c in enumerate(names)}, **(s["colors"] or {})}


class _Axis:
    def __init__(self, scale, linthresh):
        if scale not in ("linear", "log"):
            raise ValueError(f"scale must be 'linear' or 'log', not {scale!r}")
        self.scale, self.linthresh = scale, linthresh
        self.t = SymmetricalLogTransform(10, linthresh, 1) if scale == "log" else None

    def fwd(self, v):
        v = np.asarray(v, float)
        return self.t.transform(v) if self.t else v

    def inv(self, v):
        v = np.asarray(v, float)
        return self.t.inverted().transform(v) if self.t else v

    def auto_range(self, values, percentiles):
        values = np.asarray(values, float); values = values[np.isfinite(values)]
        if values.size == 0:
            return (0.0, 1.0)
        lo, hi = self.fwd(np.percentile(values, percentiles))
        pad = .06 * (hi - lo or 1)
        return tuple(self.inv([lo - pad, hi + pad]))

    def bins(self, lo, hi, n):
        return self.inv(np.linspace(*self.fwd([lo, hi]), n + 1))

    def apply(self, ax, which):
        if self.scale == "log":
            (ax.set_xscale if which == "x" else ax.set_yscale)("symlog", linthresh=self.linthresh)


def _limits(setting, axis, values, percentiles):
    if isinstance(setting, (tuple, list)):
        return {k: tuple(setting) for k in values}
    if setting == "shared":
        allv = np.concatenate([np.asarray(v, float) for v in values.values()] or [np.array([])])
        r = axis.auto_range(allv, percentiles)
        return {k: r for k in values}
    if setting == "panel":
        return {k: axis.auto_range(v, percentiles) for k, v in values.items()}
    raise ValueError(f"limits must be 'panel', 'shared' or (low, high), not {setting!r}")


def _hist(ax, values, bins, color, horizontal, normalize):
    counts, edges = np.histogram(values, bins=bins)
    if normalize and counts.sum():
        counts = counts / counts.sum()
    ax.stairs(counts, edges, orientation="horizontal" if horizontal else "vertical", baseline=None, color=color, lw=1.4)


def _lanes(ax, samples, bins, colors, horizontal, normalize):
    """One lane per condition on a common height scale: filled histogram, solid median and dotted mean.

    ``samples`` is a list of (condition, values to bin, values for the median/mean, axis limits). The
    first condition gets the top lane (top marginal) or the left lane (side marginal)."""
    counts = []
    for _, v, _, _ in samples:
        c, edges = np.histogram(v, bins=bins)
        counts.append(c / c.sum() if normalize and c.sum() else c.astype(float))
    step = max([c.max() for c in counts if len(c)] + [0]) * 1.12 or 1.0
    n = len(samples)
    for k, ((name, _, raw, (lo, hi)), c) in enumerate(zip(samples, counts)):
        base = k * step if horizontal else (n - 1 - k) * step
        color = colors[name]
        o = "horizontal" if horizontal else "vertical"
        ax.stairs(c + base, edges, baseline=base, fill=True, alpha=.28, color=color, orientation=o, lw=0)
        ax.stairs(c + base, edges, baseline=base, color=color, orientation=o, lw=1.3)
        line = ax.axvline if horizontal else ax.axhline
        line(base, color=MUTED, lw=.5, alpha=.6)
        if len(raw):
            for value, style in [(np.median(raw), "-"), (np.mean(raw), ":")]:
                v = float(np.clip(value, lo, hi))
                span = [base, base + step * .92]
                ax.plot(span if horizontal else [v, v], [v, v] if horizontal else span, style, color=color,
                        lw=2.2 if style == "-" else 1.8, solid_capstyle="butt")
    (ax.set_xlim if horizontal else ax.set_ylim)(0, n * step)
    (ax.set_xticks if horizontal else ax.set_yticks)([])


def _holm(ps):
    order, out, prev = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, ps[k] * (len(ps) - i))); out[k] = prev
    return out


def _between(samples):
    """Kruskal-Wallis across >2 conditions, and two-sided Mann-Whitney per pair with Holm correction."""
    samples = [(c, np.asarray(v, float)) for c, v in samples if len(v) >= 3]
    names = [c for c, _ in samples]
    kw = kruskal(*[v for _, v in samples]).pvalue if len(samples) > 2 else np.nan
    pairs = list(itertools.combinations(range(len(samples)), 2))
    ps = np.array([mannwhitneyu(samples[a][1], samples[b][1]).pvalue for a, b in pairs])
    holm = _holm(ps) if len(ps) else ps
    return kw, [(names[a], names[b], p, h) for (a, b), p, h in zip(pairs, ps, holm)]


def strain_differences(data, figure, *settings, **overrides):
    """Per panel and variable: n, median and mean per condition, and between-condition tests.

    Uses the founders a figure shows as filled markers (outliers removed). Pairwise p values are
    Holm-corrected within one panel and variable; Kruskal-Wallis is reported for panels of 3+ conditions."""
    s = _settings(settings, overrides)
    keep = set(_conditions(data, s, figure))
    rows = []
    for group in _panels(data, s):
        group = [c for c in group if c in keep]
        used = {c: _select(data.founders, c, *FIGURES[figure], s)[0] for c in group}
        for var in FIGURES[figure]:
            samples = [(c, used[c][var].dropna().to_numpy()) for c in group]
            kw, pairs = _between(samples) if len(group) > 1 else (np.nan, [])
            for c, v in samples:
                rows.append({"figure": figure, "panel": " + ".join(group), "variable": var, "kind": "summary",
                             "condition": c, "n": len(v), "median": np.median(v) if len(v) else np.nan,
                             "mean": np.mean(v) if len(v) else np.nan, "p": np.nan, "p_holm": np.nan})
            if np.isfinite(kw):
                rows.append({"figure": figure, "panel": " + ".join(group), "variable": var, "kind": "Kruskal-Wallis",
                             "condition": "all", "p": kw, "p_holm": np.nan})
            for a, b, p, h in pairs:
                rows.append({"figure": figure, "panel": " + ".join(group), "variable": var, "kind": "Mann-Whitney",
                             "condition": f"{a} vs {b}", "p": p, "p_holm": h})
    return pd.DataFrame(rows)


def _fmt(v):
    return "–" if not np.isfinite(v) else f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def _p(p):
    stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    return f"{p:.1g} {stars}" if p >= 1e-99 else f"<1e-99 {stars}"


def _summary_text(table, x, y, group):
    """Monospace block: per-condition n / median (mean), then the between-condition tests."""
    if table.empty:
        return ""
    t = table[table.kind == "summary"]
    w = max(len(c) for c in group) + 2
    head = f"{'':{w}}{'n':>5}   {'median (mean)':<24}{'median (mean)'}"
    sub = f"{'':{w}}{'':>5}   {VARIABLES[x].split(' (')[0]:<24}{VARIABLES[y].split(' (')[0]}"
    lines = [sub, head]
    for c in group:
        rx, ry = t[(t.condition == c) & (t.variable == x)], t[(t.condition == c) & (t.variable == y)]
        if rx.empty:
            continue
        rx, ry = rx.iloc[0], ry.iloc[0]
        lines.append(f"{c:{w}}{int(ry.n):>5}   {_fmt(rx['median']) + ' (' + _fmt(rx['mean']) + ')':<24}"
                     f"{_fmt(ry['median'])} ({_fmt(ry['mean'])})")
    if len(group) > 1:
        lines.append("Between strains (Mann-Whitney, Holm-corrected" + (", Kruskal-Wallis across all):" if len(group) > 2 else "):"))
        for var in (x, y):
            tv = table[table.variable == var]
            kw = tv[tv.kind == "Kruskal-Wallis"]
            pairs = [f"{r.condition.replace(' vs ', '–')} {_p(r.p_holm)}" for r in tv[tv.kind == "Mann-Whitney"].itertuples()]
            name = VARIABLES[var].split(" (")[0]
            first = f"  {name}: " + (f"KW p {_p(kw.p.iloc[0])}; " if len(kw) else "")
            chunks, line = [], first
            for pr in pairs:
                if len(line) + len(pr) > 96:
                    chunks.append(line.rstrip("; ")); line = "    "
                line += pr + "; "
            chunks.append(line.rstrip("; "))
            lines.extend(chunks)
    return "\n".join(lines)


def _panel_axes(fig, spec, normalize, s=None, n_lanes=1, summary_lines=0, legend_rows=1):
    lanes = (s or {}).get("marginal_style", "overlay") == "lanes" and n_lanes > 1
    top_h, side_w = (min(1 + .32 * n_lanes, 2.2), min(.9 + .25 * n_lanes, 1.9)) if lanes else (1, 1)
    sub = spec.subgridspec(5, 2, height_ratios=[.25 + .4 * legend_rows, top_h, 4.2, .55 if summary_lines else .01,
                                                .14 * summary_lines + .01],
                           width_ratios=[4.2, side_w], hspace=.05, wspace=.04)
    leg = fig.add_subplot(sub[0, :]); leg.axis("off")
    top = fig.add_subplot(sub[1, 0]); main = fig.add_subplot(sub[2, 0], sharex=top)
    side = fig.add_subplot(sub[2, 1], sharey=main)
    text = fig.add_subplot(sub[4, :]); text.axis("off")
    unit = ("fraction" if normalize else "cells") + ("\n(same scale\nin every lane)" if lanes else "")
    for a in (top, side):
        a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7, colors=MUTED)
    top.tick_params(labelbottom=False); side.tick_params(labelleft=False)
    top.set_ylabel(unit, fontsize=7, color=MUTED); side.set_xlabel(unit, fontsize=7, color=MUTED)
    main.spines[["top", "right"]].set_visible(False); main.grid(axis="y", alpha=.16); main.set_axisbelow(True)
    main.tick_params(labelsize=8, colors=INK)
    return leg, top, main, side, text


def _finish(fig, s):
    if s["save_to"]:
        path = Path(s["save_to"]); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=140, bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    if s["show"]:
        plt.show()
    else:
        plt.close(fig)


# ============================================================ scatter figures

def plot(data, figure, *settings, **overrides):
    """Draw one figure (a key of FIGURES): one panel per group of conditions, marginal histograms.

    Returns the statistics table; ``table.attrs["removed_ids"]`` lists founders dropped as outliers
    (``removed_outliers`` turns several tables into one founder table)."""
    s = _settings(settings, overrides)
    x, y = FIGURES[figure]
    zone = {**LINEAR_ZONE, **(s["log_linear_zone"] or {})}
    ax_x, ax_y = _Axis(s["x_scale"], zone[x]), _Axis(s["y_scale"], zone[y])
    lag = y == "lag_h"
    colors = _colors(data, s)
    stats = statistics(data, figure, s)
    keep = set(_conditions(data, s, figure))
    panels = [[c for c in group if c in keep] for group in _panels(data, s)]
    panels = [g for g in panels if g]
    if not panels:
        raise ValueError(f"No condition has the data for {figure}")
    pts = {i: {c: _select(data.founders, c, x, y, s) for c in g} for i, g in enumerate(panels)}
    xv = {i: np.concatenate([np.r_[u[x], e[x]] for u, e, _ in pts[i].values()]) for i in pts}
    yv = {i: np.concatenate([u[y].to_numpy() for u, e, _ in pts[i].values()]) for i in pts}
    xlim = _limits(s["x_limits"], ax_x, xv, s["auto_limit_percentiles"])
    ylim = _limits(s["y_limits"], ax_y, yv, s["auto_limit_percentiles"])
    if lag and not isinstance(s["y_limits"], (tuple, list)):
        ylim = {i: (min(0, lo) if s["y_scale"] == "linear" else lo, hi) for i, (lo, hi) in ylim.items()}

    diffs = strain_differences(data, figure, s) if s["strain_summary"] else pd.DataFrame()
    texts = {i: _summary_text(diffs[diffs.panel == " + ".join(g)] if len(diffs) else diffs, x, y, g)
             for i, g in enumerate(panels)}
    n_text = max([t.count("\n") + 1 if t else 0 for t in texts.values()])
    cols = min(s["columns"], len(panels)); rows = -(-len(panels) // cols)
    leg_rows = max(-(-(len(g) + (s["marginal_style"] == "lanes" and len(g) > 1)) // min(2, len(g))) for g in panels)
    extra_h = .17 * n_text + .35 * (leg_rows - 1)
    fig = plt.figure(figsize=(s["panel_size"][0] * cols, (s["panel_size"][1] + extra_h) * rows))
    outer = fig.add_gridspec(rows, cols, hspace=.45, wspace=.28)
    fig.suptitle(f"{VARIABLES[y]} vs {VARIABLES[x]}", fontsize=13, color=INK, x=.01, ha="left")
    for i, group in enumerate(panels):
        leg, top, ax, side, text = _panel_axes(fig, outer[i // cols, i % cols], s["normalize_histograms"], s,
                                               len(group), n_text + 1 if n_text else 0, leg_rows)
        lanes = s["marginal_style"] == "lanes" and len(group) > 1
        top_samples, side_samples = [], []
        (x0, x1), (y0, y1) = xlim[i], ylim[i]
        fy0, fy1 = ax_y.fwd([y0, y1]); h = fy1 - fy0
        band, ytop = (ax_y.inv([fy1 + .10 * h, fy1 + .19 * h]) if lag and s["include_non_dividers"] else (None, y1))
        xb, yb = ax_x.bins(x0, x1, s["bins"]), ax_y.bins(y0, y1, s["bins"])
        handles = []
        for k, c in enumerate(group):
            used, extra, removed = pts[i][c]
            r = stats[stats.condition == c].iloc[0]
            color, marker = colors[c], MARKERS[k % len(MARKERS)]
            yy = np.clip(used[y], y0, y1) if lag else used[y]
            ax.scatter(used[x], yy, s=30, c=color, marker=marker, alpha=.8, edgecolor="white", linewidth=.5, zorder=3)
            if len(extra):
                if lag:
                    jit = np.random.default_rng(i).uniform(-.025, .025, len(extra)) * h
                    off = (k - (len(group) - 1) / 2) * .03 * h
                    ax.scatter(extra[x], ax_y.inv(ax_y.fwd(band) + jit + off), s=30, facecolor="none",
                               edgecolor=color, marker=marker, linewidth=1.1, alpha=.85, zorder=3)
                else:
                    ax.scatter(extra[x], extra[y], s=30, facecolor="none", edgecolor=color, marker=marker, zorder=3)
            straight = s["x_scale"] == "linear" and s["y_scale"] == "linear"
            if s["trend_lines"] and straight and np.isfinite(r.theil_sen_slope):
                xx = np.linspace(max(used[x].min(), x0), min(used[x].max(), x1), 200)
                yy = r.theil_sen_intercept + r.theil_sen_slope * xx
                inside = (yy >= y0) & (yy <= y1)
                ax.plot(xx[inside], yy[inside], "--", color=color, lw=1.5, zorder=2)
            if lanes:
                top_samples.append((c, np.clip(np.r_[used[x], extra[x]], x0, x1), used[x].to_numpy(), (x0, x1)))
                side_samples.append((c, np.clip(used[y], y0, y1), used[y].to_numpy(), (y0, y1)))
            else:
                _hist(top, np.clip(np.r_[used[x], extra[x]], x0, x1), xb, color, False, s["normalize_histograms"])
                _hist(side, np.clip(used[y], y0, y1), yb, color, True, s["normalize_histograms"])
            stat = (f"ρ {r.spearman_rho:+.2f}, p {r.p:.2g} (Holm {r.p_holm:.2g})" if np.isfinite(r.p)
                    else "too few for a test")
            what = f"{r.n} divided" if lag else f"{r.n} founders"
            if len(extra):
                what += f", {len(extra)} {'no division' if lag else 'non-dividers'} (hollow)"
            if len(removed):
                what += f", {len(removed)} outliers removed"
            handles.append(Line2D([0], [0], marker=marker, color=color, markersize=7, markeredgecolor="white",
                                  linestyle="--" if s["trend_lines"] and straight else "",
                                  label=f"{c}: {what}\n{stat}"))
        ax_x.apply(ax, "x"); ax_y.apply(ax, "y")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, ytop)
        if band is not None:
            for a in (ax, side):
                a.axhline(y1, color=MUTED, lw=.8, ls=":", zorder=1)
            ax.text(1, ax_y.inv(ax_y.fwd(band) + .05 * h), f"no division by end ({data.recording_end_h:.1f} h)",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color=MUTED)
            ax.set_yticks([t for t in ax.get_yticks() if y0 <= t <= y1]); ax.set_ylim(y0, ytop)
        if x0 < 0 < x1:
            ax.axvline(0, color=MUTED, lw=.6, zorder=1)
        if not lag and y0 < 0 < y1:
            ax.axhline(0, color=MUTED, lw=.6, zorder=1)
        out_x = int(((xv[i] < x0) | (xv[i] > x1)).sum()); out_y = int(((yv[i] < y0) | (yv[i] > y1)).sum())
        if out_x or out_y:
            ax.text(1, -.17, f"outside the limits: {out_x} in x, {out_y} in y (drawn at the edge)",
                    transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
        if lanes:
            _lanes(top, top_samples, xb, colors, False, s["normalize_histograms"])
            _lanes(side, side_samples, yb, colors, True, s["normalize_histograms"])
            handles.append(Line2D([0], [0], color=MUTED, lw=2.2, label="histograms: median (solid), mean (dotted)"))
        else:
            top.set_ylim(0, None); side.set_xlim(0, None)
        if texts[i]:
            text.text(0, 1, texts[i], transform=text.transAxes, va="top", ha="left", family="monospace",
                      fontsize=7.4, color=INK)
        leg.text(0, 1, " + ".join(group), transform=leg.transAxes, fontsize=11, color=INK, va="top")
        leg.legend(handles=handles, fontsize=7.6, frameon=False, loc="lower left", ncol=min(2, len(group)),
                   labelcolor=INK, handlelength=2, borderaxespad=0)
        ax.set_xlabel(VARIABLES[x], fontsize=8.5, color=INK); ax.set_ylabel(VARIABLES[y], fontsize=8.5, color=INK)
    _finish(fig, s)
    return stats


def removed_outliers(data, tables):
    """Founders dropped as outliers by any of the given statistics tables, one row per figure."""
    rows = [data.founders[data.founders.id.isin(t.attrs.get("removed_ids", []))].assign(figure=t.figure.iloc[0])
            for t in tables if len(t)]
    return pd.concat(rows, ignore_index=True) if rows else data.founders.iloc[:0]


# ============================================================ YFP histograms

def yfp_histograms(data, *settings, times_h="auto", n_frames=4, max_h=4.0, **overrides):
    """Histograms of per-cell YFP (founders and their daughters) at a few YFP frames, one row per panel group.

    times_h: "auto" = ``n_frames`` evenly spaced YFP frames within the first ``max_h`` hours, or a list
    of hours (each uses the nearest YFP frame; panels with nothing within 0.75 h stay empty)."""
    s = _settings(settings, overrides)
    colors = _colors(data, s)
    m = data.measurements
    yfp = m[(m.measurement == "YFP_series") & (m.status == "ok")].copy()
    yfp["t_h"] = yfp.time_s / 3600
    keep = set(_conditions(data, s, "YFP_vs_PI"))
    panels = [[c for c in g if c in keep] for g in _panels(data, s)]
    panels = [g for g in panels if g]
    frames = sorted(yfp.frame.unique())
    times = yfp.groupby("frame").t_h.median()
    if times_h == "auto":
        early = [f for f in frames if times[f] <= max_h] or frames[:1]
        chosen = [early[int(round(j))] for j in np.linspace(0, len(early) - 1, min(n_frames, len(early)))]
    else:
        chosen = [min(frames, key=lambda f: abs(times[f] - t)) for t in times_h]
        chosen = [f if abs(times[f] - t) <= .75 else None for f, t in zip(chosen, times_h)]
    ax_x = _Axis(s["x_scale"], (s["log_linear_zone"] or {}).get("initial_YFP", LINEAR_ZONE["initial_YFP"]))
    values = {(i, j): yfp[(yfp.condition.isin(g)) & (yfp.frame == f)].value
              for i, g in enumerate(panels) for j, f in enumerate(chosen) if f is not None}
    lim = _limits(s["x_limits"], ax_x, values, s["auto_limit_percentiles"])
    fig, axes = plt.subplots(len(panels), len(chosen), figsize=(3.6 * len(chosen), 2.6 * len(panels)),
                             squeeze=False, sharey=False)
    for i, g in enumerate(panels):
        for j, f in enumerate(chosen):
            ax = axes[i, j]
            ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7, colors=MUTED)
            if f is None:
                ax.set_title("no YFP frame near this time", fontsize=8, color=MUTED); continue
            lo, hi = lim[(i, j)]
            bins = ax_x.bins(lo, hi, s["bins"])
            parts = []
            for c in g:
                v = yfp[(yfp.condition == c) & (yfp.frame == f)].value
                _hist(ax, np.clip(v, lo, hi), bins, colors[c], False, s["normalize_histograms"])
                if len(v):
                    ax.axvline(np.median(v), color=colors[c], ls=":", lw=1)
                parts.append(f"{c} n={len(v)} med={np.median(v):.0f}" if len(v) else f"{c} n=0")
            ax_x.apply(ax, "x"); ax.set_xlim(lo, hi)
            ax.set_title(f"frame {f} · {times[f]:.1f} h\n" + "; ".join(parts), fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel(" + ".join(g) + ("\nfraction" if s["normalize_histograms"] else "\ncells"), fontsize=8)
            if i == len(panels) - 1:
                ax.set_xlabel("YFP per cell (ADU/pixel)", fontsize=8)
    fig.tight_layout()
    _finish(fig, s)
    return chosen

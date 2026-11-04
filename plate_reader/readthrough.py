"""Stop-codon readthrough from a dual-fluorescent reporter measured on a plate reader.

The method is that of Hinnu et al. 2024, Sci Rep 14:6883 (doi:10.1038/s41598-024-57597-8):

1. Measure red autofluorescence in cells with a plasmid lacking the red gene, in each condition.
2. Subtract it from the red signal of the reporter wells.
3. Take the red:green ratio (R:G) of each reporter well.
4. Readthrough (%) = 100 x R:G of the mutant reporter / R:G of the intact reporter.

The paper does not normalize to OD or subtract green autofluorescence, so this doesn't either.
All values here have already had medium background subtracted (reader_background.py), which the
paper does not describe but this lab always does.

Errors come from biological replicates: each replicate is one well of its own culture in each
condition. Nothing here assumes which replicates pair up across reporters, because they are
independent cultures.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLE_IN_FRAME = "in_frame"
ROLE_AUTOFLUORESCENCE = "autofluorescence"
ROLE_TEST = "test"


@dataclass
class ReadthroughDesign:
    """Which channels and plate-map values play which part. Channel names are the loaded ones."""

    red: str
    green: str
    reporter_trait: str
    in_frame: object
    autofluorescence: object
    test: list
    condition_trait: str
    replicate_trait: str
    od: str = None
    min_od: float = 0.1
    roles: dict = field(init=False)

    def __post_init__(self):
        self.test = list(self.test)
        self.roles = {self.in_frame: ROLE_IN_FRAME, self.autofluorescence: ROLE_AUTOFLUORESCENCE}
        self.roles.update({value: ROLE_TEST for value in self.test})


def window_mean(time, y, t, window_h):
    """Mean of the reads within window_h of time t, and how many reads that was."""
    time, y = np.asarray(time, dtype=float), np.asarray(y, dtype=float)
    n = min(len(time), len(y))
    time, y = time[:n], y[:n]
    keep = np.isfinite(time) & np.isfinite(y) & (np.abs(time - t) <= window_h)
    if not keep.any():
        return np.nan, 0
    return float(y[keep].mean()), int(keep.sum())


def reporter_table(exp, corrected, design, wells, time_h, window_h=0.25):
    """One row per well: its part in the design and its background-subtracted values at time_h."""
    channels = {"red": design.red, "green": design.green}
    if design.od:
        channels["od"] = design.od
    rows = []
    for well in wells:
        reporter = exp.well_to_trait_value(well, design.reporter_trait)
        if reporter not in design.roles:
            continue
        row = {"well": well, "reporter": reporter, "role": design.roles[reporter],
               "condition": exp.well_to_trait_value(well, design.condition_trait),
               "replicate": exp.well_to_trait_value(well, design.replicate_trait)}
        reads = []
        for name, channel in channels.items():
            loaded = corrected[channel]
            row[name], n = window_mean(exp.data[(loaded, "time")], exp.data[(loaded, well)], time_h, window_h)
            reads.append(n)
        row["reads_in_window"] = min(reads)
        rows.append(row)
    return pd.DataFrame(rows)


def add_ratios(table, design):
    """Subtract the mean autofluorescence of each condition from the red signal, then take R:G.

    A well gets no ratio if it has no green signal or, when OD is known, its OD is below min_od,
    because the ratio isn't meaningful there.
    """
    table = table.copy()
    autofluorescence = table[table["role"] == ROLE_AUTOFLUORESCENCE].groupby("condition")["red"].agg(["mean", "std", "count"])
    table["red_autofluorescence"] = table["condition"].map(autofluorescence["mean"])
    table["red_corrected"] = table["red"] - table["red_autofluorescence"]
    usable = table["green"] > 0
    if "od" in table:
        usable &= table["od"] >= design.min_od
    table["ratio"] = (table["red_corrected"] / table["green"]).where(usable & (table["role"] != ROLE_AUTOFLUORESCENCE))
    return table


def _relative_variance(mean, se):
    return (se / mean) ** 2 if mean != 0 else np.nan


def readthrough(table, design):
    """Readthrough per test reporter and condition, with errors from biological replicates.

    Returns (summary, per_replicate). In the summary, readthrough_percent is 100 x mean test R:G /
    mean in-frame R:G. se_percent propagates the standard errors of those two means, including
    the uncertainty of the autofluorescence, which matters when the test signal is small. It is
    NaN with fewer than 2 replicates on either side. sd_replicates_percent is the sample SD of
    the test replicates, each divided by the mean in-frame R:G.
    """
    table = add_ratios(table, design)
    summary, per_replicate = [], []
    autofluorescence = table[table["role"] == ROLE_AUTOFLUORESCENCE]
    for condition, in_condition in table.groupby("condition", sort=True):
        in_frame = in_condition[in_condition["role"] == ROLE_IN_FRAME]
        in_frame_ratio = in_frame["ratio"].dropna()
        blank = autofluorescence[autofluorescence["condition"] == condition]["red"].dropna()
        blank_se = blank.std(ddof=1) / np.sqrt(len(blank)) if len(blank) > 1 else np.nan
        for test_value in design.test:
            test = in_condition[in_condition["reporter"] == test_value]
            test_ratio = test["ratio"].dropna()
            row = {"reporter": test_value, "condition": condition, "n_test": len(test_ratio),
                   "n_in_frame": len(in_frame_ratio), "n_autofluorescence": len(blank),
                   "test_ratio": test_ratio.mean(), "in_frame_ratio": in_frame_ratio.mean(),
                   "red_autofluorescence": blank.mean() if len(blank) else np.nan}
            if len(test_ratio) == 0 or len(in_frame_ratio) == 0 or not row["in_frame_ratio"]:
                row.update(readthrough_percent=np.nan, se_percent=np.nan, sd_replicates_percent=np.nan)
                summary.append(row)
                continue
            percent = 100 * test_ratio / row["in_frame_ratio"]
            row["readthrough_percent"] = 100 * row["test_ratio"] / row["in_frame_ratio"]
            row["sd_replicates_percent"] = percent.std(ddof=1) if len(percent) > 1 else np.nan
            if len(test_ratio) > 1 and len(in_frame_ratio) > 1:
                test_se = test_ratio.std(ddof=1) / np.sqrt(len(test_ratio))
                in_frame_se = in_frame_ratio.std(ddof=1) / np.sqrt(len(in_frame_ratio))
                variance = _relative_variance(row["test_ratio"], test_se) + _relative_variance(row["in_frame_ratio"], in_frame_se)
                if np.isfinite(blank_se):
                    variance += _relative_variance(row["test_ratio"], blank_se / test["green"].mean())
                    variance += _relative_variance(row["in_frame_ratio"], blank_se / in_frame["green"].mean())
                row["se_percent"] = abs(row["readthrough_percent"]) * np.sqrt(variance)
            else:
                row["se_percent"] = np.nan
            summary.append(row)
            for (_, source), value in zip(test.dropna(subset=["ratio"]).iterrows(), percent):
                per_replicate.append({"reporter": test_value, "condition": condition, "replicate": source["replicate"],
                                      "well": source["well"], "ratio": source["ratio"], "readthrough_percent": value})
    return pd.DataFrame(summary), pd.DataFrame(per_replicate)


def time_course(exp, corrected, design, wells, times, window_h=0.25):
    """The summary of readthrough() at each time, with a time_h column."""
    parts = []
    for t in times:
        summary, _ = readthrough(reporter_table(exp, corrected, design, wells, t, window_h), design)
        parts.append(summary.assign(time_h=t))
    return pd.concat(parts, ignore_index=True)


def raw_well_series(path, sheet, label, well):
    """Time (h) and values of a well straight from a Tecan sheet, without the library.

    Returns None for other formats, so the check can say it was not done.
    """
    import openpyxl

    rows = list(openpyxl.load_workbook(path, data_only=True, read_only=True)[sheet].iter_rows(values_only=True))
    starts = [i for i, row in enumerate(rows) if str(row[0]).startswith("Cycle Nr.")]
    name = str(label).replace("Label:", "").strip()
    named = [i for i in starts if str(rows[i - 1][0]).replace("Label:", "").strip() == name]
    if not named and len(starts) == 1:
        named = starts
    if not named:
        return None
    start = named[0]
    header = rows[start]

    def numbers(values):
        return np.array([v if isinstance(v, (int, float)) else np.nan for v in values], dtype=float)

    if str(header[1]).startswith("Time"):
        if well not in header:
            return None
        column = header.index(well)
        data = []
        for row in rows[start + 1:]:
            if row[1] is None or str(row[0]).startswith("End Time"):
                break
            data.append(row)
        return numbers([r[1] for r in data]) / 3600, numbers([r[column] for r in data])
    block = {}
    for row in rows[start:]:
        if row[0] is None:
            break
        block[row[0]] = row[1:]
    if well not in block or "Time [s]" not in block:
        return None
    return numbers(block["Time [s]"]) / 3600, numbers(block[well])


def _status(ok, warn=False):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


def sanity_checks(exp, corrected, background, design, wells, time_h, window_h=0.25, expected_percent=(0.1, 10.0),
                  control_wells=(), measurements=None, reader_file=None, sheet=None, stability_times=None,
                  od_jump=0.05):
    """Checks that the readthrough number can be trusted. Returns a table of check, status, detail.

    FAIL means don't interpret the number until it is explained. WARN means look at it and report it.
    """
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    raw = reporter_table(exp, corrected, design, wells, time_h, window_h)
    table = add_ratios(raw, design)
    summary, _ = readthrough(raw, design)
    conditions = sorted(table["condition"].unique(), key=float if all(_is_number(c) for c in table["condition"]) else str)
    groups = table.groupby("role")

    counts = table.groupby(["condition", "role"])["well"].count().unstack(fill_value=0)
    per_test = table[table["role"] == ROLE_TEST].groupby(["condition", "reporter"])["well"].count()
    missing = [role for role in (ROLE_IN_FRAME, ROLE_AUTOFLUORESCENCE, ROLE_TEST) if role not in counts or (counts[role] == 0).any()]
    fewest = min([int(counts[ROLE_IN_FRAME].min()) if ROLE_IN_FRAME in counts else 0, int(per_test.min()) if len(per_test) else 0])
    if missing:
        add("Design complete", "FAIL", f"No wells in some conditions for: {', '.join(missing)}.")
    elif fewest < 2:
        add("Design complete", "FAIL", "Fewer than 2 biological replicates in some condition, so there's no error estimate.")
    elif fewest == 2:
        add("Design complete", "WARN", "All parts are present in every condition, but with only 2 biological replicates, so the error estimate is rough.")
    else:
        add("Design complete", "PASS", f"All parts are present in every condition, with at least {fewest} biological replicates.")
    no_reads = table.loc[table["reads_in_window"] == 0, "well"].tolist()
    add("Reads at the chosen time", _status(not no_reads), f"No reads within {window_h} h of {time_h} h: {no_reads}." if no_reads
        else f"Every well has reads within {window_h} h of {time_h} h (at least {int(table['reads_in_window'].min())}).")

    if len(background.controls):
        unused = background.controls.loc[~background.controls["used"], "well"].tolist()
        add("Medium controls clean", _status(not unused, warn=True),
            f"Not used: {unused}." if unused else f"All {len(background.controls)} controls passed the contamination check.")
    else:
        add("Medium controls clean", "WARN", "This plate has no medium controls.")
    for channel in [c for c in (design.od, design.green, design.red) if c]:
        residual = [window_mean(exp.data[(corrected[channel], "time")], exp.data[(corrected[channel], w)], time_h, window_h)[0] for w in control_wells]
        residual = [abs(v) for v in residual if np.isfinite(v)]
        if not residual:
            continue
        signal = {design.od: "od", design.green: "green", design.red: "red"}[channel]
        reference = table.loc[table["role"] != ROLE_AUTOFLUORESCENCE, signal].median()
        fraction = max(residual) / abs(reference) if reference else np.inf
        add(f"Medium stays flat in {channel}", _status(fraction < 0.05, warn=True),
            f"After subtraction the medium controls are off by up to {max(residual):.4g}, which is {100 * fraction:.1f}% of the typical well.")

    if "od" in table:
        low = table.loc[table["od"] < design.min_od, "well"].tolist()
        add("Wells have grown", _status(not low), f"OD below {design.min_od} at {time_h} h: {low}." if low
            else f"All reporter wells are above OD {design.min_od} at {time_h} h (lowest {table['od'].min():.2f}).")
        jumps = []
        for well in table["well"]:
            y = np.asarray(exp.data[(corrected[design.od], well)], dtype=float)
            y = y[np.isfinite(y)][1:]
            step = np.abs(np.diff(y))
            if len(step) and step.max() > od_jump:
                jumps.append(f"{well} ({step.max():.2f})")
        add("No sudden OD jumps", _status(not jumps, warn=True),
            f"OD changes by more than {od_jump} between consecutive reads, which suggests clumps, a bubble or condensation, in: {', '.join(jumps)}." if jumps
            else f"No well's OD changes by more than {od_jump} between consecutive reads.")

    in_frame = table[table["role"] == ROLE_IN_FRAME]
    autofluor = table[table["role"] == ROLE_AUTOFLUORESCENCE]
    test = table[table["role"] == ROLE_TEST]
    weak = table.loc[table["green"] <= 0, "well"].tolist()
    add("Green signal present", _status(not weak), f"No green signal above medium in: {weak}." if weak
        else f"All reporter wells have green above medium (lowest {table['green'].min():.0f}).")

    fraction = autofluor.groupby("condition")["red"].mean().abs() / in_frame.groupby("condition")["red"].mean()
    worst = fraction.max()
    add("Autofluorescence small next to the intact reporter", "PASS" if worst < 0.05 else ("WARN" if worst < 0.2 else "FAIL"),
        f"Red of the no-red-gene wells is at most {100 * worst:.1f}% of the intact reporter's.")

    if len(autofluor) >= 3 and autofluor["green"].nunique() > 1:
        slope = np.polyfit(autofluor["green"], autofluor["red"], 1)[0]
        leak = slope * in_frame["green"].median() / in_frame["red"].median()
        add("No green signal leaking into red", _status(abs(leak) < 0.05, warn=True),
            f"Across the no-red-gene wells, red changes by {slope:.2g} per unit of green. At the intact reporter's green that's {100 * leak:.1f}% of its red.")

    ratios = in_frame.groupby("condition")["ratio"].mean().dropna()
    if len(ratios) > 1:
        spread = ratios.std(ddof=1) / ratios.mean()
        add("Intact R:G the same across conditions", _status(spread < 0.2, warn=True),
            f"Red and green share one transcript, so this should barely move. Mean R:G runs from {ratios.min():.3f} to {ratios.max():.3f} (CV {100 * spread:.0f}%).")

    if all(_is_number(c) for c in conditions) and len(conditions) >= 3:
        rising = []
        for reporter, part in table.groupby("reporter"):
            mean = part.groupby("condition")["green"].mean()
            mean = mean.loc[sorted(mean.index, key=float)]
            rho = pd.Series(mean.values).corr(pd.Series([float(i) for i in mean.index]), method="spearman")
            rising.append((reporter, rho, mean.iloc[-1] / mean.iloc[0]))
        low = [r for r, rho, fold in rising if rho < 0.8 or fold < 1.2]
        add("Green responds to the condition", _status(not low, warn=True),
            "; ".join(f"{r}: {fold:.1f}-fold, Spearman {rho:.2f}" for r, rho, fold in rising))

    for name, part in (("intact", in_frame), ("test", test)):
        if part["ratio"].notna().sum() == 0:
            continue
        spread = part.groupby(["reporter", "condition"])["ratio"].agg(lambda v: (v.max() - v.min()) / abs(v.mean()) if v.mean() else np.nan)
        add(f"Biological replicates agree ({name})", _status(spread.max() < 0.25, warn=True),
            f"Replicates' R:G differ by up to {100 * spread.max():.0f}% of their mean.")

    for reporter, part in test.groupby("reporter"):
        fold = part.groupby("condition")["green"].mean() / in_frame.groupby("condition")["green"].mean()
        add(f"Green matches the intact reporter ({reporter})", _status(((fold > 0.67) & (fold < 1.5)).all(), warn=True),
            f"Green comes before the mutation, so these should match. {reporter} / intact runs from {fold.min():.2f} to {fold.max():.2f}.")

    low, high = expected_percent
    values = summary["readthrough_percent"].dropna()
    outside = values[(values < low) | (values > high)]
    if len(values) == 0:
        add("Readthrough in the expected range", "FAIL", "No readthrough value could be calculated.")
    elif len(outside):
        extra = " Values this high mean the test wells are nearly as red as the intact reporter. Check the stop codon in the plasmid and the identity and positions of the strains before reading anything into them." if values.max() > 50 else ""
        add("Readthrough in the expected range", "FAIL",
            f"Readthrough runs from {values.min():.3g}% to {values.max():.3g}%, against an expected {low}–{high}%.{extra}")
    else:
        add("Readthrough in the expected range", "PASS", f"Readthrough runs from {values.min():.3g}% to {values.max():.3g}%, within the expected {low}–{high}%.")

    if stability_times:
        course = time_course(exp, corrected, design, wells, stability_times, window_h)
        by_group = course.groupby(["reporter", "condition"])["readthrough_percent"]
        ratio = by_group.max() / by_group.min()
        add("Readthrough stable over time", _status((ratio < 1.5).all(), warn=True),
            f"Over {min(stability_times)}–{max(stability_times)} h, readthrough in one condition changes by up to {ratio.max():.2f}-fold.")

    if reader_file and measurements:
        problems, compared = [], 0
        wells_table = background.wells
        examples = [part["well"].iloc[0] for _, part in table.groupby("reporter")]
        for channel in [c for c in (design.red, design.green, design.od) if c]:
            for well in examples:
                series = raw_well_series(reader_file, sheet, measurements[channel], well)
                if series is None:
                    continue
                entry = wells_table[(wells_table["measurement"] == channel) & (wells_table["well"] == well)].iloc[0]
                expected = window_mean(series[0], series[1], time_h, window_h)[0] - entry["background"] + (entry["offset"] if "offset" in entry and pd.notna(entry["offset"]) else 0.0)
                found = window_mean(exp.data[(corrected[channel], "time")], exp.data[(corrected[channel], well)], time_h, window_h)[0]
                compared += 1
                if not np.isclose(expected, found, rtol=1e-6, atol=1e-9):
                    problems.append(f"{channel} {well}: raw {expected:.6g}, pipeline {found:.6g}")
        if compared:
            add("Pipeline matches the raw sheet", _status(not problems), "; ".join(problems) if problems
                else f"Recomputed {compared} values directly from the reader sheet, without the library, and all match.")
        else:
            add("Pipeline matches the raw sheet", "WARN", "This reader format can't be rechecked automatically. Recompute a well by hand.")
    return pd.DataFrame(checks)


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC"]
INK, MUTED = "#333333", "#8a8a8a"


def _condition_positions(conditions):
    return {c: i for i, c in enumerate(sorted(conditions, key=lambda c: (float(c) if _is_number(c) else 0, str(c))))}


def plot_readthrough(summary, per_replicate, condition_label, expected_percent=None, title=None):
    """Readthrough against condition: mean ± SE, with each biological replicate as a dot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    positions = _condition_positions(summary["condition"].unique())
    reporters = list(summary["reporter"].unique())
    for index, reporter in enumerate(reporters):
        color = COLORS[index % len(COLORS)]
        part = summary[summary["reporter"] == reporter].sort_values("condition", key=lambda c: c.map(positions))
        x = part["condition"].map(positions) + 0.08 * (index - (len(reporters) - 1) / 2)
        ax.errorbar(x, part["readthrough_percent"], yerr=part["se_percent"], color=color, marker="o", markersize=7, linewidth=2,
                    capsize=3, label=f"{reporter} (mean ± SE)")
        points = per_replicate[per_replicate["reporter"] == reporter]
        if len(points):
            ax.scatter(points["condition"].map(positions) + 0.08 * (index - (len(reporters) - 1) / 2), points["readthrough_percent"],
                       s=24, facecolor="white", edgecolor=color, linewidth=1.2, zorder=3, label="biological replicates")
    if expected_percent:
        ax.axhspan(expected_percent[0], expected_percent[1], color="0.88", zorder=0, label=f"expected {expected_percent[0]}–{expected_percent[1]}%")
    ax.set_xticks(list(positions.values()), [str(c) for c in positions])
    ax.set_xlabel(condition_label, fontsize=12, color=INK)
    ax.set_ylabel("Readthrough (% of intact reporter's R:G)", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=13, color=INK)
    return fig, ax


def plot_ratio_panels(table, design, condition_label, title=None):
    """Red and green of every well at one time, by reporter, on two panels with their own scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    positions = _condition_positions(table["condition"].unique())
    for ax, signal, label in zip(axes, ("green", "red"), (f"{design.green}, background subtracted", f"{design.red}, background subtracted")):
        for index, (reporter, part) in enumerate(table.groupby("reporter", sort=False)):
            means = part.groupby("condition")[signal].mean()
            means = means.loc[sorted(means.index, key=lambda c: positions[c])]
            color = COLORS[index % len(COLORS)]
            ax.plot([positions[c] for c in means.index], means.values, color=color, linewidth=2, label=str(reporter))
            ax.scatter(part["condition"].map(positions), part[signal], s=22, facecolor="white", edgecolor=color, linewidth=1.1, zorder=3)
        ax.set_xticks(list(positions.values()), [str(c) for c in positions])
        ax.set_xlabel(condition_label, fontsize=11, color=INK)
        ax.set_ylabel(label, fontsize=11, color=INK)
        ax.tick_params(labelsize=9, colors=INK)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13, color=INK)
    return fig, axes


def plot_time_course(course, condition_label, chosen_time=None, title=None):
    """Readthrough over the run, one line per condition, one panel per test reporter."""
    import matplotlib.pyplot as plt

    reporters = list(course["reporter"].unique())
    fig, axes = plt.subplots(1, len(reporters), figsize=(6.5 * len(reporters), 4.2), squeeze=False, constrained_layout=True)
    conditions = sorted(course["condition"].unique(), key=lambda c: (float(c) if _is_number(c) else 0, str(c)))
    shades = plt.cm.Blues(np.linspace(0.35, 1, len(conditions)))
    for ax, reporter in zip(axes[0], reporters):
        part = course[course["reporter"] == reporter]
        for shade, condition in zip(shades, conditions):
            line = part[part["condition"] == condition]
            ax.plot(line["time_h"], line["readthrough_percent"], color=shade, linewidth=1.8, label=str(condition))
        if chosen_time is not None:
            ax.axvline(chosen_time, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("Time [h]", fontsize=11, color=INK)
        ax.set_ylabel("Readthrough (%)", fontsize=11, color=INK)
        ax.set_title(str(reporter), fontsize=12, color=INK)
        ax.tick_params(labelsize=9, colors=INK)
        ax.legend(title=condition_label, frameon=False, fontsize=8, title_fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13, color=INK)
    return fig, axes

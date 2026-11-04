"""Medium-background subtraction and QC for plate-reader experiments.

Works on a ``reader_functions.ReaderExperiment``. Every measurement channel is
corrected separately, and the rule for each well is:

1. A clean medium-control well exists in the well's medium: subtract one constant
   per medium and channel (the mean of the clean controls over the run, after
   each control's initial jump).
2. Otherwise, subtract the well's own start: the median of the reads after the
   initial jump. This is valid only if the start is plausibly medium alone.
3. Otherwise stop. ``BackgroundUnresolved`` says why, and the caller must supply a
   decision (``manual_background``, ``control_decisions``, ``exclude_wells`` or
   ``accept_own_start``). Nothing is guessed.

Typical use::

    result = rb.compute_background(exp, control_wells=["E10", "F10"], match_traits=["medium"])
    corrected = rb.apply_background(exp, result, offset={"OD": 0.0})   # {"OD": "OD_bg"}
    print(rb.summarize(result))
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class BackgroundUnresolved(Exception):
    """The background of some wells or controls needs a decision from the experimenter."""


@dataclass(frozen=True)
class BackgroundSettings:
    """Thresholds. OD values are in raw reader units; other channels scale to their own noise."""
    baseline_reads: int = 5          # reads in the start window, after the jump
    max_jump_reads: int = 3          # the initial jump is searched for in this many leading reads
    rolling_reads: int = 5           # running-median width, so single spikes don't count as a rise
    od_jump: float = 0.002           # a leading read this far from the plateau after it is a jump
    od_plateau: float = 0.004        # the reads after a jump must agree within this
    od_flat: float = 0.004           # a start window spanning more than this gets flagged
    od_contaminated_rise: float = 0.02   # a control rising more than this is contaminated
    od_suspect_change: float = 0.005     # a control moving more than this needs a human look
    od_control_outlier: float = 0.005    # a control this far from its medium's median needs a look
    od_max_start: float = 0.12       # a start above this is not medium alone (typical blank 0.08-0.095)
    od_start_outlier: float = 0.01   # a well starting this far above others in its medium gets flagged
    od_negative: float = 0.005       # corrected OD below minus this gets flagged
    min_relative_tol: float = 0.02   # other channels: tolerances are at least this fraction of the signal
    blank_drift_note: float = 0.05   # a control changing by this fraction of its constant gets flagged
    blank_drift_stop: float = 0.25   # beyond this, one constant does not describe the blank: ask


# ---------------------------------------------------------------------- helpers
def _resolve_trait(exp, trait):
    """Return the plate-map key for ``trait``, tolerating stray spaces in the plate map."""
    keys = [k for k in exp.plate_map_dict if k not in ("well",)]
    if trait in keys:
        return trait
    stripped = [k for k in keys if str(k).strip() == str(trait).strip()]
    if len(stripped) == 1:
        return stripped[0]
    raise KeyError(f"Trait {trait!r} is not in the plate map. Traits there: {keys}")


def _key_part(value):
    """One trait value in a form that compares reliably: 'LB ' and 'lb' agree, and so do 1, 1.0 and '1'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    try:
        return f"{float(text):.12g}"
    except ValueError:
        return text.casefold()


def _match_key(exp, well, traits):
    """Label for what must match between a well and its control, e.g. 'lb|0.5'."""
    if not traits:
        return "all"
    return "|".join(_key_part(exp.well_to_trait_value(well, trait)) for trait in traits)


def _normal_key(key):
    return "|".join(_key_part(part) for part in str(key).split("|"))


def _measured(y):
    return np.asarray(y, dtype=float)[np.isfinite(np.asarray(y, dtype=float))]


def _running_median(z, width):
    return pd.Series(z).rolling(width, center=True, min_periods=max(1, width // 2 + 1)).median().to_numpy()


def _tolerances(z, is_od, s):
    """(jump, plateau, flat, negative) tolerances for one trace."""
    if is_od:
        return s.od_jump, s.od_plateau, s.od_flat, s.od_negative
    early = z[:30]
    noise = 1.4826 * np.median(np.abs(np.diff(early))) / np.sqrt(2) if len(early) > 1 else 0.0
    jump = max(3 * noise, s.min_relative_tol * abs(np.median(early)), 1e-9)
    return jump, 2 * jump, 2 * jump, 2 * jump


def detect_jump(z, is_od=True, settings=None):
    """Number of leading reads to drop from the measured reads ``z``.

    Read k is part of the jump if the three reads after it agree with each other and
    read k is farther from them than they are from one another, in either direction.
    That keeps a later spike or the onset of growth from being read as a jump.
    """
    s = settings or BackgroundSettings()
    jump_tol, plateau_tol, _, _ = _tolerances(z, is_od, s)
    n_drop = 0
    for k in range(s.max_jump_reads):
        plateau = z[k + 1:k + 4]
        if len(plateau) < 3:
            break
        span = np.ptp(plateau)
        if span <= plateau_tol and abs(z[k] - np.median(plateau)) > max(jump_tol, span):
            n_drop = k + 1
    return n_drop


def jump_assessable(y, is_od=True, settings=None):
    """Whether the reads after the first read agree well enough for a jump to be detected at all.

    In a well already changing fast, such as one lysing in SDS, they don't. detect_jump then
    finds nothing, which doesn't mean the first read is sound.
    """
    s = settings or BackgroundSettings()
    z = _measured(y)
    if len(z) < 4:
        return False
    _, plateau_tol, _, _ = _tolerances(z, is_od, s)
    return bool(np.ptp(z[1:4]) <= plateau_tol)


def well_start(y, is_od=True, settings=None):
    """Describe the start of one trace: reads dropped, start value, and span of the start window."""
    s = settings or BackgroundSettings()
    z = _measured(y)
    if len(z) == 0:
        return {"jump_reads": np.nan, "start": np.nan, "start_span": np.nan, "flat": False}
    n_drop = detect_jump(z, is_od, s)
    window = z[n_drop:n_drop + s.baseline_reads]
    _, _, flat_tol, _ = _tolerances(z, is_od, s)
    span = float(np.ptp(window))
    return {"jump_reads": n_drop, "start": float(np.median(window)), "start_span": span, "flat": bool(span <= flat_tol)}


# ---------------------------------------------------------------------- controls
def assess_medium_controls(exp, control_wells, match_traits=(), od_measurement="OD", settings=None):
    """Classify each medium-control well as clean, suspect, contaminated, not_measured or unverified.

    Contamination is judged on OD, because that is what growth raises. A file with no
    OD channel can't be judged, so its controls come back ``unverified``.
    """
    s = settings or BackgroundSettings()
    traits = [_resolve_trait(exp, t) for t in match_traits]
    has_od = od_measurement is not None and (od_measurement, "time") in exp.data
    rows = []
    for well in control_wells:
        row = {"well": well, "match": _match_key(exp, well, traits), "start_od": np.nan,
               "jump_reads": np.nan, "rise_od": np.nan, "drop_od": np.nan,
               "status": "unverified", "reason": "no OD channel to judge contamination"}
        if has_od:
            z = _measured(exp.data[(od_measurement, well)])
            if len(z) == 0:
                row.update(status="not_measured", reason="no readings in the file")
            elif len(z) <= s.baseline_reads:
                row.update(status="unverified", reason=f"only {len(z)} reads, too few to judge")
            else:
                start = well_start(z, True, s)
                tail = z[start["jump_reads"]:]
                smooth = _running_median(tail, s.rolling_reads)
                rise = float(np.nanmax(smooth) - start["start"])
                drop = float(start["start"] - np.nanmin(smooth))
                row.update(start_od=start["start"], jump_reads=start["jump_reads"], rise_od=rise, drop_od=drop)
                if start["start"] > s.od_max_start:
                    row.update(status="contaminated", reason=f"starts at OD {start['start']:.3f}")
                elif rise > s.od_contaminated_rise:
                    row.update(status="contaminated", reason=f"rises by OD {rise:.3f}")
                elif rise > s.od_suspect_change:
                    row.update(status="suspect", reason=f"rises by OD {rise:.4f}")
                elif drop > s.od_suspect_change:
                    row.update(status="suspect", reason=f"falls by OD {drop:.4f}")
                else:
                    row.update(status="clean", reason="")
        rows.append(row)
    table = pd.DataFrame(rows, columns=["well", "match", "start_od", "jump_reads", "rise_od", "drop_od", "status", "reason"])

    # A control far from the others in its medium is suspect even if it is flat.
    for _, group in table.groupby("match"):
        usable = group[group["status"].isin(["clean", "suspect"])]
        if len(usable) == 2 and abs(usable["start_od"].iloc[0] - usable["start_od"].iloc[1]) > s.od_control_outlier:
            for idx in usable.index:
                if table.loc[idx, "status"] == "clean":
                    table.loc[idx, "status"] = "suspect"
                    table.loc[idx, "reason"] = (f"the two controls of this medium start at "
                                                f"{usable['start_od'].iloc[0]:.4f} and {usable['start_od'].iloc[1]:.4f}")
        if len(usable) < 3:
            continue
        center = usable["start_od"].median()
        for idx, row in usable.iterrows():
            if row["status"] == "clean" and abs(row["start_od"] - center) > s.od_control_outlier:
                table.loc[idx, "status"] = "suspect"
                table.loc[idx, "reason"] = f"starts at OD {row['start_od']:.4f}, others near {center:.4f}"
    return table


# ---------------------------------------------------------------------- background
@dataclass
class BackgroundResult:
    controls: pd.DataFrame     # one row per control well, with status and decision
    constants: pd.DataFrame    # one row per channel and medium that has a usable control
    wells: pd.DataFrame        # one row per channel and well: method, value and flags
    unresolved: list = field(default_factory=list)   # what still needs a decision
    settings: BackgroundSettings = field(default_factory=BackgroundSettings)
    corrected: dict = field(default_factory=dict)    # channel -> name of the corrected channel
    od_measurement: str = "OD"


def _manual_value(manual, measurement, well, key):
    """A manual background, looked up by well, then by medium, then for the whole channel."""
    entry = (manual or {}).get(measurement)
    if entry is None:
        return None
    if isinstance(entry, dict):
        if well in entry:
            return entry[well]
        for given, value in entry.items():
            if _normal_key(given) == key:
                return value
        return None
    return entry


def compute_background(exp, control_wells=None, match_traits=(), od_measurement="OD", measurements=None,
                       control_decisions=None, manual_background=None, exclude_wells=None,
                       accept_blank_drift=None, accept_own_start=None, settings=None, raise_if_unresolved=True):
    """Work out the background of every well and channel. Does not change ``exp``.

    Parameters
    ----------
    control_wells : list of str
        Wells holding medium alone. Never inferred from names, because 'control' and
        'ctrl' also mean other things in plate maps.
    match_traits : list of str
        Traits a control must share with a well to serve as its background, such as
        ["medium", "glucose"]. Leave empty if the whole plate has one medium.
    od_measurement : str or None
        The OD channel. It judges contamination and whether a well's start is medium alone.
    control_decisions : dict
        {well: "keep" | "drop"}, the experimenter's call on controls flagged by the QC.
    manual_background : dict
        The experimenter's blank where no clean control exists: {channel: value} for all
        such wells in the channel, or {channel: {match_key or well: value}} for some of
        them. Wells it doesn't cover keep their own start.
    exclude_wells : list of str
        Wells left out of correction altogether. They come out as NaN.
    accept_blank_drift : list of str
        Channels where the experimenter accepts one constant although the clean controls
        change by more than ``settings.blank_drift_stop`` over the run.
    accept_own_start : list of str or True
        Wells, or True for all, where the experimenter accepts the well's own start although
        it sits above the others in its medium or the well later falls below it.
    """
    s = settings or BackgroundSettings()
    traits = [_resolve_trait(exp, t) for t in match_traits]
    decisions = control_decisions or {}
    channels = list(measurements) if measurements is not None else list(exp.measurements)
    loaded = set(exp.wells)
    excluded = set(exclude_wells or [])
    accepted_start = set(loaded) if accept_own_start is True else set(accept_own_start or [])
    control_wells = list(control_wells or [])
    for name, wells_given in (("control_wells", control_wells), ("exclude_wells", excluded),
                              ("control_decisions", decisions), ("accept_own_start", accepted_start)):
        unknown = [w for w in wells_given if w not in loaded]
        if unknown:
            raise KeyError(f"{name} names wells that aren't loaded from the plate map: {sorted(unknown)}")
    # An excluded control is out altogether: it can't set the constant for other wells.
    control_wells = [w for w in control_wells if w not in excluded]
    unknown_channels = sorted(set(accept_blank_drift or []) - set(channels))
    if unknown_channels:
        raise ValueError(f"accept_blank_drift names channels {unknown_channels}. Loaded channels: {channels}")

    has_od = od_measurement is not None and (od_measurement, "time") in exp.data
    controls = assess_medium_controls(exp, control_wells, match_traits, od_measurement, s)
    unresolved = []

    media = {_match_key(exp, well, traits) for well in exp.wells}
    for m, entry in (manual_background or {}).items():
        if m not in channels:
            raise ValueError(f"manual_background names channel {m!r}. Loaded channels: {channels}")
        for key in (entry if isinstance(entry, dict) else {}):
            if key not in loaded and _normal_key(key) not in media:
                raise ValueError(f"manual_background[{m!r}] has key {key!r}, which is neither a well nor a medium. "
                                 f"Media on this plate: {sorted(media)}")

    def _use(row):
        if traits and any(not part for part in row["match"].split("|")):
            unresolved.append(f"control {row['well']} has no value for {traits} in the plate map, so it can't be "
                              "matched to a medium. Fill it in, or list it in control_decisions as 'drop'."
                              if decisions.get(row["well"]) != "drop" else "")
            return False
        decision = decisions.get(row["well"])
        if decision is not None:
            if decision not in ("keep", "drop"):
                raise ValueError(f"control_decisions[{row['well']!r}] must be 'keep' or 'drop'")
            return decision == "keep"
        if row["status"] == "clean":
            return True
        if row["status"] in ("suspect", "unverified"):
            unresolved.append(f"control {row['well']} ({row['match']}) is {row['status']}: {row['reason']}. "
                              "Set control_decisions to 'keep' or 'drop'.")
        return False

    controls["used"] = [_use(row) for _, row in controls.iterrows()] if len(controls) else []
    unresolved = [item for item in unresolved if item]
    usable = controls[controls["used"]] if len(controls) else controls

    # one constant per channel and medium
    constant_rows = []
    for m in channels:
        for key, group in usable.groupby("match") if len(usable) else []:
            per_well, traces, contributing = [], [], []
            for well in group["well"]:
                z = _measured(exp.data[(m, well)])
                if len(z) == 0:
                    continue
                n_drop = detect_jump(z, m == od_measurement, s)
                per_well.append(float(np.mean(z[n_drop:])))
                traces.append(z[n_drop:])
                contributing.append(well)
            if not per_well:
                continue
            length = min(len(t) for t in traces)
            mean_trace = _running_median(np.mean([t[:length] for t in traces], axis=0), s.rolling_reads)
            drift = float(np.nanmax(mean_trace) - np.nanmin(mean_trace))
            value = float(np.mean(per_well))
            fraction = drift / abs(value) if value else (np.inf if drift > 0 else 0.0)
            constant_rows.append({"measurement": m, "match": key, "background": value,
                                  "n_controls": len(per_well), "wells": ",".join(contributing),
                                  "drift": drift, "drift_fraction": fraction})
    constants = pd.DataFrame(constant_rows, columns=["measurement", "match", "background", "n_controls",
                                                     "wells", "drift", "drift_fraction"])
    for row in constants.itertuples(index=False):
        if row.drift_fraction > s.blank_drift_stop and row.measurement not in set(accept_blank_drift or []):
            unresolved.append(
                f"{row.measurement}, medium '{row.match}': the clean controls change by {row.drift:.4g} over the run, "
                f"{row.drift_fraction:.0%} of the constant {row.background:.4g}, so one constant does not describe this "
                "blank. Check for crosstalk or drift, then either list the channel in accept_blank_drift or decide "
                "on another treatment for it.")

    # every well, every channel
    control_set = set(control_wells)
    dropped = set(controls.loc[~controls["used"], "well"]) if len(controls) else set()
    cells_at_start, no_od, no_trait = {}, {}, []
    rows = []
    for m in channels:
        is_od = m == od_measurement
        for well in exp.wells:
            y = np.asarray(exp.data[(m, well)], dtype=float)
            key = _match_key(exp, well, traits)
            start = well_start(y, is_od, s)
            row = {"measurement": m, "well": well, "role": "control" if well in control_set else "sample",
                   "match": key, "method": None, "background": np.nan, "jump_reads": start["jump_reads"],
                   "start": start["start"], "start_span": start["start_span"], "flags": []}
            if not np.isfinite(y).any():
                row.update(method="not_measured")
            elif well in excluded:
                row.update(method="excluded")
            elif well in dropped:
                row.update(method="control_not_used")
            elif traits and any(not part for part in key.split("|")):
                row.update(method="unresolved")
                if well not in no_trait:
                    no_trait.append(well)
            else:
                match = constants[(constants["measurement"] == m) & (constants["match"] == key)]
                manual = _manual_value(manual_background, m, well, key)
                if len(match):
                    if isinstance((manual_background or {}).get(m), dict) and manual is not None:
                        raise ValueError(f"manual_background[{m!r}] covers {well}, which has a clean control. "
                                         "Remove the manual value, or drop the control with control_decisions.")
                    row.update(method="medium_control", background=float(match["background"].iloc[0]))
                    if is_od and start["start"] - row["background"] < -s.od_negative:
                        row["flags"].append("starts_below_control")
                elif manual is not None:
                    row.update(method="manual", background=float(manual))
                else:
                    if has_od:
                        od_start = start if is_od else well_start(exp.data[(od_measurement, well)], True, s)
                        if not np.isfinite(od_start["start"]):
                            row.update(method="unresolved")
                            no_od.setdefault((m, key), []).append(well)
                        elif od_start["start"] > s.od_max_start:
                            row.update(method="unresolved")
                            cells_at_start.setdefault((m, key), []).append((well, od_start["start"]))
                        else:
                            row.update(method="well_start", background=start["start"])
                            if not start["flat"]:
                                row["flags"].append("start_not_flat")
                    else:
                        row.update(method="unresolved")
                        no_od.setdefault((m, key), []).append(well)
            rows.append(row)
    wells = pd.DataFrame(rows)

    doubts_high, doubts_fall = [], []
    if has_od and od_measurement in channels:
        od_rows = wells[(wells["measurement"] == od_measurement) & (wells["role"] == "sample") & wells["start"].notna()]
        for key, group in od_rows.groupby("match"):
            reference = group if len(group) >= 3 else od_rows
            center = reference["start"].median()
            high = group[(group["method"] == "well_start") & (group["start"] - center > s.od_start_outlier)]
            for idx in high.index:
                wells.at[idx, "flags"] = wells.at[idx, "flags"] + ["start_above_others"]
                if wells.at[idx, "well"] not in accepted_start:
                    doubts_high.append(wells.at[idx, "well"])
        for idx in od_rows[od_rows["method"] == "well_start"].index:
            z = _measured(exp.data[(od_measurement, wells.at[idx, "well"])])[int(wells.at[idx, "jump_reads"]):]
            if len(z) and np.nanmin(_running_median(z, s.rolling_reads)) - wells.at[idx, "start"] < -s.od_negative:
                wells.at[idx, "flags"] = wells.at[idx, "flags"] + ["dips_below_background"]
                if wells.at[idx, "well"] not in accepted_start:
                    doubts_fall.append(wells.at[idx, "well"])

    if no_trait:
        unresolved.append(f"Wells {', '.join(no_trait)} have no value for {traits} in the plate map, so they can't be "
                          "matched to a medium. Fill it in, or leave them out with exclude_wells.")
    if traits and len(usable):
        sample_keys = set(wells.loc[wells["role"] == "sample", "match"])
        for row in usable.itertuples(index=False):
            if row.match not in sample_keys:
                unresolved.append(f"control {row.well} ({row.match}) matches no sample well, so it would be ignored. "
                                  "Check how the medium is written in the plate map.")
    if doubts_high or doubts_fall:
        parts = []
        if doubts_high:
            parts.append(f"{len(doubts_high)} start more than {s.od_start_outlier} OD above others in their medium "
                         f"({', '.join(doubts_high[:12])}{' ...' if len(doubts_high) > 12 else ''})")
        if doubts_fall:
            parts.append(f"{len(doubts_fall)} later fall more than {s.od_negative} OD below their start "
                         f"({', '.join(doubts_fall[:12])}{' ...' if len(doubts_fall) > 12 else ''})")
        unresolved.append("Wells using their own start, which may hold more than medium: " + "; ".join(parts) + ". "
                          "Give a blank with manual_background, leave them out with exclude_wells, or accept their "
                          "own start with accept_own_start.")

    for (m, key), found in sorted(cells_at_start.items(), key=str):
        names = ", ".join(w for w, _ in found[:8]) + (" ..." if len(found) > 8 else "")
        top = max(v for _, v in found)
        unresolved.append(
            f"{m}, medium '{key}': {len(found)} wells start above OD {s.od_max_start} (up to {top:.3f}), so their "
            f"start may not be medium alone, and there is no clean control: {names}. Give a blank with "
            "manual_background, for the channel, this medium or these wells, or leave the wells out with exclude_wells.")
    for (m, key), found in sorted(no_od.items(), key=str):
        unresolved.append(f"{m}, medium '{key}': no clean control and no OD reads to show that {len(found)} wells "
                          f"({', '.join(found[:12])}) start in medium alone. Set manual_background or exclude_wells.")

    result = BackgroundResult(controls=controls, constants=constants, wells=wells, unresolved=unresolved, settings=s,
                              od_measurement=od_measurement)
    if unresolved and raise_if_unresolved:
        raise BackgroundUnresolved("Background needs a decision:\n- " + "\n- ".join(unresolved))
    return result


def apply_background(exp, result, offset=None, suffix="_bg"):
    """Add a background-subtracted copy of each channel to ``exp.data``; raw data stay as they are.

    ``offset`` is {channel: value}, added after subtraction, and defaults to 0.
    Controls that were not used and unmeasured wells come out as NaN.
    Returns {channel: corrected channel name}.
    """
    if result.unresolved:
        raise BackgroundUnresolved("Resolve these first:\n- " + "\n- ".join(result.unresolved))
    s = result.settings
    offset = offset or {}
    corrected = {}
    flags_col = result.wells["flags"].tolist()
    minima = []
    for pos, row in enumerate(result.wells.itertuples(index=False)):
        m, well = row.measurement, row.well
        y = np.asarray(exp.data[(m, well)], dtype=float)
        new_name = f"{m}{suffix}"
        corrected[m] = new_name
        exp.data[(new_name, "time")] = exp.data[(m, "time")]
        if np.isfinite(row.background):
            value = y - row.background
            exp.data[(new_name, well)] = value + float(offset.get(m, 0.0))
            z = value[np.isfinite(value)]
            n_drop = int(row.jump_reads) if np.isfinite(row.jump_reads) else 0
            low = float(np.nanmin(_running_median(z[n_drop:], s.rolling_reads))) if len(z) > n_drop else np.nan
            minima.append(low)
            _, _, _, negative_tol = _tolerances(z, m == result.od_measurement, s)
            if np.isfinite(low) and low < -negative_tol and row.role == "sample" and "dips_below_background" not in row.flags:
                flags_col[pos] = list(row.flags) + ["dips_below_background"]
        else:
            exp.data[(new_name, well)] = np.full(y.shape, np.nan)
            minima.append(np.nan)
    result.wells["flags"] = flags_col
    result.wells["min_corrected"] = minima
    result.wells["offset"] = [float(offset.get(m, 0.0)) for m in result.wells["measurement"]]
    result.corrected = corrected
    return corrected


def channel_ratio(exp, numerator, denominator, new_name, min_denominator):
    """Add ``numerator / denominator`` as a channel, blank where the denominator is below ``min_denominator``.

    Channels of one run can differ by a read if it was stopped mid-cycle, so the
    shorter length is used.
    """
    time = np.asarray(exp.data[(numerator, "time")], dtype=float)
    for well in exp.wells:
        a = np.asarray(exp.data[(numerator, well)], dtype=float)
        b = np.asarray(exp.data[(denominator, well)], dtype=float)
        n = min(len(a), len(b))
        out = np.full(a.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:n] = np.where(b[:n] >= min_denominator, a[:n] / b[:n], np.nan)
        exp.data[(new_name, well)] = out
    exp.data[(new_name, "time")] = time
    return new_name


# ---------------------------------------------------------------------- reporting
def trim_trailing_empty_reads(exp):
    """Drop reads after the last timestamp. Returns how many were dropped per channel.

    Plex exports keep columns for cycles that never ran. Their time is NaN, which breaks
    axis limits and time lookups in the library's plotting functions.
    """
    dropped = {}
    for m in exp.measurements:
        time = np.asarray(exp.data[(m, "time")], dtype=float)
        finite = np.flatnonzero(np.isfinite(time))
        n_keep = int(finite[-1]) + 1 if len(finite) else 0
        dropped[m] = len(time) - n_keep
        for key in [k for k in exp.data if k[0] == m]:
            exp.data[key] = np.asarray(exp.data[key], dtype=float)[:n_keep]
    return dropped


def blank_time_windows(exp, windows):
    """Set reads inside the given time windows to NaN, in every well and channel.

    ``windows`` is a list of (start_h, end_h), for events that disturb the whole plate, such as
    the plate being taken out. Returns the number of reads blanked per channel.
    """
    blanked = {}
    for m in exp.measurements:
        time = np.asarray(exp.data[(m, "time")], dtype=float)
        inside = np.zeros(time.shape, dtype=bool)
        for start, end in windows:
            inside |= (time >= start) & (time <= end)
        blanked[m] = int(inside.sum())
        for key in [k for k in exp.data if k[0] == m and k[1] != "time"]:
            values = np.array(exp.data[key], dtype=float)
            values[inside] = np.nan
            exp.data[key] = values
    return blanked


def verify_loaded(exp):
    """One row per channel: reads, time span, wells with data. Raises if time runs backwards."""
    rows = []
    for m in exp.measurements:
        time = np.asarray(exp.data[(m, "time")], dtype=float)
        finite = time[np.isfinite(time)]
        if len(finite) > 1 and not np.all(np.diff(finite) > 0):
            raise ValueError(f"Time in channel {m!r} does not increase. Check the sheet and the reader parser.")
        with_data = [w for w in exp.wells if np.isfinite(np.asarray(exp.data[(m, w)], dtype=float)).any()]
        no_data = [w for w in exp.wells if w not in set(with_data)]
        rows.append({"measurement": m, "reads": len(finite),
                     "first_h": float(finite[0]) if len(finite) else np.nan,
                     "last_h": float(finite[-1]) if len(finite) else np.nan,
                     "median_interval_min": float(np.median(np.diff(finite)) * 60) if len(finite) > 1 else np.nan,
                     "wells_with_data": len(with_data), "wells_without_data": ",".join(map(str, no_data))})
    return pd.DataFrame(rows)


def summarize(result):
    """Plain-text account of what was subtracted and what needs a look."""
    lines = []
    if len(result.controls):
        lines.append("Medium controls:")
        for row in result.controls.itertuples(index=False):
            use = "used" if row.used else "NOT used"
            why = f" ({row.reason})" if row.reason else ""
            lines.append(f"  {row.well} [{row.match}]: {row.status}{why}, {use}")
    else:
        lines.append("Medium controls: none on this plate")
    for row in result.constants.itertuples(index=False):
        lines.append(f"Constant {row.measurement} [{row.match}]: {row.background:.5g} from {row.n_controls} controls "
                     f"({row.wells}); drift over the run {row.drift:.3g} ({row.drift_fraction:.1%})")
        if row.drift_fraction > result.settings.blank_drift_note:
            lines.append(f"FLAG blank_drift in {row.measurement} [{row.match}]: the blank moves {row.drift_fraction:.1%} "
                         "around one constant")
    counts = result.wells.groupby(["measurement", "match", "method"]).size()
    for (m, key, method), n in counts.items():
        where = "" if key == "all" else f" [{key}]"
        lines.append(f"{m}{where}: {n} wells by {method}")
    flagged = result.wells[result.wells["flags"].map(len) > 0]
    for flag in sorted({f for fl in flagged["flags"] for f in fl}):
        hits = flagged[flagged["flags"].map(lambda fl: flag in fl)]
        for m, group in hits.groupby("measurement"):
            names = ",".join(group["well"])
            lines.append(f"FLAG {flag} in {m}: {names}")
    for item in result.unresolved:
        lines.append(f"UNRESOLVED: {item}")
    return "\n".join(lines)


def background_table(result):
    """The per-well table, ready to save as CSV."""
    table = result.wells.copy()
    table["flags"] = table["flags"].map(";".join)
    return table


def plot_background_qc(exp, result, measurement, save_path=None, max_start_reads=15):
    """Three panels: control traces, start of the sample wells, and background across the plate."""
    import matplotlib.pyplot as plt

    INK, MUTED, USED, NOT_USED = "#333333", "#9a9a9a", "#2a6f97", "#b23a48"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    time = np.asarray(exp.data[(measurement, "time")], dtype=float)
    is_od = measurement == result.od_measurement

    ax = axes[0]
    drawn = {}
    for row in result.controls.itertuples(index=False):
        color, style, label = (USED, "-", "control, used") if row.used else (NOT_USED, "--", "control, not used")
        line, = ax.plot(time, exp.data[(measurement, row.well)], color=color, linestyle=style, linewidth=1.2, label=label)
        drawn[label] = line
    constants = result.constants[result.constants["measurement"] == measurement]
    finite_time = time[np.isfinite(time)]
    for row in constants.itertuples(index=False):
        drawn["constant subtracted"], = ax.plot(
            [finite_time[0], finite_time[-1]], [row.background, row.background],
            color=INK, linewidth=1.5, linestyle=":", label="constant subtracted")
    if drawn:
        ax.legend(handles=list(drawn.values()), frameon=False, fontsize=9)
        ax.set_xlabel("Time [h]", fontsize=11)
        ax.set_ylabel(f"{measurement}, raw", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No medium controls used", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Medium controls", fontsize=12)

    ax = axes[1]
    samples = result.wells[(result.wells["measurement"] == measurement) & (result.wells["role"] == "sample")]
    for row in samples.itertuples(index=False):
        if np.isfinite(row.background):
            y = np.asarray(exp.data[(measurement, row.well)], dtype=float)[:max_start_reads] - row.background
            ax.plot(np.arange(len(y)), y, color=USED, alpha=0.35, linewidth=0.8)
    ax.axhline(0, color=INK, linewidth=0.8)
    if is_od:
        ax.set_ylim(-0.04, 0.08)
    ax.set_title(f"Sample wells, first {max_start_reads} reads", fontsize=12)
    ax.set_xlabel("Read number (0 is the first read)", fontsize=11)
    ax.set_ylabel(f"{measurement} minus background", fontsize=11)

    ax = axes[2]
    wells_2d = exp.wells_2d
    table = result.wells[result.wells["measurement"] == measurement]
    lookup = dict(zip(table["well"], table["background"]))
    grid = np.full(wells_2d.shape, np.nan)
    for i in range(wells_2d.shape[0]):
        for j in range(wells_2d.shape[1]):
            grid[i, j] = lookup.get(wells_2d[i, j], np.nan)
    image = ax.imshow(grid, cmap="Blues", aspect="equal")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("background subtracted", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_xticks(range(wells_2d.shape[1]))
    ax.set_xticklabels([str(c + 1) for c in range(wells_2d.shape[1])])
    ax.set_yticks(range(wells_2d.shape[0]))
    ax.set_yticklabels([chr(ord("A") + r) for r in range(wells_2d.shape[0])])
    ax.grid(False)
    ax.set_title("Background by well (white: none)", fontsize=12)

    for ax in axes:
        ax.tick_params(labelsize=9)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axes

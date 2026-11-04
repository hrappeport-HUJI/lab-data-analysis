import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import util
from scipy.signal import savgol_filter
from scipy import stats
import seaborn as sns
import re
import plotly.colors as pc
from util import suppress_warnings
from itertools import product
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from colorsys import rgb_to_hsv, hsv_to_rgb
import datetime

util.set_style()

def _rgba_to_rgb_tuple(c):
    """Convert matplotlib color to RGB tuple in [0,1]."""
    from matplotlib.colors import to_rgb
    return to_rgb(c)


def _shade_from_base(base_rgb, rank_i, n_ranks, s_scale_range=(1.0, 0.6), v_scale_range=(0.7, 1.0)):
    """
    Adjust saturation/value of base color to create a distinct shade for a given rank.
    rank_i in [0..n_ranks-1].
    """
    r, g, b = base_rgb
    h, s, v = rgb_to_hsv(r, g, b)

    # Interpolate scales across reps
    if n_ranks <= 1:
        s_scale = 1.0
        v_scale = 1.0
    else:
        t = rank_i / (n_ranks - 1)
        s_scale = s_scale_range[0] + t * (s_scale_range[1] - s_scale_range[0])
        v_scale = v_scale_range[0] + t * (v_scale_range[1] - v_scale_range[0])

    s2 = np.clip(s * s_scale, 0.0, 1.0)
    v2 = np.clip(v * v_scale, 0.0, 1.0)
    r2, g2, b2 = hsv_to_rgb(h, s2, v2)
    return (r2, g2, b2)

# Function to generate colors based on trait values
def natural_sort_key(text):
    """
    Convert a string into a list of mixed strings and integers for natural sorting.
    Example: "1-10" becomes ["", 1, "-", 10, ""]
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


# def generate_color_palette(trait_values):
#     unique_values = list(dict.fromkeys(trait_values))  # Preserve order and uniqueness
#     color_scale = pc.qualitative.Plotly  # Choose a qualitative color scale
#     # color_scale = pc.qualitative.Alphabet  # Choose a qualitative color scale
#     colors = color_scale[:len(unique_values)]
#     return {value: color for value, color in zip(unique_values, colors)}


def generate_color_palette(trait_values):
    unique_values = list(dict.fromkeys(trait_values))  # Preserve order and uniqueness
    # Combine multiple qualitative palettes for more unique colors
    color_scale = (
        pc.qualitative.Plotly
        + pc.qualitative.D3
        + pc.qualitative.G10
        + pc.qualitative.Alphabet
    )
    # Cycle through colors if there are more values than available colors
    colors = [color_scale[i % len(color_scale)] for i in range(len(unique_values))]
    return {value: color for value, color in zip(unique_values, colors)}

# Only 4 line styles are available in plotly
def generate_line_styles(trait_values):
    unique_values = list(set(trait_values))
    line_styles = ['solid', 'dash', 'dot', 'dashdot'][:len(unique_values)]
    return {value: style for value, style in zip(unique_values, line_styles)}

def generate_marker_symbols(trait_values):
    unique_values = list(set(trait_values))
    symbols = ['circle', 'square', 'diamond', 'cross', 'x'][:len(unique_values)]
    return {value: symbol for value, symbol in zip(unique_values, symbols)}




class ReaderExperiment:
    def __init__(self, df_path, plate_map_path, measurements, reader="synergy", t_0=0, sheet_name=None,
                 missing_well_value=np.nan):
        self.df_path = df_path
        self.plate_map_path = plate_map_path
        self.measurements = measurements
        self.reader = reader
        self.t_0 = t_0
        self.plate_map_dict = ReaderExperiment.get_plate_map_dict(plate_map_path)
        self.wells_2d = self.plate_map_dict["well"]
        self.wells = self.plate_map_dict["well"].flatten()
        if "well_status" in self.plate_map_dict:
            self.wells = self.wells[(self.plate_map_dict["well_status"] != "empty").flatten()]

        self.data = ReaderExperiment.get_reader_data(
            df_path,
            measurements,
            self.wells,
            reader,
            sheet_name=sheet_name,
            missing_well_value=missing_well_value,
        )
        start_sheet = sheet_name[0] if isinstance(sheet_name, (list, tuple, pd.Index, np.ndarray)) else sheet_name
        self.start_time = ReaderExperiment.get_start_time(df_path, reader, sheet=start_sheet)

    def well_to_trait_value(self, well, trait_name):
        well_idx = np.where(self.wells_2d == well)
        return self.plate_map_dict[trait_name][well_idx][0]

    def traits_to_well(self, trait_values):
        idx = np.where(np.all([self.plate_map_dict[trait_name] == trait_value for trait_name, trait_value in trait_values.items()], axis=0))
        return self.wells_2d[idx]

    def well_to_name(self, well):
        return " ".join([str(self.well_to_trait_value(well, trait)) for trait in self.plate_map_dict.keys() if trait not in ["well", "well_status"]])

    @staticmethod
    def _odd_window(n, target, min_val=3):
        target = int(target)
        max_odd = n if n % 2 == 1 else n - 1
        win = min(target, max_odd)
        if win % 2 == 0:
            win -= 1
        return win if win >= min_val else None

    @staticmethod
    def estimate_max_log_growth(
            time,
            y,
            *,
            t_span=(None, None),
            smooth_window_points=11,
            fit_window_points=9,
            polyorder=3,
            min_abs_signal=0.03,
            min_fold_over_baseline=1.25,
            max_y_fraction=0.90,
            min_max_signal=0.08,
            min_fit_points=5,
            min_r2=0.97,
            min_slope=0.0,
            exclude_edge_points=3,
    ):
        """
        Estimate max exponential growth rate as max local slope of log(y) vs time.

        Returns dict with:
          mu         : max log-slope, h^-1
          t0         : tangent point time
          y0         : fitted/smoothed y at tangent point
          log_y0     : log(y0)
          idx        : local index in filtered arrays
          r2         : local log-linear fit R^2
          intercept  : local fit intercept
        """
        time = np.asarray(time, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(time) & np.isfinite(y) & (y > 0)

        if t_span[0] is not None:
            mask &= time >= t_span[0]
        if t_span[1] is not None:
            mask &= time <= t_span[1]

        t = time[mask]
        yy = y[mask]

        empty = {
            "mu": np.nan,
            "t0": np.nan,
            "y0": np.nan,
            "log_y0": np.nan,
            "idx": None,
            "r2": np.nan,
            "intercept": np.nan,
        }

        if len(t) < min_fit_points or np.nanmax(yy) < min_max_signal:
            return empty

        order = np.argsort(t)
        t = t[order]
        yy = yy[order]

        baseline = np.nanmedian(yy[:min(5, len(yy))])
        low_cut = max(min_abs_signal, baseline * min_fold_over_baseline)
        high_cut = np.nanmax(yy) * max_y_fraction

        candidate = (yy >= low_cut) & (yy <= high_cut)
        z = np.log(yy)

        smooth_win = ReaderExperiment._odd_window(
            len(z),
            smooth_window_points,
            min_val=polyorder + 2,
        )

        if smooth_win is not None and smooth_win > polyorder:
            try:
                z_s = ReaderExperiment.smooth_data(
                    z,
                    window_length=smooth_win,
                    polyorder=min(polyorder, smooth_win - 2),
                )
            except Exception:
                z_s = z
        else:
            z_s = z

        fit_win = ReaderExperiment._odd_window(
            len(z_s),
            fit_window_points,
            min_val=min_fit_points,
        )

        if fit_win is None:
            return empty

        half = fit_win // 2
        edge = max(exclude_edge_points, half)
        best = empty.copy()

        for k in range(edge, len(t) - edge):
            if not candidate[k]:
                continue

            sl = slice(k - half, k + half + 1)
            local_ok = candidate[sl] & np.isfinite(z_s[sl])

            if np.sum(local_ok) < min_fit_points:
                continue

            x_fit = t[sl][local_ok]
            z_fit = z_s[sl][local_ok]

            if np.ptp(x_fit) <= 0:
                continue

            slope, intercept = np.polyfit(x_fit, z_fit, 1)

            if slope <= min_slope:
                continue

            pred = intercept + slope * x_fit
            ss_res = np.sum((z_fit - pred) ** 2)
            ss_tot = np.sum((z_fit - np.mean(z_fit)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            if min_r2 is not None and (not np.isfinite(r2) or r2 < min_r2):
                continue

            if not np.isfinite(best["mu"]) or slope > best["mu"]:
                best = {
                    "mu": float(slope),
                    "t0": float(t[k]),
                    "y0": float(np.exp(z_s[k])),
                    "log_y0": float(z_s[k]),
                    "idx": int(k),
                    "r2": float(r2),
                    "intercept": float(intercept),
                }

        return best

    def get_max_log_growth_fit(self, well, measurement, **kwargs):
        time = self.data[(measurement, "time")]
        y = self.data[(measurement, well)]
        return ReaderExperiment.estimate_max_log_growth(time, y, **kwargs)

    def get_max_log_growth_rate(self, well, measurement, **kwargs):
        return self.get_max_log_growth_fit(well, measurement, **kwargs)["mu"]

    def plot_plate_grid(self, measurement, t_span=(None, None),
                        yscale="linear", ylim=None,
                        color_by_trait=None, color_map=None,
                        annotate_well_by_trait=None,
                        line_kwargs=None,
                        figsize=None, title=None,
                        show_well_id=False,
                        save_fig_path=None):
        """
        Plot all traces of a given measurement laid out in the plate's physical grid.

        The grid dimensions are inferred from self.wells_2d. Wells with no
        measurement (empty / not present in self.wells) are left as blank panels
        so that all rows/columns remain aligned. Inner panels share x and y axes;
        only the leftmost column shows y tick labels and only the bottom row shows
        x tick labels. Row labels (e.g. A, B, ...) appear on the left and column
        labels (e.g. 1, 2, ...) appear on the top.

        Parameters
        ----------
        measurement : str
            Measurement key, must exist in self.data.
        t_span : tuple (start, end)
            Time window in hours; None means no bound.
        yscale : str
            "linear" or "log".
        ylim : tuple or None
            Y-axis limits shared across all panels.
        color_by_trait : str or None
            If given, color each trace by this trait value.
        color_map : dict or None
            Optional mapping {trait_value: color}. Used together with
            color_by_trait. If None, an automatic palette is built.
        annotate_well_by_trait : str, tuple[str, ...], or None
            If given, annotate each plotted well with this trait value. If a tuple
            of traits is given, annotate with the trait values joined by "_".
        line_kwargs : dict or None
            Extra kwargs forwarded to ax.plot (e.g. {"linewidth": 1}).
        figsize : tuple or None
            Figure size. If None, scaled automatically from grid shape.
        title : str or None
            Optional figure title.
        show_well_id : bool
            If True, annotate each panel with its well id (small, top-left).
        save_fig_path : str or None
            If given, save figure here.

        Returns
        -------
        fig, axes
            The matplotlib Figure and the 2D ndarray of Axes (shape = grid).
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        wells_2d = self.wells_2d
        n_rows, n_cols = wells_2d.shape

        # Derive row labels (letters) and column labels (numbers) from well ids like "A1", "B12"
        def _split_well(w):
            if w is None or (isinstance(w, float) and np.isnan(w)):
                return None, None
            s = str(w)
            m = re.match(r"^\s*([A-Za-z]+)\s*(\d+)\s*$", s)
            if m:
                return m.group(1).upper(), m.group(2)
            return None, None

        row_labels = [None] * n_rows
        col_labels = [None] * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                r, c = _split_well(wells_2d[i, j])
                if r is not None and row_labels[i] is None:
                    row_labels[i] = r
                if c is not None and col_labels[j] is None:
                    col_labels[j] = c

        # Fallbacks if parsing failed
        row_labels = [
            rl if rl is not None else chr(ord("A") + i)
            for i, rl in enumerate(row_labels)
        ]
        col_labels = [
            cl if cl is not None else str(j + 1)
            for j, cl in enumerate(col_labels)
        ]

        # Time and global slice
        time = np.asarray(self.data[(measurement, "time")], dtype=float)

        def _get_slice(ts):
            start_idx = 0 if ts[0] is None else int(np.searchsorted(time, ts[0]))
            end_idx = len(time) if ts[1] is None else int(np.searchsorted(time, ts[1]))
            return slice(start_idx, end_idx)

        sl = _get_slice(t_span)
        x_plot = time[sl]

        # Color handling
        valid_wells_set = set(
            self.wells.tolist() if hasattr(self.wells, "tolist") else list(self.wells)
        )

        if color_by_trait:
            trait_vals = []
            for w in wells_2d.flatten():
                if w in valid_wells_set:
                    try:
                        trait_vals.append(self.well_to_trait_value(w, color_by_trait))
                    except Exception:
                        pass
            if color_map is None:
                auto = generate_color_palette(trait_vals)
                color_map = auto

        def _format_trait_annotation(well, trait):
            if isinstance(trait, tuple):
                vals = [self.well_to_trait_value(well, t) for t in trait]
                return "_".join(str(v) for v in vals)
            return str(self.well_to_trait_value(well, trait))

        # Figure & grid
        if figsize is None:
            figsize = (max(6, 1.2 * n_cols + 1), max(4, 0.9 * n_rows + 1))

        fig = plt.figure(figsize=figsize)

        # Leave some margins for headers; almost no spacing between panels
        gs = gridspec.GridSpec(
            n_rows, n_cols, figure=fig,
            wspace=0.0, hspace=0.0,
            left=0.07, right=0.99, top=0.93, bottom=0.09,
        )

        line_kwargs = dict(line_kwargs) if line_kwargs else {}
        line_kwargs.setdefault("linewidth", 1.0)

        axes = np.empty((n_rows, n_cols), dtype=object)
        base_ax = None

        for i in range(n_rows):
            for j in range(n_cols):
                if base_ax is None:
                    ax = fig.add_subplot(gs[i, j])
                    base_ax = ax
                else:
                    ax = fig.add_subplot(gs[i, j], sharex=base_ax, sharey=base_ax)
                ax.grid(False, which="both", axis="both")
                axes[i, j] = ax

                well = wells_2d[i, j]
                has_data = (well in valid_wells_set) and ((measurement, well) in self.data)

                if has_data:
                    y = np.asarray(self.data[(measurement, well)], dtype=float)[sl]

                    color = None
                    if color_by_trait and color_map is not None:
                        try:
                            tv = self.well_to_trait_value(well, color_by_trait)
                            color = color_map.get(tv, None)
                        except Exception:
                            color = None

                    ax.plot(x_plot, y, color=color, **line_kwargs)

                    if show_well_id:
                        ax.text(
                            0.98, 0.05, str(well),
                            transform=ax.transAxes,
                            fontsize=7,
                            va="bottom",
                            ha="right",
                            color="dimgray",
                        )

                    if annotate_well_by_trait is not None:
                        try:
                            trait_text = _format_trait_annotation(
                                well,
                                annotate_well_by_trait,
                            )
                            ax.text(
                                0.02, 0.95, trait_text,
                                transform=ax.transAxes,
                                fontsize=8,
                                va="top",
                                ha="left",
                                color="k",
                            )
                        except Exception:
                            pass

                # Aesthetics: hide inner tick labels, keep axis frame for alignment
                if i != n_rows - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if j != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", direction="in", labelsize=8)

                # Column header on top row
                if i == 0:
                    ax.set_title(str(col_labels[j]), fontsize=10, pad=4)

                # Row header on leftmost column
                if j == 0:
                    ax.set_ylabel(
                        str(row_labels[i]),
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=10,
                        labelpad=10,
                    )

        # Apply scale / limits once via the shared base axis
        if base_ax is not None:
            base_ax.set_yscale(yscale)
            if ylim is not None:
                base_ax.set_ylim(ylim)
            if len(x_plot) > 0:
                base_ax.set_xlim(x_plot[0], x_plot[-1])

        # Common axis labels (on figure)
        fig.supxlabel("Time [h]", fontsize=11)
        fig.supylabel(str(measurement), fontsize=11)

        if title:
            fig.suptitle(title, fontsize=12)

        if save_fig_path:
            fig.savefig(save_fig_path, dpi=200, bbox_inches="tight")

        return fig, axes

    def transform_measurement(self, measurement, new_measurement_name, func, wells=None):
        if wells is None:
            wells = self.wells
        for well in wells:
            self.data[(new_measurement_name, well)] = func(self.data[(measurement, well)])
        self.data[(new_measurement_name, "time")] = self.data[(measurement, "time")]

    @staticmethod
    def to_hours(t):
        return t.total_seconds() / 3600 if isinstance(t, datetime.timedelta) \
                           else t.hour + t.minute / 60 + t.second / 3600 + t.microsecond / 3600000000
    @staticmethod
    def to_hours_array(t):
        return np.array(list(map(ReaderExperiment.to_hours, t)))

    # @staticmethod
    # def get_plate_map_dict(path):
    #     plate_map = pd.read_excel(path, sheet_name="Sheet1")
    #     # the df at place 0,0 is the name of the first characteristic of the plate map
    #     # the rectangle from 1,0 to the first nans is the values for each cell of said characteristic
    #     # we start by finding the indices of the first nan in the first column
    #     m = np.argmax(plate_map.iloc[1:, 0].isnull())
    #     n = len(plate_map.iloc[1, :])
    #     # n_characteristics = int((len(plate_map) + 1) / (m + 2))
    #     n_characteristics = len(plate_map)//(m+2) + 1
    #
    #     plate_map_dict = {}
    #     for i in range(n_characteristics):
    #         characteristic = plate_map.iloc[i*(m+2), 0]
    #         values = plate_map.iloc[i*(m+2)+1:i*(m+2)+1+m, 0:n].values
    #         plate_map_dict[characteristic] = values
    #     return plate_map_dict

    @staticmethod
    def get_plate_map_dict(path):
        plate_map = pd.read_excel(path, sheet_name="Sheet1")
        # the df at place 0,0 is the name of the first characteristic of the plate map
        # the rectangle from 1,0 to the first nans is the values for each cell of said characteristic
        # we start by finding the indices of the first nan in the first column
        m = np.argmax(plate_map.iloc[1:, 0].isnull())
        n = len(plate_map.iloc[1, :])
        # n_characteristics = int((len(plate_map) + 1) / (m + 2))
        n_characteristics = len(plate_map)//(m+2) + 1
        plate_map_dict = {}
        for i in range(n_characteristics):
            characteristic = plate_map.iloc[i*(m+2), 0]
            start_row = i*(m+2) + 1
            end_row = start_row + m
            # Extract the block for this characteristic (may be smaller than m x n
            # if the last block is incomplete)
            block = plate_map.iloc[start_row:end_row, 0:n].values
            # Ensure all characteristics have the same m x n shape by padding with NaNs
            full_block = np.full((m, n), np.nan, dtype=block.dtype)
            full_block[:block.shape[0], :block.shape[1]] = block
            plate_map_dict[characteristic] = full_block
        return plate_map_dict

    @staticmethod
    def _is_well_column(column):
        return bool(re.match(r"^[A-Za-z]+\d+$", str(column).strip()))

    @staticmethod
    def _well_sort_key(column):
        match = re.match(r"^([A-Za-z]+)(\d+)$", str(column).strip())
        if not match:
            return (str(column), 0)
        row, col = match.groups()
        return (row.upper(), int(col))

    @staticmethod
    def concatenate_reader_sheets_old(df_path, measurement, sheet_names, missing_well_value=np.nan):
        """Concatenate old/Tecan-reader sheets using each sheet's recorded start time.

        Parameters
        ----------
        df_path : str
            Reader workbook path.
        measurement : str
            Measurement marker in the workbook, e.g. ``"Label: OD"``.
        sheet_names : list-like
            Sheets to concatenate in chronological order. The first sheet defines
            elapsed time zero.
        missing_well_value : scalar, default np.nan
            Value inserted for wells absent from a sheet. Use ``np.nan`` to
            represent unmeasured wells, or ``0`` if missing wells should be
            treated as zero signal.

        Returns
        -------
        pandas.DataFrame
            A reader-style measurement dataframe with ``Time [s]`` adjusted to
            elapsed seconds from the first sheet start.
        """
        if isinstance(sheet_names, str) or sheet_names is None:
            raise ValueError("sheet_names must be a non-empty list of sheet names")

        sheet_names = list(sheet_names)
        if len(sheet_names) == 0:
            raise ValueError("sheet_names must contain at least one sheet")

        starts = [
            ReaderExperiment.get_start_time(df_path, reader="old", sheet=sheet)
            for sheet in sheet_names
        ]
        t0 = starts[0]

        sheet_dfs = []
        all_well_columns = set()
        ordered_non_well_columns = []

        for sheet, start_time in zip(sheet_names, starts):
            df = ReaderExperiment.get_measurement_df_from_reader_file_old(
                df_path,
                measurement,
                sheet_name=sheet,
            ).copy()

            if "Time [s]" not in df.columns:
                raise ValueError(f"Sheet {sheet!r} does not contain a 'Time [s]' column")

            elapsed_offset_s = (start_time - t0).total_seconds()
            df["Time [s]"] = pd.to_numeric(df["Time [s]"], errors="coerce") + elapsed_offset_s

            for col in df.columns:
                if ReaderExperiment._is_well_column(col):
                    all_well_columns.add(col)
                elif col not in ordered_non_well_columns:
                    ordered_non_well_columns.append(col)

            sheet_dfs.append(df)

        all_well_columns = sorted(all_well_columns, key=ReaderExperiment._well_sort_key)
        out_dfs = []
        for df in sheet_dfs:
            df = df.copy()
            for well in all_well_columns:
                if well not in df.columns:
                    df[well] = missing_well_value
            for col in ordered_non_well_columns:
                if col not in df.columns:
                    df[col] = np.nan
            out_dfs.append(df[ordered_non_well_columns + all_well_columns])

        concatenated = pd.concat(out_dfs, ignore_index=True)
        if "Cycle Nr." in concatenated.columns:
            concatenated["Cycle Nr."] = np.arange(1, len(concatenated) + 1)
        return concatenated

    @staticmethod
    def _tecan_block_start(df, measurement, df_path):
        """Return the row of the "Cycle Nr." header of the block named ``measurement``.

        A block is named by the row directly above its "Cycle Nr." header. Single-label
        exports have no such row; their name comes from the "Label: X" metadata row.
        ``"Label: X"`` and ``"X"`` are treated as the same request.

        Raises ValueError when the name matches no block, rather than returning
        another block's data under the requested name.
        """
        def _strip_label(value):
            return re.sub(r"^Label:\s*", "", str(value).strip())

        col0 = df.iloc[:, 0]
        col0_str = col0.astype(str)
        cycle_rows = list(df.index[col0_str.str.startswith("Cycle Nr.")])
        if len(cycle_rows) == 0:
            raise IndexError(f"No data block ('Cycle Nr.') found in the file {df_path}.")

        label_rows = [_strip_label(v) for v in col0_str[col0_str.str.match(r"^Label:\s*")]]
        names = []
        for i, row in enumerate(cycle_rows):
            above = col0.iloc[row - 1] if row > 0 else np.nan
            if pd.notna(above) and str(above).strip():
                names.append(_strip_label(above))
            elif len(cycle_rows) == 1 and len(label_rows) == 1:
                names.append(label_rows[0])
            else:
                names.append(None)

        requested = _strip_label(measurement)
        for row, name in zip(cycle_rows, names):
            if name == requested:
                return row
        raise ValueError(
            f"Measurement {measurement!r} not found in {df_path}. "
            f"Blocks in this sheet: {[n for n in names if n is not None]}"
        )

    @staticmethod
    def get_measurement_df_from_reader_file_old(df_path, measurement, sheet_name=None, missing_well_value=np.nan):
        if sheet_name is None:
            sheet_name = "Sheet2"
        if isinstance(sheet_name, (list, tuple, pd.Index, np.ndarray)):
            return ReaderExperiment.concatenate_reader_sheets_old(
                df_path,
                measurement,
                sheet_name,
                missing_well_value=missing_well_value,
            )
        df = pd.read_excel(df_path, sheet_name=sheet_name)
        start = ReaderExperiment._tecan_block_start(df, measurement, df_path)
        # get the df from the start index to the end of the df
        d = df.iloc[start + 1:, 1].isnull()
        end = d[d].index[0]
        measurement_df = df.iloc[start + 1:end, :]
        # set the column names as the first row
        measurement_df.columns = df.iloc[start, :]

        # set indices to start from 0
        measurement_df.index = np.arange(len(measurement_df))
        return measurement_df

    @staticmethod
    def get_measurement_df_from_reader_file_synergy(df_path, measurement, sheet_name=None):
        if sheet_name is None:
            sheet_name = "Plate 1 - Sheet1"
        df = pd.read_excel(df_path, sheet_name=sheet_name, header=None)

        # Try to detect format type
        # Format 1: measurement label in column 0 (original format)
        # Format 2: numeric wavelength in column 0 (new format with Time column)
        indices = df[df.iloc[:, 0] == measurement].index

        if len(indices) == 0:
            # No match in column 0, might be numeric or different format
            raise ValueError(f"Measurement '{measurement}' not found in column 0. Check measurement name or file format.")

        start = indices[0]

        # Check if this is the new format (has 'Time' in column structure)
        # In new format, data starts 2 rows after the measurement marker
        # and has 'Time' and 'T°' columns
        if start + 2 < len(df) and df.iloc[start + 2, 1] == 'Time':
            # New format: numeric marker (like 600), then blank, then header with Time
            header_row = start + 2
            data_start = start + 3

            # Find where data ends (first row with all NaN in data columns)
            data_section = df.iloc[data_start:, 2:]  # Skip Time and T° columns
            # Find first row where all values are NaN
            all_nan = data_section.isnull().all(axis=1)
            end_indices = all_nan[all_nan].index
            if len(end_indices) > 0:
                end = end_indices[0]
            else:
                end = len(df)

            # Extract data
            measurement_df = df.iloc[data_start:end, 1:].copy()
            measurement_df.columns = df.iloc[header_row, 1:].values

            # Convert Time to numeric (hours)
            time_col = measurement_df.iloc[:, 0]  # Time column

            # Handle both string and datetime.time objects
            if pd.api.types.is_object_dtype(time_col):
                # Check if first non-null value is datetime.time
                first_val = time_col.dropna().iloc[0] if len(time_col.dropna()) > 0 else None
                if hasattr(first_val, 'hour') and hasattr(first_val, 'minute'):
                    # datetime.time below 24 h; Excel switches to timedelta from 24 h on
                    measurement_df['time'] = time_col.apply(
                        lambda t: ReaderExperiment.to_hours(t) if pd.notna(t) else np.nan
                    )
                else:
                    # String format, use pd.to_timedelta
                    measurement_df['time'] = pd.to_timedelta(time_col).dt.total_seconds() / 3600
            else:
                measurement_df['time'] = pd.to_timedelta(time_col).dt.total_seconds() / 3600

            # Drop original Time and T° columns, keep numeric time
            measurement_df = measurement_df.drop(columns=[measurement_df.columns[0], measurement_df.columns[1]])

        else:
            # Original format: measurement in column 0, data starts at start+2
            d = df.iloc[start+1:, 2].isnull()
            end = d[d].index[0]
            measurement_df = df.iloc[start+2:end, 1:]
            # set the column names as the first row
            measurement_df.columns = df.iloc[start+1, 1:]

        # set indices to start from 0
        measurement_df.index = np.arange(len(measurement_df))
        return measurement_df

    @staticmethod
    def get_measurement_df_from_reader_file_plex(df_path, measurement, sheet_name=None):
        if sheet_name is None:
            sheet_name = "Sheet2"
        df = pd.read_excel(df_path, sheet_name=sheet_name, header=None)
        start = ReaderExperiment._tecan_block_start(df, measurement, df_path)

        # find end index where column 2 is null (starting from start+1)
        d = df.iloc[start + 1:, 2].isnull()
        end = d[d].index[0]

        data = df.iloc[start + 1:end, :]  # columns 1:end
        columns = data.iloc[:, 0].values  # first column in block (values to become columns)
        measurement_df = data.iloc[:, 1:].transpose()
        measurement_df.columns = columns
        measurement_df.index = np.arange(len(measurement_df))
        return measurement_df

    def get_first_crossing_time(self, well, measurement, threshold, threshold_fold=None, t_0=0):
        time = np.asarray(self.data[(measurement, "time")], dtype=float)
        y = np.asarray(self.data[(measurement, well)], dtype=float)
        finite = np.isfinite(time) & np.isfinite(y)

        if threshold_fold:
            finite_y = y[finite]
            if len(finite_y) == 0:
                return np.nan
            threshold = finite_y[0] * threshold_fold

        if len(y) == 0 or not np.any(finite):
            return np.nan

        crosses = finite & (y > threshold)
        if not np.any(crosses):
            return np.nan

        crossing_time = time[np.argmax(crosses)] - t_0
        return crossing_time

    @staticmethod
    def smooth_data(y, window_length=10, polyorder=3):
        y = np.asarray(y, dtype=float)
        out = np.full(y.shape, np.nan, dtype=float)
        finite = np.isfinite(y)

        if y.ndim != 1:
            return savgol_filter(y, window_length, polyorder)
        if not np.any(finite):
            return out

        idx = np.flatnonzero(finite)
        run_breaks = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, run_breaks)

        for run in runs:
            vals = y[run]
            win = ReaderExperiment._odd_window(
                len(vals),
                window_length,
                min_val=polyorder + 2,
            )
            if win is None or win <= polyorder:
                out[run] = vals
            else:
                out[run] = savgol_filter(vals, win, min(polyorder, win - 2))
        return out

    def get_stationary_time(self, well, measurement, min_stationary_measurement=0.2):
        fit = self.get_max_log_growth_fit(
            well,
            measurement,
            min_max_signal=min_stationary_measurement,
        )
        if not np.isfinite(fit["t0"]):
            return np.nan
        return fit["t0"] - getattr(self, "t_0", 0.0)

    def get_time_in_stationary(self, well, measurement):
        t_0 = self.get_stationary_time(well, measurement)
        t_1 = self.data[(measurement, "time")][-1]
        return t_1 - t_0


    def get_slope(self, well, measurement):
        y = self.data[(measurement, well)]
        smooth_y = ReaderExperiment.smooth_data(y)
        return np.diff(smooth_y)

    @staticmethod
    def get_start_time(df_path, reader="synergy", sheet=None):
        """Extract experiment start time as datetime object."""
        if sheet is None:
            sheet = "Plate 1 - Sheet1" if reader == "synergy" else "Sheet2"

        df = pd.read_excel(df_path, sheet_name=sheet, header=None)

        if reader == "synergy":
            # Date in row with "Date" label, Time in row with "Time" label
            date_row = df[df.iloc[:, 0] == "Date"].index[0]
            time_row = df[df.iloc[:, 0] == "Time"].index[0]
            date_val = df.iloc[date_row, 1]  # datetime.datetime
            time_val = df.iloc[time_row, 1]  # datetime.time
            return datetime.datetime.combine(date_val.date(), time_val)

        elif reader in ("old", "plex"):
            # Look for "Start Time:" row
            start_row = df[df.iloc[:, 0] == "Start Time:"].index[0]
            start_str = df.iloc[start_row, 1]  # e.g., "1/2/2023 9:04:36 AM"
            return datetime.datetime.strptime(start_str, "%m/%d/%Y %I:%M:%S %p")
        return None

    @staticmethod
    def get_reader_data(df_path, measurements, wells, reader, sheet_name=None, missing_well_value=np.nan):
        data = {}
        for (measurement, measurement_name_in_reader) in measurements.items():
            if reader == "synergy":
                df = ReaderExperiment.get_measurement_df_from_reader_file_synergy(df_path, measurement_name_in_reader, sheet_name=sheet_name)
            elif reader == "plex":
                df = ReaderExperiment.get_measurement_df_from_reader_file_plex(df_path, measurement_name_in_reader, sheet_name=sheet_name)
            elif reader == "old":
                df = ReaderExperiment.get_measurement_df_from_reader_file_old(
                    df_path,
                    measurement_name_in_reader,
                    sheet_name=sheet_name,
                    missing_well_value=missing_well_value,
                )
            else:
                raise ValueError(f"Unknown reader: {reader}")

            # Extract time data once per measurement (not per well)
            if reader == "synergy":
                # New format already has 'time' column, old format has 'Time'
                if 'time' in df.columns:
                    data[(measurement, "time")] = pd.to_numeric(df['time'], errors="coerce").to_numpy(dtype=float)
                else:
                    data[(measurement, "time")] = ReaderExperiment.to_hours_array(df['Time'])
            elif reader == "plex":
                data[(measurement, "time")] = (
                    pd.to_numeric(df['Time [s]'], errors="coerce").to_numpy(dtype=float) / 3600
                )
            elif reader == "old":
                data[(measurement, "time")] = (
                    pd.to_numeric(df['Time [s]'], errors="coerce").to_numpy(dtype=float) / 3600
                )

            for well in wells:
                # Convert reader artifacts (OVER, *, Wait rows, absent wells) to NaN.
                if well in df.columns:
                    series = df[well]
                else:
                    series = pd.Series(np.nan, index=df.index)
                cleaned = series.replace("OVER", np.nan)
                cleaned = cleaned.apply(lambda x: str(x).replace("*", "") if isinstance(x, str) else x)
                data[(measurement, well)] = pd.to_numeric(cleaned, errors="coerce").to_numpy(dtype=float)

        return data

    @staticmethod
    def generate_hue_variations(base_colors, trait_values, hue_trait_values):
        """Generate hue/shade variations of base colors based on a secondary trait."""
        import colorsys

        # Get unique hue trait values and sort them
        unique_hue_values = sorted(set(hue_trait_values))
        n_variations = len(unique_hue_values)

        if n_variations == 0:
            return base_colors

        # Create mapping from trait combinations to colors
        color_map = {}

        for trait_val, hue_trait_val in zip(trait_values, hue_trait_values):
            if trait_val not in base_colors:
                continue

            base_color = base_colors[trait_val]

            # Convert hex to RGB
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*rgb)

            # Find the position of this hue trait value
            hue_index = unique_hue_values.index(hue_trait_val)

            # Vary the value (brightness) or saturation based on position
            # Keep hue the same, vary brightness from 0.5 to 1.0
            if n_variations > 1:
                # Create variations in brightness
                v_variation = 0.5 + (0.5 * hue_index / (n_variations - 1))
                new_v = v * v_variation

                # Optionally also vary saturation slightly
                s_variation = 0.7 + (0.3 * hue_index / (n_variations - 1))
                new_s = s * s_variation
            else:
                new_v = v
                new_s = s

            # Convert back to RGB
            new_rgb = colorsys.hsv_to_rgb(h, new_s, new_v)

            # Convert to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(new_rgb[0] * 255),
                int(new_rgb[1] * 255),
                int(new_rgb[2] * 255)
            )

            # Store with combined key
            color_map[(trait_val, hue_trait_val)] = hex_color

        return color_map

    def plot_data(self, measurement, wells=None, t_span=(None, None), t_span_per_well=None,
                  color_by_trait="", hue_by_trait="", line_style_by_trait="",
                  ylim=None, yscale="linear", marker_by_trait="", show_fig=False,
                  save_fig_path=None, save_html_path=None, title="", mark_stationary=False,
                  traces_initially_visible=True):
        if wells is None:
            wells = self.wells

        initial_trace_visibility = True if traces_initially_visible else "legendonly"

        # Handle color mapping with optional hue variations
        if color_by_trait and hue_by_trait:
            # First get base colors
            base_colors = generate_color_palette(
                [self.well_to_trait_value(well, color_by_trait) for well in wells])

            # Then create hue variations
            color_map = ReaderExperiment.generate_hue_variations(
                base_colors,
                [self.well_to_trait_value(well, color_by_trait) for well in wells],
                [self.well_to_trait_value(well, hue_by_trait) for well in wells]
            )
        elif color_by_trait:
            temp_map = generate_color_palette(
                [self.well_to_trait_value(well, color_by_trait) for well in wells])
            # Convert to simple mapping if no hue trait
            color_map = {(k, None): v for k, v in temp_map.items()}
        else:
            color_map = None

        line_style_map = generate_line_styles(
            [self.well_to_trait_value(well, line_style_by_trait) for well in wells]) if line_style_by_trait else None
        marker_map = generate_marker_symbols(
            [self.well_to_trait_value(well, marker_by_trait) for well in wells]) if marker_by_trait else None

        time = self.data[(measurement, "time")]

        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.02)

        def get_slice(t_span):
            start_idx = 0 if t_span[0] is None else np.searchsorted(time, t_span[0])
            end_idx = len(time) if t_span[1] is None else np.searchsorted(time, t_span[1])
            return slice(start_idx, end_idx)

        # Track which trait combinations we've already added to legend
        legend_entries = set()

        for well in wells:
            well_name = self.well_to_name(well)
            if t_span_per_well is not None and well in t_span_per_well:
                plot_slice = get_slice(t_span_per_well[well])
            else:
                plot_slice = get_slice(t_span)

            x = time
            y = self.data[(measurement, well)]

            # Get color based on trait combination
            if color_map:
                if hue_by_trait:
                    color_key = (self.well_to_trait_value(well, color_by_trait),
                                 self.well_to_trait_value(well, hue_by_trait))
                else:
                    color_key = (self.well_to_trait_value(well, color_by_trait), None)
                line_color = color_map.get(color_key)
            else:
                line_color = None

            # Determine legend group and whether to show this entry
            color_val = self.well_to_trait_value(well, color_by_trait) if color_by_trait else None
            hue_val = self.well_to_trait_value(well, hue_by_trait) if hue_by_trait else None
            line_val = self.well_to_trait_value(well, line_style_by_trait) if line_style_by_trait else None
            marker_val = self.well_to_trait_value(well, marker_by_trait) if marker_by_trait else None

            # Create legend entry key
            legend_key = (color_val, hue_val, line_val, marker_val)
            show_legend = legend_key not in legend_entries
            legend_entries.add(legend_key)

            # Build legend name based on active traits
            legend_parts = []
            if color_by_trait:
                legend_parts.append(f"{color_by_trait}: {color_val}")
            if hue_by_trait:
                legend_parts.append(f"{hue_by_trait}: {hue_val}")
            if line_style_by_trait:
                legend_parts.append(f"{line_style_by_trait}: {line_val}")
            if marker_by_trait:
                legend_parts.append(f"{marker_by_trait}: {marker_val}")

            legend_name = ", ".join(legend_parts) if legend_parts else well_name

            fig.add_trace(go.Scatter(
                x=x[plot_slice],
                y=y[plot_slice],
                name=legend_name,
                legendgroup=legend_name,
                showlegend=show_legend,
                mode='lines+markers',
                line=dict(
                    color=line_color,
                    dash=line_style_map[
                        self.well_to_trait_value(well, line_style_by_trait)] if line_style_by_trait else None,
                ),
                marker=dict(
                    symbol=marker_map[self.well_to_trait_value(well, marker_by_trait)]
                ) if marker_by_trait else None,
                hovertemplate=f"<b>{well_name}</b><br>" +
                              f"Time: %{{x:.2f}}h<br>" +
                              f"{measurement}: %{{y:.2f}}<br>" +
                              "<extra></extra>",
                visible=initial_trace_visibility
            ), row=1, col=1)

            if mark_stationary:
                stationary_time = self.get_stationary_time(well, measurement)
                if not np.isnan(stationary_time):
                    y_val = y[np.searchsorted(x, stationary_time)]
                    fig.add_trace(go.Scatter(
                        x=[stationary_time],
                        y=[y_val],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color=line_color if line_color else 'red',
                            line=dict(color='black', width=1)
                        ),
                        name=f'{legend_name} (stationary)',
                        legendgroup=legend_name,
                        showlegend=False,
                        hovertemplate=f"<b>{well_name}</b><br>" +
                                      f"Stationary Time: {stationary_time:.2f}h<br>" +
                                      f"{measurement}: {y_val:.2f}<br>" +
                                      "<extra></extra>",
                        visible=initial_trace_visibility
                    ), row=1, col=1)

        # Update layout with legend configuration
        fig.update_layout(
            title=title,
            xaxis_title="Time [h]",
            yaxis_title=measurement,
            legend=dict(
                title=dict(text="Traits"),
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest'
        )
        fig.update_yaxes(row=1, col=1, type=yscale, range=ylim)

        # Add legend section headers if multiple trait types are used
        if sum([bool(color_by_trait), bool(hue_by_trait), bool(line_style_by_trait), bool(marker_by_trait)]) > 1:
            # Create dummy traces for legend section headers
            trait_types = []
            if color_by_trait:
                trait_types.append(color_by_trait)
            if hue_by_trait:
                trait_types.append(hue_by_trait)
            if line_style_by_trait:
                trait_types.append(line_style_by_trait)
            if marker_by_trait:
                trait_types.append(marker_by_trait)

            # Update legend title
            fig.update_layout(legend_title_text=" / ".join(trait_types))

        if show_fig:
            fig.show()
        if save_fig_path:
            fig.write_image(save_fig_path)
        if save_html_path:
            fig.write_html(save_html_path)
        return fig


    def plot_data_static(self, measurement, wells=None, t_span=(None, None), t_span_per_well=None,
                         trait_color_map=None, rep_trait=None,
                         ylim=None, yscale="linear",
                         save_fig_path=None, title="", mark_stationary=False):
        """
        Static (matplotlib) version of plot_data.

        Parameters
        ----------
        measurement : str
            Name of the measurement to plot (must exist in self.data).
        wells : array-like or None
            Wells to include. Defaults to self.wells.
        t_span : tuple (start, end)
            Global time window. None entries mean no bound.
        t_span_per_well : dict or None
            Per-well overrides of t_span, keyed by well id.
        trait_color_map : str, tuple, or None
            * str  – trait name used for coloring (auto palette).
            * tuple (trait_name, {trait_value: color, ...}) – trait name
              plus an explicit color dict.
            * None – every curve gets its own auto color.
        rep_trait : str, list of str, or None
            Trait(s) that distinguish biological / technical replicates.
            When provided, wells that share every trait *except* rep_trait
            are averaged and plotted as mean ± std (shaded).
        ylim : tuple or None
            Y-axis limits.
        yscale : str
            "linear" or "log".
        save_fig_path : str or None
            If given, figure is saved to this path.
        title : str
            Figure title.
        mark_stationary : bool
            If True, mark the estimated stationary-phase entry point.

        Returns
        -------
        fig, ax
        """
        from collections import defaultdict
        from matplotlib.colors import to_rgb

        if wells is None:
            wells = self.wells

        # ---- Parse trait_color_map ------------------------------------------------
        color_trait = None
        explicit_colors = None
        if trait_color_map is not None:
            if isinstance(trait_color_map, str):
                color_trait = trait_color_map
            elif isinstance(trait_color_map, (tuple, list)) and len(trait_color_map) == 2:
                color_trait, explicit_colors = trait_color_map[0], trait_color_map[1]
            else:
                raise ValueError("trait_color_map must be a trait name (str) or (trait_name, {value: color}) tuple")

        # ---- Parse rep_trait ------------------------------------------------------
        if rep_trait is not None:
            if isinstance(rep_trait, str):
                rep_traits = [rep_trait]
            else:
                rep_traits = list(rep_trait)
        else:
            rep_traits = []

        # ---- Time helpers ---------------------------------------------------------
        time = self.data[(measurement, "time")]

        def get_slice(ts):
            start_idx = 0 if ts[0] is None else np.searchsorted(time, ts[0])
            end_idx = len(time) if ts[1] is None else np.searchsorted(time, ts[1])
            return slice(start_idx, end_idx)

        global_slice = get_slice(t_span)

        # ---- Identify all non-rep traits for grouping -----------------------------
        all_traits = [t for t in self.plate_map_dict.keys() if t not in ("well", "well_status")]
        group_traits = [t for t in all_traits if t not in rep_traits]

        # ---- Build a color palette if color_trait is given -------------------------
        if color_trait is not None:
            unique_color_vals = list(dict.fromkeys(
                self.well_to_trait_value(w, color_trait) for w in wells
            ))
            if explicit_colors is None:
                # Build auto palette using matplotlib tab10 / tab20
                cmap = plt.cm.get_cmap("tab10") if len(unique_color_vals) <= 10 else plt.cm.get_cmap("tab20")
                auto_colors = {v: cmap(i % cmap.N) for i, v in enumerate(unique_color_vals)}
            else:
                auto_colors = explicit_colors
        else:
            auto_colors = None

        # ---- Create figure --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))

        # ---- CASE 1: rep_trait given → group, average, shade ----------------------
        if rep_traits:
            # Group wells by their non-rep trait signature
            groups = defaultdict(list)
            for w in wells:
                key = tuple(self.well_to_trait_value(w, t) for t in group_traits)
                groups[key].append(w)

            legend_entries = set()

            for key, group_wells in groups.items():
                # Determine the plot slice (use the first well's per-well span if any,
                # otherwise global)
                if t_span_per_well is not None:
                    # pick first matching override, else global
                    pw_slice = None
                    for gw in group_wells:
                        if gw in t_span_per_well:
                            pw_slice = get_slice(t_span_per_well[gw])
                            break
                    if pw_slice is None:
                        pw_slice = global_slice
                else:
                    pw_slice = global_slice

                x = np.asarray(time[pw_slice], dtype=float)

                # Stack y values from all replicate wells
                ys = []
                for gw in group_wells:
                    ys.append(np.asarray(self.data[(measurement, gw)][pw_slice], dtype=float))
                ys = np.array(ys, dtype=float)

                mean_y = np.nanmean(ys, axis=0)
                std_y = np.nanstd(ys, axis=0)

                # Build label from group traits (excluding rep traits)
                label_parts = []
                for t, v in zip(group_traits, key):
                    label_parts.append(f"{t}: {v}")
                label = ", ".join(label_parts)

                # Color
                if color_trait is not None:
                    # All wells in the group share the same color-trait value
                    c_val = self.well_to_trait_value(group_wells[0], color_trait)
                    color = auto_colors.get(c_val, None)
                else:
                    color = None

                show_legend = label not in legend_entries
                legend_entries.add(label)

                ax.plot(x, mean_y, label=label if show_legend else None, color=color, linewidth=1.5)
                ax.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2, color=color)

                if mark_stationary:
                    # Average stationary time across reps
                    stat_times = []
                    for gw in group_wells:
                        st = self.get_stationary_time(gw, measurement)
                        if not np.isnan(st):
                            stat_times.append(st)
                    if stat_times:
                        avg_stat_time = np.mean(stat_times)
                        idx = np.searchsorted(time, avg_stat_time)
                        idx = min(idx, len(time) - 1)
                        # Recompute mean_y at that index if within slice
                        if pw_slice.start <= idx < (pw_slice.stop if pw_slice.stop else len(time)):
                            local_idx = idx - pw_slice.start
                            if 0 <= local_idx < len(mean_y):
                                ax.scatter([time[idx]], [mean_y[local_idx]], marker='*', s=120,
                                           color=color if color else 'red',
                                           edgecolors='black', linewidths=0.8, zorder=5)

        # ---- CASE 2: no rep_trait → plot individual wells -------------------------
        else:
            legend_entries = set()

            for well in wells:
                well_name = self.well_to_name(well)

                if t_span_per_well is not None and well in t_span_per_well:
                    plot_slice = get_slice(t_span_per_well[well])
                else:
                    plot_slice = global_slice

                x = np.asarray(time[plot_slice], dtype=float)
                y = np.asarray(self.data[(measurement, well)][plot_slice], dtype=float)

                # Color
                if color_trait is not None:
                    c_val = self.well_to_trait_value(well, color_trait)
                    color = auto_colors.get(c_val, None)
                    label = f"{color_trait}: {c_val}"
                else:
                    color = None
                    label = well_name

                show_legend = label not in legend_entries
                legend_entries.add(label)

                ax.plot(x, y, label=label if show_legend else None, color=color, linewidth=1)

                if mark_stationary:
                    stationary_time = self.get_stationary_time(well, measurement)
                    if not np.isnan(stationary_time):
                        st_idx = np.searchsorted(time, stationary_time)
                        st_idx = min(st_idx, len(time) - 1)
                        y_val = self.data[(measurement, well)][st_idx]
                        ax.scatter([stationary_time], [y_val], marker='*', s=120,
                                   color=color if color else 'red',
                                   edgecolors='black', linewidths=0.8, zorder=5)

        # ---- Axes formatting ------------------------------------------------------
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(measurement)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25)

        plt.tight_layout()

        if save_fig_path:
            fig.savefig(save_fig_path, dpi=200, bbox_inches="tight")

        return fig, ax



    @suppress_warnings(UserWarning)
    def analyze_crossing_times(self, measurement="OD", threshold=0.3, condition=None):
        """
        Analyze crossing times for different dilutions and fit linear curves.

        Parameters:
        -----------
        self : object
            Reader experiment object containing spectrophotometer data
        measurement : str
            Type of measurement (default: "OD")
        threshold : float
            Threshold value for crossing time determination
        figsize : tuple
            Figure size for the plot

        Returns:
        --------
        results_df : DataFrame
            DataFrame containing fitted parameters and statistics
        fig : matplotlib.figure.Figure
            Figure object containing the plots
        """

        assert_condition = condition if condition is not None else lambda well: True
        # Get all wells from the experiment
        wells = self.wells

        # Get unique values for each trait
        # u_strains = np.unique([self.well_to_trait_value(well, "strain") for well in wells])
        u_strains = sorted(
            np.unique([self.well_to_trait_value(well, "strain")  for well in wells]),
            key=natural_sort_key
        )
        u_dilutions = np.unique([self.well_to_trait_value(well, "dilution")  for well in wells])
        u_reps = np.unique([self.well_to_trait_value(well, "rep") for well in wells])

        # Store results
        results = []
        ct_data = []

        # Collect crossing times for each strain-replicate combination
        for strain in u_strains:
            for rep in u_reps:
                ct_values = []
                dilution_values = []

                # Collect crossing times across dilutions
                for dilution in u_dilutions:
                    # try:
                    wells_found = self.traits_to_well({
                        "strain": strain,
                        "dilution": dilution,
                        "rep": rep
                    })
                    wells_found = [well for well in wells_found if assert_condition(well)]

                    if wells_found:
                        well = wells_found[0]
                        ct = self.get_first_crossing_time(
                            well=well,
                            measurement=measurement,
                            threshold=threshold
                        )

                        if ct is not None and not np.isnan(ct):
                            ct_values.append(ct)
                            dilution_values.append(dilution)
                            ct_data.append({
                                'strain': strain,
                                'rep': rep,
                                'dilution': dilution,
                                'ct': ct,
                                'log_dilution': np.log10(dilution)
                            })
                    # except Exception as e:
                    #     print("Warning: could not determine crossing time for strain '{}'".format(strain))
                    #     print(e)

                # Fit linear model if we have enough data points
                if len(ct_values) >= 2:
                    log_dilutions = np.log10(dilution_values)

                    # Perform linear regression: ct = a + b * log10(dilution)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        log_dilutions, ct_values
                    )

                    # Calculate growth rate (inverse of slope, converted to per hour)
                    # The slope represents hours per log10(dilution)
                    # Growth rate ≈ ln(10) / slope (in per hour units)
                    growth_rate = np.log(10) / slope if slope != 0 else np.nan

                    results.append({
                        'strain': strain,
                        'rep': rep,
                        'slope': slope,
                        'intercept': intercept,
                        'growth_rate': growth_rate,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'std_err': std_err,
                        'n_points': len(ct_values)
                    })

        # Convert to DataFrames
        results_df = pd.DataFrame(results)
        ct_df = pd.DataFrame(ct_data)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Define color palette for strains
        strain_colors = sns.color_palette("husl", len(u_strains))
        strain_color_map = dict(zip(u_strains, strain_colors))

        # Plot 1: Crossing time vs log(dilution) with fitted lines
        ax1 = axes[0]

        # Plot all data points with unique markers for replicates
        rep_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        rep_marker_map = dict(zip(u_reps, rep_markers[:len(u_reps)]))

        for strain in u_strains:
            strain_data = ct_df[ct_df['strain'] == strain]
            color = strain_color_map[strain]

            # Plot individual points for each replicate with different markers
            for rep in u_reps:
                rep_data = strain_data[strain_data['rep'] == rep]
                if not rep_data.empty:
                    marker = rep_marker_map[rep]
                    ax1.scatter(rep_data['log_dilution'], rep_data['ct'],
                                color=color, alpha=0.7, s=80, marker=marker,
                                label=f'{strain} rep{rep}', edgecolors='black', linewidth=0.5)

                    # Plot fitted line for this strain-rep combination
                    fit_data = results_df[(results_df['strain'] == strain) &
                                          (results_df['rep'] == rep)]
                    if not fit_data.empty:
                        x_fit = np.linspace(ct_df['log_dilution'].min() - 0.1,
                                            ct_df['log_dilution'].max() + 0.1, 100)
                        y_fit = fit_data.iloc[0]['intercept'] + fit_data.iloc[0]['slope'] * x_fit
                        ax1.plot(x_fit, y_fit, color=color, alpha=0.8,
                                 linewidth=2.5, linestyle='--')

        ax1.set_xlabel('log₁₀(Dilution)', fontsize=12)
        ax1.set_ylabel('Crossing Time (hours)', fontsize=12)
        ax1.set_title('Crossing Time vs Dilution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # ax1.legend(loc='lower right', fontsize=9, ncol=1)

        # Plot 2: Growth rates comparison
        ax2 = axes[1]

        # Prepare data for growth rate plot
        growth_data = []
        for strain in u_strains:
            strain_results = results_df[results_df['strain'] == strain]
            for _, row in strain_results.iterrows():
                growth_data.append({
                    'strain': strain,
                    'rep': row['rep'],
                    'growth_rate': row['growth_rate'],
                    'r_squared': row['r_squared']
                })

        growth_df = pd.DataFrame(growth_data)

        # Create grouped bar plot for growth rates
        x_positions = np.arange(len(u_strains))
        width = 0.8 / len(u_reps)

        for i, rep in enumerate(u_reps):
            rep_rates = []
            rep_errors = []
            for strain in u_strains:
                data = growth_df[(growth_df['strain'] == strain) &
                                 (growth_df['rep'] == rep)]
                if not data.empty:
                    rep_rates.append(data.iloc[0]['growth_rate'])
                    # Use R² as a proxy for confidence (higher R² = smaller error bar)
                    rep_errors.append(0.1 * data.iloc[0]['growth_rate'] *
                                      (1 - data.iloc[0]['r_squared']))
                else:
                    rep_rates.append(0)
                    rep_errors.append(0)

            ax2.bar(x_positions + i * width, rep_rates, width,
                    yerr=rep_errors, capsize=5,
                    label=f'Rep {rep}', alpha=0.8,
                    color=[strain_color_map[s] for s in u_strains])

        ax2.set_xlabel('Strain', fontsize=12)
        ax2.set_ylabel('Growth Rate (h⁻¹)', fontsize=12)
        ax2.set_title('Estimated Growth Rates', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_positions + width * (len(u_reps) - 1) / 2)
        ax2.set_xticklabels(u_strains, rotation=45, ha='right')
        # ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3, axis='y')

        # # Add text annotations with fit statistics
        # text_y = 0.25
        # for strain in u_strains:
        #     strain_results = results_df[results_df['strain'] == strain]
        #     avg_r2 = strain_results['r_squared'].mean()
        #     avg_growth = strain_results['growth_rate'].mean()
        #     std_growth = strain_results['growth_rate'].std()
        #
        #     text = f"{strain}: μ = {avg_growth:.3f}±{std_growth:.3f} h⁻¹, R² = {avg_r2:.3f}"
        #     ax2.text(0.02, text_y, text, transform=ax2.transAxes,
        #              fontsize=9, verticalalignment='top',
        #              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        #     text_y -= 0.05

        # Plot 3: Intercepts (Lag period) comparison
        ax3 = axes[2]

        # Prepare data for intercept plot
        intercept_data = []
        for strain in u_strains:
            strain_results = results_df[results_df['strain'] == strain]
            for _, row in strain_results.iterrows():
                intercept_data.append({
                    'strain': strain,
                    'rep': row['rep'],
                    'intercept': row['intercept'],
                    'r_squared': row['r_squared']
                })

        intercept_df = pd.DataFrame(intercept_data)

        # Create grouped bar plot for intercepts (lag periods)
        x_positions = np.arange(len(u_strains))
        width = 0.8 / len(u_reps)

        for i, rep in enumerate(u_reps):
            rep_intercepts = []
            rep_errors = []
            for strain in u_strains:
                data = intercept_df[(intercept_df['strain'] == strain) &
                                    (intercept_df['rep'] == rep)]
                if not data.empty:
                    rep_intercepts.append(data.iloc[0]['intercept'])
                    # Use R² as a proxy for confidence
                    rep_errors.append(0.1 * abs(data.iloc[0]['intercept']) *
                                      (1 - data.iloc[0]['r_squared']))
                else:
                    rep_intercepts.append(0)
                    rep_errors.append(0)

            ax3.bar(x_positions + i * width, rep_intercepts, width,
                    yerr=rep_errors, capsize=5,
                    label=f'Rep {rep}', alpha=0.8,
                    color=[strain_color_map[s] for s in u_strains])

        ax3.set_xlabel('Strain', fontsize=12)
        ax3.set_ylabel('Lag Period (hours)', fontsize=12)
        ax3.set_title('Estimated Lag Periods (Intercept at log₁₀(dilution)=0)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_positions + width * (len(u_reps) - 1) / 2)
        ax3.set_xticklabels(u_strains, rotation=45, ha='right')
        # ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3, axis='y')

        # # Add text annotations with average lag periods
        # text_y = 0.25
        # for strain in u_strains:
        #     strain_results = results_df[results_df['strain'] == strain]
        #     avg_lag = strain_results['intercept'].mean()
        #     std_lag = strain_results['intercept'].std()
        #
        #     text = f"{strain}: lag = {avg_lag:.2f}±{std_lag:.2f} h"
        #     ax3.text(0.02, text_y, text, transform=ax3.transAxes,
        #              fontsize=9, verticalalignment='top',
        #              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        #     text_y -= 0.05

        return results_df, fig

    def plot_value_bargrid(
        self,
        value_fn,
        layout,
        *,
        filter_traits=None,
        orderings=None,
        figsize=(12, 6),
        sharey=True,
        jitter=0.06,
        bar_alpha=0.85,
        point_size=40,
        edgecolor="black",
        capsize=4,
        title=None,
        show_group_labels=False,
        min_reps=1,
        point_hue_by=None,
        show_point_hue_legend=False,
        # Generic pre-plot reduction of points (per-well values) into transformed/paired points:
        #  - None: skip
        #  - dict: built-ins {"type":"collapse", ...} or {"type":"pair", ...}
        #  - callable: f(df_raw) -> df_raw_transformed (must provide columns: value, __ROW, __COL, __GROUP, __COLOR, __HUE)
        point_reduce=None,
    ):
        """
        Build a grid of grouped barplots with per-point (rep) markers:
          - Rows: layout["rows"] (row headers)
          - Cols: layout["cols"] (column headers)
          - X-axis clusters: layout["groups"]
          - Bars within cluster: layout["colors"]
          - Points overlaid: marker encodes groups, color encodes colors with per-rep hue shading.

        Pre-plot transformation (point_reduce):
          Use this to derive new per-point values before plotting (e.g., delta per rep, or averaging tech reps):
          - Built-in 'collapse':
              point_reduce={
                  "type": "collapse",
                  "over": ["tech_rep"],                  # list of trait cols to collapse over
                  "how": np.nanmean,                    # reducer(values) -> scalar
                  "label": None                         # optional new color label; defaults to keeping original __COLOR
              }
            Result: each point is the reducer across all rows that differ only by 'over'.
          - Built-in 'pair':
              point_reduce={
                  "type": "pair",
                  "pivot": "strain",                    # trait column over which to pair (default: layout["colors"])
                  "levels": ("epec_tol", "epec_wt"),    # A, B levels along pivot
                  "how": lambda a, b: a - b,            # reducer for pair (A, B) -> scalar
                  "label": "tol - wt",                  # new synthetic color label for the reduced value
                  "match_on": None                      # list of traits to match pairs on; default: ["__ROW","__COL","__GROUP","__HUE"]
              }
            Result: each point becomes how(value[A], value[B]) for matching units; plotting colors collapse to a single label.
          - Callable:
              def my_reduce(df_raw: pd.DataFrame) -> pd.DataFrame:
                  # must return DataFrame with columns: value, __ROW, __COL, __GROUP, __COLOR, __HUE
                  return df_transformed

        Examples:
          1) Average across tech reps per unit:
             point_reduce={"type":"collapse","over":["tech_rep"],"how":np.nanmean}
          2) Per-rep delta between two strains:
             layout["colors"]="strain", layout["reps"]="rep"
             point_reduce={"type":"pair","pivot":"strain","levels":("epec_tol","epec_wt"),"how":lambda a,b: a-b,"label":"Δ tol-wt"}

        """
        # Helpers to support composite traits (lists/tuples) expanded as Cartesian products
        def _as_list(x):
            if x is None or x == "":
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        def _flatten_traits(*specs):
            out = []
            for sp in specs:
                out.extend(_as_list(sp))
            return out

        def _ordered_levels(series, tname):
            if tname in ords:
                return [lvl for lvl in ords[tname] if lvl in set(series)]
            uniq = list(pd.unique(series))
            try:
                return sorted(uniq)
            except Exception:
                return uniq

        def _build_levels_and_assign(df, traits, out_col):
            # traits: list of column names. If 0 -> single "All".
            if len(traits) == 0:
                levels = ["All"]
                df[out_col] = "All"
                return levels
            if len(traits) == 1:
                t = traits[0]
                levels = _ordered_levels(df[t], t)
                df[out_col] = df[t]
                return levels
            # Composite: cartesian product of per-trait ordered levels, assign tuple keys
            comp_levels_lists = [_ordered_levels(df[t], t) for t in traits]
            levels = list(product(*comp_levels_lists))
            df[out_col] = list(zip(*[df[t] for t in traits]))
            return levels

        def _format_level_label(traits, level_val):
            if len(traits) == 0:
                return ""
            if len(traits) == 1:
                return f"{traits[0]} = {level_val}"
            # Composite
            parts = []
            for t, v in zip(traits, level_val):
                parts.append(f"{t}={v}")
            return ", ".join(parts)

        rows_trait_spec   = layout.get("rows")
        cols_trait_spec   = layout.get("cols")
        groups_trait_spec = layout.get("groups")
        colors_trait_spec = layout.get("colors")
        reps_trait_spec   = layout.get("reps")

        # Normalize to lists
        rows_traits   = _as_list(rows_trait_spec)
        cols_traits   = _as_list(cols_trait_spec)
        groups_traits = _as_list(groups_trait_spec)
        colors_traits = _as_list(colors_trait_spec)
        reps_traits   = _as_list(reps_trait_spec)

        filt = filter_traits or {}
        ords = orderings or {}

        # Determine hue source for per-point shade (can also be composite)
        if point_hue_by is None:
            point_hue_by = reps_trait_spec
        hue_traits = _as_list(point_hue_by)

        # Gather per-well raw values
        needed_traits = set(_flatten_traits(rows_traits, cols_traits, groups_traits, colors_traits, reps_traits, hue_traits)) | set(filt.keys())
        records = []
        for w in getattr(self, "wells", []):
            # Pre-filter by fixed trait constraints
            if any(self.well_to_trait_value(w, t) != v for t, v in filt.items()):
                continue
            try:
                val = value_fn(w)
            except Exception:
                continue
            if val is None or not np.isfinite(val):
                continue

            rec = {"well": w, "value": float(val)}
            for t in needed_traits:
                rec[t] = self.well_to_trait_value(w, t)
            records.append(rec)

        df_raw = pd.DataFrame.from_records(records)
        if df_raw.empty:
            raise ValueError("No data after filtering and value computation.")

        # Assign plotting categories and compute ordered levels, supporting composite specs
        row_levels   = _build_levels_and_assign(df_raw, rows_traits,   "__ROW")
        col_levels   = _build_levels_and_assign(df_raw, cols_traits,   "__COL")
        group_levels = _build_levels_and_assign(df_raw, groups_traits, "__GROUP")
        color_levels = _build_levels_and_assign(df_raw, colors_traits, "__COLOR")
        hue_levels   = _build_levels_and_assign(df_raw, hue_traits,    "__HUE")

        # ------------------------------
        # Pre-plot reduction (optional)
        # ------------------------------
        def _collapse(df, over, how, new_label=None):
            # Group by everything except columns in "over" and measurements/well
            drop_cols = set(over) | {"well"}
            keep_cols = [c for c in df.columns if c not in drop_cols and c != "value"]
            agg = df.groupby(keep_cols, dropna=False)["value"].apply(lambda s: how(s.values)).reset_index()
            if new_label is not None:
                agg["__COLOR"] = new_label
            return agg

        def _pair(df, pivot, levels, how, label, match_on=None):
            # Map pivot to internal if it references any layout trait spec or internal names
            pivot_col = pivot
            # Accept internal names directly
            if pivot in {"__COLOR", "__ROW", "__COL", "__GROUP", "__HUE"}:
                pivot_col = pivot
            else:
                # If pivot equals entire spec (including composite lists/tuples) or one of its components
                if pivot == colors_trait_spec or pivot in colors_traits:
                    pivot_col = "__COLOR"
                elif pivot == rows_trait_spec or pivot in rows_traits:
                    pivot_col = "__ROW"
                elif pivot == cols_trait_spec or pivot in cols_traits:
                    pivot_col = "__COL"
                elif pivot == groups_trait_spec or pivot in groups_traits:
                    pivot_col = "__GROUP"
                elif pivot == point_hue_by or pivot in hue_traits:
                    pivot_col = "__HUE"

            A, B = levels
            # By default, pair within unit: row/col/group/hue (i.e., per rep if hue=reps)
            if match_on is None:
                match_on = ["__ROW", "__COL", "__GROUP", "__HUE"]

            # Build a tidy table: one row per (match_on + pivot) with single value
            cols_needed = match_on + [pivot_col, "value"]
            tbl = df[cols_needed].copy()

            # Pivot pivot_col to wide with columns [A, B]
            wide = tbl.pivot_table(index=match_on, columns=pivot_col, values="value", aggfunc="mean").reset_index()
            if A not in wide.columns or B not in wide.columns:
                # No pairs found; return empty safe frame with expected columns
                out = df.head(0)[["__ROW","__COL","__GROUP","__HUE","__COLOR","value"]].copy()
                return out

            # Compute pairwise reduction
            wide["value"] = wide.apply(lambda r: how(r[A], r[B]), axis=1)

            # Rebuild a narrow frame with single synthetic color
            out = wide[match_on + ["value"]].copy()
            out["__COLOR"] = label
            return out

        if point_reduce is not None:
            if callable(point_reduce):
                df_raw = point_reduce(df_raw.copy())
                # Ensure required columns exist
                required_cols = {"value", "__ROW", "__COL", "__GROUP", "__COLOR", "__HUE"}
                missing = required_cols - set(df_raw.columns)
                if missing:
                    raise ValueError(f"Custom reducer must return columns {required_cols}, missing: {missing}")
                # Recompute levels (color_levels might have collapsed)
                row_levels   = list(pd.unique(df_raw["__ROW"]))
                col_levels   = list(pd.unique(df_raw["__COL"]))
                group_levels = list(pd.unique(df_raw["__GROUP"]))
                color_levels = list(pd.unique(df_raw["__COLOR"]))
                hue_levels   = list(pd.unique(df_raw["__HUE"]))
            elif isinstance(point_reduce, dict):
                pr = point_reduce
                pr_type = pr.get("type", "").lower()
                if pr_type == "collapse":
                    over = pr.get("over")
                    how = pr.get("how", np.nanmean)
                    label = pr.get("label", None)
                    if not over or not callable(how):
                        raise ValueError("collapse reducer requires 'over' (list) and callable 'how'.")
                    df_raw = _collapse(df_raw, over=over, how=how, new_label=label)
                    # If we relabeled colors to a single label, update color_levels
                    if label is not None:
                        color_levels = [label]
                    else:
                        color_levels = list(pd.unique(df_raw["__COLOR"]))
                    # Update other levels
                    row_levels   = list(pd.unique(df_raw["__ROW"]))
                    col_levels   = list(pd.unique(df_raw["__COL"]))
                    group_levels = list(pd.unique(df_raw["__GROUP"]))
                    hue_levels   = list(pd.unique(df_raw["__HUE"]))
                elif pr_type == "pair":
                    pivot = pr.get("pivot", colors_trait_spec or "__COLOR")
                    levels = pr.get("levels")
                    how = pr.get("how", lambda a, b: a - b)
                    label = pr.get("label", "reduced")
                    match_on = pr.get("match_on", None)
                    if not levels or not callable(how):
                        raise ValueError("pair reducer requires 'levels' (A,B) and callable 'how'.")
                    df_raw = _pair(df_raw, pivot=pivot, levels=levels, how=how, label=label, match_on=match_on)
                    # After pair, we typically have a single color label
                    color_levels = [label]
                    # Update other levels
                    row_levels   = list(pd.unique(df_raw["__ROW"]))
                    col_levels   = list(pd.unique(df_raw["__COL"]))
                    group_levels = list(pd.unique(df_raw["__GROUP"]))
                    hue_levels   = list(pd.unique(df_raw["__HUE"]))
                else:
                    raise ValueError("Unsupported point_reduce['type']. Use 'collapse', 'pair', or a callable.")
            else:
                raise ValueError("point_reduce must be None, dict, or callable.")

        # Set categoricals for consistent plotting order
        for col_name, levels in [
            ("__ROW", row_levels), ("__COL", col_levels),
            ("__GROUP", group_levels), ("__COLOR", color_levels), ("__HUE", hue_levels)
        ]:
            df_raw[col_name] = pd.Categorical(df_raw[col_name], categories=levels, ordered=True)

        # Aggregate for bars: mean/std per (row,col,group,color)
        agg_keys = ["__ROW", "__COL", "__GROUP", "__COLOR"]
        df_stats = df_raw.groupby(agg_keys, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()

        # Fast count lookup for min_reps filtering
        counts_lookup = {
            (r, c, g, col): int(cnt)
            for r, c, g, col, cnt in zip(
                df_stats["__ROW"], df_stats["__COL"], df_stats["__GROUP"], df_stats["__COLOR"], df_stats["count"]
            )
        }

        # Subplots
        n_rows, n_cols = len(row_levels), len(col_levels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey, squeeze=False)

        # Styles: color palette and group markers
        palette = sns.color_palette("tab10", n_colors=max(1, len(color_levels)))
        color_map = dict(zip(color_levels, palette))
        marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
        marker_map = dict(zip(group_levels, marker_cycle[:len(group_levels)]))

        # Per-hue shading for points
        point_color_map = {}
        n_h = max(1, len(hue_levels))
        for c_lvl in color_levels:
            base_rgb = _rgba_to_rgb_tuple(color_map[c_lvl])
            for i, h_lvl in enumerate(hue_levels):
                point_color_map[(c_lvl, h_lvl)] = _shade_from_base(base_rgb, i, n_h)

        # Column headers (top row only)
        if len(cols_traits) > 0:
            for j, c_val in enumerate(col_levels):
                axes[0, j].set_title(_format_level_label(cols_traits, c_val), y=1.02, fontsize=12)

        # Row headers (leftmost column only)
        if len(rows_traits) > 0:
            for i, r_val in enumerate(row_levels):
                axes[i, 0].set_ylabel(_format_level_label(rows_traits, r_val), fontsize=12)

        # Plot per subplot
        for (r_i, r_val), (c_j, c_val) in product(enumerate(row_levels), enumerate(col_levels)):
            ax = axes[r_i, c_j]
            sub_stats = df_stats[(df_stats["__ROW"] == r_val) & (df_stats["__COL"] == c_val)]
            sub_raw   = df_raw[(df_raw["__ROW"] == r_val) & (df_raw["__COL"] == c_val)]
            if sub_stats.empty and sub_raw.empty:
                ax.set_visible(False)
                continue

            # Grouped bars: x-axis clusters by group; within each cluster bars by color
            gx = np.arange(len(group_levels), dtype=float)
            nC = max(1, len(color_levels))
            width = 0.8 / nC
            offsets = (np.arange(nC) - (nC - 1) / 2.0) * width

            # Draw bars and phantom bars
            for c_idx, c_val2 in enumerate(color_levels):
                xs, ys, es, masks = [], [], [], []
                for g_idx, g_val in enumerate(group_levels):
                    xs.append(gx[g_idx] + offsets[c_idx])
                    g = sub_stats[(sub_stats["__GROUP"] == g_val) & (sub_stats["__COLOR"] == c_val2)]
                    cnt = counts_lookup.get((r_val, c_val, g_val, c_val2), 0)
                    if g.empty or cnt < min_reps:
                        ys.append(0.0); es.append(0.0); masks.append(False)
                    else:
                        ys.append(float(g["mean"].iloc[0]))
                        es.append(0.0 if pd.isna(g["std"].iloc[0]) else float(g["std"].iloc[0]))
                        masks.append(True)

                xs = np.array(xs); ys = np.array(ys); es = np.array(es)
                real_mask = np.array(masks, dtype=bool)
                if np.any(real_mask):
                    ax.bar(xs[real_mask], ys[real_mask], yerr=es[real_mask], width=width,
                           color=color_map[c_val2], alpha=bar_alpha,
                           edgecolor=edgecolor, linewidth=0.8, capsize=capsize)
                phantom_mask = ~real_mask
                if np.any(phantom_mask):
                    ax.bar(xs[phantom_mask], np.zeros_like(xs[phantom_mask]),
                           width=width, color="none", edgecolor="none", linewidth=0.0)

            # Overlay points with per-hue shading
            if not sub_raw.empty:
                x_pos = {(g, c): gx[g_i] + offsets[c_i]
                         for g_i, g in enumerate(group_levels)
                         for c_i, c in enumerate(color_levels)}
                for _, row in sub_raw.iterrows():
                    g = row["__GROUP"]; c = row["__COLOR"]; h = row["__HUE"]
                    x_center = x_pos.get((g, c))
                    if x_center is None:
                        continue
                    xj = float(x_center + np.random.uniform(-jitter, jitter) * width)
                    shade_col = point_color_map.get((c, h), _rgba_to_rgb_tuple(color_map[c]))
                    ax.scatter([xj], [row["value"]],
                               s=point_size, marker=marker_map[g], color=shade_col,
                               edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3)

            ax.set_xticks(gx)
            if show_group_labels:
                ax.set_xticklabels([str(x) for x in group_levels])
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)

            if len(rows_traits) == 0 and c_j == 0:
                ax.set_ylabel("Value")
            ax.grid(True, axis="y", alpha=0.25)

        # Global title
        if title:
            fig.suptitle(title, y=0.98)

        # Global legend for colors and groups
        color_handles = [Patch(facecolor=color_map[lvl], edgecolor="black", label=str(lvl)) for lvl in color_levels]
        marker_handles = [Line2D([0], [0], marker=marker_map[lvl], color="black", linestyle="None",
                                 markerfacecolor="white", markersize=9, label=str(lvl))
                          for lvl in group_levels]

        legend_sections = []
        if len(color_levels) > 0:
            legend_sections.append(("colors", color_handles))
        if len(group_levels) > 0:
            legend_sections.append(("groups", marker_handles))

        # Optional legend for hue ramp (e.g., reps)
        hue_handles = []
        if show_point_hue_legend and len(hue_traits) > 0 and len(hue_levels) > 0 and len(color_levels) > 0:
            base_c = color_levels[0]
            for h_lvl in hue_levels:
                hue_color = point_color_map[(base_c, h_lvl)]
                hue_handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None",
                                          markerfacecolor=hue_color, markersize=8, label=str(h_lvl)))
            # Show label based on provided hue trait names
            legend_sections.append((", ".join(hue_traits), hue_handles))

        if legend_sections:
            handles = []
            titles = []
            for sec_title, hs in legend_sections:
                handles.extend(hs); titles.append(sec_title)
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                       frameon=False, title="; ".join(titles))

        plt.tight_layout(rect=[0, 0, 0.85, 0.97] if legend_sections else [0, 0, 1, 0.97])
        return fig, axes, df_raw, df_stats


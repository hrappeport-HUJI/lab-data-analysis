"""Interactive per-condition histograms (plotly): one panel per condition, shared bins and x axis.

The x axis runs from 0 to 1.1 x the largest value; drag to zoom, double-click to reset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# fixed categorical order (validated palette, light surface); a condition keeps its slot
CONDITION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8a85"
TEXT = "#3b3b38"
GRID = "#e4e3de"


def condition_color(k: int) -> str:
    return CONDITION_COLORS[k] if k < len(CONDITION_COLORS) else GREY


def bin_edges(values: np.ndarray, width: float | None = None) -> np.ndarray:
    """Edges covering all values. Bins are centred on the smallest value, so values sampled every
    ``width`` (frame times) each fall in the middle of their own bin."""
    lo, hi = float(values.min()), float(values.max())
    if width is None or width <= 0:
        auto = np.histogram_bin_edges(values, "fd") if hi > lo else np.array([lo, lo + 1])
        width = max(auto[1] - auto[0], (hi - lo) / 200) if hi > lo else 1.0
    n = int(np.floor((hi - lo) / width + 0.5)) + 1
    return lo - width / 2 + width * np.arange(n + 1)


def histograms(colonies: pd.DataFrame, conditions: list[str], metric: str, title: str, xlabel: str, unit: str,
               width: float | None = None) -> go.Figure:
    pooled = colonies[metric].dropna().to_numpy()
    edges = bin_edges(pooled, width) if len(pooled) else np.array([0.0, 1.0])
    centers, w = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    n = len(conditions)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.08, 0.3 / n),
                        subplot_titles=[f"<b>{c}</b>" for c in conditions])
    for k, cond in enumerate(conditions, start=1):
        v = colonies.loc[colonies.condition == cond, metric].dropna().to_numpy()
        counts, _ = np.histogram(v, edges)
        frac = counts / max(len(v), 1)
        fig.add_trace(go.Bar(
            x=centers, y=frac, width=w * 0.92, marker_color=condition_color(k - 1), name=cond, showlegend=False,
            customdata=np.c_[edges[:-1], edges[1:], counts],
            hovertemplate=(f"<b>{cond}</b><br>%{{customdata[0]:.3g}} to %{{customdata[1]:.3g}} {unit}<br>"
                           "%{customdata[2]} colonies (%{y:.1%})<extra></extra>")), row=k, col=1)
        if len(v):
            med = float(np.median(v))
            fig.add_vline(x=med, line=dict(color=TEXT, width=1.5, dash="dash"), row=k, col=1)
            note = f"median {med:.3g} {unit}, n = {len(v)}"
        else:
            note = "no colonies"
        fig.add_annotation(text=note, xref="x domain", yref="y domain", x=0.99, y=0.97, xanchor="right",
                           yanchor="top", showarrow=False, font=dict(size=12, color=TEXT), row=k, col=1)
        fig.update_yaxes(title_text="fraction", tickformat=".0%", gridcolor=GRID, zeroline=False, row=k, col=1)
    x_max = 1.1 * float(pooled.max()) if len(pooled) and pooled.max() > 0 else edges[-1]
    fig.update_xaxes(range=[0, x_max], showgrid=False, linecolor=GREY, ticks="outside")
    fig.update_xaxes(title_text=xlabel, row=n, col=1)
    fig.update_layout(title=dict(text=title, x=0.01), height=180 + 230 * n, bargap=0, plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=TEXT), margin=dict(l=70, r=30, t=80, b=60),
                      hovermode="closest")
    fig.update_annotations(font_size=13, selector=dict(text=None))
    return fig


def write_html(figures: list[go.Figure], path, title: str) -> None:
    """All figures in one self-contained page (plotly.js inlined, opens offline)."""
    parts = [f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False,
                       config={"displaylogo": False})
             for i, f in enumerate(figures)]
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
                    f"<body style='font-family:sans-serif;max-width:1100px;margin:auto;background:white'>"
                    + "".join(parts) + "</body></html>", encoding="utf-8")

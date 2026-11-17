// series: per-row curves (e.g. a signal over time) drawn in the clicked point's card.
// Data: B.series = {row id: {curve name: [[x...], [y...]]}}, written by write_viewer(..., series=long_table).
// features.series = {"title": "PI over time", "x_label": "Time (h)", "y_label": "PI (ADU/pixel)", "y_scale": "linear" | "log",
//   "styles": {"curve name": {"mode": "markers" | "lines" | "markers+lines", "dash": "dash", "color": "#hex", "opacity": 1}},
//   "lines": [{"column": "lag", "label": "first division"}]}      // vertical lines at a column's value for that row
const SERIES = B.series || {};
let seriesLog = null;
function drawSeries(cfg, r, el) {
  const curves = SERIES[r.id] || {}, names = Object.keys(curves);
  if (!names.length) { el.textContent = `No ${(cfg.title || "series").toLowerCase()} for this row.`; el.className = "hint"; return; }
  const ink = css("--ink"), muted = css("--muted"), line = css("--line"), base = colorOf(state.colorBy, r[state.colorBy]);
  const log = seriesLog ?? cfg.y_scale === "log";
  const traces = names.map(n => { const st = (cfg.styles || {})[n] || {}, [x, y] = curves[n], mode = st.mode || "markers+lines";
    return { type: "scatter", mode, name: n, x, y: log ? y.map(v => (v > 0 ? v : null)) : y, opacity: st.opacity ?? 1,
      marker: { color: st.color || base, size: 5 }, line: { color: st.color || base, width: 1.6, dash: st.dash || "solid" },
      hovertemplate: `${n}<br>%{x:.3g}, %{y:.4g}<extra></extra>` }; });
  const shapes = [], annotations = [];
  for (const l of cfg.lines || []) { const v = r[l.column]; if (!ok(v)) continue;
    shapes.push({ type: "line", xref: "x", x0: v, x1: v, yref: "paper", y0: 0, y1: 1, line: { color: muted, width: 1.2, dash: l.dash || "dot" } });
    annotations.push({ xref: "x", x: v, yref: "paper", y: 1, text: l.label || l.column, showarrow: false, xanchor: "left", yanchor: "top", textangle: -90, font: { size: 10.5, color: muted } }); }
  Plotly.react(el, traces, { height: 260, margin: { l: 56, r: 10, t: 8, b: 40 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--sans"), color: ink, size: 11.5 }, showlegend: names.length > 1, legend: { orientation: "h", x: 0, y: 1.14, font: { size: 11 } },
    xaxis: { title: { text: cfg.x_label || "" }, gridcolor: line, linecolor: muted, showline: true, zeroline: false },
    yaxis: { title: { text: cfg.y_label || "" }, type: log ? "log" : "linear", gridcolor: line, linecolor: muted, showline: true, zeroline: false },
    shapes, annotations }, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    toImageButtonOptions: { format: "svg", filename: `${r.id}_series` } });
}
viewerModule("series", {
  pickBody(cfg, r, box) {
    const wrap = document.createElement("div"); wrap.className = "series";
    const bar = document.createElement("div"); bar.className = "series-bar";
    const h = document.createElement("h2"); h.textContent = cfg.title || "Series"; h.style.margin = "0";
    const tog = document.createElement("button"); tog.className = "btn"; tog.type = "button";
    const el = document.createElement("div"); el.style.width = "100%";
    const sync = () => { tog.textContent = (seriesLog ?? cfg.y_scale === "log") ? "Linear Y" : "Log Y"; };
    tog.onclick = () => { seriesLog = !(seriesLog ?? cfg.y_scale === "log"); sync(); drawSeries(cfg, r, el); };
    sync(); bar.append(h, tog); wrap.append(bar, el); box.append(wrap); drawSeries(cfg, r, el);
  },
});
TOOLS.push({ name: "get_series", description: "The per-row curves (e.g. a signal over time) for one row id: {curve name: {x: [...], y: [...]}}.",
  inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  execute: i => Object.fromEntries(Object.entries(SERIES[i.id] || {}).map(([n, [x, y]]) => [n, { x, y }])) });

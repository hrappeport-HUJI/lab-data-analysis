// cdf: 1 - CDF ("survival") curves of one numeric column, one step curve per group of the page (the "Colour by" choice,
// split by the second grouping when one is set), drawn in the group's colour and line style.
// features.cdf = {
//   column: "appearance_h",               // shown first
//   columns: ["appearance_h", "doubling_time_h"],   // offered (default: every numeric column)
//   scale: "log",                         // y scale at first: "log" (default) or "linear"
//   x_scale: "linear",                    // x scale at first: "linear" (default) or "log"
//   title: "Appearance: fraction of colonies not yet seen",
//   y_label: "Fraction not yet appeared",
// }
// Rows used: pass the filters, have a value in the column, and were not removed by the person. Rows with no Y value
// and far outliers of the scatter are kept, because the tail is what a 1 - CDF is for. On a log x axis, values <= 0
// cannot be drawn: they still count in n and the fractions, and the card says how many there are.
const CDF = { cfg: null, col: null, scale: { x: "linear", y: "log" } };
CDF.store = () => STORE + ":cdf";
CDF.save = () => { try { localStorage.setItem(CDF.store(), JSON.stringify({ col: CDF.col, scale: CDF.scale })); } catch (e) {} };
CDF.offered = () => (CDF.cfg.columns && CDF.cfg.columns.length ? CDF.cfg.columns : NUM).filter(c => COLS[c] && COLS[c].kind !== "category");

CDF.init = cfg => {
  CDF.cfg = cfg; const offered = CDF.offered();
  if (!offered.length) { document.getElementById("cdf-card").hidden = true; return; }
  CDF.col = offered.includes(cfg.column) ? cfg.column : offered[0];
  CDF.scale = { x: cfg.x_scale === "log" ? "log" : "linear", y: cfg.scale === "linear" ? "linear" : "log" };
  try { const s = JSON.parse(localStorage.getItem(CDF.store()) || "null");
    if (s) { if (offered.includes(s.col)) CDF.col = s.col;
      if (s.scale && typeof s.scale === "object") for (const a of ["x", "y"]) if (s.scale[a] === "linear" || s.scale[a] === "log") CDF.scale[a] = s.scale[a]; } } catch (e) {}
  if (cfg.title) document.getElementById("cdf-title").textContent = cfg.title;
  const sel = document.getElementById("cdf-col");
  offered.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = L(c); sel.append(o); });
  sel.value = CDF.col; sel.onchange = () => { CDF.col = sel.value; CDF.save(); CDF.draw(); };
  document.querySelectorAll("[data-cdf-scale]").forEach(b => b.onclick = () => { CDF.scale[b.dataset.cdfAxis] = b.dataset.cdfScale; CDF.save(); CDF.draw(); });
  CDF.draw();
};

// Points of the step curve: starts at 1 at the smallest value, drops by 1/n at each value.
CDF.curve = vals => {
  const v = vals.slice().sort((a, b) => a - b), n = v.length, x = [], y = [];
  if (!n) return { x, y, n };
  x.push(v[0]); y.push(1);
  for (let i = 0; i < n; i++) { if (i + 1 < n && v[i + 1] === v[i]) continue; x.push(v[i]); y.push((n - i - 1) / n); }
  return { x, y, n };
};

CDF.draw = () => {
  if (!CDF.cfg || !CDF.col || !window.Plotly) return;
  document.querySelectorAll("[data-cdf-scale]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.cdfScale === CDF.scale[b.dataset.cdfAxis])));
  const ink = css("--ink"), muted = css("--muted"), line = css("--line"), col = CDF.col, logY = CDF.scale.y === "log", logX = CDF.scale.x === "log";
  const traces = [];
  let minPos = 1, hiddenX = 0;
  for (const g of groupsShown()) {
    const vals = ROWS.filter(r => groupKey(r) === g && passFilters(r) && !state.excluded.has(r.id) && ok(r[col])).map(r => r[col]);
    const c = CDF.curve(vals); if (!c.n) continue;
    minPos = Math.min(minPos, 1 / c.n);
    // A log y axis cannot show the final 0 (the curve ends at 1/n); a log x axis cannot show values <= 0.
    const keep = c.y.map((y, i) => (!logY || y > 0) && (!logX || c.x[i] > 0));
    if (logX) hiddenX += vals.filter(v => v <= 0).length;
    const st = styleOf(g);
    traces.push({ type: "scatter", mode: "lines", line: { shape: "hv", color: st.color, dash: st.dash, width: 2 }, name: `${g} (${c.n})`, legendgroup: g,
      x: c.x.filter((_, i) => keep[i]), y: c.y.filter((_, i) => keep[i]),
      hovertemplate: `<b>${g}</b><br>${short(col)} %{x:.4g}<br>above: %{y:.3g}<extra></extra>` });
  }
  const yLabel = CDF.cfg.y_label || `Fraction with ${short(col)} above x`;
  const axis = { gridcolor: line, linecolor: muted, showline: true, zeroline: false, ticks: "outside", tickcolor: muted, exponentformat: "power" };
  const layout = { margin: { l: 64, r: 12, t: 10, b: 52 }, showlegend: true, hovermode: "closest",
    legend: { orientation: "h", x: 0, y: -0.18, font: { size: 11.5, color: ink } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: css("--sans"), color: ink, size: 12 },
    xaxis: { ...axis, title: { text: L(col), font: { size: 12.5 } }, type: logX ? "log" : "linear" },
    yaxis: { ...axis, title: { text: yLabel, font: { size: 12.5 } }, type: logY ? "log" : "linear", range: logY ? [Math.log10(minPos) - .1, 0.05] : [0, 1.03] } };
  Plotly.react("cdf-plot", traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    toImageButtonOptions: { format: "svg", filename: `one_minus_cdf_${col}` } });
  const hint = document.getElementById("cdf-hint"); hint.dataset.base = hint.dataset.base || hint.textContent;
  hint.textContent = hint.dataset.base + (hiddenX ? ` ${hiddenX} value${hiddenX > 1 ? "s" : ""} at or below 0 cannot be drawn on the log x axis (still counted).` : "");
};

viewerModule("cdf", {
  init(cfg) { CDF.init(cfg); },
  drawn() { CDF.draw(); },
});

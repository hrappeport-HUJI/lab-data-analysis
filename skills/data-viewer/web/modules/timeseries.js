// timeseries: stacked curve panels under the scatter; the picked row is highlighted in every panel.
// features.timeseries = {
//   series:   {series_id: {channel: {t: [...], v: [...]}}},       // one entry per measured object (a well, a cell, ...)
//   channels: {channel: {label: "OD630, background subtracted", scale: "log"}},
//   time_label: "Time since inoculation (h)",
//   panels: [{name: "Regrowth well", series_col: "well_regrowth",          // row column holding the series id
//             events: {"transfer": "t_transfer_h"},                         // row columns holding event times (same clock as t)
//             channels: ["OD", "A560"], x: "transfer", scale: "log", channel: "OD"}],  // offered channels and defaults
//   show: [0, 1, 2],   // panels shown at first (indices into panels), each a separate subplot
//   max_panels: 6,
// }
const TS = { cfg: null, choice: [], n: 0, fade: true };
TS.def = k => TS.cfg.panels[k];
TS.channelsOf = def => (def.channels && def.channels.length ? def.channels : Object.keys(TS.cfg.channels || {}));
TS.defaultChoice = k => { const d = TS.def(k), ch = TS.channelsOf(d);
  return { panel: k, channel: d.channel && ch.includes(d.channel) ? d.channel : ch[0], x: d.x || "absolute", scale: d.scale || (TS.cfg.channels[d.channel || ch[0]] || {}).scale || "linear" }; };
TS.store = () => STORE + ":timeseries";
TS.save = () => { try { localStorage.setItem(TS.store(), JSON.stringify({ choice: TS.choice, n: TS.n, fade: TS.fade })); } catch (e) {} };

TS.init = cfg => {
  TS.cfg = cfg; const P = cfg.panels || [];
  if (!P.length) { document.getElementById("ts-card").hidden = true; return; }
  const max = Math.max(1, Math.min(cfg.max_panels || 6, 8));
  const show = (cfg.show && cfg.show.length ? cfg.show : P.map((_, k) => k)).filter(k => P[k]).slice(0, max);
  TS.choice = Array.from({ length: max }, (_, i) => TS.defaultChoice(show[i] ?? i % P.length));
  TS.n = show.length;
  try { const s = JSON.parse(localStorage.getItem(TS.store()) || "null");
    if (s && Array.isArray(s.choice)) { s.choice.forEach((c, i) => { if (i < max && P[c.panel]) TS.choice[i] = { ...TS.defaultChoice(c.panel), ...c }; }); TS.n = Math.max(1, Math.min(max, s.n | 0 || TS.n)); TS.fade = s.fade !== false; } } catch (e) {}
  const nsel = document.getElementById("ts-n");
  nsel.innerHTML = Array.from({ length: max }, (_, i) => `<option value="${i + 1}">${i + 1}</option>`).join("");
  nsel.onchange = () => { TS.n = +nsel.value; TS.save(); TS.controls(); TS.draw(); };
  const fade = document.getElementById("ts-fade"); fade.onchange = () => { TS.fade = fade.checked; TS.save(); TS.draw(); };
  document.getElementById("ts-clear").onclick = () => showPick(null);
  TS.controls(); TS.draw();
  document.getElementById("ts-plot").on("plotly_click", ev => { const p = ev.points && ev.points[0]; if (p && p.data && p.data.meta != null) showPick(ROWS[p.data.meta]); });
};

TS.controls = () => {
  document.getElementById("ts-n").value = String(TS.n); document.getElementById("ts-fade").checked = TS.fade;
  const box = document.getElementById("ts-controls"); box.replaceChildren();
  for (let i = 0; i < TS.n; i++) {
    const c = TS.choice[i], d = TS.def(c.panel), row = document.createElement("div"); row.className = "ts-row";
    const num = document.createElement("span"); num.className = "num"; num.textContent = i + 1;
    const mk = (opts, val, aria, on) => { const s = document.createElement("select"); s.setAttribute("aria-label", aria);
      opts.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; s.append(o); }); s.value = val; s.onchange = () => on(s.value); return s; };
    const ps = mk(TS.cfg.panels.map((p, k) => [String(k), p.name]), String(c.panel), `Panel ${i + 1} curves`, v => { TS.choice[i] = TS.defaultChoice(+v); TS.save(); TS.controls(); TS.draw(); });
    const chs = TS.channelsOf(d);
    const cs = mk(chs.map(k => [k, (TS.cfg.channels[k] || {}).label || k]), c.channel, `Panel ${i + 1} measurement`, v => { c.channel = v; c.scale = (TS.cfg.channels[v] || {}).scale || c.scale; TS.save(); TS.controls(); TS.draw(); });
    const xs = mk([["absolute", TS.cfg.time_label || "Time (h)"], ...Object.keys(d.events || {}).map(e => [e, `Hours since ${e}`])], c.x, `Panel ${i + 1} time axis`, v => { c.x = v; TS.save(); TS.draw(); });
    const seg = document.createElement("div"); seg.className = "seg"; seg.setAttribute("role", "group"); seg.setAttribute("aria-label", `Panel ${i + 1} y scale`);
    for (const [v, t] of [["linear", "Linear"], ["log", "Log"]]) { const b = document.createElement("button"); b.type = "button"; b.textContent = t; b.setAttribute("aria-pressed", String(c.scale === v));
      b.onclick = () => { c.scale = v; TS.save(); TS.controls(); TS.draw(); }; seg.append(b); }
    row.append(num, ps, cs, xs, seg); box.append(row);
  }
};

TS.draw = () => {
  if (!TS.cfg || !(TS.cfg.panels || []).length || !window.Plotly) return;
  const ink = css("--ink"), muted = css("--muted"), line = css("--line"), accent = css("--accent");
  const rows = ROWS.filter(r => passFilters(r) && !state.excluded.has(r.id));
  const n = TS.n, gap = .06, h = (1 - gap * (n - 1)) / n;
  const traces = [], shapes = [], annotations = [];
  const layout = { height: 70 + 230 * n, margin: { l: 64, r: 12, t: 26, b: 46 }, showlegend: false, hovermode: "closest",
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: css("--sans"), color: ink, size: 12 }, shapes, annotations };
  for (let i = 0; i < n; i++) {
    const c = TS.choice[i], d = TS.def(c.panel), ax = i ? String(i + 1) : "", top = 1 - i * (h + gap);
    const xLabel = c.x === "absolute" ? (TS.cfg.time_label || "Time (h)") : `Hours since ${c.x}`;
    layout["xaxis" + ax] = { domain: [0, 1], anchor: "y" + ax, title: { text: xLabel, font: { size: 11.5 }, standoff: 4 }, gridcolor: line, linecolor: muted, showline: true, zeroline: false, ticks: "outside", tickcolor: muted };
    layout["yaxis" + ax] = { domain: [top - h, top], anchor: "x" + ax, type: c.scale === "log" ? "log" : "linear", title: { text: (TS.cfg.channels[c.channel] || {}).label || c.channel, font: { size: 11.5 } },
      gridcolor: line, linecolor: muted, showline: true, zeroline: false, ticks: "outside", tickcolor: muted, exponentformat: "power" };
    annotations.push({ xref: "paper", yref: "paper", x: 0, y: top, yanchor: "bottom", xanchor: "left", showarrow: false, text: `<b>${i + 1}. ${d.name}</b>`, font: { size: 12, color: ink } });
    const seen = new Set(), pickedSid = picked ? picked[d.series_col] : null, mine = [];
    for (const r of rows) {
      const sid = r[d.series_col]; if (sid == null || seen.has(sid)) continue; seen.add(sid);
      const s = ((TS.cfg.series || {})[sid] || {})[c.channel]; if (!s) continue;
      const off = c.x === "absolute" ? 0 : r[(d.events || {})[c.x]]; if (off == null) continue;
      const isP = pickedSid != null && sid === pickedSid, faded = picked && TS.fade && !isP;
      const col = styleOf(groupKey(r), r).color;
      const tr = { type: "scattergl", mode: "lines", xaxis: "x" + ax, yaxis: "y" + ax, x: s.t.map(t => t - off), y: s.v, meta: (isP ? picked : r).i,
        line: { color: faded ? hexA(col, .16) : col, width: isP ? 3.2 : 1.2 }, opacity: faded ? 1 : (isP ? 1 : .8),
        hovertemplate: `<b>${label(isP ? picked : r)}</b><br>${sid}<br>%{x:.2f} h, %{y:.4g}<extra></extra>` };
      (isP ? mine : traces).push(tr);
    }
    traces.push(...mine);  // the picked curve on top
    if (picked) for (const [e, col] of Object.entries(d.events || {})) {
      const t = picked[col]; if (t == null) continue; const off = c.x === "absolute" ? 0 : picked[(d.events || {})[c.x]]; if (off == null) continue;
      shapes.push({ type: "line", xref: "x" + ax, x0: t - off, x1: t - off, yref: "paper", y0: top - h, y1: top, line: { color: accent, width: 1.2, dash: "dash" } });
      annotations.push({ xref: "x" + ax, x: t - off, yref: "paper", y: top, yanchor: "top", xanchor: "left", showarrow: false, text: " " + e, font: { size: 10.5, color: accent } });
    }
  }
  const hint = document.getElementById("ts-hint");
  hint.dataset.base = hint.dataset.base || hint.textContent;
  hint.textContent = (picked ? `Highlighted: ${label(picked)}. ` : "") + hint.dataset.base;
  Plotly.react("ts-plot", traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"], toImageButtonOptions: { format: "svg", filename: "curves" } });
};

viewerModule("timeseries", {
  init(cfg) { TS.init(cfg); },
  drawn() { TS.draw(); },
});

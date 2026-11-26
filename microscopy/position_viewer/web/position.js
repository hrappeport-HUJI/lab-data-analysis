"use strict";
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const palettes = ["grey", "red", "green", "fire", "HiLo"];
let channels = [];
const tones = { division: "#69e9ae", "probably dead (PI)": "#ff4d8d", loss: "#ffa257", exclusion: "#b0a2f3" };
const fates = { division: tones.division, "probably dead (PI)": tones["probably dead (PI)"], loss: tones.loss, "no division": "#7db8ff", exclusion: tones.exclusion, "not assessed": "#bccbd1" };
const outcomes = {
  divided: "division",
  unresolved: "loss",
  excluded: "exclusion",
  not_grown: "no division",
  pi_positive_death: "probably dead (PI)",
};
const STORAGE_BC = "pv2_channels_v1";
const STORAGE_OVERLAY = "pv2_overlay_v1";
const maxFrame = () => (meta ? (meta.times.length ?? Object.keys(meta.times).length) - 1 : 0);
const configs = Object.create(null),
  positions = new Map(),
  cache = new Map();
const overlayPrefs = { lineWidth: 2.0, labelSize: 12 };
let pos,
  meta,
  f = 0,
  width = 2048,
  height = 2048,
  cx = width / 2,
  cy = height / 2,
  factor = 1,
  chosen = null,
  editing = false,
  bc = null;
let epoch = 0,
  playing = false,
  overviewSlow = false,
  loopId = 0,
  scene = null,
  pointer = null,
  drag = null;

// Histogram interactive slider state
const HIST_PAD_L = 42,
  HIST_PAD_R = 14,
  HIST_TOP = 20,
  HIST_BOT = 112;
let histState = { min: 0, max: 65535, dragMode: null, startX: 0, startLo: 0, startHi: 0, lockedDomain: null };
let channelPrefs = Object.create(null);

function loadSavedPrefs() {
  try {
    const rawOv = localStorage.getItem(STORAGE_OVERLAY);
    if (rawOv) {
      const ov = JSON.parse(rawOv);
      if (typeof ov.lineWidth === "number") overlayPrefs.lineWidth = clamp(ov.lineWidth, 0.5, 4.5);
      if (typeof ov.labelSize === "number") overlayPrefs.labelSize = clamp(Math.round(ov.labelSize), 8, 24);
    }
  } catch (_) {}
  try {
    const rawBc = localStorage.getItem(STORAGE_BC);
    if (rawBc) channelPrefs = Object.assign(Object.create(null), JSON.parse(rawBc));
  } catch (_) {}
}
function saveChannelPrefs() {
  try {
    for (const ch of channels) {
      const s = configs[ch];
      if (!s) continue;
      channelPrefs[ch] = {
        on: Boolean(s.on),
        lut: s.lut,
        lo: s.lo,
        hi: s.hi,
        auto: Boolean(s.auto),
        pinned: Boolean(s.pinned),
      };
    }
    localStorage.setItem(STORAGE_BC, JSON.stringify(channelPrefs));
  } catch (_) {}
}
function saveOverlayPrefs() {
  try {
    localStorage.setItem(STORAGE_OVERLAY, JSON.stringify(overlayPrefs));
  } catch (_) {}
}

const svgNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs, parent) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent.append(el);
  return el;
}
function button(label, fn, parent) {
  const b = document.createElement("button");
  b.textContent = label;
  b.onclick = fn;
  parent.append(b);
  return b;
}
function options(select, values) {
  for (const v of values) select.add(new Option(v, v));
}
function failure(e) {
  stop();
  $("status").textContent = "Load failed · " + e.message;
}
function get(route, args = {}, binary = false) {
  const url =
    "/position-data/" + route + "?" + new URLSearchParams({ pos, ...args });
  if (cache.has(url)) return cache.get(url);
  const result = fetch(url)
    .then((r) => {
      if (!r.ok) throw Error(route + " returned " + r.status);
      return binary ? r.arrayBuffer() : r.json();
    })
    .catch((err) => {
      cache.delete(url);
      throw err;
    });
  cache.set(url, result);
  if (cache.size > 300) cache.delete(cache.keys().next().value);
  return result;
}
function stillSize() {
  return 1024;
}
function geometry() {
  const n = clamp(Math.round($("zoom").clientWidth / factor), 1, Math.min(1024, width, height));
  const x = clamp(Math.round(cx - n / 2), 0, width - n),
    y = clamp(Math.round(cy - n / 2), 0, height - n);
  return { x, y, n };
}
function overviewSize(level) {
  const longest = Math.max(width, height);
  return [width, height].map((side) => Math.max(1, Math.round(side * Math.min(level, longest) / longest)));
}
function level(hist, pct) {
  if (!hist) return [0, 65535];
  const [v, c] = hist,
    tail = (c.reduce((s, n) => s + n, 0) * pct) / 200;
  let a = 0,
    b = c.length - 1,
    sum = 0;
  while (a < b && sum + c[a] <= tail) sum += c[a++];
  sum = 0;
  while (b > a && sum + c[b] <= tail) sum += c[b--];
  return [Math.min(65534, v[a]), Math.max(v[b], Math.min(65534, v[a]) + 1)];
}
const _IS_LE = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
function makeLut(s) {
  const key = [s.lo, s.hi, s.lut].join("-");
  if (s.key === key) return;
  s.key = key;
  if (!s.table) s.table = new Uint8Array(65536);
  if (!s.u32) s.u32 = new Uint32Array(65536);
  if (!s.rgb) s.rgb = [new Uint8Array(256), new Uint8Array(256), new Uint8Array(256)];
  const [R, G, B] = s.rgb;
  const u32_256 = new Uint32Array(256);
  for (let v = 0; v < 256; v++) {
    let r = v, g = v, b = v;
    if (s.lut === "red") { g = 0; b = 0; }
    else if (s.lut === "green") { r = 0; b = 0; }
    else if (s.lut === "fire") { r = 3 * v; g = 3 * v - 255; b = 3 * v - 510; }
    else if (s.lut === "HiLo") {
      if (v === 0) { r = 0; g = 70; b = 255; }
      else if (v === 255) { r = 255; g = 0; b = 0; }
    }
    r = clamp(r, 0, 255); g = clamp(g, 0, 255); b = clamp(b, 0, 255);
    R[v] = r; G[v] = g; B[v] = b;
    u32_256[v] = _IS_LE
      ? ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0
      : ((r << 24) | (g << 16) | (b << 8) | 255) >>> 0;
  }
  const lo = s.lo, span = Math.max(1, s.hi - lo), scale = 255 / span;
  const tbl = s.table, u32 = s.u32;
  for (let i = 0; i < 65536; i++) {
    const v = i <= lo ? 0 : i >= s.hi ? 255 : ((i - lo) * scale + 0.5) | 0;
    tbl[i] = v;
    u32[i] = u32_256[v];
  }
}
function unpack(buffer, w, h, acquired, mask) {
  const planes = Object.create(null),
    len = w * h;
  acquired.forEach(
    (ch, i) => (planes[ch] = new Uint16Array(buffer, 2 * len * i, len)),
  );
  return {
    planes,
    mask: mask ? new Uint8Array(buffer, 2 * len * acquired.length, len) : null,
    w, h,
  };
}
function paint(id, tile) {
  const { w, h } = tile;
  const c = $(id).querySelector("canvas");
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
    c._img = null;
  }
  c.classList.toggle(
    "shrunk",
    w > $(id).clientWidth * (window.devicePixelRatio || 1),
  );
  const ctx = c._ctx || (c._ctx = c.getContext("2d", { alpha: false }));
  if (!c._img) {
    c._img = ctx.createImageData(w, h);
    c._u32 = new Uint32Array(c._img.data.buffer);
  }
  const img = c._img, data = img.data, u32 = c._u32, len = w * h;
  let first = true;
  for (const ch of channels) {
    const s = configs[ch], raw = tile.planes[ch];
    if (!s.on || !raw) continue;
    makeLut(s);
    if (first) {
      const lut = s.u32;
      for (let i = 0; i < len; i++) u32[i] = lut[raw[i]];
      first = false;
    } else {
      const tbl = s.table, [r, g, b] = s.rgb;
      for (let i = 0, j = 0; i < len; i++, j += 4) {
        const v = tbl[raw[i]];
        data[j] += r[v];
        data[j + 1] += g[v];
        data[j + 2] += b[v];
      }
    }
  }
  if (first) u32.fill(_IS_LE ? 0xff000000 : 0x000000ff);
  const m = tile.mask;
  if (m) {
    const lastRow = len - w;
    for (let i = 0, j = 0; i < len; i++, j += 4) {
      if (!m[i]) continue;
      const col = i % w,
        edge =
          (col > 0 && !m[i - 1]) ||
          (col < w - 1 && !m[i + 1]) ||
          (i >= w && !m[i - w]) ||
          (i < lastRow && !m[i + w]);
      if (edge) {
        data[j] = 255;
        data[j + 1] = 230;
        data[j + 2] = 0;
      } else {
        data[j] = data[j] * 0.65 + 89;
        data[j + 1] = data[j + 1] * 0.65 + 77;
        data[j + 2] *= 0.65;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
const record = (i) => meta.records[i];
const outcomeAliases = {
  division: "division",
  no_division: "no division",
  not_assessed: "not assessed",
  lost: "loss",
  ghost_lysis: "loss",
};
function kind(r) {
  if ((r.decision === "not_grown" || r.decision === "no_division") && r.pi_status === "probably_dead") {
    return "probably dead (PI)";
  }
  return outcomes[r.decision] ?? outcomeAliases[r.decision];
}
function when(r) {
  return { division: r.division_frame, "probably dead (PI)": r.pi_surge_frame ?? r.pi_high_frame ?? r.event_frame ?? r.last_valid_frame, loss: r.last_valid_frame, exclusion: r.baseline_frame ?? 0 }[
    kind(r)
  ];
}
function eventRows() {
  return meta.records
    .map((r, i) => ({ r, i, t: when(r) }))
    .filter((e) => e.t != null)
    .sort((a, b) => a.t - b.t || a.i - b.i);
}
function anchor(i, assigned) {
  const p = assigned[i]?.[0];
  if (!p?.length) return meta.anchors[f][i];
  return [0, 1].map((k) => p.reduce((sum, pt) => sum + pt[k], 0) / p.length);
}
function overlays(id, assigned, unassigned, box) {
  const el = $(id).querySelector("svg"),
    whole = id === "overview";
  const { x, y, n } = whole ? { x: 0, y: 0, n: width } : box;
  const unit = n / Math.max(1, $(id).clientWidth);
  const lw = overlayPrefs.lineWidth;
  const fs = overlayPrefs.labelSize;
  el.replaceChildren();
  el.setAttribute("viewBox", [x, y, n, whole ? height : n].join(" "));
  el.setAttribute("preserveAspectRatio", "none");
  function polygon(p, color, width, dashed = false) {
    if (
      !whole &&
      !p.some(
        ([a, b]) =>
          a >= x - 32 && a <= x + n + 32 && b >= y - 32 && b <= y + n + 32,
      )
    )
      return;
    const attrs = {
      points: p.map((pt) => pt.join(",")).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": width * unit,
      "stroke-linejoin": "round",
    };
    if (dashed) attrs["stroke-dasharray"] = (4 * unit) + " " + (3 * unit);
    svg("polygon", attrs, el);
  }
  if ($("master").checked) {
    if (!whole && $("outlines").checked && unassigned.length) {
      const parts = [];
      for (const p of unassigned) {
        if (
          !p.length ||
          !p.some(
            ([a, b]) =>
              a >= x - 32 && a <= x + n + 32 && b >= y - 32 && b <= y + n + 32,
          )
        )
          continue;
        let seg = "M" + p[0][0] + "," + p[0][1];
        for (let k = 1; k < p.length; k++) seg += "L" + p[k][0] + "," + p[k][1];
        parts.push(seg + "Z");
      }
      if (parts.length)
        svg(
          "path",
          {
            d: parts.join(" "),
            fill: "none",
            stroke: "#bccbd166",
            "stroke-width": Math.max(0.7, lw * 0.55) * unit,
            "stroke-linejoin": "round",
          },
          el,
        );
    }
    for (const [idx, polys] of Object.entries(assigned)) {
      const i = Number(idx),
        r = record(i),
        color =
          $("colourby").value === "fate"
            ? fates[kind(r)]
            : r.decision === "excluded"
              ? tones.exclusion
              : `hsl(${(i * 137.508) % 360},83%,67%)`;
      if ($("outlines").checked)
        for (const p of polys)
          polygon(p, i === chosen ? "white" : color, i === chosen ? lw + 1.0 : lw, r.decision === "excluded");
      if ($("ids").checked) {
        const [a, b] = anchor(i, assigned);
        svg(
          "text",
          {
            x: a + 4 * unit,
            y: b - 3 * unit,
            fill: color,
            "font-size": fs * unit,
            "font-weight": "600",
            stroke: "#101517",
            "stroke-width": Math.max(2, fs * 0.22) * unit,
            "paint-order": "stroke",
          },
          el,
        ).textContent = r.founder_id;
      }
    }
    if ($("events").checked)
      for (const { r, i, t } of eventRows())
        if (t === f) {
          const [a, b] = anchor(i, assigned);
          svg(
            "circle",
            {
              cx: a,
              cy: b,
              r: (9 + lw * 1.5) * unit,
              fill: "none",
              stroke: tones[kind(r)],
              "stroke-width": (lw + 0.8) * unit,
              class: "pulse",
            },
            el,
          );
        }
  }
  if (whole)
    svg(
      "rect",
      {
        x: box.x,
        y: box.y,
        width: box.n,
        height: box.n,
        fill: "transparent",
        stroke: "#fff",
        "stroke-width": 1.6 * unit,
        "stroke-dasharray": 6 * unit + " " + 3 * unit,
      },
      el,
    );
}
function status() {
  if (!scene) return;
  const g = scene.box,
    texts = [];
  if (pointer) {
    const [id, px, py] = pointer,
      whole = id === "overview",
      tile = scene[id];
    const a = clamp(Math.floor(px * tile.w), 0, tile.w - 1),
      b = clamp(Math.floor(py * tile.h), 0, tile.h - 1);
    const x = whole ? Math.floor(px * width) : g.x + a,
      y = whole ? Math.floor(py * height) : g.y + b;
    texts.push(
      "x " + clamp(x, 0, width - 1) + "  y " + clamp(y, 0, height - 1),
      whole && (tile.w < width || tile.h < height)
        ? "downsampled mean · ADU"
        : "native ADU",
    );
    for (const ch of channels)
      texts.push(
        ch + " " + (tile.planes[ch]?.[b * tile.w + a] ?? (meta.channels[ch].includes(f) ? "off" : "not acquired")),
      );
    const i = hit(x, y);
    texts.push(i == null ? "Founder —" : "Founder " + record(i).founder_id);
  } else {
    texts.push("Move over either image for raw ADU");
    for (const ch of channels)
      if (!meta.channels[ch].includes(f)) texts.push(ch + " not acquired");
  }
  $("status").textContent = texts.join("  ·  ");
  $("transform").textContent =
    channels
      .filter((ch) => configs[ch].on)
      .map((ch) => {
        const s = configs[ch];
        return (
          ch +
          ": linear " +
          s.lo.toLocaleString() +
          "–" +
          s.hi.toLocaleString() +
          " ADU, " +
          (s.auto ? "auto every frame" : s.pinned ? "locked across pos" : "fixed over frames")
        );
      })
      .join("  |  ") +
      (document.body.classList.contains("smooth") ? "  |  Smoothed display" : "") ||
    "All channels off";
}
function hit(x, y, tolerance = 0) {
  if (!scene) return null;
  for (const [i, polys] of Object.entries(scene.assigned))
    for (const p of polys) {
      let inside = false;
      for (let a = 0, b = p.length - 1; a < p.length; b = a++) {
        const [ax, ay] = p[a],
          [bx, by] = p[b];
        if (tolerance) {
          const dx = bx - ax,
            dy = by - ay,
            t = clamp(
              ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1),
              0,
              1,
            );
          if (Math.hypot(x - ax - t * dx, y - ay - t * dy) <= tolerance)
            return Number(i);
        }
        if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax)
          inside = !inside;
      }
      if (inside) return Number(i);
    }
  return null;
}
async function render() {
  if (!meta) return;
  const token = ++epoch,
    box = geometry(),
    args = { f, mask: Number($("master").checked && $("mask").checked) },
    ovN = playing ? (overviewSlow ? 512 : 1024) : stillSize(),
    started = performance.now();
  const acquired = channels.filter((ch) => meta.channels[ch].includes(f));
  const wanted = acquired.filter((ch) => configs[ch].on);
  const chArg = wanted.join(",");
  const [ov, zm, assigned, other, hists] = await Promise.all([
    get("overview", { ...args, s: ovN, ch: chArg }, true),
    get("zoom", { ...args, ...box, ch: chArg }, true),
    get("a", { f }),
    $("master").checked && $("outlines").checked ? get("u", { f }) : [],
    Promise.all(
      channels.map((ch) =>
        (bc === ch || configs[ch].auto || !configs[ch].initialized) &&
        acquired.includes(ch)
          ? get("hist", { f, ch })
          : null,
      ),
    ),
  ]);
  if (token !== epoch) return;
  overviewSlow =
    performance.now() - started >
    700 / clamp(Number($("fps").value) || 5, 1, 30);
  channels.forEach((ch, i) => {
    const s = configs[ch];
    s.hist = hists[i];
    if (s.hist && (s.auto || (!s.initialized && !s.pinned))) {
      [s.lo, s.hi] = level(s.hist, s.pct);
      s.initialized = true;
    } else if (s.pinned) {
      s.initialized = true;
    }
  });
  scene = {
    overview: unpack(ov, ...overviewSize(ovN), wanted, args.mask),
    ovN,
    zoom: unpack(zm, box.n, box.n, wanted, args.mask),
    assigned,
    other,
    box,
  };
  paint("overview", scene.overview);
  paint("zoom", scene.zoom);
  overlays("overview", assigned, [], box);
  overlays("zoom", assigned, other, box);
  $("time").textContent = "Frame " + f + "/" + maxFrame();
  if (Number.isFinite(meta.times[f]) && Number.isFinite(meta.times[0])) {
    const mins = Math.round((meta.times[f] - meta.times[0]) / 60);
    $("time").textContent += " · " + Math.floor(mins / 60) + ":" + String(mins % 60).padStart(2, "0");
  }
  $("scale").textContent = box.n + " px";
  $("frame").value = f;
  document
    .querySelectorAll("#zooms button")
    .forEach((b) => b.classList.toggle("active", b.value == factor));
  status();
  if (bc) histogram();
  if (!document.body.dataset.ready) performance.mark("position-first-frame");
  document.body.dataset.ready = pos + "-" + f;
  prefetchAround(f, box, ovN, args.mask, token);
}
function prefetchAround(curF, box, ovN, maskFlag, token) {
  if (!meta) return;
  const mf = maxFrame(), curPos = pos;
  const targets = playing ? [curF + 1, curF + 2] : [curF + 1];
  setTimeout(() => {
    if (token !== epoch || pos !== curPos || !meta) return;
    const needU = $("master").checked && $("outlines").checked;
    for (const nf of targets) {
      if (nf < 0 || nf > mf) continue;
      const acq = channels.filter((ch) => meta.channels[ch].includes(nf));
      const chArg = acq.filter((ch) => configs[ch].on).join(",");
      get("overview", { f: nf, mask: maskFlag, s: ovN, ch: chArg }, true).catch(() => {});
      get("zoom", { f: nf, mask: maskFlag, ...box, ch: chArg }, true).catch(() => {});
      get("a", { f: nf }).catch(() => {});
      if (needU) get("u", { f: nf }).catch(() => {});
    }
  }, 15);
}
function repaintOverlaysOnly() {
  if (!scene) return;
  overlays("overview", scene.assigned, [], scene.box);
  overlays("zoom", scene.assigned, scene.other || [], scene.box);
}
let pixelRaf = 0;
function repaintPixelsOnly() {
  if (!scene || pixelRaf) return;
  pixelRaf = requestAnimationFrame(() => {
    pixelRaf = 0;
    if (!scene) return;
    paint("zoom", scene.zoom);
    paint("overview", scene.overview);
    status();
    if (bc) histogram();
  });
}
const redraw = () => render().catch(failure);
function stop() {
  playing = false;
  loopId++;
  $("play").textContent = "Play";
  if (scene && scene.ovN !== stillSize()) redraw();
}
async function jump(e) {
  stop();
  const m = meta;
  chosen = e.i;
  editing = false;
  f = e.t;
  const id = ++epoch,
    assigned = await get("a", { f });
  if (meta !== m || id !== epoch) return;
  [cx, cy] = anchor(e.i, assigned);
  card();
  await render();
}
async function advance(target, check = false) {
  target = clamp(target, 0, maxFrame());
  if (check && $("review").checked && $("stop").value !== "none") {
    const e = eventRows().find(
      (e) =>
        e.t === target && ($("stop").value === "every event" || !e.r.reviewed),
    );
    if (e) return jump(e);
  }
  f = target;
  await render();
}
async function play() {
  if (playing) {
    stop();
    return;
  }
  if (!meta) return;
  playing = true;
  const id = ++loopId;
  $("play").textContent = "Pause";
  while (playing && id === loopId) {
    const start = performance.now();
    const remaining =
      $("review").checked &&
      $("stop").value !== "none" &&
      chosen != null &&
      when(record(chosen)) === f &&
      eventRows().find(
        (e) =>
          e.t === f &&
          e.i > chosen &&
          ($("stop").value === "every event" || !e.r.reviewed),
      );
    if (remaining) {
      await jump(remaining);
      break;
    }
    await advance(f === maxFrame() ? 0 : f + 1, true);
    if (id !== loopId) break;
    if (f === maxFrame() || !playing) {
      stop();
      break;
    }
    await new Promise((done) =>
      setTimeout(
        done,
        Math.max(
          0,
          1000 / clamp(Number($("fps").value) || 5, 1, 30) -
            (performance.now() - start),
        ),
      ),
    );
  }
}
function navigate(dir) {
  const rows = eventRows();
  const here = rows.findIndex((e) => e.t === f && e.i === chosen);
  const next =
    here >= 0
      ? rows[here + dir]
      : dir > 0
        ? rows.find((e) => e.t > f)
        : rows
            .slice()
            .reverse()
            .find((e) => e.t < f);
  if (next) jump(next).catch(failure);
}
function fateKey() {
  const key = $("fatekey");
  key.hidden = $("colourby").value !== "fate";
  key.replaceChildren(
    ...Object.entries(fates).map(([name, c]) => {
      const s = document.createElement("span");
      s.className = "fate";
      s.style.setProperty("--tone", c);
      s.textContent = name;
      return s;
    }),
  );
}
function sidebar() {
  const list = $("list"),
    ticks = $("ticks");
  list.replaceChildren();
  ticks.replaceChildren();
  const rows = eventRows();
  for (const e of rows) {
    const { r, i, t } = e,
      type = kind(r),
      color = tones[type],
      y = { division: 2, "probably dead (PI)": 4, loss: 6, exclusion: 8.5 }[type] ?? 5;
    svg(
      "rect",
      {
        x: t + 0.12,
        y: y - 1.6,
        width: 0.8,
        height: 3.2,
        fill: r.reviewed ? color : "none",
        stroke: color,
        "stroke-width": 0.24,
      },
      ticks,
    );
    if (
      ($("type").selectedIndex && $("type").value !== type) ||
      ($("checked").selectedIndex &&
        ($("checked").value === "reviewed") !== r.reviewed)
    )
      continue;
    const b = button("", () => jump(e).catch(failure), list);
    b.className = "event";
    const dot = document.createElement("i");
    dot.className = "dot";
    dot.style.setProperty("--tone", color);
    dot.style.setProperty("--fill", r.reviewed ? color : "transparent");
    const end = document.createElement("span");
    end.textContent = (r.reviewed ? "Reviewed" : "Unreviewed") + " · f " + t;
    b.title = r.reviewed ? "Reviewed" : "Unreviewed";
    const tag = r.is_doublet
      ? " [doublet]"
      : type === "exclusion" && (r.excluded_reason || r.exclude_reason)
        ? " (" + (r.excluded_reason || r.exclude_reason) + ")"
        : "";
    b.append(dot, type + " · #" + r.founder_id + tag, end);
  }
  $("count").textContent = list.children.length + " / " + rows.length;
  card();
}
function card() {
  const el = $("card");
  el.replaceChildren();
  if (chosen == null) {
    el.textContent = "Select a founder or event to review.";
    return;
  }
  const r = record(chosen),
    type = kind(r),
    reason =
      r.reason ??
      (type === "exclusion"
        ? r.exclude_reason || r.excluded_reason
        : type === "loss"
          ? r.tracking_end_reason
          : "");
  function line(tag, text) {
    const p = document.createElement(tag);
    p.textContent = text;
    el.append(p);
  }
  line("strong", "Founder " + r.founder_id + (r.is_doublet ? " (frame-0 doublet)" : ""));
  line(
    "p",
    (type === "exclusion" ? "Excluded" : r.is_doublet ? "Included (doublet pair)" : "Included") +
      " · " +
      (r.reviewed ? "reviewed" : "unreviewed"),
  );
  line(
    "p",
    type +
      (type === "division"
        ? " · frames " + r.last_single_frame + " → " + r.division_frame
        : when(r) != null
          ? " · frame " + when(r)
          : ""),
  );
  if (reason) line("p", (type === "exclusion" ? "Exclusion reason: " : "Reason: ") + reason);
  if (r.note) line("p", r.note);
  if (r.confidence) line("p", "Confidence · " + r.confidence);
  if (!editing) {
    button("Agree ✓", agree, el);
    button(
      "Edit",
      () => {
        editing = true;
        card();
      },
      el,
    );
    return;
  }
  const select = document.createElement("select");
  options(select, Object.values(outcomes));
  select.value = type;
  function field(label, input) {
    const l = document.createElement("label");
    l.append(label + " ", input);
    el.append(l);
  }
  field("Outcome", select);
  const reasonBox = document.createElement("input"),
    conf = document.createElement("input"),
    at = document.createElement("input");
  reasonBox.value = reason || "";
  conf.value = r.confidence || "";
  at.type = "checkbox";
  field("Reason", reasonBox);
  field("Confidence", conf);
  field("Event is at this frame", at);
  button(
    "Apply edit",
    () => {
      const v = select.value,
        t = at.checked || type !== v ? f : when(r);
      r.decision = Object.keys(outcomes).find((k) => outcomes[k] === v);
      if (v === "division") {
        r.division_frame = t;
        r.last_single_frame =
          t === 0 ? null : Math.min(r.last_single_frame ?? t - 1, t - 1);
      }
      if (v === "loss") r.last_valid_frame = t;
      r.reason = reasonBox.value;
      r.confidence = conf.value;
      r.reviewed = true;
      editing = false;
      sidebar();
      redraw();
    },
    el,
  );
  button(
    "Cancel",
    () => {
      editing = false;
      card();
    },
    el,
  );
}
function agree() {
  if (!$("review").checked || chosen == null) return;
  record(chosen).reviewed = true;
  editing = false;
  sidebar();
  redraw();
}

function histX(val, min, max, width) {
  const plotW = width - HIST_PAD_L - HIST_PAD_R;
  return HIST_PAD_L + ((val - min) / Math.max(1, max - min)) * plotW;
}
function histVal(px, min, max, width) {
  const plotW = width - HIST_PAD_L - HIST_PAD_R;
  const frac = clamp((px - HIST_PAD_L) / plotW, 0, 1);
  return Math.round(min + frac * (max - min));
}

function histogram() {
  const s = configs[bc],
    h = s.hist,
    c = $("hist"),
    ctx = c.getContext("2d");
  const W = c.width,
    H = c.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#bfccd3";
  ctx.font = "11px sans-serif";
  $("bcname").textContent = bc + " · Brightness & Contrast";
  $("perframe").checked = s.auto;
  $("bclock").textContent = s.auto ? "auto/frame" : s.pinned ? "locked across pos" : "auto init";
  for (const key of ["lo", "hi"]) {
    $(key + "num").value = s[key];
  }
  if (!h) {
    ctx.fillText("Not acquired at this frame", 12, 74);
    return;
  }
  const [v, counts] = h;
  const min = histState.lockedDomain ? histState.lockedDomain[0] : Math.min(s.lo, v[0]);
  const max = histState.lockedDomain ? histState.lockedDomain[1] : Math.max(s.hi, v[v.length - 1]);
  histState.min = min;
  histState.max = max;

  const plotW = W - HIST_PAD_L - HIST_PAD_R;
  const plotH = HIST_BOT - HIST_TOP;
  const nBins = 220;
  const bins = new Float64Array(nBins);
  v.forEach(
    (a, i) =>
      (bins[clamp(Math.floor(((a - min) / Math.max(1, max - min)) * (nBins - 1)), 0, nBins - 1)] +=
        counts[i]),
  );
  const peak = Math.max(1, ...bins);

  const xLo = clamp(histX(s.lo, min, max, W), HIST_PAD_L, W - HIST_PAD_R);
  const xHi = clamp(histX(s.hi, min, max, W), HIST_PAD_L, W - HIST_PAD_R);

  // Shaded active window between lo and hi
  ctx.fillStyle = "rgba(137, 222, 203, 0.14)";
  ctx.fillRect(xLo, HIST_TOP, Math.max(1, xHi - xLo), plotH);

  // Histogram bars
  ctx.fillStyle = "#8aa0ab";
  for (let i = 0; i < nBins; i++) {
    const bh = (plotH - 4) * (bins[i] / peak);
    ctx.fillRect(HIST_PAD_L + (i * plotW) / nBins, HIST_BOT - bh, plotW / nBins + 0.4, bh);
  }

  // LUT transfer curve
  ctx.strokeStyle = "#8ce8cf";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(HIST_PAD_L, HIST_BOT);
  ctx.lineTo(xLo, HIST_BOT);
  ctx.lineTo(xHi, HIST_TOP);
  ctx.lineTo(W - HIST_PAD_R, HIST_TOP);
  ctx.stroke();

  // Interactive Min & Max handle lines + grabber thumbs on histogram
  function drawHandle(x, color, label, isLo) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, HIST_TOP - 4);
    ctx.lineTo(x, HIST_BOT + 4);
    ctx.stroke();

    // Top thumb pill
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x - 6, HIST_TOP - 14, 12, 12, 3);
    ctx.fill();
    ctx.fillStyle = "#10151a";
    ctx.fillRect(x - 1, HIST_TOP - 11, 2, 6);

    // Bottom handle triangle
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x, HIST_BOT + 2);
    ctx.lineTo(x - 5, HIST_BOT + 10);
    ctx.lineTo(x + 5, HIST_BOT + 10);
    ctx.closePath();
    ctx.fill();

    // Handle value readout
    ctx.fillStyle = color;
    ctx.font = "bold 10px ui-monospace, Consolas, monospace";
    ctx.textAlign = isLo ? (x < HIST_PAD_L + 36 ? "left" : "right") : (x > W - HIST_PAD_R - 36 ? "right" : "left");
    ctx.fillText(label, isLo ? (x < HIST_PAD_L + 36 ? x + 4 : x - 4) : (x > W - HIST_PAD_R - 36 ? x - 4 : x + 4), HIST_BOT + 24);
  }
  drawHandle(xLo, "#38bdf8", String(s.lo), true);
  drawHandle(xHi, "#f59e0b", String(s.hi), false);

  // Axis labels
  ctx.fillStyle = "#9ab0bc";
  ctx.font = "10.5px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Pixels", 3, 14);
  ctx.fillText(peak.toLocaleString(), 3, 28);
  ctx.fillText(min.toLocaleString(), HIST_PAD_L, H - 5);
  ctx.textAlign = "center";
  ctx.fillText(Math.round((min + max) / 2).toLocaleString() + " ADU", HIST_PAD_L + plotW / 2, H - 5);
  ctx.textAlign = "right";
  ctx.fillText(max.toLocaleString(), W - HIST_PAD_R, H - 5);
  ctx.textAlign = "left";
}

function setupHistogramInteractions() {
  const c = $("hist");
  const getCanvasX = (e) => {
    const r = c.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * c.width;
  };
  const pickMode = (px) => {
    const s = configs[bc];
    if (!s || !s.hist) return null;
    const xLo = histX(s.lo, histState.min, histState.max, c.width);
    const xHi = histX(s.hi, histState.min, histState.max, c.width);
    const dLo = Math.abs(px - xLo);
    const dHi = Math.abs(px - xHi);
    if (dLo <= 10 && dLo <= dHi) return "lo";
    if (dHi <= 10) return "hi";
    if (px > xLo && px < xHi) return "mid";
    return dLo < dHi ? "lo" : "hi";
  };
  c.onpointerdown = (e) => {
    if (!bc || !configs[bc]?.hist) return;
    const px = getCanvasX(e);
    const mode = pickMode(px);
    if (!mode) return;
    c.setPointerCapture(e.pointerId);
    const s = configs[bc];
    histState.dragMode = mode;
    histState.startX = px;
    histState.startLo = s.lo;
    histState.startHi = s.hi;
    histState.lockedDomain = [histState.min, histState.max];
    if (mode === "lo" || mode === "hi") {
      const val = histVal(px, histState.min, histState.max, c.width);
      if (mode === "lo") manual(clamp(val, 0, s.hi - 1), s.hi, true);
      else manual(s.lo, clamp(val, s.lo + 1, 65535), true);
    }
  };
  c.onpointermove = (e) => {
    if (!bc || !configs[bc]?.hist) return;
    const px = getCanvasX(e);
    if (!histState.dragMode) {
      const m = pickMode(px);
      c.style.cursor = m === "mid" ? "grab" : "ew-resize";
      return;
    }
    c.style.cursor = histState.dragMode === "mid" ? "grabbing" : "ew-resize";
    const s = configs[bc];
    const [dMin, dMax] = histState.lockedDomain;
    if (histState.dragMode === "lo") {
      const val = histVal(px, dMin, dMax, c.width);
      manual(clamp(val, 0, s.hi - 1), s.hi, true);
    } else if (histState.dragMode === "hi") {
      const val = histVal(px, dMin, dMax, c.width);
      manual(s.lo, clamp(val, s.lo + 1, 65535), true);
    } else if (histState.dragMode === "mid") {
      const plotW = c.width - HIST_PAD_L - HIST_PAD_R;
      const dVal = Math.round(((px - histState.startX) / plotW) * (dMax - dMin));
      const span = histState.startHi - histState.startLo;
      const newLo = clamp(histState.startLo + dVal, 0, 65535 - span);
      manual(newLo, newLo + span, true);
    }
  };
  const endDrag = () => {
    if (!histState.dragMode) return;
    histState.dragMode = null;
    histState.lockedDomain = null;
    saveChannelPrefs();
    histogram();
  };
  c.onpointerup = endDrag;
  c.onpointercancel = endDrag;
}

function manual(lo, hi, fast = false) {
  lo = Math.round(lo);
  hi = Math.round(hi);
  if (
    !Number.isInteger(lo) ||
    !Number.isInteger(hi) ||
    lo < 0 ||
    hi > 65535 ||
    lo >= hi
  )
    return;
  Object.assign(configs[bc], {
    lo,
    hi,
    auto: false,
    pinned: true,
    pct: 0.35,
    initialized: true,
  });
  if (fast && scene) {
    repaintPixelsOnly();
  } else {
    saveChannelPrefs();
    redraw();
  }
}

function channelControls() {
  $("channels").replaceChildren();
  channels.forEach((ch, i) => {
    const sv = channelPrefs[ch] || {};
    const s = (configs[ch] = {
      on: sv.on !== undefined ? Boolean(sv.on) : i === 0,
      lut: palettes.includes(sv.lut) ? sv.lut : palettes[i % palettes.length],
      lo: Number.isInteger(sv.lo) ? clamp(sv.lo, 0, 65534) : 0,
      hi: Number.isInteger(sv.hi) ? clamp(sv.hi, 1, 65535) : 65535,
      pct: 0.35,
      auto: Boolean(sv.auto),
      pinned: Boolean(sv.pinned) && Number.isInteger(sv.lo) && Number.isInteger(sv.hi) && sv.lo < sv.hi,
      initialized: Boolean(sv.pinned),
    });
    const group = document.createElement("fieldset"),
      toggle = document.createElement("input"),
      sel = document.createElement("select"),
      label = document.createElement("label");
    toggle.type = "checkbox";
    toggle.checked = s.on;
    toggle.onchange = () => {
      s.on = toggle.checked;
      saveChannelPrefs();
      if (!s.on || (scene && scene.overview.planes[ch] && scene.zoom.planes[ch]))
        repaintPixelsOnly();
      else redraw();
    };
    label.append(toggle, ch + " ");
    group.append(label);
    sel.ariaLabel = ch + " LUT";
    options(sel, palettes);
    sel.value = s.lut;
    sel.onchange = () => {
      s.lut = sel.value;
      saveChannelPrefs();
      if (scene) repaintPixelsOnly();
      else redraw();
    };
    group.append(sel);
    button(
      "B&C",
      () => {
        bc = ch;
        $("bc").hidden = false;
        redraw();
      },
      group,
    );
    $("channels").append(group);
  });
}

function controls() {
  loadSavedPrefs();
  // Compact Min / Max numeric inputs under histogram (synced with on-histogram sliders)
  for (const key of ["lo", "hi"]) {
    const label = document.createElement("label");
    label.append(key === "lo" ? "Min" : "Max");
    const input = document.createElement("input");
    input.type = "number";
    input.min = 0;
    input.max = 65535;
    input.id = key + "num";
    input.onchange = () => manual(Number($("lonum").value), Number($("hinum").value));
    label.append(input);
    $("ranges").append(label);
  }
  setupHistogramInteractions();

  $("auto").onclick = () => {
    const s = configs[bc];
    if (!s.hist) return;
    const pct = s.nextPct || 0.35;
    const [lo, hi] = level(s.hist, pct);
    s.nextPct = Math.min(80, pct * 2);
    s.pct = pct;
    manual(lo, hi);
  };
  $("reset").onclick = () => {
    const s = configs[bc];
    if (!s.hist) return;
    s.nextPct = 0.35;
    manual(...level(s.hist, 0));
  };
  $("set").onclick = () =>
    manual(Number($("lonum").value), Number($("hinum").value));
  $("close").onclick = () => {
    bc = null;
    $("bc").hidden = true;
  };
  $("perframe").onchange = (e) => {
    configs[bc].auto = e.target.checked;
    if (e.target.checked) configs[bc].pinned = false;
    saveChannelPrefs();
    redraw();
  };

  // Outline width & label size controls
  $("lineWidth").value = overlayPrefs.lineWidth;
  $("lineWidthVal").textContent = overlayPrefs.lineWidth.toFixed(1);
  $("lineWidth").oninput = () => {
    overlayPrefs.lineWidth = Number($("lineWidth").value);
    $("lineWidthVal").textContent = overlayPrefs.lineWidth.toFixed(1);
    saveOverlayPrefs();
    repaintOverlaysOnly();
  };
  $("labelSize").value = overlayPrefs.labelSize;
  $("labelSizeVal").textContent = String(overlayPrefs.labelSize);
  $("labelSize").oninput = () => {
    overlayPrefs.labelSize = Number($("labelSize").value);
    $("labelSizeVal").textContent = String(overlayPrefs.labelSize);
    saveOverlayPrefs();
    repaintOverlaysOnly();
  };

  for (const n of [1, 2, 4, 8])
    button(n + "×", () => zoom(n), $("zooms")).value = n;
  $("review").onchange = () => {
    document.body.classList.toggle("raw", !$("review").checked);
    if (!$("review").checked) {
      chosen = null;
      editing = false;
      card();
    }
    resize();
  };
  $("smooth").onchange = (e) => {
    document.body.classList.toggle("smooth", e.target.checked);
    status();
  };
  for (const id of ["master", "outlines", "ids", "events", "mask"])
    $(id).onchange = redraw;
  $("colourby").onchange = () => {
    fateKey();
    redraw();
  };
  fateKey();
  for (const id of ["type", "checked"]) $(id).onchange = sidebar;
  $("play").onclick = () => play().catch(failure);
  $("prev").onclick = () => {
    stop();
    advance(f - 1).catch(failure);
  };
  $("next").onclick = () => {
    stop();
    advance(f + 1).catch(failure);
  };
  $("frame").oninput = () => {
    stop();
    advance(Number($("frame").value)).catch(failure);
  };
  document.onkeydown = (e) => {
    if (e.target.matches("input,select,textarea") || e.metaKey || e.ctrlKey)
      return;
    const actions = {
      " ": () => play().catch(failure),
      ArrowLeft: () => {
        stop();
        advance(f - 1).catch(failure);
      },
      ArrowRight: () => {
        stop();
        advance(f + 1).catch(failure);
      },
      "[": () => navigate(-1),
      "]": () => navigate(1),
      "+": () => zoom(factor * 2),
      "=": () => zoom(factor * 2),
      "-": () => zoom(factor / 2),
      a: agree,
      A: agree,
      o: () => {
        $("master").checked = !$("master").checked;
        redraw();
      },
      O: () => {
        $("master").checked = !$("master").checked;
        redraw();
      },
      s: () => {
        $("smooth").checked = !$("smooth").checked;
        document.body.classList.toggle("smooth", $("smooth").checked);
        status();
      },
      S: () => {
        $("smooth").checked = !$("smooth").checked;
        document.body.classList.toggle("smooth", $("smooth").checked);
        status();
      },
      Escape: () => {
        bc = null;
        $("bc").hidden = true;
      },
    };
    if (actions[e.key]) {
      e.preventDefault();
      actions[e.key]();
    }
  };
}
function zoom(next) {
  factor = clamp(next, 1, 8);
  redraw();
}
function interactions(id) {
  const el = $(id),
    whole = id === "overview";
  const coords = (e) => {
    const r = el.getBoundingClientRect();
    return [
      clamp((e.clientX - r.left) / r.width, 0, 1),
      clamp((e.clientY - r.top) / r.height, 0, 1),
    ];
  };
  el.onpointerdown = (e) => {
    if (!scene || e.button) return;
    el.setPointerCapture(e.pointerId);
    const [u, v] = coords(e),
      g = scene.box;
    const within =
      u * width >= g.x &&
      u * width <= g.x + g.n &&
      v * height >= g.y &&
      v * height <= g.y + g.n;
    drag = { u, v, x: g.x + g.n / 2, y: g.y + g.n / 2, n: g.n, moved: false };
    if (whole && !within) {
      drag.x = cx = u * width;
      drag.y = cy = v * height;
      redraw();
    }
  };
  el.onpointermove = (e) => {
    const [u, v] = coords(e);
    pointer = [id, u, v];
    if (drag) {
      const dx = u - drag.u,
        dy = v - drag.v;
      if ((Math.abs(dx) + Math.abs(dy)) * el.clientWidth > 2) drag.moved = true;
      if (drag.moved) {
        cx = clamp(drag.x + dx * (whole ? width : -drag.n), drag.n / 2, width - drag.n / 2);
        cy = clamp(drag.y + dy * (whole ? height : -drag.n), drag.n / 2, height - drag.n / 2);
        redraw();
      }
    }
    status();
  };
  el.onpointerup = (e) => {
    if (drag && !drag.moved) {
      const [u, v] = coords(e);
      if (whole) {
        cx = u * width;
        cy = v * height;
        redraw();
      } else if ($("review").checked && $("master").checked) {
        const g = scene.box,
          i = hit(g.x + u * g.n, g.y + v * g.n, 2 / factor);
        if (i != null) {
          chosen = i;
          editing = false;
          card();
          repaintOverlaysOnly();
        }
      }
    }
    drag = null;
  };
  el.onpointercancel = () => {
    drag = null;
  };
  el.onpointerleave = () => {
    pointer = null;
    status();
  };
  if (!whole)
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        zoom(factor * (e.deltaY < 0 ? 2 : 0.5));
      },
      { passive: false },
    );
}
function sizeFields() {
  for (const id of ["overview", "zoom"]) {
    const el = $(id),
      parent = el.parentElement;
    const aspect = id === "overview" ? width / height : 1;
    const paneWidth = Math.min(parent.clientHeight * aspect, parent.clientWidth);
    el.style.width = Math.max(1, Math.floor(paneWidth)) + "px";
    el.style.height = Math.max(1, Math.floor(paneWidth / aspect)) + "px";
  }
}
function resize() {
  sizeFields();
  redraw();
}
async function selectPosition() {
  stop();
  epoch++;
  pos = $("pos").value;
  const requested = pos;
  meta = null;
  scene = null;
  chosen = null;
  editing = false;
  pointer = null;
  $("status").textContent = "Loading Pos" + pos + "…";
  const m = positions.get(pos) || (await get("meta"));
  if (pos !== requested) return;
  positions.set(pos, m);
  meta = m;
  width = meta.width ?? 2048;
  height = meta.height ?? 2048;
  $("dimensions").textContent = width + " × " + height;
  const names = Object.keys(meta.channels);
  const structure = meta.structure_channel ?? names.find((ch) => ch.toLowerCase() === "phase");
  channels = names.includes(structure) ? [structure, ...names.filter((ch) => ch !== structure)] : names;
  bc = null;
  $("bc").hidden = true;
  for (const ch of Object.keys(configs)) delete configs[ch];
  channelControls();
  f = 0;
  const mf = maxFrame();
  $("frame").max = mf;
  $("ticks").setAttribute("viewBox", `0 0 ${mf + 1} 10`);
  cx = width / 2;
  cy = height / 2;
  factor = 1;
  sizeFields();
  sidebar();
  await render();
}
async function init() {
  controls();
  interactions("overview");
  interactions("zoom");
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe($("zoom").parentElement);
  const list = await get("list");
  for (const p of list)
    $("pos").add(new Option(p.label || ("Pos" + p.pos + " · " + p.strain), p.pos));
  if (!list.length) throw Error("position cache unavailable");
  $("pos").onchange = () => selectPosition().catch(failure);
  await selectPosition();
}
init().catch(failure);

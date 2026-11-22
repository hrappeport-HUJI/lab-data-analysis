"""Build a data-viewer page from a table: one HTML file you can double-click on any computer.

    from build_viewer import write_viewer
    write_viewer(df, spec, "out.html")                       # defaults: data embedded, no chat, Plotly from CDN
    write_viewer(df, spec, "out.html", data="file")          # data in out.data.js next to the page
    write_viewer(df, spec, "out.html", plotly="inline")      # works offline too (+3.5 MB)

Command line (no Python code needed):
    python build_viewer.py build table.csv spec.json out.html [--data auto|embed|file] [--plotly cdn|inline]
    python build_viewer.py dump page.html table.csv          # get the table back out of a page

The spec says how to show the table; every key but `columns` is optional:

    spec = {
      "title": "Short name", "subtitle": "one line",
      "notes": "definitions shown on the page and given to agents and the chat",
      "row_meaning": "what one row is",
      "id": "key",                         # unique row id column (default: row number)
      "label": "P{pos} · founder {num}",   # hover / card label, {column} placeholders
      "color_by": "strain",                # default colour column (a category column)
      "split_by": "day",                   # optional second grouping: each colour group split by this category
      "split_style": "shade",              # how the split groups differ: shade (of the colour), style (marker/dash/pattern), color
      "filters": ["strain", "group"],      # category columns offered as checkbox filters
      "hover": ["group"],                  # extra columns in the hover text
      "hollow_when": {"column": "lag_source", "values": ["not divided"]},   # drawn as hollow markers
      "ring_when":   {"column": "lag_source", "values": ["estimated"]},     # drawn with a ring
      "views": [{"name": "...", "x": "a", "y": "b", "x_scale": "log", "color_by": "strain", "split_by": "day", "split_style": "style",
                 "filters": {"strain": ["KLY"]}}],
      "features": {                        # optional modules, see SKILL.md; leave out for none
        "links": [{"name": "Open in review viewer", "url": "http://host:8891/?founder={id}"}],
        "chat": {"placeholder": "an example question"},
        "cdf": {"column": "a", "scale": "log", "x_scale": "linear", "title": "..."},
      },
      "columns": {
        "a":      {"label": "A (unit)", "kind": "number", "scale": "log", "zone": 10, "short": "A",
                   "description": "...", "never_outlier": False, "missing": "what a blank means"},
        "strain": {"label": "Strain", "kind": "category", "order": [...], "colors": {"KLY": "#1f6fd1"}},
      },
    }
Columns not in spec["columns"] are left out. Numbers are rounded to 4 significant digits.
"""
import argparse
import copy
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
MODULES = WEB / "modules"
PLOTLY_VERSION = "2.27.0"
PLOTLY_URL = f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{PLOTLY_VERSION}/plotly.min.js"
EMBED_LIMIT_MB = 5
SPLIT_STYLES = ("shade", "style", "color")  # how groups split by a second category differ  # data="auto": embed below this, separate .data.js file above


def available_modules():
    return sorted(p.stem for p in MODULES.glob("*.js"))


# ---------------------------------------------------------------- spec + data
def check_spec(df, spec):
    """Raise ValueError naming every problem, so a bad spec fails at build time, not in the browser."""
    cols, errs = spec.get("columns") or {}, []
    if not cols:
        errs.append("spec['columns'] is empty")
    for name, meta in cols.items():
        if name not in df.columns:
            errs.append(f"column {name!r} is not in the table")
        if meta.get("kind", "number") not in ("number", "category"):
            errs.append(f"column {name!r}: kind must be 'number' or 'category'")
    cats = {k for k, m in cols.items() if m.get("kind") == "category"}
    nums = set(cols) - cats
    if spec.get("id") and spec["id"] not in df.columns:
        errs.append(f"id column {spec['id']!r} is not in the table")
    elif spec.get("id") and df[spec["id"]].duplicated().any():
        errs.append(f"id column {spec['id']!r} has duplicate values")
    if not nums:
        errs.append("at least one numeric column is needed for the axes")
    for k in spec.get("filters", []) + ([spec["color_by"]] if spec.get("color_by") else []):
        if k not in cats:
            errs.append(f"{k!r} is used as a filter or colour but is not a category column")
    if spec.get("split_by") and spec["split_by"] not in cats:
        errs.append(f"split_by {spec['split_by']!r} is not a category column")
    if spec.get("split_style", "shade") not in SPLIT_STYLES:
        errs.append(f"split_style must be one of {', '.join(SPLIT_STYLES)}")
    for rule in ("hollow_when", "ring_when"):
        if spec.get(rule) and spec[rule].get("column") not in cols:
            errs.append(f"{rule}: column {spec[rule].get('column')!r} is not in spec['columns']")
    for v in spec.get("views", []):
        for axis in ("x", "y"):
            if v.get(axis) and v[axis] not in nums:
                errs.append(f"view {v.get('name')!r}: {axis} {v[axis]!r} is not a numeric column")
        if v.get("color_by") and v["color_by"] not in cats:
            errs.append(f"view {v.get('name')!r}: color_by {v['color_by']!r} is not a category column")
        if v.get("split_by") and v["split_by"] not in cats:
            errs.append(f"view {v.get('name')!r}: split_by {v['split_by']!r} is not a category column")
        if v.get("split_style", "shade") not in SPLIT_STYLES:
            errs.append(f"view {v.get('name')!r}: split_style must be one of {', '.join(SPLIT_STYLES)}")
        for c in (v.get("filters") or {}):
            if c not in spec.get("filters", []):
                errs.append(f"view {v.get('name')!r}: filter {c!r} is not in spec['filters']")
    for name in (spec.get("features") or {}):
        if name not in available_modules():
            errs.append(f"feature {name!r} does not exist; available: {', '.join(available_modules())}")
    ts = (spec.get("features") or {}).get("timeseries")
    if ts is not None:
        series, channels = ts.get("series") or {}, ts.get("channels") or {}
        for k, pnl in enumerate(ts.get("panels") or []):
            name = pnl.get("name", k)
            if pnl.get("series_col") not in cats:
                errs.append(f"timeseries panel {name!r}: series_col {pnl.get('series_col')!r} is not a category column")
            elif pnl["series_col"] in df.columns:
                missing = sorted({str(v) for v in df[pnl["series_col"]].dropna()} - set(series))
                if missing:
                    errs.append(f"timeseries panel {name!r}: no series for {', '.join(missing[:5])}")
            for e, col in (pnl.get("events") or {}).items():
                if col not in nums:
                    errs.append(f"timeseries panel {name!r}: event {e!r} column {col!r} is not a numeric column")
            for ch in pnl.get("channels") or []:
                if ch not in channels:
                    errs.append(f"timeseries panel {name!r}: channel {ch!r} is not in 'channels'")
            if pnl.get("x", "absolute") not in ["absolute", *(pnl.get("events") or {})]:
                errs.append(f"timeseries panel {name!r}: x {pnl.get('x')!r} is neither 'absolute' nor an event")
        for sid, chans in series.items():
            for ch, s in chans.items():
                if len(s.get("t", [])) != len(s.get("v", [])):
                    errs.append(f"timeseries series {sid!r} channel {ch!r}: t and v differ in length")
    cdf = (spec.get("features") or {}).get("cdf")
    if cdf is not None:
        for c in [cdf.get("column")] * bool(cdf.get("column")) + list(cdf.get("columns") or []):
            if c not in nums:
                errs.append(f"cdf: column {c!r} is not a numeric column")
        if cdf.get("scale", "log") not in ("log", "linear"):
            errs.append(f"cdf: scale must be 'log' or 'linear', not {cdf.get('scale')!r}")
        if cdf.get("x_scale", "linear") not in ("log", "linear"):
            errs.append(f"cdf: x_scale must be 'log' or 'linear', not {cdf.get('x_scale')!r}")
    if errs:
        raise ValueError("data-viewer spec problems:\n  " + "\n  ".join(errs))


def _num(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return float(f"{float(v):.4g}")


def bundle(df, spec):
    """The page's data: {spec, ids, data: {column: [values]}, n}. Column-wise keeps it small and fast."""
    check_spec(df, spec)
    spec = copy.deepcopy(spec)
    data = {}
    for name, meta in spec["columns"].items():
        s = df[name]
        if meta.get("kind", "number") == "number":
            data[name] = [_num(v) for v in pd.to_numeric(s, errors="coerce")]
        else:
            data[name] = [None if pd.isna(v) else str(v) for v in s]
            seen = {v for v in data[name] if v is not None}
            meta["order"] = [str(v) for v in meta.get("order", [])] + sorted(seen - {str(v) for v in meta.get("order", [])})
    ids = [str(v) for v in df[spec["id"]]] if spec.get("id") else [str(i) for i in range(len(df))]
    return {"spec": spec, "ids": ids, "data": data, "n": len(df)}


# ---------------------------------------------------------------- agent guide
def agent_guide(b, data_note):
    """Plain-text instructions at the top of the page for an agent that opens or reads the file."""
    s, cols = b["spec"], b["spec"]["columns"]
    lines = [
        "AGENT GUIDE: interactive data viewer (built by the data-viewer skill)",
        f"Title: {s.get('title', '')}. {s.get('subtitle', '')}".strip(),
        f"Rows: {b['n']}, each one {s.get('row_meaning', 'record')}. Row id column: {s.get('id') or '(row number)'}.",
        f"Data: {data_note}",
        "  Shape: {spec, ids: [row ids], data: {column: [one value per row]}, n}. null = missing.",
        "  From Python: build_viewer.read_viewer_data(path) returns (DataFrame, spec).",
        "Columns:",
    ]
    for k, m in cols.items():
        desc = f" [{', '.join(m['order'])}]" if m.get("kind") == "category" else ""
        extra = "; ".join(x for x in (m.get("description"), f"blank means {m['missing']}" if m.get("missing") else None) if x)
        lines.append(f"  {k}: {m.get('label', k)} ({m.get('kind', 'number')}){desc}{' - ' + extra if extra else ''}")
    if s.get("notes"):
        lines += ["Definitions:", "  " + s["notes"]]
    if s.get("views"):
        lines.append("Preset views: " + "; ".join(f"{v['name']} ({v.get('y')} vs {v.get('x')})" for v in s["views"]))
    lines += [
        "In a browser, the page exposes window.viewer with the same tools a chat model would use:",
        "  describe_data(), query_rows({where, columns, sort_by, descending, limit}), summarize({group_by, columns}),",
        "  correlate({x, y, group_by}), compare_groups({column, group_by}), set_view({x, y, x_scale, filters, ...}),",
        "  remove_points({ids | where, restore}). window.viewer.tools lists their input schemas.",
        "  where = [{column, op, value}], op one of == != < <= > >= in 'not in' 'is null' 'not null'.",
        f"Features in this page: {', '.join(s.get('features') or {}) or 'none'}.",
    ]
    return "<!--\n" + "\n".join(lines).replace("-->", "- ->") + "\n-->"


# ---------------------------------------------------------------- page
def _plotly_tag(plotly, cache_dir=None):
    if plotly == "cdn":
        return f'<script src="{PLOTLY_URL}"></script>'
    path = Path(plotly) if plotly not in ("inline",) else Path(cache_dir or Path.home() / ".cache" / "data-viewer") / f"plotly-{PLOTLY_VERSION}.min.js"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PLOTLY_URL, timeout=60) as r:
            path.write_bytes(r.read())
    return "<script>" + path.read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>"


def write_viewer(df, spec, out_html, data="auto", plotly="cdn", embed_limit_mb=EMBED_LIMIT_MB, plotly_cache=None):
    """Write the viewer page (and out.data.js when the data goes in a separate file).

    data:   "embed" (one file), "file" (out.data.js beside the page) or "auto" (embed below embed_limit_mb).
    plotly: "cdn" (needs internet), "inline" (downloaded once, then cached) or a path to plotly.min.js.
    Returns {"html": path, "data_file": path or None, "html_bytes": n, "data_bytes": n}.
    """
    out_html = Path(out_html)
    b = bundle(df, spec)
    text = json.dumps(b, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if data == "auto":
        data = "embed" if len(text.encode("utf-8")) < embed_limit_mb * 2**20 else "file"
    if data not in ("embed", "file"):
        raise ValueError("data must be 'auto', 'embed' or 'file'")
    data_file = out_html.with_name(out_html.stem + ".data.js") if data == "file" else None
    if data == "embed":
        data_tag = '<script id="viewer-data" type="application/json">' + text.replace("<", "\\u003c") + "</script>"  # no "</script" or "<!--" inside
        note = 'embedded in this file as JSON, in the script element with id "viewer-data" (type application/json).'
    else:
        data_tag = f'<script src="{data_file.name}"></script>'
        note = f"in {data_file.name} next to this file, as window.VIEWER_BUNDLE = {{...}}; (keep the two files together)."
    features = list(b["spec"].get("features") or {})
    css = "".join((MODULES / f"{m}.css").read_text(encoding="utf-8") for m in features if (MODULES / f"{m}.css").exists())
    cards = "".join((MODULES / f"{m}.html").read_text(encoding="utf-8") for m in features if (MODULES / f"{m}.html").exists())
    js = "\n".join((MODULES / f"{m}.js").read_text(encoding="utf-8") for m in features)
    page = (WEB / "core.html").read_text(encoding="utf-8")
    title = re.sub(r"[<&]", "", b["spec"].get("title") or "Data viewer")
    for slot, value in (("<!--AGENT_GUIDE-->", agent_guide(b, note)), ("<title>Data Viewer</title>", f"<title>{title}</title>"),
                        ("/*MODULE_CSS*/", css), ("<!--SLOT:cards-->", cards), ("/*MODULE_JS*/", js),
                        ("<!--PLOTLY-->", _plotly_tag(plotly, plotly_cache)), ("<!--DATA-->", data_tag)):
        if slot not in page:
            raise RuntimeError(f"core.html lost its {slot} slot")
        page = page.replace(slot, value, 1)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    if data_file:
        data_file.write_text("window.VIEWER_BUNDLE = " + text + ";\n", encoding="utf-8")
    elif out_html.with_name(out_html.stem + ".data.js").exists():
        out_html.with_name(out_html.stem + ".data.js").unlink()  # stale sidecar from an earlier build
    return {"html": out_html, "data_file": data_file, "html_bytes": len(page.encode("utf-8")),
            "data_bytes": data_file.stat().st_size if data_file else 0}


# ---------------------------------------------------------------- reading a page back
def read_viewer_data(path):
    """(DataFrame, spec) from a built page or its .data.js file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".js":
        raw = text[text.index("=") + 1:].strip().rstrip(";")
    else:
        m = re.search(r'^<script id="viewer-data" type="application/json">(.*?)</script>', text, re.S | re.M)
        if m:
            raw = m.group(1)
        else:
            side = path.with_name(path.stem + ".data.js")
            if not side.exists():
                raise FileNotFoundError(f"no embedded data and no {side.name} next to {path.name}")
            return read_viewer_data(side)
    b = json.loads(raw)
    df = pd.DataFrame(b["data"])
    df.insert(0, "_id", b["ids"])
    return df, b["spec"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Build or read a data-viewer page.")
    sub = p.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build", help="table (.csv/.parquet) + spec (.json) -> page")
    bp.add_argument("table"), bp.add_argument("spec"), bp.add_argument("out")
    bp.add_argument("--data", default="auto", choices=["auto", "embed", "file"])
    bp.add_argument("--plotly", default="cdn", help="cdn, inline, or a path to plotly.min.js")
    dp = sub.add_parser("dump", help="page -> csv")
    dp.add_argument("page"), dp.add_argument("out")
    a = p.parse_args(argv)
    if a.cmd == "build":
        df = pd.read_parquet(a.table) if a.table.endswith(".parquet") else pd.read_csv(a.table)
        r = write_viewer(df, json.loads(Path(a.spec).read_text(encoding="utf-8")), a.out, data=a.data, plotly=a.plotly)
        print(f"{r['html']} ({r['html_bytes'] / 1024:.0f} KB)" + (f" + {r['data_file']} ({r['data_bytes'] / 1024:.0f} KB)" if r["data_file"] else ""))
    else:
        df, _ = read_viewer_data(a.page)
        df.to_csv(a.out, index=False)
        print(f"{len(df)} rows -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())

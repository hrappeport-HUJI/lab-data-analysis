---
name: data-viewer
description: Turn a results table into an interactive viewer page (scatter with marginal histograms, filters, live statistics, saved views) that opens by double-click on any computer, can be emailed, and carries instructions for other agents. Optional modules add click-through links to the source data and a chat box. Use when someone wants to explore, share or present per-item results (cells, wells, colonies, samples) interactively instead of as static figures.
---

`build_viewer.py` turns a table plus a short spec into one HTML page. The page knows nothing about any experiment: everything specific (labels, units, colours, what a blank means, definitions, preset views) is in the spec, and travels with the data.

## Principles

These are what make a viewer worth sending. Keep them when adding plot types or modules.

1. **Generic page, specific spec.** The page is the same for every dataset. The spec gives each column a label with units, a kind, a scale, and what a blank means; it gives the definitions text, the preset views and the colours. A reader (or agent) learns what a value means from the page itself.
2. **One file that opens anywhere.** The default page carries its own data and needs nothing but Plotly from the internet, which can also go inside the file for offline use. Large tables go in a `.data.js` file beside the page, never a `.csv` or `.json` file, because a double-clicked page may not read neighbouring data files but may load neighbouring scripts.
3. **Every point leads back to its source.** Each row has a stable id, and links (the `links` module) are templates over the row, so a point can open its images, its review page or its raw file.
4. **Measured and inferred look different.** Estimated values are ringed, censored ones hollow, rows with no Y value sit in a labelled band. n is always shown. A result that depends on an assumption should be visibly marked.
5. **Statistics are live, conservative and stated.** Rank-based tests (Spearman with permutation p, Mann-Whitney, Kruskal-Wallis), Holm-corrected, recomputed on exactly what is shown, and the method is written under the tables. Outlier removal is optional, rule-based and drawn as grey crosses; points a person removes are listed and can be put back. Nothing disappears silently.
6. **Any view can be named and recreated.** Presets come from the spec; people save their own views or copy one as text to send.
7. **Small and fast.** Data are column-wise and rounded to 4 significant digits, plots use WebGL. Embedded pages stay fast to roughly 50,000 rows. Raw data (images, time series) stay behind links, not inside the page.
8. **No secrets in the page.** Pages get forwarded. Access keys and credentials stay in each viewer's browser (the review viewer remembers its key per browser), never in the spec.
9. **Agents are first-class readers.** Every page starts with an agent guide (an HTML comment: what a row is, columns, definitions, where the data sit) and exposes `window.viewer` with the same data tools a chat model uses. An agent can read the file, or open it and call the tools, without guessing.
10. **The chat is optional and a fallback.** When on, it computes only through the page's tools, shows each tool call, and can change the plot. The controls stay the main interface.

## Build a viewer

```python
import sys; sys.path.insert(0, "<repo>/skills/data-viewer")
from build_viewer import write_viewer
write_viewer(df, spec, "results/viewer.html")
```

or without Python code: `python build_viewer.py build table.csv spec.json out.html`. The spec format is in the module docstring of `build_viewer.py`; `check_spec` fails the build with every problem listed (missing columns, a filter that is not a category, a view naming an unknown column, an unknown feature).

Options of `write_viewer`:

| Option | Default | Other values |
|---|---|---|
| `data` | `"auto"`: embedded below 5 MB of data, otherwise `out.data.js` beside the page | `"embed"`, `"file"` |
| `plotly` | `"cdn"`: loaded from cdnjs (needs internet) | `"inline"` (downloaded once to `~/.cache/data-viewer`, +3.5 MB, works offline), or a path to `plotly.min.js` |

Writing the spec: start from the questions the person asked. Give each preset view a name that says what it answers ("Initial PI vs lag"), set `hollow_when`/`ring_when` for censored or estimated rows, write `missing` for every column that can be blank, and put the operational definitions in `notes`. Leave out columns nobody will plot or filter on.

Reading a page back: `read_viewer_data(path)` returns `(DataFrame, spec)`; `python build_viewer.py dump page.html out.csv` writes the table.

## Modules (optional features)

Switch a module on by naming it in `spec["features"]`; a module not named is not put in the page.

| Module | Config | What it does |
|---|---|---|
| `links` | `[{"name": "Open in review viewer", "url": "http://host:8891/?founder={id}"}]` | Adds links to the card that opens when a point is clicked. `{column}` placeholders, `{id}` is the row id. |
| `series` | `{"title", "x_label", "y_label", "y_scale", "styles": {curve: {mode, dash, color, opacity}}, "lines": [{"column", "label"}]}`, plus `write_viewer(..., series=long_table)` with columns `id, series, x, y` | Draws each clicked row's own curves (e.g. a signal over time, its fit, the descendants' mean) in the card, with vertical lines at that row's values of the named columns (e.g. lag) and a log/linear toggle. Adds a `get_series` tool for agents and the chat. |
| `chat` | `{"placeholder": "an example question"}` | A chat box answered by Claude. Works only when the page is published as a claude.ai artifact with `capabilities: {"sample": {}}`; elsewhere it says so. Uses the viewer's own Claude account. |

Without modules, clicking a point opens a card with all its values and a Remove / Put back button.

Adding a module: put `web/modules/<name>.js` (and optionally `<name>.css`, and `<name>.html` for a card under the plot) in the folder. The JS calls `viewerModule("<name>", {init(cfg) {}, pickActions(cfg, row, headerElement) {}, pickBody(cfg, row, cardElement) {}})`; `cfg` is `spec.features[name]`. A module can also push data tools onto `TOOLS`; they then appear in `window.viewer` and the chat. It can use the page's globals (`S` spec, `COLS`, `ROWS`, `state`, `draw()`, `TOOLS`). Add a row to the table above and a test in `tests/test_build_viewer.py`. New plot types (time series, image grids, heatmaps) should come in the same way, sharing the spec, filters, links and statistics.

## Sharing

| To | Do |
|---|---|
| Send to anyone | Email or share the `.html` (plus `.data.js` if there is one, as a zip). It opens by double-click. |
| Work offline (a talk, a flight) | Build with `plotly="inline"`. |
| Public web page | Commit the page to a public GitHub repo with Pages on. Free Pages serves only public repos, so the data become public: only for published data. |
| Use the chat | Build with `features: {"chat": {}}` and publish it as a private claude.ai artifact with the `sample` capability. |

Before sharing, check that the page holds no unpublished data the person did not mean to send, and no access keys in link templates.

## Check

`python -m pytest skills/data-viewer/tests`. With Chrome installed, the tests also open built pages from disk headless and check that the plot, statistics, click card and agent tools work; without Chrome that check is skipped, so after changing the page code open a built page in a real browser yourself.

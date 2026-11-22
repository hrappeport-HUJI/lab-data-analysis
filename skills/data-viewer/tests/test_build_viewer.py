import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_viewer as bv  # noqa: E402


def toy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"key": [f"r{i}" for i in range(40)], "grp": ["a", "b"] * 20,
                       "x": rng.normal(size=40), "y": rng.lognormal(size=40)})
    df.loc[3, "y"] = np.nan
    df.loc[5, "grp"] = "</script><!-- odd"
    spec = {"title": "Toy", "id": "key", "label": "row {key}", "color_by": "grp", "filters": ["grp"],
            "notes": "x and y are made up.", "row_meaning": "one toy record",
            "views": [{"name": "y vs x", "x": "x", "y": "y", "y_scale": "log"}],
            "columns": {"grp": {"label": "Group", "kind": "category", "order": ["a", "b"]},
                        "x": {"label": "X (u)"}, "y": {"label": "Y (u)", "scale": "log", "missing": "not measured"}}}
    return df, spec


def scripts(page):
    return [m.group(1) for m in re.finditer(r"<script>(.*?)</script>", page, re.S)]


def test_default_is_one_self_contained_file_without_chat(tmp_path):
    df, spec = toy()
    r = bv.write_viewer(df, spec, tmp_path / "v.html")
    page = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert r["data_file"] is None and not (tmp_path / "v.data.js").exists()
    assert page.startswith("<!doctype html>") and '<meta charset="utf-8">' in page
    assert "AGENT GUIDE" in page and "window.viewer" in page and "y: Y (u) (number) - blank means not measured" in page
    assert "chat-card" not in page and "initChat" not in page and "claude.use" not in page
    # plotly, data, page code: the odd category value cannot close a tag (the guide comment may quote it harmlessly)
    assert re.sub(r"<!--.*?-->", "", page, count=1, flags=re.S).count("</script>") == 3
    assert "plotly" in page and "cdnjs" in page and "<title>Toy</title>" in page
    back, back_spec = bv.read_viewer_data(tmp_path / "v.html")
    assert list(back["_id"]) == list(df.key) and back.grp[5] == "</script><!-- odd"
    assert back.y.isna().sum() == 1 and np.allclose(back.x, df.x, rtol=1e-3)
    assert back_spec["columns"]["grp"]["order"][:2] == ["a", "b"]


def test_data_in_separate_file(tmp_path):
    df, spec = toy()
    r = bv.write_viewer(df, spec, tmp_path / "v.html", data="file")
    assert r["data_file"] == tmp_path / "v.data.js"
    page = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert '<script src="v.data.js"></script>' in page and "viewer-data" not in page.split("<!--DATA-->")[0][-50:]
    assert (tmp_path / "v.data.js").read_text(encoding="utf-8").startswith("window.VIEWER_BUNDLE = {")
    assert len(bv.read_viewer_data(tmp_path / "v.html")[0]) == 40
    # auto switches to a file above the limit, and a later embedded build removes the stale file
    assert bv.write_viewer(df, spec, tmp_path / "w.html", embed_limit_mb=1e-4)["data_file"] is not None
    bv.write_viewer(df, spec, tmp_path / "w.html", data="embed")
    assert not (tmp_path / "w.data.js").exists()


def test_features_are_opt_in(tmp_path):
    df, spec = toy()
    spec["features"] = {"links": [{"name": "Source", "url": "http://h/?id={id}&g={grp}"}], "chat": {"placeholder": "Ask"}}
    bv.write_viewer(df, spec, tmp_path / "v.html")
    page = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert 'viewerModule("links"' in page and 'viewerModule("chat"' in page and 'id="chat-card"' in page
    assert ".chat-log" in page and "Features in this page: links, chat" in page


def test_plotly_inline_from_a_local_copy(tmp_path):
    df, spec = toy()
    fake = tmp_path / "plotly.min.js"
    fake.write_text("window.Plotly={react(){},};", encoding="utf-8")
    bv.write_viewer(df, spec, tmp_path / "v.html", plotly=str(fake))
    page = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert "cdnjs" not in page and "window.Plotly={react(){},};" in page


def test_spec_errors_are_reported_together():
    df, spec = toy()
    spec["filters"] = ["x"]
    spec["views"][0]["x"] = "nope"
    spec["features"] = {"teleport": {}}
    spec["columns"]["missing_col"] = {"label": "M"}
    with pytest.raises(ValueError) as e:
        bv.bundle(df, spec)
    msg = str(e.value)
    for part in ("'missing_col' is not in the table", "'x' is used as a filter", "x 'nope' is not a numeric column", "feature 'teleport'"):
        assert part in msg


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_code_parses(tmp_path):
    df, spec = toy()
    spec["features"] = {name: ({} if name != "links" else [{"name": "S", "url": "http://h/{id}"}]) for name in bv.available_modules()}
    bv.write_viewer(df, spec, tmp_path / "v.html")
    code = scripts((tmp_path / "v.html").read_text(encoding="utf-8"))[-1]
    (tmp_path / "page.js").write_text(code, encoding="utf-8")
    subprocess.run(["node", "--check", str(tmp_path / "page.js")], check=True)


def _chrome():
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if Path(p).exists():
            return p
    return None


PROBE = """<script>setTimeout(() => { const o = document.createElement('pre'); o.id = 'probe';
try { showPick(ROWS[0]); o.textContent = JSON.stringify({ svg: !!document.querySelector('#plot .main-svg'),
  within: document.querySelectorAll('#t-within tbody tr').length, links: [...document.querySelectorAll('#pick a')].map(a => a.href),
  groups: window.viewer.summarize({group_by: ['grp'], columns: ['x']}).length });
} catch (e) { o.textContent = 'ERROR ' + e.message; } document.body.appendChild(o); }, 2000);</script>"""


@pytest.mark.skipif(_chrome() is None, reason="no Chrome for a headless render check")
@pytest.mark.parametrize("data", ["embed", "file"])
def test_page_draws_when_opened_from_disk(tmp_path, data):
    """Open the page from file:// as a double-click would: plot, statistics, click card, agent tools."""
    import json
    df, spec = toy()
    spec["features"] = {"links": [{"name": "Source", "url": "http://h/?id={id}"}]}
    bv.write_viewer(df, spec, tmp_path / "v.html", data=data)
    page = tmp_path / "v.html"
    page.write_text(page.read_text(encoding="utf-8").replace("</body>", PROBE + "</body>"), encoding="utf-8")
    try:
        out = subprocess.run([_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=10000", "--dump-dom", page.as_uri()],
                             capture_output=True, text=True, encoding="utf-8", timeout=120).stdout
    except subprocess.TimeoutExpired:
        pytest.skip("Chrome did not finish")
    m = re.search(r'<pre id="probe">(.*?)</pre>', out, re.S)
    if m is None and "did not load" in out:
        pytest.skip("Plotly could not be downloaded (offline)")
    assert m, out[:500]
    got = json.loads(m.group(1))
    assert got["svg"] and got["within"] == 3 and got["groups"] == 3  # a, b and the odd group
    assert got["links"] == ["http://h/?id=r0"]


def test_command_line_build_and_dump(tmp_path):
    df, spec = toy()
    df.to_csv(tmp_path / "t.csv", index=False)
    (tmp_path / "s.json").write_text(__import__("json").dumps(spec), encoding="utf-8")
    bv.main(["build", str(tmp_path / "t.csv"), str(tmp_path / "s.json"), str(tmp_path / "v.html")])
    bv.main(["dump", str(tmp_path / "v.html"), str(tmp_path / "back.csv")])
    assert len(pd.read_csv(tmp_path / "back.csv")) == 40


def test_timeseries_module(tmp_path):
    df, spec = toy()
    df["well"] = [f"w{i % 5}" for i in range(40)]
    df["t_event"] = 2.0
    spec["columns"]["well"] = {"label": "Well", "kind": "category"}
    spec["columns"]["t_event"] = {"label": "Event time (h)"}
    series = {f"w{k}": {"OD": {"t": [0, 1, 2, 3], "v": [0.1, 0.2, 0.4, 0.8]}} for k in range(5)}
    spec["features"] = {"timeseries": {"series": series, "channels": {"OD": {"label": "OD", "scale": "log"}},
                                       "panels": [{"name": "Well OD", "series_col": "well", "events": {"drug": "t_event"}, "x": "drug"}]}}
    bv.write_viewer(df, spec, tmp_path / "t.html")
    page = (tmp_path / "t.html").read_text(encoding="utf-8")
    assert 'id="ts-card"' in page and 'viewerModule("timeseries"' in page and ".ts-row" in page
    assert bv.read_viewer_data(tmp_path / "t.html")[1]["features"]["timeseries"]["series"]["w0"]["OD"]["v"][-1] == 0.8
    bad = copy_spec(spec)
    bad["features"]["timeseries"]["panels"][0].update(series_col="grp", events={"drug": "nope"}, x="later")
    with pytest.raises(ValueError) as e:
        bv.write_viewer(df, bad, tmp_path / "bad.html")
    msg = str(e.value)
    assert "no series for" in msg and "'nope' is not a numeric column" in msg and "neither 'absolute' nor an event" in msg


def test_cdf_module(tmp_path):
    df, spec = toy()
    spec["features"] = {"cdf": {"column": "x", "columns": ["x", "y"], "scale": "linear", "title": "X survival"}}
    bv.write_viewer(df, spec, tmp_path / "c.html")
    page = (tmp_path / "c.html").read_text(encoding="utf-8")
    assert 'id="cdf-card"' in page and 'viewerModule("cdf"' in page and "#cdf-plot" in page
    bad = copy_spec(spec)
    bad["features"]["cdf"].update(column="grp", scale="sqrt")
    with pytest.raises(ValueError) as e:
        bv.write_viewer(df, bad, tmp_path / "bad.html")
    assert "'grp' is not a numeric column" in str(e.value) and "scale must be" in str(e.value)


def _render(page):
    """Run a page in headless Chrome and return the JSON its probe script wrote."""
    import json
    try:
        out = subprocess.run([_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=10000", "--dump-dom", page.as_uri()],
                             capture_output=True, text=True, encoding="utf-8", timeout=120).stdout
    except subprocess.TimeoutExpired:
        pytest.skip("Chrome did not finish")
    m = re.search(r'<pre id="probe">(.*?)</pre>', out, re.S)
    if m is None and "did not load" in out:
        pytest.skip("Plotly could not be downloaded (offline)")
    assert m, out[:500]
    import html
    text = html.unescape(m.group(1))
    assert not text.startswith("ERROR"), text
    return json.loads(text)


SPLIT_PROBE = """<script>setTimeout(() => { const o = document.createElement('pre'); o.id = 'probe';
try { const before = (document.getElementById('cdf-plot').data || []).length;
  window.viewer.set_view({split_by: 'day', split_style: 'style'});
  const tr = document.getElementById('cdf-plot').data;
  document.querySelector('[data-cdf-axis=x][data-cdf-scale=log]').click();
  o.textContent = JSON.stringify({ before, cdf: tr.map(t => t.name.split(' (')[0]), dashes: [...new Set(tr.map(t => t.line.dash))].length,
    within: document.querySelectorAll('#t-within tbody tr').length, xlog: document.getElementById('cdf-plot').layout.xaxis.type,
    view: window.viewer.describe_data().current_view.split_by });
} catch (e) { o.textContent = 'ERROR ' + e.message; } document.body.appendChild(o); }, 2000);</script>"""


@pytest.mark.skipif(_chrome() is None, reason="no Chrome for a headless render check")
def test_split_groups_and_cdf_axes_in_the_browser(tmp_path):
    df, spec = toy()
    df["day"] = ["d1"] * 20 + ["d2"] * 20
    spec["columns"]["day"] = {"label": "Day", "kind": "category", "order": ["d1", "d2"]}
    spec["features"] = {"cdf": {"column": "x"}}
    df["x"] = df["x"].abs() + 0.1
    bv.write_viewer(df, spec, tmp_path / "s.html")
    page = tmp_path / "s.html"
    page.write_text(page.read_text(encoding="utf-8").replace("</body>", SPLIT_PROBE + "</body>"), encoding="utf-8")
    got = _render(page)
    assert got["before"] == 3                      # a, b and the odd group
    # 3 groups x 2 days, minus the empty combination (the odd group's one row is on d1)
    assert len(got["cdf"]) == 5 and "a · d1" in got["cdf"] and "b · d2" in got["cdf"]
    assert got["dashes"] == 2 and got["within"] == 5 and got["xlog"] == "log" and got["view"] == "day"


def test_split_spec_is_checked(tmp_path):
    df, spec = toy()
    spec["split_by"] = "x"
    spec["split_style"] = "hue"
    spec["views"][0]["split_by"] = "nope"
    with pytest.raises(ValueError) as e:
        bv.write_viewer(df, spec, tmp_path / "bad.html")
    msg = str(e.value)
    assert "split_by 'x' is not a category column" in msg and "split_style must be one of" in msg and "split_by 'nope'" in msg


def copy_spec(spec):
    import copy
    return copy.deepcopy(spec)


def test_log_ticks_cover_narrow_ranges():
    page = (Path(bv.WEB) / "core.html").read_text(encoding="utf-8")
    # narrow log ranges fall back to 1-2-5 steps and then to even ticks, so an axis never loses its labels
    assert "for (const m of [1, 2, 5])" in page and "if (inside.length < 3) inside = niceTicks(" in page
    assert 'hook("drawn", sel)' in page and 'hook("picked", r)' in page

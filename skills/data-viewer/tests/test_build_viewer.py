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


def toy_series(df):
    rows = [(k, "signal", t, 100 * np.exp(-.3 * t)) for k in df.key[:5] for t in np.arange(0, 3, .5)]
    rows += [(k, "fit", t, 99 * np.exp(-.3 * t)) for k in df.key[:5] for t in (0, 3)] + [("not-a-row", "signal", 0, 1)]
    return pd.DataFrame(rows, columns=["id", "series", "x", "y"])


def series_spec(spec):
    spec["features"] = {"series": {"title": "Signal", "x_label": "t", "styles": {"fit": {"mode": "lines", "dash": "dash"}},
                                   "lines": [{"column": "x", "label": "x"}]}}
    return spec


def test_series_round_trip_and_pairing(tmp_path):
    df, spec = toy()
    with pytest.raises(ValueError, match="go together"):
        bv.write_viewer(df, series_spec(spec), tmp_path / "v.html")
    with pytest.raises(ValueError, match="go together"):
        bv.write_viewer(df, toy()[1], tmp_path / "v.html", series=toy_series(df))
    bv.write_viewer(df, series_spec(spec), tmp_path / "v.html", series=toy_series(df))
    page = (tmp_path / "v.html").read_text(encoding="utf-8")
    assert 'viewerModule("series"' in page and "get_series({id})" in page and "Per-row curves" in page
    s = bv.read_viewer_series(tmp_path / "v.html")
    assert set(s.id) == set(df.key[:5]) and len(s) == 5 * 8  # the row that is not in the table is dropped
    assert bv.read_viewer_series(bv.write_viewer(df, toy()[1], tmp_path / "w.html")["html"]).empty


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
    bv.write_viewer(df, spec, tmp_path / "v.html", series=toy_series(df))
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
  within: document.querySelectorAll('#t-within tbody tr').length, links: [...document.querySelectorAll('#pick .pick-head a')].map(a => a.href),
  groups: window.viewer.summarize({group_by: ['grp'], columns: ['x']}).length,
  series_traces: document.querySelectorAll('#pick .series .scatterlayer .trace').length,
  series_tool: Object.keys(window.viewer.get_series ? window.viewer.get_series({id: ROWS[0].id}) : {}) });
} catch (e) { o.textContent = 'ERROR ' + e.message; } document.body.appendChild(o); }, 2000);</script>"""


@pytest.mark.skipif(_chrome() is None, reason="no Chrome for a headless render check")
@pytest.mark.parametrize("data", ["embed", "file"])
def test_page_draws_when_opened_from_disk(tmp_path, data):
    """Open the page from file:// as a double-click would: plot, statistics, click card, agent tools."""
    import json
    df, spec = toy()
    spec = series_spec(spec)
    spec["features"]["links"] = [{"name": "Source", "url": "http://h/?id={id}"}]
    bv.write_viewer(df, spec, tmp_path / "v.html", data=data, series=toy_series(df))
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
    assert got["series_traces"] == 2 and sorted(got["series_tool"]) == ["fit", "signal"]


def test_command_line_build_and_dump(tmp_path):
    df, spec = toy()
    df.to_csv(tmp_path / "t.csv", index=False)
    (tmp_path / "s.json").write_text(__import__("json").dumps(spec), encoding="utf-8")
    bv.main(["build", str(tmp_path / "t.csv"), str(tmp_path / "s.json"), str(tmp_path / "v.html")])
    bv.main(["dump", str(tmp_path / "v.html"), str(tmp_path / "back.csv")])
    assert len(pd.read_csv(tmp_path / "back.csv")) == 40

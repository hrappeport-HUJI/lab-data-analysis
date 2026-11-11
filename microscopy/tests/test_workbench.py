"""Restored founder-review viewer running on microscopy-package projects."""

import hashlib
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from microscopy.demo import create
from microscopy.events import summarize
from microscopy.review import materialize, save
from microscopy.storage import Project
from microscopy.workbench.adapter import RunReviewModel, FROM_STORE
from microscopy.workbench.server import WorkbenchServer


@pytest.fixture
def project(tmp_path):
    return create(tmp_path / "demo")


def db_digest(project):
    path = project.root / "reviews.sqlite"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def test_records_match_review_store_and_old_viewer_vocabulary(project):
    model = RunReviewModel(project.root)
    records = {r["id"]: r for r in model.all_effective()}
    run_id = project.latest_run()
    rows = summarize(project, materialize(project, run_id, project.review(run_id)["operations"]))
    assert set(records) == {r["id"] for r in rows}
    for row in rows:
        r = records[row["id"]]
        assert r["decision"] == FROM_STORE[row["decision"]]
        assert r["condition"] == row["condition"]
        assert r["frame_count"] == 8  # from the manifest, not a constant
        if r["decision"] == "divided":
            assert r["last_single_frame"] == row["lower_frame"] < r["division_frame"] == row["upper_frame"]
            assert r["event_frame"] == r["division_frame"]
        if r["decision"] == "unresolved":
            assert (r["last_valid_frame"], r["event_frame"]) == (row["lower_frame"], row["upper_frame"])
    cfg = model.config()
    assert cfg["structure_channel"] == "phase"
    assert cfg["channels"] == [{"id": "PI", "label": "PI"}]
    assert (cfg["image_width"], cfg["image_height"]) == (128, 96)
    assert cfg["fields"]["1"]["times_s"][1] - cfg["fields"]["1"]["times_s"][0] == pytest.approx(600)


def test_saved_review_is_picked_up_without_restart(project):
    model = RunReviewModel(project.root)
    run_id = project.latest_run()
    target = next(r for r in model.all_effective() if r["decision"] == "divided")
    assert not target["reviewed"]
    save(project, run_id, {"type": "set_event", "founder_id": target["id"], "decision": "lost",
                           "lower_frame": 1, "upper_frame": 2, "note": "checked"}, 0)
    updated = model.effective(target["id"])
    assert updated["reviewed"] and updated["decision"] == "unresolved"
    assert (updated["last_valid_frame"], updated["event_frame"], updated["note"]) == (1, 2, "checked")
    assert updated["original"]["decision"] == "divided"  # automatic call stays visible


def test_render_is_small_window_with_outline_and_metadata(project):
    model = RunReviewModel(project.root)
    r = next(r for r in model.all_effective() if r["decision"] == "divided")
    png, meta = model.render(r, 0, mode="overlay")
    image = Image.open(io.BytesIO(png))
    assert image.size == (640, 640)
    assert meta["frame"] == 0 and meta["has_outline"] and meta["channel_label"] == "PI"
    assert meta["structure_usable"] and not meta["fluorescence_missing"]
    # Display controls never change the review state.
    before = db_digest(project)
    model.render(r, 3, mode="fluor", gain=4, full=True)
    model.render(r, 3, mode="phase", width=192, outlines=False)
    assert db_digest(project) == before
    with pytest.raises(ValueError):
        model.render(r, 99)


def test_missing_channel_frame_is_reported_not_zero(project):
    model = RunReviewModel(project.root)
    r = model.all_effective()[0]
    model.project.frames.pop((r["position"], 2, "PI"))  # simulate a frame without fluorescence
    _, meta = model.render(r, 2, mode="overlay")
    assert meta["fluorescence_missing"] is True


def test_server_is_read_only_and_guards_session(project):
    server = WorkbenchServer(project.root, 0, read_only=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = server.origin
        token = json.loads(urllib.request.urlopen(base + "/api/session").read())["token"]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/project")
        assert e.value.code == 403
        data = json.loads(urllib.request.urlopen(
            urllib.request.Request(base + "/api/project", headers={"X-Review-Token": token})).read())
        assert data["config"]["read_only"] is True and len(data["founders"]) > 0
        fid = data["founders"][0]["id"]
        response = urllib.request.urlopen(urllib.request.Request(
            f"{base}/api/frame?id={fid}&frame=1&mode=overlay", headers={"X-Review-Token": token}))
        assert response.headers["Content-Type"] == "image/png"
        assert json.loads(response.headers["X-Frame-Meta"])["frame"] == 1
        before = db_digest(project)
        request = urllib.request.Request(
            base + "/api/review", method="POST",
            data=json.dumps({"id": fid, "changes": {"decision": "excluded"}, "expected_revision": 0}).encode(),
            headers={"X-Review-Token": token, "Content-Type": "application/json", "Origin": base})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(request)
        assert e.value.code == 403 and "read-only" in json.loads(e.value.read())["error"]
        assert db_digest(project) == before
    finally:
        server.shutdown()
        server.server_close()


def test_static_stage_has_no_drift(project):
    model = RunReviewModel(project.root)
    r = model.all_effective()[0]
    assert model.center(r, 0).tolist() == model.center(r, 7).tolist()


def post(base, token, path, body):
    request = urllib.request.Request(
        base + path, method="POST", data=json.dumps(body).encode(),
        headers={"X-Review-Token": token, "Content-Type": "application/json", "Origin": base})
    return json.loads(urllib.request.urlopen(request).read())


def test_tagging_saves_reverts_and_round_trips(project):
    server = WorkbenchServer(project.root, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = server.origin
        token = json.loads(urllib.request.urlopen(base + "/api/session").read())["token"]
        founders = server.model.all_effective()
        a, b = founders[0]["id"], founders[1]["id"]
        # Division at frame 3, with the old viewer's field names.
        x = post(base, token, "/api/review", {"id": a, "expected_revision": 0, "changes": {
            "decision": "divided", "last_single_frame": 2, "division_frame": 3, "last_valid_frame": 5,
            "note": "clear septum", "confidence": "high", "reviewer": "HR"}})
        assert x["revision"] == 1
        r = x["record"]
        assert (r["decision"], r["last_single_frame"], r["division_frame"], r["last_valid_frame"]) == ("divided", 2, 3, 5)
        assert r["reviewed"] and r["saved_by_viewer"] and r["note"] == "clear septum" and r["reviewer"] == "HR"
        # Doublet exclusion keeps its reason.
        x = post(base, token, "/api/review", {"id": b, "expected_revision": 1, "changes": {
            "decision": "excluded", "exclude_reason": "doublet"}})
        assert x["record"]["decision"] == "excluded" and x["record"]["exclude_reason"] == "doublet"
        # Stale revision is refused, nothing written.
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base, token, "/api/review", {"id": a, "expected_revision": 1, "changes": {"decision": "excluded"}})
        assert e.value.code == 409
        # Invalid timing is refused with a readable message.
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base, token, "/api/review", {"id": a, "expected_revision": 2, "changes": {
                "decision": "unresolved", "last_valid_frame": 4, "event_frame": 4}})
        assert e.value.code == 400
        # Not-grown needs the full-window confirmation.
        with pytest.raises(urllib.error.HTTPError):
            post(base, token, "/api/review", {"id": a, "expected_revision": 2, "changes": {"decision": "not_grown"}})
        # Per-founder revert restores only that founder.
        exported = json.loads(urllib.request.urlopen(urllib.request.Request(
            base + "/api/export/json", headers={"X-Review-Token": token})).read())
        x = post(base, token, "/api/revert", {"id": a, "expected_revision": 2})
        assert x["revision"] == 3 and not x["record"]["reviewed"]
        assert server.model.effective(b)["exclude_reason"] == "doublet"
        with pytest.raises(urllib.error.HTTPError):
            post(base, token, "/api/revert", {"id": a, "expected_revision": 3})
        # Importing the earlier export restores both decisions as a new revision.
        x = post(base, token, "/api/import", {"reviews": exported, "expected_revision": 3})
        restored = {f["id"]: f for f in x["founders"]}
        assert x["revision"] == 4 and restored[a]["division_frame"] == 3 and restored[b]["decision"] == "excluded"
        # History keeps every revision.
        assert [r["revision"] for r in [project.review(project.latest_run())]] == [4]
    finally:
        server.shutdown()
        server.server_close()


def test_tagging_sets_are_validated_and_served(project, tmp_path):
    fields = sorted(RunReviewModel(project.root).fields)
    good = tmp_path / "sets.json"
    good.write_text(json.dumps({"sets": {"A": [int(fields[0])], "B": fields[1:]}}))
    server = WorkbenchServer(project.root, 0, read_only=True, tagging_sets=good)
    try:
        assert server.tagging_sets == [{"name": "A", "positions": [fields[0]]}, {"name": "B", "positions": fields[1:]}]
    finally:
        server.server_close()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sets": {"A": ["999"]}}))
    with pytest.raises(ValueError, match="999"):
        WorkbenchServer(project.root, 0, read_only=True, tagging_sets=bad)

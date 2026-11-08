import copy
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from microscopy.demo import create
from microscopy.storage import Project, Conflict, file_digest, write_json
from microscopy.importing import initialize, template
from microscopy.pipeline import run, validate_graph, validate_labels
from microscopy.review import materialize, save, polygon_mask
from microscopy.events import summarize
from microscopy.measurement import measure
from microscopy.analysis import analyze, pairwise
from microscopy.images import read_frame


@pytest.fixture
def project(tmp_path):
    return create(tmp_path / "demo")


def state(p):
    rid = p.latest_run()
    r = p.review(rid)
    d = materialize(p, rid, r["operations"])
    return rid, d, summarize(p, d)


def test_end_to_end_and_frozen_results(project):
    rid, d, founders = state(project)
    assert {
        s: sum(f["decision"] == s for f in founders)
        for s in ["division", "lost", "no_division"]
    } == {"division": 4, "lost": 4, "no_division": 4}
    for f in founders:
        if f["decision"] == "division":
            upper = 3 + int(f["field"]) % 2
            assert (f["lower_frame"], f["upper_frame"]) == (upper - 1, upper)
            assert f["division_midpoint_s"] == (upper - 0.5) * 600
    result = analyze(project, rid)
    assert result["review_revision"] == 0
    assert all(
        t["status"] == "field_exchangeability_not_confirmed"
        for t in result["comparisons"]
    )
    assert len(result["plots"]) == 4
    path = project.root / "results" / result["result_id"] / "summary.json"
    before = file_digest(path)
    save(
        project,
        rid,
        {
            "type": "set_event",
            "founder_id": founders[0]["id"],
            "decision": "exclude",
            "note": "ambiguous founder",
        },
        0,
    )
    result2 = analyze(project, rid, reviewed_only=True)
    assert result2["review_revision"] == 1 and result2["counts"]["exclude"] == 1
    assert file_digest(path) == before


def test_review_revision_conflicts_undo_import_and_new_runs(project):
    rid, d, fs = state(project)
    run_hash = file_digest(project.run_path(rid) / "run.json")
    op = {"type": "set_event", "founder_id": fs[0]["id"], "decision": "exclude"}
    save(project, rid, op, 0)
    with pytest.raises(Conflict):
        save(project, rid, op, 0)
    exported = project.export_review(rid)
    save(project, rid, {"type": "undo"}, 1)
    assert project.review(rid)["operations"] == []
    save(project, rid, {"type": "import", "review": exported}, 2)
    assert project.review(rid)["operations"] == [op]
    new = run(project, fields=["1"])["run_id"]
    assert project.review(new)["revision"] == 0 and project.review(rid)["revision"] == 3
    with pytest.raises(ValueError, match="another run"):
        save(project, new, {"type": "import", "review": exported}, 0)
    assert file_digest(project.run_path(rid) / "run.json") == run_hash


def test_split_no_inherited_track_or_duplicate_measurement(project):
    rid, d, fs = state(project)
    f = next(f for f in fs if f["field"] == "1" and f["decision"] == "division")
    op = {
        "type": "split_founder",
        "founder_id": f["id"],
        "polygons": [
            [[24, 22], [30, 22], [30, 28], [24, 28]],
            [[34, 22], [40, 22], [40, 28], [34, 28]],
        ],
    }
    save(project, rid, op, 0)
    _, data, founders = state(project)
    parent = next(r for r in founders if r["id"] == f["id"])
    children = [r for r in founders if r.get("split_parent") == f["id"]]
    assert parent["decision"] == "exclude" and len(children) == 2
    assert all(
        c["decision"] == "lost" and len(c["observation_ids"]) == 1 for c in children
    )
    measurements = measure(project, rid, data, founders)
    assert f["id"] not in {m["founder_id"] for m in measurements}
    assert all(c["id"] in {m["founder_id"] for m in measurements} for c in children)
    with pytest.raises(ValueError, match="overlap"):
        materialize(
            project, rid, [{**op, "polygons": [op["polygons"][0], op["polygons"][0]]}]
        )


def test_quality_gap_and_manual_event_requirements(project):
    rid, d, fs = state(project)
    f = next(f for f in fs if f["field"] == "1" and f["decision"] == "no_division")
    save(
        project,
        rid,
        {
            "type": "set_frame_quality",
            "field": "1",
            "frame": 2,
            "channel": "phase",
            "usable": False,
        },
        0,
    )
    _, d, fs = state(project)
    f2 = next(r for r in fs if r["id"] == f["id"])
    assert f2["decision"] == "lost" and f2["last_identity_frame"] == 1
    with pytest.raises(ValueError, match="throughout"):
        save(
            project,
            rid,
            {"type": "set_event", "founder_id": f["id"], "decision": "no_division"},
            1,
        )
    with pytest.raises(ValueError, match="last-certain"):
        save(
            project,
            rid,
            {
                "type": "set_event",
                "founder_id": f["id"],
                "decision": "division",
                "lower_frame": 3,
                "upper_frame": 2,
            },
            1,
        )
    save(
        project,
        rid,
        {
            "type": "set_event",
            "founder_id": f["id"],
            "decision": "division",
            "lower_frame": 1,
            "upper_frame": 4,
        },
        1,
    )
    _, d, fs = state(project)
    manual = next(r for r in fs if r["id"] == f["id"])
    assert manual["reviewed"] and manual["division_midpoint_s"] == 1500


def test_terminal_division_requires_following_confirmation(project):
    rid, d, fs = state(project)
    d["config"]["tracking"]["confirmation_frames"] = 6
    assert all(f["decision"] != "division" for f in summarize(project, d))


def test_measurement_offsets_exposure_missing_and_negative(project):
    rid, d, fs = state(project)
    rows = measure(project, rid, d, fs)
    for row in rows:
        assert row["status"] == "ok"
        assert abs(row["net_mean"] - (40 + int(row["field"]) * 10)) < 2
        assert (
            abs(row["background_median"] - (100 if int(row["field"]) < 3 else 400)) < 3
        )
    recipe = project.config["measurements"][0]
    recipe.update(normalize_exposure=True)
    with pytest.raises(ValueError, match="linear"):
        measure(project, rid, d, fs)
    recipe.update(linear_response_verified=True, reference_exposure_ms=200)
    normalized = measure(project, rid, d, fs)
    assert normalized[0]["value"] == pytest.approx(rows[0]["net_mean"] * 2)
    key = (rows[0]["field"], 0, "PI")
    project.frames[key]["exposure_ms"] = None
    missing = measure(project, rid, d, fs)
    assert missing[0]["status"] == "missing_exposure" and missing[0]["value"] is None
    recipe["normalize_exposure"] = False
    d["quality_overrides"] = [
        {"field": rows[0]["field"], "frame": 0, "channel": "PI", "usable": False}
    ]
    assert measure(project, rid, d, fs)[0]["status"] == "unusable_image_or_mask"


def test_alternative_tracker_and_custom_segmenter_contract(project):
    from microscopy.adapters.classical import CentroidTracker, ThresholdSegmenter

    result = run(
        project, tracker=CentroidTracker(), segmenter=ThresholdSegmenter(), fields=["1"]
    )
    assert (
        result["tracker_class"].endswith("CentroidTracker")
        and len(result["founders"]) == 3
    )

    class Bad:
        def segment(self, image, **kwargs):
            return np.zeros(image.shape, float)

    with pytest.raises(ValueError, match="integer"):
        run(project, segmenter=Bad(), fields=["1"])
    with pytest.raises(ValueError, match="Merges"):
        validate_graph(
            [
                {"id": "a", "field": "1", "frame": 0},
                {"id": "b", "field": "1", "frame": 0},
                {"id": "c", "field": "1", "frame": 1},
            ],
            [["a", "c"], ["b", "c"]],
        )


def test_polygon_checks():
    assert polygon_mask([[1, 1], [7, 1], [7, 7], [1, 7]], (10, 10)).sum() == 49
    with pytest.raises(ValueError, match="cross"):
        polygon_mask([[1, 1], [7, 7], [1, 7], [7, 1]], (10, 10))
    with pytest.raises(ValueError, match="outside"):
        polygon_mask([[-1, 1], [7, 1], [1, 7]], (10, 10))
    # Disjoint collinear edges in a valid concave polygon must be accepted.
    assert (
        polygon_mask(
            [[1, 1], [3, 1], [3, 3], [5, 3], [5, 1], [7, 1], [7, 7], [1, 7]], (10, 10)
        ).sum()
        > 0
    )


def test_import_refuses_stack_and_never_assumes_timing(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "Pos1" / "phase"
    path.mkdir(parents=True)
    cfg = template()
    cfg["structure_channel"] = "phase"
    write_json(tmp_path / "cfg.json", cfg)
    tifffile.imwrite(
        path / "img_0.tif",
        np.zeros((3, 20, 20), dtype="uint16"),
        photometric="minisblack",
    )
    with pytest.raises(ValueError, match="2D"):
        initialize(raw, tmp_path / "cfg.json", tmp_path / "project")
    tifffile.imwrite(path / "img_0.tif", np.zeros((20, 20), dtype="uint16"))
    p = initialize(raw, tmp_path / "cfg.json", tmp_path / "project")
    assert p.manifest[0]["time_s"] is None and p.manifest[0]["exposure_ms"] is None
    with pytest.raises(ValueError, match="outside"):
        initialize(raw, tmp_path / "cfg.json", raw / "analysis")
    tifffile.imwrite(path / "img_0.tif", np.ones((20, 20), dtype="uint16"))
    with pytest.raises(ValueError, match="changed"):
        read_frame(p, "1", 0, "phase")


def test_statistics_use_units_and_only_planned_pairs():
    rows = [
        {
            "founder_id": str(i),
            "metric": "PI",
            "condition": c,
            "field": str(i),
            "biological_replicate": str(i),
            "value": v,
        }
        for i, (c, v) in enumerate(zip("AAABBB", [1, 2, 3, 4, 5, 6]))
    ]
    table = pd.DataFrame(rows)
    test = pairwise(table, [["A", "B"]], unit="biological_replicate")[0]
    assert test["p_raw"] == pytest.approx(0.1) and test["n_units_1"] == 3
    duplicated = pd.concat([table] * 20, ignore_index=True)
    assert (
        pairwise(duplicated, [["A", "B"]], unit="biological_replicate")[0]["p_raw"]
        == test["p_raw"]
    )
    assert pairwise(table, [["A", "B"]])[0]["p_raw"] is None
    assert pairwise(table, []) == []
    table["biological_replicate"] = "same"
    assert pairwise(table, [["A", "B"]], unit="biological_replicate")[0][
        "status"
    ].startswith("paired_design")


def test_http_review_save_and_origin_protection(project):
    from microscopy.server import ReviewServer

    server = ReviewServer(project.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(server.origin + "/api/state") as r:
            s = json.load(r)
        body = json.dumps(
            {
                "revision": s["revision"],
                "operation": {
                    "type": "set_event",
                    "founder_id": s["founders"][0]["id"],
                    "decision": "exclude",
                },
            }
        ).encode()
        bad = urllib.request.Request(
            server.origin + "/api/review",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 403
        good = urllib.request.Request(
            server.origin + "/api/review",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": server.origin,
                "X-Review-Token": s["token"],
            },
        )
        with urllib.request.urlopen(good) as r:
            assert json.load(r)["revision"] == 1
        with urllib.request.urlopen(
            server.origin + "/api/image?field=1&frame=0&channel=phase"
        ) as r:
            assert r.read().startswith(b"\x89PNG")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(server.origin + "/results/../../project.json")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_negative_signal_and_manual_mask_are_quantitative(project, monkeypatch):
    import microscopy.measurement as measurement

    rid, d, fs = state(project)
    f = fs[0]
    image = np.full((96, 128), 100, dtype=np.uint16)
    image[22:29, 24:41] = 90
    monkeypatch.setattr(measurement, "read_frame", lambda *args: image)
    rows = measure(project, rid, d, [f])
    assert rows[0]["value"] == -10
    op = {
        "type": "set_mask",
        "observation_id": f["observation_id"],
        "polygon": [[24, 22], [27, 22], [27, 25], [24, 25]],
    }
    changed = materialize(project, rid, [op])
    rows2 = measure(project, rid, changed, summarize(project, changed)[:1])
    assert rows2[0]["area_px"] == 16 and rows2[0]["net_integrated"] == -160


def test_manual_gap_cannot_imply_continuous_no_division(project):
    rid, d, fs = state(project)
    f = next(f for f in fs if f["field"] == "1" and f["decision"] == "no_division")
    byframe = {
        n["frame"]: n["id"]
        for n in d["observations"]
        if n["id"] in f["observation_ids"]
    }
    data = materialize(
        project,
        rid,
        [
            {"type": "set_links", "observation_id": byframe[1], "children": []},
            {
                "type": "set_links",
                "observation_id": byframe[0],
                "children": [byframe[2]],
            },
        ],
    )
    item = next(r for r in summarize(project, data) if r["id"] == f["id"])
    assert item["contains_tracking_gap"] and item["decision"] == "lost"


def test_sparse_channel_and_event_relative_selection(project):
    rid, d, fs = state(project)
    f = next(f for f in fs if f["field"] == "1" and f["decision"] == "division")
    project.config["measurements"] = [
        {
            "name": "before",
            "channel": "PI",
            "selection": "event_relative",
            "offset_s": -300,
            "tolerance_s": 1,
        }
    ]
    rows = measure(project, rid, d, [f])
    assert len(rows) == 1 and rows[0]["frame"] == f["lower_frame"]
    project.config["measurements"][0].update(selection="frame", frame=0)
    project.frames.pop(("1", 0, "PI"))
    assert measure(project, rid, d, [f])[0]["status"] == "channel_not_acquired"


def test_focus_and_configuration_rejections():
    from microscopy.config import validate

    cfg = template()
    cfg["tracking"]["confirmation_frames"] = -1
    with pytest.raises(ValueError, match="confirmation"):
        validate(cfg)
    cfg = template()
    cfg["analysis"]["comparisons"] = [["A", "B"], ["B", "A"]]
    with pytest.raises(ValueError, match="distinct"):
        validate(cfg)


def test_baseline_run_does_not_claim_outcomes(project):
    result = run(project, baseline_only=True, fields=["1"])
    assert {n["frame"] for n in result["observations"]} == {0}
    data = materialize(project, result["run_id"], [])
    assert {f["decision"] for f in summarize(project, data)} == {"not_assessed"}
    assert not result["edges"]


def test_parallel_processing_and_explicit_exclusion(project):
    project.config["segmentation"]["workers"] = 2
    project.config["quality"]["excluded_structure_frames"] = {"1": [2]}
    result = run(project, fields=["1"])
    assert not any(n["frame"] == 2 for n in result["observations"])
    assert (
        next(q for q in result["quality"] if q["frame"] == 2)["reason"]
        == "explicit_image_exclusion"
    )
    data = materialize(project, result["run_id"], [])
    assert all(f["decision"] == "lost" for f in summarize(project, data))


def test_parallel_fields_preserve_canonical_result(project):
    before = project.load_run(project.latest_run())
    project.config["field_workers"] = 2
    after = run(project)
    for key in ["observations", "edges", "founders", "quality"]:
        assert before[key] == after[key]


def test_field_override_recovers_cells_when_hot_pixel_inflates_range():
    from microscopy.adapters.classical import ThresholdSegmenter

    image = np.full((64, 64), 1000, dtype=np.uint16)
    image[10:18, 10:30] = 500
    image[45, 45] = 65000
    parameters = {
        "min_area_px": 20,
        "max_area_px": 300,
        "threshold_by_field": {"special": 100},
    }
    segmenter = ThresholdSegmenter()
    labels = segmenter.segment(
        image, context={"field": "special"}, parameters=parameters
    )
    assert labels[14, 20] > 0
    assert (
        segmenter.segment(image, context={"field": "ordinary"}, parameters=parameters)[
            14, 20
        ]
        == 0
    )

"""Colony-following founder events and the watershed segmenter."""

import numpy as np
import pytest

from microscopy.adapters.classical import ThresholdSegmenter
from microscopy.adapters.watershed import WatershedSegmenter
from microscopy.colony import ColonyEvents, field_shift
from microscopy.demo import create
from microscopy.events import summarize
from microscopy.pipeline import run
from microscopy.review import materialize

SHAPE = (120, 160)


def rod(labels, label, y, x0, x1, half=4):
    labels[y - half:y + half + 1, x0:x1 + 1] = label


def nodes_for(labels, frame):
    out = []
    for label in range(1, int(labels.max()) + 1):
        yy, xx = np.nonzero(labels == label)
        if len(yy):
            out.append({"id": f"f0-t{frame}-o{label}", "label": label, "area_px": int(len(yy)),
                        "x": float(xx.mean()), "y": float(yy.mean())})
    return out


def feed(sequence, parameters=None, skip=()):
    """sequence: list of label images, frame index = position."""
    events = ColonyEvents({"drift_radius_px": 30, **(parameters or {})})
    first = sequence[0]
    nodes = nodes_for(first, 0)
    founders = [{"id": n["id"], "observation_id": n["id"]} for n in nodes]
    events.start(0, first, nodes, founders)
    for frame, labels in enumerate(sequence[1:], 1):
        if frame in skip:
            events.skip(frame)
        else:
            events.update(frame, labels, nodes_for(labels, frame))
    return events.finish(len(sequence) - 1)


def scene(frame, divide_at=4, shift=(0, 0), fuse_at=None):
    """Founder A grows then divides into touching daughters; B never grows; C, D, E are static
    anchors (needed to estimate stage shift); F grows towards G and fuses with it."""
    L = np.zeros(SHAPE, np.int32)
    grow = min(frame, divide_at + 2) * 3
    if frame < divide_at:
        rod(L, 1, 20, 20, 40 + grow)
    else:  # two daughters, touching end to end -> one object for a plain threshold, two here
        rod(L, 1, 20, 20, 35 + grow // 2)
        rod(L, 7, 20, 36 + grow // 2, 52 + grow)
    rod(L, 2, 60, 20, 40)
    rod(L, 3, 100, 20, 40)
    rod(L, 4, 100, 100, 120)
    rod(L, 5, 60, 120, 140)
    if fuse_at is not None:
        if frame < fuse_at:
            rod(L, 6, 40, 60, 75)
            rod(L, 8, 40, 82, 95)
        else:
            rod(L, 6, 40, 60, 95)
    dy, dx = shift
    if dy or dx:
        L = np.roll(L, (dy, dx), axis=(0, 1))
    # relabel consecutively
    ids = np.unique(L[L > 0])
    lut = np.zeros(L.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[L]


def test_division_non_grower_and_fusion():
    seq = [scene(t, divide_at=4, fuse_at=3) for t in range(8)]
    events = feed(seq, {"min_colony_growth": 1.0, "static_split_growth": 0})
    byx = {e: v for e, v in events.items()}
    a = byx["f0-t0-o1"]
    assert a["decision"] == "division" and (a["lower_frame"], a["upper_frame"]) == (3, 4)
    static = [v for k, v in events.items() if v["decision"] == "no_division"]
    assert len(static) == 4 and all(v["upper_frame"] == 7 for v in static)
    fused = [v for v in events.values() if v["tracking_end_reason"] == "fused"]
    assert len(fused) == 2 and all(v["upper_frame"] == 3 for v in fused)


def test_touching_non_growers_are_split_by_footprint_not_fused():
    seq = [scene(t, divide_at=99, fuse_at=3) for t in range(7)]
    events = feed(seq)
    touched = [v for v in events.values() if "touched_neighbour" in v["flags"]]
    assert len(touched) == 2 and all(v["decision"] == "no_division" for v in touched)


def test_stage_jitter_is_not_a_loss_or_fusion():
    seq = [scene(t, divide_at=99, shift=(12, 9) if t == 3 else (0, 0)) for t in range(7)]
    assert field_shift(seq[2], seq[3]) == (12, 9)
    events = feed(seq)
    assert all(v["decision"] == "no_division" for v in events.values()), events


def test_vanished_loss_interval_starts_where_it_went_missing():
    seq = [scene(t, divide_at=99) for t in range(8)]
    for t in range(4, 8):
        seq[t] = np.where(seq[t] == 2, 0, seq[t])  # founder at y=60,x=20..40 disappears at frame 4
    events = feed(seq)
    lost = [v for v in events.values() if v["decision"] == "lost"]
    assert len(lost) == 1
    assert (lost[0]["lower_frame"], lost[0]["upper_frame"], lost[0]["tracking_end_reason"]) == (3, 4, "vanished")


def test_unusable_trailing_frames_censor_instead_of_no_division():
    seq = [scene(t, divide_at=99) for t in range(8)]
    events = feed(seq, skip={6, 7})
    assert all(v["decision"] == "lost" and v["tracking_end_reason"] == "unusable_frames" for v in events.values())


def test_watershed_splits_touching_daughters_at_a_bright_septum():
    rng = np.random.default_rng(0)
    image = 1000 + rng.normal(0, 5, (80, 120))
    image[36:45, 20:60] -= 600   # daughter 1 interior
    image[36:45, 62:100] -= 600  # daughter 2 interior
    image[36:45, 60:62] -= 250   # septum: still dark enough to join the foreground, brighter than cells
    image = np.clip(image, 0, 65535).astype(np.uint16)
    params = {"threshold": 150, "background_sigma_px": 20, "min_area_px": 25, "h_rel": 1.0, "min_piece_px": 100}
    plain = ThresholdSegmenter().segment(image, context={}, parameters={**params, "max_area_px": 10**6})
    split = WatershedSegmenter().segment(image, context={}, parameters=params)
    assert plain.max() == 1
    assert split.max() == 2
    assert {int(split[40, 30]), int(split[40, 90])} == {1, 2}


def test_watershed_keeps_large_blobs_and_rejects_bright_polarity():
    image = np.full((200, 200), 1000, np.uint16)
    image[20:180, 20:180] = 300  # one huge dark object, far over the old 1500 px cap
    labels = WatershedSegmenter().segment(image, context={}, parameters={"threshold": 100, "background_sigma_px": 40})
    assert (labels > 0).sum() > 20000  # kept (possibly split), never deleted for being large
    with pytest.raises(ValueError):
        WatershedSegmenter().segment(image, context={}, parameters={"polarity": "bright"})


def test_pipeline_colony_mode_on_demo(tmp_path):
    project = create(tmp_path / "demo")
    project.config["segmentation"].update(adapter="watershed", threshold_from_first_frame=False)
    project.config["tracking"].update(events="colony", colony={"min_colony_growth": 0.0})
    result = run(project)
    assert result["founder_events"] and result["segmenter_class"].endswith("WatershedSegmenter")
    rows = summarize(project, materialize(project, result["run_id"], []))
    import json

    truth = {t["field"]: t for t in json.loads((tmp_path / "demo" / "synthetic_truth.json").read_text())}
    for field, t in truth.items():
        mine = [r for r in rows if r["field"] == field]
        divided = [r for r in mine if r["decision"] == "division"]
        assert len(divided) == 1
        assert [divided[0]["lower_frame"], divided[0]["upper_frame"]] == t["division_interval"]
        lost = [r for r in mine if r["decision"] == "lost"]
        assert len(lost) == 1 and [lost[0]["lower_frame"], lost[0]["upper_frame"]] == t["loss_interval"]
        assert sum(r["decision"] == "no_division" for r in mine) == 1
        # outlines for the viewer: every colony object is reachable
        assert len(divided[0]["observation_ids"]) > len(set(n.split("-o")[0] for n in divided[0]["observation_ids"]))


def test_threshold_from_first_frame_is_recorded(tmp_path):
    project = create(tmp_path / "demo")
    project.config["segmentation"].update(adapter="watershed", threshold_from_first_frame=True, threshold=None)
    result = run(project, fields=["1"])
    assert set(result["field_thresholds"]) == {"1"} and result["field_thresholds"]["1"] > 0


def test_split_only_in_the_last_frame_is_unconfirmed_not_a_division():
    seq = [scene(t, divide_at=5) for t in range(6)]  # daughters appear only in the final frame
    a = feed(seq, {"min_colony_growth": 1.0})["f0-t0-o1"]
    assert a["decision"] == "lost" and a["tracking_end_reason"] == "unconfirmed_division_candidate"
    assert (a["lower_frame"], a["upper_frame"]) == (4, 5)


def test_division_lower_bound_is_the_last_observed_frame():
    seq = [scene(t, divide_at=4) for t in range(8)]
    a = feed(seq, {"min_colony_growth": 1.0}, skip={3})["f0-t0-o1"]
    assert a["decision"] == "division" and (a["lower_frame"], a["upper_frame"]) == (2, 4)
    assert a["contains_gap"]


def test_skipped_frame_keeps_static_contact_splitting():
    seq = [scene(t, divide_at=99, fuse_at=3) for t in range(7)]
    events = feed(seq, skip={2})
    touched = [v for v in events.values() if "touched_neighbour" in v["flags"]]
    assert len(touched) == 2 and not any(v["tracking_end_reason"] == "fused" for v in events.values())


def _colony_project(tmp_path):
    project = create(tmp_path / "demo")
    project.config["segmentation"].update(adapter="watershed", threshold_from_first_frame=False)
    project.config["tracking"].update(events="colony", colony={"min_colony_growth": 0.0})
    return project, run(project, fields=["1"])


def test_gap_or_late_frame_exclusion_never_yields_a_stale_or_full_window_call(tmp_path):
    project, result = _colony_project(tmp_path)
    rid = result["run_id"]
    rows = summarize(project, materialize(project, rid, []))
    assert sorted(r["decision"] for r in rows) == ["division", "lost", "no_division"]
    # excluding a frame in review after the run invalidates the cached automatic calls that used it
    op = {"type": "set_frame_quality", "field": "1", "frame": 2, "channel": "phase", "usable": False, "reason": "blur"}
    later = summarize(project, materialize(project, rid, [op]))
    assert all(r["decision"] == "lost" for r in later)
    assert all(r["tracking_end_reason"] in {"frames_excluded_after_run", "baseline_unusable"} for r in later)
    # a frame excluded before the run is a gap: no full-window "no division" call
    project.config["quality"]["excluded_structure_frames"] = {"1": [2]}
    gapped = run(project, fields=["1"])
    rows = summarize(project, materialize(project, gapped["run_id"], []))
    assert "no_division" not in [r["decision"] for r in rows]
    assert any(r["tracking_end_reason"] == "incomplete_window" for r in rows)


def test_run_reports_progress_and_status(tmp_path):
    project = create(tmp_path / "demo")
    project.config["field_workers"] = 2
    result = run(project)
    directory = project.run_path(result["run_id"])
    import json

    assert json.loads((directory / "status.json").read_text())["state"] == "complete"
    progress = [json.loads(f.read_text()) for f in sorted((directory / "progress").glob("*.json"))]
    assert len(progress) == 4 and all(p["done"] == p["total"] for p in progress)


def test_unguarded_launch_script_fails_fast_instead_of_hanging(tmp_path):
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "noguard.py"  # no `if __name__ == "__main__":` guard, parallel fields
    script.write_text(textwrap.dedent(f'''
        import sys; sys.path.insert(0, {str(__import__("pathlib").Path(__file__).parents[2])!r})
        from microscopy.demo import create
        from microscopy.pipeline import run
        p = create({str(tmp_path / "demo")!r})
        p.config["field_workers"] = 2
        run(p)
    '''))
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "__main__" in done.stderr and "field worker process died" in done.stderr


def test_resume_reuses_saved_frames_and_matches_a_fresh_run(tmp_path, monkeypatch):
    import shutil

    project = create(tmp_path / "demo")
    fresh = run(project)
    # simulate a run that died part-way: drop its result and half its label planes
    partial = project.run_path(fresh["run_id"])
    (partial / "run.json").unlink()
    planes = sorted((partial / "labels").glob("*.npz"))
    for f in planes[len(planes) // 2:]:
        f.unlink()
    calls = []
    original = ThresholdSegmenter.segment
    monkeypatch.setattr(ThresholdSegmenter, "segment", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    resumed = run(project, resume_from=fresh["run_id"])
    assert 0 < len(calls) < len(planes)  # only the missing frames were segmented again
    for key in ["founders", "edges", "quality"]:
        assert resumed[key] == fresh[key]
    strip = lambda obs: [{k: v for k, v in o.items() if k != "labels_file"} for o in obs]
    assert strip(resumed["observations"]) == strip(fresh["observations"])
    assert resumed["resumed_from"] == fresh["run_id"]
    # different segmentation settings must not reuse anything
    project.config["segmentation"]["threshold"] = 99
    with pytest.raises(ValueError, match="different segmentation settings"):
        run(project, resume_from=fresh["run_id"])


def test_raw_folder_moved_behind_a_symlink_still_reads(tmp_path):
    import os
    from microscopy.images import read_frame
    from microscopy.storage import Project

    project = create(tmp_path / "demo")
    raw = project.raw
    moved = tmp_path / "moved_raw"
    os.rename(raw, moved)
    os.symlink(moved, raw)  # data relocated, link left at the recorded raw_root
    again = Project(project.root)
    key = next(iter(again.frames))
    assert read_frame(again, *key).shape == tuple(again.frames[key]["shape"])
    record = again.frames[key]
    record["path"] = "../../etc/passwd"  # escaping the acquisition folder is still refused
    with pytest.raises(ValueError, match="outside acquisition"):
        read_frame(again, *key)


def test_colony_run_stores_only_founder_colonies_and_still_measures(tmp_path):
    from microscopy.analysis import analyze

    project, result = _colony_project(tmp_path)
    used = {i for e in result["founder_events"].values() for i in e["observation_ids"]}
    used |= {f["observation_id"] for f in result["founders"]}
    assert {o["id"] for o in result["observations"]} == used
    assert result["graph_scope"] == "founder_colonies"
    frames = {(o["field"], o["frame"]) for o in result["observations"]}
    assert all(f"{f}|{t}" in result["label_planes"] for f, t in frames)
    out = analyze(project, result["run_id"])  # measurements and plots work on the compact graph
    assert out


def test_run_stops_cleanly_when_disk_runs_low(tmp_path, monkeypatch):
    import microscopy.pipeline as pipeline

    project = create(tmp_path / "demo")
    monkeypatch.setattr(pipeline, "free_disk_gb", lambda path: 1.0)
    with pytest.raises(RuntimeError, match="Not enough free disk"):
        run(project)


def test_oversized_run_file_is_refused_before_loading(tmp_path):
    from microscopy.storage import check_loadable

    path = tmp_path / "run.json"
    path.write_text("{}" + " " * 1000)
    check_loadable(path)  # small: fine
    with pytest.raises(MemoryError, match="needs about"):
        check_loadable(path, expansion=10**9)

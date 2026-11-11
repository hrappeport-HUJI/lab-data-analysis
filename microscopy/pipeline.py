"""Algorithm runs are immutable; later corrections are independent review revisions."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from collections import deque
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy import ndimage
from . import __version__
from .adapters import get_adapter
from .colony import ColonyEvents
from .images import read_frame, quality
from .storage import Project, utc, write_json, file_digest
from importlib.metadata import version, PackageNotFoundError
import platform
import shutil
import os


STATELESS_SEGMENTERS = {"ThresholdSegmenter", "WatershedSegmenter"}

# Resource guards: a run must never fill the disk or starve the machine.
MIN_FREE_DISK_GB = 3.0  # a running job stops (resumably) below this
LABEL_PLANE_MB = 0.6  # generous per-frame estimate for compressed label planes
WORKER_NICE = 10  # field worker processes run at lower priority so the computer stays usable


def free_disk_gb(path):
    return shutil.disk_usage(path).free / 2**30


def check_disk(path, need_gb=0.0):
    free = free_disk_gb(path)
    if free < MIN_FREE_DISK_GB + need_gb:
        raise RuntimeError(
            f"Not enough free disk: {free:.1f} GB free, this needs about {need_gb:.1f} GB plus a "
            f"{MIN_FREE_DISK_GB:.0f} GB safety margin. Free space, then resume with resume_from."
        )


def validate_labels(labels, shape):
    labels = np.asarray(labels)
    if (
        labels.shape != tuple(shape)
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
    ):
        raise ValueError(
            "Segmenter must return nonnegative integer labels matching Y,X shape"
        )
    if labels.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Label IDs exceed supported range")
    return labels.astype(np.int32)


def validate_graph(observations, edges):
    byid = {r["id"]: r for r in observations}
    if len(byid) != len(observations):
        raise ValueError("Observation IDs must be unique")
    parents, children = {}, {}
    seen = set()
    for a, b in edges:
        if (a, b) in seen or a not in byid or b not in byid:
            raise ValueError("Invalid or duplicate tracking link")
        seen.add((a, b))
        if byid[a]["field"] != byid[b]["field"] or byid[a]["frame"] >= byid[b]["frame"]:
            raise ValueError("Tracking links must move forward within one field")
        parents.setdefault(b, []).append(a)
        children.setdefault(a, []).append(b)
    if any(len(p) > 1 for p in parents.values()) or any(
        len(c) > 2 for c in children.values()
    ):
        raise ValueError(
            "Merges or >2 daughters are ambiguous, not supported lineage links"
        )
    return byid, parents, children


def _process_field(job):
    """One field, start to finish. Module level so fields can run in separate processes."""
    p, cfg, segmenter, tracker, baseline_only, directory, n_selected, fi, field, reuse = job
    if __import__("multiprocessing").current_process().name != "MainProcess":
        try:
            os.nice(int(cfg.get("worker_nice", WORKER_NICE)))
        except OSError:
            pass
    sp, tp = cfg.get("segmentation", {}), cfg.get("tracking", {})
    structural = cfg["structure_channel"]
    observations, edges, founders, qc = [], [], [], []
    records = sorted(
        (
            r
            for r in p.manifest
            if r["field"] == field and r["channel"] == structural
        ),
        key=lambda r: r["frame"],
    )
    if baseline_only:
        records = records[:1]
    previous, previous_labels, last_frame = [], None, None
    baseline = None
    fsp = sp
    threshold_used = None
    if sp.get("threshold_from_first_frame") and field not in sp.get("threshold_by_field", {}):
        # One threshold per field, fixed at the first frame: an automatic threshold that rises
        # as colonies fill the field drops faint, non-growing cells late in the recording.
        from .adapters.watershed import auto_threshold

        first = read_frame(p, field, records[0]["frame"], structural)
        threshold_used = auto_threshold(first, float(sp.get("background_sigma_px", 8)))
        fsp = {**sp, "threshold_by_field": {**sp.get("threshold_by_field", {}), field: threshold_used}}
    colony = ColonyEvents(tp.get("colony")) if tp.get("events") == "colony" else None
    colony_started = False

    def prepare(record):
        frame = record["frame"]
        q = {"field": field, "frame": frame, "channel": structural}
        if frame in cfg.get("quality", {}).get(
            "excluded_structure_frames", {}
        ).get(field, []):
            return (
                record,
                None,
                {**q, "usable": False, "reason": "explicit_image_exclusion"},
                None,
            )
        image = read_frame(p, field, frame, structural)
        q.update(quality(image))
        minimum = cfg.get("quality", {}).get("min_focus_score")
        q["usable"] = minimum is None or q["focus_score"] >= minimum
        q["reason"] = "" if q["usable"] else "focus_below_configured_threshold"
        saved = reuse / f"{frame:09}.npz" if reuse is not None else None
        if q["usable"] and saved is not None and saved.exists():
            # Resuming: this frame was segmented by an earlier, identically configured run.
            with np.load(saved) as stored:
                labels = validate_labels(stored["labels"], image.shape)
        else:
            labels = (
                validate_labels(
                    segmenter.segment(image, context=record, parameters=fsp),
                    image.shape,
                )
                if q["usable"]
                else None
            )
        return record, image, q, labels

    def prepared():
        workers = int(sp.get("workers", 1))
        if not 1 <= workers <= 8:
            raise ValueError("Segmentation workers must be 1–8")
        if workers > 1 and type(segmenter).__name__ not in STATELESS_SEGMENTERS:
            raise ValueError(
                "Parallel segmentation currently requires the stateless reference segmenter"
            )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            source = iter(records)
            pending = deque()
            for _ in range(workers):
                rec = next(source, None)
                if rec is not None:
                    pending.append(pool.submit(prepare, rec))
            while pending:
                result = pending.popleft().result()
                rec = next(source, None)
                if rec is not None:
                    pending.append(pool.submit(prepare, rec))
                yield result

    for record, image, q, labels in prepared():
        frame = record["frame"]
        qc.append(q)
        if not q["usable"]:
            previous, previous_labels, last_frame = [], None, frame
            if colony_started:
                colony.skip(frame)
            continue
        label_file = f"labels/{fi:04}_{frame:09}.npz"
        (directory / "labels").mkdir(exist_ok=True)
        check_disk(directory)  # stop cleanly (resumable) before the disk fills
        np.savez_compressed(
            directory / label_file,
            labels=labels.astype(np.uint16) if labels.max(initial=0) < 65536 else labels,
        )
        current = []
        for label, region in enumerate(ndimage.find_objects(labels), 1):
            if region is None:
                continue
            yy, xx = np.where(labels[region] == label)
            yy += region[0].start
            xx += region[1].start
            oid = f"f{fi}-t{frame}-o{label}"
            node = {
                "id": oid,
                "field": field,
                "frame": frame,
                "time_s": record["time_s"],
                "label": int(label),
                "labels_file": label_file,
                "x": float(xx.mean()),
                "y": float(yy.mean()),
                "area_px": int(len(xx)),
                "bbox": [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max()) + 1,
                    int(yy.max()) + 1,
                ],
            }
            current.append(node)
        can_start = frame == records[0]["frame"] or cfg.get("founders", {}).get(
            "allow_delayed_start", False
        )
        if baseline is None and can_start:
            baseline = frame
            margin = cfg.get("founders", {}).get("edge_margin_px", 3)
            radius = cfg.get("founders", {}).get("isolation_px", 0)
            for node in current:
                x0, y0, x1, y1 = node["bbox"]
                reason = (
                    "edge"
                    if min(x0, y0, image.shape[1] - x1, image.shape[0] - y1)
                    < margin
                    else ""
                )
                if radius > 0:
                    mask = labels == node["label"]
                    nearby = ndimage.binary_dilation(
                        mask, iterations=int(radius)
                    )
                    if np.any(nearby & (labels > 0) & ~mask):
                        reason = "not_isolated"
                founders.append(
                    {
                        "id": node["id"],
                        "observation_id": node["id"],
                        "field": field,
                        "baseline_frame": frame,
                        "delayed_start": frame != records[0]["frame"],
                        "excluded_reason": reason,
                    }
                )
        if colony is not None:
            if colony_started:
                colony.update(frame, labels, current)
            elif baseline == frame:
                colony.start(frame, labels, current, founders)
                colony_started = True
        step = int(cfg.get("expected_frame_step", 1))
        if previous and last_frame is not None and frame == last_frame + step:
            edges.extend(
                tracker.link(
                    previous,
                    current,
                    previous_labels=previous_labels,
                    current_labels=labels,
                    parameters=tp,
                )
            )
        observations.extend(current)
        # Heartbeat: lets a person or agent see that a long run is progressing (or has stalled).
        write_json(directory / "progress" / f"{fi:04}.json",
                   {"field": field, "frame": frame, "done": len(qc), "total": len(records), "updated": utc()})
        previous, previous_labels, last_frame = current, labels, frame
    print(f"Processed field {field} ({fi+1}/{n_selected})", flush=True)
    events = colony.finish(records[-1]["frame"]) if colony_started else {}
    planes = {}
    if colony_started:
        # Keep only what the founder events refer to (plus every baseline object). Storing every
        # object of every frame made one 24-field run.json 5.9 GB, too large to load in 16 GB RAM.
        # Label planes stay on disk and are listed so measurements still see all cells.
        planes = {f"{o['field']}|{o['frame']}": o["labels_file"] for o in observations}
        keep = {f["observation_id"] for f in founders}
        for event in events.values():
            keep.update(event["observation_ids"])
        observations = [o for o in observations if o["id"] in keep]
        edges = [e for e in edges if e[0] in keep and e[1] in keep]
    return observations, edges, founders, qc, events, threshold_used, planes


def _reusable_labels(p, resume_from, cfg, segmenter, selected):
    """Map field -> labels path prefix in an earlier run with identical segmentation settings."""
    started = p.run_path(resume_from) / "started.json"
    if not started.exists():
        raise ValueError(f"Run {resume_from} has no started.json; it cannot be resumed")
    earlier = json.loads(started.read_text())

    def settings(c):
        seg = {k: v for k, v in c.get("segmentation", {}).items() if k not in {"workers"}}
        return seg, c["structure_channel"], c.get("quality", {})

    if settings(earlier["config"]) != settings(cfg) or earlier["segmenter_class"] != _class(segmenter):
        raise ValueError(f"Run {resume_from} used different segmentation settings; start a fresh run")
    labels = p.run_path(resume_from) / "labels"
    index = {field: i for i, field in enumerate(earlier["fields"])}
    return {f: _Prefix(labels, index[f]) for f in selected if f in index}


class _Prefix:
    """labels/<fi>_<frame>.npz of an earlier run, addressable as prefix / f"{frame:09}.npz"."""

    def __init__(self, folder, fi):
        self.folder, self.fi = Path(folder), fi

    def __truediv__(self, name):
        return self.folder / f"{self.fi:04}_{name}"


def _class(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def run(project, *, segmenter=None, tracker=None, fields=None, baseline_only=False, resume_from=None):
    """Run segmentation and tracking. ``resume_from``: id of an earlier run (usually one that failed
    part-way) whose saved label planes are reused when its segmentation settings are identical."""
    p = project if isinstance(project, Project) else Project(project)
    cfg = json.loads(json.dumps(p.config))
    cfg["analysis_scope"] = "baseline_only" if baseline_only else "time_series"
    sp, tp = cfg.get("segmentation", {}), cfg.get("tracking", {})
    segmenter = segmenter or get_adapter("segmenter", sp.get("adapter", "threshold"))
    tracker = tracker or get_adapter("tracker", tp.get("adapter", "overlap"))
    run_id = p.new_id("run")
    directory = p.run_path(run_id)
    directory.mkdir(parents=True)
    observations, edges, founders, qc = [], [], [], []
    structural = cfg["structure_channel"]
    selected = sorted({r["field"] for r in p.manifest if r["channel"] == structural})
    if fields is not None:
        unknown = set(fields) - set(selected)
        if unknown:
            raise ValueError(f"Unknown pilot fields: {unknown}")
        selected = [f for f in selected if f in fields]
    reuse = _reusable_labels(p, resume_from, cfg, segmenter, selected) if resume_from else {}
    n_frames = sum(1 for r in p.manifest if r["channel"] == structural and r["field"] in selected)
    check_disk(directory, need_gb=n_frames * LABEL_PLANE_MB / 1024)
    write_json(directory / "started.json", {"config": cfg, "fields": selected, "segmenter_class": _class(segmenter),
                                            "resumed_from": resume_from, "created_at": utc()})
    try:


        field_workers = int(cfg.get("field_workers", 1))
        if not 1 <= field_workers <= 8:
            raise ValueError("field_workers must be 1–8")
        if field_workers > 1 and (
            type(segmenter).__name__ not in STATELESS_SEGMENTERS
            or type(tracker).__name__ not in {"OverlapTracker", "CentroidTracker"}
        ):
            raise ValueError(
                "Parallel fields currently require stateless reference adapters"
            )
        jobs = [
            (p, cfg, segmenter, tracker, baseline_only, directory, len(selected), fi, field, reuse.get(field))
            for fi, field in enumerate(selected)
        ]
        # Separate processes when parallel: the per-frame bookkeeping is Python code that threads
        # cannot run concurrently. Built-in adapters are stateless and picklable.
        Pool = ProcessPoolExecutor if field_workers > 1 else ThreadPoolExecutor
        (directory / "progress").mkdir(exist_ok=True)
        write_json(directory / "status.json", {"state": "running", "pid": os.getpid(), "fields": len(selected),
                                               "parallel": Pool.__name__, "started": utc()})
        field_pool = Pool(max_workers=field_workers)
        failed = True
        try:
            founder_events, field_thresholds = {}, {}
            label_planes = {}
            for (obs, links, roots, checks, events, used, planes), field in zip(
                field_pool.map(_process_field, jobs), selected
            ):
                founder_events.update(events)
                label_planes.update(planes)
                if used is not None:
                    field_thresholds[field] = used
                observations.extend(obs)
                edges.extend(links)
                founders.extend(roots)
                qc.extend(checks)
            failed = False
        except BrokenProcessPool as e:
            raise RuntimeError(
                "A field worker process died. If you started this run from your own script, put the "
                "call inside `if __name__ == \"__main__\":` (worker processes re-import the script), "
                "or set field_workers to 1."
            ) from e
        finally:
            # On failure do not wait for workers that may never finish (e.g. a worker that could not
            # start because the launching script lacks an `if __name__ == "__main__":` guard).
            field_pool.shutdown(wait=not failed, cancel_futures=failed)
        validate_graph(observations, edges)
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        dependencies = {}
        for name in ["numpy", "scipy", "pandas", "tifffile", "Pillow", "matplotlib", "scikit-image"]:
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = None
        source_hashes = {
            str(path.relative_to(Path(__file__).parent)): file_digest(path)
            for path in Path(__file__).parent.rglob("*.py")
            if "tests" not in path.parts
        }
        result = {
            "python_version": platform.python_version(),
            "dependencies": dependencies,
            "source_sha256": source_hashes,
            "schema_version": 1,
            "run_id": run_id,
            "manifest_digest": cfg["manifest_digest"],
            "created_at": utc(),
            "package_version": __version__,
            "code_commit": commit,
            "config": cfg,
            "segmenter_class": type(segmenter).__module__
            + "."
            + type(segmenter).__qualname__,
            "tracker_class": type(tracker).__module__
            + "."
            + type(tracker).__qualname__,
            "fields": selected,
            "observations": observations,
            "edges": [list(e) for e in edges],
            "founders": founders,
            "quality": qc,
        }
        if tp.get("events") == "colony":
            result["founder_events"] = founder_events
            result["label_planes"] = label_planes
            result["graph_scope"] = "founder_colonies"
        if field_thresholds:
            result["field_thresholds"] = field_thresholds
        if resume_from:
            result["resumed_from"] = resume_from
        write_json(directory / "run.json", result)
        write_json(directory / "status.json", {"state": "complete", "ended": utc()})
        write_json(p.root / "latest_run.json", {"run_id": run_id})
        return result
    except Exception as e:
        write_json(directory / "failed.json", {"error": str(e), "created_at": utc()})
        write_json(directory / "status.json", {"state": "failed", "error": str(e)[:500], "ended": utc()})
        raise

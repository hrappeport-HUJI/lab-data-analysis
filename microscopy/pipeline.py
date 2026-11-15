"""Algorithm runs are immutable; later corrections are independent review revisions."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from collections import deque
from functools import partial
import tempfile
import time
import multiprocessing
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
from .system import lower_priority
from importlib.metadata import version, PackageNotFoundError
import platform
import shutil
import os


STATELESS_SEGMENTERS = {"ThresholdSegmenter", "WatershedSegmenter"}

# Resource guards: a run must never fill the disk or starve the machine.
MIN_FREE_DISK_GB = 3.0  # a running job stops (resumably) below this
LABEL_PLANE_MB = 1.2  # per-frame budget for compressed labels, nodes and atomic temporary files
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


def _field_plan(p, cfg, baseline_only, field):
    sp = cfg.get("segmentation", {})
    structural = cfg["structure_channel"]
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
    fsp = sp
    threshold_used = None
    if sp.get("threshold_from_first_frame") and field not in sp.get("threshold_by_field", {}):
        # One threshold per field, fixed at the first frame: an automatic threshold that rises
        # as colonies fill the field drops faint, non-growing cells late in the recording.
        from .adapters.watershed import auto_threshold

        first = read_frame(p, field, records[0]["frame"], structural)
        threshold_used = auto_threshold(first, float(sp.get("background_sigma_px", 8)))
        fsp = {**sp, "threshold_by_field": {**sp.get("threshold_by_field", {}), field: threshold_used}}
    return {"field": field, "records": records, "parameters": fsp,
            "threshold_used": threshold_used}


_WORKER = None
_NATIVE_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _initialize_worker(project_root, cfg, segmenter, directory):
    global _WORKER
    # Also retain the inherited caps for libraries imported lazily in a worker.
    for name in _NATIVE_THREAD_VARS:
        os.environ[name] = "1"
    lower_priority(cfg.get("worker_nice", WORKER_NICE))
    _WORKER = Project(project_root), cfg, segmenter, Path(directory)


def _bounded_results(pool, function, jobs, limit):
    """Consume completions, keeping at most limit jobs (and their metadata) outstanding."""
    source = iter(jobs)
    pending = set()
    def fill():
        while len(pending) < limit:
            job = next(source, None)
            if job is None:
                break
            pending.add(pool.submit(function, job))
    fill()
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
        fill()


def _frame_jobs(plans, reuse):
    queues = deque((fi, plan, iter(plan["records"])) for fi, plan in enumerate(plans))
    while queues:
        fi, plan, records = queues.popleft()
        record = next(records, None)
        if record is not None:
            yield fi, plan["field"], record, plan["parameters"], reuse.get(plan["field"])
            queues.append((fi, plan, records))


def _sidecar(directory, fi, frame):
    return directory / "prepared" / f"{fi:04}_{frame:09}.json"


def _prepare_frame(job, *, state=None):
    p, cfg, segmenter, directory = state if state is not None else _WORKER
    fi, field, record, fsp, reuse = job
    structural = cfg["structure_channel"]
    frame = record["frame"]
    q = {"field": field, "frame": frame, "channel": structural}
    if frame in cfg.get("quality", {}).get(
        "excluded_structure_frames", {}
    ).get(field, []):
        q.update(usable=False, reason="explicit_image_exclusion")
        write_json(_sidecar(directory, fi, frame),
                   {"schema_version": 1, "quality": q, "shape": None, "nodes": []})
        return fi, frame
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

    label_file = f"labels/{fi:04}_{frame:09}.npz"
    current = []
    if labels is not None:
        check_disk(directory)
        fd, temporary = tempfile.mkstemp(prefix=".labels-", suffix=".npz", dir=directory / "labels")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, labels=labels.astype(np.uint16)
                                    if labels.max(initial=0) < 65536 else labels)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / label_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
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
    # These sidecars are produced afresh for this run, never trusted across resumes.
    # A completion marker is published only after its atomic label-plane replacement.
    write_json(_sidecar(directory, fi, frame),
               {"schema_version": 1, "quality": q, "shape": list(image.shape), "nodes": current})
    return fi, frame


def _follow_field(job):
    cfg, tracker, directory, n_selected, fi, plan = job
    field, records = plan["field"], plan["records"]
    tp = cfg.get("tracking", {})
    observations, edges, founders, qc = [], [], [], []
    previous, previous_labels, last_frame = [], None, None
    baseline = None
    threshold_used = plan["threshold_used"]
    colony = ColonyEvents(tp.get("colony")) if tp.get("events") == "colony" else None
    colony_started = False

    for record in records:
        frame = record["frame"]
        prepared = json.loads(_sidecar(directory, fi, frame).read_text())
        if prepared["schema_version"] != 1:
            raise ValueError("Unknown prepared-frame schema")
        q, current = prepared["quality"], prepared["nodes"]
        qc.append(q)
        if not q["usable"]:
            previous, previous_labels, last_frame = [], None, frame
            if colony_started:
                colony.skip(frame)
            continue
        with np.load(directory / f"labels/{fi:04}_{frame:09}.npz") as stored:
            labels = validate_labels(stored["labels"], prepared["shape"])
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
                    if min(x0, y0, labels.shape[1] - x1, labels.shape[0] - y1)
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
    # Publish a final replay count even when the last (or every) frame was excluded.
    write_json(directory / "progress" / f"{fi:04}.json",
               {"field": field, "frame": records[-1]["frame"], "done": len(qc),
                "total": len(records), "updated": utc()})
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
    return fi, (observations, edges, founders, qc, events, threshold_used, planes)


def _reusable_labels(p, resume_from, cfg, segmenter, selected):
    """Map field -> labels path prefix in an earlier run with identical segmentation settings."""
    started = p.run_path(resume_from) / "started.json"
    if not started.exists():
        raise ValueError(f"Run {resume_from} has no started.json; it cannot be resumed")
    earlier = json.loads(started.read_text())

    def settings(c):
        seg = {k: v for k, v in c.get("segmentation", {}).items() if k not in {"workers"}}
        return seg, c["structure_channel"], c.get("quality", {}), c.get("manifest_digest")

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
            raise ValueError("field_workers must be 1-8")
        if field_workers > 1 and (
            type(segmenter).__name__ not in STATELESS_SEGMENTERS
            or type(tracker).__name__ not in {"OverlapTracker", "CentroidTracker"}
        ):
            raise ValueError(
                "Parallel fields currently require stateless reference adapters"
            )
        frame_workers = int(cfg.get("frame_workers", max(field_workers, int(sp.get("workers", 1)))))
        if not 1 <= frame_workers <= 8 or not 1 <= int(sp.get("workers", 1)) <= 8:
            raise ValueError("Segmentation workers must be 1–8")
        workers = frame_workers
        if workers > 1 and (type(segmenter).__name__ not in STATELESS_SEGMENTERS
                            or type(tracker).__name__ not in {"OverlapTracker", "CentroidTracker"}):
            raise ValueError("Parallel segmentation requires stateless reference adapters")
        plans = [_field_plan(p, cfg, baseline_only, field) for field in selected]
        write_json(directory / "field_plan.json", plans)
        for folder in ("progress", "labels", "prepared"):
            (directory / folder).mkdir(exist_ok=True)
        write_json(directory / "status.json", {"state": "running", "phase": "segmentation",
                   "pid": os.getpid(), "workers": workers, "started": utc()})
        # A single pool, with a phase barrier. The serial path keeps custom adapters usable.
        Pool = ProcessPoolExecutor if workers > 1 else ThreadPoolExecutor
        kwargs = dict(max_workers=workers)
        if workers > 1:
            kwargs.update(initializer=_initialize_worker, initargs=(str(p.root), cfg, segmenter, directory),
                          mp_context=multiprocessing.get_context("spawn"))
        # Windows spawn inherits these before importing NumPy/SciPy in the child.
        saved_environment = {name: os.environ.get(name) for name in _NATIVE_THREAD_VARS}
        for name in _NATIVE_THREAD_VARS:
            os.environ[name] = "1"
        field_pool = Pool(**kwargs)
        failed = True
        phase_start = time.perf_counter()
        try:
            prepare = (_prepare_frame if workers > 1 else
                       partial(_prepare_frame, state=(p, cfg, segmenter, directory)))
            counts = [0] * len(plans)
            for fi, frame in _bounded_results(field_pool, prepare,
                                              _frame_jobs(plans, reuse), 2 * frame_workers):
                counts[fi] += 1
                write_json(directory / "progress" / f"segmented-{fi:04}.json",
                           {"field": selected[fi], "frame": frame, "segmented": counts[fi],
                            "total": len(plans[fi]["records"]), "updated": utc()})
            segmentation_s = time.perf_counter() - phase_start
            write_json(directory / "status.json", {"state": "running", "phase": "replay",
                       "pid": os.getpid(), "workers": workers, "updated": utc()})
            phase_start = time.perf_counter()
            jobs = [(cfg, tracker, directory, len(selected), fi, plan)
                    for fi, plan in enumerate(plans)]
            results = dict(_bounded_results(field_pool, _follow_field, jobs, min(field_workers, workers)))
            replay_s = time.perf_counter() - phase_start
            founder_events, field_thresholds, label_planes = {}, {}, {}
            for fi, field in enumerate(selected):
                obs, links, roots, checks, events, used, planes = results.pop(fi)
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
                'A field worker process died. Put the call inside `if __name__ == "__main__":` '
                '(worker processes re-import the script), or set worker counts to 1.'
            ) from e
        finally:
            field_pool.shutdown(wait=not failed, cancel_futures=failed)
            for name, value in saved_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        write_json(directory / "timing.json", {"segmentation_s": segmentation_s,
                   "replay_s": replay_s, "workers": workers, "field_workers": field_workers})
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

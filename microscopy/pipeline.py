"""Algorithm runs are immutable; later corrections are independent review revisions."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy import ndimage
from . import __version__
from .adapters import get_adapter
from .images import read_frame, quality
from .storage import Project, utc, write_json, file_digest
from importlib.metadata import version, PackageNotFoundError
import platform


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


def run(project, *, segmenter=None, tracker=None, fields=None, baseline_only=False):
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
    try:

        def process_field(item):
            fi, field = item
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
                labels = (
                    validate_labels(
                        segmenter.segment(image, context=record, parameters=sp),
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
                if workers > 1 and type(segmenter).__name__ != "ThresholdSegmenter":
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
                    continue
                label_file = f"labels/{fi:04}_{frame:09}.npz"
                (directory / "labels").mkdir(exist_ok=True)
                np.savez_compressed(directory / label_file, labels=labels)
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
                previous, previous_labels, last_frame = current, labels, frame
            print(f"Processed field {field} ({fi+1}/{len(selected)})", flush=True)
            return observations, edges, founders, qc

        field_workers = int(cfg.get("field_workers", 1))
        if not 1 <= field_workers <= 8:
            raise ValueError("field_workers must be 1–8")
        if field_workers > 1 and (
            type(segmenter).__name__ != "ThresholdSegmenter"
            or type(tracker).__name__ not in {"OverlapTracker", "CentroidTracker"}
        ):
            raise ValueError(
                "Parallel fields currently require stateless reference adapters"
            )
        with ThreadPoolExecutor(max_workers=field_workers) as field_pool:
            for obs, links, roots, checks in field_pool.map(
                process_field, enumerate(selected)
            ):
                observations.extend(obs)
                edges.extend(links)
                founders.extend(roots)
                qc.extend(checks)
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
        for name in ["numpy", "scipy", "pandas", "tifffile", "Pillow", "matplotlib"]:
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
        write_json(directory / "run.json", result)
        write_json(p.root / "latest_run.json", {"run_id": run_id})
        return result
    except Exception as e:
        write_json(directory / "failed.json", {"error": str(e), "created_at": utc()})
        raise

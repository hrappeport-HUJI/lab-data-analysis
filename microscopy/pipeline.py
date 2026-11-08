"""Algorithm runs are immutable; later corrections are independent review revisions."""

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


def run(project, *, segmenter=None, tracker=None, fields=None):
    p = project if isinstance(project, Project) else Project(project)
    cfg = json.loads(json.dumps(p.config))
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
        for fi, field in enumerate(selected):
            records = sorted(
                (
                    r
                    for r in p.manifest
                    if r["field"] == field and r["channel"] == structural
                ),
                key=lambda r: r["frame"],
            )
            previous, previous_labels, last_frame = [], None, None
            baseline = None
            for record in records:
                frame = record["frame"]
                image = read_frame(p, field, frame, structural)
                q = {
                    "field": field,
                    "frame": frame,
                    "channel": structural,
                    **quality(image),
                }
                minimum = cfg.get("quality", {}).get("min_focus_score")
                q["usable"] = minimum is None or q["focus_score"] >= minimum
                q["reason"] = "" if q["usable"] else "focus_below_configured_threshold"
                qc.append(q)
                if not q["usable"]:
                    previous, previous_labels, last_frame = [], None, frame
                    continue
                labels = validate_labels(
                    segmenter.segment(image, context=record, parameters=sp), image.shape
                )
                label_file = f"labels/{fi:04}_{frame:09}.npy"
                (directory / "labels").mkdir(exist_ok=True)
                np.save(directory / label_file, labels, allow_pickle=False)
                current = []
                for label in np.unique(labels):
                    if label == 0:
                        continue
                    yy, xx = np.where(labels == label)
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

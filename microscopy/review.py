"""Replay review operations without altering algorithm runs or raw data."""

from functools import lru_cache
import copy
import math
import numpy as np
from PIL import Image, ImageDraw
from .pipeline import validate_graph


def polygon_mask(polygon, shape):
    if not isinstance(polygon, list) or len(polygon) < 3 or len(polygon) > 10000:
        raise ValueError("Draw a polygon with at least three vertices")
    points = []
    for point in polygon:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in point
            )
        ):
            raise ValueError("Invalid polygon coordinates")
        x, y = point
        if not 0 <= x < shape[1] or not 0 <= y < shape[0]:
            raise ValueError("Polygon extends outside the image")
        points.append((x, y))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 2, len(points)):
            if i == 0 and j == len(points) - 1:
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            bounds_overlap = max(min(a[0], b[0]), min(c[0], d[0])) <= min(
                max(a[0], b[0]), max(c[0], d[0])
            ) and max(min(a[1], b[1]), min(c[1], d[1])) <= min(
                max(a[1], b[1]), max(c[1], d[1])
            )
            if (
                bounds_overlap
                and cross(a, b, c) * cross(a, b, d) <= 0
                and cross(c, d, a) * cross(c, d, b) <= 0
            ):
                raise ValueError(
                    "Polygon edges cross or touch; redraw a simple outline"
                )
    mask = Image.new("1", (shape[1], shape[0]))
    ImageDraw.Draw(mask).polygon(points, fill=1)
    result = np.asarray(mask, dtype=bool)
    if result.sum() < 3:
        raise ValueError("Outline is too small")
    return result


@lru_cache(maxsize=4)
def _labels(path, modified_ns, size):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            array = loaded["labels"]
    else:
        array = loaded
    array.setflags(write=False)
    return array


def observation_mask(project, run_id, node):
    shape = project.frames[
        (node["field"], node["frame"], project.config["structure_channel"])
    ]["shape"]
    if "polygon" in node:
        return polygon_mask(node["polygon"], shape)
    return label_image(project, run_id, node) == node["label"]


def label_image(project, run_id, node):
    path = project.run_path(run_id) / node["labels_file"]
    if not path.resolve().is_relative_to(project.run_path(run_id)):
        raise ValueError("Mask path outside run")
    stat = path.stat()
    return _labels(str(path), stat.st_mtime_ns, stat.st_size)


def materialize(project, run_id, operations):
    original = project.load_run(run_id)
    data = copy.deepcopy(original)
    nodes = {r["id"]: r for r in data["observations"]}
    founders = {r["id"]: r for r in data["founders"]}
    data.update(
        event_overrides={}, measurement_frames={}, quality_overrides=[], needs_review=[]
    )
    allowed = {
        "set_event",
        "set_mask",
        "split_founder",
        "add_observation",
        "set_links",
        "set_frame_quality",
        "set_measurement_frame",
    }
    for op in operations:
        kind = op.get("type")
        if kind not in allowed:
            raise ValueError("Unknown review operation")
        if kind == "set_event":
            fid = op["founder_id"]
            if fid not in founders or founders[fid].get("split_children"):
                raise ValueError("Review an active founder")
            decision = op["decision"]
            if decision not in {"division", "lost", "no_division", "exclude"}:
                raise ValueError("Unknown decision")
            lower, upper = op.get("lower_frame"), op.get("upper_frame")
            field = founders[fid]["field"]
            valid_frames = {
                r["frame"]
                for r in project.manifest
                if r["field"] == field
                and r["channel"] == project.config["structure_channel"]
            }
            if decision in {"division", "lost"} and (
                type(lower) is not int
                or lower not in valid_frames
                or type(upper) is not int
                or upper not in valid_frames
                or lower >= upper
                or lower < founders[fid]["baseline_frame"]
            ):
                raise ValueError(
                    "Event needs a valid last-certain and later event frame"
                )
            if decision == "no_division" and op.get("full_window_verified") is not True:
                raise ValueError(
                    "Confirm identity throughout the recording for a no-division decision"
                )
            if decision == "no_division" and founders[fid]["delayed_start"]:
                raise ValueError(
                    "Delayed-start founder cannot establish full-window absence of division"
                )
            data["event_overrides"][fid] = copy.deepcopy(op)
        elif kind in {"set_mask", "add_observation"}:
            oid = op["observation_id"]
            if kind == "add_observation":
                if oid in nodes or not oid.startswith("manual-"):
                    raise ValueError(
                        "New observation needs a unique manual- identifier"
                    )
                key = (
                    str(op["field"]),
                    op["frame"],
                    project.config["structure_channel"],
                )
                rec = project.frames[key]
                nodes[oid] = {
                    "id": oid,
                    "field": str(op["field"]),
                    "frame": op["frame"],
                    "time_s": rec["time_s"],
                    "manual": True,
                }
            if oid not in nodes:
                raise ValueError("Unknown observation")
            node = nodes[oid]
            shape = project.frames[
                (node["field"], node["frame"], project.config["structure_channel"])
            ]["shape"]
            mask = polygon_mask(op["polygon"], shape)
            yy, xx = np.where(mask)
            node.update(
                polygon=op["polygon"],
                x=float(xx.mean()),
                y=float(yy.mean()),
                area_px=int(mask.sum()),
                bbox=[
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max()) + 1,
                    int(yy.max()) + 1,
                ],
            )
            data["event_overrides"].clear()
            data["needs_review"].append(
                "Mask edited; existing tracking links were not automatically refitted. Reconfirm event decisions."
            )
        elif kind == "split_founder":
            fid = op["founder_id"]
            if fid not in founders or founders[fid].get("split_children"):
                raise ValueError("Split an unsplit founder")
            if len(op["polygons"]) != 2:
                raise ValueError("Draw exactly two initial cells")
            parent = founders[fid]
            node = nodes[parent["observation_id"]]
            shape = project.frames[
                (node["field"], node["frame"], project.config["structure_channel"])
            ]["shape"]
            masks = [polygon_mask(poly, shape) for poly in op["polygons"]]
            if np.any(masks[0] & masks[1]):
                raise ValueError("Initial cells overlap")
            # Prevent duplicating another active initial founder.
            for other in founders.values():
                if (
                    other["id"] != fid
                    and other["field"] == parent["field"]
                    and other["baseline_frame"] == parent["baseline_frame"]
                    and not other.get("excluded_reason")
                ):
                    occupied = observation_mask(
                        project, run_id, nodes[other["observation_id"]]
                    )
                    if any(np.any(m & occupied) for m in masks):
                        raise ValueError("Split overlaps another founder")
            children = []
            for i, (poly, mask) in enumerate(zip(op["polygons"], masks), 1):
                child = f"{fid}-cell{i}"
                if child in nodes:
                    raise ValueError("Child already exists")
                yy, xx = np.where(mask)
                nodes[child] = {
                    "id": child,
                    "field": node["field"],
                    "frame": node["frame"],
                    "time_s": node["time_s"],
                    "polygon": poly,
                    "x": float(xx.mean()),
                    "y": float(yy.mean()),
                    "area_px": int(mask.sum()),
                    "bbox": [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max()) + 1,
                        int(yy.max()) + 1,
                    ],
                    "manual": True,
                }
                founders[child] = {
                    "id": child,
                    "observation_id": child,
                    "field": parent["field"],
                    "baseline_frame": parent["baseline_frame"],
                    "delayed_start": parent["delayed_start"],
                    "excluded_reason": "",
                    "split_parent": fid,
                }
                children.append(child)
            parent.update(
                split_children=children, excluded_reason="replaced_by_manual_split"
            )
            data["edges"] = [e for e in data["edges"] if e[0] != fid and e[1] != fid]
            data["event_overrides"].clear()
        elif kind == "set_links":
            parent = op["observation_id"]
            children = op["children"]
            if parent not in nodes or not isinstance(children, list):
                raise ValueError("Unknown parent or invalid children")
            data["edges"] = [e for e in data["edges"] if e[0] != parent] + [
                [parent, c] for c in children
            ]
            validate_graph(list(nodes.values()), data["edges"])
            data["event_overrides"].clear()
            data["needs_review"].append(
                "Links edited; preceding event decisions need reconfirmation."
            )
        elif kind == "set_frame_quality":
            key = (str(op["field"]), op["frame"], op["channel"])
            if key not in project.frames or type(op["usable"]) is not bool:
                raise ValueError("Unknown image or invalid quality flag")
            data["quality_overrides"] = [
                r
                for r in data["quality_overrides"]
                if (r["field"], r["frame"], r["channel"]) != key
            ] + [{**op, "field": str(op["field"])}]
            if op["channel"] == project.config["structure_channel"]:
                data["event_overrides"].clear()
        elif kind == "set_measurement_frame":
            fid = op["founder_id"]
            channel = op["channel"]
            frame = op["frame"]
            if (
                fid not in founders
                or (founders[fid]["field"], frame, channel) not in project.frames
            ):
                raise ValueError("Unknown founder/image")
            data["measurement_frames"][f"{fid}|{channel}"] = frame
    data["observations"] = list(nodes.values())
    data["founders"] = list(founders.values())
    validate_graph(data["observations"], data["edges"])
    return data


def save(project, run_id, operation, expected_revision):
    return project.save_review(
        run_id,
        operation,
        expected_revision,
        lambda ops: materialize(project, run_id, ops),
    )

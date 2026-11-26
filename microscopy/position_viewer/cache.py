"""Build immutable position caches from arrays or from a lab-microscopy run."""

import json
from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from ..events import summarize
from ..images import read_frame
from ..review import label_image
from ..storage import Project
from ._format import LEVELS, NATIVE, PIXEL_DTYPE, channel_name, component, overview_pixels, overview_shape


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")), encoding="utf-8")


@contextmanager
def _new_cache(output_dir):
    """Publish a complete cache on the same filesystem, keeping existing caches intact."""
    output = Path(output_dir).absolute()
    if output.exists() and (not output.is_dir() or output.is_symlink() or any(output.iterdir())):
        raise FileExistsError(f"Cache destination must be empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".position-viewer-", dir=output.parent) as staging:
        root = Path(staging) / "cache"
        root.mkdir()
        yield root
        if output.is_dir():
            output.rmdir()  # fails safely if another process populated the destination
        root.rename(output)


def _reader(source):
    return source if callable(source) else source.__getitem__


def _adu(frame, shape):
    frame = np.asarray(frame)
    if frame.shape != shape:
        raise ValueError(f"Expected a frame of shape {shape}, got {frame.shape}")
    if frame.dtype == PIXEL_DTYPE:
        return frame
    if frame.dtype.kind not in "buif" or not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 65535:
        raise ValueError("The viewer requires finite intensities in the uint16 ADU range")
    if frame.dtype.kind == "f":
        frame = np.rint(frame)
    return frame.astype(PIXEL_DTYPE)


def _array_stream(stack, path, dtype, shape):
    stream = stack.enter_context(path.open("wb"))
    np.lib.format.write_array_header_2_0(stream, {
        "descr": dtype.str, "fortran_order": False, "shape": shape,
    })
    return stream


def _write_channel(folder, channel, frames, load, shape, first=None):
    levels = {
        level: shape if level == NATIVE else overview_shape(*shape, level)
        for level in LEVELS
    }
    with ExitStack() as stack:
        streams = {
            level: _array_stream(stack, folder / f"{channel}-{level}.npy", PIXEL_DTYPE,
                                 (len(frames), *size))
            for level, size in levels.items()
        }
        for i, frame in enumerate(frames):
            plane = _adu(first if i == 0 and first is not None else load(frame), shape)
            for level, stream in streams.items():
                stream.write(overview_pixels(plane, levels[level]).tobytes())
            counts = np.bincount(plane.ravel(), minlength=65536)
            values = np.flatnonzero(counts)
            _write_json(folder / f"{channel}-{frame}-hist.json", [values.tolist(), counts[values].tolist()])


def _write_position(root, entry, times, records, channels, overlays, colony_masks=None,
                    tracking=None):
    folder = root / entry["pos"]
    folder.mkdir()
    if len({channel_name(ch).casefold() for ch in channels}) != len(channels):
        raise ValueError("Cache channel names must be unique regardless of case")
    primary = next(iter(channels))
    first_frames, first_loader = channels[primary]
    first = np.asarray(first_loader(first_frames[0]))
    if first.ndim != 2 or not all(first.shape):
        raise ValueError("Expected nonempty 2-D frames")
    height, width = first.shape
    for name, (frames, load) in channels.items():
        _write_channel(folder, name, frames, load, first.shape, first if name == primary else None)

    byte_width = (width + 7) // 8
    empty_mask = bytes(height * byte_width)
    with ExitStack() as stack:
        packed = _array_stream(stack, folder / "colony.npy", np.dtype("u1"),
                               (len(times), height, byte_width))
        for frame in range(len(times)):
            assigned, unassigned, mask = overlays(frame)
            if colony_masks is not None:
                mask = colony_masks(frame)
            if mask is None:
                packed.write(empty_mask)
            else:
                mask = np.asarray(mask)
                if mask.shape != first.shape:
                    raise ValueError("Colony mask and frame dimensions differ")
                packed.write(np.packbits(mask != 0, axis=-1).tobytes())
            _write_json(folder / f"a-{frame}.json", assigned)
            _write_json(folder / f"u-{frame}.json", unassigned)
    _write_json(folder / "channels.json", {name: frames for name, (frames, _) in channels.items()})
    _write_json(folder / "tracking.json", dict(
        tracking or {}, times=times, records=records, height=height, width=width,
        condition=entry["strain"], label=entry["label"],
    ))


def build_position_cache(output_dir, phase_stack_or_loader, *, pos="1", strain="",
                         label=None, n_frames=None, times=None, records=(),
                         phase_channel="Phase", channels=None, colony_masks=None,
                         assigned=None, unassigned=None):
    """Build a one-position cache and return its root.

    The phase source is a stack or a callable taking a zero-based timeline index.
    Supply `times` (elapsed seconds, with None for missing values) or `n_frames`
    for a callable. Additional channels are stacks aligned to the timeline or
    mappings from acquired timeline indices to frames. Colony masks may be a stack
    or a callable. Assigned and unassigned overlays are mappings by frame and use
    the viewer's polygon format: {record_index: [polygon, ...]} and [polygon, ...].
    """
    if n_frames is None:
        if times is not None:
            n_frames = len(times)
        elif callable(phase_stack_or_loader):
            raise ValueError("A phase loader requires times or n_frames")
        else:
            n_frames = len(phase_stack_or_loader)
    frame_ids = list(range(n_frames))
    times = [None] * n_frames if times is None else [None if t is None else float(t) for t in times]
    if not frame_ids or len(times) != n_frames:
        raise ValueError("Provide a nonempty timeline with one time per frame")
    if not callable(phase_stack_or_loader) and len(phase_stack_or_loader) != n_frames:
        raise ValueError("The phase stack must cover the timeline")
    sources = {phase_channel: (frame_ids, _reader(phase_stack_or_loader))}
    for channel, source in (channels or {}).items():
        if channel == phase_channel:
            raise ValueError("The structure channel is already supplied")
        if isinstance(source, Mapping):
            frames = sorted(map(index, source))
        else:
            if len(source) != n_frames:
                raise ValueError(f"Channel {channel!r} needs a full stack or a mapping of acquired frames")
            frames = frame_ids
        if any(frame not in range(n_frames) for frame in frames):
            raise ValueError(f"Channel {channel!r} has frames outside the timeline")
        sources[channel] = (frames, _reader(source))
    entry = _entry(pos, strain, label)

    def polygons(frame):
        a = assigned.get(frame, {}) if assigned is not None else {}
        u = unassigned.get(frame, []) if unassigned is not None else []
        return a, u, None

    with _new_cache(output_dir) as root:
        _write_position(root, entry, times, list(records), sources, polygons,
                        _reader(colony_masks) if colony_masks is not None else None,
                        tracking={"structure_channel": phase_channel})
        _write_json(root / "positions.json", [entry])
    return Path(output_dir).absolute()


def _entry(pos, strain, label=None):
    pos = component(str(pos))
    return {"pos": pos, "strain": strain, "label": label or f"Pos {pos}"}


def _record(row, event, timeline):
    decision = {
        "division": "divided", "lost": "unresolved",
        "exclude": "excluded", "no_division": "not_grown",
    }.get(row["decision"], row["decision"])
    divided = decision == "divided"
    return dict(
        founder_id=row["id"], x=row["initial_x"], y=row["initial_y"],
        decision=decision, reviewed=False, excluded_reason=row.get("excluded_reason"),
        tracking_end_reason=row["tracking_end_reason"], is_doublet=bool(event.get("is_doublet")),
        pi_status=event.get("pi_status"),
        last_single_frame=timeline.get(row["lower_frame"]) if divided else None,
        division_frame=timeline.get(row["upper_frame"]) if divided else None,
        last_valid_frame=timeline.get(row["last_identity_frame"]),
        event_frame=timeline.get(event.get("event_frame", row["upper_frame"])),
        pi_surge_frame=timeline.get(event.get("pi_surge_frame")),
        pi_high_frame=timeline.get(event.get("pi_high_frame")),
        baseline_frame=timeline[row["baseline_frame"]],
    )


def _label_polygons(labels):
    # Bound the object-index array even for externally supplied sparse label IDs.
    if labels.max() > labels.size:
        ids, inverse = np.unique(labels, return_inverse=True)
        # find_objects reserves zero for background, even if the plane has none.
        indexed = (inverse.reshape(labels.shape) + int(ids[0] != 0)).astype(np.int32)
        if ids[0] != 0:
            ids = np.insert(ids, 0, 0)
    else:
        indexed = labels
        ids = None
    for idx, box in enumerate(ndimage.find_objects(indexed), 1):
        if box is None:
            continue
        label = int(ids[idx]) if ids is not None else idx
        ys, xs = box
        cropped = np.pad(indexed[box] == idx, 1)
        polygons = [
            (curve[:, ::-1] + (xs.start - 1, ys.start - 1)).tolist()
            for curve in find_contours(cropped, 0.5)
        ]
        yield label, polygons


def _project_position(root, project, run, field, rows, manifest, nodes):
    run_id = run["run_id"]
    structure = run["config"]["structure_channel"]
    by_channel = defaultdict(dict)
    for item in manifest:
        by_channel[item["channel"]][item["frame"]] = item["time_s"]
    frames = sorted({frame for channel in by_channel.values() for frame in channel})
    structure_times = by_channel[structure]
    # Preserve sparse channel-only acquisitions and prefer the structure timestamp.
    times = [
        structure_times[f] if f in structure_times else next(
            by_channel[ch][f] for ch in by_channel if f in by_channel[ch]
        )
        for f in frames
    ]
    timeline = {frame: idx for idx, frame in enumerate(frames)}
    channels = {}
    for ch in dict.fromkeys((structure, *by_channel)):
        acquired = sorted(timeline[f] for f in by_channel[ch])
        channels[ch] = (acquired, lambda idx, ch=ch: read_frame(project, field, frames[idx], ch))

    records = []
    assignments = defaultdict(lambda: defaultdict(list))
    paths = {n["frame"]: n["labels_file"] for n in nodes.values()}
    for row in rows:
        idx = len(records)
        event = run.get("founder_events", {}).get(row["id"], {})
        records.append(_record(row, event, timeline))
        for oid in row["observation_ids"]:
            node = nodes[oid]
            assignments[node["frame"]][node["label"]].append(idx)

    field_index = run["fields"].index(field)

    def overlays(idx):
        frame = frames[idx]
        file = run.get("label_planes", {}).get(f"{field}|{frame}", paths.get(frame))
        if file is None:
            # Standard runs can have an empty label plane with no observation to reference it.
            candidate = f"labels/{field_index:04}_{frame:09}.npz"
            if not (project.run_path(run_id) / candidate).is_file():
                return {}, [], None
            file = candidate
        labels = label_image(project, run_id, {"labels_file": file})
        assigned, unassigned = defaultdict(list), []
        owners = assignments.get(frame, {})
        for label, polygons in _label_polygons(labels):
            if label in owners:
                for idx in owners[label]:
                    assigned[idx].extend(polygons)
            else:
                unassigned.extend(polygons)
        return assigned, unassigned, labels

    entry = _entry(field, project.config.get("conditions", {}).get(field, ""))
    _write_position(root, entry, times, records, channels, overlays, tracking={
        "run_id": run_id, "frames": frames, "structure_channel": structure,
    })
    return entry


def build_project_viewer_cache(project_dir, output_dir, run_id=None, fields=None):
    """Snapshot a run's algorithm outputs and raw images into a new viewer cache.

    The project and its review database are read-only inputs. The mask layer shows
    the union of the source label plane. Founder assignments use the run's
    canonical event/lineage summary.
    """
    project = Project(project_dir)
    if Path(output_dir).resolve().is_relative_to(project.raw.resolve()):
        raise ValueError("Place the viewer cache outside the raw acquisition")
    run_id = run_id or project.latest_run()
    run = project.load_run(run_id)
    if isinstance(fields, str):
        fields = [fields]
    selected = list(dict.fromkeys(map(str, fields))) if fields is not None else run["fields"]
    if not selected or set(selected) - set(run["fields"]):
        raise ValueError("Select fields present in the run")
    selected_set = set(selected)
    manifest = defaultdict(list)
    for item in project.manifest:
        manifest[item["field"]].append(item)
    subset = {**run, "founders": [f for f in run["founders"] if f["field"] in selected_set],
              "event_overrides": {}, "quality_overrides": []}
    rows = defaultdict(list)
    for row in summarize(project, subset):
        rows[row["field"]].append(row)
    nodes = defaultdict(dict)
    for node in run["observations"]:
        if node["field"] in selected_set:
            nodes[node["field"]][node["id"]] = node
    with _new_cache(output_dir) as root:
        entries = [
            _project_position(root, project, run, field, rows[field], manifest[field], nodes[field])
            for field in selected
        ]
        _write_json(root / "positions.json", entries)
    return Path(output_dir).absolute()

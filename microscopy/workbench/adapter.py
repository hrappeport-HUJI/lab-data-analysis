"""Read model that lets the previous founder-review viewer run on microscopy-package projects.

The previous viewer talked to a ``Project`` object with founder records in its own
vocabulary (``divided``/``unresolved``/``excluded``/``not_grown`` and frame fields such as
``last_single_frame``). This module presents a microscopy run plus its SQLite review
store in that vocabulary. The run, the manifest, the masks and the review database
remain the only sources of truth; nothing here writes to them.

Rendering follows the previous viewer: the server cuts a small window around the
founder and returns a display-sized image, instead of sending full camera frames.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import csv
import io
import json
import sqlite3
import threading

import numpy as np
from PIL import Image
from scipy import ndimage

from ..storage import Project, read_json
from ..images import read_frame
from ..review import materialize, label_image, polygon_mask, save
from ..events import summarize, quality_map

# Previous-viewer decision names <-> review-store decision names.
TO_STORE = {
    "divided": "division",
    "unresolved": "lost",
    "not_grown": "no_division",
    "excluded": "exclude",
}
FROM_STORE = {v: k for k, v in TO_STORE.items()}
FROM_STORE["not_assessed"] = "unresolved"

DEFAULT_WINDOW = 256
OUTPUT_SIZE = 640


class Memo:
    """Small thread-safe LRU that computes each key once, even under concurrent requests."""

    def __init__(self, size):
        self.size = size
        self.items = OrderedDict()
        self.pending = {}
        self.lock = threading.Lock()

    def get(self, key, compute):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
            event = self.pending.get(key)
            owner = event is None
            if owner:
                event = self.pending[key] = threading.Event()
        if not owner:
            event.wait()
            with self.lock:
                if key in self.items:
                    return self.items[key]
            return compute()
        try:
            value = compute()
            with self.lock:
                self.items[key] = value
                while len(self.items) > self.size:
                    self.items.popitem(last=False)
            return value
        finally:
            with self.lock:
                self.pending.pop(key, None)
            event.set()

    def __contains__(self, key):
        with self.lock:
            return key in self.items


class _CachedRunProject(Project):
    """Project whose run JSON is parsed once; ``materialize`` deep-copies it anyway."""

    def __init__(self, root):
        super().__init__(root)
        self._runs = {}

    def load_run(self, run_id):
        if run_id not in self._runs:
            self._runs[run_id] = super().load_run(run_id)
        return self._runs[run_id]


class RunReviewModel:
    def __init__(self, root, run_id=None, read_only=True):
        self.project = _CachedRunProject(root)
        self.root = self.project.root
        self.run_id = run_id or self.project.latest_run()
        self.project.load_run(self.run_id)
        self.read_only = read_only
        cfg = self.project.config
        self.structure = cfg["structure_channel"]
        self.fluorescence = list(cfg.get("fluorescence_channels", []))
        self.lock = threading.RLock()
        self._state = None
        self.arrays = Memo(48)  # raw uint16 frames, ~8 MB each at 2048x2048
        self.signals = Memo(12)  # background-subtracted fluorescence, float32
        self.pngs = Memo(400)  # rendered display images
        self.limits = Memo(4096)  # fixed phase contrast per founder
        self.prefetcher = ThreadPoolExecutor(max_workers=2)
        self.fields = self._field_index()
        self.channel_labels = self._channel_labels()
        self.baseline = self._baseline_measurements()

    # ----- static acquisition metadata -------------------------------------------------
    def _field_index(self):
        fields = {}
        for r in self.project.manifest:
            f = fields.setdefault(r["field"], {"channels": {}, "times": {}, "shape": r["shape"]})
            f["channels"].setdefault(r["channel"], set()).add(r["frame"])
            if r["channel"] == self.structure:
                f["times"][r["frame"]] = r["time_s"]
        for f in fields.values():
            frames = sorted(f["channels"].get(self.structure, []))
            f["frames"] = frames
            f["frame_count"] = (frames[-1] + 1) if frames else 0
        return fields

    def _channel_labels(self):
        labels = {self.structure: "Phase"}
        for m in self.project.config.get("measurements", []):
            name = m.get("name", "")
            if name.startswith("baseline_") and m.get("channel") in self.fluorescence:
                labels.setdefault(m["channel"], name[len("baseline_"):])
        for c in self.fluorescence:
            labels.setdefault(c, c)
        labels.update(self.project.config.get("viewer", {}).get("channel_labels", {}))
        return labels

    def _baseline_measurements(self):
        """Baseline fluorescence per founder from the latest frozen result, for display."""
        pointer = self.root / "latest_results.json"
        if not pointer.exists():
            return {}
        info = read_json(pointer)
        if info.get("run_id") != self.run_id:
            return {}
        path = self.root / "results" / info["result_id"] / "founder_metrics.csv"
        values = {}
        if path.exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["metric"].startswith("baseline_") and row["value"] not in ("", "nan"):
                        values.setdefault(row["founder_id"], {})[row["metric"][9:]] = float(row["value"])
        return values

    # ----- review state, cached per revision ---------------------------------------------
    def _db_revision(self):
        db_path = self.root / "reviews.sqlite"
        if not db_path.exists():
            return 0, []
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        try:
            row = db.execute(
                "SELECT revision,state FROM revisions WHERE run_id=? ORDER BY revision DESC LIMIT 1",
                (self.run_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        finally:
            db.close()
        return (row[0], json.loads(row[1])) if row else (0, [])

    def state(self):
        revision, operations = self._db_revision()
        with self.lock:
            if self._state and self._state["revision"] == revision:
                return self._state
            data = materialize(self.project, self.run_id, operations)
            rows = summarize(self.project, data)
            original = summarize(self.project, {**data, "event_overrides": {}})
            nodes = {n["id"]: n for n in data["observations"]}
            usable = quality_map(data)
            quality_reason = {
                (q["field"], q["frame"], q["channel"]): q.get("reason", "")
                for q in data["quality"] + data["quality_overrides"]
            }
            by_frame = {}
            for row in rows:
                for oid in row["observation_ids"]:
                    by_frame.setdefault((row["id"], nodes[oid]["frame"]), []).append(nodes[oid])
            drift = self._drift(data, nodes)
            records = [
                self._record(row, orig, data["event_overrides"].get(row["id"]))
                for row, orig in zip(rows, original)
            ]
            self._state = {
                "revision": revision,
                "records": records,
                "byid": {r["id"]: r for r in records},
                "nodes_at": by_frame,
                "drift": drift,
                "usable": usable,
                "quality_reason": quality_reason,
                "warnings": list(dict.fromkeys(data["needs_review"])),
            }
            return self._state

    def _drift(self, data, nodes):
        """Cumulative stage drift per field and frame, in raw pixels.

        Estimated as the median displacement of one-to-one tracking links between
        consecutive frames. Plays the role of the previous viewer's drift files: the
        review window follows the stage, not any particular cell, so context after
        tracking loss stays where it was rather than jumping to a neighbour.
        """
        children = {}
        for a, b in data["edges"]:
            children.setdefault(a, []).append(b)
        steps = {}
        for a, kids in children.items():
            if len(kids) != 1:
                continue
            na, nb = nodes[a], nodes[kids[0]]
            if nb["frame"] != na["frame"] + 1:
                continue
            steps.setdefault((na["field"], nb["frame"]), []).append((nb["x"] - na["x"], nb["y"] - na["y"]))
        drift = {}
        for field, info in self.fields.items():
            total = np.zeros(2)
            per_frame = {}
            for frame in range(info["frame_count"]):
                moves = steps.get((field, frame))
                if moves and len(moves) >= 3:
                    total = total + np.median(np.array(moves), axis=0)
                per_frame[frame] = total.copy()
            drift[field] = per_frame
        return drift

    def _frames(self, row):
        return {
            "divided": dict(
                last_single_frame=row["lower_frame"],
                division_frame=row["upper_frame"],
                event_frame=row["upper_frame"],
                last_valid_frame=max(row["last_identity_frame"], row["upper_frame"]),
            ),
            "unresolved": dict(
                last_single_frame=None,
                division_frame=None,
                last_valid_frame=row["lower_frame"],
                event_frame=row["upper_frame"],
            ),
            "not_grown": dict(
                last_single_frame=None,
                division_frame=None,
                last_valid_frame=row["upper_frame"],
                event_frame=row["upper_frame"],
            ),
            "excluded": dict(
                last_single_frame=None,
                division_frame=None,
                last_valid_frame=row["last_identity_frame"],
                event_frame=row["baseline_frame"],
            ),
        }[FROM_STORE[row["decision"]]]

    def _record(self, row, orig, override):
        field = row["field"]
        decision = FROM_STORE[row["decision"]]
        label = row["id"].rsplit("-o", 1)[-1]
        founder_label = label if row["baseline_frame"] == 0 else f"{label}@f{row['baseline_frame']}"
        reason = orig["excluded_reason"] or orig["tracking_end_reason"].replace("_", " ")
        if orig.get("flags"):
            reason += "; check: " + ", ".join(f.replace("_", " ") for f in orig["flags"])
        override = override or {}
        record = {
            "id": row["id"],
            "position": field,
            "founder_id": founder_label,
            "condition": row["condition"],
            "decision": decision,
            "baseline_frame": row["baseline_frame"],
            "x": row["initial_x"],
            "y": row["initial_y"],
            "crop_width": DEFAULT_WINDOW,
            "tracker_endpoint_frame": orig["last_identity_frame"],
            "tracking_end_reason": orig["tracking_end_reason"],
            "excluded_reason": row["excluded_reason"],
            "frame_count": self.fields[field]["frame_count"],
            "reviewed": bool(row["reviewed"]),
            "note": row.get("note", ""),
            "confidence": override.get("confidence", ""),
            "reviewer": override.get("reviewer", ""),
            "full_window_reviewed": bool(override.get("full_window_verified", False)),
            "baseline": self.baseline.get(row["id"], {}),
            **self._frames(row),
        }
        for key in ["review_anchor_x", "review_anchor_y", "review_anchor_frame", "review_crop_width",
                    "exclude_reason"]:
            if key in override:
                record[key] = override[key]
        if override.get("last_certain_frame") is not None:
            record["last_valid_frame"] = override["last_certain_frame"]
        record["saved_by_viewer"] = override.get("source") == "workbench"
        record["original"] = {
            "decision": FROM_STORE[orig["decision"]],
            "note": f"Automatic call: {reason}." if reason else "",
            **self._frames(orig),
        }
        return record

    def all_effective(self):
        return self.state()["records"]

    def effective(self, fid):
        record = self.state()["byid"].get(fid)
        if record is None:
            raise ValueError("Unknown founder ID")
        return record

    def config(self):
        cfg = self.project.config
        widths = [f["shape"][1] for f in self.fields.values()]
        heights = [f["shape"][0] for f in self.fields.values()]
        return {
            "project_id": f"{self.root.name}:{self.run_id}",
            "title": f"{cfg.get('acquisition_take', self.root.name)} founder review",
            "run_id": self.run_id,
            "image_width": max(widths),
            "image_height": max(heights),
            "frame_count": max(f["frame_count"] for f in self.fields.values()),
            "structure_channel": self.structure,
            "channels": [
                {"id": c, "label": self.channel_labels[c]} for c in self.fluorescence
            ],
            "fields": {
                name: {
                    "frame_count": f["frame_count"],
                    "times_s": [f["times"].get(i) for i in range(f["frame_count"])],
                    "channels": {
                        c: sorted(frames) for c, frames in f["channels"].items() if c != self.structure
                    },
                }
                for name, f in self.fields.items()
            },
            "time_origin": cfg.get("time_origin", ""),
            "read_only": self.read_only,
            "condition_grouping_label": "Conditions from project.json",
        }

    def excluded_frames(self):
        """Structure frames marked unusable, per field: shown on the timeline."""
        st = self.state()
        out = {}
        for (field, frame, channel), ok in st["usable"].items():
            if channel == self.structure and not ok:
                out.setdefault(field, []).append(frame)
        return {k: sorted(v) for k, v in out.items()}

    # ----- images ---------------------------------------------------------------------
    def _raw(self, field, frame, channel):
        key = (field, frame, channel)
        if key not in self.project.frames:
            return None
        return self.arrays.get(key, lambda: read_frame(self.project, field, frame, channel))

    def _signal(self, field, frame, channel):
        """Fluorescence minus a smooth local background, scaled to 100 ms (display only)."""
        key = (field, frame, channel)
        if key not in self.project.frames:
            return None

        def compute():
            raw = self._raw(field, frame, channel).astype(np.float32)
            h, w = raw.shape
            small = np.asarray(
                Image.fromarray(raw).resize((max(1, w // 4), max(1, h // 4)), Image.BOX)
            )
            background = ndimage.gaussian_filter(small, 10)
            background = np.asarray(Image.fromarray(background).resize((w, h), Image.BILINEAR))
            exposure = float(self.project.frames[key].get("exposure_ms") or 100)
            out = (raw - background) * (100.0 / exposure)
            out.setflags(write=False)
            return out

        return self.signals.get(key, compute)

    @staticmethod
    def _window(shape, center, width):
        x0 = int(round(center[0] - width / 2))
        y0 = int(round(center[1] - width / 2))
        return x0, y0, int(round(width))

    @staticmethod
    def _cut(array, box, out):
        """Square window at raw resolution, zero padded outside the image, resized to ``out``."""
        x0, y0, w = box
        h_img, w_img = array.shape
        region = np.zeros((w, w), np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w_img, x0 + w), min(h_img, y0 + w)
        if sx1 > sx0 and sy1 > sy0:
            region[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
        if w == out:
            return region
        method = Image.BOX if w > out else Image.BILINEAR
        return np.asarray(Image.fromarray(region).resize((out, out), method))

    def center(self, record, frame):
        x = record.get("review_anchor_x", record["x"])
        y = record.get("review_anchor_y", record["y"])
        anchor_frame = int(record.get("review_anchor_frame", record["baseline_frame"]))
        drift = self.state()["drift"].get(record["position"], {})
        zero = np.zeros(2)
        return np.array([float(x), float(y)]) + drift.get(frame, zero) - drift.get(anchor_frame, zero)

    def _outline_mask(self, record, frame, box, out):
        """Boundary pixels of the founder's tracked masks at this frame, in display pixels."""
        nodes = self.state()["nodes_at"].get((record["id"], frame), [])
        if not nodes:
            return None, 0
        x0, y0, w = box
        shape = self.fields[record["position"]]["shape"]
        mask = np.zeros((w, w), bool)
        for node in nodes:
            if "polygon" in node:
                full = polygon_mask(node["polygon"], shape)
            else:
                full = label_image(self.project, self.run_id, node) == node["label"]
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(shape[1], x0 + w), min(shape[0], y0 + w)
            if sx1 > sx0 and sy1 > sy0:
                mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] |= full[sy0:sy1, sx0:sx1]
        if not mask.any():
            return None, len(nodes)
        scaled = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((out, out), Image.NEAREST)) > 127
        edge = scaled & ~ndimage.binary_erosion(scaled, iterations=2)
        return edge, len(nodes)

    def phase_limits(self, record):
        def compute():
            frame = record["baseline_frame"]
            raw = self._raw(record["position"], frame, self.structure)
            box = self._window(raw.shape, self.center(record, frame), record["crop_width"])
            crop = self._cut(raw.astype(np.float32), box, 256)
            values = crop[crop > 0] if (crop > 0).any() else crop
            lo, hi = np.percentile(values, [0.4, 99.6])
            return float(lo), float(hi)

        key = (record["id"], record.get("review_anchor_x"), record.get("review_anchor_y"))
        return self.limits.get(key, compute)

    def render(self, record, frame, mode="overlay", channel=None, width=None, gain=1.0,
               alpha=1.0, outlines=True, full=False, auto_contrast=False):
        field = record["position"]
        if frame not in self.fields[field]["channels"].get(self.structure, ()):
            raise ValueError("Frame outside this field's recording")
        channel = channel or (self.fluorescence[0] if self.fluorescence else None)
        if channel is not None and channel not in self.fluorescence:
            raise ValueError("Unknown fluorescence channel")
        revision = self.state()["revision"]
        key = (revision, record["id"], frame, mode, channel, width, gain, alpha, outlines, full,
               auto_contrast, record.get("review_anchor_x"), record.get("review_anchor_y"))
        return self.pngs.get(key, lambda: self._render(record, frame, mode, channel, width, gain,
                                                       alpha, outlines, full, auto_contrast))

    def _render(self, record, frame, mode, channel, width, gain, alpha, outlines, full, auto_contrast):
        field = record["position"]
        out = OUTPUT_SIZE
        phase = self._raw(field, frame, self.structure)
        if full:
            width = max(phase.shape)
            center = np.array([phase.shape[1] / 2, phase.shape[0] / 2])
        else:
            width = float(width or record.get("review_crop_width") or record["crop_width"])
            center = self.center(record, frame)
        box = self._window(phase.shape, center, width)
        ph = self._cut(phase.astype(np.float32), box, out)
        if auto_contrast or full:
            values = ph[ph > 0] if (ph > 0).any() else ph
            lo, hi = np.percentile(values, [0.4, 99.6])
        else:
            lo, hi = self.phase_limits(record)
        gray = np.uint8(np.clip((ph - lo) / max(hi - lo, 1) * 255, 0, 255))
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        fluor_missing = False
        if mode in ("overlay", "fluor") and channel:
            signal = self._signal(field, frame, channel)
            if signal is None:
                fluor_missing = True
            else:
                fl = self._cut(signal, box, out)
                view = self.project.config.get("viewer", {})
                soft = float(view.get("fluorescence_softening", 50))
                vmax = float(view.get("fluorescence_vmax", 1500))
                bright = np.clip(np.arcsinh(np.maximum(fl, 0) * gain / soft) / np.arcsinh(vmax / soft), 0, 1)
                color = np.array([255, 20, 15]) if channel == self.fluorescence[0] else np.array([40, 255, 60])
                if mode == "fluor":
                    rgb = (bright[:, :, None] * color).astype(np.uint8)
                else:
                    mix = bright * 0.9 * alpha
                    rgb = (rgb * (1 - mix[:, :, None]) + color * mix[:, :, None]).clip(0, 255).astype(np.uint8)
        count = 0
        if outlines:
            edge, count = self._outline_mask(record, frame, box, out)
            if edge is not None:
                rgb[edge] = (0, 230, 255)
        else:
            count = len(self.state()["nodes_at"].get((record["id"], frame), []))
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG", compress_level=1)
        st = self.state()
        meta = {
            "frame": frame,
            "time_s": self.fields[field]["times"].get(frame),
            "center": center.tolist(),
            "width": float(box[2]),
            "phase_limits": [float(lo), float(hi)],
            "has_outline": count > 0,
            "outline_count": count,
            "outline_source": "tracker",
            "fluorescence_missing": fluor_missing,
            "channel_label": self.channel_labels.get(channel, channel) if channel else None,
            "structure_usable": st["usable"].get((field, frame, self.structure), True),
            "quality_reason": st["quality_reason"].get((field, frame, self.structure), ""),
        }
        return stream.getvalue(), meta

    def prefetch(self, record, frame, channel=None):
        """Warm raw/fluorescence arrays for the next frames so stepping and playback stay quick."""
        field = record["position"]
        channel = channel or (self.fluorescence[0] if self.fluorescence else None)
        for f in (frame + 1, frame + 2, frame - 1):
            if (field, f, self.structure) not in self.project.frames:
                continue
            if (field, f, self.structure) not in self.arrays:
                self.prefetcher.submit(self._raw, field, f, self.structure)
            if channel and (field, f, channel) in self.project.frames and (field, f, channel) not in self.signals:
                self.prefetcher.submit(self._signal, field, f, channel)

    # ----- saving: previous-viewer actions -> review-store operations ----------------
    EXTRA = ("note", "confidence", "reviewer", "exclude_reason", "review_anchor_x", "review_anchor_y",
             "review_anchor_frame", "review_crop_width")

    def _writable(self):
        if self.read_only:
            raise PermissionError("This viewer was started read-only; nothing was changed.")

    def _event_operation(self, fid, changes):
        current = self.effective(fid)
        merged = {**current, **changes}
        decision = merged.get("decision")
        if decision not in TO_STORE:
            raise ValueError("Choose a decision")
        op = {"type": "set_event", "founder_id": fid, "decision": TO_STORE[decision], "source": "workbench"}

        def frame(key, label):
            value = merged.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Enter the {label}")
            return value

        if decision == "divided":
            op["lower_frame"] = frame("last_single_frame", "last frame before visible division")
            op["upper_frame"] = frame("division_frame", "first divided frame")
            if merged.get("last_valid_frame") is not None:
                op["last_certain_frame"] = frame("last_valid_frame", "last certain tracking frame")
        elif decision == "unresolved":
            op["lower_frame"] = frame("last_valid_frame", "last certain tracking frame")
            op["upper_frame"] = frame("event_frame", "first uncertain frame")
        elif decision == "not_grown":
            op["full_window_verified"] = merged.get("full_window_reviewed") is True
        for key in self.EXTRA:
            value = merged.get(key)
            if value is None or value == "":
                continue
            if key in ("note", "confidence", "reviewer", "exclude_reason"):
                if not isinstance(value, str) or len(value) > 12000:
                    raise ValueError(f"Invalid {key}")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {key}")
            if key == "exclude_reason" and decision != "excluded":
                continue
            op[key] = value
        return op

    def _apply(self, operation, expected_revision):
        self._writable()
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("Missing review revision")
        result = save(self.project, self.run_id, operation, expected_revision)
        return result["revision"]

    def save_override(self, fid, changes, expected_revision):
        if not isinstance(changes, dict):
            raise ValueError("Invalid review")
        revision = self._apply(self._event_operation(fid, changes), expected_revision)
        return revision, self.effective(fid)

    def revert(self, fid, expected_revision):
        self.effective(fid)
        revision = self._apply({"type": "clear_event", "founder_id": fid}, expected_revision)
        return revision, self.effective(fid)

    def import_reviews(self, payload, expected_revision):
        if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
            raise ValueError("Choose a review file exported from this viewer")
        revision = self._apply({"type": "import", "review": payload}, expected_revision)
        return revision, self.all_effective()

    # ----- exports --------------------------------------------------------------------
    def export_review(self):
        revision, operations = self._db_revision()
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "manifest_digest": self.project.config["manifest_digest"],
            "revision": revision,
            "operations": copy.deepcopy(operations),
        }

    def latest_results(self):
        pointer = self.root / "latest_results.json"
        if not pointer.exists():
            return None
        info = read_json(pointer)
        rel = Path("results") / info["result_id"]
        summary = self.root / rel / "summary.json"
        if not summary.exists():
            return None
        data = read_json(summary)
        return {
            "relative_dir": str(rel),
            "review_revision": info.get("review_revision", data.get("review_revision")),
            "reviewed_only": data.get("reviewed_only", False),
            "plots": data.get("plots", []),
            "counts": data.get("counts", {}),
        }

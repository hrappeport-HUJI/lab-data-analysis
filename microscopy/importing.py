"""Discovery and an explicit acquisition manifest for 2D TIFF folders."""

from pathlib import Path
import csv
import json
import re
import tifffile
from .storage import Project, digest, file_digest, write_json, utc


def inspect(folder):
    root = Path(folder).resolve()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff"})
    examples = []
    for p in files[:20]:
        with tifffile.TiffFile(p) as t:
            examples.append(
                {
                    "path": str(p.relative_to(root)),
                    "shape": list(t.series[0].shape),
                    "axes": t.series[0].axes,
                    "dtype": str(t.series[0].dtype),
                }
            )
    return {
        "root": str(root),
        "tiff_files": len(files),
        "examples": examples,
        "note": "Channel identity, elapsed time, exposure, conditions and biological replicates require explicit metadata; none are inferred from brightness or file modification times.",
    }


def template():
    return {
        "file_pattern": r"Pos(?P<field>\d+)/(?P<channel>[^/]+)/img_(?P<frame>\d+).*\.tif",
        "structure_channel": "REPLACE",
        "fluorescence_channels": [],
        "frame_interval_s": None,
        "exposure_ms": {},
        "metadata_csv": None,
        "conditions": {},
        "biological_replicates": {},
        "pixel_size_um": None,
        "segmentation": {
            "adapter": "threshold",
            "polarity": "dark",
            "threshold": None,
            "min_area_px": 12,
            "max_area_px": 5000,
            "background_sigma_px": 8,
        },
        "tracking": {
            "adapter": "overlap",
            "max_displacement_px": 15,
            "division_area_ratio": [0.65, 1.5],
            "min_child_fraction": 0.2,
            "confirmation_frames": 1,
        },
        "quality": {"min_focus_score": None},
        "founders": {
            "edge_margin_px": 3,
            "isolation_px": 0,
            "allow_delayed_start": False,
        },
        "measurements": [],
        "analysis": {
            "comparisons": [],
            "unit": "field",
            "allow_field_inference": False,
        },
    }


def initialize(raw_root, config_path, destination):
    raw_root, destination = Path(raw_root).resolve(), Path(destination).resolve()
    if destination == raw_root or destination.is_relative_to(raw_root):
        raise ValueError("Project outputs must be outside the raw acquisition folder")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Destination must be empty; existing reviews will never be replaced"
        )
    from .config import validate

    cfg = validate(json.loads(Path(config_path).read_text()))
    pattern = re.compile(cfg["file_pattern"])
    if not {"field", "frame", "channel"} <= pattern.groupindex.keys():
        raise ValueError("file_pattern needs named field, frame and channel groups")
    structure = cfg.get("structure_channel")
    if not structure or structure == "REPLACE":
        raise ValueError("Identify the structure channel explicitly")
    metadata = {}
    if cfg.get("metadata_csv"):
        meta_path = Path(cfg["metadata_csv"])
        if not meta_path.is_absolute():
            meta_path = Path(config_path).resolve().parent / meta_path
        with open(meta_path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["field"], int(r["frame"]), r["channel"])
                if key in metadata:
                    raise ValueError("Duplicate metadata record")
                metadata[key] = r
    records, keys = [], set()
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        m = pattern.fullmatch(path.relative_to(raw_root).as_posix())
        if not m:
            continue
        field, frame, channel = m["field"], int(m["frame"]), m["channel"]
        key = (field, frame, channel)
        if key in keys:
            raise ValueError(f"Multiple files identify {key}")
        keys.add(key)
        with tifffile.TiffFile(path) as t:
            if len(t.series) != 1 or len(t.series[0].shape) != 2:
                raise ValueError(
                    f"Expected one 2D image: {path.name}. Use a reader adapter/export for stacks or volumes."
                )
            shape, dtype = list(t.series[0].shape), str(t.series[0].dtype)
        md = metadata.get(key, {})
        interval = cfg.get("frame_interval_s")
        time = (
            float(md["time_s"])
            if md.get("time_s")
            else frame * float(interval) if interval is not None else None
        )
        exposure = (
            float(md["exposure_ms"])
            if md.get("exposure_ms")
            else cfg.get("exposure_ms", {}).get(channel)
        )
        if time is not None and (not __import__("math").isfinite(time) or time < 0):
            raise ValueError("Elapsed times must be finite and nonnegative")
        if exposure is not None and (
            not __import__("math").isfinite(exposure) or exposure <= 0
        ):
            raise ValueError("Exposures must be positive and finite")
        records.append(
            {
                "field": field,
                "frame": frame,
                "channel": channel,
                "path": str(path.relative_to(raw_root)),
                "shape": shape,
                "dtype": dtype,
                "time_s": time,
                "exposure_ms": exposure,
                "sha256": file_digest(path),
            }
        )
    if not records:
        raise ValueError("No images match the configured pattern")
    fields = sorted({r["field"] for r in records})
    for field in fields:
        structural = sorted(
            (r for r in records if r["field"] == field and r["channel"] == structure),
            key=lambda r: r["frame"],
        )
        if not structural:
            raise ValueError(f"No structure channel for field {field}")
        if len({tuple(r["shape"]) for r in records if r["field"] == field}) != 1:
            raise ValueError(
                f"Channels in field {field} differ in dimensions; register/export first"
            )
        times = [r["time_s"] for r in structural]
        if all(t is not None for t in times) and any(
            b <= a for a, b in zip(times, times[1:])
        ):
            raise ValueError("Structure timestamps must strictly increase")
    channels = {r["channel"] for r in records}
    if set(cfg.get("fluorescence_channels", [])) - channels:
        raise ValueError("A requested fluorescence channel has no images")
    cfg.update(
        schema_version=1,
        raw_root=str(raw_root),
        manifest_digest=digest(records),
        created_at=utc(),
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "manifest.json", records)
    write_json(destination / "project.json", cfg)
    return Project(destination)

"""Build the per-image metadata table from a Micro-Manager acquisition's metadata.txt files.

Each position folder (``Pos*/``) holds a ``metadata.txt`` JSON with one ``FrameKey-<frame>-<channel>-<slice>``
entry per image. The table this writes is what ``metadata_csv`` in a project configuration expects:
``field,frame,channel,time_s,exposure_ms,file`` plus the wall-clock time, pixel size and every
light-source level that was recorded (columns ``light_<name>``), so excitation changes are visible.

``time_s`` is Micro-Manager's ``ElapsedTime-ms`` / 1000: one clock for the whole acquisition, shared by
all positions, starting when the acquisition started. Nothing is inferred from file modification times.
"""
import csv
import json
import re
from pathlib import Path

BASE_COLUMNS = ["field", "frame", "channel", "time_s", "exposure_ms", "file", "wall_time", "pixel_size_um"]
LEVEL_KEY = re.compile(r"^(?P<device>[A-Za-z0-9]+)-(?P<name>[A-Za-z0-9]+)_Level$")


def read_position(metadata_path):
    """Rows for one position folder's metadata.txt."""
    metadata_path = Path(metadata_path)
    field = re.fullmatch(r"Pos(\d+)", metadata_path.parent.name)
    if not field:
        raise ValueError(f"Expected a Pos<N> folder: {metadata_path.parent}")
    data = json.loads(metadata_path.read_text())
    rows = []
    for key, entry in data.items():
        if not key.startswith("FrameKey-"):
            continue
        row = {
            "field": field.group(1),
            "frame": int(entry["Frame"]),
            "channel": entry["Channel"],
            "time_s": float(entry["ElapsedTime-ms"]) / 1000.0,
            "exposure_ms": float(entry["Exposure-ms"]),
            "file": f"{metadata_path.parent.name}/{entry['FileName']}",
            "wall_time": entry.get("Time", ""),
            "pixel_size_um": entry.get("PixelSizeUm", ""),
        }
        for name, value in entry.items():
            m = LEVEL_KEY.match(name)
            if m:
                row[f"light_{m['device']}_{m['name']}".lower()] = value
        rows.append(row)
    return rows


def build_table(acquisition, output):
    """Write the metadata table for every Pos*/metadata.txt under ``acquisition``. Returns a summary."""
    acquisition, output = Path(acquisition), Path(output)
    if output.exists():
        raise ValueError(f"{output} exists; choose a new file")
    if acquisition.resolve() in output.resolve().parents:
        raise ValueError("Write the table outside the raw acquisition folder")
    files = sorted(acquisition.glob("Pos*/metadata.txt"), key=lambda p: int(p.parent.name[3:]))
    if not files:
        raise FileNotFoundError(f"No Pos*/metadata.txt under {acquisition}")
    rows = [r for f in files for r in read_position(f)]
    seen = set()
    for r in rows:
        key = (r["field"], r["frame"], r["channel"])
        if key in seen:
            raise ValueError(f"Duplicate metadata record {key}")
        seen.add(key)
        if not (acquisition / r["file"]).is_file():
            raise FileNotFoundError(f"metadata.txt lists a missing image: {r['file']}")
    rows.sort(key=lambda r: (int(r["field"]), r["frame"], r["channel"]))
    extra = sorted({k for r in rows for k in r} - set(BASE_COLUMNS))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS + extra)
        writer.writeheader()
        writer.writerows(rows)
    channels = sorted({r["channel"] for r in rows})
    return {
        "output": str(output),
        "positions": len(files),
        "images": len(rows),
        "channels": {c: sum(r["channel"] == c for r in rows) for c in channels},
        "frames_per_position": max(r["frame"] for r in rows) + 1,
        "exposure_ms": sorted({r["exposure_ms"] for r in rows}),
    }

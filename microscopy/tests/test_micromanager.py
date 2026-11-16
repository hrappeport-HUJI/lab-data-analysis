import csv
import json

import pytest

from microscopy.micromanager import build_table


def _position(root, name, frames):
    folder = root / name
    folder.mkdir(parents=True)
    entries = {"Summary": {"ChNames": ["Phase", "PI"]}}
    for frame in range(frames):
        for ci, channel in enumerate(["Phase", "PI"]):
            file = f"img_{frame:09d}_{channel}_000.tif"
            (folder / file).write_bytes(b"")
            entries[f"FrameKey-{frame}-{ci}-0"] = {
                "Frame": frame, "Channel": channel, "FileName": file,
                "ElapsedTime-ms": 1000 * (420 * frame + ci), "Exposure-ms": 200,
                "Time": "2026-11-12 17:28:49 +0200", "PixelSizeUm": 0,
                "Spectra-Green_Level": "10" if channel == "PI" else "0", "Spectra-White_Level": "100",
            }
    (folder / "metadata.txt").write_text(json.dumps(entries))


def test_table_from_metadata_files(tmp_path):
    raw = tmp_path / "raw"
    _position(raw, "Pos0", 2)
    _position(raw, "Pos10", 2)
    summary = build_table(raw, tmp_path / "meta.csv")
    assert summary["positions"] == 2 and summary["images"] == 8
    rows = list(csv.DictReader(open(tmp_path / "meta.csv")))
    assert [r["field"] for r in rows[:4]] == ["0"] * 4 and rows[-1]["field"] == "10"
    pi = next(r for r in rows if r["field"] == "0" and r["frame"] == "1" and r["channel"] == "PI")
    assert float(pi["time_s"]) == 421.0 and float(pi["exposure_ms"]) == 200.0
    assert pi["file"] == "Pos0/img_000000001_PI_000.tif" and pi["light_spectra_green"] == "10"


def test_refuses_missing_images_and_overwrites(tmp_path):
    raw = tmp_path / "raw"
    _position(raw, "Pos0", 1)
    (raw / "Pos0" / "img_000000000_PI_000.tif").unlink()
    with pytest.raises(FileNotFoundError):
        build_table(raw, tmp_path / "meta.csv")
    (tmp_path / "exists.csv").write_text("")
    with pytest.raises(ValueError):
        build_table(raw, tmp_path / "exists.csv")
    with pytest.raises(ValueError):
        build_table(raw, raw / "inside.csv")

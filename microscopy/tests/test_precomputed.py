import numpy as np
import pytest

from microscopy.adapters import get_adapter
from microscopy.demo import create
from microscopy.pipeline import run


def _write(folder, field, frame, labels):
    path = folder / f"Pos{field}" / f"{frame:09d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels)


def test_reads_filters_and_renumbers(tmp_path):
    labels = np.zeros((20, 20), np.uint16)
    labels[1:6, 1:6] = 7      # 25 px
    labels[10:12, 10:12] = 9  # 4 px: below min_area
    labels[14:19, 14:19] = 3
    _write(tmp_path, "4", 2, labels)
    seg = get_adapter("segmenter", "precomputed")
    out = seg.segment(np.zeros((20, 20)), context={"field": "4", "frame": 2},
                      parameters={"labels_dir": str(tmp_path), "min_area_px": 10})
    assert out.dtype == np.int32 and set(np.unique(out)) == {0, 1, 2}
    assert (out[1:6, 1:6] == 2).all() and (out[14:19, 14:19] == 1).all() and out[10, 10] == 0
    with pytest.raises(FileNotFoundError):
        seg.segment(np.zeros((20, 20)), context={"field": "4", "frame": 3}, parameters={"labels_dir": str(tmp_path)})
    with pytest.raises(ValueError, match="do not match"):
        seg.segment(np.zeros((10, 10)), context={"field": "4", "frame": 2}, parameters={"labels_dir": str(tmp_path)})


def test_pipeline_with_precomputed_equals_the_source_segmentation(tmp_path):
    p = create(tmp_path / "demo")
    first = run(p, fields=["1"])
    folder = tmp_path / "labels"
    for plane in sorted(p.run_path(first["run_id"]).glob("labels/*.npz")):
        fi, frame = plane.stem.split("_")
        with np.load(plane) as stored:
            _write(folder, "1", int(frame), stored["labels"])
    p.config["segmentation"] = {"adapter": "precomputed", "labels_dir": str(folder), "min_area_px": 0}
    p.config["frame_workers"] = 2
    second = run(p, fields=["1"])
    assert second["founders"] == first["founders"]
    assert [o["area_px"] if "area_px" in o else o.get("area") for o in second["observations"]] == \
           [o["area_px"] if "area_px" in o else o.get("area") for o in first["observations"]]

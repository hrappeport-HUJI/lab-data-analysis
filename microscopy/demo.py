"""Deterministic synthetic acquisition; no biological benchmark claim."""

from pathlib import Path
import numpy as np
import tifffile
from .importing import template, initialize
from .storage import write_json
from .pipeline import run


def create(folder):
    folder = Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Demo destination must be empty")
    raw = folder / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:96, :128]
    truth = []
    for field in range(1, 5):
        division = 3 + (field % 2)
        for frame in range(8):
            labels = np.zeros((96, 128), np.uint16)
            if frame < division:
                labels[(abs(xx - 32) <= 8) & (abs(yy - 25) <= 3)] = 1
            else:
                labels[(abs(xx - 27) <= 3) & (abs(yy - 25) <= 3)] = 1
                labels[(abs(xx - 37) <= 3) & (abs(yy - 25) <= 3)] = 2
            if frame < 5:
                labels[(abs(xx - 85) <= 6) & (abs(yy - 25) <= 3)] = 3
            labels[(abs(xx - 35) <= 6) & (abs(yy - 70) <= 3)] = 4
            phase = 1000 + rng.normal(0, 2, labels.shape) - 500 * (labels > 0)
            fluorescence = (
                (100 if field < 3 else 400)
                + rng.normal(0, 2, labels.shape)
                + (labels > 0) * (40 + field * 10)
            )
            for channel, image in [("phase", phase), ("PI", fluorescence)]:
                dest = raw / f"Pos{field}" / channel
                dest.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    dest / f"img_{frame:03}.tif",
                    np.clip(image, 0, 65535).astype("uint16"),
                )
        truth.append(
            {
                "field": str(field),
                "division_interval": [division - 1, division],
                "loss_interval": [4, 5],
                "persistent_cell": "no observed division through frame 7",
            }
        )
    cfg = template()
    cfg.update(
        structure_channel="phase",
        fluorescence_channels=["PI"],
        frame_interval_s=600,
        exposure_ms={"PI": 100},
        conditions={"1": "A", "2": "A", "3": "B", "4": "B"},
    )
    cfg["segmentation"].update(threshold=100, background_sigma_px=20, min_area_px=20)
    cfg["measurements"] = [
        {"name": "baseline_PI", "channel": "PI", "selection": "baseline"}
    ]
    cfg["analysis"]["comparisons"] = [["A", "B"]]
    write_json(folder / "config.json", cfg)
    write_json(folder / "synthetic_truth.json", truth)
    project = initialize(raw, folder / "config.json", folder / "project")
    run(project)
    return project

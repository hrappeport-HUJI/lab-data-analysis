"""Run a fine-tuned model on every frame of test-set plates and track MATLAB's colonies by seed.

    python scripts/track_with_model.py --plates EPEC260224 [--testset testset_matlab] [--model models/unet_scanlag_v1]

``--plates`` is a plate-ID prefix. Per plate, writes to ``<testset>/model_runs/<model>/plates/<plate>/``:
``labels.zarr`` (uint16 [T, H, W]) and ``tracks.parquet`` (frame, time_min, colony_id, area_px,
merged), in the same coordinates and with the same colony IDs as the MATLAB plate. Plates that
already have tracks.parquet are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scanlag.detect.unet import load_detector
from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.testset.tiles import PREVIOUS_GAP, plate_window
from scanlag.track.seed_areas import seed_areas, tracks_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plates", required=True, help="plate-ID prefix, e.g. EPEC260224")
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "unet_scanlag_v1",
                        help="trained model folder (default: models/unet_scanlag_v1)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    det = load_detector(args.model)
    plates = sorted(p for p in (args.testset / "plates").iterdir() if p.name.startswith(args.plates))
    io = ThreadPoolExecutor(2)
    for plate_dir in plates:
        out = ("model_runs", args.model.name, "plates", plate_dir.name)
        if store.path(*out, "tracks.parquet").exists():
            print(f"{plate_dir.name}: done already", flush=True)
            continue
        info = json.loads((plate_dir / "plate.json").read_text())
        colonies = pd.read_parquet(plate_dir / "colonies.parquet")
        shape = np.array(Image.open(plate_dir / "mask.png")).shape
        n_frames = len(info["raw_scans"])
        read = lambda f: plate_window(info, f, 0, 0, shape)
        background = read(0)
        labels = store.create_labels((n_frames, *shape), *out, "labels.zarr")
        sx, sy = colonies.seed_x.to_numpy(int), colonies.seed_y.to_numpy(int)
        area = np.zeros((n_frames, len(colonies)), np.int64)
        merged = np.zeros_like(area, bool)
        t0 = time.time()
        pending = io.submit(lambda: (read(0), read(0)))
        for f in range(n_frames):
            image, previous = pending.result()
            if f + 1 < n_frames:  # prefetch the next frame while the GPU works
                pending = io.submit(lambda g=f + 1: (read(g), read(max(g - PREVIOUS_GAP, 0))))
            pred = det.segment([det.prepare(image, background, previous)])[0]
            labels[f] = pred
            area[f], merged[f] = seed_areas(pred, sx, sy)
        tracks = tracks_table(area, merged, colonies.colony_id.to_numpy(), info["times_min"])
        store.write_table(tracks, *out, "tracks.parquet")
        print(f"{plate_dir.name}: {n_frames} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

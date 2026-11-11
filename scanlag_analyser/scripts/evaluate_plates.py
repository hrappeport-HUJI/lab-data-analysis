"""Compare trained models and MATLAB on whole plates, against the colonies the user kept.

    python scripts/evaluate_plates.py --testset testset_vapc [--models models/cpsam_scanlag_v1 models/unet_scanlag_v1]

The test set comes from ``scripts/build_testset.py`` (MATLAB replayed bit-exactly). Each plate is
scored at two frames: the median appearance frame of its kept colonies ("mid") and the last frame.
See ``scanlag/eval/seeds.py`` for the matching rule.

Detectors:
  matlab_all  every blob MATLAB's threshold found at that frame (before review)
  <model>     each model in --models, run on the full plate crop

Writes ``<testset>/eval/plates.csv`` (one row per plate, frame and detector, with seconds per frame),
``summary.csv`` and each model's labels ``pred/<detector>/<plate>_f<frame>.png``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image
from scipy import ndimage as ndi

from scanlag.detect.unet import load_detector
from scanlag.eval.seeds import score_frame, summarize
from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.testset.tiles import PREVIOUS_GAP, plate_window

DEFAULT_MODELS = [PROJECT_ROOT / "models" / "cpsam_scanlag_v1", PROJECT_ROOT / "models" / "unet_scanlag_v1"]


def matlab_blobs(labels: np.ndarray) -> np.ndarray:
    """MATLAB's blobs at one frame: kept colonies and IGNORE are both blobs (8-connected)."""
    return ndi.label(labels > 0, structure=np.ones((3, 3)))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, required=True, help="test set folder (scripts/build_testset.py)")
    parser.add_argument("--models", type=Path, nargs="+", default=DEFAULT_MODELS,
                        help="trained model folders to evaluate (default: models/cpsam_scanlag_v1 models/unet_scanlag_v1)")
    parser.add_argument("--unet-cores", type=float, nargs="+", default=[0.7],
                        help="U-Net watershed core thresholds to compare (one detector each; higher splits less)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    detectors = []
    for m in args.models:
        first = load_detector(m)
        if hasattr(first, "core"):
            detectors += [load_detector(m, core=c, _net=first.net()) for c in args.unet_cores]
        else:
            detectors.append(first)
    for det in detectors:  # load weights and warm up before timing
        det.segment([np.zeros((256, 256, 3), np.float32)])
    rows = []
    for plate_dir in sorted((args.testset / "plates").iterdir()):
        info = json.loads((plate_dir / "plate.json").read_text())
        colonies = pd.read_parquet(plate_dir / "colonies.parquet")
        mask = np.array(Image.open(plate_dir / "mask.png")).astype(bool)
        labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
        last = labels.shape[0] - 1
        mid = int(np.median(colonies.loc[colonies.status == 0, "appearance_frame"]))
        background = plate_window(info, 0, 0, 0, mask.shape)
        for kind, frame in (("mid", mid), ("last", last)):
            image = plate_window(info, frame, 0, 0, mask.shape)
            previous = plate_window(info, max(frame - PREVIOUS_GAP, 0), 0, 0, mask.shape)
            preds = [("matlab_all", matlab_blobs(labels[frame]), np.nan)]
            for det in detectors:
                t0 = time.perf_counter()
                pred = det.segment([det.prepare(image, background, previous)])[0]
                preds.append((det.name, pred, time.perf_counter() - t0))
                store.write_image(pred, "eval", "pred", det.name, f"{plate_dir.name}_f{frame}.png")
            for name, p, seconds in preds:
                row = score_frame(p, colonies, frame, mask)
                rows.append({"plate_id": plate_dir.name, "frame_kind": kind, "frame": frame, "detector": name,
                             "seconds": seconds, **row})
                print(f"{plate_dir.name} {kind:4s} f{frame:3d} {name:22s} det={row['n_det']:4d} "
                      f"P={row['tp'] / max(row['n_det'], 1):.3f} R={row['kept_found'] / max(row['n_kept'], 1):.3f}", flush=True)

    df = pd.DataFrame(rows)
    store.write_table(df, "eval", "plates.csv")
    summary = []
    df = df.assign(condition=df.plate_id.str.split("__").str[0])
    for keys, g in list(df.groupby(["frame_kind", "detector", "condition"])) + list(df.groupby(["frame_kind", "detector"])):
        kind, name = keys[0], keys[1]
        subset = keys[2] if len(keys) == 3 else "all"
        summary.append({"frame_kind": kind, "detector": name, "subset": subset,
                        "seconds_per_frame": g.seconds.mean(), **summarize(g.to_dict("records"))})
    store.write_table(pd.DataFrame(summary), "eval", "summary.csv")


if __name__ == "__main__":
    main()

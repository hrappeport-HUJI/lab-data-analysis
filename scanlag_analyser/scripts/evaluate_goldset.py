"""Score colony detectors against the hand-corrected gold tiles.

    python scripts/evaluate_goldset.py [--testset testset_matlab] [--detectors matlab cellpose]

Detectors:
  matlab_all   every blob the MATLAB threshold found (detector output before tracking and review)
  matlab_kept  only colonies kept after MATLAB tracking and manual review (status 0)
  cellpose_*   Cellpose-SAM zero-shot, over the input / diameter / min_size grid in CELLPOSE_GRID
  model:<dir>  a trained model from scripts/train_cellpose.py or train_unet.py, e.g. model:models/unet_scanlag_v1

Writes to ``<testset>/eval/``: ``summary.csv`` (one row per detector, updated in place),
``tiles_<detector>.csv`` and the predicted labels ``pred/<detector>/<tile>.png``.

Caveat: the gold labels started from MATLAB's segmentation, so MATLAB's boundaries (and hence
its IoU) are favoured wherever the annotator did not redraw a colony.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scanlag.eval.instances import counts, match, summarize
from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.testset.gold import load_gold_set

CELLPOSE_GRID = [
    {"input": inp, "diameter": d, "min_size": m}
    for inp in ("rgb", "diff", "diff_rgb")
    for d in (None, 10.0, 6.0)
    for m in (15, 5)
]


def predictions(detector: str, tiles):
    if detector == "matlab_all":
        yield detector, [t.matlab for t in tiles]
    elif detector == "matlab_kept":
        yield detector, [t.matlab_kept for t in tiles]
    elif detector == "cellpose":
        from scanlag.detect.cellpose_backend import CellposeDetector

        model = None
        for params in CELLPOSE_GRID:
            det = CellposeDetector(**params, _model=model)
            images = [det.prepare(t.image, t.background, t.previous) for t in tiles]
            yield det.name, det.segment(images)
            model = det._model
    elif detector.startswith("model:"):
        from scanlag.detect.unet import load_detector

        det = load_detector(detector.removeprefix("model:"))
        yield det.name, det.segment([det.prepare(t.image, t.background, t.previous) for t in tiles])
    else:
        raise ValueError(detector)


def evaluate(tiles, preds, iou: float) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    for t, p in zip(tiles, preds):
        rows.append({"tile": t.name, "category": t.category, **counts(match(t.labels, p, iou, t.ignore))})
    summaries = [{"subset": "all", **summarize(rows)}]
    for cat in sorted({r["category"] for r in rows}):
        summaries.append({"subset": cat, **summarize([r for r in rows if r["category"] == cat])})
    return pd.DataFrame(rows), summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    parser.add_argument("--detectors", nargs="+", default=["matlab_all", "matlab_kept", "cellpose"],
                        help="detectors to score, see above (default: matlab_all matlab_kept cellpose)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU for a detection to match a gold colony (default 0.5)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    tiles = load_gold_set(args.testset)
    summary_path = store.path("eval", "summary.csv")
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=["detector", "iou"])

    new = []
    for detector in args.detectors:
        for name, preds in predictions(detector, tiles):
            for t, p in zip(tiles, preds):
                store.write_image(p, "eval", "pred", name, f"{t.name}.png")
            per_tile, summaries = evaluate(tiles, preds, args.iou)
            store.write_table(per_tile, "eval", f"tiles_{name}.csv")
            new += [{"detector": name, "iou": args.iou, **s} for s in summaries]
            s = summaries[0]
            print(f"{name:34s} F1={s['f1']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} "
                  f"smallF1={s['small_f1']:.3f} IoU={s['mean_iou_matched']:.2f}")

    new = pd.DataFrame(new)
    keep = ~old.set_index(["detector", "iou"]).index.isin(new.set_index(["detector", "iou"]).index)
    store.write_table(pd.concat([old[keep], new], ignore_index=True), "eval", "summary.csv")


if __name__ == "__main__":
    main()

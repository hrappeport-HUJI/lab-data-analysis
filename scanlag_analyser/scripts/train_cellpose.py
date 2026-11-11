"""Fine-tune Cellpose-SAM on all MATLAB test-set plates plus the gold tiles.

    python scripts/train_cellpose.py --name cpsam_scanlag_v2 [--steps N] [--rebuild-tiles]

1. Builds ``<testset>/train/tiles.npz`` once (about 3k tiles read from the raw scans).
2. Trains and writes ``models/<name>/model.pt``, ``config.json`` and ``history.json``.

All plates and gold tiles are used for training, so scores on the gold set afterwards only show
that the model fits its training data. A real test needs new, unseen plates. An existing model
folder is not overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.train.cellpose_finetune import TrainConfig, train
from scanlag.train.dataset import build_training_set, load_training_set


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The docstring as written, and each option's default after its help."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab", help="test set folder (scripts/build_testset.py)")
    parser.add_argument("--name", required=True, help="model folder name under models/, e.g. cpsam_scanlag_v2")
    parser.add_argument("--force", action="store_true", help="overwrite models/<name> if it exists")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps, help="training steps")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="tiles per batch")
    parser.add_argument("--accumulate", type=int, default=TrainConfig.accumulate,
                        help="batches whose gradients are summed per optimizer step")
    parser.add_argument("--rebuild-tiles", action="store_true", help="build <testset>/train/tiles.npz again")
    args = parser.parse_args()
    if (PROJECT_ROOT / "models" / args.name).exists() and not args.force:
        parser.error(f"models/{args.name} exists: choose another --name, or --force to overwrite it")

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    tiles = args.testset / "train" / "tiles.npz"
    if args.rebuild_tiles or not tiles.exists():
        tiles = build_training_set(args.testset, store, log=lambda m: print(m, flush=True))
    x, labels, ignore, meta = load_training_set(tiles)
    print(f"{len(x)} tiles ({(meta.source == 'gold').sum()} gold)", flush=True)

    models = ResultsStore(PROJECT_ROOT / "models", raw_dirs)
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, accumulate=args.accumulate)
    out = train(x, labels, ignore, meta.source.to_numpy(), models.path(args.name, "model.pt").parent, cfg,
                log=lambda m: print(m, flush=True))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

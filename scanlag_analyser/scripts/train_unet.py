"""Train the U-Net colony detector on all MATLAB test-set tiles plus the gold tiles.

    python scripts/train_unet.py --name unet_scanlag_v3 [--steps N]

Uses ``<testset>/train/tiles.npz`` (built from the test set the first time, if missing) and writes
``models/<name>/model.pt``, ``config.json`` and ``history.json``. An existing model folder is not
overwritten unless --force is given. (unet_scanlag_v1 was trained with --steps 10000.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.train.dataset import build_training_set, load_training_set
from scanlag.train.unet_train import UNetConfig, train


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The docstring as written, and each option's default after its help."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab", help="test set folder (scripts/build_testset.py)")
    parser.add_argument("--name", required=True, help="model folder name under models/, e.g. unet_scanlag_v3")
    parser.add_argument("--force", action="store_true", help="overwrite models/<name> if it exists")
    parser.add_argument("--steps", type=int, default=UNetConfig.steps, help="training steps")
    parser.add_argument("--scale", type=float, default=UNetConfig.scale, help="image scale the model works at")
    parser.add_argument("--min-upper-limit", type=float, default=0.0,
                        help="drop MATLAB tiles from plates whose contrast upper limit is below this "
                             "(low limits give noisy, oversized colony outlines)")
    args = parser.parse_args()
    if (PROJECT_ROOT / "models" / args.name).exists() and not args.force:
        parser.error(f"models/{args.name} exists: choose another --name, or --force to overwrite it")

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    tiles = args.testset / "train" / "tiles.npz"
    if not tiles.exists():
        tiles = build_training_set(args.testset, store, log=lambda m: print(m, flush=True))
    x, labels, ignore, meta = load_training_set(tiles)
    dropped = []
    if args.min_upper_limit > 0:
        upper = {pid: json.loads((args.testset / "plates" / pid / "plate.json").read_text())["limits"][1]
                 for pid in meta.plate_id.unique()}
        low = meta.plate_id.map(upper).lt(args.min_upper_limit) & (meta.source == "matlab")
        dropped = sorted(meta.plate_id[low].unique())
        print(f"dropping {int(low.sum())} tiles from {len(dropped)} plates with upper limit < {args.min_upper_limit}")
        keep = ~low.to_numpy()
        x, labels, ignore, meta = x[keep], labels[keep], ignore[keep], meta[keep].reset_index(drop=True)
    print(f"{len(x)} tiles ({(meta.source == 'gold').sum()} gold)", flush=True)

    models = ResultsStore(PROJECT_ROOT / "models", raw_dirs)
    cfg = UNetConfig(steps=args.steps, scale=args.scale)
    out = train(x, labels, ignore, meta.source.to_numpy(), models.path(args.name, "model.pt").parent, cfg,
                log=lambda m: print(m, flush=True))
    config_path = out.parent / "config.json"
    config = json.loads(config_path.read_text())
    config.update(min_upper_limit=args.min_upper_limit, dropped_plates=dropped)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

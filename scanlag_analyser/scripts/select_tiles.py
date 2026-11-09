"""Choose and export tiles from the MATLAB test set for manual correction.

    python scripts/select_tiles.py [--testset testset_matlab] [--per-category 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.testset.tiles import export_tile, select_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    parser.add_argument("--per-category", type=int, default=8, help="tiles chosen per category (default 8)")
    args = parser.parse_args()

    raw_dirs = json.loads((args.testset / REGISTRY_FILE).read_text())["raw_dirs"]
    store = ResultsStore(args.testset, raw_dirs)
    if (args.testset / "manual").exists():
        raise SystemExit(f"{args.testset / 'manual'} exists; remove it first to re-select tiles")
    specs = select_tiles(args.testset, args.per_category)
    for i, spec in enumerate(specs, 1):
        folder = export_tile(args.testset, spec, i, store)
        print(f"{folder.name}: {spec.plate_id} frame {spec.frame} ({spec.score} targets)")


if __name__ == "__main__":
    main()

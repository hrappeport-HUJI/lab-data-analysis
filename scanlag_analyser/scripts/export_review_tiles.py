"""Export training tiles from a reviewed run: where the review found the detector wrong.

    python scripts/export_review_tiles.py <results folder> [--out DIR] [--per-plate 20]
                                          [--categories excluded,doublet,split,fragment,track]
                                          [--unreviewed] [--include-unreviewed]

Run it after the colony review. Per reviewed plate it writes a 256 px tile around every colony you
excluded by hand, in the category of its flags: possible doublets ("doublet"), irregular splits
("split"), pieces split off a neighbour ("fragment"), jumps and flicker ("track"), or none of these
("excluded": debris, writing, specks). Colonies you included were fine and give no tile.
--unreviewed adds included colonies with segmentation flags you didn't review
(scanlag/train/review_tiles.py).

The tiles go to <results>/training_tiles/ (or --out) in the layout of the MATLAB gold set, so
    python scripts/annotate_tiles.py --testset <results>/training_tiles
opens them in napari for correction, and the corrected tiles can be added to a training run.
Existing tiles are kept (a tile is only written if its folder doesn't exist yet). The raw scans are
only read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scanlag.cli import split_prompt_args
from scanlag.pipeline.run import load_plate_dirs
from scanlag.store import REGISTRY_FILE, ResultsStore
from scanlag.train.review_tiles import CATEGORIES, plate_review_tiles, write_tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="results folder (the pipeline's --out)")
    parser.add_argument("--out", type=Path, help="tile folder (default <results>/training_tiles)")
    parser.add_argument("--per-plate", type=int, default=20, help="tiles per plate at most (default 20)")
    parser.add_argument("--categories", default=",".join(CATEGORIES),
                        help=f"comma-separated, from {', '.join(CATEGORIES)} (default all)")
    parser.add_argument("--unreviewed", action="store_true",
                        help="also included colonies with segmentation flags not reviewed yet")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="also plates never opened in the review window (implies --unreviewed there)")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))

    results = args.results.resolve()
    registry = results / REGISTRY_FILE
    if not registry.exists():
        sys.exit(f"{results} is not a results folder (no {REGISTRY_FILE})")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if unknown := set(categories) - set(CATEGORIES):
        sys.exit(f"unknown categories {sorted(unknown)}: use {', '.join(CATEGORIES)}")
    store = ResultsStore(args.out or results / "training_tiles", json.loads(registry.read_text())["raw_dirs"])

    counts, skipped = Counter(), []
    for d in load_plate_dirs(results):
        review = d / "review.json"
        visited = review.exists() and json.loads(review.read_text()).get("visited", False)
        if not visited and not args.include_unreviewed:
            skipped.append(f"{d.parent.name}/{d.name}")
            continue
        for tile in plate_review_tiles(d, categories, args.per_plate, unreviewed=args.unreviewed or not visited):
            name = f"tile_{tile.scanner}_p{tile.plate}_c{tile.colony_id}_{tile.category}"
            if (store.root / "manual" / name).exists():
                counts["kept"] += 1
                continue
            write_tile(d, tile, store)
            counts[tile.category] += 1
        print(f"{d.parent.name}/{d.name}: done", flush=True)
    if skipped:
        print(f"skipped {len(skipped)} plates never opened in the review window (--include-unreviewed to add them): "
              + ", ".join(skipped))
    new = sum(v for k, v in counts.items() if k != "kept")
    print(f"{new} new tiles in {store.root / 'manual'} ("
          + ", ".join(f"{k} {v}" for k, v in counts.items() if k != "kept") + ")"
          + (f"; {counts['kept']} existing kept" if counts["kept"] else ""))
    print(f"correct them with: python scripts/annotate_tiles.py --testset \"{store.root}\"")


if __name__ == "__main__":
    main()

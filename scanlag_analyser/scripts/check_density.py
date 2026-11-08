"""Quick density check of every plate, to find TNTC plates before the pipeline runs.

    python scripts/check_density.py <scan folder> [--out RESULTS] [--prefix P] [--scanner S]

No detector is used, so it takes seconds per scanner. Each plate's last scan is compared with the
background scan, and the colonies in the working area are estimated from the area that changed
and the size of a typical colony (``scanlag/pipeline/density.py``). Each plate gets a verdict:
TNTC (more than --tntc), dense (more than --dense: expect many merged colonies) or ok.

Writes to the results folder (default <project root>/results/<scan folder name>):

  density/density.csv          one row per plate: coverage, median patch size, estimate, verdict
  density/<scanner>.png        the last scan with each plate circled in its verdict's colour
                               (green ok, orange dense, red TNTC) and its estimate

Nothing is excluded here. Look at the pictures, decide, and pass the plates to leave out to the
pipeline, e.g. ``scanlag_pipeline.py <scans> --exclude 1-4.1u:1-6 --max-colonies 0``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanlag.cli import split_prompt_args
from scanlag.pipeline.density import DENSE, TNTC, DensityConfig, density_check, flagged
from scanlag.pipeline.scans import find_scanners, select_scanners
from scanlag.store import ResultsStore, default_results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans", type=Path, help="folder with the raw scans")
    parser.add_argument("--out", type=Path, help="results folder (default: <project root>/results/<scan folder name>)")
    parser.add_argument("--prefix", action="append", default=[], help="only scanners with this file name prefix")
    parser.add_argument("--scanner", action="append", default=[], help="only this scanner (repeatable)")
    parser.add_argument("--plates", type=int, default=6, help="plates per scanner (default 6)")
    parser.add_argument("--mask-fraction", type=float, default=0.8, help="working area / plate radius (default 0.8)")
    parser.add_argument("--tntc", type=float, default=TNTC, help=f"TNTC above this estimate (default {TNTC})")
    parser.add_argument("--dense", type=float, default=DENSE, help=f"dense above this estimate (default {DENSE})")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))

    raw = args.scans.resolve()
    if not raw.is_dir():
        sys.exit(f"{raw} is not a folder")
    try:
        scanners = select_scanners(find_scanners(raw), args.prefix, args.scanner)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    store = ResultsStore(args.out or default_results_dir(raw), [raw])
    cfg = DensityConfig(n_plates=args.plates, mask_fraction=args.mask_fraction, tntc=args.tntc, dense=args.dense)
    table, pictures = density_check(scanners, cfg)

    store.write_table(table.drop(columns=["cx", "cy", "r"]).round(4), "density", "density.csv")
    for name, picture in pictures.items():
        store.write_image(picture, "density", f"{name}.png")
    print(table[["scanner", "plate", "coverage", "median_blob_px", "est_colonies", "verdict"]]
          .round({"coverage": 3, "median_blob_px": 0, "est_colonies": 0}).to_string(index=False))
    if tntc := flagged(table):
        print("TNTC: " + "; ".join(f"{s} plates {','.join(map(str, p))}" for s, p in tntc.items()))
        print("to leave them out: " + " ".join(f"--exclude {s}:{','.join(map(str, p))}" for s, p in tntc.items()))
    print(f"written to {store.path('density')}")


if __name__ == "__main__":
    main()

"""Quality report of processed plates, to read before the colony review.

    python scripts/scanlag_qc.py <results folder> [--galleries N]

Writes <results>/qc/: qc.csv (one row per plate: the density estimate against the tracked and
whole-plate counts, merges, flags by kind, possible doublets, bad frames), notes.txt (warnings
worth reading first) and index.html (both, with each plate's last frame and colony outlines, and
galleries of flagged colonies over time). Reads the raw scans (read only) for the pictures.
scanlag_pipeline.py runs this by itself after segmenting, --reflag and --retrack (unless --no-qc).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scanlag.cli import split_prompt_args
from scanlag.pipeline.qc import GALLERY_PER_KIND, write_qc
from scanlag.pipeline.run import load_plate_dirs
from scanlag.store import REGISTRY_FILE, ResultsStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="results folder (the pipeline's --out)")
    parser.add_argument("--galleries", type=int, default=GALLERY_PER_KIND,
                        help=f"colonies per flag kind and plate in the galleries (default {GALLERY_PER_KIND}; "
                             "0: tables and outline pictures only)")
    args = parser.parse_args(split_prompt_args(sys.argv[1:]))
    results = args.results.resolve()
    plate_dirs = load_plate_dirs(results)
    if not plate_dirs:
        sys.exit(f"no processed plates in {results}")
    registry = results / REGISTRY_FILE
    if not registry.exists():
        sys.exit(f"{results} is not a results folder (no {REGISTRY_FILE})")
    store = ResultsStore(results, json.loads(registry.read_text())["raw_dirs"])
    write_qc(results, plate_dirs, store, args.galleries)
    notes = (results / "qc" / "notes.txt").read_text().strip()
    print("notes:\n" + notes if notes else "notes: none")


if __name__ == "__main__":
    main()

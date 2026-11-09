"""Build the MATLAB-derived reference test set.

    python scripts/build_testset.py D:\\Alon\\Documents\\Experiments\\microcolony-seq [--out testset_matlab]

Reads raw scans and legacy ROI/data.mat files (never modifies them) and writes into ``--out``.
"""

from __future__ import annotations

import argparse
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scanlag.io.matlab_import import find_analyzed_plates
from scanlag.store import PROJECT_ROOT, ResultsStore
from scanlag.testset.build import build_plate, plate_id


def _build(plate_dir: Path, data_root: Path, out: Path) -> dict:
    store = ResultsStore(out, [data_root])
    try:
        return build_plate(plate_dir, data_root, store)
    except Exception as exc:  # report, drop partial output, continue with the other plates
        shutil.rmtree(store.path("plates", plate_id(plate_dir, data_root)), ignore_errors=True)
        return {"plate_id": str(plate_dir), "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path, help="folder with the experiments' raw scans and MATLAB ROI folders")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    parser.add_argument("--workers", type=int, default=6, help="plates built in parallel processes (default 6)")
    parser.add_argument("--retry-failed", action="store_true", help="only rebuild plates that errored last time")
    args = parser.parse_args()

    store = ResultsStore(args.out, [args.data_root])
    plates = find_analyzed_plates(args.data_root)
    rows = []
    if args.retry_failed:
        previous = pd.read_csv(store.path("index.csv"))
        failed = previous["error"].notna() if "error" in previous else pd.Series(False, index=previous.index)
        retry = set(previous.loc[failed, "plate_id"])
        plates = [p for p in plates if str(p) in retry]
        rows = previous[~failed].to_dict("records")
    print(f"{len(plates)} analyzed plates to build")
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_build, p, args.data_root, args.out) for p in plates]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            status = row.get("error") or f"kept {row['n_kept']}/{row['n_colonies']}, parity {row['parity_exact']}"
            print(f"[{i}/{len(plates)}] {row['plate_id']}: {status}", flush=True)
    index = pd.DataFrame(rows).sort_values("plate_id")
    store.write_table(index.drop(columns=["trace"], errors="ignore"), "index.csv")
    print(f"done; index at {store.path('index.csv')}")


if __name__ == "__main__":
    main()

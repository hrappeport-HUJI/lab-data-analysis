"""ScanLag analysis pipeline: raw scans -> colonies -> review GUI -> result tables.

    python scripts/scanlag_pipeline.py <scan folder> [options]

1. Finds every scanner's time-lapse in <scan folder> (also in subfolders; MATLAB ROI folders are
   skipped). Scans must be named <prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f]. The first scan of each
   scanner is the background. --prefix / --scanner keep only some of the scanners.
2. Segments the last frame of every plate to estimate its density, and offers to exclude plates
   with more than --max-colonies blobs.
3. Finds the plates (default 6 per scanner, numbered as in MATLAB: left column top to bottom,
   then the right column), aligns every frame, segments it and tracks the colonies.
4. Opens the review GUI (PlateAnalyzer-style) to include/exclude colonies, with suspicious tracks
   and shapes flagged.
5. "Finish & export" in the GUI (or --no-review / --export-only) writes the tables of every plate in
   the results folder (only the selected scanners' plates with --prefix / --scanner):
   <scanner>/plate_<k>/ colonies.csv, tracks.csv, area_matrix.csv, and
   all_colonies.csv, all_tracks.csv, summary.csv at the top. Results go to <project root>/results/<folder name>.

The raw folder is only read. Re-running skips the segmentation of scanners that are already
done (use --redo to repeat it), so --review-only or a plain re-run reopens the GUI with the
decisions made so far. The script asks before --redo deletes existing results. Closed input
(no terminal) answers every question with n.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from scanlag.pipeline.export import export
from scanlag.pipeline.run import (RunConfig, last_frame_counts, load_plate_dirs, make_segmenter, model_dir,
                                  process_scanner, refresh_flags, retrack_plate)
from scanlag.pipeline.scans import find_scanners, select_scanners
from scanlag.cli import split_prompt_args
from scanlag.store import ResultsStore, default_results_dir


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The docstring as written, and each option's default after its help."""


def ask(question: str) -> bool:
    """A y/n question in the terminal. Closed input (no terminal) counts as n."""
    while True:
        try:
            answer = input(f"{question} y/n ").strip().lower()
        except EOFError:
            print("(no terminal: answering n)")
            return False
        if answer in ("y", "yes", "n", "no"):
            return answer.startswith("y")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("scans", type=Path, help="folder with the raw scans")
    parser.add_argument("--detector", default="unet",
                        help="unet (U-Net v1, ~0.15 s per plate frame on GPU), cellpose (Cellpose-SAM v1, ~5.5 s per "
                             "plate frame), classical (MATLAB threshold, no GPU) or a trained model folder")
    parser.add_argument("--out", type=Path, help="results folder (default: <project root>/results/<scan folder name>)")
    parser.add_argument("--prefix", action="append", default=[],
                        help="only the scanners whose file name prefix starts with this (repeatable)")
    parser.add_argument("--scanner", action="append", default=[],
                        help="only this scanner, by scanner folder or file name, e.g. 1-4.1u (repeatable)")
    parser.add_argument("--max-colonies", type=int, default=500,
                        help="offer to exclude plates with more blobs than this on the last frame (0 = no check)")
    parser.add_argument("--plates", type=int, default=6, help="plates per scanner")
    parser.add_argument("--mask-fraction", type=float, default=0.8,
                        help="radius of the working area as a fraction of the plate radius (MATLAB: 0.8)")
    parser.add_argument("--threshold", type=float, default=0.2, help="threshold TH of the classical detector")
    parser.add_argument("--tracker", choices=["footprint", "seed"], default="footprint",
                        help="footprint: merged colonies keep their own area; seed: MATLAB, they report the merged blob")
    parser.add_argument("--workers", type=int, default=6, help="plates processed in parallel (1 = one at a time)")
    parser.add_argument("--no-review", action="store_true", help="export without opening the review window")
    parser.add_argument("--overwrite", action="store_true", help="don't ask before --redo deletes existing results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--redo", action="store_true",
                      help="segment again even if results exist (deletes them, review decisions included)")
    mode.add_argument("--review-only", action="store_true", help="no segmentation: open the window on existing results")
    mode.add_argument("--export-only", action="store_true", help="no segmentation, no window: export existing results")
    mode.add_argument("--reflag", action="store_true",
                      help="recompute the review flags of existing results (keeps review decisions)")
    mode.add_argument("--retrack", action="store_true",
                      help="track existing results again with --tracker, from the stored segmentation. Colony "
                           "numbers change, so the review starts afresh (old tables kept as *.bak)")
    args = parser.parse_args(argv)
    if args.review_only and args.no_review:
        parser.error("--review-only with --no-review does nothing but export: use --export-only")
    return args


def to_segment(scanners, store: ResultsStore, redo: bool) -> tuple[list, list, list]:
    """(scanners to segment, of those the ones whose complete results will be deleted, the ones whose
    folder is an interrupted run)."""
    todo, replace, incomplete = [], [], []
    for s in scanners:
        folder = store.root / s.name
        done = folder / "scanner.json"
        if done.exists() and not redo:
            print(f"{s.name}: already segmented with {json.loads(done.read_text())['detector']} (use --redo to repeat)")
            continue
        todo.append(s)
        if done.exists():
            replace.append(s)
        elif folder.exists():
            incomplete.append(s)
    return todo, replace, incomplete


def confirm_redo(replace, store: ResultsStore) -> bool:
    """Ask before --redo deletes complete results, saying how many plates have review decisions."""
    if not replace:
        return True
    reviewed = sum(1 for s in replace for _ in (store.root / s.name).glob("plate_*/review.json"))
    names = ", ".join(s.name for s in replace)
    note = f", including review decisions on {reviewed} plate(s)" if reviewed else ""
    return ask(f"warning: --redo will delete the results of {names}{note}. proceed?")


def dense_plates(scanners, cfg: RunConfig, seg, limit: int) -> dict[str, list[int]]:
    """Plates per scanner whose last frame has more than ``limit`` blobs."""
    dense = {}
    for s in scanners:
        counts = last_frame_counts(s, cfg, seg, log=lambda m: print(m, flush=True))
        print(f"{s.name}: last-frame blobs " + ", ".join(f"plate {k}: {c}" for k, c in counts.items()), flush=True)
        if over := [k for k, c in counts.items() if c > limit]:
            dense[s.name] = over
    return dense


def ask_exclude(dense: dict[str, list[int]], limit: int) -> bool:
    n = sum(map(len, dense.values()))
    listed = " and ".join(f"{name}, plates {', '.join(map(str, plates))}" for name, plates in dense.items())
    print("(counts are blobs on the last frame: merged colonies count once, so a plate has at least that many)")
    return ask(f"warning: {listed} {'is' if n == 1 else 'are'} too dense (n>{limit}). "
               f"expect long runtimes and low quality tracks. exclude these plates?")


def segment(todo, replace, incomplete, store: ResultsStore, cfg: RunConfig, args) -> None:
    """Density check, then segmentation and tracking of every scanner in ``todo``."""
    log = lambda m: print(m, flush=True)
    if not args.overwrite and not confirm_redo(replace, store):
        sys.exit("stopped")
    seg = make_segmenter(cfg, log=log)
    exclude = {}
    if args.max_colonies > 0:
        print("estimating plate density from the last frames...", flush=True)
        dense = dense_plates(todo, cfg, seg, args.max_colonies)
        if dense and ask_exclude(dense, args.max_colonies):
            exclude = dense
    for s in todo:
        if s in incomplete:
            print(f"{s.name}: removing the results of an interrupted run")
        if (store.root / s.name).exists():
            shutil.rmtree(store.path(s.name))
        plate_dirs = process_scanner(s, store, cfg, log=log, seg=seg, exclude=exclude.get(s.name, ()))
        if not plate_dirs:
            continue
        settings = {k: v for k, v in vars(cfg).items() if k != "workers"}  # workers doesn't change the results
        store.write_json({"detector": cfg.detector, "model": model_dir(cfg.detector), "n_scans": len(s.scans),
                          "plates": [d.name for d in plate_dirs], "excluded_plates": exclude.get(s.name, []),
                          "config": settings}, s.name, "scanner.json")


def main() -> None:
    args = parse_args(split_prompt_args(sys.argv[1:]))
    raw = args.scans.resolve()
    if not raw.is_dir():
        sys.exit(f"{raw} is not a folder")
    existing_only = args.review_only or args.export_only or args.reflag or args.retrack
    if not existing_only:  # the detector is only used to segment
        try:
            model_dir(args.detector)
        except ValueError as e:
            sys.exit(str(e))
    store = ResultsStore(args.out or default_results_dir(raw), [raw])
    print(f"results: {store.root}")
    cfg = RunConfig(detector=args.detector, n_plates=args.plates, mask_fraction=args.mask_fraction,
                    threshold=args.threshold, tracker=args.tracker, workers=args.workers)

    filtered = bool(args.prefix or args.scanner)
    scanners = []
    if filtered or not existing_only:
        try:
            scanners = select_scanners(find_scanners(raw), args.prefix, args.scanner)
        except ValueError as e:
            sys.exit(str(e))
    if not existing_only:
        print(f"{len(scanners)} scanner(s): "
              + ", ".join(f"{s.name} ({len(s.scans)} scans, time zero from {s.time_zero})" for s in scanners))
        todo, replace, incomplete = to_segment(scanners, store, args.redo)
        if todo:
            segment(todo, replace, incomplete, store, cfg, args)

    all_plates = load_plate_dirs(store.root)
    selected = {s.name for s in scanners}
    plate_dirs = [d for d in all_plates if d.parent.name in selected] if filtered else all_plates
    if not plate_dirs:
        sys.exit("no processed plates found")
    if args.retrack:
        for d in plate_dirs:
            r = retrack_plate(store, d, cfg.tracker)
            print(f"{d.parent.name}/{d.name}: {r['old_tracker']} -> {cfg.tracker} tracker, "
                  f"{r['old_colonies']} -> {r['colonies']} colonies, {r['flagged']} flagged", flush=True)
    elif args.reflag:
        for d in plate_dirs:
            print(f"{d.parent.name}/{d.name}: {refresh_flags(d)} flagged colonies")
    if hidden := len(all_plates) - len(plate_dirs):
        print(f"note: only the {len(plate_dirs)} plates of the selected scanners are exported; all_colonies.csv, "
              f"all_tracks.csv and summary.csv will leave out the other {hidden}")
    if args.no_review or args.export_only:
        export(store.root, store, plate_dirs=plate_dirs)
        return
    from scanlag.gui.review import run_review

    run_review(plate_dirs, on_finish=lambda: export(store.root, store, plate_dirs=plate_dirs))


if __name__ == "__main__":
    main()

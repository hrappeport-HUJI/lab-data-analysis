"""ScanLag analysis pipeline: raw scans -> colonies -> review GUI -> result tables.

    python scripts/scanlag_pipeline.py <scan folder> [options]

1. Finds every scanner's time-lapse in <scan folder> (also in subfolders; MATLAB ROI folders are
   skipped). Scans must be named <prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f]. The first scan of each
   scanner is the background. --prefix / --scanner keep only some of the scanners.
2. Estimates every plate's density from its last scan, without the detector (as
   check_density.py), and offers to exclude plates with more than --max-colonies colonies in the
   working area (TNTC). --exclude leaves plates out without asking.
3. Finds the plates (default 6 per scanner, numbered as in MATLAB: left column top to bottom,
   then the right column), aligns every frame, segments it and tracks the colonies.
4. Writes the QC report <results>/qc/ (as scanlag_qc.py: counts against the density estimate,
   flags by kind, possible doublets, bad frames, outline pictures and galleries of flagged
   colonies; notes.txt lists what to look at). Also after --reflag and --retrack; not with --no-qc.
5. Opens the review GUI (PlateAnalyzer-style) to include/exclude colonies, with suspicious tracks
   and shapes flagged.
6. "Finish & export" in the GUI (or --no-review / --export-only) writes the tables of every plate in
   the results folder (only the selected scanners' plates with --prefix / --scanner):
   <scanner>/plate_<k>/ colonies.csv, tracks.csv, area_matrix.csv, and
   all_colonies.csv, all_tracks.csv, summary.csv at the top. Each plate's last frame is also
   segmented out to --full-fraction of the plate radius, for the final colony count of the whole
   plate (final_objects.csv per plate; n_final_* columns in summary.csv). Results go to <project root>/results/<folder name>.

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
import time
from pathlib import Path

from scanlag.pipeline.export import export
from scanlag.pipeline.qc import GALLERY_PER_KIND, write_qc
from scanlag.pipeline.density import DensityConfig, density_check
from scanlag.pipeline.run import (FULL_FRACTION, RunConfig, load_plate_dirs, make_segmenter, model_dir,
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
                        help="offer to exclude plates whose estimated colonies in the working area exceed this "
                             "(TNTC; 0 = no check)")
    parser.add_argument("--exclude", action="append", default=[], metavar="SCANNER:PLATES",
                        help="leave these plates out, e.g. 1-4.1u:1-6 or 1-4.2u:2,5 (repeatable)")
    parser.add_argument("--full-fraction", type=float, default=FULL_FRACTION,
                        help="radius of the final whole-plate count, as a fraction of the plate radius "
                             "(beyond about 0.93 the dish wall's reflections start)")
    parser.add_argument("--plates", type=int, default=6, help="plates per scanner")
    parser.add_argument("--mask-fraction", type=float, default=0.8,
                        help="radius of the working area as a fraction of the plate radius (MATLAB: 0.8)")
    parser.add_argument("--threshold", type=float, default=0.2, help="threshold TH of the classical detector")
    parser.add_argument("--tracker", choices=["footprint", "seed"], default="footprint",
                        help="footprint: merged colonies keep their own area; seed: MATLAB, they report the merged blob")
    parser.add_argument("--workers", type=int, default=6, help="plates processed in parallel (1 = one at a time)")
    parser.add_argument("--no-review", action="store_true", help="export without opening the review window")
    parser.add_argument("--no-qc", action="store_true",
                        help="don't write the QC report (<results>/qc/) after segmenting, --reflag or --retrack")
    parser.add_argument("--qc-galleries", type=int, default=GALLERY_PER_KIND,
                        help=f"colonies per flag kind and plate in the QC galleries (default {GALLERY_PER_KIND})")
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
    try:
        args.exclude = parse_exclude(args.exclude)
    except ValueError as e:
        parser.error(str(e))
    if args.review_only and args.no_review:
        parser.error("--review-only with --no-review does nothing but export: use --export-only")
    return args


def parse_exclude(specs: list[str]) -> dict[str, list[int]]:
    """["1-4.1u:1-6", "1-4.2u:2,5"] -> {"1-4.1u": [1, ..., 6], "1-4.2u": [2, 5]}."""
    from scanlag.analysis.conditions import parse_plates

    out: dict[str, list[int]] = {}
    for spec in specs:
        scanner, _, plates = spec.rpartition(":")
        if not scanner or not plates:
            raise ValueError(f"--exclude {spec!r}: expected SCANNER:PLATES, e.g. 1-4.1u:1-6")
        out.setdefault(scanner.strip(), []).extend(parse_plates(plates))
    return out


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


def dense_plates(scanners, cfg: RunConfig, limit: int, store: ResultsStore) -> dict[str, list[int]]:
    """Plates per scanner whose estimated colonies in the working area exceed ``limit`` (no detector:
    scanlag.pipeline.density). The table and overview pictures go to <results>/density/."""
    table, pictures = density_check(scanners, DensityConfig(n_plates=cfg.n_plates, mask_fraction=cfg.mask_fraction,
                                                            tntc=limit))
    store.write_table(table.drop(columns=["cx", "cy", "r"]).round(4), "density", "density.csv")
    for name, picture in pictures.items():
        store.write_image(picture, "density", f"{name}.png")
    dense = {}
    for name, rows in table.groupby("scanner", sort=False):
        print(f"{name}: estimated colonies " + ", ".join(f"plate {r.plate}: {r.est_colonies:.0f} ({r.verdict})"
                                                         for r in rows.itertuples()), flush=True)
        if over := [int(r.plate) for r in rows.itertuples() if r.est_colonies > limit]:
            dense[name] = over
    return dense


def ask_exclude(dense: dict[str, list[int]], limit: int, store: ResultsStore) -> bool:
    n = sum(map(len, dense.values()))
    listed = " and ".join(f"{name}, plates {', '.join(map(str, plates))}" for name, plates in dense.items())
    print(f"(estimates from the last scan; pictures in {store.path('density')})")
    return ask(f"warning: {listed} {'looks' if n == 1 else 'look'} TNTC (estimated n>{limit}). "
               f"expect long runtimes and low quality tracks. exclude these plates?")


MIN_FREE_GB = 2.0  # free disk space needed in the results folder to start segmenting
GB_PER_SCANNER = 0.3  # results of one scanner of 6 plates x 200 scans (pilot 2026_10_15: about 0.1 GB)


def check_disk(store: ResultsStore, n_scanners: int) -> None:
    """Stop before segmenting if the results folder's disk is nearly full (a full disk mid-run leaves
    broken zarr arrays); warn if it is tight."""
    free = shutil.disk_usage(store.root).free / 1e9
    need = max(MIN_FREE_GB, GB_PER_SCANNER * n_scanners)
    if free < need:
        sys.exit(f"only {free:.1f} GB free on the disk of {store.root}; segmenting {n_scanners} scanner(s) needs "
                 f"at least {need:.1f} GB. Free some space and run again")
    if free < 3 * need:
        print(f"note: {free:.1f} GB free on the results disk", flush=True)


def torch_device_name() -> str | None:
    try:
        from scanlag.device import torch_device

        return str(torch_device())
    except Exception:  # noqa: BLE001 - classical detector without torch
        return None


def segment(todo, replace, incomplete, store: ResultsStore, cfg: RunConfig, args) -> None:
    """Density check, then segmentation and tracking of every scanner in ``todo``."""
    log = lambda m: print(m, flush=True)
    if not args.overwrite and not confirm_redo(replace, store):
        sys.exit("stopped")
    check_disk(store, len(todo))
    seg = make_segmenter(cfg, log=log)
    exclude = {name: list(plates) for name, plates in args.exclude.items()}
    if args.max_colonies > 0:
        print("estimating plate density from the last scans...", flush=True)
        dense = dense_plates(todo, cfg, args.max_colonies, store)
        dense = {name: [p for p in plates if p not in exclude.get(name, [])] for name, plates in dense.items()}
        dense = {name: plates for name, plates in dense.items() if plates}
        if dense and ask_exclude(dense, args.max_colonies, store):
            for name, plates in dense.items():
                exclude.setdefault(name, []).extend(plates)
    for s in todo:
        if s in incomplete:
            print(f"{s.name}: removing the results of an interrupted run")
        if (store.root / s.name).exists():
            shutil.rmtree(store.path(s.name))
        t0 = time.time()
        plate_dirs = process_scanner(s, store, cfg, log=log, seg=seg, exclude=exclude.get(s.name, ()))
        if not plate_dirs:
            continue
        settings = {k: v for k, v in vars(cfg).items() if k != "workers"}  # workers doesn't change the results
        store.write_json({"detector": cfg.detector, "model": model_dir(cfg.detector), "n_scans": len(s.scans),
                          "plates": [d.name for d in plate_dirs], "excluded_plates": exclude.get(s.name, []),
                          "device": None if cfg.detector == "classical" else torch_device_name(),
                          "minutes": round((time.time() - t0) / 60, 1),
                          "finished": time.strftime("%Y-%m-%d %H:%M"), "config": settings}, s.name, "scanner.json")


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
                    threshold=args.threshold, tracker=args.tracker, workers=args.workers,
                    full_fraction=args.full_fraction)

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
    changed = bool(not existing_only and todo) or args.reflag or args.retrack

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
    if changed and not args.no_qc:
        write_qc(store.root, plate_dirs, store, args.qc_galleries, log=lambda m: print(m, flush=True))
        notes = store.path("qc", "notes.txt").read_text().strip()
        print("qc notes:\n" + notes if notes else "qc notes: none", flush=True)
    if args.no_review or args.export_only:
        export(store.root, store, plate_dirs=plate_dirs)
        return
    from scanlag.gui.review import run_review

    run_review(plate_dirs, on_finish=lambda: export(store.root, store, plate_dirs=plate_dirs))


if __name__ == "__main__":
    main()

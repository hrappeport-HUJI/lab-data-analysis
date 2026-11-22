---
name: scanlag-analysis
description: Analyze a ScanLag run (flatbed-scanner time-lapse of colonies on agar plates) with ScanLag Analyser, from its scan folder and a free-text description of the plates, ending in appearance-time and growth-rate histograms per condition and whole-plate colony counts. Checks the scans and time zero, finds TNTC plates and asks before excluding them, confirms the plate-to-condition map, runs the segmentation, tracking and QC report, opens the colony review and waits for it, exports training tiles from the review, then builds the histograms with the appearance-vs-growth control. Use whenever someone has a folder of scanner TIFFs (<prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tiff) or asks for colony appearance times, lag, growth rates or colony counts from scanners.
metadata:
  status: draft
  code: scanlag_analyser (this skill's parent folder)
---

# ScanLag analysis

Turn one ScanLag run into reviewed colony tables, appearance-time and growth-rate histograms per condition, and whole-plate colony counts. These instructions are for any agent that can run Python and read and write files. They use no agent-specific tools.

A wrong plate-to-condition map, a wrong time zero or a TNTC plate left in gives plausible histograms that are wrong, and nobody catches them later. So when something is unclear, stop and ask the experimenter rather than guess. Ask everything open at once.

**Local setups.** A lab may wrap this skill in its own skill or notes that say where the code, the environment, the scans and the results live, and what to do with the results. Where those say something different about locations, environment or record-keeping, they win. The procedure and the checks below stay the same.

## Locations

| Name | What it is | Default |
|---|---|---|
| `CODE` | The `scanlag_analyser` folder: `scripts/`, the `scanlag` package, and `README.md`, which documents every option and output. This skill is at `CODE/skills/scanlag-analysis/` | Two levels above this file |
| `RAW` | The scan folder: the scans and the robot's `run_*.log` files, possibly with scanners in subfolders (`CODE/README.md`, section 3). Read only | Ask the experimenter |
| `RESULTS` | Everything the analysis writes: `density/`, `qc/` (QC report), one folder per scanner, `summary.csv`, `training_tiles/`, `analysis/`. It must not be inside `RAW`; the code refuses if it is | A `scanlag_results/` folder next to `RAW`. Without `--out` the scripts write to `CODE/results/<scan folder name>/`, inside the code folder, so always pass `--out RESULTS` |
| `ENV` | A Python environment with `scanlag` installed | The conda env `scanlag` from `CODE/environment.yml` (NVIDIA GPU) or `CODE/environment-cpu.yml` (everything else, including every Mac) |

Run every script with `ENV` active: `python CODE/scripts/…`. If `import scanlag` fails, the environment is missing or broken: ask before creating or rebuilding one (`CODE/README.md`, section 2). Before building, check the free disk space (a full disk leaves a broken env folder); with a conda older than 23.10 build with mamba or micromamba, whose solver is much faster. Check a new env with `python -m pytest -q` in `CODE` (needs `environment-training.yml` on top) and `python -c "from scanlag.device import torch_device; print(torch_device())"`, which prints `cuda`, `mps` or `cpu`.

If a path doesn't exist, ask where it is. Never write into `RAW` or move its files. Don't change `CODE` unless asked.

## 1. Inspect the scans

```
python CODE/scripts/inspect_scans.py RAW
```

Tell the experimenter what you found: the scanners, scans per scanner, the interval, how long the run lasted, gaps, and **where time zero comes from** for each scanner. Stop and ask when:

- A `GAP` line appears, or scanners have very different numbers of scans or end times. A `background pause` line (no scans for a while after the first, background, scan) is the usual setup: mention it, don't stop for it.
- Time zero differs between scanners of one run, or comes from `first scan` for some and a log for others. Time zero is the start of the robot run (the latest `run_*.log` started before the first scan). Ask whether that is the time zero they want, or whether it should be when the plates were plated or went into the incubator, and how long before the first scan that was.
- Files are reported as not following the naming rule.

## 2. Check for TNTC plates

```
python CODE/scripts/check_density.py RAW --out RESULTS
```

It estimates the colonies in each plate's working area from the last scan, without the detector, in seconds per scanner: each changed patch counts as its area over a single colony's size, so clusters count as several colonies. On the developers' pilot, the colonies the pipeline later tracked were 0-20 % above this estimate. Open `RESULTS/density/<scanner>.png` and show the pictures to the experimenter together with the table. Each plate is circled green (ok), orange (dense: many colonies will merge) or red (TNTC, above 500 by default). The plates' handwritten labels are often readable in these pictures, which also helps with step 3.

Propose excluding the TNTC plates and ask for confirmation. Don't exclude anything without it, and don't decide about dense plates alone: merging makes their growth rates less reliable, but they may be the condition of interest. Record the decision as `--exclude SCANNER:PLATES` arguments for step 4.

## 3. Map plates to conditions

The experimenter describes the plates in free text. Plates are numbered per scanner as on the board: plates 1–3 are the left column from top to bottom and 4–6 the right column, as in the `density/` pictures (boards with another number of plates: `--plates`, and check the pictures). Check the description against this list and ask only about what is missing:

- The condition of every plate that stays in, on every scanner. Plates without a condition are left out of the analysis, so say which.
- What the replicates are, and whether plates of one condition on different scanners should be pooled.
- Anything that changes time zero for some plates only (plated at another time, moved between scanners).

Write `RESULTS/conditions.csv`, one row per condition and scanner (`CODE/README.md`, section 6):

```
condition,scanner,plates
CTRL,1-4.3u,1-3
GLU 0.2%,1-4.3u,4-6
```

Show it back as a board per scanner (two columns of three, with each plate's handwritten label if readable) and get a confirmation. A mistake is cheapest to fix here.

## 4. Segment and track

```
python CODE/scripts/scanlag_pipeline.py RAW --out RESULTS --exclude SCANNER:PLATES --max-colonies 0 --no-review
```

`--max-colonies 0` turns off the pipeline's own TNTC question, which step 2 has already settled. `--no-review` writes the tables with automatic statuses; the review comes in step 5. Re-running skips scanners that are done.

Run it in the background. As a guide, on an Apple M1 (GPU) it takes about 9.5 min per scanner of 6 plates × 200 scans; it is faster on an NVIDIA GPU and much slower on a CPU. Within the first minutes, confirm it is making progress (new files in `RESULTS/<scanner>/`) before saying it is running; once the first scanner is done (it prints the time per scanner) give the experimenter an estimate for the rest. `RESULTS/<scanner>/scanner.json` records the device and the minutes. It stops at once if the results disk is nearly full. If it fails with an error naming `mps` or `cuda`, run it again with `SCANLAG_DEVICE=cpu` in front and say so.

The pipeline then writes the QC report and prints its notes. Read `RESULTS/qc/notes.txt` and `RESULTS/qc/qc.csv`, and give the experimenter the notes and the path of `RESULTS/qc/index.html` (outline pictures and galleries of flagged colonies). Explain every note:

- a tracked count far from the density estimate (`tracked_vs_est`): look at that plate's outline picture before going on
- many flagged colonies to review (`n_flags_unreviewed`): that plate needs a careful review
- merge excess (`n_final_full_est` above `n_final_full`): many merged clusters, so growth rates on that plate rest on fewer colonies
- possible doublets: two colonies tracked as one, so the plate holds that many more colonies than counted
- bad frames: a blurred or badly lit scan, ignored by the flags and the growth metrics
- the `control` lines: Spearman rho of appearance time against growth rate per scanner (step 6)

Flags only point the reviewer at colonies; they never change a status. If the code changes after a run (new flag rules), re-apply them with `--reflag` (about a minute, keeps statuses and review decisions); back up the plates' `colonies.parquet`, `tracks.parquet`, `plate.json` and `review.json` first. Never run `--redo` or `--retrack` on reviewed plates without asking: they reset the review.

## 5. Pause for the review

The review is done by a person in a desktop window. Open it for them in a terminal they can see and stop (it keeps running after your turn), or give them the command:

```
python CODE/scripts/scanlag_pipeline.py RAW --out RESULTS --review-only
```

Tell them which plates the QC notes point at. In the window, every flagged colony starts in the "To review" list; they step through it with F (clicking a colony in a list or its curve zooms the plate onto it, Esc shows the whole plate), include (Ctrl/Cmd+I) or exclude (Ctrl/Cmd+E) each, and it moves to "Reviewed" (green included, red excluded); Ctrl/Cmd+U puts a reviewed colony back. Then they press "Finish & export". Wait until they say it is done. `CODE/README.md`, section 5, has all the controls.

- Decisions are saved as they are made, so the window can be closed and reopened. While it is open, don't run anything that writes the plates' tables (`--reflag`, `--retrack`, `--redo`); reading and `--export-only` are fine. After a code change the window has to be closed and reopened to get the new version.
- "Visited" in `summary.csv`'s `reviewed` column only means the plate was opened. `n_flags_unreviewed` shows what is still to review; report it.
- If they choose to skip the review, note in the report that the statuses are automatic.

Then export training tiles from the review: one per colony they excluded, in the category of its flags (doublet, bad split, fragment, jump/flicker, or plain excluded):

```
python CODE/scripts/export_review_tiles.py RESULTS
```

Report the number of tiles per category, and offer to open them for correction (`python CODE/scripts/annotate_tiles.py --testset RESULTS/training_tiles`, needs napari). Corrected tiles are training data for a better segmentation model (`CODE/README.md`, section 12).

## 6. Histograms

```
python CODE/scripts/scanlag_analysis.py RESULTS --growth both --no-open
```

It reads `RESULTS/conditions.csv` and writes `RESULTS/analysis/`: `histograms.html` (appearance time, radial growth rate and doubling time per condition), `summary.csv` per condition, and per-condition colony tables. Defaults: radial growth is the slope of sqrt(area) over each colony's first 12 frames, stopping at its first merge; doubling time is 20 → 80 px. Change them only if the experimenter asks (`--fit-frames`, `--doubling-range`, `--merged-growth split`).

Before reporting, check:

- Every condition has the plates and the scanners you expect (the script prints them). A `warning: no results for` line means a plate in the layout wasn't processed.
- How many included colonies per condition have no growth rate or doubling time (NaN in `analysis/<condition>/colonies.csv`), and why: mostly merges before enough frames, or never reaching 80 px. If a condition loses many more than the others, its growth histogram is biased, so say so.
- Appearance times start after time zero plus the first scans, and conditions differ in the direction the experimenter expects or the difference is pointed out.
- **The appearance-vs-growth control** (printed per condition, and `appearance_vs_growth_rho`, `_p`, `_n` in `analysis/summary.csv`). Appearance time is the first scan in which a colony is detected (a blob of about 10 px), not the end of its lag: on the developers' pilot, colonies were first seen at about 12 px, 1.5-2 h after the start back-extrapolated from their own growth, and the delay is shorter for fast growers. A clearly negative Spearman rho means appearance times partly reflect growth rate, so a difference in appearance between conditions can be a difference in growth rate; say so whenever conditions differ in both. Compare with the per-plate values in `qc/qc.csv` (`lag_growth_rho`): a pooled rho much stronger than the per-plate ones comes partly from differences between plates (pilot: -0.29 pooled, -0.09 to -0.18 per plate).
- Every number you report comes from `analysis/summary.csv` or the colony tables.

Report: scanners, time zero and the run time and device, excluded plates and why, the layout, the QC notes and what was done about them, per condition the plates, colonies, median appearance time and growth rate, colonies without growth estimates, whole-plate counts and possible doublets, bad frames, flagged colonies left unreviewed, the appearance-vs-growth control, the training tiles exported, and the paths of `histograms.html`, `qc/index.html` and the tables.

Pitfalls: read the colony tables' flag column as `df["flags"]` (`df.flags` is a pandas attribute); review decisions live in `colonies.parquet` (`status`, `flags_reviewed`), and `review.json` is only their log.

## 7. Keep a record

Ask whether and where the run should be written up (a lab notebook, an electronic notebook, a results note). If a local setup says how, follow it. Point to `RAW` and `RESULTS` rather than copying scans or result tables into the notes; a static picture of the histograms is usually enough as the key figure.

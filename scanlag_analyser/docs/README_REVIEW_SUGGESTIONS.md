# Usability review: suggested changes

Written by the README sub-agent on 2026-10-22, while it wrote `README.md`. It reviewed the repo for a user
who did not develop it, and was not allowed to change any code. The file:line references are from that day,
so later edits may have moved them.

## Status on 2026-10-26

Re-checked by the README agent on 2026-10-26 (see also `docs/READABILITY_REVIEW_SUGGESTIONS.md`).
Nothing in this list is fully resolved. Two items have moved a little:

| Item | Status | Note |
|---|---|---|
| 3 (`--help` of `scanlag_pipeline.py`) | **partly done** | The new options (`--overwrite`, `--prefix`, `--scanner`, `--max-colonies`, `--workers`) have `help=`. The older ones (`--detector`, `--plates`, `--mask-fraction`, `--threshold`, `--redo`, `--review-only`, `--no-review`, `--reflag`, `--retrack`) still have none, and `--detector` is still not validated (`scripts/scanlag_pipeline.py:83-99`). |
| 4 (`--redo` deletes review decisions without asking) | **partly done** | A y/n question now appears when the results folder is not empty (`scripts/scanlag_pipeline.py:106-108`). It is generic: it does not say that `--redo` deletes review decisions, it also appears when nothing will be deleted, and `--overwrite` skips it. The delete itself is now at line 144-145. |
| 1, 2, 5-29 | open | Line numbers have moved in `run.py` (e.g. item 29's `PREVIOUS_GAP` import is now `run.py:41`, and it is also used by the new `last_frame_counts`), `unet.py` (item 2 is now lines 98-99), `review.py` (item 6 is now `PlateData._read`, line 127-130; item 20 is line 43) and `scanlag_pipeline.py` (item 19 is line 150). |

Test-suite count on 2026-10-26: 56 passed (README updated).

## Status after batch 1 (2026-10-26)

Checked against the working tree after "batch 1" (uncommitted). The README describes the new behaviour.

| Item | Status | What the code does now |
|---|---|---|
| 1 dependencies | **done** | `pyproject.toml`: `dl` adds `segmentation-models-pytorch>=0.5` and `fastremap`, a new `cellpose = ["cellpose>=4"]` extra, and `pydantic` / `pyyaml` are removed. |
| 2 `normalize` from the training module | open | `scanlag/detect/unet.py:98-99` still imports `normalize` from `scanlag.train.cellpose_finetune` and `model_channels` from `scanlag.train.dataset`. |
| 3 `--help`, `--detector` validation | **done** | Every option of `scanlag_pipeline.py` has `help=`, with defaults shown. An unknown `--detector` exits with `unknown detector ...` (`scanlag.pipeline.run.model_dir`). |
| 4 `--redo` deletes without asking | **done** (no backup) | `confirm_redo` asks before deleting complete results and counts the plates with a `review.json`. `--overwrite` skips the question. No backup is made. Interrupted folders are removed with a printed message. |
| 5 training `--name` | **done** | `--name` is required in `train_unet.py` and `train_cellpose.py`; an existing `models/<name>` is refused unless `--force` is given. |
| 6 absolute `raw_dir` | open | |
| 7 standalone review command | open | `--export-only` was added, but review still goes through `scanlag_pipeline.py <scans> --review-only`. |
| 8 LICENSE, model in plain git | open | |
| 9 `models/` registered as a results folder | open | Training still writes through `ResultsStore(PROJECT_ROOT / "models", raw_dirs)`. |
| 10 training docstrings and help texts | **done** | `train_unet.py` docstring fixed; help strings added to the training scripts, `build_testset`, `select_tiles`, `evaluate_goldset`, `track_with_model`, `plot_area_tracks`, `compare_with_matlab`, `annotate_tiles`, `evaluate_plates` and `scanlag_analysis.py --growth`. |
| 11 `--condition` duplicate check | **done** | Both paths go through `conditions.expand_layout`. |
| 12 two growth-time definitions | open | |
| 13 `summary.csv` plate as text | open | |
| 14 `plates.py` docstring | **done** | |
| 15 hard-coded constants | open | The plate error message now at least names the 90 mm / 300 dpi assumption. |
| 16 `ColonyStatus` location | open | |
| 17 time zero | **done** (print only) | The scanner list prints `time zero from first scan` or the log name (`Scanner.time_zero`). Parent folders are still not searched. |
| 18 plate-count error | **done** | It now hints at `--plates`, the plate diameter and the dpi. |
| 19 detector in `scanner.json` | **done** | New `model` key (resolved model folder, `null` for classical). |
| 20 `matplotlib.use` on import | open | `scanlag/gui/review.py:43`. |
| 21 `main.py` | **done** | Deleted (staged). |
| 22 `docs/PLAN.md:4` status | **done** | |
| 23 `tiles.py` docstring `.tif` | **done** | |
| 24 condition folder clash | **done** | `check_folder_names` in `scanlag_analysis.py` stops with an error. |
| 25 `_HHMMSS` scan names | open | |
| 26 shortcuts dialog, undo | open | |
| 27 hard-coded D: path in the test | **done** | `SCANLAG_LEGACY_DATA`, with the D: path as default. |
| 28 log file | open | |
| 29 `PREVIOUS_GAP` | **done** (differently) | Moved to the new `scanlag/detect/channels.py` (still importable from `scanlag.testset.tiles`). `model_channels` stays in `scanlag/train/dataset.py`, so the two are not next to each other. |

When it ran (2026-10-22), the test suite passed (37 tests).

What it did not check:
- It did not launch the review GUI. The GUI section of the README comes from reading `review.py`.
- It did not run the pipeline or the analysis end to end. It checked output columns against the existing
  `results_EPEC260224` files. That folder's `analysis/` also has `*_hist.png` files, which the current code no
  longer writes, so they are left over from an older version.
- It did not test a clean install. The list of missing dependencies comes from reading the imports.
- The speed figures in the README come from docstrings and `docs/PLAN.md`, not from measurement.
- It did not check that `cpsam_v2` is a valid Cellpose model name.

## High priority

1. **`pyproject.toml:6-25,27-30`**: undeclared dependencies.
   - The default U-Net detector needs `segmentation-models-pytorch` (`scanlag/detect/unet.py:36`) and `fastremap`.
     `unet.py:96` imports `scanlag/train/cellpose_finetune.py`, which imports `fastremap` at line 24.
   - Neither is declared, and `cellpose` is not declared either.
   - `pydantic` and `pyyaml` are declared but never imported.
   - **Fix:** add `segmentation-models-pytorch` and `fastremap` to the `dl` extra, add
     `cellpose = ["cellpose>=4"]`, and drop `pydantic` and `pyyaml`.
2. **`scanlag/detect/unet.py:96-99`**: inference takes `normalize` from the training module, which pulls in
   fastremap and Cellpose code.
   - **Fix:** move `normalize` and `model_channels` into a small neutral module such as
     `scanlag/detect/channels.py`.
3. **`scripts/scanlag_pipeline.py:49-57`**: no option has `help=`, so `--help` lists bare flags. The explanations
   are only in the docstring at lines 18-28.
   - An unknown `--detector` is treated as a path and fails with an unclear `FileNotFoundError` on `config.json`
     (`run.py:107`).
   - **Fix:** add `help=` and `choices=` to the options, and validate `--detector`.
4. **`scripts/scanlag_pipeline.py:73-78`**: `--redo` deletes the scanner folder, review decisions included
   (`shutil.rmtree`), without asking. A folder with no `scanner.json` is also deleted without warning.
   - **Fix:** ask first, or first move `colonies.parquet` and `review.json` to a timestamped backup.
5. **`scripts/train_unet.py:23`, `scripts/train_cellpose.py:26`**: the default `--name` is the shipped model's
   name. Training without `--name` overwrites `models/unet_scanlag_v1` (which is in git) or `cpsam_scanlag_v1`.
   - **Fix:** make `--name` required, or refuse to overwrite an existing folder unless `--force` is given.
6. **`scanlag/gui/review.py:116-119`, the `raw_dir` key in `plate.json`**: the GUI reads images from the absolute
   raw path saved at processing time. Moving the data, or a change of drive letter, breaks review.
   - **Fix:** add a `--raw-dir` override, or resolve paths against the registered raw dir.
7. **No standalone review or export command**: `--review-only` still needs the scan folder
   (`scanlag_pipeline.py:48,60-63`).
   - **Fix:** add `scripts/scanlag_review.py <results>`, or `[project.scripts]` entry points
     (`scanlag-pipeline`, `scanlag-review`, `scanlag-analysis`).
8. **Repository basics**: there is no LICENSE (the README is now done). `models/unet_scanlag_v1/model.pt` (98 MB)
   is in plain git, close to GitHub's 100 MB limit.
   - **Fix:** add a LICENSE, and use Git LFS or a release download for the weights.

## Medium priority

9. **`scripts/train_unet.py:48`, `train_cellpose.py:41`, the committed `models/.scanlag_results.json`**: `models/`
   is registered as a results folder tied to the microcolony-seq raw dir. Training from any other test set
   raises `FileExistsError`.
   - **Fix:** write models without registering them in `ResultsStore`, or register `models/` with no raw dirs.
10. **`scripts/train_unet.py:3-6`**: the docstring is out of date.
    - It says tiles are "built by scripts/train_cellpose.py", but the script builds them itself.
    - It shows `--steps 6000`, but v1 was trained with 10000.
    - Options without help text: `--testset`, `--name`, `--steps` and `--scale` here, and options in
      `train_cellpose.py:25-30`, `build_testset.py:34-37`, `select_tiles.py:18-19`, `evaluate_goldset.py:73-75`,
      `track_with_model.py:32-33`, `plot_area_tracks.py:42-44`, and `compare_with_matlab.py:127` (`results`).
    - **Fix:** add help strings and use `description=__doc__` as the main scripts do.
11. **`scripts/scanlag_analysis.py:100-103`**: plates given with `--condition` skip the duplicate-plate check
    that `read_layout` runs (`conditions.py:77-80`).
    - **Fix:** run both paths through one validator.
12. **`scanlag/pipeline/export.py:47-48` vs `scanlag/analysis/conditions.py:106`**: one quantity, two
    definitions under two names.
    - `growth_20_80px_min` keeps zero values and is fixed at 20/80 px.
    - `doubling_time` blanks values ≤ 0 and its range can be changed.
    - **Fix:** use one function and one name.
13. **`scanlag/pipeline/export.py:92`**: `summary.csv` writes `plate` as `"plate_1"`, while every other table
    uses an integer.
14. **`scanlag/pipeline/plates.py:5-6`**: the docstring says plates are numbered by x "rounded to 10 % of the
    scan width". The code (lines 64-74) groups them by gaps in x.
15. **Hard-coded physical constants**: none of these can be changed from the command line.
    - 300 dpi: `conditions.py:36` `UM_PER_PX`, `plates.py:49`.
    - 90 mm plates: `plates.py:22`.
    - Empty-slot gray level 25: `plates.py:164`.
    - Near-border 15 px: `seed_tracker.py:17`.
    - `MIN_AREA` 10: `seed_tracker.py:16`.
    - `min_upper_limit` 0.06: `run.py:51`.
    - Flag thresholds: `flags.py`.
    - **Fix:** add `--dpi`, `--plate-diameter`, `--min-area` and `--border-px`; save them in `plate.json`;
      read `UM_PER_PX` from there.
16. **`scanlag/io/matlab_import.py:20-26`**: `ColonyStatus` is used across the whole pipeline but lives in the
    MATLAB importer.
    - **Fix:** move it to `scanlag/status.py` and re-export it from the old place.
17. **`scanlag/pipeline/scans.py:39-45`**: the `run_*.log` file is only searched for in the scan folder itself,
    and the output never says which time zero was used.
    - **Fix:** also search parent folders, and print where time zero came from (the log or the first scan).
18. **`scanlag/pipeline/plates.py:59-61`**: "found N plate-sized circles, expected 6" gives no hint about what to do.
    - **Fix:** mention `--plates`, the dpi and the plate size.
19. **`scripts/scanlag_pipeline.py:80` vs `run.py:170,217`**: `scanner.json` records the detector as `unet`,
    but `plate.json` records `unet_scanlag_v1`.
    - **Fix:** record both the alias and the model folder it resolved to.
20. **`scanlag/gui/review.py:35`**: `matplotlib.use("QtAgg")` runs on import, so importing the module in a
    notebook switches the plotting backend.
    - **Fix:** set the backend inside `run_review`.

## Low priority

21. **`main.py:1-16`**: leftover PyCharm template code. **Fix:** delete it.
22. **`docs/PLAN.md:4`**: still says "Status: plan only, no code yet".
23. **`scanlag/testset/tiles.py:10-12`**: the docstring says `labels_init.tif` / `labels_corrected.tif`, but the
    code writes `.png` (line 161; `gold.py:57`).
24. **`scripts/scanlag_analysis.py:72-73`**: two condition names that become the same folder name (for example
    "A B" and "A/B") overwrite each other.
    - **Fix:** detect the clash and stop with an error.
25. **`scanlag/pipeline/scans.py:20`**: scan names with seconds (`_HHMMSS`) don't match, although
    `io/raw.py:18` parses them.
    - **Fix:** allow `(\d{4}|\d{6})`.
26. **`scanlag/gui/review.py:340-347`**: there is no list of shortcuts in the app, and no undo for a single change.
    - **Fix:** add a Help → Shortcuts dialog, and Ctrl+Z to undo the last change in the log.
27. **`tests/test_classical.py:10`**: hard-coded path on D:.
    - **Fix:** read it from an environment variable such as `SCANLAG_LEGACY_DATA`.
28. **Logging**: the code uses `print` and keeps no log in the results folder.
    - **Fix:** write a `pipeline.log` (versions, settings, timings) into the results folder.
29. **`scanlag/pipeline/run.py:35`**: the pipeline imports `PREVIOUS_GAP` from `scanlag.testset.tiles`, a
    test-set module.
    - **Fix:** move it next to the model-channel definition.

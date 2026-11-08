# ScanLag Analyser

ScanLag Analyser measures **when bacterial colonies appear** and **how fast they grow** on agar plates
imaged over time by flatbed scanners. It is the Python successor of the MATLAB "ScanLag" code
(`CreateBoardHint`, `CropROI`, `ProcessPlates`, `PlateAnalyzer`, `epec_*.m`, ...).

For every scanner in an experiment it:

1. finds the Petri dishes on the first (background) scan,
2. aligns every later scan to it,
3. detects and segments colonies in each frame (U-Net model by default; Cellpose-SAM or the original
   MATLAB threshold are alternatives),
4. tracks each colony over time (appearance time, area per frame, merges with neighbours),
5. flags suspicious colonies and lets you include or exclude colonies in a review window,
6. exports tidy CSV tables, and
7. groups plates into experimental conditions and draws interactive histograms of appearance time and
   growth rate.

The raw scans are **only ever read**. Every output goes to a separate results folder.

---

## Contents

1. [Workflow at a glance](#1-workflow-at-a-glance)
2. [Requirements and installation](#2-requirements-and-installation)
3. [Input data: the raw scan folder](#3-input-data-the-raw-scan-folder)
4. [Step 1: the pipeline (`scanlag_pipeline.py`)](#4-step-1-the-pipeline-scanlag_pipelinepy)
5. [Step 2: the review window](#5-step-2-the-review-window)
6. [Step 3: condition analysis (`scanlag_analysis.py`)](#6-step-3-condition-analysis-scanlag_analysispy)
7. [Output files reference](#7-output-files-reference)
8. [Concepts: status codes, merges, flags, coordinates, units](#8-concepts-status-codes-merges-flags-coordinates-units)
9. [Comparing with an old MATLAB analysis (`compare_with_matlab.py`)](#9-comparing-with-an-old-matlab-analysis-compare_with_matlabpy)
10. [Running from PyCharm](#10-running-from-pycharm)
11. [Using the code from Python or a notebook](#11-using-the-code-from-python-or-a-notebook)
12. [Models: where they live, training and evaluation](#12-models-where-they-live-training-and-evaluation)
13. [Troubleshooting and FAQ](#13-troubleshooting-and-faq)
14. [Running the tests](#14-running-the-tests)
15. [Project structure](#15-project-structure)

---

## 1. Workflow at a glance

```
 raw scan folder (read only)            <repo>/results/<folder name>/  (by default)
 ───────────────────────────            ──────────────────────────────────────────────────────────────
 <prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tiff
             │
             ▼
 ┌──────────────────────────────┐
 │ 1. scripts/scanlag_pipeline.py│  find plates ─ align ─ segment ─ track ─ flag
 └──────────────────────────────┘      → <scanner>/plate_<k>/ plate.json, mask.png, labels.zarr,
             │                            tracks.parquet, colonies.parquet
             ▼
 ┌──────────────────────────────┐
 │ 2. Review window (opens       │  include / exclude colonies; every click is saved
 │    automatically)             │      → colonies.parquet + review.json
 └──────────────────────────────┘
             │  "Finish & export"
             ▼
      CSV tables per plate + all_colonies.csv, all_tracks.csv, summary.csv
             │
             ▼
 ┌──────────────────────────────┐
 │ 3. scripts/scanlag_analysis.py│  conditions.csv → per-condition tables + histograms.html
 └──────────────────────────────┘      → analysis/
```

A typical session (Windows, from the project folder, with the virtual environment active):

```bat
:: 1 + 2: process the scans, then review the colonies in the window that opens
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224"

:: 3: first run writes a conditions.csv template; fill it in, then run again
python scripts\scanlag_analysis.py EPEC260224
python scripts\scanlag_analysis.py EPEC260224 --growth both
```

Step 1 writes its results to `results/EPEC260224` in the project folder (this repository).
`scanlag_analysis.py` accepts the bare experiment name and looks it up there, unless a folder of that name
exists in the current directory. If you gave the pipeline `--out`, pass that folder instead. The examples are
for the Windows Command Prompt (`cmd`), run from the project folder.

---

## 2. Requirements and installation

### What you need

| Item | Notes |
|---|---|
| Python **3.11 or newer** | `pyproject.toml` says `requires-python = ">=3.11"`. The conda environments use 3.12; the developers' venv uses 3.11.5. |
| Windows, macOS or Linux | Developed and tested on Windows 11. Paths in the examples are Windows paths. |
| conda | [Miniforge](https://github.com/conda-forge/miniforge) is recommended (conda-forge only, free for any use). An existing Anaconda or Miniconda works too. |
| NVIDIA GPU with CUDA (optional) | Strongly recommended for the U-Net and Cellpose detectors, and for training. Everything also runs on the CPU, more slowly. The classical (MATLAB) detector and the analysis need no GPU. Macs with Apple silicon (M1 or later) use their own GPU. |
| Disk space | A few MB to a few tens of MB per plate for results (compressed label stacks). The raw scans are not copied. |
| Access to the raw scans | Needed for processing **and** for the review window, which reads the images from the original location (see [FAQ](#13-troubleshooting-and-faq)). |

### Install with conda (recommended)

Three environment files are in the project folder. All packages come from conda-forge, except the
project itself, `opencv-python-headless` and Cellpose, which come from pip. conda-forge's Cellpose package
includes the Cellpose GUI, which would install a second Qt library and a second OpenCV:

| File | What it installs | Use it on |
|---|---|---|
| `environment.yml` | the pipeline with the default U-Net detector, the review window, the analysis; PyTorch with CUDA | Windows or Linux with an **NVIDIA GPU** |
| `environment-cpu.yml` | the same, with the PyTorch build without CUDA (a much smaller download; on Apple silicon it still uses the Mac's GPU) | any machine **without an NVIDIA GPU**: macOS (Apple silicon and Intel), Windows, Linux |
| `environment-training.yml` | an add-on to either: Cellpose-SAM and pytest | training and evaluating models, the Cellpose detector, running the tests |

Cellpose-SAM is not in the user environment: it is only needed for `--detector cellpose`, whose model is
not shipped (see [Model weights](#model-weights)), and for training.

In a terminal (on Windows: the "Miniforge Prompt" or "Anaconda Prompt"):

```bat
:: 1. get the code (or copy the folder) and go into it
cd C:\path\to\scanlag_analyser

:: 2. create the environment (a few minutes), then activate it.
::    NVIDIA GPU:
conda env create -f environment.yml
::    no NVIDIA GPU (and every Mac):
conda env create -f environment-cpu.yml

conda activate scanlag

:: 3. (optional) add what training needs
conda env update -n scanlag -f environment-training.yml
```

**CUDA version.** With `environment.yml`, conda reads the NVIDIA driver's CUDA version and picks the
newest PyTorch built for it, so an older driver gets an older PyTorch (any version from 2.6 works). Without
an NVIDIA driver, `environment.yml` stops with `nothing provides __cuda`: use `environment-cpu.yml`.

**Which processor runs the networks.** The U-Net and Cellpose run on an NVIDIA GPU (CUDA) if there is one,
else on the GPU of an Apple-silicon Mac (Metal, "MPS"), else on the CPU (`scanlag/device.py`). To choose
yourself, set the environment variable `SCANLAG_DEVICE` to `cuda`, `mps` or `cpu` before running a script,
e.g. `SCANLAG_DEVICE=cpu python scripts/scanlag_pipeline.py ...` on macOS/Linux, or
`set SCANLAG_DEVICE=cpu` first on Windows. The networks compute in bfloat16 on CUDA (as they were trained),
float16 on the Apple GPU and float32 on the CPU (most CPUs emulate bfloat16, ~1000× slower), so areas can
differ from a CUDA run by a pixel here and there. The U-Net post-processing and the tracking run on the CPU
on every machine.

The environment installs the project itself in editable mode (`pip install -e .`, the last line of the
file), so the scripts in `scripts/` can `import scanlag` and code changes take effect without
reinstalling. Keep the project folder where it was when the environment was created, or run
`pip install -e .` again from its new place.

To update after pulling a new version: `conda env update -n scanlag -f environment.yml --prune`
(add `environment-training.yml` again afterwards if you use it). To start over:
`conda env remove -n scanlag`.

### Alternative: pip and a virtual environment

Without conda, use Python 3.11 or newer and the dependency groups of `pyproject.toml`:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

:: (NVIDIA GPU only) the CUDA build of PyTorch first; see pytorch.org for the right index URL
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -e ".[gui,dl]"
:: (optional) training and tests
pip install -e ".[cellpose,dev]"
```

On macOS/Linux, activate with `source .venv/bin/activate`. Without the `--index-url` line, pip gives the
CPU build on Windows, a CUDA build on Linux, and the CPU/Apple build on macOS (Apple silicon only; PyTorch
publishes no pip wheels for Intel Macs).

### Optional dependency groups (from `pyproject.toml`)

These are for the pip route; the conda files install the same packages (`environment.yml` = always + `gui`
+ `dl`; `environment-training.yml` adds `cellpose` + `dev`). `tests/test_environment.py` checks that the
files and `pyproject.toml` list the same packages, and that no file asks conda for a package that brings a
second Qt library (PySide6) or a second OpenCV: `matplotlib` (use `matplotlib-base`), `cellpose` or
`segment-anything` (from pip instead).

| Group | Installs | Needed for |
|---|---|---|
| *(always)* | numpy, scipy, scikit-image, opencv-python-headless, tifffile, imagecodecs, zarr>=3, pandas, pyarrow, pillow, matplotlib, tqdm, plotly | the classical detector, tracking, export, analysis, histograms |
| `gui` | `napari[pyqt5]` | the review window (it uses PyQt5, which comes with this) and `annotate_tiles.py` (napari) |
| `dl` | torch, torchvision, segmentation-models-pytorch>=0.5, fastremap | the default **U-Net** detector, the Cellpose detector (with `cellpose`), training |
| `cellpose` | cellpose>=4 | the Cellpose detector, Cellpose fine-tuning, `evaluate_goldset.py --detectors cellpose` |
| `dev` | pytest | running the tests |

`environment.yml` was tested in October 2026 on Windows 11 with an RTX 4080: it installed PyTorch 2.14.1
(CUDA 13.4), all tests passed, and the U-Net gave the same appearance times as the developers' venv, with
identical areas on 97 % of the track rows (the others mostly 1 px, at most 35 px: rounding differences
between PyTorch versions). The developers' venv has: torch 2.6.0+cu124, segmentation_models_pytorch 0.5.0, cellpose 4.2.1.1,
napari 0.9.2, PyQt5 5.15.11, numpy 2.4, pandas 3.0, zarr 3.1, plotly 7.1.

### Check the installation

```bat
conda activate scanlag
python -c "import torch; from scanlag.device import torch_device; print('torch', torch.__version__, '- runs on:', torch_device())"
python scripts\scanlag_pipeline.py --help
:: with environment-training.yml installed:
python -m pytest -q
```

### Model weights

The detectors load trained weights from the `models/` folder of the project:

| `--detector` value | Folder | In git? |
|---|---|---|
| `unet` (default) | `models/unet_scanlag_v1/` (`model.pt` ≈ 98 MB, `config.json`, `history.json`) | yes |
| `cellpose` | `models/cpsam_scanlag_v1/` (`model.pt` ≈ 1.2 GB) | **no** (git-ignored; copy it from the developers) |
| any folder path | that folder, which must hold `config.json` and `model.pt` | e.g. `models/unet_scanlag_v2` (git-ignored) |

The folder names are resolved relative to the project folder, not the current directory
(`scanlag/pipeline/run.py`, `MODELS`). A model folder is recognised as a U-Net if its `config.json` has
`"type": "unet"`; otherwise it is loaded as a fine-tuned Cellpose model (`scanlag.detect.unet.load_detector`).
The GPU (NVIDIA, or Apple silicon) is used automatically, see
[which processor runs the networks](#install-with-conda-recommended).

---

## 3. Input data: the raw scan folder

### File names

The pipeline looks (recursively) for files named

```
<prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif      or  .tiff   (case-insensitive)
```

for example `epec-ctrl1_1-4.1u_20260224_1336.tiff` → prefix `epec-ctrl1`, scanner `1-4.1u`,
taken on 24 Feb 2026 at 13:36. Rules (from `scanlag/pipeline/scans.py`):

* The **scanner** is the text between the last two underscores before the date. It may not contain `_`.
  The prefix may. The prefix may also be empty (`_1-4.1u_20261015_1307.tiff`) or missing together with
  its underscore (`1-4.1u_20261015_1307.tiff`).
* Scan times come **from the file name**, not from file dates, so copying the data is safe. Times have
  minute resolution; names with seconds (`_HHMMSS`) are *not* picked up by the pipeline.
* Files that do not match (e.g. `*_motions.mat`, `*.log`) are ignored. Folders named `ROI` (MATLAB crops)
  are skipped.
* Scans of each scanner are sorted by time. **The first scan of each scanner is the background**
  (taken before colonies are visible); every later scan is compared with it.

### Folder layout

Either all scans in one folder, or one sub-folder per condition/scanner. A real example:

```
EPEC260224\                         ← pass this folder to the pipeline
  CTRL1\
    epec-ctrl1_1-4.1u_20260224_1336.tiff     (background)
    epec-ctrl1_1-4.1u_20260224_1536.tiff
    ...
    run_epec-ctrl120260224_1336.log          (scan robot log, optional)
    ROI\                                      (old MATLAB output, ignored)
  CTRL2\ ...
  AMP7\ ...
```

Each (folder, scanner) pair is processed as one time-lapse. The results sub-folder is named after the
scanner (`1-4.1u`). If the same scanner name occurs in two folders, the folder name is put in front
(`CTRL1_1-4.1u`).

### Time zero

Times are reported in minutes since time zero, which is the earlier of

* the `YYYY/MM/DD HH:MM:SS - Starting.` line of a `run_*.log` file **in the same folder as the scans**, and
* the first scan.

The pipeline prints which one it used for each scanner (`time zero from first scan` or
`time zero from run_<...>.log`). Only the scan folder itself is searched for the log, not its parent folders.

### Image assumptions (hard-coded)

* 8-bit RGB TIFFs (extra channels are ignored). Uncompressed TIFFs are memory-mapped; compressed ones
  are decoded into memory.
* **300 dpi** (used to compute the expected plate radius and the µm conversion).
* **90 mm** round plates (`PLATE_DIAMETER_MM`), by default **6 plates per scanner** (`--plates`).
* An empty holder slot (median gray inside the plate < 25, i.e. as dark as the scanner lid) is skipped
  and reported as an empty slot.

---

## 4. Step 1: the pipeline (`scanlag_pipeline.py`)

**Purpose:** raw scans → plates → colonies → review window → CSV tables.

```
python scripts/scanlag_pipeline.py <scan folder> [--detector unet|cellpose|classical|<model dir>]
                                   [--out DIR] [--prefix P ...] [--scanner NAME ...] [--max-colonies N]
                                   [--plates N] [--mask-fraction F] [--threshold TH] [--tracker footprint|seed]
                                   [--workers N] [--no-review] [--overwrite]
                                   [--redo | --review-only | --export-only | --reflag | --retrack]
```

`python scripts\scanlag_pipeline.py --help` lists every option with its default.

### Arguments

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `scans` (positional) | folder | *required* | Folder with the raw scans (searched recursively). |
| `--detector` | text | `unet` | `unet` = U-Net v1 (`models/unet_scanlag_v1`, about 0.15 s per plate per frame on GPU); `cellpose` = fine-tuned Cellpose-SAM v1 (`models/cpsam_scanlag_v1`, about 5.5 s per plate per frame); `classical` = MATLAB's threshold detector (no GPU or torch needed); or the path of any trained model folder (one with a `config.json`). Anything else stops with `unknown detector ...`. |
| `--out` | folder | `<project folder>/results/<scan folder name>` | Results folder. Must not be inside the raw-data folder (and vice versa). |
| `--prefix` | text, repeatable | all | Only the scanners whose file name prefix starts with this (case-insensitive), e.g. `--prefix epec-amp`. |
| `--scanner` | text, repeatable | all | Only this scanner, e.g. `--scanner 1-4.1u` (case-insensitive). It matches either the results sub-folder name (`CTRL1_1-4.1u`) or the scanner name from the file names (`1-4.1u`, which selects all folders with that scanner). With `--prefix`, a scanner must match both. |
| `--max-colonies` | integer | `500` | Before segmenting, the last frame of every plate is segmented once and its blobs of at least 10 px are counted (`<scanner>: last-frame blobs plate 1: 74, ...`). Colonies that have merged by then count once, so the true number of colonies is at least that. If plates have more blobs than this, the script asks `warning: <scanner>, plates ... are too dense (n>500). expect long runtimes and low quality tracks. exclude these plates? y/n`. Answering `n`, or no terminal, keeps them. Excluded plates get no results folder and are recorded in `scanner.json` as `excluded_plates`. `0` turns the check off. The check only covers scanners segmented in this run (not those already done, and not with `--review-only`, `--export-only`, `--reflag` or `--retrack`). |
| `--plates` | integer | `6` | Number of plates per scanner. |
| `--mask-fraction` | number | `0.8` | Radius of the working area (where colonies are counted) as a fraction of the plate radius. 0.8 is the MATLAB default. |
| `--threshold` | number | `0.2` | Threshold `TH` of the **classical** detector only (ignored by models). |
| `--tracker` | `footprint`, `seed` | `footprint` | `footprint`: each colony keeps its own pixels, and a blob of merged colonies is split between them, so every colony keeps its own area after merging. `seed`: MATLAB's seed-pixel tracker, where merged colonies all report the merged blob's area (section 8, Merges). |
| `--workers` | integer | `6` | Plates processed in parallel threads (`1` = one after another). The results do not depend on it; lower it if memory is short. |
| `--no-review` | flag | off | Do not open the window; export the tables straight away (plates not reviewed yet keep their automatic statuses). Cannot be combined with `--review-only` (use `--export-only`). |
| `--overwrite` | flag | off | Do not ask before `--redo` deletes existing results (for unattended runs). |
| `--redo` | flag | off | Segment again even if a scanner was already done. **Deletes that scanner's results folder, including your review decisions.** First asks `warning: --redo will delete the results of A, B, including review decisions on N plate(s). proceed? y/n` (unless `--overwrite`); `n` or no terminal stops. No backup is made. |
| `--review-only` | flag | off | Skip segmentation; open the review window on the existing results. |
| `--export-only` | flag | off | Skip segmentation and the window; export the existing results. |
| `--reflag` | flag | off | Recompute the review flags of existing results (keeps statuses and review decisions), then continue as usual (window, or export with `--no-review`). No segmentation is done. |
| `--retrack` | flag | off | Track existing results again with `--tracker` (default `footprint`), from the stored segmentation (`labels.zarr`): no scans are read and no model runs, a few seconds per plate. Then continue as usual (window, or export with `--no-review`). **Colony numbers change, so each plate's review starts afresh**; the old `colonies.parquet` and `review.json` are kept as `colonies.<old tracker>.bak.parquet` and `review.<old tracker>.bak.json`. |

`--redo`, `--review-only`, `--export-only`, `--reflag` and `--retrack` are mutually exclusive: give at most
one of them.

**Scanner filters (`--prefix`, `--scanner`)** apply to every step: the density check, the segmentation,
`--reflag`, `--retrack`, the review window and the export. "Finish & export", `--no-review` and
`--export-only` write the CSV tables of the selected scanners' plates only, and `all_colonies.csv`,
`all_tracks.csv` and `summary.csv` then hold only those plates (an export without a filter writes them for
all plates again). When the results folder holds other plates, the script says so (`note: only the N plates
of the selected scanners are exported; all_colonies.csv, all_tracks.csv and summary.csv will leave out the
other M`). After every export it lists the plates that were never opened in the review
window (`K of them were never opened in the review window (automatic statuses): ...`).

**Questions and unattended runs.** The script asks y/n only before `--redo` deletes results and when plates
are too dense. Without a terminal (closed input) it prints `(no terminal: answering n)`: a `--redo` then
stops, and dense plates are kept. Use `--overwrite` and `--max-colonies 0` (or a higher limit) for batch jobs.

### What happens

1. **Scans** are found and grouped by scanner (section 3), and filtered by `--prefix` / `--scanner`. The
   script prints each scanner with its number of scans and where its time zero came from. Scanners that
   already have results (a `scanner.json`) are skipped (unless `--redo`). For the others, the last frame of each
   plate is segmented to estimate its density, and you may exclude very dense plates (`--max-colonies`).
2. **Plates** are found with a Hough circle transform on the background scan, refined by a least-squares
   fit to the plate rim, and numbered like MATLAB's `CreateBoardHint`: by column from left to right, and
   top to bottom within a column (on a 2 × 3 board, plates 1-3 are the left column).
3. **Alignment:** for each plate and frame, an integer (dx, dy) shift relative to the background is
   measured (phase correlation of edges, then refined ±1 px).
4. **Segmentation** of each plate crop with the chosen detector. The models see three channels:
   gray(frame − background), gray(frame) and gray(frame 4 scans earlier). The classical detector uses
   MATLAB's contrast limits from the plate's **last** frame, with a floor of 0.06 on the upper limit so
   that empty plates are not flooded with noise.
5. **Tracking** (all detectors): a colony is born when a blob of at least **10 px** overlaps no existing
   colony; its "seed" is the rounded centroid of that first blob. In later frames, a blob belongs to the
   colonies whose pixels of the previous frame it overlaps, and a blob of several colonies is split between
   them (footprint tracker, the default; section 8, Merges). With `--tracker seed`, the blob covering the
   seed belongs to the colony, as in MATLAB.
6. **Status and flags** are set automatically (section 8).
7. The **review window** opens with all plates in the results folder (only those of the selected scanners
   if `--prefix` / `--scanner` was given).

Re-running the same command skips scanners that are already segmented (it prints
`already segmented with ... (use --redo to repeat)`) and reopens the window with the decisions made so far,
without asking anything. A scanner folder left without `scanner.json` (e.g. an interrupted run) is deleted
and redone without a question; the script prints `<scanner>: removing the results of an interrupted run`.

Plates excluded by the density check are not processed later on a plain re-run, because their scanner
counts as done. To process them after all, run that scanner again with
`--redo --scanner <name> --max-colonies 0` (this deletes the scanner's review decisions).

### Examples

```bat
:: default: U-Net detector, results in results\EPEC260224 (project folder), then the review window
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224"

:: MATLAB-style threshold detector, results somewhere else, 4 plates per scanner
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --detector classical --out "E:\results\EPEC260224_classical" --plates 4

:: reopen the review window later (same scan folder, same --out if you used one)
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --review-only

:: only one scanner, and only the scanners of the epec-amp7 files
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --scanner 1-4.1u
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --prefix epec-amp7

:: batch run without a window, using another trained model
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --detector models\unet_scanlag_v2 --no-review

:: write the CSV tables again from the existing results (no segmentation, no window)
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --export-only
```

Rough speed (from the developers' notes): EPEC260224, 4 scanners and 371 scans, took 11.5 min with the
U-Net on an RTX 4080.

---

## 5. Step 2: the review window

The review window (`scanlag/gui/review.py`, PyQt5 + matplotlib) is modelled on MATLAB's `PlateAnalyzer`.

**How to open it:** it opens automatically at the end of `scanlag_pipeline.py`, or with

```bat
python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260224" --review-only
```

There is no separate review command; from Python you can call
`scanlag.gui.review.run_review(plate_dirs)` (section 11). The window reads the images from the **raw
scans** at the path stored in each `plate.json`, so the raw data must be reachable.

### Layout

* **Top bar:** plate selector with ◀ ▶, *Find colony no.* box (type a number, press Enter), buttons
  *Exclude (Ctrl+E)*, *Include (Ctrl+I)*, *Exclude all*, *Reset plate*, *◀ flagged*, *flagged ▶ (F)*,
  check boxes, and *Finish & export*.
* **Left:** the plate image (last frame by default), with a time slider under it.
  Below that: the image layer (*Raw* or *Difference (frame − background)*, which is what MATLAB
  thresholds), *Black* / *White* level sliders and *Auto contrast*.
* **Middle:** area of each colony over time (log scale by default); a dashed line marks the current frame.
  Each panel has a matplotlib toolbar for zoom and pan (while zoom/pan is active, clicks do not select).
* **Right:** list of **flagged colonies** (check these first), with a box above it to show one kind of flag only and *Exclude selected* / *Exclude all listed* buttons below it, and details of the selected colony
  (status, detected status, appearance time, final area, first merge, flags).

Click a colony on the plate, or its curve in the graph, to select it (cyan ring, thick curve).

### Keyboard shortcuts

| Key | Action |
|---|---|
| Ctrl+E | Exclude the selected colony |
| Ctrl+I | Include the selected colony (also works for a colony excluded automatically, e.g. near the border) |
| Left / Right | Previous / next frame |
| Home / End | First / last frame |
| Page Up / Page Down | Previous / next plate |
| F / Shift+F | Next / previous flagged colony |
| Enter (in *Find colony no.*) | Select that colony |

### Check boxes

| Box | Default | Effect |
|---|---|---|
| Outlines | on | Draw colony outlines on the image |
| Numbers | on | Draw colony numbers (hidden while dragging the slider) |
| Labels only | off | Show only the segmented blobs (MATLAB's analysis view) |
| Show excluded | off | Also plot curves of non-included colonies (dotted) |
| Flagged only | off | Plot only the colonies in the flagged list (follows the flag-kind box) |
| Log scale | on | Log y axis for the area graph |

### Colours

| Number colour | Meaning |
|---|---|
| white | included |
| red | included, appears in this frame |
| orange with "!" | included and flagged |
| magenta | near the border (excluded automatically) |
| yellow, bold | excluded by you |
| gray | zero area in the last frame (excluded automatically) |

Outlines use a stable colour per included colony; magenta = near border, yellow = excluded, gray = zero
final area, dark gray = a blob that is not a colony (too small or never tracked). Merged colonies keep their
own colour; with "Labels only", the line where their blob was split is white. On plates tracked with
`--tracker seed`, **white = a blob holding several colonies (merged)**. The area outside the working-area circle is darkened.

### Bulk actions

* **Exclude all** sets every *included* colony on the current plate to excluded (asks first). Colonies
  already excluded automatically keep their status.
* **Flagged list:** choose a flag kind in the box above the list (e.g. *large at appearance*) to list only those
  colonies. *Exclude selected* excludes the colonies selected in the list (Ctrl+click / Shift+click to select
  several); *Exclude all listed* excludes every included colony the list shows. Both ask first and are logged
  in `review.json` as `exclude flagged: <kind>`.
* **Reset plate** undoes all your changes on the current plate and returns to the automatic statuses
  (asks first).

### What is saved, and where

Every change is saved **immediately**; you can close the window at any time and reopen it later.

* `<results>/<scanner>/plate_<k>/colonies.parquet`: columns `status` and `status_name` are updated
  (`auto_status` keeps the original automatic status).
* `<results>/<scanner>/plate_<k>/review.json`: `{"visited": true/false, "log": [...]}`, an append-only log.
  Single changes are logged as `{"time", "colony_id", "from", "to"}`, bulk actions as
  `{"time", "action": "exclude all" | "reset", "colony_ids": [...]}`. `visited` becomes true when the
  plate is opened.

**Finish & export** (warns if some plates shown in the window were never opened) writes the CSV tables for
the plates in the window (section 7): all plates in the results folder, or only the selected scanners' with
`--prefix` / `--scanner`; then it closes the window. The terminal then lists every exported plate that was never opened in
the window (those keep their automatic statuses). You can also export without the window with
`--export-only`, which skips segmentation and goes straight to export.

---

## 6. Step 3: condition analysis (`scanlag_analysis.py`)

**Purpose:** group reviewed plates into experimental conditions, compute per-colony growth metrics and
draw per-condition histograms. Only **included** colonies (status 0) are used, read from the current
`colonies.parquet` files, so the latest review decisions always apply (you do not need to export first).

```
python scripts/scanlag_analysis.py <results folder | experiment name> [--layout CSV | --condition NAME=scanner:plates ...]
                                   [--out DIR] [--growth radial|doubling|both] [--fit-frames FIRST-LAST]
                                   [--doubling-range LB UB] [--bin MIN] [--rate-bin W] [--no-open]
```

### Arguments

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `results` (positional) | folder or name | *required* | The results folder written by the pipeline, or just the experiment name (e.g. `EPEC260224`): a bare name that is not a folder in the current directory is looked up in `results` in the project folder. |
| `--layout` | file | `<results>/conditions.csv` | Condition layout CSV (see below). |
| `--condition` | text, repeatable | none | Layout on the command line instead of a file: `NAME=scanner:plates[;scanner:plates]`, one per condition. When given, `--layout` is not read. The same checks as for `conditions.csv` apply (a plate in two conditions is an error). |
| `--out` | folder | `<results>/analysis` | Output folder. |
| `--growth` | `radial`, `doubling`, `both` | `radial` | Which growth histogram(s) to draw. The tables always contain both metrics. |
| `--merged-growth` | `exclude`, `split` | `exclude` | `exclude`: both growth metrics stop at the colony's first merge (as MATLAB). `split`: also use the merged frames, with the colony's own share of the merged blob (footprint tracker; plates tracked with `--tracker seed` fall back to `exclude` with a warning). |
| `--fit-frames` | `FIRST-LAST` | `1-12` | Frames of each colony used for the radial-growth fit, counted from its appearance (1 = appearance frame). Must satisfy 1 ≤ FIRST < LAST. |
| `--doubling-range` | two numbers | `20 80` | Area range LB UB (pixels) for the doubling time. Must satisfy 0 < LB < UB. |
| `--bin` | number (min) | the scan interval | Bin width of the appearance and doubling-time histograms. The default is the median interval between scans. |
| `--rate-bin` | number | `0.05` | Bin width of the radial-growth histogram (px^0.5/h). |
| `--no-open` | flag | off | Do not open `histograms.html` in the browser. |

### The condition layout (`conditions.csv`)

One row per (condition, scanner) pair. Columns `condition`, `scanner`, `plates` (header names are
case-insensitive). `plates` takes a comma-separated list and/or ranges; quote it when it contains commas:

```csv
condition,scanner,plates
CTRL,1-4.1u,"1,2,4,5"
CTRL,1-5.1u,1-3
AMP5,1-4.1u,"3,6"
AMP5,1-5.1u,4-6
```

* A condition can span several scanners; a scanner's plates can be split over several conditions.
* Rows with an empty `condition` are ignored (so you can leave plates out).
* A plate listed in two conditions is an error.
* A listed plate without results (an empty slot, or not processed) is skipped with a warning.
* Two condition names that would get the same output folder (e.g. `A B` and `A/B`, see "Condition folder
  names" below) are an error; rename one.
* The `scanner` column takes scanner folders, i.e. the results sub-folder names (e.g. `1-4.1u`, or
  `CTRL1_1-4.1u` when a scanner name occurs in two scan folders; see the glossary in section 8). The script prints
  `plates found: ...` at the start to help you.

**If `conditions.csv` does not exist**, the script writes a template there (one row per scanner, with all
its plates and an empty condition) and stops:

```
no condition layout yet: wrote a template to ...\conditions.csv
fill in the condition column (split a scanner's plates over several rows if needed) and run again
```

Open it in Excel or a text editor, fill in the `condition` column, save as CSV and run again.

### Metric definitions (per included colony)

| Metric | Column(s) | Definition |
|---|---|---|
| Appearance time | `appearance_min`, `appearance_h`, `appearance_frame` | Time of the first frame in which the tracker assigns the colony a blob (for a new colony this requires a blob of ≥ 10 px). Measured from time zero (section 3). |
| Radial growth rate | `radial_growth_sqrtpx_per_h`, `radial_fit_r2`, `radial_fit_points` | Slope of a least-squares straight line through **sqrt(area in px)** against time (h), using the colony's frames `--fit-frames` counted from appearance, and only frames **before its first merge** (unless `--merged-growth split`). Frames where the colony is not detected simply contribute no point. Needs at least 4 points (else empty). R² and the number of points are stored. |
| Radial growth in µm/h | `radial_growth_um_per_h` | The same slope converted to the growth of the radius of a disc of equal area: slope / √π × 84.67 µm/px (300 dpi). |
| Doubling time | `doubling_time_min`, `doubling_time_h` | Time from the first frame with area ≥ LB to the first frame with area ≥ UB (default 20 → 80 px: the radius doubles; MATLAB `getAppearanceGrowth`). Only frames where the colony is not merged count. Empty (NaN) if the colony merged before reaching UB, never reached UB, or crossed both thresholds between two scans. |

**Merges:** a colony is merged in a frame when it shares a detected blob with another colony. By default
(`--merged-growth exclude`) merged frames are excluded from both growth metrics, and the old MATLAB rule
(drop colonies that merge before reaching UB) is kept for the doubling time. With `--merged-growth split`,
the footprint tracker's split areas are used instead, so colonies that touch a neighbour early still get
growth values. `first_merged_min` gives the time of the first merge.

### Outputs (`<results>/analysis/` by default)

| File | Content |
|---|---|
| `<condition>/colonies.csv` | One row per included colony of that condition (columns below). |
| `<condition>/area_tracks.csv` | Area per colony and frame, long format: `condition, scanner, plate, colony_id, frame, time_min, time_h, area_px, x, y, merged`. |
| `<condition>/area_matrix.csv` | One row per colony (`condition, scanner, plate, colony_id`), one column per frame (`frame_0`, `frame_1`, ...), area in px, 0 when absent. Frame numbers are only comparable within a scanner (each scanner has its own scan times; use `area_tracks.csv` for times). |
| `all_colonies.csv` | All conditions together. |
| `summary.csv` | Per condition: `n_scanners`, `n_plates`, `n_colonies`, and for `appearance_min`, `radial_growth_sqrtpx_per_h`, `radial_growth_um_per_h`, `doubling_time_min` the `_n`, `_mean`, `_median`, `_std`. |
| `settings.json` | `fit`: the fit settings (`fit_first`, `fit_last`, `min_points`, `lb`, `ub`, `merged_growth`); `layout`: the layout used. |
| `histograms.html` | Interactive histograms (opens in the browser; self-contained, works offline). |

Columns of `colonies.csv` / `all_colonies.csv`: `condition, scanner, plate, colony_id, appearance_frame,
appearance_min, appearance_h, radial_growth_sqrtpx_per_h, radial_growth_um_per_h, radial_fit_r2,
radial_fit_points, doubling_time_min, doubling_time_h, first_merged_min, final_area_px, seed_x, seed_y,
n_flags, flags`.

Condition folder names are the condition names with spaces and `<>:"/\|?*` replaced by `_`. If two conditions
would get the same folder, the script stops with `conditions that would share an output folder, ...`.

### Reading the histograms

`histograms.html` contains one figure per metric: **appearance time (min)** always, then the **radial
growth rate** (`--growth radial`, default), the **doubling time (min)** (`--growth doubling`), or both.

* One panel per condition, stacked, in the order of the layout, sharing the same bins and x axis.
* The y axis is the **fraction** of that condition's colonies in each bin (so conditions with different
  colony numbers can be compared).
* The dashed vertical line is the **median**; the note in the corner gives the median and *n*.
* Time bins are one scan interval wide and centred on the scan times, so every scan falls in its own bin.
* The x axis runs from 0 to 1.1 × the largest value; drag to zoom, double-click to reset, hover a bar for
  its range and colony count.

The script also prints the summary table (median appearance, median growth, n per condition) at the end.

### Examples

```bat
:: first run: writes results\EPEC260224\conditions.csv and stops; fill it in, then:
python scripts\scanlag_analysis.py EPEC260224

:: both growth histograms, fit over frames 2-10 after appearance, 30 min time bins
python scripts\scanlag_analysis.py EPEC260224 --growth both --fit-frames 2-10 --bin 30

:: a results folder written with --out is passed as a path
python scripts\scanlag_analysis.py "E:\results\EPEC260224_classical"

:: conditions on the command line, written to a separate folder, no browser
python scripts\scanlag_analysis.py EPEC260224 --condition "CTRL=1-4.1u:1-3;1-5.1u:1-3" --condition "AMP7=1-4.1u:4-6;1-5.1u:4-6" --out "results\EPEC260224\analysis_v2" --no-open
```

---

## 7. Output files reference

Default results folder: `<project folder>/results/<scan folder name>/`.

```
EPEC260224/                      e.g. <project folder>\results\EPEC260224
  .scanlag_results.json          which raw folder(s) this results folder belongs to
  conditions.csv                 your condition layout (section 6)
  all_colonies.csv               export: every plate's colonies.csv, stacked
  all_tracks.csv                 export: every plate's tracks.csv, stacked
  summary.csv                    export: colony counts per plate
  <scanner>/
    scanner.json                 detector, model (resolved model folder, null for classical), number of scans,
                                 plates, excluded_plates (density check), config (settings used, without workers)
    plate_<k>/                   k = plate number (empty slots have no folder)
      plate.json                 geometry, scans, times, alignment, settings
      mask.png                   working area (1-bit PNG)
      labels.zarr/               segmentation, uint16 [frames, height, width]
      colonies.zarr/             colony ID per pixel, uint16 [frames, height, width] (footprint tracker)
      tracks.parquet             one row per colony and frame (working file)
      colonies.parquet           one row per colony, with review statuses (working file)
      review.json                review log
      colonies.csv               export
      tracks.csv                 export
      area_matrix.csv            export
  analysis/                      scanlag_analysis.py output (section 6)
```

The `.parquet` and `labels.zarr` working files are what the review window and the analysis read; the CSV
files are written by the export for use in Excel, R, etc. Parquet files open in pandas with
`pd.read_parquet(path)`.

### `plate.json`

`scanner`, `scanner_label`, `plate`, `raw_dir` (absolute path of the scans), `raw_scans` (file names, in
frame order), `times_min` (time of each frame), `start_time` (time zero), `circle` (`number, cx, cy, r,
crop_x0, crop_y0, crop_size` in background-scan pixels), `crop_xy_size` (`[x0, y0, size]`), `shift_xy`
(per-frame `[dx, dy]` alignment), `mask_fraction`, `detector` (e.g. `unet_scanlag_v1` or `classical`),
`threshold` and `contrast_limits` (classical only), `min_area`.

### `tracks.parquet` / `tracks.csv`

| Column | Meaning |
|---|---|
| `scanner`, `plate` | (CSV only) |
| `frame` | Frame index, 0 = background scan |
| `time_min`, `time_h` | Time since time zero (`time_h` in CSV only) |
| `colony_id` | Colony number on this plate (1, 2, ...), numbered in order of appearance |
| `area_px` | Area in pixels: the colony's own pixels, also when it is merged (footprint tracker). **With `--tracker seed`, if `merged` is true, this is the area of the whole merged blob.** |
| `x`, `y` | Centroid of the colony's own pixels (seed tracker: of its blob) (0-based pixel coordinates in the plate crop, section 8) |
| `blob` | (parquet only) label in that frame of `labels.zarr` of the blob holding (most of) the colony |
| `merged` | The colony shares its blob with another colony |
| `included` | (CSV only) the colony's status is 0 |

Only frames in which the colony is detected (area > 0) have a row.

### `colonies.parquet` / `colonies.csv`

| Column | Meaning |
|---|---|
| `scanner`, `plate` | (CSV only) |
| `colony_id` | Colony number |
| `status`, `status_name` | Current status (section 8), including your review decisions |
| `auto_status` | Status given by the pipeline before review |
| `included` | (CSV only) `status == 0` |
| `appearance_frame`, `appearance_min`, `appearance_h` | First frame / time the colony is detected (`appearance_h` CSV only) |
| `growth_20_80px_min` | (CSV only) time from 20 px to 80 px, NaN if the colony merged before reaching 80 px. Fixed at 20/80; the analysis script's doubling time is configurable and also blanks values ≤ 0. |
| `first_merged_frame` (parquet) / `first_merged_min` (CSV) | First merge, −1 / empty if never |
| `final_area_px` | Area in the last frame |
| `seed_x`, `seed_y` | The colony's seed pixel (0-based, plate-crop coordinates) |
| `flags`, `n_flags` | Review flags (text separated by `; `) and their number |

### `area_matrix.csv` (per plate)

Rows = frames; first column `time_min`, then one column `colony_<id>` per **included** colony; area in px,
0 when absent. This is the equivalent of MATLAB's `Area` matrix.

### `summary.csv` (top level)

`scanner, plate, reviewed` (plate opened in the window), `n_colonies, n_included, n_valid, n_near_border,
n_manually_excluded, n_zero_final_area, n_flagged_included`. Note: here `plate` is the folder name
(`plate_1`), while elsewhere it is a number.

### `labels.zarr`

A compressed [frames × height × width] `uint16` array with the detector's output for every frame: 0 =
background, other values = blob labels **of that frame** (they are not colony IDs; use the `blob` column of
`tracks.parquet` to map). Open it with `zarr.open_array(path, mode="r")`.

### `colonies.zarr`

Written by the footprint tracker only: the same shape as `labels.zarr`, with each pixel's **colony ID** (0 =
no colony). A blob of merged colonies is divided between them here.

---

## 8. Concepts: status codes, merges, flags, coordinates, units

### Glossary

| Term | Meaning |
|---|---|
| scanner name | The scanner as written in the scan file names, e.g. `1-4.1u` (`Scanner.label`, `scanner_label` in `plate.json`). |
| scanner folder | The results sub-folder of one time-lapse (`Scanner.name`, `scanner` in `plate.json` and the tables). Usually the scanner name; `<scan folder>_<scanner name>` (e.g. `CTRL1_1-4.1u`) when the same scanner name occurs in two scan folders. The `scanner` column of `conditions.csv` and `compare_with_matlab.py --scanner` take scanner folders; the pipeline's `--scanner` takes either. |
| plate | One Petri dish on the scanner board, numbered 1 to `--plates` in MATLAB board order (section 4). Results are in `<scanner folder>/plate_<k>/`. |
| slot | A position on the board. An empty slot (no dish) is detected and skipped, so its plate number has no folder. |
| blob | One connected object the detector found in one frame (a label in `labels.zarr`). A blob can hold several colonies (a merge) or be debris. |
| colony | A tracked object with its own number (`colony_id`), followed from its first blob of at least 10 px to the last frame. |
| seed | The colony's rounded centroid in its first frame (`seed_x`, `seed_y`). The seed tracker follows a colony by this pixel. |
| footprint | A colony's own pixels in the previous frame. The footprint tracker (default) follows colonies by their footprints and splits merged blobs between them (`colonies.zarr`). |
| auto status | The status the pipeline gave a colony before review (`auto_status`; "detected as" in the review window). `status` is the current one, including your decisions. |

### Colony status codes

Same codes as MATLAB's `IgnoredColonies` (`scanlag.io.matlab_import.ColonyStatus`):

| Code | Name | Set by | Meaning |
|---|---|---|---|
| 0 | `VALID` | pipeline / you | Included in the analysis |
| 1 | `NEAR_BORDER` | pipeline | The colony's seed is within 15 px of the edge of the working area |
| 2 | `MANUALLY_EXCLUDED` | you | Excluded in the review window |
| 3 | `ZERO_FINAL_AREA` | pipeline | Not detected in the last frame |

### Merges

Colonies do not move and only grow, so the footprint tracker (`scanlag/track/tracker.py`, the default)
follows each colony by its pixels of the previous frame. A detected blob belongs to the colonies it
overlaps by at least 10 % of their previous area (or whose seed it covers); a blob overlapping colonies only
less than that goes to the one it overlaps most, so a fragment of a colony never starts a new colony. When
two colonies touch, they form one blob, and each is marked `merged` for that frame. The blob is then split:
each colony keeps the pixels it already had, and every new pixel goes to the nearest colony (watershed on
the distance to their previous pixels, inside the blob). Neighbours grow by about the same step between two
scans, so the split follows the line where their fronts met, and each colony keeps a continuous area track
instead of jumping to the merged blob's area.

Limitation: a colony that already touches a neighbour in the first frame it is detected cannot be separated
and becomes part of the neighbour (whose track then usually gets a `jump` flag).

With `--tracker seed` (MATLAB), a colony is followed by its seed pixel; each colony whose seed is under a
merged blob reports the blob's area and centroid. With the U-Net, touching colonies are often still
separated (each keeps its own core), so merges are less frequent than with the classical detector.

### Review flags

Flags mark colonies the reviewer should look at first (`scanlag/pipeline/flags.py`). They never change a
status by themselves.

| Flag | Rule |
|---|---|
| `gap (n frames)` | Disappears for some frames after appearing (not counting merged frames) |
| `transient merge (n frames)` | Merged with another colony and separate again later |
| `shrinks to x%` | Area drops below 75 % of its largest area so far (only once that area is ≥ 20 px). With the footprint tracker merged frames count too, so a bad split or a colony the detector partly lost shows up here |
| `jump xN` | Area grows more than 2× between consecutive frames (from ≥ 15 px) |
| `stays small (n px)` | Still below 20 px at the end (dust, speck, bubble) |
| `large at appearance (n px)` | Already above 20 px in its first frame. New colonies start at 10-15 px; a bigger one is usually a piece split off a neighbouring colony for one frame, which then "merges" back and follows the neighbour's blob (common on dense plates). Note the neighbour is then marked merged from that frame, which shortens its growth fit |
| `irregular shape (solidity s)` | Solidity < 0.85 (in its last unmerged frame, if ≥ 30 px; the colony's own pixels from `colonies.zarr` with the footprint tracker) |
| `elongated (eccentricity e)` | Eccentricity > 0.85 (scratches, writing, rim reflections) |

Run the pipeline with `--reflag` to recompute flags (e.g. after the rules change) without losing
decisions.

### Coordinates

* All pixel coordinates in the outputs are **0-based**, **x = column, y = row**, origin at the top-left.
* `x`, `y`, `seed_x`, `seed_y` are in the **plate crop**: a square of side `crop_size` around the plate.
* To go back to the raw scan of frame *k*: `x_raw = x + crop_x0 + shift_xy[k][0]`,
  `y_raw = y + crop_y0 + shift_xy[k][1]` (values from `plate.json`).
* The plate circle (`cx`, `cy`, `r`) is in background-scan pixels.
* MATLAB files use 1-based coordinates; the importer and comparison script convert them.

### Units

| Quantity | Unit | Conversion (300 dpi) |
|---|---|---|
| length | px | 1 px = 25 400 / 300 = **84.67 µm** |
| area | px | 1 px = 84.67² ≈ **7 169 µm²** |
| time | minutes (`_min`) or hours (`_h`) | |
| radial growth | px^0.5/h = slope of sqrt(area in px) | × 84.67 / √π → µm/h of the equivalent-disc radius (`radial_growth_um_per_h`) |

---

## 9. Comparing with an old MATLAB analysis (`compare_with_matlab.py`)

**Purpose:** check the new results against a MATLAB ScanLag analysis of the **same scans**: which colonies
both found, and how their appearance time, 20-80 px time and radial growth differ.

```
python scripts/compare_with_matlab.py <results folder> <MATLAB ROI folder> [--scanner NAME]
                                      [--matlab-scanner N] [--max-dist PX]
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `results` (positional) | folder | *required* | Pipeline results folder (must contain `.scanlag_results.json`). |
| `roi` (positional) | folder | *required* | MATLAB `ROI` folder holding `sep_<n>_<plate>/` folders with `sep_<n>_<plate>_data.mat` and the ROI images. |
| `--scanner` | text | the only scanner | Pipeline scanner to compare (required if the results hold several). |
| `--matlab-scanner` | text | `1` | The `<n>` in MATLAB's `sep_<n>_<plate>`. |
| `--max-dist` | number (px) | `8` | Largest distance between seeds for two colonies to match. |

How it works: each MATLAB ROI image is located in the raw scan by template matching (correlation must be
≥ 0.95), so MATLAB positions map onto the pipeline's plate crop. Colonies are paired one to one by seed
position (Hungarian algorithm). MATLAB and the pipeline must have used the same scans, otherwise it stops.
The growth settings are the defaults (fit frames 1-12, 20-80 px) and cannot be changed from the command line.

Output, in `<results>/matlab_comparison/`: `matches.csv` (every colony of either, with its partner and
both statuses), `plates.csv` (counts per plate), `pairs.csv` (colonies kept by MATLAB and included in the new
run, with both sets of metrics), `status_crosstab.csv`, `area_tracks.png`, `paired_metrics.png`. A summary
is printed.

```bat
python scripts\compare_with_matlab.py "results\EPEC260224" "D:\Experiments\EPEC260224\CTRL1\ROI" --scanner 1-4.1u

:: MATLAB folders named sep_2_<plate>, stricter matching
python scripts\compare_with_matlab.py "results\EPEC260224" "D:\Experiments\EPEC260224\CTRL2\ROI" --scanner 1-4.4.1u --matlab-scanner 2 --max-dist 5
```

---

## 10. Running from PyCharm

The project contains two shared run configurations (`.idea/runConfigurations/`):

| Configuration | Script | Prompt |
|---|---|---|
| **ScanLag pipeline** | `scripts/scanlag_pipeline.py` | "Scan folder, then any options (e.g. --detector classical, --review-only)" |
| **ScanLag analysis** | `scripts/scanlag_analysis.py` | "Experiment name in results/ or results folder (e.g. EPEC260224), then any options (e.g. --growth both)" |

1. Open the project folder in PyCharm and select the `scanlag` conda environment (or `.venv`) as interpreter
   (Settings → Project → Python Interpreter → Add Interpreter → Conda Environment → Use existing). The configurations use the project interpreter and the project folder as working directory.
2. Pick the configuration in the run-configuration drop-down and press Run (Shift+F10).
3. A box asks for the parameters. Type **the folder first, then the options**, for example:

   ```
   D:\Experiments\EPEC260224 --detector classical
   EPEC260224 --growth both --fit-frames 2-10
   ```

   For the analysis, the experiment name is enough when the results are in the project's `results`
   folder; otherwise type the full path of the results folder (PyCharm passes the text to Python without a
   shell, so do not rely on `%USERPROFILE%` or `~` there).

PyCharm passes everything typed in that box as one argument. `scanlag.cli.split_prompt_args` splits it
again **before each ` --`**, so folder names with spaces work with or without quotes, and a quoted path
ending in a backslash (`"D:\scans\exp\" --redo`) is handled. Consequences:

* Put the folder first. `--detector classical D:\scans` would be read as the detector `classical D:\scans`
  (and stop with `unknown detector ...`).
* Option values that themselves contain ` --` are not supported.

For the other scripts, create a run configuration (Run → Edit Configurations → + → Python), set the script
path, the parameters and the working directory to the project folder. Relative paths are relative to the
working directory, so `results\EPEC260224` works for the default results folder.

The pipeline may ask y/n questions (`--redo` deleting results, dense plates); type the answer in PyCharm's
Run console and press Enter.

---

## 11. Using the code from Python or a notebook

All functions below are importable after `pip install -e .`. Paths may be `str` or `pathlib.Path`.

### Reading results

```python
from pathlib import Path
import pandas as pd
from scanlag.pipeline.run import load_plate_dirs, plate_info
from scanlag.pipeline.export import plate_tables
from scanlag.store import RESULTS_ROOT

results = RESULTS_ROOT / "EPEC260224"              # the pipeline's default results folder
plates = load_plate_dirs(results)             # [.../1-4.1u/plate_1, ...]
colonies, tracks, matrix = plate_tables(plates[0])
included = colonies[colonies.included]
```

| Function | Signature | Returns |
|---|---|---|
| `scanlag.pipeline.run.load_plate_dirs` | `(results: Path) -> list[Path]` | All `<scanner>/plate_<k>` folders that have a `colonies.parquet`, sorted. |
| `scanlag.pipeline.run.plate_info` | `(plate_dir) -> dict` | Contents of `plate.json`. |
| `scanlag.pipeline.export.plate_tables` | `(plate_dir: Path, merged_growth="exclude") -> (colonies, tracks, area_matrix)` | The three per-plate DataFrames exactly as exported to CSV (section 7), using the current review statuses. `merged_growth` as in `scanlag_analysis.py --merged-growth`. |
| `scanlag.pipeline.export.export` | `(results: Path, store: ResultsStore, log=print) -> None` | Writes all CSV exports (what "Finish & export" does). |
| `scanlag.pipeline.run.refresh_flags` | `(plate_dir: Path) -> int` | Recomputes flags in place; returns the number of flagged colonies. |
| `scanlag.pipeline.run.retrack_plate` | `(store: ResultsStore, plate_dir: Path, tracker="footprint") -> dict` | What `--retrack` does for one plate; backs up the old `colonies.parquet` / `review.json`. Returns the old tracker and the old and new colony counts. |

### Condition analysis

```python
from scanlag.analysis.conditions import FitConfig, read_layout, collect, summary, area_matrix
from scanlag.analysis.plots import histograms, write_html

layout = read_layout(results / "conditions.csv")
data = collect(results, layout, FitConfig(fit_first=1, fit_last=12, lb=20, ub=80))
print(summary(data.colonies, data.conditions))
fig = histograms(data.colonies, data.conditions, "appearance_min", "Appearance time",
                 "appearance time (min)", "min", width=15)
fig.show()
```

| Name | Signature | Notes |
|---|---|---|
| `FitConfig` | dataclass `(fit_first=1, fit_last=12, min_points=4, lb=20, ub=80, merged_growth="exclude")` | Radial-fit window (frames from appearance), minimum points, doubling-time range (px), and `--merged-growth`. |
| `parse_plates` | `(text) -> list[int]` | `"1,2,4-6"` → `[1, 2, 4, 5, 6]`; raises `ValueError` on bad input. |
| `read_layout` | `(path: Path) -> DataFrame` | One row per (condition, scanner, plate); raises `ValueError` on missing columns or a plate in two conditions. |
| `expand_layout` | `(table: DataFrame) -> DataFrame` | The same from a table with columns `condition, scanner, plates` (what `--condition` builds); raises `ValueError` for a plate in two conditions. |
| `available_plates` | `(results: Path) -> dict[str, list[int]]` | scanner → processed plate numbers. |
| `layout_template` | `(results: Path) -> DataFrame` | Template layout with empty conditions. |
| `collect` | `(results: Path, layout: DataFrame, cfg: FitConfig = FitConfig()) -> ConditionData` | `ConditionData` has `.colonies`, `.tracks` (included colonies only), `.conditions` (in layout order), `.missing` (plates without results). Raises `FileNotFoundError` if no plate is found. |
| `radial_growth` | `(tracks: DataFrame, cfg: FitConfig, use_merged=False) -> DataFrame` | Per colony (`scanner, plate, colony_id`): `slope` (px^0.5/h), `r2`, `n`. Needs columns `scanner, plate, colony_id, frame, time_h, area_px, merged`. |
| `doubling_time` | `(tracks: DataFrame, lb: float, ub: float, use_merged=False) -> Series` | colony_id → **hours** from lb to ub px, for one plate. |
| `area_matrix` | `(tracks: DataFrame) -> DataFrame` | Wide colony × frame table. |
| `summary` | `(colonies: DataFrame, conditions: list[str]) -> DataFrame` | Per-condition n / mean / median / std. |
| `scanlag.analysis.plots.histograms` | `(colonies, conditions, metric, title, xlabel, unit, width=None) -> plotly Figure` | `width=None` picks bins automatically. |
| `scanlag.analysis.plots.write_html` | `(figures: list, path: Path, title: str) -> None` | One self-contained HTML page. |

### Running parts of the pipeline

```python
from scanlag.pipeline.scans import find_scanners
from scanlag.pipeline.run import RunConfig, process_scanner
from scanlag.store import ResultsStore

raw = Path(r"D:\Experiments\EPEC260224")
store = ResultsStore.for_raw_dir(raw)                    # <project folder>/results/EPEC260224
for scanner in find_scanners(raw):
    process_scanner(scanner, store, RunConfig(detector="classical"))
```

| Name | Signature | Notes |
|---|---|---|
| `scanlag.pipeline.scans.find_scanners` | `(root) -> list[Scanner]` | `Scanner` has `name` (results sub-folder), `label` (scanner name from the file names), `directory, scans, times, start, prefix, time_zero` (where `start` came from: `"first scan"` or the robot log's file name) and the property `times_min`. Raises `FileNotFoundError` if no scan matches. |
| `scanlag.pipeline.scans.select_scanners` | `(scanners, prefixes=(), names=()) -> list[Scanner]` | The `--prefix` / `--scanner` filter. Raises `ValueError` (listing the scanners found) if nothing matches. |
| `scanlag.pipeline.run.RunConfig` | dataclass `(detector="unet", n_plates=6, mask_fraction=0.8, threshold=0.2, min_upper_limit=0.06, min_area=10, tracker="footprint", workers=6)` | `min_upper_limit` (classical contrast floor) and `min_area` (px to start a colony) are not exposed on the command line. |
| `scanlag.pipeline.run.model_dir` | `(detector: str) -> Path \| None` | The model folder of `unet`, `cellpose` or a folder path; `None` for `classical`. Raises `ValueError` if the folder has no `config.json`. |
| `scanlag.pipeline.run.make_segmenter` | `(cfg, log=print)` | The detector chosen by `cfg.detector`, to reuse across scanners. |
| `scanlag.pipeline.run.last_frame_counts` | `(scanner, cfg, seg=None, log=print) -> dict[int, int]` | Plate number → colonies (blobs ≥ `min_area` px) on the last frame: the `--max-colonies` density check. |
| `scanlag.pipeline.run.process_scanner` | `(scanner, store, cfg, log=print, seg=None, exclude=()) -> list[Path]` | Segments and tracks all plates of one scanner, except the plate numbers in `exclude`; returns the plate folders. `seg` reuses a segmenter from `make_segmenter`. Does **not** write `scanner.json` (the script does). |
| `scanlag.store.ResultsStore` | `(results_dir, raw_dirs: list)`; `ResultsStore.for_raw_dir(raw_dir, results_dir=None)` | The only writer; refuses paths inside raw data. Methods `path(*parts)`, `write_json`, `write_table` (`.csv` or parquet), `write_mask`, `write_image`, `create_labels`. |
| `scanlag.store.default_results_dir` | `(raw_dir, root=RESULTS_ROOT) -> Path` | `<root>/<raw folder name>`; `RESULTS_ROOT` is `<project folder>/results`. |
| `scanlag.gui.review.run_review` | `(plate_dirs: list[Path], on_finish=None) -> None` | Opens the review window (blocks until closed). Importing this module switches matplotlib to the Qt backend. |

### Images, plates, detectors, tracking

| Name | Signature | Notes |
|---|---|---|
| `scanlag.io.raw.read_scan` | `(path) -> ndarray` | RGB scan, read-only (memory-mapped when uncompressed). |
| `scanlag.io.raw.parse_timestamp` | `(path) -> datetime \| None` | Time from `_YYYYMMDD_HHMM[SS]` in the name. |
| `scanlag.io.raw.ScanSeries` | `ScanSeries.from_pattern(directory, pattern)`; `.read(i)`, `.read_window(i, rows, cols)`, `len()` | Time-sorted series (falls back to file time if the name has no stamp). |
| `scanlag.pipeline.plates.find_plates` | `(background, n_plates=6, dpi=300.0, diameter_mm=90.0) -> list[Plate]` | `Plate(number, cx, cy, r)` with `.crop` = `(x0, y0, size)`. Raises `RuntimeError` if too few circles are found. |
| `scanlag.pipeline.plates.circle_mask` | `(size, plate, fraction=0.8) -> bool array` | Working-area mask in crop coordinates. |
| `scanlag.pipeline.plates.crop` | `(image, x0, y0, size) -> ndarray` | Square crop, zero-padded outside the image. |
| `scanlag.pipeline.plates.Aligner` | `(background, plate, margin=40, refine=1, mask_fraction=0.8)`; `.shift(image) -> (dx, dy)` | Frame alignment. |
| `scanlag.pipeline.plates.is_empty_slot` | `(background, plate) -> bool` | Empty holder slot test. |
| `scanlag.detect.unet.load_detector` | `(model_dir, **kwargs)` | U-Net or fine-tuned Cellpose detector from a model folder. Both have `.prepare(image, background, previous)` and `.segment(list_of_prepared) -> list[label images]`. |
| `scanlag.detect.unet.UNetDetector` | `(model_dir, fg=0.4, core=0.7, min_size=5)` | `fg`: foreground probability threshold; `core`: higher splits touching colonies less (≥ 0.85 almost never splits); `min_size` in px. `.predict(x)` returns the foreground and distance maps. |
| `scanlag.detect.cellpose_backend.CellposeDetector` | `(input="rgb", diameter=None, min_size=15, flow_threshold=0.4, cellprob_threshold=0.0, pretrained_model="cpsam_v2", norm_limits=None, label=None)`; `CellposeDetector.finetuned(model_dir, **kwargs)` | `input` is `rgb`, `diff` or `diff_rgb` for zero-shot use. |
| `scanlag.detect.classical` | `subtract_background(image, background)`, `stretch_limits(gray, low=0.01, high=0.99)`, `binarize(image, background, limits, threshold, mask)`, `label_column_major(binary) -> (labels, n)`, `rgb2gray_uint8(rgb)` | Bit-exact ports of MATLAB `cleanImage`, `stretchlim`, `im2L`, `bwlabel`, `rgb2gray`. |
| `scanlag.track.tracker.FootprintTracker` | `(min_area=10, min_overlap=0.1)`; `own = .update(labels, n)` returns the frame's colony ID image; `.tracks.as_arrays()` as below | The default tracker: colonies keep their own pixels, merged blobs are split between them. |
| `scanlag.track.seed_tracker.SeedTracker` | `(min_area=10)`; `.update(labels, n)`; `.tracks.as_arrays() -> (area[T,N], centroid[T,N,2], blob[T,N])` | MATLAB's seed tracker. Labels must be numbered column-major (`label_column_major`). Centroids are 1-based (MATLAB style). |
| `scanlag.track.seed_tracker.border_status` | `(mask, seeds, radius=15) -> bool array` | True if the colony is not near the border. |
| `scanlag.io.matlab_import.load_plate` | `(plate_dir) -> MatlabPlate` | Reads a MATLAB `<plate_dir>/<name>_data.mat` (area, centroid, status, threshold, limits, mask, motions, `times_min`). |
| `scanlag.io.matlab_import.find_analyzed_plates` | `(root) -> list[Path]` | MATLAB plate folders with tracking results. |
| `scanlag.io.matlab_import.ColonyStatus` | `IntEnum` | `VALID=0, NEAR_BORDER=1, MANUALLY_EXCLUDED=2, ZERO_FINAL_AREA=3`. |
| `scanlag.cli.split_prompt_args` | `(argv: list[str]) -> list[str]` | PyCharm prompt splitting (section 10). |

---

## 12. Models: where they live, training and evaluation

Most users only need the shipped U-Net (`models/unet_scanlag_v1`). This section is for retraining or
benchmarking. All of it assumes access to old MATLAB-analysed experiments (the developers use
`D:\Alon\Documents\Experiments\microcolony-seq`), which are used as training labels.

### The models

* **U-Net v1** (`models/unet_scanlag_v1`, default): U-Net with a ResNet-34 encoder
  (`segmentation_models_pytorch`), two outputs (foreground and distance to the colony edge); touching
  colonies are separated by a watershed from colony cores. Input upscaled 2×. Trained 10 000 steps.
* **Cellpose-SAM v1** (`models/cpsam_scanlag_v1`): Cellpose-SAM fine-tuned on the same tiles; about 40×
  slower than the U-Net.
* **U-Net v2** (`models/unet_scanlag_v2`, git-ignored): v1 minus plates with a low MATLAB contrast limit
  (`--min-upper-limit 0.08`). The developers found no clear improvement; v1 stays the default.

Each model folder holds `model.pt`, `config.json` (training settings, input channels and the fixed
normalisation limits used at inference) and `history.json` (loss curve). See `docs/PLAN.md` §10-14 for the
scores.

### Training workflow

```
build_testset.py  →  select_tiles.py  →  annotate_tiles.py  →  train_unet.py / train_cellpose.py
 (MATLAB replay)      (pick 256-px tiles)   (correct them by hand)   (builds train/tiles.npz once, trains)
                                                                        ↓
                                     evaluate_goldset.py / evaluate_plates.py  →  scanlag_pipeline.py --detector models\<name>
```

The training scripts need `--name` (the new folder under `models/`, e.g. `--name unet_scanlag_v3`), and refuse
an existing `models/<name>` unless `--force` is given, so the shipped models are not overwritten by accident.
Every script in this section prints its options with `--help`.

### `build_testset.py` — MATLAB reference test set

Replays MATLAB detection and tracking (bit-exact) for every analysed MATLAB plate under `data_root`, from
the ROI images, and writes per-frame labels (kept colonies; everything else MATLAB segmented is marked
IGNORE = 65535). Only reads the raw data.

```
python scripts/build_testset.py <data_root> [--out DIR] [--workers N] [--retry-failed]
```

| Argument | Default | Meaning |
|---|---|---|
| `data_root` | *required* | Root folder searched for MATLAB `*_data.mat` plate folders. |
| `--out` | `<project>/testset_matlab` | Output folder (`index.csv`, `plates/<exp>__<cond>__<plate>/`). |
| `--workers` | `6` | Parallel processes. |
| `--retry-failed` | off | Only rebuild plates that failed last time (per `index.csv`). |

```bat
python scripts\build_testset.py "D:\Alon\Documents\Experiments\microcolony-seq" --workers 4
```

### `select_tiles.py` — choose tiles for hand correction

Picks 256 × 256 px tiles in three categories (`early`, `merged`, `excluded`) and writes
`<testset>/manual/tile_XX_<category>/` (`image.png`, `background.png`, `previous.png`, `labels_init.png`,
`tile.json`). Refuses to run if `manual/` already exists.

| Argument | Default | Meaning |
|---|---|---|
| `--testset` | `<project>/testset_matlab` | Test-set folder. |
| `--per-category` | `8` | Tiles per category. |

### `annotate_tiles.py` — correct the tiles in napari

```
python scripts/annotate_tiles.py [--testset DIR]
```

Opens napari with the layers `colonies` (editable labels), `image`, `difference`, and hidden `previous`
and `background`. Goal: every colony is one label and nothing else is labelled (split merged colonies with a
fresh label, press M for a new ID; delete debris by filling with label 0; add missed colonies). Keys:
**S** save (`labels_corrected.png`), **N** / Page Down next tile, **Shift+N** / Page Up previous tile.
(B belongs to napari's labels layer.)

### `train_unet.py`

```
python scripts/train_unet.py --name NAME [--force] [--testset DIR] [--steps N] [--scale S] [--min-upper-limit L]
```

| Argument | Default | Meaning |
|---|---|---|
| `--testset` | `<project>/testset_matlab` | Test set with gold tiles; `train/tiles.npz` is built there if missing. |
| `--name` | *required* | Output folder `models/<name>/`. |
| `--force` | off | Overwrite `models/<name>/` if it exists. |
| `--steps` | `6000` | Optimizer steps (batch 16). v1 was trained with 10 000 (about 12 min on an RTX 4080). |
| `--scale` | `2.0` | Input upscaling factor. |
| `--min-upper-limit` | `0.0` | Drop MATLAB tiles from plates whose contrast upper limit is below this. |

Other settings (learning rate 1e-3, AdamW, cosine schedule, gold tiles sampled 5× more often, loss
BCE + Dice + 5 × distance MSE) are in `scanlag/train/unet_train.py` (`UNetConfig`). The first run downloads
the ImageNet ResNet-34 encoder weights (internet needed).

```bat
python scripts\train_unet.py --name unet_scanlag_v3 --steps 10000
```

### `train_cellpose.py`

```
python scripts/train_cellpose.py --name NAME [--force] [--testset DIR] [--steps N] [--batch-size B] [--accumulate A] [--rebuild-tiles]
```

| Argument | Default | Meaning |
|---|---|---|
| `--testset` | `<project>/testset_matlab` | Test set. |
| `--name` | *required* | Output folder `models/<name>/`. |
| `--force` | off | Overwrite `models/<name>/` if it exists. |
| `--steps` | `8000` | Optimizer steps (about 65 min on an RTX 4080). |
| `--batch-size` | `2` | Tiles per forward pass. |
| `--accumulate` | `2` | Forward passes per optimizer step (effective batch = 4). |
| `--rebuild-tiles` | off | Rebuild `train/tiles.npz` even if it exists. |

Needs `environment-training.yml` (pip route: the `cellpose` extra); the base Cellpose-SAM weights are fetched by Cellpose on first use.

### Evaluation scripts

| Script | Usage | Arguments (default) | Output |
|---|---|---|---|
| `evaluate_goldset.py` | Score detectors on the hand-corrected tiles (F1, precision, recall, small-colony F1, by IoU). | `--testset` (`testset_matlab`), `--detectors` (`matlab_all matlab_kept cellpose`; also `model:<dir>`), `--iou` (`0.5`) | `<testset>/eval/summary.csv` (updated in place), `tiles_<detector>.csv`, `pred/<detector>/*.png` |
| `evaluate_plates.py` | Score models on whole plates against the colonies kept in MATLAB, at the median-appearance frame and the last frame. | `--testset` (**required**), `--models` (`models/cpsam_scanlag_v1 models/unet_scanlag_v1`), `--unet-cores` (`0.7`) | `<testset>/eval/plates.csv`, `summary.csv`, `pred/...` |
| `track_with_model.py` | Run a model on every frame of test-set plates and follow MATLAB's colonies by their seeds. | `--plates` (**required**, plate-ID prefix such as `EPEC260224`), `--testset`, `--model` (`models/unet_scanlag_v1`) | `<testset>/model_runs/<model>/plates/<plate>/labels.zarr`, `tracks.parquet` |
| `plot_area_tracks.py` | Plot MATLAB vs model area tracks (needs `track_with_model.py` first). | `--plates` (**required**), `--testset`, `--model` (`unet_scanlag_v1`, a folder *name*) | `<testset>/model_runs/<model>/figures/*.png`, `timing_<prefix>.csv` |

```bat
python scripts\evaluate_goldset.py --detectors model:models\unet_scanlag_v3
python scripts\evaluate_plates.py --testset testset_vapc --models models\unet_scanlag_v1 models\unet_scanlag_v3 --unet-cores 0.5 0.7 0.85
```

Caveats from the developers (`docs/PLAN.md`): all MATLAB plates and gold tiles are used for training, so
gold-set scores show training fit, not performance on new data; test on plates from an experiment that was
not used for training.

---

## 13. Troubleshooting and FAQ

**`ModuleNotFoundError: No module named 'scanlag'`**
Activate the environment first (`conda activate scanlag`). If it is active, install the package in it:
`pip install -e .` from the project folder. In PyCharm, select the `scanlag` interpreter.

**`No module named 'segmentation_models_pytorch'` or `'fastremap'`**
The default U-Net detector needs both. They are in `environment.yml` (pip route: the `dl` extra,
`pip install -e ".[dl]"`). Or use `--detector classical`.

**`No module named 'cellpose'`**
Cellpose is not in the user environment: `conda env update -n scanlag -f environment-training.yml`.

**The U-Net runs on the CPU although the machine has an NVIDIA GPU**
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `False`: the
environment was created from `environment-cpu.yml`, or the driver was updated or broken since. Check
`nvidia-smi`, then `conda env remove -n scanlag` and create it again from `environment.yml`. Also check
that `SCANLAG_DEVICE` is not set to `cpu`.

**An error mentioning `mps` on a Mac**
Some operation is missing from PyTorch's Apple-GPU backend. Run on the CPU instead with
`SCANLAG_DEVICE=cpu` (see [which processor runs the networks](#install-with-conda-recommended)) and
please report the error.

**`nothing provides __cuda` when creating the environment**
`environment.yml` needs an NVIDIA GPU with its driver installed. Use `environment-cpu.yml`.

**`unknown detector ...: use unet, cellpose, classical or a trained model folder (one with a config.json)`**
`--detector` was neither an alias nor a folder holding a trained model. Check the spelling or the path; for
`cellpose`, the folder `models/cpsam_scanlag_v1` must have been copied in (it is not in git).

**`... no scans named [<prefix>]_<scanner>_<YYYYMMDD>_<HHMM>.tif under ...`**
Check the file names (section 3) and that you passed the experiment folder, not a single file.

**`found N plate-sized circles, expected 6. If the board holds fewer plates, use --plates; ...`**
Fewer dish-like circles were found on the background scan. If the board holds fewer plates, use
`--plates N`. The detector expects 90 mm plates scanned at 300 dpi (neither can be changed from the command
line).

**`... already holds results for [...]; choose another results directory for [...]`**
Each results folder remembers its raw folder (`.scanlag_results.json`). Two experiments whose folders have
the same name (e.g. two `AMP7`) would share `results/AMP7`; give one of them `--out`.

**`warning: --redo will delete the results of ..., including review decisions on N plate(s). proceed? y/n`**
`--redo` is about to delete complete results of those scanners. `y` deletes them and segments again; `n` (or
no terminal: `(no terminal: answering n)`) stops with `stopped`. No backup is made, so copy
`colonies.parquet` and `review.json` first if you want to keep the decisions. `--overwrite` skips the question
(for batch jobs).

**`warning: ... plates ... are too dense (n>500). ... exclude these plates? y/n`**
The density check found more blobs on the last frame than `--max-colonies` (merged colonies count once, so
there are at least that many colonies). `y` leaves those plates out (recorded in `scanner.json` as
`excluded_plates`); `n`, or no terminal, processes them anyway. Raise the limit or use `--max-colonies 0` to
skip the check.

**`--review-only with --no-review does nothing but export: use --export-only`** / **`not allowed with argument ...`**
`--redo`, `--review-only`, `--export-only`, `--reflag` and `--retrack` cannot be combined; give one of them.

**`no scanner matches the filter; found: ...`**
`--prefix` / `--scanner` matched nothing. The message lists every scanner with its prefix; copy a name from it.

**`RawDataProtectionError: ... must be separate from raw data`**
`--out` points inside the raw-data folder (or the reverse). Raw data is never written to; choose a
results folder elsewhere.

**Is my raw data safe?**
Yes, by design: scans are opened read-only, and all writes go through `ResultsStore`, which refuses any
path inside a raw-data folder. Results go to `<project folder>/results/<folder>` by default. You can keep raw
data on a read-only drive.

**I re-ran with `--redo` and my review decisions are gone.**
`--redo` deletes the scanner's results folder, including `colonies.parquet` and `review.json` (it asks first,
unless `--overwrite`). Copy those files first if you want to keep a record. To recompute only the flags, use `--reflag` instead; to track
again (e.g. switch an experiment processed before the footprint tracker to it), use `--retrack`, which keeps
the segmentation and backs up the old decisions as `*.bak` files.

**How do I switch an older experiment to the footprint tracker?**
`python scripts\scanlag_pipeline.py "D:\Experiments\EPEC260216" --retrack`. It reuses the stored
segmentation, so it takes seconds per plate, then opens the review window (the review starts afresh because
colony numbers change).

**The review window fails to show images / `FileNotFoundError` for a `.tiff`.**
The window reads images from the raw folder recorded in `plate.json` (`raw_dir`, an absolute path). If the
data moved or the drive letter changed, make it available at the old path again (there is currently no
option to point to a new location).

**`--review-only` asks for the scan folder.**
Yes: the scan folder is always the first argument; the results folder is derived from it (or from
`--out`). Use the same `--out` as in the first run.

**`scanlag_analysis.py` stopped and said it wrote a template.**
That is expected on the first run: open `<results>/conditions.csv`, fill in the `condition` column and run
again (section 6). Use `--condition` to skip the file.

**`plates assigned to more than one condition: ...`**
A plate is listed under two conditions in `conditions.csv` or in the `--condition` options. Each plate may
belong to one condition only.

**`conditions that would share an output folder, rename one of each: ...`**
Two condition names become the same folder name (e.g. `A B` and `A/B` both become `A_B`). Rename one.

**`... is not a folder` from `scanlag_analysis.py`**
Neither the path nor `<project folder>/results/<name>` exists. Pass the full path of the results folder (e.g. one written
with `--out`).

**`warning: no results for 1-4.1u plate 3 (empty slot or not processed), skipped`**
That plate has no results folder (the slot was empty, or the scanner was not processed). Harmless if
expected.

**PyCharm prompt: my folder path with spaces or a trailing backslash is split wrongly.**
Put the folder first and the options after it (section 10). Quotes are optional.

**A colony's growth rate or doubling time is empty.**
Radial growth needs ≥ 4 unmerged frames inside `--fit-frames`; the doubling time is empty if the colony
merged before reaching UB, never reached UB, or crossed LB and UB between two scans.

**Appearance times differ from MATLAB by 15-60 min on some plates.**
MATLAB's threshold depends on the contrast of the plate's last frame, so its sensitivity varies between
plates; the U-Net uses fixed normalisation. See `docs/PLAN.md` §12.1. `--detector classical` reproduces
MATLAB's detection (with a floor on very low contrast limits).

**It is slow.**
Check where the networks run (`python -c "from scanlag.device import torch_device; print(torch_device())"`
prints `cuda`, `mps` or `cpu`). Without a GPU, use `--detector classical` or expect much longer runs.
Cellpose is about 40× slower than the U-Net.

---

## 14. Running the tests

```bat
conda activate scanlag
python -m pytest -q
```

The tests cover file-name parsing and the scanner filters, raw-data protection, the classical detector
(bit-exactness with MATLAB), tracking, flags, plate geometry, re-tracking a stored plate, the
speed-ups (same results as the slower reference code), the command-line scripts (`tests/test_scripts.py`:
y/n questions, conflicting flags, `--redo` bookkeeping, dense-plate exclusion, analysis checks, the export
note), the U-Net and Cellpose training utilities, metrics, the condition analysis and the consistency of the
environment files. They need `environment-training.yml` on top of `environment.yml` (pip route: the `dl`,
`cellpose` and `dev` extras, `pip install -e ".[gui,dl,cellpose,dev]"`). One test
(`test_parity_with_matlab_on_dense_plate`) uses a legacy MATLAB plate and is skipped when that data is not
present. It looks under the folder in the environment variable `SCANLAG_LEGACY_DATA` (default: the developers'
`D:\Alon\Documents\Experiments\microcolony-seq`). All tests should pass (about 2 min).

---

## 15. Project structure

```
scanlag_analyser/
  pyproject.toml              package metadata and dependencies
  environment.yml             conda environment for users with an NVIDIA GPU
  environment-cpu.yml         the same with CPU-only PyTorch (no NVIDIA GPU, macOS)
  environment-training.yml    add-on for training, Cellpose and the tests
  README.md                   this file
  docs/PLAN.md                developer design notes and benchmark results (sections 1-15)
  models/                     trained detectors (unet_scanlag_v1 in git; others git-ignored)
  scripts/                    command-line tools
    scanlag_pipeline.py         raw scans → results → review window → export
    scanlag_analysis.py         conditions → tables + histograms
    compare_with_matlab.py      new results vs a MATLAB analysis
    build_testset.py            MATLAB replay → reference test set
    select_tiles.py             choose tiles for hand correction
    annotate_tiles.py           correct tiles in napari
    train_unet.py               train the U-Net
    train_cellpose.py           fine-tune Cellpose-SAM
    evaluate_goldset.py         score detectors on the gold tiles
    evaluate_plates.py          score detectors on whole plates
    track_with_model.py         model tracks for MATLAB colonies
    plot_area_tracks.py         MATLAB vs model area-track figures
  scanlag/                    the Python package
    cli.py                      PyCharm argument splitting
    device.py                   where the networks run: CUDA, Apple GPU or CPU (SCANLAG_DEVICE)
    store.py                    ResultsStore: the only writer, raw-data protection
    io/raw.py                   read-only scan access, timestamps
    io/matlab_import.py         MATLAB *_data.mat reader, ColonyStatus codes
    pipeline/scans.py           find scans, group by scanner, time zero
    pipeline/plates.py          plate detection, alignment, masks
    pipeline/run.py             segmentation + tracking per scanner, per-plate outputs
    pipeline/flags.py           review flags
    pipeline/export.py          CSV export
    detect/channels.py          model input settings shared with training (PREVIOUS_GAP)
    detect/classical.py         MATLAB threshold detector (bit-exact port)
    detect/unet.py              U-Net detector, model loading
    detect/cellpose_backend.py  Cellpose-SAM detector
    track/tracker.py            footprint tracker (default): splits merged colonies
    track/seed_tracker.py       MATLAB seed-pixel tracker, border test
    track/seed_areas.py         follow known seeds on another detector's labels
    gui/review.py               review window
    analysis/conditions.py      condition layout and growth metrics
    analysis/plots.py           plotly histograms
    train/                      training tiles, U-Net and Cellpose training loops
    testset/                    MATLAB replay, tile selection, gold set
    eval/                       detection metrics (IoU and seed based)
  tests/                      pytest suite
  .idea/runConfigurations/    PyCharm run configurations
```

Licence: no licence file is included in the repository; ask the authors before redistributing.

# ScanLag Analyser (Python) — Design Plan

Python successor to the MATLAB ScanLag code in `C:\Users\owner\Documents\Code\scan_lag_raya`.
Status: written as the plan on 2026-10-21, before any code. It has been implemented since; the README
describes the current behaviour, and this file is kept for the design reasons.

---

## 1. What the MATLAB code does

Scripts in the root are copy-paste templates of two kinds. They differ only in hard-coded paths and in which
lines are commented out:

* `*_prep.m` / `fullAnalysis_*.m`: **plate preparation and processing**
* `epec_*.m` / `Data_*.m`: **comparing conditions** (histograms and plots)

### 1.1 Pipeline

| Step | MATLAB entry point | What it does | Output |
|---|---|---|---|
| 1. Board geometry | `CreateBoardHint` (run once per scanner) | Hough circles (`imfindcircles`) on one scan. Saves plate centers and radii relative to image size, plus an alignment rectangle | `BoardHint01.mat` |
| 2. Align + crop | `CropROI` → `FindPlates`, `fullPyrMotion_trans`, `saveROI` | Sorts scans by **file mtime**. Refines each plate circle with Hough. Registers consecutive scans (translation-only pyramidal Lucas-Kanade inside the alignment rectangle), shifts each scan, and **writes a TIFF crop per plate per scan** | `ROI/sep_S_P/*.tif`, `<scan>_motions.mat` **written into the raw folder**, `sep_S_P_data.mat` |
| 3. Mask | `setMaskApp` (GUI) | Edit a circular or polygon working-area mask | `CircleMask` or `FullMask` in data.mat |
| 4. Detect + track | `ProcessPlates` → `Images2Colonies` | Per frame: `gray(frame − frame0)`, contrast stretch (limits from the last frame), threshold `TH=0.2`, 3×3 median filter, mask, `bwlabel`. **Tracking**: each colony is identified by the pixel at its first centroid. A blob in a later frame belongs to that colony if it covers that pixel. Unmatched blobs with ≥10 px become new colonies. Colonies whose seed is within 15 px of the mask border get flagged | `Area`, `Centroid`, `BBox` [frames × colonies], `IgnoredColonies` |
| 5. Start time | `SetStartingTime` | Parses the first timestamp in the scanner `.log` | `StartingTime` |
| 6. Review | `PlateAnalyzer` (GUI) | Time slider, overlay, area curves, manual include/exclude (Ctrl+E / Ctrl+I) | `IgnoredColonies` = 2 |
| 7. Condition analysis | `getAppearanceTime`, `getAppearanceGrowth`, `getColoniesGrowthRate`, `getMergedColonies`, `getSurvivalCurve`, `getDistrStatistics` | Appearance time is the first frame with area > 0. Growth time is t(area ≥ ub) − t(area ≥ lb) (lb=20, ub=80 px). Colonies that merge before reaching ub are dropped. A merge is detected when two colonies share an identical centroid | Results struct and plots |

`IgnoredColonies` codes: 0 = valid, 1 = near border, 2 = manually excluded, 3 = zero area in the last frame.

### 1.2 Dead or out-of-scope code (will not be ported)

* `StitchImages`, `CalcMotionsAndStitch`, `calcMotions`, `warp_images`, `pyrMotion_trans`, `motion_trans`: panorama-stitching leftovers. They call functions that don't exist (`findMinCut`, `stabilizeRGB_*`, `pyrMotion_rot`).
* `GeneralUtilities/Instruments/*` (Tecan, Wallac, thermometer, FCS readers) and `ConvertToV16`: not part of ScanLag.
* `SetMask.m` (empty), `Plates2Colonies.m` (calls a nonexistent `images2Colonies`), 3 duplicate copies of `GetDataName.m`.
* `Misc/hist2d`, `MPNindex`: can be added later if needed.

### 1.3 Problems to fix

1. **Raw folder is modified**: `*_motions.mat` files are written next to the raw scans.
2. **Disk bloat**: the ROI crops take 2.2 GB of a 6.6 GB experiment (EPEC260323/AMP7). They are recomputable from the raw data and a few numbers.
3. **Time comes from file mtime** (`dir().datenum`). Copying or moving files breaks it. Scan filenames already carry `_YYYYMMDD_HHMM`.
4. **Merged colonies are lost.** Once blobs touch, every colony in the blob reports the merged area, and the colony is excluded from growth statistics. Dense plates (e.g. EPEC260216/AMP24 sep_2_2, 1166 colonies) are mostly touching colonies at the end.
5. **Identity depends on a single seed pixel.** The tracker can lose a colony if its blob no longer covers the first-centroid pixel.
6. Contrast limits come from the *last* frame, so detection of early frames depends on how the plate ends up.
7. Minor: `CropROI` saves `PlateCirc` from `PlatePos{i}` (destination index) and not `PlatePos{Plates2Cut(i)}` (plate index), so the radius can come from a different plate.
8. Configuration is done by commenting lines in and out. State lives in `global`s and in `.mat` files mutated with `-append`.

---

## 2. Design principles

1. **Raw data is never written to.** See §4.
2. **No derived images on disk.** Store geometry, transforms, masks and labels. Compute pixel crops lazily from the raw scans. They are uncompressed TIFFs, so `tifffile` memory-mapped windowed reads are cheap.
3. **One config file per experiment** (YAML, validated with pydantic) instead of editing scripts.
4. **Swappable detector.** `Detector` protocol with `classical` (MATLAB parity), `unet` and `cellpose` backends. Tracking and analysis don't depend on which one is used.
5. **Tidy outputs.** Per-colony and per-frame tables in Parquet/CSV that pandas can read directly. No opaque struct blobs.
6. **Testable.** Pure functions, a regression suite against the existing 130 MATLAB-analyzed plates, and a test that the raw data is unchanged.

---

## 3. Architecture

```
scanlag/
  config.py            # pydantic models: Experiment, Scanner, Plate, DetectionParams, AnalysisParams
  io/
    raw.py             # ScanSeries: read-only, sorted by filename timestamp, memmap window reads
    scanner_log.py     # start time, interval, scanner names from *.log
    matlab_import.py   # read *_data.mat, BoardHint.mat, *_motions.mat (migration + regression)
  store.py             # ProjectStore: the only place that writes; guards raw paths; manifest
  prepare/
    plates.py          # board-hint generation + per-plate Hough refinement (skimage/OpenCV)
    registration.py    # per-frame translation (phase correlation, subpixel) -> offsets table
    masks.py           # circle/polygon masks, border-distance map
  detect/
    base.py            # Detector protocol: frames -> label images
    classical.py       # bit-for-bit port of im2L (bg subtraction, stretch, TH, medfilt)
    unet.py            # PyTorch model + tiled inference + watershed post-processing
    cellpose_backend.py
  track/
    tracker.py         # overlap-based tracking with merge splitting (see §5.3)
  analysis/
    metrics.py         # appearance, lb->ub growth time, survival, distribution stats
    conditions.py      # group plates into conditions, build tidy result tables
    plots.py           # matplotlib versions of the comparison figures in epec_*.m
    export.py          # per-colony CSV for microcolony-seq picking (colony_info_analyzed.csv)
  train/
    pseudolabels.py    # build training targets from MATLAB results (§5.2)
    dataset.py, train.py, evaluate.py
  gui/                 # napari widgets: mask editor (replaces setMaskApp), plate review (replaces PlateAnalyzer)
  cli.py               # `scanlag prepare | detect | track | review | analyze | train`
configs/example_experiment.yaml
tests/
```

**Environment:** a new conda env `scanlag-py` on Python 3.11 with torch (CUDA, RTX 4080 present), numpy, scipy, scikit-image, opencv, tifffile, zarr, pandas, pyarrow, matplotlib, pydantic, typer and napari[pyqt]. cellpose is optional. The existing `scanlag` env runs Python 3.14 without torch and won't be used, and the base env has a broken `imagecodecs`.

---

## 4. Storage and raw-data safety

### 4.1 Output layout

The results directory is a parameter. Its default is `<project root>/results/<data folder name>` (§7).

```
<project root>/results/<experiment>/
  experiment.yaml          # frozen copy of the config used
  manifest.json            # every raw file: path, size, mtime, sha256, parsed timestamp
  registration.parquet     # scanner, frame, filename, timestamp, dx, dy
  plates/<scanner>_<plate>/
    plate.json             # circle + bbox in raw coordinates, mask type, detector + params, code version
    mask.png               # 1-bit working-area mask (a few KB)
    labels.zarr            # uint16 [T, H, W] colony labels, chunked + compressed
    tracks.parquet         # colony_id, frame, area, cx, cy, bbox, merged_flag
    colonies.parquet       # colony_id, status, appearance_min, growth_lb_ub_min, ...
    review.json            # manual include/exclude decisions (append-only log)
```

Estimated size per plate: label stacks are mostly background and compress well, so a few MB, compared with
about 360 MB of ROI TIFFs per plate today. Crops for display or training come from
`ScanSeries.read_window(frame, bbox, offset)`.

### 4.2 Guarantees

* `ScanSeries` opens files read-only (`tifffile.memmap(mode="r")` / `open(..., "rb")`). It has no write methods.
* All writes go through `ProjectStore`, which resolves every target path and **refuses** any path that is inside a registered raw directory or matches a manifest file.
* On every run, sizes and mtimes are compared against the manifest, with full sha256 on demand. A change raises a warning that names the files.
* CI test: run the full pipeline on a fixture experiment and assert that the raw-directory hash is unchanged.
* The old `*_motions.mat` files are only ever *read* (importer).

---

## 5. Deep-learning colony detection and tracking

### 5.1 Pre-trained models reviewed

| Candidate | Fit | Verdict |
|---|---|---|
| **Cellpose-SAM** (cellpose 4.0.6, already installed in base env) | PyTorch generalist instance segmentation. Handles touching round objects and can be fine-tuned. Trained on microscopy, not flatbed scans. Weak on 3–5 px objects | **Best starting point.** Benchmark zero-shot, then fine-tune |
| AGAR-based Faster R-CNN / YOLOv8 detectors (e.g. `leodr/colony-detection`) | Bounding boxes only, so no area. Trained on endpoint photos of plates with large colonies under different imaging. No official AGAR weights confirmed | Not suitable |
| Colony Grounded SAM2 (2026 preprint, zero-shot) | Endpoint images. Heavy. Unverified on tiny early colonies | Optional quick test, not a foundation |
| Lens-free / TFT time-lapse DNNs | Custom hardware, no released weights for our setting | Not applicable |

Conclusion: **no pre-trained model fits ScanLag data as is.** It is 300 dpi flatbed RGB, colonies first appear at
10–20 px, and the signal is temporal (change relative to background). Plan: a custom PyTorch model trained on
our data, with fine-tuned Cellpose-SAM as the baseline it has to beat.

### 5.2 Training data

* **Pseudo-labels from existing results:** 130 analyzed plates and about 10.8k tracked colonies on `D:\Alon\...\microcolony-seq`.
  The MATLAB pipeline doesn't save masks, so `classical.py` regenerates them exactly from the saved `TH`, `Limits` and mask,
  which also validates the port. Then:
  * keep colonies with status 0 (manually reviewed plates are the most trustworthy);
  * **split merged blobs** by seeded watershed from the colony instances of the last pre-merge frame, so the targets learn to separate touching colonies;
  * mark uncertain regions (excluded colonies, merge boundaries with low confidence, plate rim) as *ignore* in the loss.
* **Gold set for evaluation:** about 20–30 tiles of 256×256 px across plates and time points (early, mid, dense), corrected by hand in napari. Roughly 2–3 h of annotation. This is required: scoring only against pseudo-labels would only show that we reproduce MATLAB.

### 5.3 Model

* **Input** (per tile, at frame *t*): RGB(t) − RGB(background), gray(t), gray(t−k), as 5–7 channels. Time context is what separates a new colony from dust or condensation.
* **Network:** U-Net with a ResNet-34 encoder (ImageNet init). Heads: foreground probability, colony-center heatmap, and distance-to-boundary map.
* **Post-processing:** seeded watershed on the distance map with peaks from the center heatmap. This gives instance labels, and touching colonies come out separated.
* **Training:** 256² tiles, flip/rot90/brightness/noise augmentation, Dice + BCE (+ MSE for distance), ignore mask. Expect under an hour on the 4080.
* **Inference:** tiled with overlap, about 1 s per plate-frame on GPU, with a CPU fallback.

### 5.4 Tracking (all detectors)

Colonies don't move and only grow. That allows a simpler and more robust tracker than seed pixels:

1. Match instances at frame *t* to instances at *t−1* by pixel overlap (IoU / Hungarian).
2. If one blob at *t* overlaps several colonies from *t−1* (a merge), split it by watershed seeded from those colonies, so **each colony keeps its own area after merging**. Flag the colony `merged` from that frame on.
   **Done** (`scanlag/track/tracker.py`, `--tracker footprint`, the default): each colony keeps its previous pixels inside the blob, new pixels go to the nearest previous footprint. A blob overlapping colonies only weakly (< 10 % of their previous area) goes to the main one instead of starting a new colony. On EPEC260216 1-4.1u plate 2 (1201 colonies with the seed tracker), replaying the stored U-Net labels: 1144 colonies, the median area ratio at merge onset drops from 2.64× to 1.06×, and "large at appearance" flags from 45 to 18. Growth on merged frames is opt-in (`--merged-growth split`); the default still excludes them.
3. A new colony needs `min_area` and persistence over N frames, which rejects one-frame noise.
4. Border status comes from a distance transform of the mask instead of convolution.

Effect on the science: growth time lb→ub becomes measurable for colonies that touch a neighbour before reaching ub
(today these are dropped). This removes a bias against dense plates and fast-growing neighbours. The `merged`
flag is kept, so the old exclusion rule can still be applied for comparison.

### 5.5 Evaluation and acceptance

* Detection F1 at IoU 0.5 on the gold set, overall and for small (< 30 px) colonies.
* Error in appearance time (minutes) against the gold set.
* Accuracy of separating merged colonies.
* At the population level, against MATLAB on the same plates: KS distance between appearance-time distributions, and the median shift.
* Appearance is defined as the first frame with area ≥ `appearance_area` (configurable). Otherwise a more sensitive detector would shift every distribution earlier and make results hard to compare with older experiments.

---

## 6. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | Env, package skeleton, config, `ScanSeries` (read-only), log parser, MATLAB importer, `ProjectStore` + raw guard, tests | Old experiments load. Raw-hash test passes |
| M1 | Classical parity pipeline: plates → registration → mask → classical detect → MATLAB-style tracker → metrics | On the 130 plates: same colony count ±1% and per-colony appearance time identical for ≥ 95% of matched colonies |
| M2 | napari GUIs: mask editor and plate review (time slider, overlays, area curves, include/exclude) | Feature parity with setMaskApp and PlateAnalyzer |
| M3 | Condition analysis and plots (port of `epec_*.m`), microcolony-seq CSV export | Figures reproduce the MATLAB ones for a past experiment |
| M4 | DL: pseudo-labels, gold set, Cellpose-SAM zero-shot/fine-tuned baseline, U-Net training and evaluation | U-Net ≥ baseline on gold-set F1 and appearance-time error |
| M5 | New tracker with merge splitting as the default, plus a report comparing it with MATLAB | Documented differences in appearance and growth distributions |

---

## 7. Decisions (2026-10-21)

1. **Output location:** separate from raw data, and always a parameter. Default: `<project root>/results/<data folder name>`
   (`scanlag.store.default_results_dir`). A results directory records its raw directories and refuses to be
   reused for another experiment that has the same folder name. Outputs include masks and area-tracking tables.
2. **GUI:** napari.
3. **Test set:** the existing MATLAB detections, keeping only colonies with `IgnoredColonies == 0` (see §8),
   plus a set of tiles for manual correction.
4. **Old ROI folders:** keep them for now. No cleanup command.

**Environment:** conda's base install currently fails (`JSONDecodeError` while loading channel data), so the
project uses a venv at `.venv` (Python 3.11.5, torch 2.6 + CUDA 12.4, napari 0.9.2).

> **Side note (TODO):** the venv is temporary. Once the project is working and validated, move to a conda
> environment. Likely fix for the conda error: `conda clean --index-cache`, then
> `conda create -n scanlag-py python=3.11` and `pip install -e .[gui,dl,dev]` from `pyproject.toml`. Re-run the
> test suite (including the MATLAB parity test) in the new env before switching.

## 8. MATLAB reference test set (`testset_matlab/`)

Built by `scripts/build_testset.py`. For each of the 123 analyzed plates, MATLAB detection and tracking are
replayed from the original ROI images. The port is **bit-exact**: every Area entry matches, including the
1,166-colony plate. This recovers the per-frame masks MATLAB never saved.

```
testset_matlab/
  index.csv                         # one row per plate: counts, parity, errors
  plates/<exp>__<cond>__<plate>/
    labels.zarr   uint16 [T,H,W]    # 0 background, k = MATLAB colony k (kept colonies), 65535 = IGNORE
    mask.png                        # working-area mask
    tracks.parquet                  # frame, time_min, colony_id, area_px, x, y, x_raw, y_raw, merged
    colonies.parquet                # status, appearance, first merge, final area, seed
    plate.json                      # raw scan per frame, crop offset in each raw scan, TH, limits, parity
  manual/tile_XX_<category>/        # tiles for manual correction (scripts/select_tiles.py)
```

IGNORE covers everything MATLAB segmented that isn't a single kept colony: colonies that were excluded
manually, near the border or with zero area; merged blobs; and blobs below the 10 px minimum. Evaluation
must neither reward nor penalize predictions there.

Crop → raw mapping (verified exactly): `ROI[k] = raw[k][y0+round(Σv) : , x0+round(Σu) : ]`, with (x0, y0) from
template matching of frame 0 and (u, v) the MATLAB motions.

**Result (2026-10-21):** 122/123 plates built (188 MB in total), all bit-exact with MATLAB, all raw positions exact.
8,950 kept colonies out of 10,788. Excluded: `EPEC260323/AMP7/ROI/sep_2_6`, because 11 of its ROI images are missing.
A before/after snapshot of the raw data tree (27,153 files, 363 GB) showed no changes. Manual tiles: 24 tiles
(8 early / 8 merged / 8 excluded) across 7 experiments, in `testset_matlab/manual`, corrected with
`scripts/annotate_tiles.py`.

**Gold set (2026-10-21):** all 24 tiles corrected (`labels_corrected.png`). Objects went from 569 in the MATLAB
labels to 562 after correction. Early tiles: +3 missed colonies. Merged tiles: +7 from splits (tiles 09 and 14).
Excluded tiles: −18, mostly in tile 17 (12 → 3). 11 tiles were left unchanged. No IGNORE pixels remain, and no
label is split into disconnected pieces. **Known gap:** the merged tiles still hold 89 MATLAB merged blobs that
were never edited (tiles 10–13, 15 and 16 are unchanged), and Cellpose splits 68 of them into 2–10 colonies.
Until they are split, the merged-category scores favour MATLAB.

## 9. Gold-set evaluation (`scripts/evaluate_goldset.py`)

`scanlag/eval/instances.py` matches predictions to gold colonies (IoU > 0.5, so each colony has at most one match). It
reports pooled F1, precision and recall, plus small-colony F1 (< 30 px: recall over small gold colonies,
precision over small predictions). Objects lying mostly outside the plate mask are ignored. Outputs are written to
`testset_matlab/eval/` (`summary.csv`, per-tile CSVs, predicted labels, `overlays.png`). Gold colonies: 561 in
the working area, 320 of them small (median area 18 px).

| Detector | F1 | P | R | small F1 | early F1 | merged F1 | excluded F1 |
|---|---|---|---|---|---|---|---|
| MATLAB, all thresholded blobs | 0.961 | 0.954 | 0.968 | 0.952 | 0.973 | 0.978 | 0.820 |
| MATLAB, kept colonies only (after tracking and review) | 0.731 | 1.000 | 0.576 | 0.723 | 0.742 | 0.675 | 0.892 |
| Cellpose-SAM zero-shot, best (`diff`, diameter 6, min_size 5) | 0.462 | 0.456 | 0.467 | 0.309 | — | — | — |
| Cellpose-SAM zero-shot, `diff`, d10, min5 | 0.446 | 0.430 | 0.463 | 0.274 | 0.288 | 0.532 | 0.883 |
| Cellpose-SAM zero-shot, RGB, defaults | 0.339 | 0.414 | 0.287 | 0.000 | | | |

Notes:
* The MATLAB numbers are optimistic. The gold labels started from MATLAB's segmentation, so its outlines match
  exactly (mean IoU 1.00), and its merged blobs are still in the gold set (see above).
* Zero-shot Cellpose-SAM fails mainly on tiny early colonies. It needs rescaling (`diameter` 6–10) and
  `min_size` < 15 to find them at all. At IoU 0.1 the best variant reaches F1 0.75, so part of the gap is
  outline precision on 3–5 px objects, where one pixel changes IoU a lot.
* Cellpose-SAM separates touching colonies that MATLAB keeps as one blob. That is the behaviour the merged
  category should reward once the gold set is split.
* The Cellpose grid (3 inputs × 3 diameters × 2 min_size) was chosen on the gold set itself. Treat the "best"
  row as an upper bound for zero-shot use, not as a held-out score.

## 10. Fine-tuned Cellpose-SAM (`scripts/train_cellpose.py`)

Decision (2026-10-21): train on **all** MATLAB test-set plates plus all 24 gold tiles. Nothing is held out.
New, unseen data for testing will be provided later. Until then, gold-set scores only show how well the model fits
its own training data.

* **Data** (`scanlag/train/dataset.py`, `testset_matlab/train/tiles.npz`): 2,610 tiles of 256×256 px. That is 2,586
  MATLAB tiles from 88 plates plus the 24 gold tiles. MATLAB frames are taken at the 5–95 % quantiles of
  colony appearance, plus one frame before any colony appears and the last frame. Tiles are centred on colonies,
  plus one random tile per frame. MATLAB tiles that overlap a gold tile are dropped. 34 plates give no tiles: their
  working-area mask is a few hand-drawn patches (in most of them under 1 % of the plate) around 0–18 colonies.
* **Labels:** kept colonies. The loss ignores IGNORE pixels (merged, excluded or rejected blobs), everything
  outside the mask, and in the gold merged tiles the MATLAB merged blobs that were never edited. Pixels marked as
  background in MATLAB tiles include faint colonies that are just below the threshold, so the model learns MATLAB's
  appearance threshold, not an earlier one.
* **Input:** 3 channels `[gray(frame − background), gray(frame), gray(frame − 4)]`, normalised with fixed
  per-channel limits from the training set, which are stored in `config.json`.
* **Training** (`scanlag/train/cellpose_finetune.py`): a custom loop with Cellpose's flow + cell-probability loss,
  masked by the ignore region. Tiles are upscaled ×3 (±20 %), so inference uses `diameter = 10`. Augmentation:
  rot90/flip, gain and offset. AdamW, lr 3e-5, warmup then cosine decay, 8,000 steps × effective batch 4
  (2 × 2 accumulation; batch 4 needs about 19 GB on the 16 GB 4080). About 65 min.
* **Output:** `models/cpsam_scanlag_v1/{model.pt, config.json, history.json}` (1.2 GB). Scored with
  `python scripts/evaluate_goldset.py --detectors model:models/cpsam_scanlag_v1`.

**Result v1 (2026-10-21, training fit, NOT a test score):** training took 67 min, and the loss went from 0.042 to about 0.01 (noisy).
With cellprob threshold −2 (chosen on the gold tiles; 0 gives F1 0.744 with recall 0.62, −1 gives 0.819, −3 gives 0.775):

| Subset | F1 | P | R | small F1 | mean IoU |
|---|---|---|---|---|---|
| all | 0.851 | 0.919 | 0.791 | 0.782 | 0.83 |
| early | 0.804 | 0.940 | 0.702 | 0.800 | 0.75 |
| merged | 0.896 | 0.883 | 0.909 | 0.308 | 0.90 |
| excluded | 0.935 | 1.000 | 0.878 | — | 0.90 |

Compared with zero-shot (F1 0.46), it now rejects writing and rim reflections, and finds far more tiny colonies.
It is still below MATLAB on its own training tiles, for two reasons:
* It misses some faint early colonies (early recall 0.70). MATLAB tiles label faint colonies below the threshold
  as background, which teaches the model to be cautious.
* **It skips dense clusters of touching colonies** (see `testset_matlab/eval/overlays_v1.png`, tile_16). Every
  touching cluster in the MATLAB labels is IGNORE, so the model never saw a labelled one. Fix: split merged blobs by
  seeded watershed from the colonies of the last frame before they merged (§5.2), and split the remaining 89 gold blobs.

## 11. First unseen test: VapC_06_08_25 (`testset_vapc/`, `scripts/evaluate_plates.py`)

`D:\Alon\Documents\Experiments\SYTOX_06_08_25\scanlag\VapC_06_08_25`: 12 plates (T2 and T24 × 6), never used in
training. MATLAB was replayed bit-exactly with `scripts/build_testset.py ... --out testset_vapc`. MATLAB found
2,338 colonies, and 2,009 were kept (0.859): 238 were excluded near the border (automatic), 82 manually, 9 with zero area.

**Reference** = the colonies the user kept, identified by their MATLAB seed pixel (`scanlag/eval/seeds.py`). A detection
is correct if it covers a kept colony's seed (including colonies MATLAB accepts only later). Recall is one-to-one
over the kept colonies that exist at that frame. The model runs on the full plate crop. Each plate is scored at
the median appearance frame of its kept colonies ("mid") and at the last frame.

| Frame | Detector | Detections | Precision | Recall | FP near border | FP user-excluded | FP no MATLAB colony | Kept seeds covered |
|---|---|---|---|---|---|---|---|---|
| mid | MATLAB, all blobs | 1,712 | 0.879 | 0.996 | 101 | 14 | 90 | 100 % |
| mid | fine-tuned v1 | 1,610 | **0.943** | 0.994 | 88 | 1 | 0 | 99.8 % |
| last | MATLAB, all blobs | 1,471 | 0.756 | 0.554 | 131 | 50 | 178 | 100 % |
| last | fine-tuned v1 | 1,291 | **0.948** | 0.609 | 62 | 3 | 2 | 83 % |

By condition (last frame), MATLAB precision is 0.64 (T2) and 0.85 (T24), and the model's is 0.95 for both.

* Nearly all of the model's false positives are colonies MATLAB excluded automatically for being near the
  border (within 15 px). The user rejected only 1–3 of them by hand. If near-border colonies are left out,
  its precision is about 0.99.
* MATLAB's extra false positives are mostly small specks and debris with no colony behind them, plus 50 blobs
  the user excluded by hand. These are exactly the ones the review step removes.
* **Recall at the last frame** is low for both detectors because 1,381 of the 2,009 kept colonies merge by the
  end, and one blob can only find one colony. The model leaves 17 % of kept seeds uncovered at the last frame;
  310 of those 337 are merged colonies (the touching-cluster gap, §10). At the mid frame, both find over 99 %.

## 12. Light model: U-Net (`scanlag/detect/unet.py`, `scripts/train_unet.py`)

Why: Cellpose-SAM takes about 5.5 s per full-plate frame, with the GPU at about 93 % (ViT-L at 3× upscale, about
170 tiles of 256² per frame). Tracking the 1,320 frames of EPEC260224 would take about 2 h.

* **Model:** `segmentation_models_pytorch` U-Net with a ResNet-34 encoder (ImageNet init), about 24M parameters
  (Cellpose-SAM has about 300M). Two outputs: foreground, and the distance to the colony edge scaled to 1 within each
  colony. Same tiles, ignore masks, input channels and fixed normalization as Cellpose v1. Input upscaled ×2. Loss:
  BCE + soft Dice + 5 × MSE (distance), masked. AdamW lr 1e-3, cosine schedule, 10k steps × batch 16: **12 min**.
* **Instances:** foreground > 0.4. Cores are where the smoothed distance (σ = 1) is > `core`; they seed a
  watershed, and pieces without a core stay whole. The predicted distance peaks at about 0.75–0.85, so `core` ≥ 0.85
  means no splitting. Default `core = 0.7`, picked before the VapC run as a middle value. On VapC, 0.7 was
  also the best of 0.5 / 0.7 / 0.85.
* **Speed:** 0.05 s network + 0.08 s post-processing per 1041² frame, about **40× faster** than Cellpose v1.

| Test | Detector | Precision | Recall | Kept seeds covered | Gold F1 (training fit) |
|---|---|---|---|---|---|
| VapC mid | MATLAB, all blobs | 0.879 | 0.996 | 100 % | 0.961 |
| VapC mid | Cellpose v1 | 0.943 | 0.994 | 99.8 % | 0.851 |
| VapC mid | U-Net v1 (core 0.7) | 0.941 | 0.998 | 99.9 % | 0.785 |
| VapC last | MATLAB, all blobs | 0.756 | 0.554 | 100 % | |
| VapC last | Cellpose v1 | **0.948** | 0.609 | 83.2 % | |
| VapC last | U-Net v1 (core 0.7) | 0.932 | **0.866** | **98.7 %** | |

* On unseen plates the U-Net is as precise as Cellpose (−0.016 at the last frame, mostly extra near-border
  colonies: 103 vs 62). It is far better on touching colonies: it covers 98.7 % of kept seeds at the last frame
  vs 83 %, and one-to-one recall is 0.87 vs 0.61.
* Its lower gold F1 (0.785) comes from the merged tiles (precision 0.67). It splits clusters that the gold set still
  holds as unsplit MATLAB blobs (§8 gap), so the gold set punishes exactly what VapC rewards. Splitting the 89
  gold blobs would settle which split is right. With `core` ≥ 0.85 (no splitting), gold F1 is 0.842.
* **U-Net v1 is now the default detector for whole-plate runs.** `load_detector(model_dir)` picks the right class
  from `config.json`.

### 12.1 Area tracks, EPEC260224 (`scripts/track_with_model.py`, `scripts/plot_area_tracks.py`)

The U-Net ran on all 1,320 frames of the 12 plates in **4 min**. Colonies are followed by their MATLAB seed
(`scanlag/track/seed_areas.py`). Figures are in `testset_matlab/model_runs/unet_scanlag_v1/figures/`. Caveat: 312
training tiles come from this experiment.

* **AMP7:** the MATLAB and U-Net tracks lie on top of each other. The median time to reach 20 px differs by 0 min
  (one frame = 15 min).
* **CTRL1 / CTRL2:** the U-Net reaches 20 px **15–60 min later** than MATLAB, and the curves meet above about 50 px.
  This follows MATLAB's contrast limits, which come from the plate's *last* frame (problem 6 in §1.3). The effective
  MATLAB threshold is TH × upper limit: about 0.028 on AMP7 plates (limit about 0.14, colonies large at 42 h), but
  0.013–0.024 on CTRL plates (limit 0.063–0.118, colonies still small at 21–24 h). The lag grows as the limit drops:
  limit 0.063 → 60 min, 0.075–0.086 → 30–45 min, 0.10–0.12 → 15 min, about 0.14 → 0 min. So MATLAB's appearance
  time depends on how long a plate was scanned. The U-Net uses fixed normalization, so it applies the same
  sensitivity to every plate.

**Leakage:** if MATLAB labels are also used as training pseudo-labels (§5.2), training and test must be split
**by experiment**, so a model is never tested on plates it was trained on.

## 13. Analysis pipeline (`scripts/scanlag_pipeline.py`)

```
python scripts/scanlag_pipeline.py <scan folder> [--detector unet|cellpose|classical|<model dir>] [--out DIR]
```

1. **Scans** (`scanlag/pipeline/scans.py`): every `<prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f]` under the folder
   (subfolders too; MATLAB `ROI` folders are skipped), grouped by scanner. The scanner name is the file-name token,
   which is also the name the robot log uses (e.g. `1-5.1u`). The first scan is the background. Time zero is the log's
   "Starting." line, or the first scan if there is no log.
2. **Plates** (`scanlag/pipeline/plates.py`): Hough circles on the background scan (90 mm at 300 dpi), refined by a
   least-squares fit to the outer rim edge along 360 rays. Plates are numbered as in MATLAB: by column, then top to
   bottom (columns are grouped by x-gap; MATLAB's 10 %-rounding misnumbers columns that straddle a rounding
   boundary, as on VapC T2). Slots whose inside is as dark as the scanner lid (median gray < 25) are skipped as
   empty. Alignment: phase correlation of the edge image, then the integer shift within ±1 px that minimises the
   mean |gray − background| inside the working area. Mask: circle of 0.8 × radius (MATLAB default).
   Checked against MATLAB on EPEC260224: centres within 1–8 px, radii within about 1 %.
3. **Segmentation + tracking** (`scanlag/pipeline/run.py`): one memory-mapped read per scan, every plate cropped
   at its aligned position. Detector: U-Net v1 (default), Cellpose-SAM v1, or classical (MATLAB im2L with
   contrast limits from the last frame, plus a **floor of 0.06 on the upper limit**, since on sparse or empty
   plates the MATLAB limit falls to a few gray levels and noise floods the plate). Tracking uses MATLAB's
   seed-pixel tracker for every detector. Status 1 (near border, 15 px) and 3 (zero final area) are set
   automatically.
4. **Flags** (`scanlag/pipeline/flags.py`): gap, transient merge, shrinks (< 75 % of the running maximum), jump
   (> 2× between frames), stays small (< 20 px), large at appearance (> 20 px in the first frame: a piece split off a
   neighbour for one frame), irregular shape (solidity < 0.85), elongated (eccentricity > 0.85).
   `--reflag` recomputes them without segmenting again. The GUI's flagged list can be limited to one kind and has
   Exclude selected / Exclude all listed.
5. **Review GUI** (`scanlag/gui/review.py`, PyQt5 + matplotlib): PlateAnalyzer layout (plate + time slider on the
   left, area curves on the right, click either to select, Ctrl+E / Ctrl+I, find colony no., PlateAnalyzer's
   number colours). New: a list of flagged colonies with their reasons, an orange "!" on the plate, and F / Shift+F to
   step through them. Also: zoom/pan, "Labels only" (MATLAB's analysis view), show excluded, flagged only, log
   scale, PageUp/PageDown between plates. Decisions are saved immediately (`colonies.parquet` + `review.json`
   log).
6. **Export** (`scanlag/pipeline/export.py`, "Finish & export"): `<project root>/results/<folder>/<scanner>/plate_<k>/`
   `tracks.csv`, `colonies.csv` (status, appearance, 20→80 px growth time, flags), `area_matrix.csv`, plus
   `all_colonies.csv`, `all_tracks.csv` and `summary.csv` at the top level. Working files (`labels.zarr`, parquet,
   `plate.json`, `mask.png`) are kept next to them, so `--review-only` reopens the GUI.

**EPEC260224 (U-Net, no review yet):** 4 scanners and 371 scans took 11.5 min. Included colonies vs MATLAB kept:
CTRL1 26/40/40/29 vs 25/43/39/24, CTRL2 69/55/61/76 vs 66/56/59/72, AMP7 7/109/5/117 vs 8/121/5/131 (plates 1/2/4/5).
AMP7 plates 3 and 6 (about 1,000 colonies each) and all of AMP5 were never analysed in MATLAB. **AMP5 is not a
proper time-lapse:** its first scan already shows grown colonies (no 2 h background wait, 16 min between the first
scans), so background subtraction leaves only crescent-shaped growth rims.

## 14. U-Net v2: training without low-contrast plates (2026-10-22)

Check of the training set for MATLAB's last-frame contrast glitch. No training tile comes from a flooded plate:
the plates with upper limit 0.012–0.059 are hand-drawn-patch plates that gave no tiles, and every plate that gave
tiles has upper limit ≥ 0.063. None of those plates has more than 10 % of its kept colonies ending below 30 px.
Noise specks are only ever in IGNORE. The remaining effect is rough, oversized kept-colony outlines on low-limit
plates.

v2 = v1 minus the 245 tiles from the 9 plates with upper limit < 0.08 (`scripts/train_unet.py --min-upper-limit 0.08`;
2,365 tiles, 10k steps, 14.7 min). Dropped plates are listed in `models/unet_scanlag_v2/config.json`.

| | VapC mid P / R | VapC last P / R | kept seeds covered (last) | gold F1 | gold early F1 | gold small F1 | gold merged F1 |
|---|---|---|---|---|---|---|---|
| U-Net v1 | 0.941 / 0.998 | 0.932 / **0.866** | 98.7 % | **0.785** | 0.789 | 0.764 | **0.757** |
| U-Net v2 | 0.941 / 0.997 | **0.937** / 0.843 | **99.5 %** | 0.749 | **0.804** | **0.784** | 0.663 |

The differences are small and point both ways. v2 is slightly more precise and finds slightly more small and early
colonies, but separates fewer touching colonies at the last frame (VapC) and splits more on the gold merged tiles
(which still hold 89 unsplit MATLAB blobs). With one training run each, this is within what a different random
seed could change. **v1 stays the default.** Removing the low-contrast plates neither helped nor hurt clearly.

## 15. Condition analysis (`scripts/scanlag_analysis.py`, `scanlag/analysis/`)

* **Layout:** `<results>/conditions.csv`, one row per (condition, scanner), with columns condition, scanner
  and plates (`1,2,3` or `1-3`). A condition can span several scanners.
  * If the file is missing, a template listing every scanner and its plates is written.
  * `--condition "NAME=scanner:plates;scanner:plates"` is the command-line alternative.
  * A plate assigned to two conditions is an error; plates without results (empty slots) are skipped
    with a warning.
* **Data:** included colonies only, read from the plate parquet files, so the current review decisions
  are used.
* **Metrics per colony:**
  * appearance time: the first frame the colony is detected;
  * radial growth rate: the slope of a least-squares fit of sqrt(area) against time, in px^0.5/h (also
    converted to µm/h of disc radius). It is fit over the colony's frames `--fit-frames`, default 1-12,
    counted from appearance (1 = the appearance frame), and stops before the colony first merges. At
    least 4 points are needed; R² and the number of points are stored;
  * doubling time: t(≥ub) − t(≥lb), with `--doubling-range`, default 20 80 px (MATLAB
    getAppearanceGrowth). It is NaN if the colony merged first or jumped past both thresholds between
    two scans.
* **Output:** `<results>/analysis/`:
  * per condition: `colonies.csv`, `area_tracks.csv` (long) and `area_matrix.csv` (colony × frame);
  * overall: `all_colonies.csv`, `summary.csv` and `settings.json`;
  * `histograms.html`: interactive plotly figures, one panel per condition, showing the fraction of
    colonies and the median, opened in the browser when the script finishes (`--no-open` to skip). The x
    axis runs from 0 to 1.1 × the largest value. Time histograms are in minutes, with bins
    one scan interval wide, centred on the scan times (`--bin` in minutes); the radial-rate bins are
    0.05 px^0.5/h wide (`--rate-bin`).
* **Outliers:** on EPEC260224 the full x range of the radial rate runs from −3 to 24 because of 57
  colonies (2%). These are short, flagged tracks on the dense AMP plates.

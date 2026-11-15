# Running the microscopy pipeline on the lab PC: suggestions

**Status: a suggestion, not a decision.** This was written by a Claude session on the Mac on 2026-11-15. It records what that session measured and how it would approach the work. The agent working on the PC should check these points and do things differently where it finds a better way. Nothing here has been tried on the PC yet, and the open questions at the end still need answers from the user.

## Goal

A general pipeline for a time-lapse microscopy experiment. It segments cells and follows each founder cell's fate. For every founder, over time, it reports:

1. size;
2. refractility, measured from phase contrast;
3. fluorescence per pixel in each channel;
4. time to first division, if the founder divides.

The tracking must be easy for a human to review. Deliverables differ per project.

**Machines**
- **Lab PC** (`DESKTOP-I8INR24`): Windows 10 Pro, i9-9900X (10 cores, 20 threads), 128 GB RAM, RTX 2080 Ti (11 GB), driver 560.94. C: has 38 GB free and E: 252 GB.
- **Mac:** an M1 Pro with 8 cores, 16 GB RAM and about 87 GB of free disk. Final review happens here.

## What the KLY/TBL8 PI pilots showed (measured on the Mac)

**Data**
- Pilot_3: 24 fields × 163 frames, 2048×2048 16-bit, with phase and mCherry (PI) at every frame and YFP every 11 frames. 64 GB raw.
- Pilot_1 and Pilot_2 are 4.7 GB and 29 GB.

**Time**
- Colony-tracker run on Pilot_3 with 4 field workers: 73 min of tracking, then 1.5 min of analysis.
- Segmentation (watershed adapter) takes 1.3–2.8 s per frame, which is roughly half of the 73 min. The other half is per-object bookkeeping, graph building and saving, and has not been profiled.
- Reading images is negligible.

**Most of each frame is empty.** Cells cover 0.2% of pixels at frame 0 and 0.8% at frame 160, yet every frame is segmented over the full 2048×2048.

**Frame QC**
- Pilot_3 field 0, frame 80 produced 8,268 objects covering 47% of the frame. That is probably a bad image segmented into noise. Nobody has looked at the image yet.
- `min_focus_score` is not set, so frames like this go through the whole pipeline.

**Accuracy on Pilot_3 Set A** (87 human-tagged founders; reproduce with `score_against_tags.py`)
- Agreement: 73 of 87.
- Divisions: 42 of 44 found. Of the 42, 21 are on the exact frame, 35 within ±1 frame and all 42 within ±2.
- Not grown: 6 of 6.
- False division calls: 6, all on founders tagged as doublets.

**Remaining error sources**
- Founders that were already doublets at frame 0.
- Touching cells.
- Debris: 131 founders have almost no initial YFP (below 8), and only 2% of them divide.
- The "possible doublet" flag fires on 22–34% of founders but caught 5 of 6 real doublets.

## Suggested division of work

**PC: the computing**
- One command per experiment: read metadata, check frame quality, segment (possibly on the GPU), track, measure, then produce review data.
- Keep data on E:, never on C:.

**Mac: review and final analysis**
- The analysis step is cheap (about 1.5 min).

**Two ways to review from the Mac:**
1. **Serve the viewer from the PC; the Mac uses a browser.** Measured on 2026-11-11:
   - Both machines are on the same university subnet (132.64.81.x). The Mac is on wired gigabit, and the round trip is 1.25 ms.
   - A display-size crop is 20–60 KB, so the network adds only a few ms per frame step.
   - The PC can cache a whole experiment in RAM.
   - Drawbacks: it only works on campus (the VPN is untested), the PC must be on, and it needs a firewall rule limited to the Mac's address plus a token.
2. **Export a review bundle and copy it to the Mac.**
   - The bundle holds per-founder crops for all frames and channels, outlines, tracks and measurements. Estimated 3–6 GB for a Pilot_3-sized experiment.
   - It works offline, and the Mac never needs raw images.
   - Copy it over SMB (the PC's port 445 answers) with rsync, which takes 1–2 min at gigabit.

The user has not yet said whether they review off campus. That decides whether option 2 is needed at all.

## Speed ideas, roughly in order of payoff

These are estimates, not measurements.

- **Profile first:** the unprofiled half of Pilot_3's 73 min.
- **More field workers:** 8 on the PC (keeping 2 cores free) instead of 4. The i9 cores are not faster than the M1 Pro's, so this is worth about 2×, not 10×.
- **Segment only around founders' colonies,** plus a margin, instead of the full frame.
- **A frame-quality gate** that skips broken frames such as field 0 / frame 80.
- **GPU segmentation** (for example Omnipose's bacterial phase-contrast model, or Cellpose). Mainly an accuracy lever for doublets and touching cells. Adopt it only if it beats watershed on Set A, and later Set B.

A realistic target might be 10–15 min for a Pilot_3-sized experiment.

## Measurement definitions to consider

1. **Size**
   - Area, plus length and width since the cells are rods.
   - Pixel size is recorded as 0 by the instrument, so µm/px must come from the objective and camera. This is an open question to the user.
   - After division, report the colony's total area and cell count.
2. **Refractility**
   - Phase darkness of the cell relative to a local background ring, using the same 3–10 px ring scheme as fluorescence.
   - Lysed or ghost cells turn phase-light, so this could flag death.
   - It is focus-dependent, so it needs frame QC. It is relative, not a refractive index.
   - The user has not yet confirmed this definition.
3. **Fluorescence:** keep the current definition, which is the mean in the cell outline minus the median of a 3–10 px ring, per channel. Missing channel frames stay missing, never zero.
4. **Division:** the midpoint of the first-division interval. Lost founders and founders still undivided at the end are censored, not "not grown".

**Output:** a tidy table with one row per founder per frame, plus a table with one row per founder.

## Review

- Order the review queue by how likely the automatic call is wrong: possible doublets, lost tracks and low-YFP objects first. Spot-check confident division calls.
- Use Set A for tuning and Set B as the held-out test, and score every change before adopting it.
- The earlier viewer plan is in `viewer_plan_2026-11-08.md`. Its implementation status has not been checked against the current code.

## Known Windows problems in the current code

- `microscopy/pipeline.py`, around line 86: `os.nice` does not exist on Windows. The code catches only `OSError`, so every field worker raises `AttributeError` once `field_workers > 1`.
- `microscopy/storage.py`, around line 42: `os.sysconf` (used to read RAM size) does not exist on Windows. Loading a run will fail.
- `project.json` stores an absolute `raw_root` (a Mac path). Repoint it on the PC.
- The experiment scripts are Mac-only:
  - `run_colony.py` calls macOS `memory_pressure` and has Mac paths;
  - the viewer launcher is a zsh `.command` file.
- `environment-scripthub.yml` targets the M1 Mac. The PC needs its own environment (CUDA PyTorch if a GPU model is used). Prefer native Windows over WSL2 for data on E:, because WSL2 reads Windows drives slowly.
- Worker processes are spawned, not forked, so keep pipeline calls under `if __name__ == "__main__":` (see `AGENTS.md`).

## Training data: what exists

- **No hand-drawn masks anywhere.** No review contains outlines or splits.
- **Pilot_3 tags** (`Pilot_3_ground_truth.csv`):
  - Founder-level labels: decision, division frame, last single-cell frame, loss frame and notes, including about 8 doublet notes.
  - Set A has 87 tagged founders in fields 2, 11, 18 and 22.
  - Set B has 81 founders in fields 0, 10, 15 and 21, not yet tagged.
- **Algorithm label images** from the colony run exist. They are useful as starting proposals, not as ground truth.
- **Suggested route to a segmentation model:**
  1. Try a pretrained model without training, and score it through the tracker on Set A.
  2. If fine-tuning is needed, have it propose masks on crops from the tagged fields, and let the user accept or fix them in the viewer. The ScanLag code already has this pattern (review training tiles, `scanlag/train/cellpose_finetune.py`).
- **The 87 tagged founders are too few to train a doublet or fate classifier.** They are enough to evaluate one.
- **The tagged fields' raw images and tags travel on a hard drive** as `2026_11_04_microscopy_Pilot_3_tagging_package`. It has its own README.

## Open questions for the user

1. How raw images reach the PC, and where they are archived long term.
2. µm per pixel, or the objective and camera.
3. Whether "refractility" means phase darkness relative to the local background.
4. Whether review happens only on campus (viewer served from the PC) or also off campus (bundle needed).

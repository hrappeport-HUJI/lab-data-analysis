# Colony-following founder events

An alternative to frame-to-frame single-cell linking, added 2026-11-09 after the linked tracker missed most divisions in Pilot_3. Enable it per project:

```json
"segmentation": {"adapter": "watershed", "h_rel": 1.0, "threshold_from_first_frame": true,
                 "min_area_px": 25, "background_sigma_px": 8},
"tracking": {"adapter": "overlap", "events": "colony", "colony": {}}
```

Do not set `max_area_px` for the watershed segmenter. Leave it out so large blobs are kept. The `overlap` tracker still builds the observation graph; with `events: "colony"` the founder decisions come from `microscopy/colony.py` and are stored in `run.json` as `founder_events`.

## Why the linked tracker failed

On the 87 human-tagged Pilot_3 founders (Set A: positions 2, 11, 18, 22), the original run found 10 of 44 divisions:

- **Size cap.** Touching daughters form one blob. Once that blob passes `max_area_px` = 1500, it is deleted, so the founder "vanishes" at its division. This explains 22 of the 33 missed divisions.
- **Strict linking.** The split was seen, but the daughters merged back into one blob in the next frame, so the linker refused to confirm the division (12 misses).
- **Threshold drift.** The automatic threshold rises as colonies fill the field (Pos11: 617 at frame 80, 748 at frame 95). Faint non-growing cells then drop out of the segmentation.

## Method

1. **Segmentation** (`adapters/watershed.py`). A local-contrast threshold is fixed per field from its first frame. There is no upper size cap. Each foreground component is split by an intensity watershed at septa that are brighter than the cell interiors by more than `h_rel` × threshold. Fragments under `min_piece_px` are merged into a neighbour.
2. **Colony following** (`colony.py`). A founder's colony is every object that overlaps the colony's previous footprint, after correcting the field-wide stage shift. The shift is the median displacement of mutually nearest centroids. Pilot_3 Pos18 jumps about 10 px for a single frame.
3. **Events.**
   - **Division** is the first frame at which the colony holds two or more cells, in at least 2 of 3 consecutive observed frames, after the colony has grown at least 1.5×. A grown object with solidity below 0.8 counts as two cells. Daughters kink at the septum before a bright line is visible, and people tag that kink as the division.
   - **Lost (fused)**: the colony shares an object with another colony before dividing. If neither touching colony has grown (both below 1.4×), the blob is instead split by the previous footprints and the founder is flagged `touched_neighbour`.
   - **Lost (vanished)**: no object for more than 2 frames. The loss interval ends at the first missing frame.
   - **Unconfirmed**: a split seen only in the last frames before the colony or the recording ends cannot be confirmed, and is reported as lost (`unconfirmed_division_candidate`).
   - **No division** requires following the colony to the last frame with no unusable frame in between and no delayed start. Otherwise the call is lost (`incomplete_window`).
   - **Interval bounds**: the lower bound of a division is the last *observed* frame before it.
   - **Doublets**: founders with solidity below 0.89 at baseline are flagged `possible_doublet`, not excluded. Excluding them automatically would also drop 2 true dividers and 5 other founders in Set A.
4. **Frames excluded in review after the run** invalidate a cached call that used them. The founder becomes lost (`frames_excluded_after_run`, flag `rerun_needed`) until the analysis is re-run.

## Validation on Pilot_3 Set A (87 human-tagged founders)

Parameters were chosen on these same founders, so the numbers are optimistic. Set B (positions 0, 10, 15, 21) is the held-out check.

Scored from a real pipeline run on a copy of the Pilot_3 project (`run-2f630e490ceb`, 4 positions in 11.5 minutes with 4 processes):

| | Linked tracker (original run) | Colony following |
|---|---:|---:|
| Agreement with the human call | 36/87 | 73/87 |
| Divisions found | 10/44 | 42/44 |
| Division frame exact / ±1 / ±2 | 7 / 9 / 9 of 10 | 21 / 35 / 42 of 42 |
| Not grown found | 0/6 | 6/6 |
| False division calls | 4 | 6 (all tagged doublets; both cells divide) |

- **Lost:** of the 22 founders both call lost, the loss frame is within 2 frames for 17.
- **Doublet flag:** `possible_doublet` marks 5 of the 6 tagged doublets and 4 of the 7 other exclusions. It also marks 6 founders that were fine.

Alternatives tried on the same tags:
- **No size cap, linked tracker**: 15/44 divisions.
- **Colony following on unsplit threshold labels**: 38/44, but divisions were called up to 11 frames late.
- **Omnipose** `bact_phase_omni` (2× downsampled, Apple MPS, 3.5 s per image) **with colony following**: 42/44 divisions and fewer doublet false calls, but always 1–4 frames early and never exact. It also needs a separate environment with numpy 1.26.
- **ScanLag code**: its footprint-based merge splitting was reused here. Its detectors target plate colonies.

## Limits

- **One acquisition.** Tested on one acquisition and one magnification, with rods about 12–16 px wide. Check `h_rel`, `dent_solidity` and `min_piece_px` on a pilot before trusting a new acquisition.
- **Neighbours.** A dividing founder that touches a growing neighbour before dividing is lost (fused). That is correct censoring, but it removes fast-growing fields' late dividers.
- **Re-runs.** Cached events are not recomputed when frame quality changes in review; re-run the analysis.

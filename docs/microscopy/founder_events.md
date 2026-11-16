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
   - **Division at contact (review required)**: if fusion prevents the usual temporal confirmation, a candidate in the immediately preceding frame can be rescued. It must be an eligible founder without a baseline doublet flag, skipped frame, or earlier artificial footprint partition. The two pre-contact frames must be consecutive: the earlier has one cell and at least 1.5x baseline area, and the later has at least 1.7x baseline area plus the existing split/kink evidence. Area must remain between 0.8x and 1.3x the preceding frame's area. For a segmented split, exactly two pieces are required, each at least 25% of their combined area. Elongation alone is insufficient. The interval runs from the last observed single-cell frame through the first detected fusion frame; `last_identity_frame` stays before fusion, and both reason and flag are `division_at_contact`. No shape or growth measurement from the frame that triggers fusion supports the call. Ordinary confirmed divisions take precedence. Disable with `tracking.colony.rescue_division_at_contact=false`; growth and maximum area step are configurable as `contact_division_growth` and `contact_max_area_step`.
   - **Lost (fused)**: the colony shares an object with another colony before dividing. If neither touching colony has grown (both below 1.4×), the blob is instead split by the previous footprints and the founder is flagged `touched_neighbour`.
   - **Lost (vanished)**: no object for more than 2 frames. The loss interval ends at the first missing frame.
   - **Unconfirmed**: a split seen only in the last frames before the colony or the recording ends cannot be confirmed, and is reported as lost (`unconfirmed_division_candidate`) unless the stricter contact-rescue rule above applies.
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

## Contact-rescue validation (2026-11-15)

All project copies and outputs are under `E:/microscopy/work/contact_rescue/`. Each pipeline replay used four workers at below-normal priority. Saved labels were reused with `resume_from`; segmentation settings and manifest digests passed the pipeline validation. All 652 Set A and 936 pilot label planes were checked equal to the copied reference planes. Original projects, raw images, runs and reviews were not modified.

Set A: reference `run-34bebd975c26`; unchanged-code replay `run-ca07b47bb820`; contact-rescue replay `run-f5d8a2804106`. The unchanged-code replay reproduced every supplied reference score. Parameters were evaluated on Set A, not the held-out Set B.

| Set A metric | Before | After |
|---|---:|---:|
| Agreement | 73/87 | 73/87 |
| Divisions found | 42/44 | 42/44 |
| Division calls correct | 42/48 | 42/48 |
| Timing exact / +/-1 / +/-2 | [21, 35, 42] | [21, 35, 42] |
| Not grown found | 6/6 | 6/6 |
| False division calls on tagged doublets | 6 | 6 |

Full confusion table: cells are **before / after**. Rows are human tags; columns are automatic calls. Zero cells are shown explicitly.

| Human tag | Divided | Excluded | Not grown | Unresolved |
|---|---:|---:|---:|---:|
| divided | 42 / 42 | 1 / 1 | 0 / 0 | 1 / 1 |
| excluded | 6 / 6 | 3 / 3 | 0 / 0 | 4 / 4 |
| not_grown | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 0 |
| unresolved | 0 / 0 | 1 / 1 | 1 / 1 | 22 / 22 |

Six-position pilot (fields 16, 2, 22, 24, 30, 6): `run-ebd647d2f174` to `run-1ccc0bad33ff`. Counts include exclusions; `no_division` is the non-grower call, and `lost` is censored, not a negative growth result.

| Condition | Division before / after | Lost before / after | No division before / after | Excluded before / after |
|---|---:|---:|---:|---:|
| EPEC Tol | 4 / 4 | 13 / 13 | 0 / 0 | 2 / 2 |
| EPEC WT | 19 / 19 | 22 / 22 | 0 / 0 | 7 / 7 |
| KLY | 21 / 23 | 21 / 19 | 1 / 1 | 2 / 2 |
| TBL3 | 4 / 4 | 3 / 3 | 5 / 5 | 0 / 0 |
| TBL5 | 7 / 7 | 18 / 18 | 0 / 0 | 9 / 9 |
| TBL8 | 4 / 4 | 20 / 20 | 9 / 9 | 5 / 5 |
| **Total (196)** | 59 / 61 | 97 / 95 | 15 / 15 | 25 / 25 |

Every changed founder (all in field 2, KLY):

| Founder | Before: decision, bounds, last identity | After: decision, bounds, last identity | Flags before -> after | Visual verdict |
|---|---|---|---|---|
| f1-t0-o16 | lost, 43..44, 43 | division, 43..45, 44 | [] -> ['division_at_contact'] | Correct: elongation, a central constriction near 43, and daughter separation/kinking at 44 before detected fusion. |
| f1-t0-o43 | lost, 18..19, 18 | division, 18..20, 19 | [] -> ['division_at_contact'] | Correct: visible constriction at 18 and a pronounced daughter kink at 19 before fusion at 20. |

Both changed from reason `unconfirmed_division_candidate` to `division_at_contact`. No other founder decision, event bounds, last-identity frame, reason or flags changed.

Rendered and personally inspected every changed founder using the scratch variant `changed_sheets.py`, modelled on the production contact-sheet script: `RunReviewModel.render`, phase mode, 96-pixel crop, no outlines, baseline plus frames around both old and new bounds. Sheets: `sheets/f1-t0-o16.png` and `sheets/f1-t0-o43.png`. Verdicts: **2 correct, 0 wrong, 0 unclear division calls**. Exact division times remain uncertain inside the reported intervals; these are visual judgements, not independent tags.

The area-step bound was added after candidate-image inspection exposed neighbour contamination (notably f5-t0-o4). The field-22 examples f2-t0-o11 and f2-t0-o16 remain lost: their apparent split frames include abrupt area gains; the latter also has three pieces. The swallowed non-grower examples remain censored. This change therefore provides a narrow division rescue, not general identity recovery through contact. Generalisation to other acquisitions, held-out tags, and long-term non-grower identity remains unverified.

Tests: `python -m pytest microscopy/tests -q -rs`: **62 passed, 1 skipped**, 2 upstream NumPy/scikit-image deprecation warnings, 86.58 seconds. The skip is the existing Windows symlink test (missing OS privilege). Synthetic label sequences cover pre-contact splits/kinks, censoring bounds and retained identity, slivers, elongation, missing confirmation without contact, skipped frames, prior growth, area jumps/drops, excluded founders, doublets, artificial partitions and disabling the rule. The required combined microscopy/plate-reader command was attempted but plate-reader collection is blocked by missing `openpyxl` and `plotly` in the supplied environment; no packages were installed.

## Limits

- **Limited acquisitions.** Original thresholds were calibrated on Pilot_3; contact rescue was additionally inspected on one six-position pilot, with rods about 12–16 px wide. Check `h_rel`, `dent_solidity` and `min_piece_px` on a pilot before trusting a new acquisition.
- **Neighbours.** Contact rescue only covers a credible split/kink immediately before fusion. It does not infer division from elongation alone or from a merged outline. In particular, a neighbour can enter a colony's overlap assignment before the multi-owner fusion detector fires; the area-step bound rejects some such cases but cannot prove purity of the pre-contact mask.
- **Swallowed non-growers.** Asymmetric footprint carving is intentionally not enabled: coverage of a compact old footprint by a larger colony cannot establish continued identity or absence of division. These founders stay censored. The existing symmetric rule (both colonies below 1.4x) is unchanged.
- **Calibration.** Contact rescue thresholds were inspected on the same six-position pilot reported in the contact validation. These visual judgements are not independent ground truth, and the rule needs blinded validation on another acquisition.
- **Re-runs.** Cached events are not recomputed when frame quality changes in review; re-run the analysis.

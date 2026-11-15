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

1. **Segmentation** (`adapters/watershed.py`). A local-contrast threshold is fixed per field from its first frame. There is no upper size cap. Each foreground component is split by an intensity watershed at septa that are brighter than the cell interiors by more than `h_rel` Ã— threshold. Fragments under `min_piece_px` are merged into a neighbour.
2. **Colony following** (`colony.py`). A founder's colony is every object that overlaps the colony's previous footprint, after correcting the field-wide stage shift. The shift is the median displacement of mutually nearest centroids. Pilot_3 Pos18 jumps about 10 px for a single frame.
3. **Events.**
   - **Division** is the first frame at which the colony holds two or more cells, in at least 2 of 3 consecutive observed frames, after the colony has grown at least 1.5Ã—. A grown object with solidity below 0.8 counts as two cells. Daughters kink at the septum before a bright line is visible, and people tag that kink as the division.
   - **Lost (fused)**: the colony shares an object with another colony before dividing. If neither touching colony has grown (both below 1.4Ã—), the blob is instead split by the previous footprints and the founder is flagged `touched_neighbour`.
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
| Division frame exact / Â±1 / Â±2 | 7 / 9 / 9 of 10 | 21 / 35 / 42 of 42 |
| Not grown found | 0/6 | 6/6 |
| False division calls | 4 | 6 (all tagged doublets; both cells divide) |

- **Lost:** of the 22 founders both call lost, the loss frame is within 2 frames for 17.
- **Doublet flag:** `possible_doublet` marks 5 of the 6 tagged doublets and 4 of the 7 other exclusions. It also marks 6 founders that were fine.

Alternatives tried on the same tags:
- **No size cap, linked tracker**: 15/44 divisions.
- **Colony following on unsplit threshold labels**: 38/44, but divisions were called up to 11 frames late.
- **Omnipose** `bact_phase_omni` (2Ã— downsampled, Apple MPS, 3.5 s per image) **with colony following**: 42/44 divisions and fewer doublet false calls, but always 1â€“4 frames early and never exact. It also needs a separate environment with numpy 1.26.
- **ScanLag code**: its footprint-based merge splitting was reused here. Its detectors target plate colonies.

## Limits

- **One acquisition.** Tested on one acquisition and one magnification, with rods about 12â€“16 px wide. Check `h_rel`, `dent_solidity` and `min_piece_px` on a pilot before trusting a new acquisition.
- **Neighbours.** A dividing founder that touches a growing neighbour before dividing is lost (fused). That is correct censoring, but it removes fast-growing fields' late dividers.
- **Re-runs.** Cached events are not recomputed when frame quality changes in review; re-run the analysis.

## Per-cell lineage adapter (2026-11-15, experimental)

Enable with `tracking: {"adapter": "cell_lineage", "confirmation_frames": 1}`
and leave `tracking.events` unset. Segmentation is independent; precomputed
Omnipose labels are the intended input. Watershed labels are supported but a
merged mask cannot establish two separate identities.

`CellLineageTracker` corrects integer stage drift using `colony.field_shift`.
It counts all nonzero label-pair intersections in one sparse pass. Each child
is assigned to the parent covering the largest fraction of its mask, requiring
at least 20% child coverage and 8% parent coverage. Competing overlaps need a
15 percentage point coverage margin; otherwise the tracker abstains and flags
`ambiguous_assignment`. Other competing overlaps are flagged
`competing_overlap`. Assignment is deterministic local overlap ownership, not
a globally optimal motion model. It has no condition-specific rules.

One child must have 0.45–1.8 times the parent's area. Two children require a
combined area ratio of 0.65–1.6 and individual ratios of 0.2–0.85. A disappearing
cell retains its unoccupied, drift-corrected footprint for one frame. Recovery
is only to one cell and carries `gap_bridge`; it cannot establish a division
inside the gap. A skipped unusable frame may likewise be bridged once. Longer
gaps break continuity. No-division decisions still require a continuous full
recording. Contact between separately segmented neighbours does not itself
terminate either lineage; near-equal ownership of a merged mask still does.

The existing graph contract remains unchanged (at most one parent and two
children). A division needs both daughters to continue independently for
`confirmation_frames` links. For this adapter only, its upper bound is the
last daughter confirmation frame, with the last observed single-parent frame
as lower bound. This deliberately reports persistence/visibility rather than
the first Omnipose constriction. The interval can span more than one frame;
other linked and colony modes retain their previous timing convention.
Terminal unconfirmed splits remain lost. Review quality overrides continue to
filter graph edges. All automatic calls require review.

Synthetic 2048×2048 label pairs containing 8,000 rectangular cells and a (3,4)
pixel shift: 8,000/8,000 links, 0.705 / 0.656 / 0.577 seconds per pair (three
fresh trackers); peak process RSS 174.3 MiB, baseline 61.5 MiB, increment
112.8 MiB, sampled every 2 ms with psutil. This is tracker-only CPU time,
excluding TIFF reading, segmentation, graph serialization and review. A
regular synthetic grid is not a worst-case biological dense colony.

A real Omnipose pilot pair (new experiment Pos24, frames 122?123,
2048×2048, 10,364 current cells) took 1.036 / 1.083 / 1.010 seconds per
pair, with 10,056 graph edges. Peak RSS was 197.9 MiB from a 96.5 MiB
baseline (101.4 MiB increment). These are throughput measurements, not link
accuracy adjudications; the current-cell count exceeds the requested 8,000-cell
stress target. Both benchmarks ran while validation jobs were active.

### Set A acceptance result

**The acceptance target was not met.** The per-cell adapter improves timing
but does not beat Omnipose+colony on agreement or divisions, and its division
precision is lower (95.2% versus 97.7%). It is an experimental option, not a
replacement for production colony following. Parameters and the timing
convention were evaluated on Set A itself; no held-out claim is made.

| Segmentation / events | Agreement /87 | Divisions /44 | Correct division calls | Exact / ±1 / ±2 | Not grown /6 |
|---|---:|---:|---:|---:|---:|
| Watershed + colony (supplied production baseline) | 73 | 42 | 42/48 | 21 / 35 / 42 | 6 |
| Omnipose + colony (supplied baseline) | 72 | 42 | 42/43 | 1 / 26 / 37 | 3 |
| Omnipose + plain overlap (supplied baseline) | 48 | 19 | 19/19 | 0 / 10 / 16 | 2 |
| Omnipose + cell lineage | 67 | 40 | 40/42 | 24 / 35 / 39 | 2 |

Omnipose run: `run-81f693fd8509`, scratch `setA/Pilot_3`. Full pipeline:
292.15 s preparation and 269.34 s replay over 652 frames with 6 preparation
workers / 4 field replays. The full graph contains 4,084,971 observations;
`run.json` is 1.61 GiB. JSON assembly, writing, parsing and review copies are
additional major costs; low per-frame tracker RSS does not describe full-run
review memory. Labels and the prepared source catalog are retained.

Confusion (human rows; automatic columns; centroid/mask matching exactly as in
the supplied scorer, including fallback to the most numerous label within a
15-pixel window):

| Human | Division | No division | Lost | Exclude | Unmatched |
|---|---:|---:|---:|---:|---:|
| Divided | 40 | 0 | 3 | 1 | 0 |
| Not grown | 0 | 2 | 3 | 1 | 0 |
| Unresolved | 1 | 1 | 18 | 2 | 2 |
| Excluded | 1 | 0 | 2 | 7 | 3 |

Signed confirmed-call minus human-tag error (40 correct division calls):
`{-3: 1, -2: 4, -1: 10, 0: 24, +1: 1}`; median 0 frames.
Raw first-split errors: `{-4: 1, -3: 4, -2: 10, -1: 24, 0: 1}`;
median -1 frame. Thus the one-frame persistence convention removes the median
early bias, but five calls remain more than one frame early. Do not apply this
convention shift to other acquisition cadences without validation.

### Watershed comparison

Same fields and adapter defaults, with the supplied production watershed
settings: `run-64a4ea41c870`, scratch `setA_watershed/Pilot_3`.

| Segmentation / events | Agreement /87 | Divisions /44 | Correct division calls | Exact / ±1 / ±2 | Not grown /6 |
|---|---:|---:|---:|---:|---:|
| Watershed + cell lineage | 59 | 29 | 29/30 | 1 / 15 / 25 | 6 |

| Human | Division | No division | Lost | Exclude | Unmatched |
|---|---:|---:|---:|---:|---:|
| Divided | 29 | 0 | 14 | 1 | 0 |
| Not grown | 0 | 6 | 0 | 0 | 0 |
| Unresolved | 0 | 2 | 21 | 1 | 0 |
| Excluded | 1 | 0 | 9 | 3 | 0 |

Signed call-tag errors (29 correct calls):
`{-1: 1, 0: 1, +1: 13, +2: 10, +3: 1, +6: 1, +7: 2}`;
median +1. Raw split median 0, distribution
`{-2: 1, -1: 1, 0: 13, +1: 10, +2: 1, +5: 1, +6: 2}`.
Preparation 754.28 s, replay 167.09 s over 652 frames. Watershed merges still
prevent independent identities, and confirmation adds delay to its later splits.

### New-experiment six-position pilot

Fields 2, 6, 16, 22, 24, 30. Corrected per-cell replay
`run-b4ff915bbf99`, scratch `pilot/project`: 227.66 s replay using completed
Omnipose preparation. The failed first attempt `run-34fabb5bb5a5` is preserved:
a drifted-out maximum label caused an undersized lookup table. The fix and a
synthetic regression test cover this case. Inputs were not changed.

Counts below are **division / lost / no_division / exclude**. Native counts
have different founder populations. The last column matches each of the 196
production founders to the Omnipose frame-0 mask at its centroid (same 15 px
fallback as Set A), adding unmatched as a fifth count.

| Condition | Production watershed+colony | Native Omnipose+lineage | Matched production founders, D/L/N/E/U |
|---|---|---|---|
| EPEC Tol | 4 / 13 / 0 / 2 | 3 / 10 / 0 / 6 | 3 / 10 / 0 / 4 / 2 |
| EPEC WT | 19 / 22 / 0 / 7 | 17 / 14 / 1 / 19 | 17 / 14 / 1 / 12 / 4 |
| KLY | 21 / 21 / 1 / 2 | 23 / 9 / 0 / 4 | 23 / 9 / 0 / 4 / 9 |
| TBL3 | 4 / 3 / 5 / 0 | 4 / 6 / 1 / 2 | 4 / 6 / 1 / 1 / 0 |
| TBL5 | 7 / 18 / 0 / 9 | 6 / 17 / 0 / 10 | 8 / 18 / 0 / 8 / 0 |
| TBL8 | 4 / 20 / 9 / 5 | 2 / 16 / 3 / 10 | 2 / 16 / 3 / 6 / 11 |
| Total | 59 / 97 / 15 / 25 | 55 / 72 / 5 / 51 | 57 / 73 / 5 / 35 / 26 |

170/196 production founders matched 164 distinct Omnipose founders (six
additional matches are duplicates); 49 matched records changed decision.
This matching is deliberately scorer-compatible, not a one-to-one identity
reconciliation. Lower loss counts do not establish improved biological recall:
there are more exclusions, unmatched objects and some false divisions.
No human reviews were transferred between runs.

### Visual review of 12 changed founders

Inspected `qc/changes_1.png` through `changes_4.png` and `qc/detail.png`,
rendered with `RunReviewModel.render(mode="phase", width=128, outlines=True)`
(detail width 96), under `E:/microscopy/work/cell_lineage/`.
Each row includes baseline, production loss/contact and the new event frames.
A review-only `pilot_review/project` projects 5,951 observations through first
events, retains every segmentation plane for background exclusion, and points
to the complete unmodified source run/catalog. These are visual verdicts on
sampled frames, not saved human overrides or new ground truth. Initial IDs
below are run-local; production and lineage IDs must not be compared directly.

| Production ID ? lineage ID | Production ? lineage | Visual verdict |
|---|---|---|
| f3-t0-o19 ? f3-t0-o21 (EPEC WT) | fused/lost 10 ? division 14 | Supported recovery: founder elongates after contact at 10; kink at 13–14 and separated daughters by 16. |
| f1-t0-o33 ? f1-t0-o24 (KLY) | fused/lost 35 ? division 13 | Rejected visible-division call: tiny object barely changes through 0–20; lobing at 10–13 persists without convincing visible daughter separation. Could be a constriction/segmentation split; not validated division. |
| f0-t0-o21 ? f0-t0-o21 (TBL5) | exclude/not_isolated ? division 75 | Supported event: rod grows, septum/constriction at 73–75, bent separated daughters by 77. Eligibility differs; not an eligible production recovery. |
| f4-t0-o7 ? f4-t0-o8 (TBL8) | fused/lost 57 ? division 12 | Rejected: persistent ring-like outline at 0,10–14,56–57; segmentation fragments it without convincing bacterial division. |
| f4-t0-o18 ? f4-t0-o16 (TBL8) | fused/lost 46 ? division 54 | Rejected: almost unchanged rod through 45–50, then mask expands into a bright phase/halo region at 52–54 among neighbours. Likely identity/segmentation error. |
| f2-t0-o4 ? f2-t0-o5 (EPEC Tol) | fused/lost 53 ? exclude | Eligibility change, no recovery: ring-like object near other baseline detections; new reason is not_isolated. |
| f3-t0-o4 ? f3-t0-o4 (EPEC WT) | fused/lost 29 ? no_division 155 | Uncertain continuity: original cell plausible at 25–29; tracked outline shifts to the crop edge by 50–75 and is absent from the central late crop. Full-window no-division is not visually validated. |
| f1-t0-o14 ? f1-t0-o13 (KLY) | fused/lost 9 ? exclude | Eligibility change: two close baseline rods violate isolation; new exclusion is reasonable, not recovered lineage. |
| f1-t0-o15 ? f1-t0-o14 (KLY) | fused/lost 9 ? exclude | Eligibility change: the other member of the same close baseline pair; reasonable isolation exclusion. |
| f0-t0-o32 ? f0-t0-o31 (TBL5) | fused/lost 34 ? exclude | Eligibility change: baseline object overlaps/abuts ring-like structure; reasonable not_isolated exclusion. |
| f0-t0-o1 ? f0-t0-o2 (TBL5) | fused/lost 33 ? exclude | Eligibility change: neighbouring ring-like baseline objects, not an independently recovered divider. |
| f4-t0-o25 ? f4-t0-o22 (TBL8) | fused/lost 70 ? no_division 155 | Uncertain, plausible continuation: same-position rod through 70 and sampled late frames 85–155, but phase contrast/mask changes in dense growth; sampled sheets do not establish full-window identity or no division. |

Verdict counts: **2 supported division events (one production-excluded),
3 rejected division calls, 2 uncertain continuations, 5 eligibility changes**.
This deliberately contact-enriched sample is not a precision estimate.

### Validation and remaining limits

`python -m pytest microscopy/tests plate_reader/tests`: 166 passed, 1 skipped
before the new tests; final `python -m pytest microscopy/tests`: 63 passed,
1 skipped, including nine new drift/contact/division/gap/ambiguity/timing
and drift-out-of-frame cases. The pipeline integration test also verifies
bridging a usable but entirely empty label frame. Plate-reader code was unchanged. Existing
skimage NumPy-deprecation warnings remain.

Local ownership can still switch identities through halos, fragmentation or
merged cells; a strong overlap winner is not a calibrated confidence estimate.
Static ring-like objects can generate false division edges. Growth/shape and
image-intensity evidence, global assignment and multi-frame segmentation repair
remain needed before promotion. One gap does not recover a division hidden
inside it. Confirmation timing depends on segmentation and frame cadence.
The all-cell graph is large: the pilot source run is 2.10 GB on disk, and full
run assembly/review requires much more memory than a single tracker pair.
Preserve source catalogs and masks when making first-event review projections.

All configs, scores, matching records, contact sheets, benchmarks and replay
scripts are retained under `E:/microscopy/work/cell_lineage/`; acquisition and
existing production/review projects were read-only. This implementation does
not satisfy the requested production accuracy threshold.

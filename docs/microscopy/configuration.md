# Configuration

Start with `lab-microscopy inspect RAW --config-template CONFIG.json`; do not copy assumptions from a different experiment silently. The configuration is saved into `project.json` on import, and into each algorithm run/results snapshot.

## Acquisition mapping

| Key | Meaning / initial default |
|---|---|
| `file_pattern` | Full-match regex of each relative image path, with named `field`, `frame`, `channel` captures. Template assumes `Pos1/phase/img_000.tif`; edit to match actual filenames. |
| `structure_channel` | Required; phase/brightfield or another channel suitable for segmentation. No intensity-based guessing. |
| `fluorescence_channels` | Explicit channel names, default empty. |
| `frame_interval_s` | Default null; when supplied, elapsed time is frame index times interval. Use metadata CSV for nonzero/irregular acquisition timestamps. |
| `exposure_ms` | Channel-to-exposure mapping; default empty/unknown. |
| `metadata_csv` | Optional CSV, relative to the initial configuration file or absolute. Populates the immutable manifest; changing this file later does not change imported times. Blank fields fall back to the declared interval/channel exposure. |
| `conditions` | Field string → condition label, e.g. `{"1":"A","2":"B"}`. Default `Unassigned`. Never baked into founder IDs. |
| `biological_replicates` | Field string → independently assigned replicate ID. No implicit inference from field number. |
| `pixel_size_um` | Optional declaration, currently stored for provenance; exported geometry remains pixels/pixels². |
| `expected_frame_step` | Default 1. Automatic linking requires this index increment; timestamps determine elapsed duration. |

Use a fresh project to change acquisition semantics (channels/time/manifest). Changing condition groups, measurements or planned comparisons in `project.json` is supported; rerun `analyze` without replacing the tracked run. Change segmentation/tracking parameters and create a new `run` to evaluate a new method. Existing run parameters remain frozen.

## Algorithms and QC

`segmentation`: `adapter="threshold"`, `polarity="dark"`, `background_sigma_px=8`, `threshold=null`, `min_area_px=12`, `max_area_px=5000`. Local contrast is Gaussian-background minus image for dark objects, reversed for bright objects. The automatic threshold is `max(median + 3 × 1.4826 × MAD, 0.03 × image range)`. Components use SciPy's default four-connectivity and fill enclosed holes. These are uncalibrated starting values in pixels, not bacterial biology standards.

`tracking`: `adapter="overlap"` (alternative `"centroid"`), `max_displacement_px=15`, `division_area_ratio=[0.65,1.5]`, `min_child_fraction=0.2`, `confirmation_frames=1`. Each child must have one unambiguous possible parent; overlap mode also requires mask intersection. A parent may continue as one child or split into two satisfying area constraints. Each daughter must be between 20% and 80% of parent area by default. Confirmation requires each daughter to continue as one child for the configured number of subsequent links. Thus the last acquired frame alone cannot confirm a new division by default. There is no motion/drift correction, learned shape model, merge recovery, or gap bridging in the reference tracker.

`quality.min_focus_score=null`: focus exclusion is off until calibrated. Score is variance of the Laplacian after per-image 1st–99th percentile scaling. Contrast and integer saturation are also recorded for structure images; they do not automatically exclude images. Manual quality flags are per field/frame/channel. Bad structure frames are retained in acquisition metadata and terminate automatic continuity.

`founders`: `edge_margin_px=3`, `isolation_px=0`, `allow_delayed_start=false`. Candidates are components in the first acquired structure frame; edge candidates remain visible but excluded. Isolation filtering is off by default because a universal separation threshold would be unjustified. If enabled, components with a neighbor within the dilation radius are excluded. Connected components can contain two cells; use the split editor. Delayed start uses the first usable structure image, labels the founder as delayed, and cannot establish no division over the whole recording.

## Fluorescence recipes

```json
{
  "measurements": [
    {"name":"baseline_PI","channel":"PI","selection":"baseline"},
    {"name":"PI_at_30min","channel":"PI","selection":"event_relative",
     "offset_s":1800,"tolerance_s":120},
    {"name":"GFP_series","channel":"GFP","selection":"all"}
  ],
  "analysis": {
    "comparisons":[["A","B"],["C","D"]],
    "unit":"biological_replicate",
    "allow_field_inference":false,
    "correlations":[["baseline_PI","first_observed_division_h"]]
  }
}
```

Measurement names must be unique. `selection` is `baseline`, `all`, `frame` (requires `frame`), or `event_relative` (relative to first-division interval midpoint; requires `tolerance_s`; `offset_s` defaults zero). Event-relative timing uses the nearest available channel timestamp with a tracked mask, within tolerance. Equidistant images are both exported, and comparisons use the declared median reduction. A per-founder/channel manual frame override supersedes recipe selection for every recipe in that channel; it is recorded explicitly.

Default measurement is mean raw mask intensity minus median local background. The background ring has inner/outer dilation radii 3/10 pixels, excluding all detected objects, with at least 30 pixels required. Configure `background_inner_px`, `background_outer_px`, `min_background_pixels`. Contaminants missed by segmentation can still enter the ring. No flat-field correction, bleaching model or spectral unmixing is assumed.

`channel_shift_yx_px=[0,0]` optionally supplies an integer translation of structure masks into fluorescence coordinates. Image sizes must already agree. This is a declared transform, not an estimated registration. A shifted mask falling outside the image is missing.

Exposure normalization is off. Enabling `normalize_exposure` requires `linear_response_verified=true`, with `reference_exposure_ms=100` by default. It multiplies net mean by reference/exposure. Raw measurements remain available. This corrects exposure scaling only, not differences in staining, gain, illumination or detector response. Saturation fraction is reported; saturated measurements are not automatically filtered in this release.

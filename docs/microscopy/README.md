# Founder microscopy framework

A local Python package and browser review application for 2D time-lapse microscopy. It starts from explicitly mapped TIFF images, isolates candidate founders, tracks their descendants, distinguishes first observed division from tracking loss, measures fluorescence, and rebuilds plots from versioned human reviews.

This is a usable first release with synthetic validation, **not a validated universal bacterial tracker**. Start each experiment with a representative pilot. The threshold segmenter and conservative overlap/centroid trackers are reference implementations intended to be replaced when better methods are appropriate. All scientific and implementation assumptions, including defaults, are listed by importance in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Install and try it

From the repository root, in your preferred Python environment:

```sh
python -m pip install -e '.[test]'
lab-microscopy demo /tmp/founder-demo
lab-microscopy analyze /tmp/founder-demo/project
lab-microscopy serve /tmp/founder-demo/project
```

Open the printed `http://127.0.0.1:8766` URL. The demo generates images outside the repository. It contains synthetic division, loss and persistent nondividing cells, with different fluorescence backgrounds in two conditions. Ground truth is saved next to the demo configuration. It demonstrates the workflow; it is not evidence of accuracy on real acquisitions.

The package can also run as `python -m microscopy`. It does not install or change the existing plate-reader pipeline.

## Analyze a folder

```sh
lab-microscopy inspect /data/acquisition --config-template /data/config.json
# Edit config.json after confirming channels, filenames and acquisition metadata.
lab-microscopy init /data/acquisition --config /data/config.json --project /data/review-project
lab-microscopy run /data/review-project --fields 1 2
lab-microscopy serve /data/review-project
```

Outputs must be outside the raw acquisition directory. Initialization refuses a nonempty destination. Files are indexed by named `field`, `frame`, and `channel` regex groups. Every matched file must be a single 2D TIFF image. Stacks, vendor formats and volumes require a separate reader/export; unsupported dimensions fail explicitly. Inspect only samples the first 20 TIFF headers; initialization validates every matched image.

Configuration is a JSON file. See the template from `inspect` and [configuration.md](configuration.md). Unknown times/exposures remain missing. If a metadata table is available, provide `field,frame,channel,time_s,exposure_ms` columns and its path in `metadata_csv`. Times must be elapsed seconds relative to the declared experimental reference, not file modification times. A constant interval is an explicit alternative, with time calculated as `frame * frame_interval_s`.

A pilot run can restrict fields. A subsequent full `run` creates a new immutable run and starts a separate review history. The pilot's reviews are preserved but **are not automatically transferred** to the new segmentation. Open a particular run with `serve --run-id RUN_ID`. Keep a reviewed run if only changing condition groups or requested analyses; those changes do not need new segmentation.

## Review workflow

The founder list can be filtered by outcome or unreviewed status. Select a founder to see phase, an optional red fluorescence overlay, outlines, a timeline, and seven frames around the candidate event. Jump to the initial frame, event, or loss; play the recording; use arrow keys to step. Full-field view provides context.

Green outlines identify the currently linked lineage. Gray outlines are other detections. Orange indicates the selected observation. Display uses the first image of each field/channel to set a fixed intensity range for that field. It does **not** normalize measurements. Set the same raw display limits across fields when comparing brightness visually. Log display is a viewing transformation only.

Decisions are **division**, **tracking lost**, **no division observed through recording**, and **exclude**. Division/loss decisions record the last certain frame and first later event frame, preserving an interval. No division requires explicit verification of identity throughout the recording. It does not mean death or permanent inability to grow. Notes are saved with the decision.

Corrections under the expandable editor:

- **Replace an outline:** select a detection, click polygon vertices, finish, then save. Numerical measurements are recalculated from raw pixels inside the corrected mask.
- **Split an initial object:** draw exactly two disjoint founder polygons. The original becomes excluded; the children receive new identities and do not inherit the ambiguous track. Link each child to its appropriate subsequent detection or review its event from the movie.
- **Recover a missing detection:** draw a new observation at the desired frame, save, and add the appropriate links.
- **Repair links:** select a parent, retain it while changing frame, select/add one or two children, then save. Saving zero children cuts outgoing links. Merges and links backward in time are rejected. An incoming link must be removed before reassigning a child.
- **Exclude a poor image:** mark its channel/frame unusable. Structure-channel exclusions break automatic continuity; fluorescence-only exclusions make those measurements missing. Marking an image usable does not generate a missing segmentation: add the outline or create a new run.
- **Choose a fluorescence frame:** select the channel and frame, then save the override for that founder/channel. A tracked mask must exist at that frame; missing masks produce an explicit missing measurement, not a zero or an unrelated nearby cell's intensity.

Mask, link, split, and structure-quality edits invalidate preceding event confirmations conservatively across the run. Reconfirm affected calls before using reviewed-only analysis. Event revisions themselves remain recoverable in SQLite. The UI saves edits explicitly; unsaved decision fields are drafts in browser local storage, not part of the scientific record. Undo creates another revision and removes the last active operation. A stale browser tab cannot overwrite a newer revision silently.

## Regenerate and export

Use **Results → Regenerate results**, or:

```sh
lab-microscopy analyze /data/review-project --reviewed-only
lab-microscopy export-reviews /data/review-project --output /data/reviews-backup.json
lab-microscopy import-reviews /data/review-project /data/reviews-backup.json --expected-revision 12
```

Review imports require the same acquisition manifest and run. A new result directory captures the active configuration and exact review state. Prior plots are not overwritten. Results include:

- Founder event intervals, outcomes, review status, and observation limits.
- Observation masks/geometry references and individual fluorescence measurements with raw mean/median, local background, corrected mean/median, integrated signal, exposure, selected time, saturation fraction and missingness status.
- One-value-per-founder tables, counts/means/medians, and outcome-stratified histograms using shared bins.
- Condition violins in linear and symmetric-log scales. Negative background-corrected values are retained.
- Requested unpaired comparisons of **unit medians**, using exact rank permutations and Holm correction; no unrequested pairwise tests.
- Optional per-condition and pooled descriptive Spearman correlations/scatter plots. Cell-level correlation p-values are deliberately not inferred without an independent-unit model.

Repeated frames and descendants are currently reduced to one median per founder in condition comparisons; the full observation table is retained. First-division summaries include only founders with a division call and known times. They are conditional descriptive distributions, not a censoring-aware population lag model. Loss/no-division categories remain in the founder table.

Statistical inference defaults off for fields. Provide independent biological replicate IDs, or explicitly justify field exchangeability before enabling field-level tests. The package refuses paired designs and exact tests exceeding 200,000 allocations instead of substituting an undocumented test. See assumptions before interpreting significance.

## Project storage and reproducibility

```
review-project/
  project.json             Current analysis/grouping configuration
  manifest.json            Acquisition paths, times, exposure, shape and SHA-256
  runs/RUN_ID/run.json      Algorithm/configuration provenance and observation graph
  runs/RUN_ID/labels/       Original segmentation labels
  reviews.sqlite           Transactional review operations and revision snapshots
  results/RESULT_ID/        Frozen tables, plots, review/configuration snapshots
  latest_run.json          Convenience pointer; does not delete prior runs
  latest_results.json      Convenience pointer; does not delete prior results
```

Raw TIFFs stay in the acquisition folder and are never rewritten. The reader checks their hashes when opening them (cached for unchanged file metadata). Back up the entire review project and raw acquisition; review JSON alone does not contain images, masks or algorithm runs. Stop the server before copying the SQLite database, or use an SQLite-aware backup tool; copying only an active `.sqlite` file can omit its WAL.

The viewer is a loopback-only application, not an Internet service or multi-user authentication system. It uses local-origin/session checks and revision conflicts. No telemetry, cloud storage, remote JavaScript or model download is included. A cloud deployment would require a different security and storage design.

The old experiment-specific viewer and reviews are untouched. This release does not automatically migrate or reinterpret legacy decisions: different legacy tracking/mask semantics need a tested migration adapter and an audit before transfer. The new folder can be analyzed directly with this package.

## Extending and validating

Read [architecture.md](architecture.md) for replaceable adapter contracts. The canonical graph and review store are independent of the tracker. Read [validation.md](validation.md) for implemented checks and the real-data validation still required. The agent workflow is in [the microscopy-experiment skill](../../skills/microscopy-experiment/SKILL.md).

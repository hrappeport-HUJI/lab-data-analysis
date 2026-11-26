# Working commands and reporting

These commands assume `lab-microscopy` is installed from the repository (`python -m pip install -e '.[test]'`). On the shared Windows PC, use the shared interpreters with your clone on `PYTHONPATH` and replace `lab-microscopy` with `python -m microscopy` using the appropriate interpreter. See [multi-user onboarding](../../../docs/multi_user_onboarding.md). Read `docs/microscopy/README.md`, `configuration.md`, and `architecture.md` from that checkout for exact options and adapter contracts; do not duplicate the numerical methods in the skill.

```sh
lab-microscopy inspect RAW_FOLDER --config-template CONFIG.json
# Run this line for supported Micro-Manager acquisitions.
lab-microscopy mm-metadata RAW_FOLDER --output METADATA.csv
lab-microscopy init RAW_FOLDER --config CONFIG.json --project REVIEW_PROJECT
lab-microscopy run REVIEW_PROJECT --fields FIELD_ID FIELD_ID
lab-microscopy serve REVIEW_PROJECT
lab-microscopy analyze REVIEW_PROJECT --reviewed-only
lab-microscopy export-reviews REVIEW_PROJECT --output NEW_BACKUP.json
```

Omit `--reviewed-only` for an explicitly labeled candidate analysis. Use `--run-id` on serve/analyze/export/import to select a previous run. `run` always creates a new run; pilot reviews remain in their original run. Import review backups only for the same run/manifest and with the explicit current `--expected-revision`. No unqualified overwriting of review files.

## v2.2 founder pilot and review

1. Verify TIFF naming, structure and fluorescence channels, actual frame coverage, and image dimensions. After extracting Micro-Manager metadata, verify the CSV keys against the filename mapping and set `metadata_csv` before `init`. The CSV uses acquisition-wide elapsed seconds. Record any change to the time reference explicitly.
2. Check disk and RAM. Use the GPU interpreter for the fast pilot. For a position with contiguous frames starting at zero and the exact filename/channel mapping documented by the runner, the entry point is:

   ```sh
   python -m microscopy.fast_omnipose --pos-dir POS_DIR --num-frames N --out-json PILOT_JSON --save-labels-dir PILOT_LABELS
   ```

   Check `--help` and confirm the phase, PI and YFP mappings in the checked-out runner before calling it. For other layouts, use `FastOmniposeRunner.run_stack` with explicit readers matching the inspected acquisition. Use a new output directory for each pilot. FP32 inference packs active ROIs into mosaics and stops following a founder once its event resolves. On the shared PC, use the automatic cross-process lock from the integrated checkout (`microscopy.gpu_lock`). Keep the common default or the administrator's shared `LAB_MICROSCOPY_GPU_LOCK` path. Keep `LAB_MICROSCOPY_DISABLE_GPU_LOCK` unset. Report lock waiting as queued and start the progress-based ETA after inference begins.
3. Inspect masks, event intervals, doublets (`is_doublet`), and every exclusion reason represented in the pilot. Verify contact and engulfment calls before interpreting loss. Check PI annotations against raw fluorescence and identity history. Preserve the fate and PI fields independently. Keep `ghost_lysis` disabled unless the experiment has a validated definition and an explicit reason to enable it.
4. For field inspection, use `python -m microscopy.position_viewer build --help` to check the cache inputs supported by your version. Build a cache in your own workspace and serve it with `python -m microscopy.position_viewer --cache-dir CACHE_DIR --host 127.0.0.1 --no-auth --port PORT`. Use no-auth only for a local session with data appropriate for other users on the lab PC to see. Use authenticated access for LAN/Tailscale.
5. For versioned analysis, set `tracking.events` to `"founder_v2"` before creating a project run, and confirm the PI/YFP channel mapping in `tracking.founder_v2`. Set `pi_channel` and `yfp_channel` explicitly to the verified channel names, or use an empty string for an absent channel. For external full-field masks, the `precomputed` segmenter expects `<labels_dir>/Pos<field>/<frame:09d>.npz` with a `labels` array for every required structure frame. Fast-run JSON and active-ROI labels are pilot outputs. Preserve complete segmentation coverage whenever the analysis uses all detections for fluorescence background exclusion. Open the project with `python -m microscopy.workbench REVIEW_PROJECT --run-id RUN_ID --port PORT` and save corrections through the review API. Use `lab-microscopy analyze REVIEW_PROJECT --run-id RUN_ID --reviewed-only` to freeze the reviewed result.
6. Load the specific frozen result directory with `microscopy.founder_plots.load(result_dir)` and call `plot` for the requested figure and known measurement recipes. For example, `plot(data, "initial_PI_vs_lag", show=False, save_to=figure_path)` requires a verified `baseline_PI` recipe and known division times. Use explicit eligibility, missingness and outlier settings, and report the plotted subset. PI-rate figures additionally require `PI_series`, and YFP figures require `baseline_YFP`.

Configuration decisions belong in the experiment project:

- Filename/reader mapping and field/channel inventory; metadata source and time origin.
- Structure and fluorescence channels; exposure, calibration and registration assumptions.
- Founder baseline, isolation/edge policy, segmentation and tracker parameters, focus policy.
- Requested fluorescence times: baseline, explicit frame, all linked observations, or a specified offset/tolerance from a division interval.
- Conditions, biological replicates, planned comparison pairs and measurement reduction rules.

The default package provides outcome counts, first-division intervals/midpoints, fluorescence mean/median/background/integrated signal, stratified histograms, condition violins, planned unit-level comparisons and optional descriptive correlations. A request outside that scope should receive a deliberate new analysis module with its assumptions and validation, not a misleading substitute from the existing menu.

A useful report explains what was detected/tracked, what was manually reviewed, what is excluded or missing and why, what the measurement means, and how conclusions depend on those choices. Link the frozen result directory and review revision. Mention insufficient independent units, unsupported designs or unvalidated model performance where they affect interpretation. For a new algorithm, compare a representative adjudicated set before treating apparent improvements in cell retention as improved tracking accuracy.

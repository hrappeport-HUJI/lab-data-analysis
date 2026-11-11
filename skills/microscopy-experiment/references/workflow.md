# Working commands and reporting

These commands assume `lab-microscopy` is installed from the repository (`python -m pip install -e '.[test]'`). Read `docs/microscopy/README.md`, `configuration.md`, and `architecture.md` from that checkout for exact options and adapter contracts; do not duplicate the numerical methods in the skill.

```sh
lab-microscopy inspect RAW_FOLDER --config-template CONFIG.json
lab-microscopy init RAW_FOLDER --config CONFIG.json --project REVIEW_PROJECT
lab-microscopy run REVIEW_PROJECT --fields FIELD_ID FIELD_ID
lab-microscopy serve REVIEW_PROJECT
lab-microscopy analyze REVIEW_PROJECT --reviewed-only
lab-microscopy export-reviews REVIEW_PROJECT --output NEW_BACKUP.json
```

Omit `--reviewed-only` for an explicitly labeled candidate analysis. Use `--run-id` on serve/analyze/export/import to select a previous run. `run` always creates a new run; pilot reviews remain in their original run. Import review backups only for the same run/manifest and with the explicit current `--expected-revision`. No unqualified overwriting of review files.

Configuration decisions belong in the experiment project:

- Filename/reader mapping and field/channel inventory; metadata source and time origin.
- Structure and fluorescence channels; exposure, calibration and registration assumptions.
- Founder baseline, isolation/edge policy, segmentation and tracker parameters, focus policy.
- Requested fluorescence times: baseline, explicit frame, all linked observations, or a specified offset/tolerance from a division interval.
- Conditions, biological replicates, planned comparison pairs and measurement reduction rules.

The default package provides outcome counts, first-division intervals/midpoints, fluorescence mean/median/background/integrated signal, stratified histograms, condition violins, planned unit-level comparisons and optional descriptive correlations. A request outside that scope should receive a deliberate new analysis module with its assumptions and validation, not a misleading substitute from the existing menu.

A useful report explains what was detected/tracked, what was manually reviewed, what is excluded or missing and why, what the measurement means, and how conclusions depend on those choices. Link the frozen result directory and review revision. Mention insufficient independent units, unsupported designs or unvalidated model performance where they affect interpretation. For a new algorithm, compare a representative adjudicated set before treating apparent improvements in cell retention as improved tracking accuracy.

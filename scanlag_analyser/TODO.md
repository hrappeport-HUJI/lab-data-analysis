# ScanLag: open work

Collected while analysing the pilot `2026_10_15_MGR_glucose` (November 2026). Most important first.

## Lag and appearance time

- [ ] **Revisit the appearance time definition.** Today `appearance_min` is the first scan in which the
  tracker gives the colony a blob of at least 10 px (`scanlag/pipeline/run.py`, `track_tables`). That is
  a detection time, not a lag: on the pilot, colonies are first seen at about 12 px, 1.5-2 h after the
  start back-extrapolated from their own growth (sqrt(area) line to 0), and the delay depends on the
  growth rate, so appearance times partly measure growth rate (Spearman rho -0.29, p 1e-36, n 1764 on
  1-4.2u). Options to compare on the pilot and on a plate with known lag differences:
  - back-extrapolated lag: the time where the sqrt(area) fit (the radial growth fit) reaches 0, or a
    fixed seed size, as an extra column `lag_extrap_h`
  - time to reach a fixed size (e.g. 20 px, the doubling-time lower bound) minus a growth-rate correction
  - keep detection time but report it with the growth rate, and test conditions on both
- [ ] The appearance-vs-growth control (`qc.py`, `conditions.summary`) is computed on every run; decide
  what rho should trigger a correction rather than a note. On the pilot the pooled per-scanner rho
  (-0.29) is about twice the per-plate values (-0.09 to -0.18): part of it is between plates. Report a
  within-plate pooled rho too (ranks within each plate, or a mixed model) so plate effects don't read as
  a lag-growth coupling.

## Merges

- [ ] **Growth from each colony's free edge**: estimate a merged colony's growth from the part of its
  perimeter that doesn't touch a neighbour, so merged colonies keep growth values.
- [ ] **Arrival-time map**: split merged blobs (and doublets present at detection) from the time each
  pixel was first covered, instead of the nearest-previous-pixels watershed.
- [ ] Doublets present at first detection are flagged (`possible doublet`) but still count as one colony:
  split them, or count them twice in `n_whole_plate`.
- [ ] `irregular split` (solidity < 0.8) misses some bad splits (pilot 1-4.3u plate 5, colony 2: an
  L-shaped share) and flags many correct concave shares in clusters. Compare with a rule on the split
  line's straightness.
- [ ] Colonies squeezed between neighbours have area tracks that jitter by 20-30 % between frames (pilot
  1-4.2u plate 6): smooth the split, or flag them separately from `shrinks`.

## Review and training

- [ ] "Save for training" key in the review window (one colony's tile at the current frame), feeding
  `export_review_tiles.py`'s folder.
- [ ] `train_unet.py`: read review tiles together with the MATLAB-derived training set, then retrain and
  compare on the gold set.

## Housekeeping

- [ ] The colony table's `flags` column clashes with pandas' `DataFrame.flags` (use `df["flags"]`);
  rename it (`review_flags`) together with the review window and the analysis.
- [ ] `scanner.json` records the device and minutes only for scanners segmented after November 2026.

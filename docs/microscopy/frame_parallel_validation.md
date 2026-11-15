# Frame-parallel validation (Windows, 2026-11-15)

Reference: `E:/microscopy/work/pilot3_bench/Pilot_3/runs/run-34bebd975c26`.
Scratch project: `E:/microscopy/work/frame_parallel/Pilot_3`, initialized by copying only the reference project's `project.json` and `manifest.json`. Fields: 2, 11, 18, 22 (163 frames each). All raw inputs and the existing reference run remained read-only; the editable installation and environment were not changed.

## Exact-equivalence gate

Final fresh run: `run-b8e92181168e`, implementation commit `ed516a684142f0714a0e70b7a7e8a856d71def09`.

`microscopy/tools/compare_runs.py` found **zero differences**: 652 label planes (array names, shapes, dtypes and every pixel/label ID), effective field thresholds, QC, 7,817 observations, 5,399 edges, 88 founders and all 88 founder events, including decisions, intervals, flags and observation IDs. It also compares semantic configuration, dependency versions and adapter classes. Only execution provenance and scheduling settings are excluded.

Reproduce with the same Python/environment:

```powershell
python microscopy/tools/compare_runs.py E:/microscopy/work/pilot3_bench/Pilot_3/runs/run-34bebd975c26 E:/microscopy/work/frame_parallel/Pilot_3/runs/run-b8e92181168e
python E:/microscopy/work/frame_parallel/score_against_tags.py run-b8e92181168e
```

The copied/adapted scorer uses `E:/microscopy/work/Pilot_3_ground_truth.csv`. Final score: agreement **73/87**, divisions **42/44**, timing exact/within 1/within 2 frames **21/35/42**, not grown **6/6**. Full logs: `compare_final.log` and `score_final.json` in the scratch root.

An earlier fresh prototype run, `run-ac4972e65d00`, also passed the identical comparison and scores. Independently recalculating baseline PI and PI series for that run and the reference produced **7,388 exactly identical measurement rows** (`compare_pi.py`, `pi_compare.log`, `pi_compare.json` in the scratch root). PI was not recalculated a second time for the final run: its complete canonical measurement inputs are identical to the reference, and measurement code is unchanged. YFP measurements were not independently recalculated.

## Timing and resource context

| Run | Frame processes | Concurrent replay fields | Preparation (s) | Replay (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | field-based, 4 field processes | 4 | not recorded | not recorded | 945 |
| Initial equivalent prototype | 8 | 4 | 454.5 | 379.6 | 837.8 |
| Final committed implementation | 8 | 4 | 471.0 | 392.3 | 866.9 |

Final wall time is **8.3% lower** than the supplied reference. The prototype preceded worker-startup hardening and the Windows atomic-write retry, so these are not controlled identical-code repetitions. On the shared Windows 10 i9-9900X (10 cores/20 threads, 128 GB RAM), other acquisition import activity and local tests overlapped these runs. Thirty-second system CPU samples for the final run averaged 38.2% (23.3–59.8%, 25 samples; a watcher restart creates a sampling gap). No cold-cache experiment or controlled repeat variability study was performed. Timing is indicative; equivalence is the gate.

Preflight: about 141 GiB free disk and 105 GiB available RAM. Final run occupies 1.188 GiB including labels, sidecars and JSON. A replay sample of coordinator plus eight workers used 2.02 GiB working set; the sum of their individual peak working sets was 2.85 GiB at that sample. These are sampled figures, not continuous peak-memory profiling. Resource logs and both successful benchmark summaries are retained in the scratch root.

One intermediate attempt (`run-c338f77cd7d4`) failed on Windows denying atomic replacement of a progress JSON during a concurrent read. `storage.write_json` now retries that transient sharing failure with bounded backoff, retaining atomic publication and propagating persistent failures. The added regression test checks success, persistent failure and temporary-file cleanup. A later watcher-only read sharing error was handled by reattaching a read-tolerant watcher without restarting the final analysis. Both failure and watcher logs are retained.

## Repository checks

- `python -m pytest microscopy/tests -vv --basetemp=E:/microscopy/work/frame_parallel/pytest_atomic`: **50 passed, 1 skipped**, two upstream scikit-image/NumPy deprecation warnings. The skip is Windows symlink privilege support.
- Scheduling coverage includes reverse completion order, bounded outstanding work, round-robin dispatch, worker counts, resume with missing labels and stale sidecars, changed field indices, exclusions and threshold precedence. Existing tests cover stage jitter, static contacts, fusion and Windows unguarded launch failure.
- `python -m pytest plate_reader/tests -q`: cannot collect three modules because the supplied environment lacks `openpyxl` and `plotly`. The environment was deliberately left unchanged.

No founder-only ROI segmentation, numerical segmentation changes, centroid optimization, fluorescence algorithm changes or review transfer was introduced.

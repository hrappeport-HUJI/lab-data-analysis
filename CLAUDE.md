# Agent notes

This repo holds code for repeating lab analyses, and no data. `README.md` describes it.

If `~/Research/Zettelkasten/Library/Skills/` exists on this machine, follow `reader-experiment/SKILL.md` there for plate-reader runs and `scanlag-experiment/SKILL.md` for ScanLag runs. Otherwise use the repository documentation and ask for any missing experiment-specific instructions.

Run `python -m pytest plate_reader/tests` (and `cd scanlag_analyser && python -m pytest tests` for ScanLag) after changing code. Commit only when asked.

## Microscopy

For microscopy use the repo-contained `skills/microscopy-experiment/SKILL.md` and `docs/microscopy/`. For v2.2, use `lab-microscopy mm-metadata` for Micro-Manager metadata, pilot `founder_v2` and `microscopy.fast_omnipose`, inspect fields with `python -m microscopy.position_viewer`, and generate requested figures from a frozen result with `microscopy.founder_plots`. Keep acquisition data, generated projects and review databases outside this code repository. Preserve original algorithm runs and save corrections through the review API. Run `python -m pytest microscopy/tests plate_reader/tests` after microscopy code changes. See `docs/multi_user_onboarding.md` for shared-PC setup.

## Shared Multi-User Lab PC

- Use your own clone and write outputs, caches and logs inside your assigned user workspace. Treat raw acquisitions as read-only.
- Reuse `E:\microscopy\env-gpu`, `E:\microscopy\env` and `E:\microscopy\models` read-only. Put your clone on `PYTHONPATH` and use the shared interpreters directly. Use the preinstalled PyTorch and CUDA stack.
- Leave other users' processes, viewers and files alone. Bind viewers to loopback by default and use your assigned viewer ports or `--port 0` where supported.
- Let the automatic GPU lock queue jobs on the shared GPU. Keep the lock enabled and use the same lock path across accounts. Check free disk and available RAM before large jobs.

## Long-running jobs (any agent)

A background job is not "running" until you have seen it make progress. After launching anything that takes more than a few minutes:

1. Within 1–2 minutes, confirm progress yourself: output files growing, or `runs/<id>/progress/*.json` advancing for microscopy runs. Check that `runs/<id>/failed.json` does not exist. Only then tell the user it is running, with an ETA computed from the measured rate.
2. Arm a watcher that reports every terminal state: done, failed, process exited without output, and stalled (no progress for a few minutes). Silence must never mean "still fine".
3. Scripts that call `microscopy.pipeline.run` with `field_workers > 1` must put the call under `if __name__ == "__main__":`. Worker processes re-import the script.
4. Check memory and disk before heavy jobs, and run them at lower priority. Never auto-chain a heavy step without checking what the previous step produced. Microscopy runs now refuse to start or continue below 3 GB of free disk, and refuse to load a run file too large for this computer's RAM.
5. Keep project copies and runs outside temporary or session scratch folders, which can be wiped.

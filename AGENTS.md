# Agent notes

This repo holds code for repeating lab analyses, and no data. `README.md` describes it.

The instructions for running an analysis are skills in the Zettelkasten, at `~/Documents/Zettelkasten/Library/Skills/`. For a plate-reader run, read `Library/Skills/reader-experiment/SKILL.md` there and follow it. For a ScanLag run, read `Library/Skills/scanlag-experiment/SKILL.md`.

Run `python -m pytest plate_reader/tests` (and `cd scanlag_analyser && python -m pytest tests` for ScanLag) after changing code. Commit only when asked.

## Microscopy

For microscopy use the repo-contained `skills/microscopy-experiment/SKILL.md` and `docs/microscopy/`. This is an explicit exception to the older external-skills convention. Keep acquisition data, generated projects and review databases outside this code repository. Preserve original algorithm runs and save corrections through the review API. Run `python -m pytest microscopy/tests plate_reader/tests` after microscopy code changes.

## Long-running jobs (any agent)

A background job is not "running" until you have seen it make progress. After launching anything that takes more than a few minutes:

1. Within 1–2 minutes, confirm progress yourself: output files growing, or `runs/<id>/progress/*.json` advancing for microscopy runs. Check that `runs/<id>/failed.json` does not exist. Only then tell the user it is running, with an ETA computed from the measured rate.
2. Arm a watcher that reports every terminal state: done, failed, process exited without output, and stalled (no progress for a few minutes). Silence must never mean "still fine".
3. Scripts that call `microscopy.pipeline.run` with `field_workers > 1` must put the call under `if __name__ == "__main__":`. Worker processes re-import the script.
4. Check memory and disk before heavy jobs, and run them at lower priority. Never auto-chain a heavy step without checking what the previous step produced. Microscopy runs now refuse to start or continue below 3 GB of free disk, and refuse to load a run file too large for this computer's RAM.
5. Keep project copies and runs outside temporary or session scratch folders, which can be wiped.

# Agent notes

This repo holds code for repeating lab analyses, and no data. `README.md` describes it.

The instructions for running an analysis are skills in the Zettelkasten, at `~/Documents/Zettelkasten/Library/Skills/`. For a plate-reader run, read `Library/Skills/reader-experiment/SKILL.md` there and follow it.

Run `python -m pytest plate_reader/tests` after changing code. Commit only when asked.

## Microscopy

For microscopy use the repo-contained `skills/microscopy-experiment/SKILL.md` and `docs/microscopy/`. This is an explicit exception to the older external-skills convention. Keep acquisition data, generated projects and review databases outside this code repository. Preserve original algorithm runs and save corrections through the review API. Run `python -m pytest microscopy/tests plate_reader/tests` after microscopy code changes.

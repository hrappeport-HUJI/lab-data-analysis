# Agent notes

This repo holds code for repeating lab analyses, and no data. `README.md` describes it.

The instructions for running an analysis are skills in the Zettelkasten, at `~/Research/Zettelkasten/Library/Skills/`. For a plate-reader run, read `Library/Skills/reader-experiment/SKILL.md` there and follow it. For a ScanLag run, read `Library/Skills/scanlag-experiment/SKILL.md`: it points to the general skill `scanlag_analyser/skills/scanlag-analysis/SKILL.md` in this repo and adds the local paths. The general skill is public and must stay free of local paths, names and record-keeping rules; put those in the Zettelkasten skill. When a ScanLag script's options or outputs change, update the general skill in the same commit.

Run `python -m pytest plate_reader/tests` (and `cd scanlag_analyser && python -m pytest tests` for ScanLag) after changing code. Commit only when asked.

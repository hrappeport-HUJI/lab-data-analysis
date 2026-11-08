# Lab data analysis

Code for analyses that repeat across experiments. It holds code only.

| What | Where |
|---|---|
| Code | This repo, cloned to `~/PycharmProjects/lab-data-analysis` |
| Data: raw files, plate maps, notebooks and results of each experiment | `~/PycharmProjects/ScriptHub/ReaderPlot/YYYY_MM_DD_name/`, backed up on Google Drive |
| How to run an analysis, and what to ask first | Skills in the Zettelkasten: `Library/Skills/` |
| What an experiment was and what it showed | Experiment notes in the Zettelkasten: `Library/LabNotebook/Experiments/` |

`.gitignore` keeps common data formats out of this repo. Don't add data, results or figures here.

## Plate reader

```
pip install -r requirements.txt
```

`plate_reader/` analyzes runs from the Tecan (Monochromator, Infinite, Plex) and Synergy readers.

| File | What it does |
|---|---|
| `reader_functions.py` | Reads the reader sheets and plate maps, and plots them |
| `reader_background.py` | Subtracts medium background, with contamination checks and QC |
| `readthrough.py` | Stop-codon readthrough from a dual-fluorescent reporter, with sanity tests |
| `scripts/inspect_reader.py` | Reports which reader was used and what it measured |
| `scripts/make_plate_map.py` | Writes a plate map from a layout description |
| `scripts/preview_background.py` | Checks a config and shows what background each well gets |
| `scripts/build_notebook.py` | Writes an experiment's notebook from its config |

To analyze a run, give an agent the `reader-experiment` skill, or follow it yourself. Each notebook prints the commit it ran on, so a result can be traced back to its code.

Tests:

```
python -m pytest plate_reader/tests
```

## ScanLag

`scanlag_analyser/` analyzes ScanLag colony appearance and growth from plate scans. It is a fork of [alongutf/scanlag_analyser](https://github.com/alongutf/scanlag_analyser), brought in as a git subtree with its history. Its own `README.md` covers setup (`environment.yml`) and the pipeline. It includes the trained U-Net weights (`models/unet_scanlag_v1/model.pt`, ~98 MB).

To pull later upstream changes into the folder:

```
git remote add scanlag-upstream https://github.com/alongutf/scanlag_analyser  # once
git subtree pull --prefix=scanlag_analyser scanlag-upstream master
```

Tests:

```
cd scanlag_analyser && python -m pytest tests
```

## Adding another kind of analysis

Each kind of repeating analysis gets a sibling of `plate_reader/`, such as `flow_cytometry/`, and a skill in `Library/Skills/`. The plate-reader pipeline is the pattern to follow:

1. **Inspect** the raw file without assuming anything: instrument, settings, what was measured.
2. **Ask** the experimenter what the file can't say. Don't guess.
3. **Write a config** for the experiment, stored in its data folder. A config lists decisions, so they stay visible.
4. **Check before analyzing**: controls and anything that needs a decision. Stop until it is decided.
5. **Build a notebook** that runs from top to bottom and carries its own sanity tests.
6. **Offer a note** for the lab notebook.

Move code into a shared folder only when a second analysis actually needs it.

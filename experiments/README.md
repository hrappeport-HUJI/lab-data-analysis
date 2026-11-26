# experiments

Tools for keeping `~/Research/Experiments` (one folder per experiment, named `YYYY-MM-DD <description>`) in order. The rules they follow are in `~/Research/ORGANIZATION.md`; the step-by-step procedure for the deep archive is the Zettelkasten skill `deep-archive`.

| Script | What it does |
|---|---|
| `build_experiments_index.py` | Writes `Experiments/index.csv` and `Experiments/INDEX.md`: per experiment, reader runs and their start times, plate-map labels, TIFF and notebook counts, size on the Mac, Zettelkasten note, deep-archive status, old locations (from `~/Research/_reorg/manifest.tsv`) and hand-written notes (`NOTES` in the script). Run with the `research` env after adding, renaming or archiving experiments. |
| `deep_archive.sh "<experiment>" <subfolder>` | Moves the heavy raw part of a finished experiment to the deep archive on Google Drive (`My Drive/Research deep archive/Experiments/<experiment>/<subfolder>/`, account hrappeport@gmail.com): writes a SHA-256 list into the experiment folder, copies, checks every file arrived with its size, appends to `DEEP_ARCHIVE.md`, and prints the command that removes the local copy. It never deletes, and refuses when the disk has less free space than the subfolder needs (Drive keeps a local copy until the upload finishes). |

Both use only the standard library and macOS tools, except that the index builder needs `openpyxl`.

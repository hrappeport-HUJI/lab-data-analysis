#!/usr/bin/env python3
"""Build ~/Research/Experiments/index.csv and INDEX.md: one row per experiment folder, with the metadata that
can be read from the files (reader runs and start times, strains in plate maps, image counts, notebooks),
the folder's old locations (from ~/Research/_reorg/manifest.tsv) and its Zettelkasten note.

Run with the research env:  conda run -n research python ~/Research/Code/lab-data-analysis/experiments/build_experiments_index.py
Re-run it after adding or renaming experiment folders. NOTES below holds hand-written remarks."""
import csv, datetime, os, re, warnings
from collections import Counter

warnings.filterwarnings("ignore")
import openpyxl

H = os.path.expanduser("~")
RD = f"{H}/Research/Experiments"
OLD_RD = f"{H}/Research/Raw_data"  # the folder's name until 2026-11-16; older manifest rows point there
ZK = f"{H}/Research/Zettelkasten/Library/LabNotebook/Experiments"
MANIFEST = f"{H}/Research/_reorg/manifest.tsv"
SKIP_DIRS = ("scanlag_results", "microscopy_tagging_package", ".zarr", "__pycache__", ".ipynb_checkpoints", "training_tiles")
WELL = re.compile(r"^[A-P]\d{1,2}$")
NOTES = {  # date decisions and splits made during the 2026-11-15 reorganization
    "2023-08-28 EPEC LB0 TBL second run": "Reader sheets 08-29..09-01; file inside is named 2023_08_23_EPEC_LB0_b.",
    "2023-09-19 TBL PAE": "Contains a copy of Adi's 2023-01-02 example file; it does not date the experiment.",
    "2024-07-14 Shani pilot": "Split from 2024_07_06_shani; holds the shared notebook (last set to the pilot).",
    "2024-11-07 KLY TBL EPEC regrowth and PI short vs long Tw": "Includes loose 2024_11_05.xlsx (sheets from 11-05): the experiment may have started 11-05 (open question).",
    "2024-12-16 REL strains": "Folder was 2024_12_17; reader started 2024-12-16 19:54.",
    "2025-03-11 Evo 0 2 4 6 lag": "Folder was 2025_03_13; first run 03-11/12.",
    "2025-08-12 EPEC WT Tol regrowth in gentamicin tobramycin streptomycin": "Folder was 2025_08_13; from-frozen step on 08-12.",
    "2025-08-31 EPEC WT Tol regrowth from different stationary conditions and after ampicillin": "Split from 2025_09_01_03; its notebook also reads the 09-08 and 09-17 folders.",
    "2025-09-28 Evo7 0 to 1 and 24 vs 96 well": "Split from 2025_09_28_evo_7; its notebook also reads the 10-04, 10-26 and 11-02 folders.",
    "2025-11-23 KLY TBL8 LB starvation regrowth and CFU": "Folder was 2025_11_25_; reader file 23_11_25 started 2025-11-23. CFU counts from estimate_CFUs merged in.",
    "2025-12-01 TBL8 evolved overnight": "Reader run started 2025-12-02.",
    "2026-01-21 Keio 85 regrowth": "Split from 2026_01_keio; its notebook also reads the 01-26 folder.",
    "2026-02-23 EPEC calibration and SDS": "Folder was 2026_02_24; first sheet 02-23.",
    "2026-05-10 Keio 85 full plate SDS and ampicillin": "Split from 2026_05_keio; cross-experiment notebooks read the 05-24 and 05-27 folders. Top-level xlsx copies duplicate original_full_keio_plate/ (kept: notebooks read both).",
    "2026-07-01 Fluorescein (florecin) reader run": "plate_transcription.xlsx is probably its plate map (unconfirmed).",
    "2026-10-12 MGR post-starvation lag exploration": "Reader file named 2026_10_15 holds Sheet2 (10-12 pilot) and Sheet3 (10-14 run).",
    "2026-10-14 MGR post-starvation lag experiment": "ScanLag scan runs 10-15 and 10-16 (the agar plating of this experiment); reader sheet is in the 10-12 folder.",
    "2026-11-04 PI uptake of KLY and TBL8 by microscopy": "Reader 11-04..11-06; images 11-05 in microscopy/ (98 GB, deep-archive candidate once analysed).",
    "2026-11-09 MGR in LB with and without glucose - SDS lysis, regrowth and ampicillin killing": "Main xlsx = full 9-sheet export; the 2-sheet early export is kept beside it.",
}


def reader_runs(path):
    """Return (kind, [start datetimes]) for a plate-reader export, or (None, [])."""
    try:
        if os.path.getsize(path) > 25_000_000:
            return None, []
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return None, []
    kind, starts = None, []
    for ws in wb.worksheets:
        date = time = None
        for i, row in enumerate(ws.iter_rows(max_row=45, max_col=6, values_only=True)):
            cells = [c for c in row if c is not None]
            if not cells:
                continue
            k = str(cells[0]).strip().rstrip(":")
            v = cells[1] if len(cells) > 1 else None
            if k == "Date" and v is not None:
                date, kind = v, kind or "Tecan"
            elif k == "Time" and v is not None and date is not None and time is None:
                time = v
            elif k in ("Start Time", "Read Start Time") or (isinstance(cells[0], str) and "Plate 1 - Sheet" in cells[0]):
                kind = kind or "BioTek"
        if date is not None:
            try:
                d = date if isinstance(date, (datetime.date, datetime.datetime)) else datetime.datetime.strptime(str(date)[:10], "%Y-%m-%d")
                d = d.date() if isinstance(d, datetime.datetime) else d
                t = ""
                if hasattr(time, "strftime"):
                    t = time.strftime("%H:%M")
                elif time:
                    for fmt in ("%I:%M:%S %p", "%H:%M:%S", "%I:%M %p", "%H:%M"):
                        try:
                            t = datetime.datetime.strptime(str(time).strip(), fmt).strftime("%H:%M")
                            break
                        except ValueError:
                            pass
                starts.append(f"{d.isoformat()} {t}".strip())
            except Exception:
                starts.append(str(date)[:16])
    return kind, sorted(set(starts))


def plate_strains(path):
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return Counter()
    c = Counter()
    for ws in wb.worksheets:
        for row in ws.iter_rows(max_row=60, max_col=30, values_only=True):
            for v in row:
                if isinstance(v, str):
                    s = v.strip()
                    if s and len(s) < 30 and not WELL.match(s) and s.lower() not in ("well", "strain", "rep", "empty", "nan", "none", "medium", "blank", "sheet1"):
                        c[s] += 1
    return c


def old_paths():
    m = {}
    if not os.path.exists(MANIFEST):
        return m
    for l in list(open(MANIFEST, encoding="utf-8"))[1:]:
        _, _, old, new = l.rstrip("\n").split("\t")
        if new.startswith(OLD_RD + "/"):
            new = RD + new[len(OLD_RD):]
        if new.startswith(RD + "/"):
            top = new[len(RD) + 1:].split("/")[0]
            if not re.search(r"/(Raw_data|Experiments)/\d{4}-\d\d-\d\d ", old):
                m.setdefault(top, []).append(old.replace(H, "~"))
    return m


def main():
    olds = old_paths()
    rows = []
    for name in sorted(os.listdir(RD)):
        p = f"{RD}/{name}"
        if not os.path.isdir(p) or name.startswith((".", "_")):
            continue
        n_files = size = tifs = imgs = nbs = 0
        readers, runs, strains = Counter(), [], Counter()
        for d, dirs, fs in os.walk(p):
            dirs[:] = [x for x in dirs if not any(s in x for s in SKIP_DIRS) or x == "microscopy"]
            deep = any(s in d for s in ("/microscopy", "/2026_10_1"))  # big image trees: count only
            for f in fs:
                fp = os.path.join(d, f)
                if f.startswith((".", "~$")):
                    continue
                n_files += 1
                try:
                    size += os.path.getsize(fp)
                except OSError:
                    pass
                fl = f.lower()
                if fl.endswith((".tif", ".tiff")):
                    tifs += 1
                elif fl.endswith((".jpg", ".jpeg", ".png")):
                    imgs += 1
                elif fl.endswith(".ipynb"):
                    nbs += 1
                elif fl.endswith(".xlsx") and not deep:
                    if "plate_map" in fl or "plate map" in fl:
                        strains += plate_strains(fp)
                    else:
                        kind, st = reader_runs(fp)
                        if kind:
                            readers[kind] += 1
                            runs += st
        zk = f"{ZK}/{name}.md"
        rows.append({
            "folder": name,
            "date": name[:10],
            "description": name[11:],
            "reader_files": "; ".join(f"{k} x{v}" for k, v in readers.items()),
            "reader_run_starts": "; ".join(sorted(set(runs))),
            "plate_map_labels": "; ".join(s for s, _ in strains.most_common(15)),
            "tiff_files": tifs,
            "jpg_png_files": imgs,
            "notebooks": nbs,
            "files": n_files,
            "size_GB": round(size / 1e9, 3),
            "zettelkasten_note": f"[[{name}]]" if os.path.exists(zk) else "",
            "deep_archived": "yes" if os.path.exists(f"{p}/DEEP_ARCHIVE.md") else "",
            "old_locations": "; ".join(sorted(set(olds.get(name, [])))),
            "notes": NOTES.get(name, ""),
        })
    with open(f"{RD}/index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{RD}/INDEX.md", "w", encoding="utf-8") as f:
        f.write(f"# Experiments index\n\nGenerated {datetime.date.today()} by `~/Research/Code/lab-data-analysis/experiments/build_experiments_index.py` "
                f"(full columns, including old locations, in `index.csv`). {len(rows)} experiments. Rules: `~/Research/ORGANIZATION.md`.\n\n")
        f.write("| Experiment | Reader runs (start) | Plate-map labels | TIFF | Notebooks | GB on Mac | Zettelkasten | Deep archive | Notes |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            cells = [r["folder"], r["reader_run_starts"].replace("; ", "<br>"), r["plate_map_labels"][:120], str(r["tiff_files"] or ""),
                     str(r["notebooks"] or ""), str(r["size_GB"]), "yes" if r["zettelkasten_note"] else "", r["deep_archived"], r["notes"]]
            f.write("| " + " | ".join(c.replace("|", "/") for c in cells) + " |\n")
    print(f"{len(rows)} experiments indexed -> {RD}/index.csv, INDEX.md")


if __name__ == "__main__":
    main()

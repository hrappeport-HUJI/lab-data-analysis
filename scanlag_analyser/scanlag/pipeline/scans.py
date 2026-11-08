"""Find the scans of an experiment and group them by scanner.

Scan robot file names are ``<prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f]``, e.g.
``epec-amp7_1-5.1u_20260224_2025.tiff`` (scanner ``1-5.1u``). The prefix may be empty
(``_1-4.1u_20261015_1307.tiff``) or left out with its underscore. The robot log
(``run_<prefix>*.log``) names the same scanners: "scanner ...NQBSCANNER1-3, named '1-3'".

The first scan of each scanner is the background (taken before the plates are incubated).
Times are minutes since the log's "Starting." line, or since the first scan if there is no log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scanlag.io.raw import parse_timestamp

SCAN_NAME = re.compile(r"^(?:(?P<prefix>.*)_)?(?P<scanner>[^_]+)_(?P<date>\d{8})_(?P<time>\d{4})\.tiff?$", re.I)
LOG_START = re.compile(r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) - Starting\.")
SKIP_DIRS = {"ROI"}


@dataclass
class Scanner:
    name: str  # folder name in the results, unique within the experiment
    label: str  # scanner name from the file names
    directory: Path
    scans: list[Path]
    times: list[datetime]
    start: datetime  # time zero
    prefix: str = ""  # file name prefix ("" if none)
    time_zero: str = "first scan"  # where ``start`` came from: "first scan" or the robot log's name

    @property
    def times_min(self) -> list[float]:
        return [(t - self.start).total_seconds() / 60 for t in self.times]


def log_start(directory: Path) -> tuple[datetime, str] | None:
    """(the "Starting." time of the first robot log in ``directory``, the log's name), or None."""
    for log in sorted(directory.glob("run_*.log")):
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if m := LOG_START.match(line):
                    return datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S"), log.name
    return None


def find_scanners(root: str | Path) -> list[Scanner]:
    """All scanners under ``root`` (searched recursively, skipping MATLAB ROI folders)."""
    root = Path(root)
    groups: dict[tuple[Path, str], list[Path]] = {}
    prefixes: dict[tuple[Path, str], str] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in (".tif", ".tiff") or SKIP_DIRS & set(path.relative_to(root).parts[:-1]):
            continue
        if m := SCAN_NAME.match(path.name):
            groups.setdefault((path.parent, m["scanner"]), []).append(path)
            prefixes.setdefault((path.parent, m["scanner"]), m["prefix"] or "")
    if not groups:
        raise FileNotFoundError(f"no scans named [<prefix>]_<scanner>_<YYYYMMDD>_<HHMM>.tif under {root}")

    labels = [label for _, label in groups]
    scanners = []
    for (directory, label), paths in sorted(groups.items()):
        paths = sorted(paths, key=parse_timestamp)
        times = [parse_timestamp(p) for p in paths]
        name = label if labels.count(label) == 1 else f"{directory.name}_{label}"
        logged = log_start(directory)
        start, source = logged if logged and logged[0] <= times[0] else (times[0], "first scan")
        scanners.append(Scanner(name, label, directory, paths, times, start, prefixes[directory, label], source))
    return scanners


def select_scanners(scanners: list[Scanner], prefixes=(), names=()) -> list[Scanner]:
    """The scanners whose file name prefix starts with one of ``prefixes`` and whose name or label is
    one of ``names`` (case-insensitive). An empty filter keeps everything."""
    prefixes = [p.lower() for p in prefixes or ()]
    names = {n.lower() for n in names or ()}
    chosen = [s for s in scanners
              if (not prefixes or any(s.prefix.lower().startswith(p) for p in prefixes))
              and (not names or s.name.lower() in names or s.label.lower() in names)]
    if not chosen:
        found = ", ".join(f"{s.name} (prefix {s.prefix!r})" for s in scanners)
        raise ValueError(f"no scanner matches the filter; found: {found}")
    return chosen

"""Read-only access to raw scanner images.

Raw files are opened for reading only; nothing in this module can modify them. Scan times are
taken from the ``_YYYYMMDD_HHMM`` stamp in the file name (file modification times change when
data is copied, so they're only used as a fallback).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

TIMESTAMP = re.compile(r"_(\d{8})_(\d{4})(\d{2})?(?=\D*$)")


def parse_timestamp(path: str | Path) -> datetime | None:
    match = TIMESTAMP.search(Path(path).stem)
    if not match:
        return None
    date, hhmm, ss = match.groups()
    return datetime.strptime(date + hhmm + (ss or "00"), "%Y%m%d%H%M%S")


@dataclass(frozen=True)
class Scan:
    path: Path
    time: datetime


class ScanSeries:
    """One scanner's time-lapse, sorted by acquisition time."""

    def __init__(self, scans: list[Scan]):
        self.scans = sorted(scans, key=lambda s: s.time)

    @classmethod
    def from_pattern(cls, directory: str | Path, pattern: str) -> ScanSeries:
        scans = []
        for path in Path(directory).glob(pattern):
            time = parse_timestamp(path) or datetime.fromtimestamp(path.stat().st_mtime)
            scans.append(Scan(path, time))
        if not scans:
            raise FileNotFoundError(f"no scans match {pattern} in {directory}")
        return cls(scans)

    def __len__(self) -> int:
        return len(self.scans)

    def read(self, index: int) -> np.ndarray:
        return read_scan(self.scans[index].path)

    def read_window(self, index: int, rows: slice, cols: slice) -> np.ndarray:
        return read_window(self.scans[index].path, rows, cols)


def read_scan(path: str | Path) -> np.ndarray:
    """Full RGB scan. Uncompressed TIFFs are memory-mapped read-only."""
    try:
        image = tifffile.memmap(path, mode="r")
    except ValueError:  # compressed or non-contiguous: decode into memory
        image = tifffile.imread(path)
    return image[..., :3]


def read_window(path: str | Path, rows: slice, cols: slice) -> np.ndarray:
    """A crop copied out of a scan. Parts outside the scan are zero."""
    image = read_scan(path)
    out = np.zeros((rows.stop - rows.start, cols.stop - cols.start, 3), image.dtype)
    r0, r1 = max(rows.start, 0), min(rows.stop, image.shape[0])
    c0, c1 = max(cols.start, 0), min(cols.stop, image.shape[1])
    if r0 < r1 and c0 < c1:
        out[r0 - rows.start : r1 - rows.start, c0 - cols.start : c1 - cols.start] = image[r0:r1, c0:c1]
    return out

"""Quick density check of every plate, before the time-lapse is processed: finds TNTC plates.

No detector and no colony tracking: on the last scan, each plate is compared with the background
scan, and the changed patches (colonies are brighter than agar) give:

* ``coverage``: the fraction of the working area that changed
* ``median_blob_px``: the median size of the changed patches (a colony or a cluster)
* ``single_colony_px``: the median size of the compact patches (solidity >= 0.9), i.e. of a single
  colony; the median of all patches if fewer than 5 are compact

``est_colonies`` counts every patch as its area over ``single_colony_px`` (rounded, at least 1), so
a cluster of touching colonies counts as several. A lawn or a TNTC plate gives a large coverage of
small patches (colonies stay small when they compete), so it scores high. Counting blobs alone, as
the detector-based check did, fails on such plates: touching colonies form a few huge blobs, and a
lawn reads as one.

Pilot 2026_10_15_MGR_glucose, last scan (52 h), working area 0.8 x plate radius, against the
colonies the pipeline then tracked in the working area:

| plates | coverage | single colony | est_colonies | tracked |
|---|---|---|---|---|
| sparse (1-4.3u) | 0.05-0.18 | 970-1,900 px | 20-79 | 20-83 |
| dense, countable (1-4.2u) | 0.19-0.25 | 310-590 px | 187-360 | 200-399 |
| TNTC (1-4.1u) | 0.31-0.44 | 60-130 px | 1,300-3,800 | (excluded) |

(Before 2026-11, ``est_colonies`` was the covered area over ``median_blob_px``, which undercounted
dense plates by 15-40 %: 160-324 on 1-4.2u.)

``verdict``: "TNTC" above ``tntc`` (default 500), "dense" above ``dense`` (default 200: expect many
merges), else "ok". Thresholds are a suggestion for the experimenter, who decides what to exclude.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from scanlag.io.raw import read_scan
from scanlag.pipeline.plates import Aligner, circle_mask, crop, find_dishes

DIFF_THRESHOLD = 15  # gray levels above the background scan
MIN_PATCH = 10  # px; smaller patches are noise
COMPACT = 0.9  # solidity of a single colony's patch
MIN_COMPACT = 5  # compact patches needed to measure a single colony's size
TNTC = 500  # est_colonies in the working area
DENSE = 200


@dataclass
class DensityConfig:
    n_plates: int = 6
    mask_fraction: float = 0.8
    tntc: float = TNTC
    dense: float = DENSE
    diff_threshold: int = DIFF_THRESHOLD


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY)


def plate_density(background: np.ndarray, image: np.ndarray, mask: np.ndarray,
                  diff_threshold: int = DIFF_THRESHOLD) -> dict:
    """coverage, n_patches, median_blob_px, single_colony_px and est_colonies of one aligned plate crop."""
    diff = np.clip(_gray(image).astype(np.int16) - _gray(background).astype(np.int16), 0, 255).astype(np.uint8)
    changed = (cv2.medianBlur(diff, 3) > diff_threshold) & mask
    patches = [r for r in regionprops(label(changed, connectivity=2)) if r.area >= MIN_PATCH]
    sizes = np.array([r.area for r in patches], float)
    compact = np.array([r.solidity >= COMPACT for r in patches], bool)
    median = float(np.median(sizes)) if len(sizes) else 0.0
    single = float(np.median(sizes[compact])) if compact.sum() >= MIN_COMPACT else median
    return {
        "coverage": float(sizes.sum()) / max(int(mask.sum()), 1),
        "n_patches": int(len(sizes)),
        "median_blob_px": median,
        "single_colony_px": single,
        "est_colonies": float(np.maximum(1, np.round(sizes / single)).sum()) if single else 0.0,
    }


def verdict(est: float, cfg: DensityConfig) -> str:
    return "TNTC" if est > cfg.tntc else "dense" if est > cfg.dense else "ok"


def scanner_density(scanner, cfg: DensityConfig = DensityConfig(), frame: int = -1) -> tuple[pd.DataFrame, np.ndarray]:
    """(one row per plate holding a dish, an overview picture of the scan) for one scanner.

    ``frame``: the scan to measure (default the last). The picture is that scan, downscaled, with
    each plate's number, estimate and verdict written on it, so the experimenter can check the call
    and see the plate's own label.
    """
    background = read_scan(scanner.scans[0])
    scan = np.array(read_scan(scanner.scans[frame]))
    plates, empty = find_dishes(background, cfg.n_plates)
    rows = []
    for p in plates:
        x0, y0, s = p.crop
        dx, dy = Aligner(background, p, mask_fraction=cfg.mask_fraction).shift(scan)
        m = plate_density(crop(background, x0, y0, s), crop(scan, x0 + dx, y0 + dy, s),
                          circle_mask(s, p, cfg.mask_fraction), cfg.diff_threshold)
        rows.append({"scanner": scanner.name, "plate": p.number, **m, "verdict": verdict(m["est_colonies"], cfg),
                     "cx": p.cx, "cy": p.cy, "r": p.r})
    table = pd.DataFrame(rows, columns=["scanner", "plate", "coverage", "n_patches", "median_blob_px",
                                        "single_colony_px", "est_colonies", "verdict", "cx", "cy", "r"])
    table.attrs["empty_slots"] = empty
    table.attrs["scan"] = scanner.scans[frame].name
    return table, overview(scan, table, empty)


COLOURS = {"ok": (60, 220, 60), "dense": (255, 190, 0), "TNTC": (255, 60, 60)}


def overview(scan: np.ndarray, table: pd.DataFrame, empty=(), width: int = 1000) -> np.ndarray:
    """The scan at ``width`` px wide (RGB), each plate circled in its verdict's colour and labelled."""
    scale = width / scan.shape[1]
    small = cv2.resize(np.ascontiguousarray(scan[..., :3]), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in table.itertuples():
        c = (int(row.cx * scale), int(row.cy * scale))
        colour = COLOURS[row.verdict]
        cv2.circle(small, c, int(row.r * scale), colour, 3)
        for i, text in enumerate([f"plate {row.plate}", f"~{row.est_colonies:.0f} {row.verdict}"]):
            org = (c[0] - 70, c[1] - 10 + 34 * i)
            cv2.putText(small, text, org, font, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(small, text, org, font, 0.9, colour, 2, cv2.LINE_AA)
    return small


def density_check(scanners, cfg: DensityConfig = DensityConfig()) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """(all plates of all ``scanners`` in one table, an overview picture per scanner)."""
    tables, pictures = [], {}
    for s in scanners:
        table, pictures[s.name] = scanner_density(s, cfg)
        tables.append(table)
    return pd.concat(tables, ignore_index=True), pictures


def flagged(table: pd.DataFrame) -> dict[str, list[int]]:
    """{scanner: plate numbers with verdict TNTC}."""
    out: dict[str, list[int]] = {}
    for row in table[table.verdict == "TNTC"].itertuples():
        out.setdefault(row.scanner, []).append(int(row.plate))
    return out

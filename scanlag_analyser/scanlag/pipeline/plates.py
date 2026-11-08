"""Plate geometry from raw scans: find the plates, align frames, build working-area masks.

Plates (PLAN §1.1 steps 1-2): circles are found on the background scan by a Hough transform,
with the expected radius from the plate diameter and the scan resolution (CreateBoardHint.m
assumes 90 mm plates). They are numbered like CreateBoardHint.m: by column, then by row. Plates
whose centres are less than a plate radius apart in x form a column (MATLAB rounds x to 10 % of
the scan width instead). On a 2 x 3 board, plates 1-3 are the left column, top to bottom.

Registration: each frame's shift relative to the background is measured per plate by phase
correlation of the plate's edge image (rim, labels, writing; colonies grow but don't move) and
rounded to whole pixels, like CropROI.m.

Mask: a circle of ``mask_fraction`` x the plate radius (MATLAB default 0.8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

PLATE_DIAMETER_MM = 90.0
MM_PER_INCH = 25.4
CROP_EXTRA = 1.0  # crop = bounding square of the plate circle (MATLAB PlateCirc rect)


@dataclass
class Plate:
    number: int  # 1-based, MATLAB board order
    cx: float  # centre in background-scan pixels
    cy: float
    r: float

    @property
    def crop(self) -> tuple[int, int, int]:
        """(x0, y0, size) of the square crop in background-scan coordinates."""
        size = int(round(2 * self.r * CROP_EXTRA))
        return int(round(self.cx - size / 2)), int(round(self.cy - size / 2)), size

    def to_dict(self) -> dict:
        x0, y0, size = self.crop
        return {**asdict(self), "crop_x0": x0, "crop_y0": y0, "crop_size": size}


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY)


def find_plates(background: np.ndarray, n_plates: int = 6, dpi: float = 300.0,
                diameter_mm: float = PLATE_DIAMETER_MM) -> list[Plate]:
    """The ``n_plates`` strongest plate-sized circles, numbered in MATLAB board order."""
    scale = 4  # search on a downsampled image, then refine at full resolution
    gray = _gray(background)
    small = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    r_expect = diameter_mm / 2 / MM_PER_INCH * dpi
    rs = r_expect / scale
    found = cv2.HoughCircles(cv2.GaussianBlur(small, (5, 5), 1.5), cv2.HOUGH_GRADIENT, dp=1, minDist=1.6 * rs,
                             param1=60, param2=20, minRadius=int(rs * 0.9), maxRadius=int(rs * 1.1))
    if found is None or found.shape[1] < n_plates:
        n = 0 if found is None else found.shape[1]
        raise RuntimeError(f"found {n} plate-sized circles, expected {n_plates}. If the board holds fewer plates, "
                           f"use --plates; plates are expected to be {diameter_mm:g} mm across in a {dpi:g} dpi scan")
    plates = [refine_circle(gray, x * scale, y * scale, r * scale) for x, y, r in found[0, :n_plates]]

    # Columns: plates whose centres are less than a radius apart in x. (Rounding x to 10 % of the
    # width, as MATLAB does, splits a column when its plates straddle a rounding boundary.)
    by_x = sorted(range(n_plates), key=lambda i: plates[i][0])
    columns, current = [], [by_x[0]]
    for i in by_x[1:]:
        if plates[i][0] - plates[current[-1]][0] > r_expect:
            columns.append(current)
            current = []
        current.append(i)
    columns.append(current)
    order = [i for col in columns for i in sorted(col, key=lambda i: plates[i][1])]
    return [Plate(k + 1, *plates[i]) for k, i in enumerate(order)]


def refine_circle(gray: np.ndarray, x: float, y: float, r: float, search: float = 0.05,
                  n_rays: int = 360) -> tuple[float, float, float]:
    """Least-squares circle through the plate's outer rim.

    Along rays from the coarse centre, the rim is the strongest bright-to-dark step going inwards
    (white holder -> dark plate). Rays where the step is weak or far off the median radius
    (labels, the holder's gaps) are dropped before the fit.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 2).astype(np.float32)
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r * (1 - search), r * (1 + search), 0.5)
    xs = x + np.outer(np.cos(theta), radii)
    ys = y + np.outer(np.sin(theta), radii)
    prof = cv2.remap(g, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    step = np.diff(prof, axis=1)  # outward derivative: rim is a large positive step
    best = step.argmax(axis=1)
    strength = step.max(axis=1)
    edge_r = radii[best] + 0.25
    ok = (strength > np.percentile(strength, 30)) & (np.abs(edge_r - np.median(edge_r)) < 0.01 * r)
    if ok.sum() < n_rays // 4:
        return float(x), float(y), float(r)
    px, py = x + edge_r[ok] * np.cos(theta[ok]), y + edge_r[ok] * np.sin(theta[ok])
    # Kasa fit: x^2 + y^2 + D x + E y + F = 0
    A = np.column_stack([px, py, np.ones_like(px)])
    D, E, F = np.linalg.lstsq(A, -(px**2 + py**2), rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    return float(cx), float(cy), float(np.sqrt(cx**2 + cy**2 - F))


def crop(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """Square crop, zero-padded where it leaves the image."""
    out = np.zeros((size, size, *image.shape[2:]), image.dtype)
    r0, c0 = max(y0, 0), max(x0, 0)
    r1, c1 = min(y0 + size, image.shape[0]), min(x0 + size, image.shape[1])
    if r0 < r1 and c0 < c1:
        out[r0 - y0 : r1 - y0, c0 - x0 : c1 - x0] = image[r0:r1, c0:c1]
    return out


def _edges(image: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(_gray(image).astype(np.float32), (0, 0), 2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


class Aligner:
    """Integer (dx, dy) of a plate in later frames relative to the background.

    Phase correlation of the plate's edge image gives a first estimate. The final shift is the
    integer shift within +-``refine`` px of it that minimises the mean absolute gray difference
    to the background over the working area. Background subtraction (the classical detector) is
    what has to line up, and a 1 px error there turns every speck's edge into a blob.
    """

    def __init__(self, background: np.ndarray, plate: Plate, margin: int = 40, refine: int = 1,
                 mask_fraction: float = 0.8):
        self.x0, self.y0, self.size = plate.crop
        self.margin, self.refine = margin, refine
        ref = _edges(crop(background, self.x0 - margin, self.y0 - margin, self.size + 2 * margin))
        self.window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
        self.ref = ref
        self.bg_gray = _gray(crop(background, self.x0, self.y0, self.size)).astype(np.int16)
        mask = circle_mask(self.size, plate, mask_fraction)
        mask[::2] = False  # every other row is plenty
        self.sample = np.nonzero(mask)

    def shift(self, image: np.ndarray) -> tuple[int, int]:
        m = self.margin
        cur = _edges(crop(image, self.x0 - m, self.y0 - m, self.size + 2 * m))
        (dx, dy), _ = cv2.phaseCorrelate(self.ref, cur, self.window)
        dx, dy = int(round(dx)), int(round(dy))
        r = self.refine
        if not r:
            return dx, dy
        # gray window covering every candidate shift, then score each candidate
        win = _gray(crop(image, self.x0 + dx - r, self.y0 + dy - r, self.size + 2 * r)).astype(np.int16)
        ys, xs = self.sample
        bg = self.bg_gray[ys, xs]
        best = min(
            ((float(np.abs(win[ys + r + j, xs + r + i] - bg).mean()), i, j)
             for j in range(-r, r + 1) for i in range(-r, r + 1)),
            key=lambda t: (t[0], abs(t[1]) + abs(t[2])),
        )
        return dx + best[1], dy + best[2]


EMPTY_SLOT_GRAY = 25  # median gray inside an empty holder slot is 2-16 (black lid), agar 36-56


def is_empty_slot(background: np.ndarray, plate: Plate) -> bool:
    """True if the slot holds no dish: its inside is as dark as the scanner lid."""
    inside = _gray(crop(background, *plate.crop))[circle_mask(plate.crop[2], plate, 0.8)]
    return float(np.median(inside)) < EMPTY_SLOT_GRAY


def circle_mask(size: int, plate: Plate, fraction: float = 0.8) -> np.ndarray:
    x0, y0, _ = plate.crop
    yy, xx = np.mgrid[:size, :size]
    return np.hypot(xx - (plate.cx - x0), yy - (plate.cy - y0)) <= fraction * plate.r

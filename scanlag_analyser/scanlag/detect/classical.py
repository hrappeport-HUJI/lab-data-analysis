"""Threshold-based colony detection, a faithful port of the MATLAB ScanLag pipeline.

MATLAB references: cleanImage.m, im2L.m, stretchlim, imadjust, im2bw, medfilt2, bwlabel.
Kept bit-compatible so legacy results can be regenerated and used as reference labels.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import ndimage as ndi

# rgb2gray coefficients: first row of inv(NTSC matrix), as used by MATLAB.
RGB2GRAY = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
DEFAULT_THRESHOLD = 0.2
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@functools.cache
def _gray_table() -> np.ndarray:
    """rgb2gray of every uint8 colour, indexed by r << 16 | g << 8 | b (16 MB)."""
    w = [np.arange(256) * c for c in RGB2GRAY]
    table = np.empty((256, 256, 256), np.uint8)
    for r in range(256):
        table[r] = np.floor(w[0][r] + w[1][:, None] + w[2][None, :] + 0.5)
    return table.ravel()


def rgb2gray_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        gray = rgb[..., :3].astype(np.float64) @ RGB2GRAY
        return np.floor(gray + 0.5).astype(np.uint8)  # MATLAB rounds half away from zero
    # uint8: the same values by table lookup, faster than the float64 product
    index = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    return _gray_table()[index]


def subtract_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """gray(image - background) with uint8 saturation, as cleanImage.m."""
    image, background = image[..., :3], background[..., :3]
    if image.dtype == background.dtype == np.uint8:
        diff = np.maximum(image, background) - background  # saturating subtraction
    else:
        diff = np.clip(image.astype(np.int16) - background, 0, 255).astype(np.uint8)
    return rgb2gray_uint8(diff)


def stretch_limits(gray: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Port of stretchlim for uint8 data (values returned in [0, 1])."""
    cdf = np.cumsum(np.bincount(gray.ravel(), minlength=256)) / gray.size
    lo = int(np.argmax(cdf > low))
    hi = int(np.argmax(cdf >= high))
    if lo == hi:
        return np.array([0.0, 1.0])
    return np.array([lo, hi]) / 255.0


def binarize(
    image: np.ndarray,
    background: np.ndarray,
    limits: np.ndarray,
    threshold: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Port of im2L.m: background subtraction, contrast stretch, threshold, 3x3 median filter."""
    gray = subtract_background(image, background) / 255.0
    lo, hi = limits
    stretched = (np.clip(gray, lo, hi) - lo) / (hi - lo)
    foreground = stretched > threshold
    # medfilt2 on a binary image with zero padding is a 5-of-9 majority vote.
    votes = ndi.correlate(foreground.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    return (votes >= 5) & mask


def label_column_major(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labelling numbered in MATLAB's column-major scan order (bwlabel).

    Label order matters: new colonies get their IDs in label order.
    """
    labels_t, n = ndi.label(binary.T, structure=EIGHT_CONNECTED)
    return np.ascontiguousarray(labels_t.T), n

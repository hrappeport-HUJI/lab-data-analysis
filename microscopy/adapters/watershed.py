"""Threshold foreground without an upper size cap, split at bright septa by an intensity watershed.

Touching daughter cells in phase contrast are separated by a thin line brighter than the cell
interiors. Each connected foreground component is split at such lines when the dark interiors on
either side are deeper than ``h_rel`` times the foreground threshold. Large merged blobs are kept
(never deleted), so a founder's outline does not disappear when its daughters touch.

Validated against human tags on Pilot_3 (2026-11-09): see docs/microscopy/founder_events.md.
"""

import numpy as np
from scipy import ndimage

from . import register_segmenter


def local_contrast(image, sigma):
    return ndimage.gaussian_filter(image, sigma) - image


def auto_threshold(image, sigma=8):
    """Robust local-contrast threshold used by both reference segmenters."""
    image = np.asarray(image, np.float32)
    contrast = local_contrast(image, sigma)
    median = np.median(contrast)
    mad = 1.4826 * np.median(np.abs(contrast - median))
    return float(max(median + 3 * mad, np.ptp(image) * 0.03))


class WatershedSegmenter:
    def segment(self, image, *, context, parameters):
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed

        if parameters.get("polarity", "dark") != "dark":
            raise ValueError("The watershed segmenter currently supports dark cells only")
        img = np.asarray(image, np.float32)
        sigma = float(parameters.get("background_sigma_px", 8))
        contrast = local_contrast(img, sigma)
        threshold = parameters.get("threshold_by_field", {}).get(
            str(context.get("field")), parameters.get("threshold")
        )
        threshold = auto_threshold(img, sigma) if threshold is None else float(threshold)
        mask = ndimage.binary_fill_holes(contrast > threshold)
        components, _ = ndimage.label(mask)
        dark = ndimage.gaussian_filter(img, float(parameters.get("darkness_background_sigma_px", 40))) - img
        h_rel = float(parameters.get("h_rel", 1.0))
        smooth = float(parameters.get("smooth_px", 1.5))
        min_piece = int(parameters.get("min_piece_px", 120))
        min_area = int(parameters.get("min_area_px", 25))
        max_area = parameters.get("max_area_px")  # None: keep every large blob
        out = np.zeros(img.shape, np.int32)
        counter = 0
        for index, box in enumerate(ndimage.find_objects(components), 1):
            if box is None:
                continue
            box = tuple(slice(max(0, s.start - 3), s.stop + 3) for s in box)
            region = components[box] == index
            area = int(region.sum())
            if area < min_area or (max_area is not None and area > max_area):
                continue
            pieces = None
            if h_rel > 0 and area >= 2 * min_piece:
                d = ndimage.gaussian_filter(dark[box], smooth)
                markers, k = ndimage.label(h_maxima(np.where(region, d, d.min()), h_rel * threshold) & region)
                if k > 1:
                    pieces = watershed(-d, markers, mask=region)
                    pieces[region & (pieces == 0)] = 1
                    pieces = _absorb_small(pieces, region, min_piece)
            if pieces is None:
                counter += 1
                out[box][region] = counter
                continue
            for j in np.unique(pieces[region]):
                counter += 1
                out[box][region & (pieces == j)] = counter
        return out


def _absorb_small(pieces, region, min_piece):
    """Merge watershed fragments smaller than ``min_piece`` into the neighbour they touch most."""
    for _ in range(4):
        areas = np.bincount(pieces.ravel())
        small = [j for j in range(1, len(areas)) if 0 < areas[j] < min_piece]
        if not small:
            break
        for j in sorted(small, key=lambda j: areas[j]):
            piece = pieces == j
            if not piece.any():
                continue
            ring = ndimage.binary_dilation(piece) & ~piece & region
            neighbours = pieces[ring]
            neighbours = neighbours[(neighbours > 0) & (neighbours != j)]
            if neighbours.size:
                pieces[piece] = np.bincount(neighbours).argmax()
    return pieces


register_segmenter("watershed", WatershedSegmenter)

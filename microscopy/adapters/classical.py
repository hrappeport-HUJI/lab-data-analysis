"""Conservative reference algorithms, not validated universal cell models."""

import numpy as np
from scipy import ndimage
from . import register_segmenter, register_tracker


class ThresholdSegmenter:
    def segment(self, image, *, context, parameters):
        image = image.astype(float)
        background = ndimage.gaussian_filter(
            image, float(parameters.get("background_sigma_px", 8))
        )
        contrast = (
            background - image
            if parameters.get("polarity", "dark") == "dark"
            else image - background
        )
        threshold = parameters.get("threshold")
        if threshold is None:
            # Robust local-contrast threshold is a proposal, requiring a pilot review.
            median = np.median(contrast)
            mad = 1.4826 * np.median(np.abs(contrast - median))
            threshold = max(median + 3 * mad, np.ptp(image) * 0.03)
        binary = ndimage.binary_fill_holes(contrast > float(threshold))
        labels, _ = ndimage.label(binary)
        result = np.zeros(labels.shape, dtype=np.int32)
        counter = 0
        for label, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            region = labels[sl] == label
            if (
                parameters.get("min_area_px", 12)
                <= region.sum()
                <= parameters.get("max_area_px", 5000)
            ):
                counter += 1
                result[sl][region] = counter
        return result


class OverlapTracker:
    require_overlap = True

    def link(self, previous, current, *, previous_labels, current_labels, parameters):
        candidates = {p["id"]: [] for p in previous}
        ambiguous_parents = set()
        maximum = float(parameters.get("max_displacement_px", 15))
        for child in current:
            options = []
            mask = current_labels == child["label"]
            for parent in previous:
                distance = np.hypot(child["x"] - parent["x"], child["y"] - parent["y"])
                overlap = int(np.sum(previous_labels[mask] == parent["label"]))
                if distance <= maximum and (overlap > 0 or not self.require_overlap):
                    options.append((parent, overlap, distance))
            # Competing parent identities are not resolved by a guessed nearest link.
            if len(options) > 1:
                ambiguous_parents.update(p["id"] for p, _, _ in options)
            if len(options) == 1:
                candidates[options[0][0]["id"]].append(child)
        edges = []
        parents = {p["id"]: p for p in previous}
        for parent_id, children in candidates.items():
            if parent_id in ambiguous_parents:
                continue
            if len(children) == 1:
                edges.append((parent_id, children[0]["id"]))
            elif len(children) == 2:
                area = parents[parent_id]["area_px"]
                low, high = parameters.get("division_area_ratio", [0.65, 1.5])
                total = sum(c["area_px"] for c in children) / area
                minimum = float(parameters.get("min_child_fraction", 0.2))
                if low <= total <= high and all(
                    minimum <= c["area_px"] / area <= 1 - minimum for c in children
                ):
                    edges.extend((parent_id, c["id"]) for c in children)
        return edges


class CentroidTracker(OverlapTracker):
    """Alternative reference adapter; still abstains on ambiguous associations."""

    require_overlap = False


register_segmenter("threshold", ThresholdSegmenter)
register_tracker("overlap", OverlapTracker)
register_tracker("centroid", CentroidTracker)

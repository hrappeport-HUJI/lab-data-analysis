"""Segmentation computed elsewhere, e.g. a GPU model run as a separate batch job.

    "segmentation": {"adapter": "precomputed", "labels_dir": "E:/.../omnipose_labels",
                     "source": "omnipose bact_phase_omni 1.1.4", "min_area_px": 25}

Reads ``<labels_dir>/Pos<field>/<frame, 9 digits>.npz`` (array ``labels``, nonnegative integers with the
image's shape). Objects smaller than ``min_area_px`` are dropped. A missing file is an error, never an
empty frame. Record the producing model in ``source``: it is part of the run's frozen configuration.
"""
from pathlib import Path

import numpy as np

from . import register_segmenter


class PrecomputedSegmenter:
    def segment(self, image, *, context, parameters):
        folder = parameters.get("labels_dir")
        if not folder:
            raise ValueError("The precomputed segmenter needs segmentation.labels_dir")
        path = Path(folder) / f"Pos{context['field']}" / f"{int(context['frame']):09d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No precomputed labels for field {context['field']} frame {context['frame']}: {path}")
        with np.load(path) as stored:
            labels = np.asarray(stored["labels"])
        if labels.shape != np.shape(image):
            raise ValueError(f"{path.name}: labels {labels.shape} do not match the image {np.shape(image)}")
        if labels.dtype.kind not in "iu" or (labels.size and labels.min() < 0):
            raise ValueError(f"{path.name}: labels must be nonnegative integers")
        min_area = int(parameters.get("min_area_px", 0))
        if not labels.size:
            return np.zeros(labels.shape, np.int32)
        areas = np.bincount(labels.ravel())
        if len(areas) <= 1:
            return np.zeros(labels.shape, np.int32)
        keep = areas >= max(1, min_area)
        keep[0] = False
        n_keep = int(keep.sum())
        if n_keep == 0:
            return np.zeros(labels.shape, np.int32)
        if n_keep == len(areas) - 1:
            return labels.astype(np.int32)
        # consecutive ids, as the other segmenters produce
        lut = np.zeros(len(areas), np.int32)
        lut[keep] = np.arange(1, n_keep + 1, dtype=np.int32)
        return lut[labels]


register_segmenter("precomputed", PrecomputedSegmenter)

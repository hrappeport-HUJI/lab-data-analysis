"""Validated raw-image access shared by algorithms and the viewer."""

from functools import lru_cache
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage
from .storage import file_digest


@lru_cache(maxsize=12)
def _read(path, modified_ns, size, expected_digest):
    if file_digest(path) != expected_digest:
        raise ValueError("A raw image changed since import. Import a new project.")
    image = tifffile.imread(path)
    if not np.isfinite(image).all():
        raise ValueError(
            "Raw image contains nonfinite intensities; define a missing-pixel policy before analysis"
        )
    image.setflags(write=False)
    return image


def read_frame(project, field, frame, channel):
    record = project.frames[(str(field), int(frame), channel)]
    path = (project.raw / record["path"]).resolve()
    if not path.is_relative_to(project.raw):
        raise ValueError("Image path outside acquisition")
    stat = path.stat()
    return _read(str(path), stat.st_mtime_ns, stat.st_size, record["sha256"])


def quality(image):
    image = np.asarray(image)
    lo, hi = np.percentile(image, [1, 99])
    contrast = float(hi - lo)
    normalized = (image.astype(float) - lo) / max(contrast, 1.0)
    score = float(np.var(ndimage.laplace(normalized)))
    saturation = (
        float(np.mean(image == np.iinfo(image.dtype).max))
        if image.dtype.kind in "ui"
        else None
    )
    return {
        "focus_score": score,
        "contrast_p01_p99": contrast,
        "saturated_fraction": saturation,
    }

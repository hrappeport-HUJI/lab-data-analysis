"""Cache geometry and naming rules.

Image requests send little-endian uint16 channel planes followed by an optional
uint8 mask. Overviews preserve aspect ratio and cap the long edge at the
requested level. The native stack keeps its historical -2048.npy suffix.
"""

import re
from pathlib import PureWindowsPath

import numpy as np
from PIL import Image

LEVELS = (512, 1024, 2048)
NATIVE = 2048  # The historical filename suffix for native pixels, whatever their shape.
PIXEL_DTYPE = np.dtype("<u2")


def component(value):
    """Validate a position or channel name before using it in a cache path."""
    path = PureWindowsPath(value)
    if (
        not value or path.drive or path.is_reserved() or value in {".", ".."}
        or value.endswith((" ", ".")) or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
    ):
        raise ValueError(f"Invalid cache name: {value!r}")
    return value


def channel_name(value):
    if "," in value:
        raise ValueError(f"Channel names cannot contain commas: {value!r}")
    return component(value)


def natural_key(value):
    return tuple(int(part) if i % 2 else part.casefold()
                 for i, part in enumerate(re.split(r"(\d+)", value, flags=re.ASCII)))


def overview_shape(height, width, level):
    longest = max(height, width)
    if longest <= level:
        return height, width
    return tuple(max(1, (dimension * level + longest // 2) // longest)
                 for dimension in (height, width))


def box_resize(plane, shape):
    if plane.shape == shape:
        return plane
    height, width = shape
    image = Image.fromarray(plane.astype(np.float32))
    return np.asarray(image.resize((width, height), Image.Resampling.BOX))


def overview_pixels(plane, shape):
    return np.rint(box_resize(plane, shape)).astype(PIXEL_DTYPE, copy=False)

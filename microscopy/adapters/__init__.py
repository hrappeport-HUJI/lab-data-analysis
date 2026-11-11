"""Small adapter contracts; installed extensions can register implementations."""

from typing import Protocol
import numpy as np


class Segmenter(Protocol):
    def segment(
        self, image: np.ndarray, *, context: dict, parameters: dict
    ) -> np.ndarray:
        """Return nonnegative integer labels with the image's Y,X shape."""


class Tracker(Protocol):
    def link(
        self,
        previous: list[dict],
        current: list[dict],
        *,
        previous_labels: np.ndarray,
        current_labels: np.ndarray,
        parameters: dict,
    ) -> list[tuple[str, str]]:
        """Return forward parent-child links. Multiple children may propose division."""


_SEGMENTERS = {}
_TRACKERS = {}


def register_segmenter(name, factory):
    if name in _SEGMENTERS:
        raise ValueError(f"Segmenter already registered: {name}")
    _SEGMENTERS[name] = factory


def register_tracker(name, factory):
    if name in _TRACKERS:
        raise ValueError(f"Tracker already registered: {name}")
    _TRACKERS[name] = factory


def get_adapter(kind, name):
    from . import classical, watershed  # registers builtins

    registry = _SEGMENTERS if kind == "segmenter" else _TRACKERS
    if name in registry:
        return registry[name]()
    from importlib.metadata import entry_points

    matches = list(entry_points(group=f"lab_microscopy.{kind}s", name=name))
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous {kind}: {name}")
    return matches[0].load()()

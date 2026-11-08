"""Area tracks for MATLAB's colonies, measured on another detector's labels.

Colonies don't move, so a colony is followed by its MATLAB seed pixel: its area at frame t is the
area of the detected object covering the seed (0 if none). An object that covers the seeds of
several colonies is a merged blob. Every colony under it gets the blob's area and ``merged=True``,
as in MATLAB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def seed_areas(labels: np.ndarray, seed_x: np.ndarray, seed_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(area, merged) per seed for one frame of ``labels``."""
    obj = labels[seed_y, seed_x]
    areas = np.bincount(labels.ravel(), minlength=int(obj.max()) + 1)
    area = np.where(obj > 0, areas[obj], 0)
    n_seeds = np.bincount(obj, minlength=int(obj.max()) + 1)
    merged = (obj > 0) & (n_seeds[obj] > 1)
    return area, merged


def tracks_table(area: np.ndarray, merged: np.ndarray, colony_ids: np.ndarray, times_min) -> pd.DataFrame:
    """Long table (frame, time_min, colony_id, area_px, merged) from [T, N] arrays, area > 0 only."""
    t, c = np.nonzero(area)
    return pd.DataFrame(
        {
            "frame": t,
            "time_min": np.asarray(times_min)[t],
            "colony_id": colony_ids[c],
            "area_px": area[t, c],
            "merged": merged[t, c],
        }
    )

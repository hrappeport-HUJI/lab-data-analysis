import os
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from scanlag.detect.classical import binarize, label_column_major, rgb2gray_uint8, stretch_limits
from scanlag.track.seed_tracker import SeedTracker, border_status

# the legacy MATLAB experiments (skipped if missing); set SCANLAG_LEGACY_DATA on another machine
LEGACY_DATA = Path(os.environ.get("SCANLAG_LEGACY_DATA", r"D:\Alon\Documents\Experiments\microcolony-seq"))
DENSE_PLATE = LEGACY_DATA / "EPEC260216" / "AMP24" / "ROI" / "sep_2_2"


def test_rgb2gray_matches_matlab_values():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], np.uint8)
    # MATLAB: rgb2gray(uint8([255 0 0; ...])) -> 76, 150, 29, 18
    assert rgb2gray_uint8(rgb).tolist() == [[76, 150, 29, 18]]


def test_labels_numbered_column_major():
    binary = np.zeros((5, 5), bool)
    binary[0, 4] = True  # first in row-major order
    binary[4, 0] = True  # first in column-major order
    labels, n = label_column_major(binary)
    assert n == 2 and labels[4, 0] == 1 and labels[0, 4] == 2


def test_binarize_median_is_majority_with_zero_padding():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    background = np.zeros_like(image)
    mask = np.ones((40, 40), bool)
    result = binarize(image, background, np.array([0.0, 1.0]), 0.5, mask)
    raw = rgb2gray_uint8(image) / 255.0 > 0.5
    expected = ndi.median_filter(raw.astype(np.uint8), size=3, mode="constant") > 0
    np.testing.assert_array_equal(result, expected)


def test_stretch_limits_constant_image():
    assert stretch_limits(np.full((4, 4), 7, np.uint8)).tolist() == [0.0, 1.0]


def test_merged_colonies_share_blob():
    tracker = SeedTracker(min_area=1)
    frame0 = np.zeros((20, 20), bool)
    frame0[2:6, 2:6] = frame0[2:6, 10:14] = True
    tracker.update(*label_column_major(frame0))
    frame1 = frame0.copy()
    frame1[2:6, 2:14] = True  # the two colonies grew together
    tracker.update(*label_column_major(frame1))
    area, _, blob = tracker.tracks.as_arrays()
    assert area[0].tolist() == [16, 16]
    assert area[1].tolist() == [48, 48] and blob[1, 0] == blob[1, 1]


def test_frame_without_blobs_after_colonies_exist():
    tracker = SeedTracker(min_area=1)
    frame = np.zeros((10, 10), bool)
    frame[2:5, 2:5] = True
    tracker.update(*label_column_major(frame))
    tracker.update(*label_column_major(np.zeros((10, 10), bool)))
    area, centroid, _ = tracker.tracks.as_arrays()
    assert area[:, 0].tolist() == [9, 0] and centroid[1, 0].tolist() == [0, 0]


def test_border_status():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    assert border_status(mask, [(50, 50), (20, 50)], radius=15).tolist() == [True, False]


@pytest.mark.skipif(not DENSE_PLATE.exists(), reason="legacy data not available")
def test_parity_with_matlab_on_dense_plate():
    from scanlag.io.matlab_import import load_plate
    from scanlag.testset.regenerate import LegacyReplay, check_parity

    plate = load_plate(DENSE_PLATE)
    replay = LegacyReplay(plate)
    for _ in replay.frames():
        pass
    area, _, _ = replay.tracker.tracks.as_arrays()
    assert check_parity(plate, area).exact

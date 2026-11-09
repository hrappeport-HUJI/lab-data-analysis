import numpy as np
from scipy import ndimage as ndi

from scanlag.track.tracker import FootprintTracker

SHAPE = (60, 80)
COLS = np.indices(SHAPE)[1]


def disk(cy, cx, r):
    yy, xx = np.indices(SHAPE)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def grow_two(tracker, radii):
    """Two colonies at x = 25 and 52 growing into each other; returns the last detected labels."""
    for r in radii:
        labels, n = ndi.label(disk(30, 25, r) | disk(30, 52, 0.8 * r))
        tracker.update(labels, n)
    return labels


def test_merged_colonies_keep_their_own_area():
    tracker = FootprintTracker(min_area=5)
    labels = grow_two(tracker, range(3, 16))
    area, centroid, blob = tracker.tracks.as_arrays()
    assert area.shape[1] == 2  # no new colony at the merge
    merged = blob[:, 0] == blob[:, 1]
    assert not merged[:-1].any() and merged[-1]  # they touch in the last frame only
    assert (np.diff(area, axis=0) > 0).all()  # both keep growing, no jump to the merged area
    assert area[-1].sum() == (labels > 0).sum() and area[-1, 0] < 1.3 * area[-2, 0]
    assert centroid[-1, 0, 0] < 30 < 45 < centroid[-1, 1, 0]  # each stays at its own place


def test_split_follows_previous_footprints():
    tracker = FootprintTracker(min_area=5)
    grow_two(tracker, range(3, 16))
    own = tracker.update(*ndi.label(disk(30, 25, 16) | disk(30, 52, 12.8)))
    assert (own[30, 9:38] == 1).all() and (own[30, 40:65] == 2).all()


def test_bad_instance_split_starts_no_new_colony():
    tracker = FootprintTracker(min_area=5)
    labels = grow_two(tracker, range(3, 16))
    pieces = labels.copy()  # the detector cuts the merged blob into three, with a bridge in the middle
    pieces[(labels > 0) & (COLS >= 36)] = 2
    pieces[(labels > 0) & (COLS > 40)] = 3
    tracker.update(pieces, 3)
    assert len(tracker.tracks.seeds) == 2
    assert tracker.tracks.area[-1].sum() == (labels > 0).sum()


def test_fragments_stay_one_colony():
    tracker = FootprintTracker(min_area=5)
    tracker.update(*ndi.label(disk(30, 30, 8)))
    frag = disk(30, 30, 9) & (COLS != 30)  # broken in two by a one-pixel gap
    tracker.update(*ndi.label(frag))
    assert len(tracker.tracks.seeds) == 1 and tracker.tracks.area[-1][0] == frag.sum()


def test_jitter_overlap_is_no_merge():
    tracker = FootprintTracker(min_area=5)
    tracker.update(*ndi.label(disk(30, 20, 6) | disk(30, 34, 6)))
    # colony 2 is detected 2 px to the right (alignment jitter); colony 1 now reaches into a few
    # pixels of colony 2's previous footprint without touching colony 2
    first, second = disk(30, 20, 8), disk(30, 36, 6)
    assert not (first & second).any() and (first & disk(30, 34, 6)).sum() < 5
    labels, n = ndi.label(first | second)
    tracker.update(labels, n)
    area, _, blob = tracker.tracks.as_arrays()
    assert blob[-1, 0] != blob[-1, 1]
    assert area[-1].tolist() == [first.sum(), second.sum()]


def test_gap_and_reappearance():
    tracker = FootprintTracker(min_area=1)
    frame = np.zeros((10, 10), int)
    frame[2:5, 2:5] = 1
    tracker.update(frame, 1)
    tracker.update(np.zeros((10, 10), int), 0)
    tracker.update(frame, 1)  # found again by its seed pixel
    area, centroid, _ = tracker.tracks.as_arrays()
    assert area[:, 0].tolist() == [9, 0, 9] and centroid[1, 0].tolist() == [0, 0]
    assert len(tracker.tracks.seeds) == 1


def test_small_unmatched_blob_is_ignored():
    tracker = FootprintTracker(min_area=10)
    frame = np.zeros((20, 20), int)
    frame[2:4, 2:4] = 1
    own = tracker.update(frame, 1)
    assert len(tracker.tracks.seeds) == 0 and not own.any()

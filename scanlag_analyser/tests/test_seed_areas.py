import numpy as np

from scanlag.track.seed_areas import seed_areas, tracks_table


def test_seed_areas_single_merged_and_missing():
    labels = np.zeros((10, 10), np.uint16)
    labels[0:2, 0:2] = 1  # one colony, 4 px
    labels[5:8, 5:8] = 2  # blob over two seeds, 9 px
    sx, sy = np.array([0, 5, 7, 9]), np.array([0, 5, 7, 0])
    area, merged = seed_areas(labels, sx, sy)
    assert area.tolist() == [4, 9, 9, 0]
    assert merged.tolist() == [False, True, True, False]


def test_tracks_table_keeps_nonzero():
    area = np.array([[0, 3], [5, 4]])
    merged = np.zeros_like(area, bool)
    t = tracks_table(area, merged, np.array([10, 11]), [0.0, 15.0])
    assert list(zip(t.frame, t.colony_id, t.area_px)) == [(0, 11, 3), (1, 10, 5), (1, 11, 4)]

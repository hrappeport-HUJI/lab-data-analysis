from datetime import datetime

import numpy as np
import tifffile

from scanlag.io.raw import ScanSeries, parse_timestamp


def test_parse_timestamp():
    assert parse_timestamp("epec-amp7_1-4.4.2u_20260324_0025.tiff") == datetime(2026, 3, 24, 0, 25)
    assert parse_timestamp("x_20260324_002530.tif") == datetime(2026, 3, 24, 0, 25, 30)
    assert parse_timestamp("epec-microcolony-seq005.tif") is None


def test_series_sorted_by_name_time_and_read_only(tmp_path):
    for stamp, value in [("20260324_0040", 2), ("20260323_2225", 0), ("20260324_0025", 1)]:
        tifffile.imwrite(tmp_path / f"s_{stamp}.tiff", np.full((8, 8, 3), value, np.uint8))
    series = ScanSeries.from_pattern(tmp_path, "s_*.tiff")
    assert [int(series.read(i)[0, 0, 0]) for i in range(3)] == [0, 1, 2]
    assert not series.read(0).flags.writeable

    window = series.read_window(2, slice(-2, 3), slice(6, 10))
    assert window.shape == (5, 4, 3)
    assert window[:2].sum() == 0 and window[2:, :2].min() == 2 and window[:, 2:].sum() == 0

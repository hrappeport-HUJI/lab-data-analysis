"""The faster gray conversion, blob statistics, centroid filter and tracker give bit-identical
results to the versions they replaced (kept here as references), and the scan feed shared by the
plate workers hands out every frame in order."""

import threading

import numpy as np
from scipy import ndimage as ndi

from scanlag.detect.classical import RGB2GRAY, rgb2gray_uint8, subtract_background
from scanlag.pipeline.run import ScanFeed, keep_centered
from scanlag.track.seed_tracker import SeedTracker, blob_properties, matlab_round
from scanlag.track.tracker import FootprintTracker, colony_properties


def ref_blob_properties(labels, n):
    idx = np.arange(1, n + 1)
    area = ndi.sum_labels(np.ones_like(labels), labels, idx).astype(np.int64)
    rows, cols = np.indices(labels.shape)
    cy = ndi.mean(rows, labels, idx) + 1
    cx = ndi.mean(cols, labels, idx) + 1
    return area, np.column_stack([cx, cy]).reshape(n, 2)


def ref_keep_centered(labels, mask):
    ids = np.unique(labels)[1:]
    if len(ids):
        cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, ids)).round().astype(int).T
        keep = ids[mask[np.clip(cy, 0, mask.shape[0] - 1), np.clip(cx, 0, mask.shape[1] - 1)]]
        lut = np.zeros(labels.max() + 1, np.int32)
        lut[keep] = np.arange(1, len(keep) + 1)
        labels = lut[labels]
    return labels


class RefFootprintTracker(FootprintTracker):
    """FootprintTracker.update as it was: one full-image pass per new blob."""

    def update(self, labels, n):
        labels = np.asarray(labels, np.int64)
        seeds = self.tracks.seeds
        prev = np.zeros(labels.shape, np.int64) if self.own is None else self.own
        owners = self._owners(labels, n, prev)
        lut = np.zeros(n + 1, np.int64)
        for b, cs in owners.items():
            if len(cs) == 1:
                lut[b] = cs[0]
        own = lut[labels]
        objects = ndi.find_objects(labels, max_label=n)
        for b, cs in owners.items():
            if len(cs) > 1:
                self._split(labels, prev, b, cs, objects[b - 1], own)
        blob_area, blob_centroid = ref_blob_properties(labels, n)
        for b in range(1, n + 1):
            if b not in owners and blob_area[b - 1] >= self.min_area:
                x, y = matlab_round(blob_centroid[b - 1])
                seeds.append((int(y) - 1, int(x) - 1))
                own[labels == b] = len(seeds)
        n_col = len(seeds)
        area, centroid = colony_properties(own, n_col)
        self.tracks.area.append(area)
        self.tracks.centroid.append(centroid)
        self.tracks.blob.append(self._main_blob(labels, own, n_col))
        self.own = own
        return own


def random_labels(rng, shape=(160, 200), n_blobs=60, gaps=False):
    seeds = np.zeros(shape, bool)
    seeds[rng.integers(0, shape[0], n_blobs), rng.integers(0, shape[1], n_blobs)] = True
    grown = ndi.binary_dilation(seeds, iterations=int(rng.integers(1, 6))) & (rng.random(shape) > 0.05)
    labels, n = ndi.label(grown)
    if gaps and n > 3:  # drop some labels without renumbering
        labels[np.isin(labels, rng.choice(np.arange(1, n + 1), n // 4, replace=False))] = 0
    return labels.astype(np.int32), n


def growing_frames(rng, n_frames=12, shape=(160, 200), n_col=70):
    """Colonies that appear at random times and grow until many of them merge."""
    yy, xx = np.indices(shape)
    cy, cx = rng.uniform(0, shape[0], n_col), rng.uniform(0, shape[1], n_col)
    start, rate = rng.integers(0, n_frames - 2, n_col), rng.uniform(0.6, 1.8, n_col)
    for k in range(n_frames):
        fg = np.zeros(shape, bool)
        for i in range(n_col):
            if k >= start[i]:
                fg |= (yy - cy[i]) ** 2 + (xx - cx[i]) ** 2 <= (1 + rate[i] * (k - start[i])) ** 2
        fg &= rng.random(shape) > 0.02  # holes and fragments
        labels, n = ndi.label(fg)
        yield labels.astype(np.int32), n


def test_blob_properties_match_ndimage():
    rng = np.random.default_rng(0)
    for i in range(20):
        labels, n = random_labels(rng, gaps=i % 2 == 1)
        area, centroid = blob_properties(labels, n)
        ref_area, ref_centroid = ref_blob_properties(labels, n)
        np.testing.assert_array_equal(area, ref_area)
        np.testing.assert_array_equal(centroid, ref_centroid)  # NaN for missing labels, as ndi.mean
    area, centroid = blob_properties(np.zeros((5, 5), np.int32), 0)
    assert area.shape == (0,) and centroid.shape == (0, 2)


def test_keep_centered_matches_center_of_mass():
    rng = np.random.default_rng(1)
    yy, xx = np.indices((160, 200))
    mask = (yy - 80) ** 2 + (xx - 100) ** 2 < 70**2
    for i in range(20):
        labels, _ = random_labels(rng, gaps=i % 2 == 1)
        np.testing.assert_array_equal(keep_centered(labels, mask), ref_keep_centered(labels, mask))
    empty = np.zeros((160, 200), np.int32)
    np.testing.assert_array_equal(keep_centered(empty, mask), empty)


def test_footprint_tracker_unchanged():
    for seed in range(4):
        new, ref = FootprintTracker(), RefFootprintTracker()
        for labels, n in growing_frames(np.random.default_rng(seed)):
            np.testing.assert_array_equal(new.update(labels, n), ref.update(labels, n))
        assert new.tracks.seeds == ref.tracks.seeds
        for a, b in zip(new.tracks.as_arrays(), ref.tracks.as_arrays()):
            np.testing.assert_array_equal(a, b)


def test_seed_tracker_unchanged():
    import scanlag.track.seed_tracker as st

    frames = list(growing_frames(np.random.default_rng(5)))
    new = SeedTracker()
    for labels, n in frames:
        new.update(labels, n)
    ref = SeedTracker()
    st.blob_properties, saved = ref_blob_properties, st.blob_properties
    try:
        for labels, n in frames:
            ref.update(labels, n)
    finally:
        st.blob_properties = saved
    assert new.tracks.seeds == ref.tracks.seeds
    for a, b in zip(new.tracks.as_arrays(), ref.tracks.as_arrays()):
        np.testing.assert_array_equal(a, b)


def ref_rgb2gray_uint8(rgb):
    return np.floor(rgb[..., :3].astype(np.float64) @ RGB2GRAY + 0.5).astype(np.uint8)


def test_gray_table_matches_float_product():
    every = np.stack(np.meshgrid(*[np.arange(256)] * 3, indexing="ij"), -1).astype(np.uint8)[::3]  # every third r, all g and b
    np.testing.assert_array_equal(rgb2gray_uint8(every), ref_rgb2gray_uint8(every))
    rng = np.random.default_rng(2)
    image, background = rng.integers(0, 256, (2, 90, 70, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rgb2gray_uint8(image[::2, 1:]), ref_rgb2gray_uint8(image[::2, 1:]))  # a view
    diff = np.clip(image.astype(np.int16) - background, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(subtract_background(image, background), ref_rgb2gray_uint8(diff))
    np.testing.assert_array_equal(rgb2gray_uint8(image.astype(float)), ref_rgb2gray_uint8(image))


def test_scan_feed_shares_frames_between_workers(monkeypatch):
    import scanlag.pipeline.run as run

    n, plates = 12, 5
    monkeypatch.setattr(run, "read_scan", lambda k: np.full(3, k))
    done = []
    feed = ScanFeed(list(range(n)), np.full(3, 0), plates, ahead=2, on_scan_done=lambda: done.append(1))
    seen = {p: [] for p in range(plates)}
    in_memory = []

    def worker(my_plates):
        for k in range(n):
            for p in my_plates:
                seen[p].append(int(feed.get(k)[0]))
                in_memory.append(len(feed.futures))

    threads = [threading.Thread(target=worker, args=(g,)) for g in ([0, 1], [2], [3, 4])]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    feed.close()
    assert all(s == list(range(n)) for s in seen.values())
    assert len(done) == n and not feed.futures
    assert max(in_memory) <= feed.ahead + 2

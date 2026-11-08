"""The detector-free density check (TNTC plates) and the final whole-plate count."""

import numpy as np
import pandas as pd

from scanlag.pipeline.density import DensityConfig, flagged, overview, plate_density, verdict
from scanlag.pipeline.plates import Plate, circle_mask


def plate_images(n_colonies: int, radius: int, size: int = 400, seed: int = 0):
    """(background, last frame) crops of a dark plate with ``n_colonies`` bright discs in the working area."""
    rng = np.random.default_rng(seed)
    background = np.full((size, size, 3), 45, np.uint8)
    image = background.copy()
    yy, xx = np.mgrid[:size, :size]
    placed = []
    while len(placed) < n_colonies:
        r = rng.uniform(0, 0.75 * size / 2)
        a = rng.uniform(0, 2 * np.pi)
        cx, cy = size / 2 + r * np.cos(a), size / 2 + r * np.sin(a)
        if all(np.hypot(cx - x, cy - y) > 2 * radius + 2 for x, y in placed):  # separate colonies
            placed.append((cx, cy))
            image[np.hypot(xx - cx, yy - cy) <= radius] = 120
    return background, image


def test_density_estimate_follows_colony_number():
    plate = Plate(1, 200, 200, 200)
    mask = circle_mask(400, plate, 0.8)
    for n, radius in [(20, 8), (120, 4)]:
        background, image = plate_images(n, radius)
        m = plate_density(background, image, mask)
        assert m["n_patches"] == n
        assert abs(m["est_colonies"] - n) <= 0.1 * n
    background, _ = plate_images(0, 4)
    assert plate_density(background, background, mask)["est_colonies"] == 0


def test_lawn_is_tntc_although_it_is_one_blob():
    plate = Plate(1, 200, 200, 200)
    mask = circle_mask(400, plate, 0.8)
    background, _ = plate_images(0, 4)
    lawn = background.copy()
    lawn[mask] = 120
    lawn[::7, :] = 45  # thin lines break the lawn into narrow strips, as cracks would
    m = plate_density(background, lawn, mask)
    assert m["coverage"] > 0.7 and m["est_colonies"] > 0  # a few big patches, but most of the plate covered


def test_verdicts_and_flagged():
    cfg = DensityConfig(tntc=500, dense=200)
    assert [verdict(x, cfg) for x in (50, 300, 1200)] == ["ok", "dense", "TNTC"]
    table = pd.DataFrame({"scanner": ["A", "A", "B"], "plate": [1, 2, 3], "verdict": ["TNTC", "ok", "TNTC"]})
    assert flagged(table) == {"A": [1], "B": [3]}


def test_overview_picture():
    scan = np.zeros((300, 200, 3), np.uint8)
    table = pd.DataFrame({"plate": [1], "cx": [100.0], "cy": [150.0], "r": [80.0], "est_colonies": [1200.0],
                          "verdict": ["TNTC"]})
    picture = overview(scan, table, width=100)
    assert picture.shape == (150, 100, 3) and picture.any()


def test_final_objects_counts_the_whole_plate():
    from scanlag.pipeline.run import RunConfig, final_objects

    plate = Plate(1, 200, 200, 200)
    yy, xx = np.mgrid[:400, :400]
    labels = np.zeros((400, 400), np.int32)
    labels[np.hypot(xx - 200, yy - 200) <= 10] = 1  # inner colony
    labels[np.hypot(xx - 230, yy - 200) <= 10] = 2  # inner colony
    labels[(np.hypot(xx - 160, yy - 160) <= 10) | (np.hypot(xx - 160, yy - 178) <= 10)] = 3  # two merged
    labels[np.hypot(xx - 370, yy - 200) <= 10] = 4  # rim colony (r_frac 0.85)
    labels[5:9, 150:250] = 6  # elongated object in the rim band (a wall reflection)
    cfg = RunConfig(mask_fraction=0.8, full_fraction=0.92)
    table, totals = final_objects(labels, plate, cfg)
    by_id = table.set_index("object_id")
    assert by_id.loc[4, "region"] == "rim" and by_id.loc[4, "counted"]
    assert by_id.loc[6, "region"] == "rim" and not by_id.loc[6, "counted"]  # elongated: rejected
    assert totals["n_rim_rejected"] == 1
    assert totals["n_full"] == totals["n_inner"] + totals["n_rim"]
    assert totals["n_full_est"] == totals["n_full"] + 1  # the merged pair counts as two

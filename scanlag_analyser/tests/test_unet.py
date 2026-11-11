import numpy as np
import torch

from scanlag.detect.unet import distance_target, instances
from scanlag.train.unet_train import unet_loss


def two_touching_disks():
    yy, xx = np.mgrid[:40, :60]
    labels = np.zeros((40, 60), np.uint16)
    labels[(yy - 20) ** 2 + (xx - 20) ** 2 <= 100] = 1
    labels[((yy - 20) ** 2 + (xx - 38) ** 2 <= 100) & (labels == 0)] = 2  # touches disk 1
    return labels


def test_distance_target_peaks_at_one_per_colony():
    labels = two_touching_disks()
    d = distance_target(labels)
    assert d[labels == 0].max() == 0
    assert np.isclose(d[labels == 1].max(), 1) and np.isclose(d[labels == 2].max(), 1)


def test_instances_split_touching_colonies():
    labels = two_touching_disks()
    out = instances((labels > 0).astype(float), distance_target(labels), core=0.5, min_size=5, smooth=0)
    assert out.max() == 2
    assert len(np.unique(out[labels == 1])) == 1 and len(np.unique(out[labels == 2])) == 1


def test_loss_ignores_masked_pixels():
    fg = torch.zeros(1, 8, 8)
    fg[:, :4] = 1
    y = torch.stack([(fg * 2 - 1) * 30, torch.full_like(fg, -30)], 1)  # right fg, distance 0 everywhere
    valid = torch.ones(1, 8, 8)
    good = unet_loss(y, fg, torch.zeros_like(fg), valid, 5.0)
    y_bad = y.clone()
    y_bad[:, 0, 4:] = 30  # wrong in the bottom half...
    valid[:, 4:] = 0  # ...which is ignored
    assert torch.isclose(unet_loss(y_bad, fg, torch.zeros_like(fg), valid, 5.0), good, atol=1e-3)

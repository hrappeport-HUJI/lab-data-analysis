import numpy as np
import torch

from scanlag.train.cellpose_finetune import BSIZE, augment, masked_loss, normalize, norm_limits, targets


def test_augment_upscales_and_keeps_label_ids():
    rng = np.random.default_rng(0)
    x = rng.random((256, 256, 3)).astype(np.float32)
    labels = np.zeros((256, 256), np.uint16)
    labels[100:110, 100:110] = 7
    ignore = np.zeros((256, 256), bool)
    ignore[:, :50] = True
    xi, li, ii = augment(x, labels, ignore, 3.0, rng)
    assert xi.shape == (BSIZE, BSIZE, 3) and li.shape == ii.shape == (BSIZE, BSIZE)
    assert set(np.unique(li)) <= {0, 7}  # nearest-neighbour: no blended IDs


def test_targets_and_masked_loss():
    labels = np.zeros((64, 64), np.int32)
    labels[10:30, 10:30] = 5
    t = torch.from_numpy(targets(labels, torch.device("cpu")))[None]
    assert t[0, 0].sum() == 400 and t[0, 1:].abs().max() <= 1.0001
    perfect = torch.cat([5 * t[:, 1:], (t[:, :1] * 2 - 1) * 50], 1)  # flows, very confident logits
    valid = torch.ones(1, 64, 64)
    assert masked_loss(perfect, t, valid) < 1e-6
    wrong = perfect.clone()
    wrong[:, 2, :32] *= -1  # wrong cell probability in the top half...
    valid[:, :32] = 0  # ...which is ignored
    assert masked_loss(wrong, t, valid) < 1e-6


def test_fixed_normalization():
    x = np.stack([np.full((10, 10), v, np.uint8) for v in (0, 100, 200)], -1)
    x[0, 0] = [50, 150, 250]
    lim = norm_limits(x, np.zeros((10, 10), bool))
    assert normalize(x, lim).shape == x.shape

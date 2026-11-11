import numpy as np

from scanlag.eval.instances import counts, drop_ignored, match, summarize


def square(labels, i, y, x, s):
    labels[y : y + s, x : x + s] = i


def test_perfect_prediction():
    true = np.zeros((50, 50), np.uint16)
    square(true, 1, 0, 0, 10)
    square(true, 2, 20, 20, 4)
    c = counts(match(true, true * 3))  # different IDs, same objects
    assert (c["tp"], c["fp"], c["fn"]) == (2, 0, 0)
    s = summarize([c])
    assert s["f1"] == 1 and s["small_f1"] == 1 and s["mean_iou_matched"] == 1


def test_iou_threshold_and_merge():
    true = np.zeros((20, 40), np.uint16)
    square(true, 1, 0, 0, 10)
    square(true, 2, 0, 10, 10)
    pred = (true > 0).astype(np.uint16)  # both colonies as one blob: IoU 0.5, not > 0.5
    c = counts(match(true, pred))
    assert (c["tp"], c["fp"], c["fn"]) == (0, 1, 2)
    c = counts(match(true, pred, iou_threshold=0.3))  # Hungarian: blob goes to one colony
    assert (c["tp"], c["fp"], c["fn"]) == (1, 0, 1)


def test_small_counts_by_own_side():
    true = np.zeros((30, 30), np.uint16)
    square(true, 1, 0, 0, 5)  # 25 px, small
    pred = np.zeros_like(true)
    square(pred, 1, 0, 0, 6)  # 36 px, not small; IoU 25/36
    c = counts(match(true, pred))
    assert c["tp"] == 1 and c["small_true_hit"] == 1 and c["small_pred"] == 0


def test_ignore_region():
    true = np.zeros((20, 20), np.uint16)
    pred = np.zeros_like(true)
    square(pred, 1, 0, 0, 4)  # entirely outside the working area
    square(pred, 2, 10, 10, 4)  # inside, no true colony -> FP
    ignore = np.zeros_like(true, bool)
    ignore[:5, :5] = True
    assert set(np.unique(drop_ignored(pred, ignore))) == {0, 2}
    c = counts(match(true, pred, ignore=ignore))
    assert (c["tp"], c["fp"], c["fn"]) == (0, 1, 0)

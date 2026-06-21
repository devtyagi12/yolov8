"""Tests for validation metrics: ap_per_class curves, confusion matrix, plots.

Run:  python tests/test_metrics.py   (or)   python -m pytest tests/test_metrics.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.utils.metrics import ConfusionMatrix, ap_per_class, smooth


def test_ap_per_class_perfect_detector():
    # 3 predictions, all true positives across all IoU thresholds, classes 0,1,2.
    n, niou = 3, 10
    tp = np.ones((n, niou), bool)
    conf = np.array([0.9, 0.8, 0.7])
    pred_cls = np.array([0, 1, 2])
    target_cls = np.array([0, 1, 2])
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    # 101-point COCO interpolation gives ~0.995 for a single-point perfect detector
    assert res["map50"] > 0.99 and res["map"] > 0.99
    assert res["mp"] > 0.99 and res["mr"] > 0.99
    # curve data present and correctly shaped
    assert res["curves"]["p"].shape == (3, 1000)
    assert res["ap50"].shape == (3,) and res["ap_class"].shape == (3,)


def test_ap_per_class_with_false_positives():
    tp = np.array([[1] * 10, [0] * 10, [1] * 10], bool)  # one FP
    conf = np.array([0.9, 0.85, 0.6])
    pred_cls = np.array([0, 0, 0])
    target_cls = np.array([0, 0])
    res = ap_per_class(tp, conf, pred_cls, target_cls)
    assert 0.0 < res["map50"] <= 1.0


def test_smooth_preserves_length():
    y = np.random.rand(100)
    assert smooth(y, 0.1).shape == y.shape


def test_confusion_matrix_counts():
    cm = ConfusionMatrix(nc=2, conf=0.25, iou_thres=0.45)
    # one correct detection (class 0) + one missed GT (class 1)
    det = np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)
    gt_boxes = np.array([[12, 12, 48, 48], [100, 100, 140, 140]], np.float32)
    gt_cls = np.array([0, 1])
    cm.process_batch(det, gt_boxes, gt_cls)
    assert cm.matrix[0, 0] == 1  # TP class 0
    assert cm.matrix[2, 1] == 1  # class-1 GT missed -> background row
    tp, fp = cm.tp_fp()
    assert tp[0] == 1


def test_plot_functions_write_files(tmp_path="/tmp/_t_plots"):
    os.makedirs(tmp_path, exist_ok=True)
    from yolo.utils.plotting import plot_confusion_matrix, plot_mc_curve, plot_pr_curve

    px = np.linspace(0, 1, 1000)
    py = np.random.rand(2, 1000)
    ap = np.random.rand(2, 10)
    prec = np.random.rand(2, 1000)
    names = {0: "a", 1: "b"}
    pr = plot_pr_curve(px, prec, ap, names, os.path.join(tmp_path, "pr.png"))
    f1 = plot_mc_curve(px, py, names, os.path.join(tmp_path, "f1.png"), ylabel="F1")
    cm = plot_confusion_matrix(np.random.rand(3, 3), names, os.path.join(tmp_path, "cm.png"))
    # matplotlib is installed in this env, so files should exist
    for f in (pr, f1, cm):
        assert f is not None and os.path.exists(f)


if __name__ == "__main__":
    test_ap_per_class_perfect_detector()
    test_ap_per_class_with_false_positives()
    test_smooth_preserves_length()
    test_confusion_matrix_counts()
    test_plot_functions_write_files()
    print("All metrics tests passed.")

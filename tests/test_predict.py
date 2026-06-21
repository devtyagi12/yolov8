"""Tests for prediction features: directory inference, save_txt/conf, feature maps.

Run:  python tests/test_predict.py   (or)   python -m pytest tests/test_predict.py -q
"""

import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo import YOLO
from yolo.engine.predictor import DetectionPredictor, Results


def _make_dir(root="/tmp/_t_pred_src", n=5):
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        im = np.full((320, 320, 3), 50, np.uint8)
        cv2.rectangle(im, (60, 60), (160, 160), (0, 0, 220), -1)
        cv2.imwrite(f"{root}/im{i}.jpg", im)
    return root


def test_directory_source_loads_all_images():
    root = _make_dir(n=5)
    images, paths = DetectionPredictor._load_source(root)
    assert len(images) == 5 and len(paths) == 5


def test_glob_source():
    root = _make_dir(n=3)
    images, _ = DetectionPredictor._load_source(f"{root}/*.jpg")
    assert len(images) == 3


def test_results_save_txt_format(tmp_path="/tmp/_t_pred_txt"):
    os.makedirs(tmp_path, exist_ok=True)
    im = np.zeros((100, 200, 3), np.uint8)
    import torch

    boxes = torch.tensor([[20.0, 30.0, 60.0, 70.0, 0.9, 1.0]])
    res = Results(im, "x.jpg", {0: "a", 1: "b"}, boxes)
    f = res.save_txt(os.path.join(tmp_path, "x.txt"), save_conf=True)
    line = open(f).read().strip().split()
    assert len(line) == 6 and line[0] == "1"  # cls cx cy w h conf
    cx, cy, w, h = map(float, line[1:5])
    assert abs(cx - 40 / 200) < 1e-3 and abs(cy - 50 / 100) < 1e-3  # normalised centre


def test_predict_directory_saves_outputs():
    src = _make_dir(n=4)
    out = "/tmp/_t_pred_out"
    shutil.rmtree(out, ignore_errors=True)
    m = YOLO("yolov8n", nc=3, verbose=False)
    res = m.predict(src, conf=0.001, imgsz=320, save=True, save_txt=True, save_conf=True,
                    visualize=True, save_dir=out)
    assert len(res) == 4
    assert len([f for f in os.listdir(out) if f.endswith(".jpg")]) == 4
    assert len(os.listdir(os.path.join(out, "labels"))) == 4
    # feature visualisation produced per-image stage grids
    feat_root = os.path.join(out, "features")
    assert os.path.isdir(feat_root) and len(os.listdir(feat_root)) == 4
    any_dir = os.path.join(feat_root, os.listdir(feat_root)[0])
    assert any(f.startswith("stage") and f.endswith(".png") for f in os.listdir(any_dir))


if __name__ == "__main__":
    test_directory_source_loads_all_images()
    test_glob_source()
    test_results_save_txt_format()
    test_predict_directory_saves_outputs()
    print("All predict tests passed.")

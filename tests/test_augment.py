"""Tests for augmentation, focusing on mosaic and label bookkeeping.

Run:  python tests/test_augment.py   (or)   python -m pytest tests/test_augment.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.data.augment import build_mosaic, xywhn2xyxy, xyxy2xywhn


def test_xywhn_xyxy_roundtrip():
    w, h = 640, 480
    xywhn = np.array([[0.5, 0.5, 0.4, 0.6], [0.25, 0.75, 0.2, 0.2]], np.float32)
    back = xyxy2xywhn(xywhn2xyxy(xywhn, w, h), w, h)
    assert np.allclose(back, xywhn, atol=1e-4)


def test_build_mosaic_shape_and_bounds():
    s = 320
    rng = np.random.default_rng(0)
    imgs, labels = [], []
    for _ in range(4):
        imgs.append((rng.random((s, s, 3)) * 255).astype(np.uint8))
        # one centred box covering the middle half of each image
        labels.append(np.array([[0, s * 0.25, s * 0.25, s * 0.75, s * 0.75]], np.float32))

    img, lab = build_mosaic(imgs, labels, s)
    assert img.shape == (s, s, 3)
    if lab.shape[0]:
        assert lab[:, 1:].min() >= 0 and lab[:, 1:].max() <= s  # boxes inside the crop
        assert ((lab[:, 3] - lab[:, 1]) > 0).all() and ((lab[:, 4] - lab[:, 2]) > 0).all()  # non-degenerate


def test_build_mosaic_preserves_a_full_object():
    """A box spanning a whole quadrant must survive somewhere in the mosaic."""
    s = 200
    imgs = [np.full((s, s, 3), 100, np.uint8) for _ in range(4)]
    # Each image fully occupied by a single object.
    labels = [np.array([[c, 5, 5, s - 5, s - 5]], np.float32) for c in range(4)]
    import random

    random.seed(1)
    img, lab = build_mosaic(imgs, labels, s)
    assert lab.shape[0] >= 1  # at least one object retained after centre-crop
    assert set(lab[:, 0].astype(int)).issubset({0, 1, 2, 3})


def test_dataset_mosaic_labels_normalised(tmp_path="/tmp/_t_mosaic"):
    import cv2

    from yolo.data.dataset import YOLODataset

    os.makedirs(f"{tmp_path}/images/train", exist_ok=True)
    os.makedirs(f"{tmp_path}/labels/train", exist_ok=True)
    for i in range(5):
        im = np.full((256, 256, 3), 40, np.uint8)
        cv2.rectangle(im, (60, 60), (190, 190), (0, 0, 220), -1)
        cv2.imwrite(f"{tmp_path}/images/train/i{i}.jpg", im)
        open(f"{tmp_path}/labels/train/i{i}.txt", "w").write("0 0.488 0.488 0.508 0.508\n")

    ds = YOLODataset(f"{tmp_path}/images/train", imgsz=256, augment=True, mosaic=1.0)
    for k in range(5):
        sample = ds[k]
        b = sample["bboxes"].numpy()
        assert sample["img"].shape == (3, 256, 256)
        if b.shape[0]:
            assert b.min() >= 0.0 and b.max() <= 1.0  # normalised
            assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all()  # positive w/h


if __name__ == "__main__":
    test_xywhn_xyxy_roundtrip()
    test_build_mosaic_shape_and_bounds()
    test_build_mosaic_preserves_a_full_object()
    test_dataset_mosaic_labels_normalised()
    print("All augmentation tests passed.")

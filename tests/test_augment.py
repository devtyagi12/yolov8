"""Tests for augmentation, focusing on mosaic and label bookkeeping.

Run:  python tests/test_augment.py   (or)   python -m pytest tests/test_augment.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.data.augment import (
    DEFAULT_HYP,
    Format,
    LetterBox,
    RandomFlip,
    RandomPerspective,
    build_mosaic,
    v8_transforms,
    xywhn2xyxy,
    xyxy2xywhn,
)


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


def test_letterbox_transform_scales_boxes():
    img = np.zeros((100, 200, 3), np.uint8)
    labels = {"img": img, "cls": np.array([[0.0]]), "bboxes": np.array([[10.0, 20.0, 60.0, 80.0]], np.float32)}
    out = LetterBox((640, 640), scaleup=True)(labels)
    assert out["img"].shape == (640, 640, 3)
    # box must stay inside the padded canvas and keep its aspect after scaling
    b = out["bboxes"][0]
    assert 0 <= b[0] < b[2] <= 640 and 0 <= b[1] < b[3] <= 640


def test_random_flip_horizontal_mirrors_boxes():
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], np.float32)
    labels = {"img": img, "cls": np.array([[1.0]]), "bboxes": boxes.copy()}
    out = RandomFlip(p=1.0, direction="horizontal")(labels)
    # x1' = W - x2, x2' = W - x1
    assert np.allclose(out["bboxes"][0], [100 - 30, 20, 100 - 10, 40])


def test_random_perspective_identity_keeps_boxes():
    # All ranges zero -> pure (near-)identity affine; boxes should be preserved.
    rng = np.random.default_rng(0)
    img = (rng.random((320, 320, 3)) * 255).astype(np.uint8)
    boxes = np.array([[40.0, 40.0, 120.0, 160.0], [200.0, 100.0, 260.0, 180.0]], np.float32)
    labels = {"img": img, "cls": np.array([[0.0], [1.0]]), "bboxes": boxes.copy()}
    tf = RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=0)
    out = tf(labels)
    assert out["img"].shape == (320, 320, 3)
    assert out["bboxes"].shape[0] == 2
    assert np.allclose(out["bboxes"], boxes, atol=1.0)


def test_v8_transforms_pipeline_outputs_tensors():
    import cv2
    import torch

    from yolo.data.dataset import YOLODataset

    root = "/tmp/_t_pipeline"
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    for i in range(6):
        im = np.full((256, 256, 3), 40, np.uint8)
        cv2.rectangle(im, (60, 60), (190, 190), (0, 0, 220), -1)
        cv2.imwrite(f"{root}/images/train/i{i}.jpg", im)
        open(f"{root}/labels/train/i{i}.txt", "w").write("0 0.488 0.488 0.508 0.508\n")

    hyp = {**DEFAULT_HYP, "mixup": 1.0, "degrees": 15, "shear": 8, "perspective": 0.0005, "flipud": 0.5, "copy_paste": 0.5}
    ds = YOLODataset(f"{root}/images/train", imgsz=256, augment=True, hyp=hyp)
    for k in range(6):
        s = ds[k]
        assert isinstance(s["img"], torch.Tensor) and s["img"].shape == (3, 256, 256)
        b = s["bboxes"].numpy()
        if b.shape[0]:
            assert b.min() >= 0.0 and b.max() <= 1.0
            assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all()


def test_close_mosaic_disables_mix():
    import cv2

    from yolo.data.dataset import YOLODataset

    root = "/tmp/_t_close"
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    for i in range(4):
        cv2.imwrite(f"{root}/images/train/i{i}.jpg", np.full((128, 128, 3), 50, np.uint8))
        open(f"{root}/labels/train/i{i}.txt", "w").write("0 0.5 0.5 0.3 0.3\n")
    ds = YOLODataset(f"{root}/images/train", imgsz=128, augment=True)
    assert ds.hyp["mosaic"] == 1.0
    ds.close_mosaic()
    assert ds.hyp["mosaic"] == 0.0 and ds.hyp["mixup"] == 0.0 and ds.hyp["copy_paste"] == 0.0
    _ = ds[0]  # still produces a valid sample


if __name__ == "__main__":
    test_xywhn_xyxy_roundtrip()
    test_build_mosaic_shape_and_bounds()
    test_build_mosaic_preserves_a_full_object()
    test_dataset_mosaic_labels_normalised()
    test_letterbox_transform_scales_boxes()
    test_random_flip_horizontal_mirrors_boxes()
    test_random_perspective_identity_keeps_boxes()
    test_v8_transforms_pipeline_outputs_tensors()
    test_close_mosaic_disables_mix()
    print("All augmentation tests passed.")

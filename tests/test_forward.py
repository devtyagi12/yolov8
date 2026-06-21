"""Fast smoke tests for the standalone YOLOv8 implementation.

Run with:  python -m pytest tests/ -q   (or)   python tests/test_forward.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo import YOLO
from yolo.nn.tasks import DetectionModel
from yolo.utils.checkpoint import load_checkpoint, remap_state_dict

# Official YOLOv8 parameter counts (sanity check the architecture).
EXPECTED_PARAMS = {"n": 3_157_200, "s": 11_166_560, "m": 25_902_640}


def test_param_counts_match_official():
    for scale, expected in EXPECTED_PARAMS.items():
        m = DetectionModel(scale, verbose=False)
        n = sum(p.numel() for p in m.parameters())
        assert n == expected, f"{scale}: got {n}, expected {expected}"


def test_forward_output_shape():
    m = DetectionModel("n", verbose=False).eval()
    y = m(torch.zeros(1, 3, 640, 640))
    out = y[0] if isinstance(y, tuple) else y
    assert out.shape == (1, 84, 8400)  # 4 box + 80 cls, 8400 anchors
    assert m.stride.tolist() == [8.0, 16.0, 32.0]


def test_checkpoint_roundtrip(tmp_path="/tmp"):
    """Save a model as an ultralytics-style {'model': obj} ckpt and reload bit-exactly."""
    src = DetectionModel("n", nc=80, verbose=False).eval()
    path = os.path.join(tmp_path, "rt.pt")
    torch.save({"model": src, "epoch": 1}, path)

    sd, meta = load_checkpoint(path)
    sd = remap_state_dict(sd)
    dst = DetectionModel("n", nc=80, verbose=False).eval()
    dst.load_state_dict_compat(sd, strict=True)

    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        a, b = src(x)[0], dst(x)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_predict_api_runs():
    m = YOLO("yolov8n", verbose=False)
    img = (np.random.rand(320, 480, 3) * 255).astype("uint8")
    res = m.predict(img, conf=0.001, imgsz=320)
    assert len(res) == 1
    assert res[0].plot().shape == img.shape


def test_loss_decreases_on_overfit():
    """One image, train mode: the loss must drop sharply within a few steps."""
    from yolo.data.build import build_dataloader
    from yolo.utils.loss import v8DetectionLoss

    # Build a 1-image dataset on the fly.
    import cv2

    root = "/tmp/_t_one"
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    im = np.full((320, 320, 3), 60, np.uint8)
    cv2.rectangle(im, (80, 90), (200, 210), (0, 0, 230), -1)
    cv2.imwrite(f"{root}/images/train/a.jpg", im)
    open(f"{root}/labels/train/a.txt", "w").write("0 0.4375 0.46875 0.375 0.375\n")

    torch.manual_seed(0)
    model = YOLO("yolov8n", nc=1, verbose=False).model.train()
    loader = build_dataloader(f"{root}/images/train", imgsz=320, batch=1, augment=False, workers=0, shuffle=False)
    batch = next(iter(loader))
    crit = v8DetectionLoss(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    first = None
    for step in range(40):
        loss, items = crit(model(batch["img"]), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(items.sum())
    last = float(items.sum())
    assert last < first * 0.5, f"loss did not drop: {first:.3f} -> {last:.3f}"


def test_corrupt_label_file_is_skipped():
    """A binary/NUL detection label is skipped (no crash); a bad line is dropped, valid kept."""
    import cv2

    from yolo.data.dataset import YOLODataset

    root = "/tmp/_t_det_corrupt"
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    for i in range(3):
        cv2.imwrite(f"{root}/images/train/i{i}.jpg", np.full((64, 64, 3), 50, np.uint8))
    open(f"{root}/labels/train/i0.txt", "w").write("0 0.5 0.5 0.3 0.3\n")            # valid
    open(f"{root}/labels/train/i1.txt", "wb").write(b"\x00\x00\x00\x00")              # corrupt
    open(f"{root}/labels/train/i2.txt", "w").write("0 0.5 0.5 0.3 0.3\nBAD a b\n")    # 1 bad line
    ds = YOLODataset(f"{root}/images/train", imgsz=64)
    assert ds.load_labels(0).shape[0] == 1   # valid
    assert ds.load_labels(1).shape[0] == 0   # corrupt -> skipped, no crash
    assert ds.load_labels(2).shape[0] == 1   # bad line dropped, valid kept


if __name__ == "__main__":
    test_param_counts_match_official()
    test_forward_output_shape()
    test_checkpoint_roundtrip()
    test_predict_api_runs()
    test_loss_decreases_on_overfit()
    test_corrupt_label_file_is_skipped()
    print("All smoke tests passed.")

"""Tests for training features: EMA, resume, cosine LR, device select, image cache.

Run:  python tests/test_training.py   (or)   python -m pytest tests/test_training.py -q
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.engine.trainer import DetectionTrainer
from yolo.nn.tasks import DetectionModel
from yolo.utils.torch_utils import ModelEMA, cosine_lr, linear_lr, select_device


def _make_ds(root="/tmp/_t_train"):
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    np.random.seed(0)
    for i in range(6):
        im = np.full((320, 320, 3), 50, np.uint8)
        cv2.rectangle(im, (60, 60), (160, 160), (0, 0, 220), -1)
        cv2.imwrite(f"{root}/images/train/i{i}.jpg", im)
        open(f"{root}/labels/train/i{i}.txt", "w").write("0 0.34 0.34 0.31 0.31\n")
    return f"{root}/images/train"


def test_select_device_cpu_and_multi():
    dev, ids = select_device("cpu")
    assert dev.type == "cpu" and ids == []
    dev, ids = select_device("0,1")  # no CUDA here -> cpu fallback
    assert dev.type == "cpu"


def test_lr_schedules():
    cl, ll = cosine_lr(0.01, 100), linear_lr(0.01, 100)
    for lf in (cl, ll):
        assert abs(lf(0) - 1.0) < 1e-6 and abs(lf(100) - 0.01) < 1e-6
    assert abs(cl(50) - 0.505) < 1e-2  # midpoint


def test_ema_tracks_model():
    torch.manual_seed(0)
    m = DetectionModel("n", nc=1, verbose=False)
    ema = ModelEMA(m)
    before = ema.ema.model[0].conv.weight.clone()
    # perturb the model, then update EMA -> EMA must move toward the new weights
    with torch.no_grad():
        m.model[0].conv.weight.add_(1.0)
    ema.update(m)
    assert ema.updates == 1
    assert not torch.allclose(ema.ema.model[0].conv.weight, before)


def test_train_ema_cache_save_and_resume():
    torch.manual_seed(0)
    path = _make_ds()
    m = DetectionModel("n", nc=1, verbose=False)
    t = DetectionTrainer(m, path, epochs=2, batch=3, imgsz=320, device="cpu", workers=0,
                         optimizer="SGD", warmup_epochs=1.0, ema=True, cos_lr=True, cache="ram",
                         close_mosaic=0, project="/tmp/_t_runs", save=True)
    t.train()
    assert t.ema is not None and t.ema.updates == 4  # 2 epochs * 2 batches
    assert os.path.exists("/tmp/_t_runs/last.pt") and os.path.exists("/tmp/_t_runs/best.pt")

    m2 = DetectionModel("n", nc=1, verbose=False)
    t2 = DetectionTrainer(m2, path, epochs=4, batch=3, imgsz=320, device="cpu", workers=0,
                          optimizer="SGD", warmup_epochs=1.0, ema=True, close_mosaic=0, save=False,
                          resume="/tmp/_t_runs/last.pt")
    assert t2.start_epoch == 2 and t2.ema.updates == 4  # state restored


def test_amp_disabled_on_cpu():
    m = DetectionModel("n", nc=1, verbose=False)
    t = DetectionTrainer(m, _make_ds(), epochs=1, batch=3, imgsz=320, device="cpu", workers=0,
                         amp=True, close_mosaic=0, save=False)
    assert t.amp is False  # AMP only enabled on CUDA


if __name__ == "__main__":
    test_select_device_cpu_and_multi()
    test_lr_schedules()
    test_ema_tracks_model()
    test_train_ema_cache_save_and_resume()
    test_amp_disabled_on_cpu()
    print("All training tests passed.")

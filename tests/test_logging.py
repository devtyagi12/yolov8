"""Tests for TensorBoard logging and training plots.

Run:  python tests/test_logging.py   (or)   python -m pytest tests/test_logging.py -q
"""

import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.engine.trainer import DetectionTrainer
from yolo.nn.tasks import DetectionModel
from yolo.utils.plotting import plot_label_histogram, plot_results


def _make_ds(root="/tmp/_t_log_ds"):
    os.makedirs(f"{root}/images/train", exist_ok=True)
    os.makedirs(f"{root}/labels/train", exist_ok=True)
    np.random.seed(0)
    for i in range(6):
        im = np.full((320, 320, 3), 50, np.uint8)
        cv2.rectangle(im, (60, 60), (160, 160), (0, 0, 220), -1)
        cv2.imwrite(f"{root}/images/train/i{i}.jpg", im)
        open(f"{root}/labels/train/i{i}.txt", "w").write("0 0.34 0.34 0.31 0.31\n")
    return f"{root}/images/train"


def test_plot_results_from_csv(tmp_path="/tmp/_t_log"):
    os.makedirs(tmp_path, exist_ok=True)
    csv = os.path.join(tmp_path, "results.csv")
    with open(csv, "w") as f:
        f.write("epoch,train/box,train/cls,metrics/mAP50\n")
        for e in range(1, 6):
            f.write(f"{e},{5 - e * 0.5},{4 - e * 0.4},{e * 0.1}\n")
    out = plot_results(csv)
    assert out and os.path.exists(out)


def test_label_histogram(tmp_path="/tmp/_t_log"):
    os.makedirs(tmp_path, exist_ok=True)
    out = plot_label_histogram([0, 0, 1, 2, 2, 2], {0: "a", 1: "b", 2: "c"}, os.path.join(tmp_path, "labels.png"))
    assert out and os.path.exists(out)


def test_trainer_writes_logs_and_plots():
    torch.manual_seed(0)
    proj = "/tmp/_t_log_run"
    shutil.rmtree(proj, ignore_errors=True)
    path = _make_ds()
    m = DetectionModel("n", nc=1, verbose=False)
    t = DetectionTrainer(m, path, data_val=path, epochs=3, batch=3, imgsz=320, device="cpu", workers=0,
                         optimizer="SGD", warmup_epochs=1.0, close_mosaic=0, save=False,
                         tensorboard=True, plots=True, project=proj)
    t.train()
    files = os.listdir(proj)
    assert "results.csv" in files and "results.png" in files and "labels.png" in files
    assert any(f.startswith("events.out.tfevents") for f in files)  # TensorBoard event file
    # results.csv has a header + 3 epoch rows
    lines = open(os.path.join(proj, "results.csv")).read().strip().splitlines()
    assert lines[0].startswith("epoch,") and len(lines) == 4


def test_per_epoch_train_and_val_visualizations():
    torch.manual_seed(0)
    proj = "/tmp/_t_log_viz"
    shutil.rmtree(proj, ignore_errors=True)
    path = _make_ds()
    m = DetectionModel("n", nc=1, verbose=False)
    t = DetectionTrainer(m, path, data_val=path, epochs=3, batch=3, imgsz=320, device="cpu", workers=0,
                         optimizer="SGD", warmup_epochs=1.0, close_mosaic=0, save=False,
                         tensorboard=False, plots=True, project=proj)
    t.train()
    # one train-batch image per epoch
    tb = sorted(os.listdir(os.path.join(proj, "train_batches")))
    assert tb == ["train_epoch001.jpg", "train_epoch002.jpg", "train_epoch003.jpg"]
    # a ground-truth + prediction image per validation
    vb = sorted(os.listdir(os.path.join(proj, "val_batches")))
    for e in (1, 2, 3):
        assert f"val_epoch{e:03d}_labels.jpg" in vb and f"val_epoch{e:03d}_pred.jpg" in vb


if __name__ == "__main__":
    test_plot_results_from_csv()
    test_label_histogram()
    test_trainer_writes_logs_and_plots()
    test_per_epoch_train_and_val_visualizations()
    print("All logging tests passed.")

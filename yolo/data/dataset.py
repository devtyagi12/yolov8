"""YOLO-format detection dataset.

Expects the standard layout::

    images/<split>/*.jpg
    labels/<split>/*.txt      # each line: "cls cx cy w h" (normalised 0-1)

The label path is derived from the image path by swapping ``/images/`` for
``/labels/`` and the extension for ``.txt`` (identical to the Ultralytics
convention), so existing YOLO datasets work unchanged.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.augment import augment_hsv, letterbox
from ..utils import LOGGER

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def img2label_path(img_path):
    """Map ``.../images/.../x.jpg`` to ``.../labels/.../x.txt`` (YOLO convention)."""
    p = Path(img_path)
    parts = p.as_posix().rsplit("/images/", 1)
    if len(parts) == 2:
        return f"{parts[0]}/labels/{parts[1]}".rsplit(".", 1)[0] + ".txt"
    return str(p.with_suffix(".txt"))


class YOLODataset(Dataset):
    """Detection dataset returning per-sample dicts ready for ``collate_fn``."""

    def __init__(self, path, imgsz=640, augment=False, hsv=(0.015, 0.7, 0.4), fliplr=0.5):
        self.imgsz = imgsz
        self.augment = augment
        self.hsv = hsv
        self.fliplr = fliplr
        self.im_files = self._gather_images(path)
        if not self.im_files:
            raise FileNotFoundError(f"No images found under {path!r}")
        self.label_files = [img2label_path(f) for f in self.im_files]
        LOGGER.info(f"YOLODataset: {len(self.im_files)} images from {path}")

    @staticmethod
    def _gather_images(path):
        path = Path(path)
        files = []
        if path.is_dir():
            files = [str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in IMG_EXTS]
        elif path.is_file() and path.suffix == ".txt":  # list file
            base = path.parent
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    p = (base / line) if not Path(line).is_absolute() else Path(line)
                    files.append(str(p))
        elif path.suffix.lower() in IMG_EXTS:
            files = [str(path)]
        return files

    def __len__(self):
        return len(self.im_files)

    def load_labels(self, idx):
        lf = self.label_files[idx]
        if Path(lf).is_file():
            with open(lf) as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines() if x], dtype=np.float32)
            if lb.size == 0:
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            lb = np.zeros((0, 5), dtype=np.float32)
        return lb  # (n, 5): cls, cx, cy, w, h (normalised)

    def __getitem__(self, idx):
        im = cv2.imread(self.im_files[idx])
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[idx]}")
        h0, w0 = im.shape[:2]
        labels = self.load_labels(idx).copy()

        if self.augment:
            augment_hsv(im, *self.hsv)
        im, ratio, pad = letterbox(im, self.imgsz, scaleup=self.augment)
        h, w = im.shape[:2]

        # Map normalised xywh (relative to original) into the letterboxed image, keep normalised.
        if labels.shape[0]:
            boxes = labels[:, 1:].copy()
            boxes[:, 0] = ratio[0] * w0 * (labels[:, 1] - labels[:, 3] / 2) + pad[0]  # x1
            boxes[:, 1] = ratio[1] * h0 * (labels[:, 2] - labels[:, 4] / 2) + pad[1]  # y1
            boxes[:, 2] = ratio[0] * w0 * (labels[:, 1] + labels[:, 3] / 2) + pad[0]  # x2
            boxes[:, 3] = ratio[1] * h0 * (labels[:, 2] + labels[:, 4] / 2) + pad[1]  # y2
            labels = labels.copy()
            labels[:, 1] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
            labels[:, 2] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
            labels[:, 3] = (boxes[:, 2] - boxes[:, 0]) / w
            labels[:, 4] = (boxes[:, 3] - boxes[:, 1]) / h

        if self.augment and self.fliplr and np.random.rand() < self.fliplr:
            im = np.fliplr(im)
            if labels.shape[0]:
                labels[:, 1] = 1 - labels[:, 1]

        img = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        labels = torch.from_numpy(labels)
        return {
            "img": torch.from_numpy(img).float() / 255.0,
            "cls": labels[:, 0:1] if labels.shape[0] else torch.zeros((0, 1)),
            "bboxes": labels[:, 1:5] if labels.shape[0] else torch.zeros((0, 4)),
            "im_file": self.im_files[idx],
            "ori_shape": (h0, w0),
        }

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([b["img"] for b in batch], 0)
        cls = torch.cat([b["cls"] for b in batch], 0)
        bboxes = torch.cat([b["bboxes"] for b in batch], 0)
        batch_idx = torch.cat([torch.full((b["cls"].shape[0],), i) for i, b in enumerate(batch)], 0)
        return {
            "img": imgs,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": batch_idx,
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
        }

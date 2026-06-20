"""YOLO-format detection dataset.

Expects the standard layout::

    images/<split>/*.jpg
    labels/<split>/*.txt      # each line: "cls cx cy w h" (normalised 0-1)

The label path is derived from the image path by swapping ``/images/`` for
``/labels/`` and the extension for ``.txt`` (identical to the Ultralytics
convention), so existing YOLO datasets work unchanged.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.augment import augment_hsv, build_mosaic, letterbox, xywhn2xyxy, xyxy2xywhn
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

    def __init__(self, path, imgsz=640, augment=False, hsv=(0.015, 0.7, 0.4), fliplr=0.5, mosaic=1.0):
        self.imgsz = imgsz
        self.augment = augment
        self.hsv = hsv
        self.fliplr = fliplr
        self.mosaic = mosaic  # probability of applying mosaic (only when augment=True)
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

    def load_image_resized(self, idx):
        """Load a BGR image resized so its long side equals ``imgsz`` plus pixel-xyxy labels."""
        im = cv2.imread(self.im_files[idx])
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[idx]}")
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (round(w0 * r), round(h0 * r)), interpolation=interp)
        h, w = im.shape[:2]

        labels = self.load_labels(idx)
        if labels.shape[0]:
            xyxy = xywhn2xyxy(labels[:, 1:], w, h)
            labels = np.concatenate([labels[:, 0:1], xyxy], 1)
        else:
            labels = np.zeros((0, 5), np.float32)
        return im, labels, (h0, w0)

    def load_mosaic(self, idx):
        """Build a 4-image mosaic centred on image ``idx``. Returns (BGR s×s img, pixel-xyxy labels)."""
        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        imgs, labels_list = [], []
        for index in indices:
            im, labels, _ = self.load_image_resized(index)
            imgs.append(im)
            labels_list.append(labels)
        return build_mosaic(imgs, labels_list, self.imgsz)

    def __getitem__(self, idx):
        use_mosaic = self.augment and self.mosaic and random.random() < self.mosaic
        if use_mosaic:
            im, labels_xyxy = self.load_mosaic(idx)  # s×s BGR, cls+xyxy pixel labels
            h, w = im.shape[:2]
            labels = np.zeros((labels_xyxy.shape[0], 5), np.float32)
            if labels_xyxy.shape[0]:
                labels[:, 0] = labels_xyxy[:, 0]
                labels[:, 1:] = xyxy2xywhn(labels_xyxy[:, 1:], w, h)
            ori_shape = (self.imgsz, self.imgsz)
        else:
            im, labels_xyxy, ori_shape = self.load_image_resized(idx)
            im, ratio, pad = letterbox(im, self.imgsz, scaleup=self.augment)
            h, w = im.shape[:2]
            labels = np.zeros((labels_xyxy.shape[0], 5), np.float32)
            if labels_xyxy.shape[0]:
                xyxy = labels_xyxy[:, 1:].copy()
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * ratio[0] + pad[0]
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * ratio[1] + pad[1]
                labels[:, 0] = labels_xyxy[:, 0]
                labels[:, 1:] = xyxy2xywhn(xyxy, w, h)

        if self.augment:
            augment_hsv(im, *self.hsv)
            if self.fliplr and np.random.rand() < self.fliplr:
                im = np.fliplr(im)
                if labels.shape[0]:
                    labels[:, 1] = 1 - labels[:, 1]

        img = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        labels = torch.from_numpy(np.ascontiguousarray(labels))
        return {
            "img": torch.from_numpy(img).float() / 255.0,
            "cls": labels[:, 0:1] if labels.shape[0] else torch.zeros((0, 1)),
            "bboxes": labels[:, 1:5] if labels.shape[0] else torch.zeros((0, 4)),
            "im_file": self.im_files[idx],
            "ori_shape": ori_shape,
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

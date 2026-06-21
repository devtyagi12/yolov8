"""YOLO-format detection dataset.

Expects the standard layout::

    images/<split>/*.jpg
    labels/<split>/*.txt      # each line: "cls cx cy w h" (normalised 0-1)

The label path is derived from the image path by swapping ``/images/`` for
``/labels/`` and the extension for ``.txt`` (identical to the Ultralytics
convention), so existing YOLO datasets work unchanged.

Augmentation is delegated to the transform pipeline in :mod:`yolo.data.augment`
(``v8_transforms``), matching the Ultralytics architecture.
"""

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..data.augment import DEFAULT_HYP, Compose, Format, LetterBox, v8_transforms, xywhn2xyxy
from ..utils import LOGGER, read_label_text

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

    def __init__(self, path, imgsz=640, augment=False, hyp=None, mosaic=None, cache=False):
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = {**DEFAULT_HYP, **(hyp or {})}
        if mosaic is not None:  # convenience override
            self.hyp["mosaic"] = mosaic
        self.im_files = self._gather_images(path)
        if not self.im_files:
            raise FileNotFoundError(f"No images found under {path!r}")
        self.label_files = [img2label_path(f) for f in self.im_files]
        # Image cache: False, "ram" (in-memory) or "disk" (.npy alongside images).
        self.cache = cache if cache in ("ram", "disk") else ("ram" if cache is True else False)
        self.ims = [None] * len(self.im_files) if self.cache == "ram" else None
        self.transforms = self.build_transforms()
        LOGGER.info(f"YOLODataset: {len(self.im_files)} images from {path}"
                    + (f" (cache={self.cache})" if self.cache else ""))

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _gather_images(path):
        path = Path(path)
        files = []
        if path.is_dir():
            files = [str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in IMG_EXTS]
        elif path.is_file() and path.suffix == ".txt":  # list file
            base = path.parent
            text = read_label_text(path)
            for line in (text or "").splitlines():
                line = line.strip()
                if line:
                    p = (base / line) if not Path(line).is_absolute() else Path(line)
                    files.append(str(p))
        elif path.suffix.lower() in IMG_EXTS:
            files = [str(path)]
        return files

    def build_transforms(self):
        """Build the augmentation pipeline (training) or a plain letterbox+format (val)."""
        if self.augment:
            return v8_transforms(self, self.imgsz, self.hyp)
        return Compose([LetterBox((self.imgsz, self.imgsz), scaleup=False), Format()])

    def close_mosaic(self):
        """Disable mosaic / mixup / copy-paste and rebuild the pipeline (for final epochs)."""
        self.hyp["mosaic"] = 0.0
        self.hyp["mixup"] = 0.0
        self.hyp["copy_paste"] = 0.0
        self.transforms = self.build_transforms()

    # ------------------------------------------------------------------ data
    def __len__(self):
        return len(self.im_files)

    def load_labels(self, idx):
        lf = self.label_files[idx]
        empty = np.zeros((0, 5), dtype=np.float32)
        if not Path(lf).is_file():
            return empty
        text = read_label_text(lf)  # None if the file is binary/corrupt (warned)
        if text is None:
            return empty
        rows, bad = [], 0
        for line in text.strip().splitlines():
            toks = line.split()
            if not toks:
                continue
            if len(toks) != 5:  # detection label: cls cx cy w h
                bad += 1
                continue
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                bad += 1
        if bad:
            LOGGER.warning(f"Skipped {bad} malformed line(s) in {lf}")
        return np.array(rows, dtype=np.float32) if rows else empty  # (n, 5)

    def _read_resized(self, idx):
        """Read + resize an image (long side = imgsz). Returns (im, (h0, w0))."""
        im = cv2.imread(self.im_files[idx])
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[idx]}")
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (round(w0 * r), round(h0 * r)), interpolation=interp)
        return im, (h0, w0)

    def load_cached_image(self, idx):
        """Return a (copy of a) resized image, using the RAM/disk cache when enabled."""
        if self.cache == "ram":
            if self.ims[idx] is None:
                self.ims[idx] = self._read_resized(idx)
            im, hw0 = self.ims[idx]
            return im.copy(), hw0
        if self.cache == "disk":
            npz = Path(self.im_files[idx]).with_suffix(".cache.npz")
            if npz.exists():
                data = np.load(str(npz))
                return data["im"], tuple(int(v) for v in data["hw0"])
            im, hw0 = self._read_resized(idx)
            np.savez(str(npz), im=im, hw0=np.array(hw0))
            return im, hw0
        return self._read_resized(idx)

    def load_image_resized(self, idx):
        """Load a BGR image resized so its long side equals ``imgsz`` plus pixel-xyxy labels."""
        im, (h0, w0) = self.load_cached_image(idx)
        h, w = im.shape[:2]

        labels = self.load_labels(idx)
        if labels.shape[0]:
            xyxy = xywhn2xyxy(labels[:, 1:], w, h)
            labels = np.concatenate([labels[:, 0:1], xyxy], 1)
        else:
            labels = np.zeros((0, 5), np.float32)
        return im, labels, (h0, w0)

    def get_image_and_label(self, idx):
        """Return a labels dict (img BGR, cls, xyxy bboxes) consumed by the transform pipeline."""
        im, labels, ori = self.load_image_resized(idx)
        return {
            "img": im,
            "cls": labels[:, 0:1] if labels.shape[0] else np.zeros((0, 1), np.float32),
            "bboxes": labels[:, 1:5] if labels.shape[0] else np.zeros((0, 4), np.float32),
            "ori_shape": ori,
            "resized_shape": im.shape[:2],
            "im_file": self.im_files[idx],
        }

    def __getitem__(self, idx):
        labels = self.get_image_and_label(idx)
        labels = self.transforms(labels)  # returns img/cls/bboxes as tensors (Format)
        return {
            "img": labels["img"],
            "cls": labels["cls"],
            "bboxes": labels["bboxes"],
            "im_file": labels.get("im_file", self.im_files[idx]),
            "ori_shape": labels.get("ori_shape", (self.imgsz, self.imgsz)),
        }

    # ------------------------------------------------------------------ batching
    @staticmethod
    def collate_fn(batch):
        import torch

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

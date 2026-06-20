"""Augmentation pipeline — a faithful, ultralytics-free port of YOLOv8's transforms.

The training pipeline mirrors Ultralytics' ``v8_transforms``::

    Mosaic -> CopyPaste -> RandomPerspective -> MixUp -> Albumentations
           -> RandomHSV -> RandomFlip(vertical) -> RandomFlip(horizontal) -> Format

Transforms operate on a ``labels`` dict carrying:
    img      : HWC BGR uint8 image
    cls      : (n, 1) float class ids
    bboxes   : (n, 4) absolute ``xyxy`` pixel boxes (normalised only by ``Format``)
    cls/bboxes stay in sync as boxes are added/removed.
"""

import math
import random

import cv2
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Default detection hyper-parameters (identical to Ultralytics' defaults).
# --------------------------------------------------------------------------- #
DEFAULT_HYP = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


# --------------------------------------------------------------------------- #
# Box helpers (numpy, absolute or normalised coordinates).
# --------------------------------------------------------------------------- #
def xywhn2xyxy(boxes, w, h, padw=0, padh=0):
    """Normalised ``[cx, cy, bw, bh]`` -> pixel ``[x1, y1, x2, y2]`` (with optional padding)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = w * (boxes[:, 0] - boxes[:, 2] / 2) + padw  # x1
    out[:, 1] = h * (boxes[:, 1] - boxes[:, 3] / 2) + padh  # y1
    out[:, 2] = w * (boxes[:, 0] + boxes[:, 2] / 2) + padw  # x2
    out[:, 3] = h * (boxes[:, 1] + boxes[:, 3] / 2) + padh  # y2
    return out


def xyxy2xywhn(boxes, w, h, clip=False, eps=0.0):
    """Pixel ``[x1, y1, x2, y2]`` -> normalised ``[cx, cy, bw, bh]`` relative to (w, h)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if clip:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - eps)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - eps)
    out = np.empty_like(boxes)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w  # cx
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h  # cy
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w  # bw
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h  # bh
    return out


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Filter boxes after a geometric transform (drop tiny / extreme-aspect / shrunk boxes).

    ``box1`` (before) and ``box2`` (after) are ``(4, n)`` arrays of ``xyxy``.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


# --------------------------------------------------------------------------- #
# Functional letterbox (used by the predictor) + a LetterBox transform class.
# --------------------------------------------------------------------------- #
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    """Resize and pad an image to ``new_shape`` preserving aspect ratio.

    Returns ``(image, (rw, rh), (dw, dh))``.
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    """Apply random HSV gain augmentation in-place to a BGR image (functional helper)."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)
    return im


def build_mosaic(imgs, labels_list, imgsz, center_range=(0.5, 1.5), fill=114, min_box=2):
    """Standalone 4-image mosaic with centre-crop to ``imgsz`` (kept for quick previews/tests).

    Returns ``(s×s image, (m, 5) cls+xyxy labels)``. The training pipeline instead uses the
    :class:`Mosaic` transform (uncropped) followed by :class:`RandomPerspective`.
    """
    img4, labels4, _ = _mosaic4_canvas(imgs, labels_list, imgsz, center_range, fill)
    off = imgsz // 2
    img_c = np.ascontiguousarray(img4[off : off + imgsz, off : off + imgsz])
    if labels4.shape[0]:
        labels4[:, [1, 3]] -= off
        labels4[:, [2, 4]] -= off
        np.clip(labels4[:, 1:], 0, imgsz, out=labels4[:, 1:])
        bw = labels4[:, 3] - labels4[:, 1]
        bh = labels4[:, 4] - labels4[:, 2]
        labels4 = labels4[(bw > min_box) & (bh > min_box)]
    return img_c, labels4


def _mosaic4_canvas(imgs, labels_list, imgsz, center_range=(0.5, 1.5), fill=114):
    """Assemble the uncropped ``2s×2s`` mosaic canvas. Returns ``(img4, labels_xyxy, (yc, xc))``."""
    assert len(imgs) == 4, "mosaic requires exactly 4 images"
    s = imgsz
    yc = int(random.uniform(s * center_range[0], s * center_range[1]))
    xc = int(random.uniform(s * center_range[0], s * center_range[1]))
    img4 = np.full((s * 2, s * 2, 3), fill, dtype=np.uint8)
    labels4 = []
    for i, (img, labels) in enumerate(zip(imgs, labels_list)):
        h, w = img.shape[:2]
        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw, padh = x1a - x1b, y1a - y1b
        if labels.shape[0]:
            lb = labels.copy().astype(np.float32)
            lb[:, [1, 3]] += padw
            lb[:, [2, 4]] += padh
            labels4.append(lb)
    labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5), np.float32)
    if labels4.shape[0]:
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
    return img4, labels4, (yc, xc)


def _split(labels):
    """Return ``(cls, bboxes)`` from a labels dict (both numpy, may be empty)."""
    cls = labels.get("cls", np.zeros((0, 1), np.float32))
    bboxes = labels.get("bboxes", np.zeros((0, 4), np.float32))
    return np.asarray(cls, np.float32).reshape(-1, 1), np.asarray(bboxes, np.float32).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# Transform classes.
# --------------------------------------------------------------------------- #
class Compose:
    """Chain a list of transforms."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, labels):
        for t in self.transforms:
            labels = t(labels)
        return labels

    def append(self, transform):
        self.transforms.append(transform)


class BaseMixTransform:
    """Base class for mix transforms (Mosaic / MixUp) that combine several images."""

    def __init__(self, dataset, pre_transform=None, p=0.0):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        if random.uniform(0, 1) > self.p:
            return labels
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            mix_labels = [self.pre_transform(d) for d in mix_labels]
        labels["mix_labels"] = mix_labels
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def get_indexes(self):
        raise NotImplementedError

    def _mix_transform(self, labels):
        raise NotImplementedError


class Mosaic(BaseMixTransform):
    """4-image mosaic. Outputs an uncropped ``2s×2s`` canvas; cropping is done by RandomPerspective."""

    def __init__(self, dataset, imgsz=640, p=1.0):
        super().__init__(dataset=dataset, pre_transform=None, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)

    def get_indexes(self):
        return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

    def _mix_transform(self, labels):
        mix = [labels] + labels["mix_labels"]
        imgs, lab_list = [], []
        for d in mix:
            cls, bboxes = _split(d)
            imgs.append(d["img"])
            lab_list.append(np.concatenate([cls, bboxes], 1) if cls.shape[0] else np.zeros((0, 5), np.float32))
        img4, labels4, _ = _mosaic4_canvas(imgs, lab_list, self.imgsz)
        labels["img"] = img4
        labels["cls"] = labels4[:, 0:1]
        labels["bboxes"] = labels4[:, 1:5]
        labels["resized_shape"] = img4.shape[:2]
        labels["mosaic_border"] = self.border
        return labels


class MixUp(BaseMixTransform):
    """Blend two images and concatenate their labels (Zhang et al., 2017)."""

    def __init__(self, dataset, pre_transform=None, p=0.0):
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"].astype(np.float32) * r + labels2["img"].astype(np.float32) * (1 - r)).astype(
            np.uint8
        )
        c1, b1 = _split(labels)
        c2, b2 = _split(labels2)
        labels["cls"] = np.concatenate([c1, c2], 0)
        labels["bboxes"] = np.concatenate([b1, b2], 0)
        return labels


class CopyPaste:
    """Box-level copy-paste: paste horizontally-flipped object crops that don't overlap existing boxes.

    The official transform relies on instance segmentation masks; this detection-only variant
    pastes rectangular box crops instead. Disabled by default (``p=0``), matching Ultralytics.
    """

    def __init__(self, p=0.0, iou_thr=0.30):
        self.p = p
        self.iou_thr = iou_thr

    def __call__(self, labels):
        cls, bboxes = _split(labels)
        if self.p == 0 or bboxes.shape[0] == 0 or random.uniform(0, 1) > self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        flipped = img[:, ::-1]
        new_cls, new_boxes = [], []
        for c, (x1, y1, x2, y2) in zip(cls[:, 0], bboxes):
            fx1, fx2 = w - x2, w - x1  # horizontally mirrored box position
            cand = np.array([fx1, y1, fx2, y2], np.float32)
            if bboxes.shape[0] and _ious(cand, bboxes).max() > self.iou_thr:
                continue
            xi1, yi1, xi2, yi2 = map(int, [max(fx1, 0), max(y1, 0), min(fx2, w), min(y2, h)])
            if xi2 - xi1 < 2 or yi2 - yi1 < 2:
                continue
            img[yi1:yi2, xi1:xi2] = flipped[yi1:yi2, xi1:xi2]
            new_cls.append(c)
            new_boxes.append([xi1, yi1, xi2, yi2])
        if new_boxes:
            labels["cls"] = np.concatenate([cls, np.array(new_cls, np.float32).reshape(-1, 1)], 0)
            labels["bboxes"] = np.concatenate([bboxes, np.array(new_boxes, np.float32)], 0)
        labels["img"] = img
        return labels


def _ious(box, boxes, eps=1e-9):
    """IoU of a single ``xyxy`` box against an ``(n, 4)`` array."""
    boxes = boxes.reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + eps)


class RandomPerspective:
    """Random affine / perspective transform (rotation, scale, shear, translation, perspective).

    When fed a mosaic, ``mosaic_border`` in the labels crops the ``2s`` canvas back to ``s``.
    Otherwise ``pre_transform`` (a LetterBox) first resizes a single image to ``imgsz``.
    """

    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        # Center.
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2
        # Perspective.
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        # Rotation and scale.
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        # Shear.
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        # Translation.
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]

        M = T @ S @ R @ P @ C
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # 4 corners
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def __call__(self, labels):
        if self.pre_transform is not None and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        border = labels.pop("mosaic_border", self.border)
        img = labels["img"]
        self.size = (img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2)  # (w, h)

        img, M, scale = self.affine_transform(img, border)
        cls, bboxes = _split(labels)
        new_boxes = self.apply_bboxes(bboxes.astype(np.float32), M)
        if new_boxes.shape[0]:
            new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, self.size[0])
            new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, self.size[1])
            keep = box_candidates(box1=bboxes.T * scale, box2=new_boxes.T, area_thr=0.10)
            new_boxes = new_boxes[keep]
            cls = cls[keep]

        labels["img"] = img
        labels["cls"] = cls
        labels["bboxes"] = new_boxes
        labels["resized_shape"] = img.shape[:2]
        return labels


class RandomHSV:
    """Random HSV gain augmentation."""

    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def __call__(self, labels):
        augment_hsv(labels["img"], self.hgain, self.sgain, self.vgain)
        return labels


class RandomFlip:
    """Random horizontal or vertical flip with bbox update."""

    def __init__(self, p=0.5, direction="horizontal"):
        assert direction in ("horizontal", "vertical")
        self.p = p
        self.direction = direction

    def __call__(self, labels):
        if random.uniform(0, 1) >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        cls, bboxes = _split(labels)
        if self.direction == "horizontal":
            img = np.ascontiguousarray(img[:, ::-1])
            if bboxes.shape[0]:
                x1 = bboxes[:, 0].copy()
                bboxes[:, 0] = w - bboxes[:, 2]
                bboxes[:, 2] = w - x1
        else:  # vertical
            img = np.ascontiguousarray(img[::-1])
            if bboxes.shape[0]:
                y1 = bboxes[:, 1].copy()
                bboxes[:, 1] = h - bboxes[:, 3]
                bboxes[:, 3] = h - y1
        labels["img"] = img
        labels["bboxes"] = bboxes
        return labels


class Albumentations:
    """Optional pixel-level augmentations via the ``albumentations`` package.

    Applies Blur / MedianBlur / ToGray / CLAHE (geometry-free, so boxes are untouched).
    If ``albumentations`` is not installed this is a no-op, exactly like Ultralytics.
    """

    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        try:
            import albumentations as A  # noqa: N812

            self.transform = A.Compose(
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_lower=75, p=0.0),
                ]
            )
        except ImportError:
            pass
        except Exception:
            self.transform = None

    def __call__(self, labels):
        if self.transform is not None and random.uniform(0, 1) < self.p:
            labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


class LetterBox:
    """Resize + pad an image (and its boxes) to a fixed shape, as a transform."""

    def __init__(self, new_shape=(640, 640), scaleup=True, center=True, color=(114, 114, 114)):
        self.new_shape = new_shape if isinstance(new_shape, (tuple, list)) else (new_shape, new_shape)
        self.scaleup = scaleup
        self.center = center
        self.color = color

    def __call__(self, labels):
        img = labels["img"]
        h, w = img.shape[:2]
        new_shape = self.new_shape
        r = min(new_shape[0] / h, new_shape[1] / w)
        if not self.scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.center:
            dw /= 2
            dh /= 2
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)

        cls, bboxes = _split(labels)
        if bboxes.shape[0]:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r + left
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r + top
        labels["img"] = img
        labels["bboxes"] = bboxes
        labels["cls"] = cls
        labels["resized_shape"] = img.shape[:2]
        return labels


class Format:
    """Final transform: image -> CHW RGB float tensor, boxes -> normalised ``xywh`` tensor."""

    def __call__(self, labels):
        img = labels["img"]
        h, w = img.shape[:2]
        cls, bboxes = _split(labels)
        nb = xyxy2xywhn(bboxes, w, h, clip=True, eps=1e-3) if bboxes.shape[0] else np.zeros((0, 4), np.float32)
        chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        labels["img"] = torch.from_numpy(chw).float() / 255.0
        labels["cls"] = torch.from_numpy(cls.astype(np.float32))
        labels["bboxes"] = torch.from_numpy(nb.astype(np.float32))
        return labels


def v8_transforms(dataset, imgsz, hyp):
    """Build the YOLOv8 detection training pipeline (mirrors Ultralytics' ``v8_transforms``)."""
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp["mosaic"])
    affine = RandomPerspective(
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        shear=hyp["shear"],
        perspective=hyp["perspective"],
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, CopyPaste(p=hyp["copy_paste"]), affine])
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp["mixup"]),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"]),
            RandomFlip(direction="vertical", p=hyp["flipud"]),
            RandomFlip(direction="horizontal", p=hyp["fliplr"]),
            Format(),
        ]
    )

"""Image augmentation / preprocessing utilities (no ultralytics dependency)."""

import random

import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    """Resize and pad an image to ``new_shape`` while preserving aspect ratio.

    Returns:
        (np.ndarray, (rh, rw), (dw, dh)): the padded image, the per-axis resize
        ratio, and the (left/right, top/bottom) padding applied.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down for better val mAP
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle padding to a multiple of stride
        dw, dh = dw % stride, dh % stride
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


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
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if clip:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - eps)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - eps)
    out = np.empty_like(boxes)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / w  # cx
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / h  # cy
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w  # bw
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h  # bh
    return out


def build_mosaic(imgs, labels_list, imgsz, center_range=(0.5, 1.5), fill=114, min_box=2):
    """Assemble a 4-image mosaic and return ``(mosaic_img, labels_xyxy)``.

    Args:
        imgs (list[np.ndarray]): four BGR images, each already resized so that its
            longest side equals ``imgsz``.
        labels_list (list[np.ndarray]): four ``(n, 5)`` arrays of ``cls, x1, y1, x2, y2``
            in each image's pixel coordinates.
        imgsz (int): output side length ``s``. A ``2s x 2s`` canvas is built and then
            centre-cropped to ``s x s`` (matching the Ultralytics default border).
    Returns:
        (np.ndarray, np.ndarray): the ``s x s`` mosaic and ``(m, 5)`` clipped labels
        (``cls, x1, y1, x2, y2``) in the cropped image's pixel coordinates.
    """
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

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # place image patch
        padw, padh = x1a - x1b, y1a - y1b  # offset of this image inside the mosaic

        if labels.shape[0]:
            lb = labels.copy().astype(np.float32)
            lb[:, [1, 3]] += padw
            lb[:, [2, 4]] += padh
            labels4.append(lb)

    labels4 = np.concatenate(labels4, 0) if labels4 else np.zeros((0, 5), np.float32)
    if labels4.shape[0]:
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    # Centre-crop the 2s mosaic down to s (Ultralytics uses border=(-s/2, -s/2)).
    off = s // 2
    img_c = np.ascontiguousarray(img4[off : off + s, off : off + s])
    if labels4.shape[0]:
        labels4[:, [1, 3]] -= off
        labels4[:, [2, 4]] -= off
        np.clip(labels4[:, 1:], 0, s, out=labels4[:, 1:])
        bw = labels4[:, 3] - labels4[:, 1]
        bh = labels4[:, 4] - labels4[:, 2]
        labels4 = labels4[(bw > min_box) & (bh > min_box)]  # drop boxes clipped away
    return img_c, labels4


def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    """Apply random HSV gain augmentation in-place to a BGR image."""
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

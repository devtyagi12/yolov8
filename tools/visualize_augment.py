#!/usr/bin/env python3
"""Visualise dataset augmentation (incl. mosaic) to verify labels track the pixels.

Pulls samples from a :class:`YOLODataset` with augmentation enabled, draws the
(possibly mosaicked) boxes back onto each image from the *post-augmentation*
normalised labels, and tiles them into a single grid image. If the boxes land on
the objects, the augmentation + label bookkeeping is correct.

Examples:
    python tools/visualize_augment.py --data datasets/mydata/images/train --n 8 --out aug.png
    python tools/visualize_augment.py --data datasets/mydata/images/train --no-mosaic --out plain.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.data.dataset import YOLODataset  # noqa: E402
from yolo.utils.plotting import COCO_NAMES, draw_normalized, image_grid  # noqa: E402


def tensor_to_bgr(img_tensor):
    """(3,H,W) RGB float tensor in [0,1] -> (H,W,3) BGR uint8."""
    arr = (img_tensor.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="images dir / list file")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=9, help="number of samples to render")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="augment_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mosaic", action="store_true", help="disable mosaic (show letterbox aug only)")
    ap.add_argument("--no-augment", action="store_true", help="disable all augmentation")
    ap.add_argument("--names", action="store_true", help="use COCO class names for labels")
    args = ap.parse_args()

    import random

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    augment = not args.no_augment
    mosaic = 0.0 if (args.no_mosaic or not augment) else 1.0
    ds = YOLODataset(args.data, imgsz=args.imgsz, augment=augment, mosaic=mosaic)
    names = COCO_NAMES if args.names else {}

    tiles, total_boxes = [], 0
    for k in range(args.n):
        sample = ds[k % len(ds)]
        bgr = tensor_to_bgr(sample["img"])
        cls = sample["cls"].numpy()
        boxes = sample["bboxes"].numpy()
        total_boxes += len(boxes)
        tile = draw_normalized(bgr, cls, boxes, names=names)
        cv2.putText(tile, f"#{k}  {len(boxes)} boxes", (6, 22), 0, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(tile)

    grid = image_grid(tiles, cols=args.cols)
    cv2.imwrite(args.out, grid)
    mode = "mosaic" if mosaic else ("augment" if augment else "raw")
    print(f"Wrote {args.out} ({args.n} {mode} samples, {total_boxes} boxes total, grid {grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualise the augmentation pipeline (mosaic, perspective, mixup, flips, HSV, ...).

Pulls samples from a :class:`YOLODataset` with augmentation enabled, draws the
*post-augmentation* boxes back onto each image from the normalised labels, and
tiles them into one grid. If the boxes land on the objects, the augmentation and
label bookkeeping are correct.

Examples:
    # default v8 pipeline (mosaic on, perspective, hsv, fliplr)
    python tools/visualize_augment.py --data datasets/mydata/images/train --n 9 --out aug.png

    # crank everything up to exercise every transform
    python tools/visualize_augment.py --data datasets/mydata/images/train \
        --degrees 20 --shear 10 --perspective 0.0005 --translate 0.2 --scale 0.7 \
        --mixup 1.0 --flipud 0.5 --copy-paste 0.5 --out aug_full.png

    python tools/visualize_augment.py --data ... --no-mosaic   # mosaic off (single-image aug)
    python tools/visualize_augment.py --data ... --no-augment  # raw letterbox, no aug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.data.augment import DEFAULT_HYP  # noqa: E402
from yolo.data.dataset import YOLODataset  # noqa: E402
from yolo.utils.plotting import COCO_NAMES, draw_normalized, image_grid  # noqa: E402


def tensor_to_bgr(img_tensor):
    """(3,H,W) RGB float tensor in [0,1] -> (H,W,3) BGR uint8."""
    arr = (img_tensor.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="images dir / list file")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=9, help="number of samples to render")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="augment_preview.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mosaic", action="store_true", help="disable mosaic")
    ap.add_argument("--no-augment", action="store_true", help="disable all augmentation")
    ap.add_argument("--names", action="store_true", help="use COCO class names for labels")
    # Per-augmentation hyper-parameter overrides (default to the v8 defaults).
    for k, v in DEFAULT_HYP.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None, help=f"override hyp.{k} (default {v})")
    args = ap.parse_args()

    import random

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    augment = not args.no_augment
    hyp = dict(DEFAULT_HYP)
    for k in DEFAULT_HYP:
        val = getattr(args, k)
        if val is not None:
            hyp[k] = val
    if args.no_mosaic:
        hyp["mosaic"] = 0.0

    ds = YOLODataset(args.data, imgsz=args.imgsz, augment=augment, hyp=hyp)
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
    active = [k for k in DEFAULT_HYP if hyp[k]] if augment else []
    print(f"Wrote {args.out} ({args.n} samples, {total_boxes} boxes, grid {grid.shape[1]}x{grid.shape[0]})")
    print(f"  augment={augment}  active_hyp={', '.join(active) if active else '(none)'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the polygon + distance YOLOv8 model on merged polygon datasets.

Datasets (YOLO layout, polygon points normalised, bbox derived from the polygon)::

    poly/images/train/*.jpg    poly/labels/train/*.txt    # "cls x1 y1 ... xn yn"
    polyd/images/train/*.jpg   polyd/labels/train/*.txt   # "cls x1 y1 ... xn yn distance"

Example:
    python examples/polygon_train.py \
        --poly poly/images/train --dist polyd/images/train \
        --val polyd/images/train --val-has-distance \
        --weights yolov8s.pt --nc 3 --epochs 100 --poly-batch 8 --dist-batch 8
"""

import argparse
import sys
from pathlib import Path
import torch
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.poly_model import YOLOPolygon


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poly", default=None, help="polygon-only images dir (optional)")
    ap.add_argument("--dist", default=None, help="polygon+distance images dir (optional)")
    ap.add_argument("--val", default=None, help="validation images dir")
    ap.add_argument("--val-has-distance", action="store_true")
    ap.add_argument("--weights", default="yolov8s", help="yolov8s or a .pt to seed box/cls weights")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--num-angles", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--poly-batch", type=int, default=8)
    ap.add_argument("--dist-batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not args.poly and not args.dist:
        ap.error("provide at least one of --poly / --dist (distance is optional)")

    model = YOLOPolygon(args.weights, nc=args.nc, num_angles=args.num_angles, device=args.device)
    model.train(
        poly_train=args.poly,
        dist_train=args.dist,
        val_data=args.val,
        val_has_distance=args.val_has_distance,
        epochs=args.epochs,
        poly_batch=args.poly_batch,
        dist_batch=args.dist_batch,
        imgsz=args.imgsz,
    )
    model.save("polygon_model.pt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train YOLOv8 detection on a YOLO-format dataset.

Dataset layout:
    <root>/images/train/*.jpg   <root>/labels/train/*.txt
    <root>/images/val/*.jpg     <root>/labels/val/*.txt

Each label line is ``cls cx cy w h`` with coordinates normalised to [0, 1].

Example:
    python examples/train.py --model yolov8n --nc 80 \
        --train datasets/coco/images/train --val datasets/coco/images/val \
        --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8n", help="scale name or .pt to fine-tune from")
    ap.add_argument("--nc", type=int, default=80, help="number of classes")
    ap.add_argument("--train", required=True, help="train images dir / list file")
    ap.add_argument("--val", default=None, help="val images dir / list file")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/train")
    args = ap.parse_args()

    model = YOLO(args.model, nc=args.nc, device=args.device)
    model.train(
        data_train=args.train,
        data_val=args.val,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        project=args.project,
    )


if __name__ == "__main__":
    main()

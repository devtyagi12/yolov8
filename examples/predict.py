#!/usr/bin/env python3
"""Run YOLOv8 detection inference on an image and save an annotated copy.

Examples:
    python examples/predict.py --weights yolov8n.pt --source bus.jpg
    python examples/predict.py --weights yolov8n --source bus.jpg   # random init
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yolov8n.pt", help="model name (yolov8n) or .pt file")
    ap.add_argument("--source", required=True, help="image path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="prediction.jpg")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    r = results[0]
    print(f"{len(r)} detection(s):")
    for d in r.summary():
        b = d["box"]
        print(f"  {d['name']:<15} {d['confidence']:.2f}  "
              f"[{b['x1']:.0f}, {b['y1']:.0f}, {b['x2']:.0f}, {b['y2']:.0f}]")
    r.save(args.out)
    print(f"Saved annotated image to {args.out}")


if __name__ == "__main__":
    main()

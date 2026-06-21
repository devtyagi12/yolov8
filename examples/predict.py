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
    ap.add_argument("--source", required=True, help="image, directory or glob pattern")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save", action="store_true", help="save annotated images")
    ap.add_argument("--save-txt", action="store_true", help="save YOLO-format labels")
    ap.add_argument("--save-conf", action="store_true", help="include confidence in labels")
    ap.add_argument("--show", action="store_true", help="display results (cv2.imshow)")
    ap.add_argument("--visualize", action="store_true", help="save backbone feature maps")
    ap.add_argument("--save-dir", default="runs/predict")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            save=args.save, save_txt=args.save_txt, save_conf=args.save_conf,
                            show=args.show, visualize=args.visualize, save_dir=args.save_dir)
    for r in results:
        print(f"{r.path}: {len(r)} detection(s)")
        for d in r.summary():
            b = d["box"]
            print(f"    {d['name']:<15} {d['confidence']:.2f}  "
                  f"[{b['x1']:.0f}, {b['y1']:.0f}, {b['x2']:.0f}, {b['y2']:.0f}]")


if __name__ == "__main__":
    main()

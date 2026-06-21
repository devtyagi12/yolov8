#!/usr/bin/env python3
"""Run polygon + distance inference and save an annotated image.

Output per detection: bounding box, class, confidence, distance, and polygon.

Example:
    python examples/polygon_predict.py --weights polygon_model.pt --nc 3 --source img.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from yolo.poly_model import YOLOPolygon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="trained polygon model .pt")
    ap.add_argument("--source", required=True, help="image path")
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--num-angles", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="polygon_prediction.jpg")
    args = ap.parse_args()

    model = YOLOPolygon("yolov8s", nc=args.nc, num_angles=args.num_angles)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.model.load_state_dict(sd, strict=False)

    results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz)
    r = results[0]
    print(f"{len(r)} detection(s):")
    for d in r.summary():
        b = d["box"]
        print(f"  {d['name']:<12} conf={d['confidence']:.2f} dist={d['distance']:.2f} "
              f"box=[{b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f}] poly_pts={len(d['polygon'])}")
    r.save(args.out)
    print(f"Saved annotated image to {args.out}")


if __name__ == "__main__":
    main()

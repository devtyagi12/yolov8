#!/usr/bin/env python3
"""Export / quantize a detection model.

Examples:
    python examples/export.py --weights yolov8n.pt --format torchscript
    python examples/export.py --weights yolov8n.pt --format onnx --dynamic
    python examples/export.py --weights yolov8n.pt --format int8     # INT8 ONNX (~4x smaller)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8n", help="model name or .pt file")
    ap.add_argument("--format", default="torchscript", choices=["torchscript", "onnx", "int8"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--dynamic", action="store_true", help="dynamic axes (onnx)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    kwargs = {}
    if args.format in ("onnx", "int8"):
        kwargs["opset"] = args.opset
    if args.format == "onnx":
        kwargs["dynamic"] = args.dynamic
    path = model.export(args.format, file=args.out, imgsz=args.imgsz, **kwargs)
    print(f"Exported to {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert an official Ultralytics ``yolov8*.pt`` into a clean, ultralytics-free file.

The output is a plain ``dict`` containing a ``model_state_dict`` (tensors only) plus
``names``/``nc`` metadata. It can be loaded by this package *and* by stock PyTorch
without the ``ultralytics`` package installed.

Usage:
    python tools/convert_weights.py yolov8n.pt yolov8n_clean.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.utils.checkpoint import load_checkpoint, remap_state_dict  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="path to an official yolov8*.pt checkpoint")
    ap.add_argument("dst", nargs="?", help="output path (default: <src stem>_clean.pt)")
    args = ap.parse_args()

    dst = args.dst or f"{Path(args.src).stem}_clean.pt"
    state_dict, meta = load_checkpoint(args.src)
    state_dict = remap_state_dict(state_dict)

    out = {
        "model_state_dict": {k: v.clone() for k, v in state_dict.items()},
        "names": meta.get("names"),
        "nc": meta.get("nc"),
    }
    torch.save(out, dst)
    print(f"Wrote {len(state_dict)} tensors to {dst} (nc={meta.get('nc')})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualise polygon datasets and the polygon augmentation pipeline.

Produces two grids:
  * ground truth   - raw polygon outline + derived bbox on each source image;
  * augmented      - the *star* polygon (decoded from the post-augmentation target)
                     + bbox, confirming polygons track through mosaic / perspective /
                     flips etc.

Also prints the per-object star calculation for the first ground-truth object.

Example:
    python tools/visualize_polygon.py --data poly/images/train --out-dir /tmp
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.data.poly_dataset import V8ParserExtended  # noqa: E402
from yolo.utils.plotting import Annotator, color_for, image_grid  # noqa: E402
from yolo.utils.poly_ops import bbox_from_polygon, polygon_to_star, split_star_targets  # noqa: E402


def draw_raw(im, polys_norm, classes, num_angles, angle_step):
    """Draw the raw polygon outline + derived bbox for each object."""
    ann = Annotator(np.ascontiguousarray(im))
    h, w = im.shape[:2]
    for poly_n, c in zip(polys_norm, classes):
        poly = poly_n.copy()
        poly[:, 0] *= w
        poly[:, 1] *= h
        col = color_for(c)
        cv2.polylines(ann.im, [poly.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        box = bbox_from_polygon(poly)
        ann.box_label(box, f"cls{c} ({len(poly)}v)", color=col)
    return ann.result()


def draw_star(im, star_targets, classes, num_angles):
    """Draw the star polygon (occupied bins, angular order) + origin from normalised targets."""
    import torch

    ann = Annotator(np.ascontiguousarray(im))
    h, w = im.shape[:2]
    for star, c in zip(star_targets, classes):
        origin, xy, conf = split_star_targets(torch.as_tensor(star), num_angles)
        col = color_for(int(c))
        ox, oy = float(origin[0]) * w, float(origin[1]) * h
        pts = []
        for b in range(num_angles):
            if conf[b] > 0.5:
                pts.append([float(xy[b, 0]) * w, float(xy[b, 1]) * h])
        if len(pts) >= 3:
            cv2.polylines(ann.im, [np.array(pts, np.int32)], True, col, 2, cv2.LINE_AA)
        cv2.circle(ann.im, (int(ox), int(oy)), 3, col, -1)
    return ann.result()


def tensor_to_bgr(t):
    arr = (t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--num-angles", type=int, default=24)
    ap.add_argument("--angle-step", type=int, default=15)
    ap.add_argument("--n-aug", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="/tmp")
    args = ap.parse_args()

    import random

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, step = args.num_angles, args.angle_step
    out = Path(args.out_dir)

    # --- ground truth grid (no augmentation) ---
    ds_raw = V8ParserExtended(args.data, imgsz=args.imgsz, augment=False, num_angles=N, angle_step=step)
    gt_tiles = []
    for i in range(len(ds_raw)):
        polys_norm, cls, _ = ds_raw.parse_label_file(i)
        im = cv2.resize(cv2.imread(ds_raw.im_files[i]), (args.imgsz, args.imgsz))
        gt_tiles.append(draw_raw(im, polys_norm, cls.astype(int), N, step))
    cv2.imwrite(str(out / "poly_gt.png"), image_grid(gt_tiles, cols=2))
    print(f"wrote {out / 'poly_gt.png'} ({len(gt_tiles)} images)")

    # --- augmented grid ---
    ds_aug = V8ParserExtended(args.data, imgsz=args.imgsz, augment=True, num_angles=N, angle_step=step,
                              hyp={"mosaic": 1.0, "degrees": 10, "scale": 0.5, "translate": 0.1, "fliplr": 0.5})
    aug_tiles = []
    for k in range(args.n_aug):
        s = ds_aug[k % len(ds_aug)]
        bgr = tensor_to_bgr(s["img"])
        aug_tiles.append(draw_star(bgr, s["poly"].numpy(), s["cls"].numpy().reshape(-1), N))
    cv2.imwrite(str(out / "poly_aug.png"), image_grid(aug_tiles, cols=2))
    print(f"wrote {out / 'poly_aug.png'} ({len(aug_tiles)} augmented samples)")


if __name__ == "__main__":
    main()

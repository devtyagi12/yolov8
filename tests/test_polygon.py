"""Tests for the polygon + distance extension.

Run:  python tests/test_polygon.py   (or)   python -m pytest tests/test_polygon.py -q
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.utils.poly_ops import (
    NO_DISTANCE,
    decode_polygons,
    encode_distance,
    poly_vector_len,
    polygon_to_star,
    split_star_targets,
    star_to_targets_torch,
)


def test_star_vector_length():
    assert poly_vector_len(24) == 2 + 3 * 24


def test_polygon_to_star_picks_max_distance_per_bin():
    # A square centred at origin -> 4 corners fall into 4 distinct 15-degree bins.
    center = np.array([0.0, 0.0])
    poly = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], np.float32)
    star = polygon_to_star(poly, center, num_angles=24, angle_step=15)
    assert star.shape[0] == 2 + 3 * 24
    _, xy, conf = split_star_targets(torch.from_numpy(star), 24)
    assert int(conf.sum()) == 4  # exactly four occupied bins
    # occupied vertices must be at distance sqrt(2)
    occ = xy[conf > 0]
    assert np.allclose(np.sqrt((occ.numpy() ** 2).sum(1)), math.sqrt(2), atol=1e-4)


def test_encode_distance_sentinel_and_log():
    assert encode_distance(None) == NO_DISTANCE
    assert encode_distance(-5) == NO_DISTANCE
    assert abs(encode_distance(10.0, 0.5, 100.0) - math.log(10.0)) < 1e-5
    assert abs(encode_distance(1000.0, 0.5, 100.0) - math.log(100.0)) < 1e-5  # clipped to max


def test_star_targets_roundtrip_angle_and_distance():
    # Build a star with a known vertex, recover dist + fractional angle.
    center = np.array([0.5, 0.5])
    # vertex at 30 degrees, distance 0.2 -> bin 2 (30/15), frac 0.0
    ang = math.radians(30.0 + 7.5)  # mid of bin 2 -> frac 0.5
    v = center + 0.2 * np.array([math.cos(ang), math.sin(ang)])
    star = polygon_to_star(np.array([v], np.float32), center, 24, 15)
    dist, frac, conf = star_to_targets_torch(torch.from_numpy(star)[None], 24, 15)
    b = int(torch.argmax(conf[0]))
    assert b == 2
    assert abs(float(dist[0, b]) - 0.2) < 1e-3
    assert abs(float(frac[0, b]) - 0.5) < 1e-2


def test_decode_polygons_inverts_angles():
    N = 24
    # zero dist -> vertices at origin; conf = sigmoid(0)=0.5
    origin = torch.tensor([[0.3, 0.4]])
    out = decode_polygons(torch.zeros(1, N), torch.zeros(1, N), torch.full((1, N), -10.0), origin, N)
    assert out.shape == (1, N, 3)
    assert torch.allclose(out[0, :, 0], origin[0, 0].expand(N), atol=1e-2)


def test_polygon_model_forward_shapes():
    from yolo.nn.poly_tasks import build_polygon_model

    m = build_polygon_model("s", nc=3, num_angles=24, verbose=False)
    m.eval()
    y, raw = m(torch.zeros(1, 3, 320, 320))
    nc, N = 3, 24
    assert y.shape[1] == 4 + nc + 1 + 3 * N  # box+cls+dist+3N poly channels
    feats, pc, pa, pd, dd = raw
    assert pc.shape[1] == N and dd.shape[1] == 1


def test_dataset_and_loss_step():
    import cv2

    from yolo.data.poly_dataset import build_merged_dataloader
    from yolo.nn.poly_tasks import build_polygon_model
    from yolo.utils.poly_loss import v8PolygonDistanceLoss

    root = "/tmp/_t_poly"
    for sub, wd in (("p", False), ("d", True)):
        os.makedirs(f"{root}/{sub}/images/train", exist_ok=True)
        os.makedirs(f"{root}/{sub}/labels/train", exist_ok=True)
        for i in range(4):
            im = np.full((256, 256, 3), 40, np.uint8)
            poly = np.array([[80, 70], [180, 80], [190, 180], [70, 175]], np.float32)
            cv2.fillPoly(im, [poly.astype(np.int32)], (0, 0, 220))
            cv2.imwrite(f"{root}/{sub}/images/train/i{i}.jpg", im)
            norm = poly.copy() / 256.0
            toks = ["0"] + [f"{v:.4f}" for v in norm.reshape(-1)]
            if wd:
                toks.append("12.5")
            open(f"{root}/{sub}/labels/train/i{i}.txt", "w").write(" ".join(toks) + "\n")

    loader, _ = build_merged_dataloader(f"{root}/p/images/train", f"{root}/d/images/train",
                                        imgsz=256, poly_batch=2, dist_batch=2, augment=True, workers=0)
    m = build_polygon_model("s", nc=1, num_angles=24, verbose=False).train()
    crit = v8PolygonDistanceLoss(m)
    batch = next(iter(loader))
    loss, items = crit(m(batch["img"]), batch)
    loss.backward()
    assert torch.isfinite(loss) and items.shape[0] == 5
    assert batch["poly"].shape[1] == poly_vector_len(24)


if __name__ == "__main__":
    test_star_vector_length()
    test_polygon_to_star_picks_max_distance_per_bin()
    test_encode_distance_sentinel_and_log()
    test_star_targets_roundtrip_angle_and_distance()
    test_decode_polygons_inverts_angles()
    test_polygon_model_forward_shapes()
    test_dataset_and_loss_step()
    print("All polygon tests passed.")

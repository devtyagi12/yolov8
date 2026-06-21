"""Polygon (star-shaped) and distance operations for the YOLOv8 polygon-distance head.

Coordinate convention
---------------------
All polygon geometry (origins, vertices, distances) is expressed in **normalised
image coordinates** in ``[0, 1]`` — the same space the dataset stores polygon
points in.  This keeps the training targets and the inference decode in one
consistent space; the predictor scales the final vertices back to the original
image (the "denormalise" step), which for a fixed input size is equivalent to the
per-level stride denormalisation described in the spec.

Star representation
-------------------
A polygon with an arbitrary number of vertices is encoded into a fixed-length
"star" vector anchored at the bounding-box centre::

    [origin_x, origin_y, x0, y0, conf0, x1, y1, conf1, ..., x_{N-1}, y_{N-1}, conf_{N-1}]

where ``N = 360 // angle_step`` angle bins.  For each bin we keep the polygon
vertex with the largest distance from the centre (``conf = 1``); empty bins store
``(0, 0, 0)``.
"""

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ANGLE_STEP = 15
DEFAULT_NUM_ANGLES = 360 // DEFAULT_ANGLE_STEP  # 24
DEFAULT_MIN_DISTANCE = 0.5
DEFAULT_MAX_DISTANCE = 100.0
NO_DISTANCE = -10.0  # sentinel for "no distance label"


def poly_vector_len(num_angles=DEFAULT_NUM_ANGLES):
    """Length of a star polygon target/prediction vector: ``2 + 3 * num_angles``."""
    return 2 + 3 * num_angles


def bbox_from_polygon(poly_xy):
    """Axis-aligned ``[x1, y1, x2, y2]`` bounding box of a ``(k, 2)`` polygon."""
    poly_xy = np.asarray(poly_xy, dtype=np.float32).reshape(-1, 2)
    if poly_xy.shape[0] == 0:
        return np.zeros(4, np.float32)
    x1, y1 = poly_xy.min(0)
    x2, y2 = poly_xy.max(0)
    return np.array([x1, y1, x2, y2], np.float32)


def polygon_to_star(poly_xy, center_xy, num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP):
    """Convert a polygon to the fixed-length star vector anchored at ``center_xy``.

    Args:
        poly_xy (np.ndarray): ``(k, 2)`` polygon vertices (same coord space as center).
        center_xy (np.ndarray): ``(2,)`` polygon centre (the bbox centre).
    Returns:
        np.ndarray: ``(2 + 3 * num_angles,)`` star vector.
    """
    poly_xy = np.asarray(poly_xy, dtype=np.float32).reshape(-1, 2)
    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    star = np.zeros((num_angles, 3), np.float32)  # x, y, conf per bin

    if poly_xy.shape[0]:
        d = poly_xy - center_xy
        dist = np.sqrt((d ** 2).sum(1))
        ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
        idx = np.floor(ang / angle_step).astype(np.int64) % num_angles
        for b in range(num_angles):
            m = idx == b
            if m.any():
                local = np.where(m)[0]
                sel = local[np.argmax(dist[m])]  # vertex with max distance in this bin
                star[b, 0] = poly_xy[sel, 0]
                star[b, 1] = poly_xy[sel, 1]
                star[b, 2] = 1.0
    return np.concatenate([center_xy, star.reshape(-1)]).astype(np.float32)


def encode_distance(distance, min_distance=DEFAULT_MIN_DISTANCE, max_distance=DEFAULT_MAX_DISTANCE):
    """``log(clip(distance))`` encoding; non-positive / missing values map to the sentinel."""
    if distance is None:
        return NO_DISTANCE
    distance = float(distance)
    if distance <= 0:
        return NO_DISTANCE
    return float(np.log(np.clip(distance, min_distance, max_distance)))


def decode_distance(pred, min_distance=DEFAULT_MIN_DISTANCE, max_distance=DEFAULT_MAX_DISTANCE):
    """Inverse of :func:`encode_distance`: ``clip(exp(pred))``."""
    return torch.clamp(torch.exp(pred), min=min_distance, max=max_distance)


def split_star_targets(star, num_angles=DEFAULT_NUM_ANGLES):
    """Split a ``(..., 2 + 3N)`` star tensor into ``(origin (...,2), xy (...,N,2), conf (...,N))``."""
    origin = star[..., :2]
    rest = star[..., 2:].reshape(*star.shape[:-1], num_angles, 3)
    xy = rest[..., :2]
    conf = rest[..., 2]
    return origin, xy, conf


def decode_polygons(poly_conf, poly_angle, poly_dist, origin_xy, num_angles=DEFAULT_NUM_ANGLES):
    """Decode raw polygon head outputs into vertices + confidence.

    Args:
        poly_conf/poly_angle/poly_dist (Tensor): ``(..., num_angles)`` raw head outputs.
        origin_xy (Tensor): ``(..., 2)`` polygon origin (predicted bbox centre, normalised).
    Returns:
        Tensor: ``(..., num_angles, 3)`` of ``[x, y, conf]`` in the origin's coord space.
    """
    dist = F.softplus(poly_dist)            # (..., N) >= 0
    frac = torch.sigmoid(poly_angle)        # (..., N) fractional bin position in [0, 1]
    conf = torch.sigmoid(poly_conf)         # (..., N) presence probability
    offsets = torch.arange(num_angles, device=poly_dist.device, dtype=poly_dist.dtype)
    ang_deg = (frac + offsets) / num_angles * 360.0
    ang_rad = torch.deg2rad(ang_deg)
    dx = dist * torch.cos(ang_rad)
    dy = dist * torch.sin(ang_rad)
    x = origin_xy[..., 0:1] + dx
    y = origin_xy[..., 1:2] + dy
    return torch.stack((x, y, conf), dim=-1)


def star_to_targets_torch(star, num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP, eps=1e-9):
    """From star targets compute per-bin target ``(dist, frac_angle, conf)`` tensors.

    The fractional angle for bin ``b`` is measured relative to that bin's base angle
    ``b * angle_step`` — i.e. the **slot index**, not a re-derived ``floor(angle/step)``.
    This keeps the target consistent with :func:`decode_polygons`, which uses the slot
    index as the angle base, even when a vertex sits exactly on a bin boundary.

    Args:
        star (Tensor): ``(M, 2 + 3N)`` target star vectors.
    Returns:
        (Tensor, Tensor, Tensor): ``dist (M, N)``, ``frac (M, N)``, ``conf (M, N)``.
    """
    origin, xy, conf = split_star_targets(star, num_angles)
    d = xy - origin.unsqueeze(-2)  # (M, N, 2)
    dist = torch.sqrt((d ** 2).sum(-1) + eps)
    ang = torch.rad2deg(torch.atan2(d[..., 1], d[..., 0])) % 360.0  # (M, N)
    slot = torch.arange(num_angles, device=star.device, dtype=ang.dtype)  # (N,)
    frac = (ang - slot * angle_step) / angle_step  # relative to the slot's base angle
    return dist, frac.clamp(0.0, 1.0), conf

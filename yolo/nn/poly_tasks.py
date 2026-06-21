"""Build a YOLOv8 detection model whose head is :class:`DetectPolygon`.

The backbone / neck and the box+class branches are identical to stock YOLOv8s, so
official ``yolov8s.pt`` weights load straight into them; only the polygon/distance
branches are new and trained from scratch.
"""

from copy import deepcopy

from ..cfg import get_cfg
from ..utils.poly_ops import DEFAULT_ANGLE_STEP, DEFAULT_NUM_ANGLES
from .tasks import DetectionModel


def polygon_cfg(scale="s", nc=80, num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP, num_dist_blocks=2):
    """Return a YOLOv8 config dict with the final Detect layer swapped for DetectPolygon."""
    cfg = get_cfg(scale=scale, nc=nc)
    cfg = deepcopy(cfg)
    head_layer = cfg["head"][-1]  # [[15, 18, 21], 1, "Detect", ["nc"]]
    assert head_layer[2] == "Detect", "expected the final head layer to be Detect"
    head_layer[2] = "DetectPolygon"
    head_layer[3] = ["nc", num_angles, angle_step, num_dist_blocks]
    cfg["num_angles"] = num_angles
    cfg["angle_step"] = angle_step
    return cfg


def build_polygon_model(
    scale="s", nc=80, num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP, num_dist_blocks=2, verbose=True
):
    """Construct a :class:`DetectionModel` with a :class:`DetectPolygon` head."""
    cfg = polygon_cfg(scale, nc, num_angles, angle_step, num_dist_blocks)
    model = DetectionModel(cfg, nc=nc, verbose=verbose)
    model.num_angles = num_angles
    model.angle_step = angle_step
    return model

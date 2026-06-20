"""Model configuration for YOLOv8 detection.

The ``YOLOV8`` dict mirrors the official ``yolov8.yaml`` architecture definition.
``scales`` provides the depth / width / max-channel multipliers that turn the
base topology into the n / s / m / l / x variants.
"""

# [depth_multiple, width_multiple, max_channels]
SCALES = {
    "n": [0.33, 0.25, 1024],
    "s": [0.33, 0.50, 1024],
    "m": [0.67, 0.75, 768],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.25, 512],
}

# Each layer: [from, repeats, module, args]
YOLOV8 = {
    "nc": 80,
    "scales": SCALES,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],      # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],     # 1-P2/4
        [-1, 3, "C2f", [128, True]],      # 2
        [-1, 1, "Conv", [256, 3, 2]],     # 3-P3/8
        [-1, 6, "C2f", [256, True]],      # 4
        [-1, 1, "Conv", [512, 3, 2]],     # 5-P4/16
        [-1, 6, "C2f", [512, True]],      # 6
        [-1, 1, "Conv", [1024, 3, 2]],    # 7-P5/32
        [-1, 3, "C2f", [1024, True]],     # 8
        [-1, 1, "SPPF", [1024, 5]],       # 9
    ],
    "head": [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 10
        [[-1, 6], 1, "Concat", [1]],                   # 11
        [-1, 3, "C2f", [512]],                         # 12
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 13
        [[-1, 4], 1, "Concat", [1]],                   # 14
        [-1, 3, "C2f", [256]],                         # 15 (P3/8-small)
        [-1, 1, "Conv", [256, 3, 2]],                  # 16
        [[-1, 12], 1, "Concat", [1]],                  # 17
        [-1, 3, "C2f", [512]],                         # 18 (P4/16-medium)
        [-1, 1, "Conv", [512, 3, 2]],                  # 19
        [[-1, 9], 1, "Concat", [1]],                   # 20
        [-1, 3, "C2f", [1024]],                        # 21 (P5/32-large)
        [[15, 18, 21], 1, "Detect", ["nc"]],           # 22 Detect(P3, P4, P5)
    ],
}


def get_cfg(scale="n", nc=80):
    """Return a deep-copied YOLOv8 config dict with the chosen scale and class count."""
    import copy

    cfg = copy.deepcopy(YOLOV8)
    cfg["nc"] = nc
    cfg["scale"] = scale
    return cfg

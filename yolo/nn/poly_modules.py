"""Polygon + distance detection head.

``DetectPolygon`` extends the standard :class:`~yolo.nn.modules.Detect` head — the
box (``cv2``) and class (``cv3``) branches are kept exactly as in stock YOLOv8 (so
official weights still load for them) — and adds:

* three polygon heads (``poly_conf`` / ``poly_angle`` / ``poly_dist``), each a
  1x1 conv-BN producing ``num_angles`` channels, fed from the **2nd-last layer of
  the box branch** (``cv2[i][:2]``);
* a per-anchor distance head (``dist``): ``num_dist_blocks`` 3x3 conv-BN layers + a
  1x1 conv-BN with a single output channel, fed from the level feature (same input
  as the box/class heads).

Output channel layout of the inference tensor (per anchor)::

    [ box(4) | cls(nc) | dist(1) | poly_conf(N) | poly_angle(N) | poly_dist(N) ]
"""

import torch
import torch.nn as nn

from ..utils.poly_ops import DEFAULT_ANGLE_STEP, DEFAULT_NUM_ANGLES
from .modules import Conv, Detect


class DetectPolygon(Detect):
    """YOLOv8 detection head augmented with polygon (star) and distance branches."""

    def __init__(self, nc=80, ch=(), num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP, num_dist_blocks=2):
        super().__init__(nc, ch)
        self.num_angles = num_angles
        self.angle_step = angle_step
        self.num_dist_blocks = num_dist_blocks

        c2 = max((16, ch[0] // 4, self.reg_max * 4))  # box-branch hidden channels (matches Detect)
        # Polygon heads take the 2nd-last box-branch feature (channels c2). 1x1 conv-BN, no activation.
        self.poly_conf = nn.ModuleList(Conv(c2, num_angles, 1, act=False) for _ in ch)
        self.poly_angle = nn.ModuleList(Conv(c2, num_angles, 1, act=False) for _ in ch)
        self.poly_dist = nn.ModuleList(Conv(c2, num_angles, 1, act=False) for _ in ch)
        # Per-anchor distance head: num_dist_blocks 3x3 conv-BN + 1x1 conv-BN(1).
        self.dist = nn.ModuleList(
            nn.Sequential(*[Conv(x, x, 3) for _ in range(num_dist_blocks)], Conv(x, 1, 1, act=False)) for x in ch
        )

    def forward(self, x):
        feats, pcs, pas, pds, dds = [], [], [], [], []
        for i in range(self.nl):
            box_feat = self.cv2[i][:2](x[i])          # 2nd-last box-branch feature (B, c2, H, W)
            box_out = self.cv2[i][2](box_feat)        # (B, 4*reg_max, H, W)
            cls_out = self.cv3[i](x[i])               # (B, nc, H, W)
            feats.append(torch.cat((box_out, cls_out), 1))
            pcs.append(self.poly_conf[i](box_feat))
            pas.append(self.poly_angle[i](box_feat))
            pds.append(self.poly_dist[i](box_feat))
            dds.append(self.dist[i](x[i]))

        b = x[0].shape[0]
        poly_conf = torch.cat([p.view(b, self.num_angles, -1) for p in pcs], 2)   # (B, N, A)
        poly_angle = torch.cat([p.view(b, self.num_angles, -1) for p in pas], 2)
        poly_dist = torch.cat([p.view(b, self.num_angles, -1) for p in pds], 2)
        dist = torch.cat([d.view(b, 1, -1) for d in dds], 2)                       # (B, 1, A)
        extras = (poly_conf, poly_angle, poly_dist, dist)

        if self.training:
            return (feats, *extras)

        y = self._inference(feats)  # decoded (B, 4 + nc, A)
        out = torch.cat((y, dist, poly_conf, poly_angle, poly_dist), 1)  # append raw extras
        return out, (feats, *extras)

    def bias_init(self):
        super().bias_init()  # initialise box/cls biases as in Detect
        for m in self.poly_conf:  # start polygon-presence logits slightly negative
            if hasattr(m, "bn"):
                nn.init.constant_(m.bn.bias, -2.0)

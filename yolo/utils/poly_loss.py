"""Loss for the polygon + distance head.

Extends :class:`~yolo.utils.loss.v8DetectionLoss` (box CIoU + DFL + class BCE) with:

* a **polygon loss** with three parts — per-bin distance (MSE on ``softplus``),
  fractional angle (BCE on ``sigmoid``) and presence confidence (BCE-with-logits);
* a per-anchor **distance loss** (L1 on the log-distance, only for objects that
  carry a valid distance label).

All terms are computed only on foreground anchors (from the Task-Aligned Assigner);
the polygon distance/angle terms are additionally masked by the per-bin target
presence (``conf``).  Targets for a foreground anchor are taken from its assigned
ground-truth object (``target_gt_idx``).
"""

import torch
import torch.nn.functional as F

from .loss import v8DetectionLoss
from .ops import make_anchors
from .poly_ops import NO_DISTANCE, star_to_targets_torch


class v8PolygonDistanceLoss(v8DetectionLoss):
    """YOLOv8 detection loss extended with polygon (star) and distance losses."""

    def __init__(
        self,
        model,
        tal_topk=10,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        poly_gain=0.1,
        dist_gain=0.1,
        poly_dist_gain=2.0,
        poly_conf_gain=0.2,
        poly_angle_gain=0.5,
    ):
        super().__init__(model, tal_topk=tal_topk, box=box, cls=cls, dfl=dfl)
        head = model.model[-1]
        self.num_angles = head.num_angles
        self.angle_step = head.angle_step
        self.poly_gain = poly_gain
        self.dist_gain = dist_gain
        self.poly_dist_gain = poly_dist_gain
        self.poly_conf_gain = poly_conf_gain
        self.poly_angle_gain = poly_angle_gain

    @staticmethod
    def _pad_per_image(values, batch_idx, batch_size, n_max):
        """Pad a flat ``(total, D)`` tensor into ``(B, n_max, D)`` grouped by image index."""
        d = values.shape[1]
        out = torch.zeros(batch_size, n_max, d, device=values.device, dtype=values.dtype)
        for j in range(batch_size):
            m = batch_idx == j
            nj = int(m.sum())
            if nj:
                out[j, :nj] = values[m][:n_max]
        return out

    def __call__(self, preds, batch):
        feats = preds[0]
        poly_conf, poly_angle, poly_dist, dist_pred = preds[1], preds[2], preds[3], preds[4]
        device = self.device

        loss = torch.zeros(5, device=device)  # box, cls, dfl, polygon, distance

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        n_max = targets.shape[1]
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # cls

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            loss[3], loss[4] = self._poly_dist_loss(
                batch, poly_conf, poly_angle, poly_dist, dist_pred, fg_mask, target_gt_idx, n_max, batch_size
            )

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["cls"]
        loss[2] *= self.hyp["dfl"]
        loss[3] *= self.poly_gain
        loss[4] *= self.dist_gain
        return loss.sum() * batch_size, loss.detach()

    def _poly_dist_loss(self, batch, poly_conf, poly_angle, poly_dist, dist_pred, fg_mask, target_gt_idx, n_max, batch_size):
        """Return ``(polygon_loss, distance_loss)`` for the foreground anchors."""
        device = self.device
        N = self.num_angles
        batch_idx = batch["batch_idx"].to(device)

        poly_t = self._pad_per_image(batch["poly"].to(device), batch_idx, batch_size, n_max)        # (B, n_max, P)
        dist_t = self._pad_per_image(batch["distance"].to(device).view(-1, 1), batch_idx, batch_size, n_max)[..., 0]

        flat_idx = (target_gt_idx + torch.arange(batch_size, device=device)[:, None] * n_max)
        poly_tgt = poly_t.view(-1, poly_t.shape[-1])[flat_idx]   # (B, A, P)
        dist_tgt = dist_t.reshape(-1)[flat_idx]                  # (B, A)

        fg = fg_mask.bool()
        num_obj = fg.sum().clamp(min=1)

        # --- polygon targets for foreground anchors ---
        poly_tgt_fg = poly_tgt[fg]                               # (Nfg, P)
        t_dist, t_frac, t_conf = star_to_targets_torch(poly_tgt_fg, N, self.angle_step)  # (Nfg, N)

        pc = poly_conf.permute(0, 2, 1)[fg]                      # (Nfg, N)
        pa = poly_angle.permute(0, 2, 1)[fg]
        pd = poly_dist.permute(0, 2, 1)[fg]

        denom = N * num_obj
        poly_mask = t_conf  # per-bin presence mask
        # distance: MSE between target dist and softplus(pred), masked by presence
        loss_pdist = (poly_mask * (F.softplus(pd) - t_dist) ** 2).sum() / denom
        # angle: BCE between fractional target angle and sigmoid(pred), masked by presence
        bce_angle = F.binary_cross_entropy(torch.sigmoid(pa), t_frac, reduction="none")
        loss_pangle = (poly_mask * bce_angle).sum() / denom
        # confidence: BCE-with-logits over all bins (foreground only)
        loss_pconf = F.binary_cross_entropy_with_logits(pc, t_conf, reduction="none").sum() / denom

        polygon_loss = (
            self.poly_dist_gain * loss_pdist
            + self.poly_conf_gain * loss_pconf
            + self.poly_angle_gain * loss_pangle
        )

        # --- per-anchor distance head (only valid distances) ---
        dd = dist_pred.permute(0, 2, 1)[fg][:, 0]               # (Nfg,)
        dist_tgt_fg = dist_tgt[fg]
        valid = dist_tgt_fg > (NO_DISTANCE + 1.0)
        if valid.any():
            distance_loss = F.l1_loss(dd[valid], dist_tgt_fg[valid], reduction="sum") / valid.sum()
        else:
            distance_loss = torch.zeros((), device=device)

        return polygon_loss, distance_loss

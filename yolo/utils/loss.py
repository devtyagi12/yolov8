"""YOLOv8 detection loss (box CIoU + DFL + classification BCE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .ops import bbox2dist, dist2bbox, make_anchors
from .tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    """Distribution Focal Loss criterion."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Combined IoU + DFL regression loss."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Criterion class computing the full YOLOv8 detection training loss."""

    def __init__(self, model, tal_topk=10, box=7.5, cls=0.5, dfl=1.5):
        device = next(model.parameters()).device
        m = model.model[-1]  # Detect head
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = {"box": box, "cls": cls, "dfl": dfl}
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Group flat (image_idx, cls, x, y, w, h) targets into (b, n_max, 5) padded tensors."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
            return out
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy_scaled(out[..., 1:5], scale_tensor)
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Compute the loss given raw model predictions and a batch dict.

        ``batch`` must contain ``batch_idx`` (n,), ``cls`` (n,1), ``bboxes`` (n,4 xywh
        normalised) and ``img`` (b, 3, H, W). Returns ``(total_loss, loss_items)``.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
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

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["cls"]
        loss[2] *= self.hyp["dfl"]
        return loss.sum() * batch_size, loss.detach()


def xywh2xyxy_scaled(x, scale_tensor):
    """Convert normalised xywh boxes to absolute xyxy using ``scale_tensor`` (w,h,w,h)."""
    y = x.clone()
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y * scale_tensor

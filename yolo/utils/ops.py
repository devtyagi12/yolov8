"""Tensor / box operations used across the model, loss and inference code.

All functions are written in plain PyTorch so the package has no dependency on
the ``ultralytics`` library.
"""

import math

import torch
import torchvision


def autopad(k, p=None, d=1):
    """Return 'same'-shape padding for a given kernel size ``k`` and dilation ``d``."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    """Round ``x`` up to the nearest multiple of ``divisor``."""
    return math.ceil(x / divisor) * divisor


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor center points and matching stride tensors from feature maps."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2], feats[i].shape[3]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # x grid
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # y grid
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Decode predicted distances (l, t, r, b) relative to anchor points into boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Encode xyxy bounding boxes into (l, t, r, b) distances for DFL targets."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def xywh2xyxy(x):
    """Convert ``[x, y, w, h]`` (center) boxes to ``[x1, y1, x2, y2]`` (corners)."""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else x.copy()
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh  # top-left
    y[..., 2:] = xy + wh  # bottom-right
    return y


def xyxy2xywh(x):
    """Convert ``[x1, y1, x2, y2]`` (corners) boxes to ``[x, y, w, h]`` (center)."""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def clip_boxes(boxes, shape):
    """Clip xyxy boxes in-place to image ``shape`` (height, width)."""
    boxes[..., 0].clamp_(0, shape[1])  # x1
    boxes[..., 1].clamp_(0, shape[0])  # y1
    boxes[..., 2].clamp_(0, shape[1])  # x2
    boxes[..., 3].clamp_(0, shape[0])  # y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """Rescale xyxy boxes from ``img1_shape`` (model input) to ``img0_shape`` (original)."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        boxes[..., 2] -= pad[0]
        boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    nc=0,
    max_wh=7680,
    max_nms=30000,
):
    """Run Non-Maximum Suppression on raw model predictions.

    Args:
        prediction (Tensor): shape (batch, 4 + nc, num_anchors).
    Returns:
        list[Tensor]: per-image (num_boxes, 6) tensors of ``[x1, y1, x2, y2, conf, cls]``.
    """
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"
    if isinstance(prediction, (list, tuple)):  # model in eval mode may output (inference, loss)
        prediction = prediction[0]

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of extra (mask) channels
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    prediction = prediction.transpose(-1, -2)  # (bs, anchors, 4+nc)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence filter
        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:  # sort by confidence, keep top boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # class offset
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return output

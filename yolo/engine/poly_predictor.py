"""Inference for the polygon + distance model.

Pipeline: preprocess (letterbox) -> forward -> NMS on the bounding boxes (carrying
the distance and polygon channels as attributes) -> decode polygon vertices &
distance -> scale boxes and polygons back to the original image.

Each result row is ``bbox, cls, conf, distance, polygon`` (the requested output
format), where ``polygon`` is an ``(num_angles, 3)`` array of ``[x, y, conf]`` in
original-image pixels (vertices with ``conf < poly_conf_thr`` are dropped).
"""

import cv2
import numpy as np
import torch
import torchvision

from ..data.augment import letterbox
from ..utils.ops import clip_boxes, xywh2xyxy
from ..utils.plotting import COCO_NAMES
from ..utils.poly_ops import (
    DEFAULT_ANGLE_STEP,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_NUM_ANGLES,
    decode_distance,
    decode_polygons,
)


class PolyResults:
    """Detections of a single image, each carrying a polygon and a distance."""

    def __init__(self, orig_img, path, names, boxes=None, dists=None, polys=None):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = str(path)
        self.names = names
        self.boxes = boxes  # (n, 6) x1y1x2y2,conf,cls
        self.dists = dists  # (n,)
        self.polys = polys  # (n, num_angles, 3) -> x, y, conf

    def __len__(self):
        return 0 if self.boxes is None else len(self.boxes)

    def summary(self):
        out = []
        if not len(self):
            return out
        for i, (*xyxy, conf, cls) in enumerate(self.boxes.cpu().numpy()):
            c = int(cls)
            poly = self.polys[i].cpu().numpy()
            verts = poly[poly[:, 2] >= 0.5][:, :2]  # keep confident vertices
            out.append(
                {
                    "name": self.names.get(c, f"class{c}"),
                    "class": c,
                    "confidence": float(conf),
                    "box": {"x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3])},
                    "distance": float(self.dists[i]),
                    "polygon": verts.tolist(),
                }
            )
        return out

    def plot(self, poly_conf_thr=0.5):
        """Draw boxes, distances and polygon outlines onto a copy of the image."""
        from ..utils.plotting import Annotator, color_for

        ann = Annotator(np.ascontiguousarray(self.orig_img))
        if len(self):
            boxes = self.boxes.cpu().numpy()
            for i, (*xyxy, conf, cls) in enumerate(boxes):
                c = int(cls)
                label = f"{self.names.get(c, f'class{c}')} {conf:.2f} d={self.dists[i]:.1f}"
                ann.box_label(xyxy, label, color=color_for(c))
                poly = self.polys[i].cpu().numpy()
                pts = poly[poly[:, 2] >= poly_conf_thr][:, :2]
                if len(pts) >= 3:
                    cv2.polylines(ann.im, [pts.astype(np.int32)], True, color_for(c), 2, cv2.LINE_AA)
        return ann.result()

    def save(self, filename):
        cv2.imwrite(str(filename), self.plot())
        return filename

    def __repr__(self):
        return f"PolyResults(path={self.path!r}, {len(self)} detections)"


def _scale_points(points, imgsz, ratio, pad, orig_shape):
    """Map normalised [0,1] (letterbox-space) points to original-image pixels."""
    pts = points.clone()
    pts[..., 0] = pts[..., 0] * imgsz  # to letterbox pixels
    pts[..., 1] = pts[..., 1] * imgsz
    pts[..., 0] = (pts[..., 0] - pad[0]) / ratio[0]
    pts[..., 1] = (pts[..., 1] - pad[1]) / ratio[1]
    pts[..., 0].clamp_(0, orig_shape[1])
    pts[..., 1].clamp_(0, orig_shape[0])
    return pts


class PolygonPredictor:
    """Run polygon + distance inference for a built polygon model."""

    def __init__(self, model, device="cpu", imgsz=640, conf=0.25, iou=0.45, max_det=300, names=None,
                 num_angles=None, angle_step=DEFAULT_ANGLE_STEP, min_distance=DEFAULT_MIN_DISTANCE,
                 max_distance=DEFAULT_MAX_DISTANCE):
        self.model = model.to(device).eval()
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.nc = model.nc
        self.num_angles = num_angles or getattr(model, "num_angles", DEFAULT_NUM_ANGLES)
        self.angle_step = angle_step
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.names = names or getattr(model, "names", None) or COCO_NAMES

    def preprocess(self, im):
        img, ratio, pad = letterbox(im, self.imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0).to(self.device), (ratio, pad)

    @torch.no_grad()
    def __call__(self, source):
        images, paths = self._load_source(source)
        results = []
        for im, path in zip(images, paths):
            tensor, (ratio, pad) = self.preprocess(im)
            out = self.model(tensor)
            y = out[0] if isinstance(out, tuple) else out  # (1, 4+nc+1+3N, A)
            boxes, dists, polys = self._postprocess(y[0], im.shape[:2], ratio, pad)
            results.append(PolyResults(im, path, self.names, boxes, dists, polys))
        return results

    def _postprocess(self, pred, orig_shape, ratio, pad):
        """Decode a single image's raw output into boxes, distances, polygons."""
        nc, N = self.nc, self.num_angles
        pred = pred.transpose(0, 1)  # (A, C)
        box = pred[:, :4]
        cls = pred[:, 4 : 4 + nc]
        off = 4 + nc
        dist_raw = pred[:, off]
        pconf = pred[:, off + 1 : off + 1 + N]
        pangle = pred[:, off + 1 + N : off + 1 + 2 * N]
        pdist = pred[:, off + 1 + 2 * N : off + 1 + 3 * N]

        conf, j = cls.max(1)
        keep = conf > self.conf
        if keep.sum() == 0:
            return None, None, None
        box, conf, j = box[keep], conf[keep], j[keep]
        dist_raw, pconf, pangle, pdist = dist_raw[keep], pconf[keep], pangle[keep], pdist[keep]

        xyxy = xywh2xyxy(box)
        i = torchvision.ops.nms(xyxy, conf, self.iou)[: self.max_det]
        xyxy, conf, j = xyxy[i], conf[i], j[i]
        dist_raw, pconf, pangle, pdist = dist_raw[i], pconf[i], pangle[i], pdist[i]

        # Polygon origin = centre of the predicted box, normalised to letterbox space.
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / self.imgsz
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / self.imgsz
        origin = torch.stack((cx, cy), 1)  # (n, 2) normalised
        poly = decode_polygons(pconf, pangle, pdist, origin, num_angles=N)  # (n, N, 3) normalised xy + conf
        poly[..., :2] = _scale_points(poly[..., :2], self.imgsz, ratio, pad, orig_shape)

        boxes6 = torch.cat((xyxy, conf[:, None], j[:, None].float()), 1)
        boxes6[:, :4] = clip_boxes(self._scale_boxes(boxes6[:, :4], ratio, pad), orig_shape)
        distance = decode_distance(dist_raw, self.min_distance, self.max_distance)
        return boxes6, distance, poly

    @staticmethod
    def _scale_boxes(xyxy, ratio, pad):
        xyxy = xyxy.clone()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio[0]
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio[1]
        return xyxy

    @staticmethod
    def _load_source(source):
        from pathlib import Path

        if isinstance(source, (str, Path)):
            im = cv2.imread(str(source))
            if im is None:
                raise FileNotFoundError(f"Could not read image: {source}")
            return [im], [source]
        if isinstance(source, np.ndarray):
            return [source], ["image0.jpg"]
        if isinstance(source, (list, tuple)):
            images, paths = [], []
            for s in source:
                im, p = PolygonPredictor._load_source(s)
                images += im
                paths += p
            return images, paths
        raise TypeError(f"Unsupported source type: {type(source)}")

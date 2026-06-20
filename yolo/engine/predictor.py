"""Inference pipeline: preprocess -> forward -> NMS -> Results."""

from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.augment import letterbox
from ..utils import LOGGER
from ..utils.ops import non_max_suppression, scale_boxes
from ..utils.plotting import COCO_NAMES, plot_detections


class Results:
    """Container for the detections of a single image."""

    def __init__(self, orig_img, path, names, boxes=None):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = str(path)
        self.names = names
        self.boxes = boxes  # tensor (n, 6): x1,y1,x2,y2,conf,cls

    def __len__(self):
        return 0 if self.boxes is None else len(self.boxes)

    @property
    def xyxy(self):
        return None if self.boxes is None else self.boxes[:, :4]

    @property
    def conf(self):
        return None if self.boxes is None else self.boxes[:, 4]

    @property
    def cls(self):
        return None if self.boxes is None else self.boxes[:, 5]

    def plot(self):
        """Return an annotated BGR image (np.ndarray)."""
        b = None if self.boxes is None else self.boxes.cpu().numpy()
        return plot_detections(self.orig_img, b, names=self.names)

    def save(self, filename):
        cv2.imwrite(str(filename), self.plot())
        return filename

    def summary(self):
        if not len(self):
            return []
        out = []
        for *xyxy, conf, cls in self.boxes.cpu().numpy():
            c = int(cls)
            out.append(
                {
                    "name": self.names.get(c, f"class{c}"),
                    "class": c,
                    "confidence": float(conf),
                    "box": {"x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3])},
                }
            )
        return out

    def __repr__(self):
        return f"Results(path={self.path!r}, {len(self)} detections)"


class DetectionPredictor:
    """Run detection inference for a loaded ``DetectionModel``."""

    def __init__(self, model, device="cpu", imgsz=640, conf=0.25, iou=0.45, max_det=300, names=None):
        self.model = model.to(device).eval()
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.names = names or getattr(model, "names", None) or COCO_NAMES

    def preprocess(self, im):
        """Letterbox + to-tensor a single BGR image. Returns (tensor, ratio_pad)."""
        img, ratio, pad = letterbox(im, self.imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor, (ratio, pad)

    @torch.no_grad()
    def __call__(self, source):
        """Run inference on a path, an image array, or a list of either."""
        images, paths = self._load_source(source)
        results = []
        for im, path in zip(images, paths):
            tensor, _ = self.preprocess(im)
            preds = self.model(tensor)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            det = non_max_suppression(
                preds, self.conf, self.iou, max_det=self.max_det, nc=len(self.names)
            )[0]
            if len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], im.shape[:2]).round()
            results.append(Results(im, path, self.names, det if len(det) else None))
        return results

    @staticmethod
    def _load_source(source):
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
                imgs, ps = DetectionPredictor._load_source(s)
                images += imgs
                paths += ps
            return images, paths
        raise TypeError(f"Unsupported source type: {type(source)}")

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

    def save_txt(self, filename, save_conf=False):
        """Write detections as YOLO-format lines: ``cls cx cy w h [conf]`` (normalised)."""
        h, w = self.orig_shape
        lines = []
        if len(self):
            for *xyxy, conf, cls in self.boxes.cpu().numpy():
                cx = ((xyxy[0] + xyxy[2]) / 2) / w
                cy = ((xyxy[1] + xyxy[3]) / 2) / h
                bw = (xyxy[2] - xyxy[0]) / w
                bh = (xyxy[3] - xyxy[1]) / h
                row = [int(cls), cx, cy, bw, bh] + ([float(conf)] if save_conf else [])
                lines.append(" ".join(f"{v:.6g}" for v in row))
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text("\n".join(lines) + ("\n" if lines else ""))
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
    def __call__(self, source, save=False, save_txt=False, save_conf=False, show=False,
                 visualize=False, save_dir="runs/predict"):
        """Run inference on an image / array / list / directory / glob.

        Options:
            save        - write annotated images to ``save_dir``;
            save_txt    - write YOLO-format predictions to ``save_dir/labels``;
            save_conf   - include confidence in the saved txt;
            show        - display via cv2.imshow (falls back to saving when headless);
            visualize   - save backbone feature-map grids to ``save_dir/features``.
        """
        images, paths = self._load_source(source)
        save_dir = Path(save_dir)
        results = []
        for im, path in zip(images, paths):
            tensor, _ = self.preprocess(im)
            if visualize:
                self._visualize_features(tensor, save_dir / "features" / Path(path).stem)
            preds = self.model(tensor)
            preds = preds[0] if isinstance(preds, (list, tuple)) else preds
            det = non_max_suppression(preds, self.conf, self.iou, max_det=self.max_det, nc=len(self.names))[0]
            if len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], im.shape[:2]).round()
            res = Results(im, path, self.names, det if len(det) else None)
            results.append(res)

            stem = Path(path).stem
            if save:
                (save_dir).mkdir(parents=True, exist_ok=True)
                res.save(save_dir / f"{stem}.jpg")
            if save_txt:
                res.save_txt(save_dir / "labels" / f"{stem}.txt", save_conf=save_conf)
            if show:
                self._show(res)
        if save or save_txt or visualize:
            LOGGER.info(f"Predictions saved to {save_dir}")
        return results

    def _visualize_features(self, tensor, out_dir):
        """Capture and save feature-map grids from the backbone via forward hooks."""
        from ..utils.plotting import feature_visualization

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        feats, hooks = {}, []
        layers = list(self.model.model)[:10]  # backbone layers (P1..SPPF)
        for li, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(lambda m, i, o, li=li: feats.__setitem__(li, o)))
        try:
            self.model(tensor)
        finally:
            for hk in hooks:
                hk.remove()
        for li, fmap in feats.items():
            if isinstance(fmap, torch.Tensor) and fmap.ndim == 4:
                feature_visualization(fmap, str(out_dir / f"stage{li}.png"))
        return out_dir

    @staticmethod
    def _show(res):
        img = res.plot()
        try:
            cv2.imshow(res.path, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:  # headless environment
            LOGGER.info("cv2.imshow unavailable (headless); use save=True to write images")

    @staticmethod
    def _load_source(source):
        from ..data.dataset import IMG_EXTS

        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():  # directory -> all images within (recursive)
                files = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
                if not files:
                    raise FileNotFoundError(f"No images found in directory: {source}")
                images, paths = [], []
                for f in files:
                    im = cv2.imread(f)
                    if im is not None:
                        images.append(im)
                        paths.append(f)
                return images, paths
            if any(ch in str(source) for ch in "*?["):  # glob pattern
                import glob

                files = sorted(glob.glob(str(source)))
                images = [cv2.imread(f) for f in files]
                return [im for im in images if im is not None], [f for f, im in zip(files, images) if im is not None]
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

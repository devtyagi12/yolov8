"""High-level :class:`YOLO` API mirroring the ergonomics of the Ultralytics one.

Typical usage::

    from yolo import YOLO

    model = YOLO("yolov8n.pt")          # load official weights (no ultralytics needed)
    results = model.predict("bus.jpg")  # inference
    results[0].save("out.jpg")

    model = YOLO("yolov8n")             # fresh model from scale name
    model.train(data_train="coco/images/train", data_val="coco/images/val", epochs=50)
"""

from pathlib import Path

import torch

from .cfg import SCALES
from .engine.predictor import DetectionPredictor
from .engine.trainer import DetectionTrainer
from .engine.validator import DetectionValidator
from .nn.tasks import DetectionModel
from .utils import LOGGER
from .utils.checkpoint import load_checkpoint, remap_state_dict
from .utils.plotting import COCO_NAMES


def _parse_scale(name):
    """Extract the n/s/m/l/x scale letter from a model name like ``yolov8s`` / ``yolov8s.pt``."""
    stem = Path(name).stem.lower()
    for s in ("n", "s", "m", "l", "x"):
        if stem.endswith(f"yolov8{s}") or stem == s or stem.endswith(f"v8{s}"):
            return s
    return None


class YOLO:
    """A YOLOv8 detection model with predict / train / val helpers."""

    def __init__(self, model="yolov8n", nc=None, device=None, verbose=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = None
        self.overrides = {}
        self.names = None

        if isinstance(model, (str, Path)) and str(model).endswith(".pt") and Path(model).exists():
            self.model = self._load_pt(str(model), nc=nc, verbose=verbose)
        else:
            scale = model if model in SCALES else (_parse_scale(str(model)) or "n")
            self.model = DetectionModel(scale, nc=nc or 80, verbose=verbose)
            self.names = self.model.names

        self.model.to(self.device)

    # ------------------------------------------------------------------ loading
    def _load_pt(self, path, nc=None, verbose=True):
        state_dict, meta = load_checkpoint(path)
        state_dict = remap_state_dict(state_dict)
        scale = _parse_scale(path) or "n"
        nc = nc or meta.get("nc") or 80
        model = DetectionModel(scale, nc=nc, verbose=verbose)
        model.load_state_dict_compat(state_dict, strict=False)

        names = meta.get("names")
        if names:
            model.names = names
        elif nc == 80:
            model.names = dict(COCO_NAMES)
        self.names = model.names
        self.ckpt_path = path
        LOGGER.info(f"Loaded weights from {path} (scale={scale}, nc={nc})")
        return model

    # ------------------------------------------------------------------ inference
    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, max_det=300):
        """Run detection inference. Returns a list of ``Results``."""
        predictor = DetectionPredictor(
            self.model, device=self.device, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, names=self.names
        )
        return predictor(source)

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    # ------------------------------------------------------------------ training
    def train(self, data_train, data_val=None, epochs=100, batch=16, imgsz=640, **kwargs):
        """Train the detection model on a YOLO-format dataset."""
        trainer = DetectionTrainer(
            self.model,
            data_train=data_train,
            data_val=data_val,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )
        self.model = trainer.train()
        return self.model

    def val(self, data, batch=16, imgsz=640, conf=0.001, iou=0.7):
        """Validate the model on a YOLO-format dataset, returning mAP metrics."""
        from .data.build import build_dataloader

        loader = build_dataloader(data, imgsz=imgsz, batch=batch, augment=False, shuffle=False)
        validator = DetectionValidator(self.model, loader, device=self.device, conf=conf, iou=iou)
        return validator()

    # ------------------------------------------------------------------ io
    def save(self, path):
        """Save a plain, ultralytics-free checkpoint (state dict + metadata)."""
        torch.save(
            {"model_state_dict": self.model.state_dict(), "nc": self.model.nc, "names": self.names, "scale": getattr(self.model, "scale", None)},
            path,
        )
        LOGGER.info(f"Saved model to {path}")
        return path

    def info(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        n_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        LOGGER.info(f"YOLO detection model: {self.model.nc} classes, {n_params:,} params ({n_grad:,} trainable)")
        return n_params

    @property
    def fuse(self):
        return self.model

"""High-level API for the polygon + distance YOLOv8 model.

Usage::

    from yolo.poly_model import YOLOPolygon

    model = YOLOPolygon("yolov8s.pt", nc=3)         # box/cls branches start from official weights
    model.train(poly_train="poly/images/train",
                dist_train="polyd/images/train",
                val_data="polyd/images/train", val_has_distance=True,
                epochs=100, poly_batch=8, dist_batch=8)
    results = model.predict("img.jpg")              # bbox, cls, conf, distance, polygon
    results[0].save("out.jpg")
"""

from pathlib import Path

import torch

from .engine.poly_predictor import PolygonPredictor
from .engine.poly_trainer import PolygonTrainer
from .engine.poly_validator import PolygonValidator
from .nn.poly_tasks import build_polygon_model
from .utils import LOGGER
from .utils.checkpoint import load_checkpoint, remap_state_dict
from .utils.plotting import COCO_NAMES
from .utils.poly_ops import DEFAULT_ANGLE_STEP, DEFAULT_NUM_ANGLES


class YOLOPolygon:
    """A YOLOv8s model extended to predict polygons (star) and distance."""

    def __init__(self, model="yolov8s", nc=80, scale="s", num_angles=DEFAULT_NUM_ANGLES,
                 angle_step=DEFAULT_ANGLE_STEP, num_dist_blocks=2, device=None, verbose=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_angles = num_angles
        self.angle_step = angle_step
        self.model = build_polygon_model(scale, nc=nc, num_angles=num_angles, angle_step=angle_step,
                                         num_dist_blocks=num_dist_blocks, verbose=verbose)

        # Optionally seed the shared box/class + backbone weights from a checkpoint.
        if isinstance(model, (str, Path)) and str(model).endswith(".pt") and Path(model).exists():
            self._load_backbone(str(model))
        self.names = {i: COCO_NAMES.get(i, f"class{i}") for i in range(nc)} if nc == 80 else {i: f"class{i}" for i in range(nc)}
        self.model.names = self.names
        self.model.to(self.device)

    def _load_backbone(self, path):
        state_dict, _ = load_checkpoint(path)
        state_dict = remap_state_dict(state_dict)
        self.model.load_state_dict_compat(state_dict, strict=False)
        LOGGER.info(f"Seeded shared backbone/box/cls weights from {path}")

    # ------------------------------------------------------------------ inference
    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, max_det=300, **kwargs):
        predictor = PolygonPredictor(self.model, device=self.device, imgsz=imgsz, conf=conf, iou=iou,
                                     max_det=max_det, names=self.names, num_angles=self.num_angles,
                                     angle_step=self.angle_step, **kwargs)
        return predictor(source)

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    # ------------------------------------------------------------------ training
    def train(self, poly_train=None, dist_train=None, val_data=None, val_has_distance=False, epochs=100,
              poly_batch=8, dist_batch=8, imgsz=640, **kwargs):
        """Train on polygon data, polygon+distance data, or both (distance is optional)."""
        trainer = PolygonTrainer(self.model, poly_train=poly_train, dist_train=dist_train, val_data=val_data,
                                 val_has_distance=val_has_distance, epochs=epochs, poly_batch=poly_batch,
                                 dist_batch=dist_batch, imgsz=imgsz, device=self.device, **kwargs)
        self.model = trainer.train()
        return self.model

    def val(self, data, has_distance=False, batch=8, imgsz=640, conf=0.25, iou=0.5):
        from .data.poly_dataset import build_poly_dataloader

        loader = build_poly_dataloader(data, has_distance=has_distance, imgsz=imgsz, batch=batch, augment=False,
                                       num_angles=self.num_angles, angle_step=self.angle_step)
        return PolygonValidator(self.model, loader, device=self.device, conf=conf, iou=iou)()

    # ------------------------------------------------------------------ io
    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "nc": self.model.nc,
                "names": self.names,
                "num_angles": self.num_angles,
                "angle_step": self.angle_step,
            },
            path,
        )
        LOGGER.info(f"Saved polygon model to {path}")
        return path

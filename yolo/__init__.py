"""
yolo: a standalone re-implementation of YOLOv8 object detection.

This package replicates the YOLOv8 detection model, training, validation and
inference pipeline *without* depending on the ``ultralytics`` library, while
remaining weight-compatible with the official ``yolov8{n,s,m,l,x}.pt`` releases.
"""

from .model import YOLO

__version__ = "0.1.0"
__all__ = ["YOLO", "__version__"]

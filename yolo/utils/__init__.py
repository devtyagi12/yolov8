"""Utility subpackage for the standalone YOLOv8 implementation."""

import logging
import os

LOGGER = logging.getLogger("yolo")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO if os.getenv("YOLO_VERBOSE", "1") != "0" else logging.WARNING)
LOGGER.propagate = False

__all__ = ["LOGGER"]

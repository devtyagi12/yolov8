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


def read_label_text(path):
    """Read a label file as UTF-8 text, or return ``None`` (with a warning) if it is
    binary / corrupt (undecodable or containing NUL bytes)."""
    try:
        data = open(path, "rb").read()
    except OSError as e:
        LOGGER.warning(f"Could not read label file {path}: {e}")
        return None
    if b"\x00" in data:
        LOGGER.warning(f"Skipping corrupt label file (contains NUL bytes): {path}")
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        LOGGER.warning(f"Skipping corrupt label file (not valid UTF-8): {path}")
        return None


__all__ = ["LOGGER", "read_label_text"]

"""Drawing utilities for detection results."""

import cv2
import numpy as np

# Default 80-class COCO names (used when a checkpoint carries no ``names``).
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train",
    7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign",
    12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Ultralytics-style color palette (hex -> BGR resolved on demand).
_HEXS = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334",
    "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF",
    "FF95C8", "FF37C7",
)


def _hex2bgr(h):
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[2], rgb[1], rgb[0]


def color_for(i):
    """Return a deterministic BGR colour tuple for class index ``i``."""
    return _hex2bgr(_HEXS[int(i) % len(_HEXS)])


class Annotator:
    """Draw boxes and labels onto a BGR image (cv2)."""

    def __init__(self, im, line_width=None, font_scale=None):
        self.im = im if im.flags.writeable else im.copy()
        h, w = im.shape[:2]
        self.lw = line_width or max(round((h + w) / 2 * 0.003), 2)
        self.sf = font_scale or self.lw / 3

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(
                self.im,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                self.sf,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    def result(self):
        return self.im


def plot_detections(im, boxes, names=None):
    """Render an annotated copy of ``im`` from a ``(n, 6)`` detection tensor/array.

    ``boxes`` rows are ``[x1, y1, x2, y2, conf, cls]`` in ``im`` pixel coordinates.
    """
    names = names or COCO_NAMES
    annotator = Annotator(np.ascontiguousarray(im))
    if boxes is not None and len(boxes):
        boxes = np.asarray(boxes)
        for *xyxy, conf, cls in boxes:
            c = int(cls)
            label = f"{names.get(c, f'class{c}')} {conf:.2f}"
            annotator.box_label(xyxy, label, color=color_for(c))
    return annotator.result()

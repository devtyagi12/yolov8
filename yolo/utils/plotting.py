"""Drawing utilities for detection results."""

from pathlib import Path

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


def draw_normalized(im_bgr, cls, boxes_xywhn, names=None):
    """Draw normalised ``[cx, cy, w, h]`` boxes (in [0, 1]) on a copy of a BGR image.

    Useful for verifying that augmentation keeps labels aligned with the pixels.
    """
    names = names or {}
    h, w = im_bgr.shape[:2]
    annotator = Annotator(np.ascontiguousarray(im_bgr))
    cls = np.asarray(cls).reshape(-1)
    boxes_xywhn = np.asarray(boxes_xywhn, dtype=np.float32).reshape(-1, 4)
    for c, (cx, cy, bw, bh) in zip(cls, boxes_xywhn):
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        ci = int(c)
        annotator.box_label([x1, y1, x2, y2], names.get(ci, str(ci)), color=color_for(ci))
    return annotator.result()


def draw_poly_star(im_bgr, cls, bboxes_xywhn, poly_star, num_angles, names=None, conf_thr=0.5):
    """Draw normalised bbox + star-polygon outline for each object (polygon targets)."""
    names = names or {}
    h, w = im_bgr.shape[:2]
    annotator = Annotator(np.ascontiguousarray(im_bgr))
    cls = np.asarray(cls).reshape(-1)
    bboxes_xywhn = np.asarray(bboxes_xywhn, dtype=np.float32).reshape(-1, 4)
    poly_star = np.asarray(poly_star, dtype=np.float32).reshape(-1, 2 + 3 * num_angles)
    for i in range(len(cls)):
        c = int(cls[i])
        col = color_for(c)
        cx, cy, bw, bh = bboxes_xywhn[i]
        annotator.box_label([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                            names.get(c, str(c)), color=col)
        verts = poly_star[i, 2:].reshape(num_angles, 3)
        pts = [[verts[b, 0] * w, verts[b, 1] * h] for b in range(num_angles) if verts[b, 2] > conf_thr]
        if len(pts) >= 3:
            cv2.polylines(annotator.im, [np.array(pts, np.int32)], True, col, 2, cv2.LINE_AA)
    return annotator.result()


def draw_poly_predictions(im_bgr, boxes6, dists, polys, names=None, conf_thr=0.5):
    """Draw predicted box + ``"cls conf d=dist"`` label + polygon outline (pixel coords).

    ``boxes6`` (n,6) ``[x1,y1,x2,y2,conf,cls]``, ``dists`` (n,), ``polys`` (n,N,3) ``[x,y,conf]``.
    """
    names = names or {}
    annotator = Annotator(np.ascontiguousarray(im_bgr))
    if boxes6 is None or len(boxes6) == 0:
        return annotator.result()
    boxes6 = np.asarray(boxes6, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32).reshape(-1)
    polys = np.asarray(polys, dtype=np.float32)
    for i, (*xyxy, conf, cls) in enumerate(boxes6):
        c = int(cls)
        col = color_for(c)
        label = f"{names.get(c, str(c))} {conf:.2f} d={dists[i]:.1f}"
        annotator.box_label(xyxy, label, color=col)
        pts = polys[i][polys[i][:, 2] >= conf_thr][:, :2]
        if len(pts) >= 3:
            cv2.polylines(annotator.im, [pts.astype(np.int32)], True, col, 2, cv2.LINE_AA)
    return annotator.result()


def image_grid(images, cols=2, pad=6, fill=30):
    """Tile equal-sized BGR images into a single grid image (row-major)."""
    if not images:
        raise ValueError("image_grid received no images")
    h, w = images[0].shape[:2]
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    grid = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), fill, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        grid[y : y + h, x : x + w] = cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im
    return grid


def _mpl():
    """Return a non-interactive matplotlib.pyplot, or None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_pr_curve(px, py, ap, names, save_path="PR_curve.png"):
    """Plot precision-recall curves (per class + mean) to ``save_path``."""
    plt = _mpl()
    if plt is None or py.size == 0:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = py.T  # (1000, nc)
    for i in range(py.shape[1]):
        label = f"{names.get(i, i)} {ap[i, 0]:.3f}" if isinstance(names, dict) else f"{i} {ap[i,0]:.3f}"
        ax.plot(px, py[:, i], linewidth=1, label=label)
    ax.plot(px, py.mean(1), linewidth=3, color="navy", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_mc_curve(px, py, names, save_path, ylabel="Metric", xlabel="Confidence"):
    """Plot a metric-vs-confidence curve (F1 / P / R) per class + mean."""
    plt = _mpl()
    if plt is None or py.size == 0:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for i in range(py.shape[0]):
        label = f"{names.get(i, i)}" if isinstance(names, dict) else str(i)
        ax.plot(px, py[i], linewidth=1, label=label)
    mean = py.mean(0)
    best = px[mean.argmax()]
    ax.plot(px, mean, linewidth=3, color="navy", label=f"all classes {mean.max():.2f} at {best:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    ax.set_title(f"{ylabel}-{xlabel} Curve")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(matrix, names, save_path="confusion_matrix.png", normalize=True):
    """Plot a confusion matrix heatmap (numbers shown for small class counts)."""
    plt = _mpl()
    if plt is None:
        return None
    array = matrix.copy()
    if normalize:
        array = array / (array.sum(0, keepdims=True) + 1e-9)
    nc = array.shape[0] - 1
    labels = [names.get(i, str(i)) if isinstance(names, dict) else str(i) for i in range(nc)] + ["background"]
    fig, ax = plt.subplots(figsize=(8, 7), tight_layout=True)
    im = ax.imshow(array, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(nc + 1))
    ax.set_yticks(range(nc + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    if nc <= 30:
        for i in range(nc + 1):
            for j in range(nc + 1):
                ax.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if array[i, j] > array.max() / 2 else "black")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_results(csv_path, save_path=None):
    """Plot per-epoch training/validation curves from a results CSV to ``results.png``."""
    plt = _mpl()
    if plt is None:
        return None
    import csv as _csv

    rows, header = [], None
    with open(csv_path) as f:
        reader = _csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append([float(x) for x in r])
    if not rows:
        return None
    data = np.array(rows)
    cols = [c for c in range(len(header)) if header[c] != "epoch"]
    epoch_idx = header.index("epoch") if "epoch" in header else 0
    n = len(cols)
    ncol = min(5, n)
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.5 * nrow), tight_layout=True, squeeze=False)
    for i, c in enumerate(cols):
        a = ax[i // ncol][i % ncol]
        a.plot(data[:, epoch_idx], data[:, c], marker=".", linewidth=1.5)
        a.set_title(header[c], fontsize=9)
        a.set_xlabel("epoch", fontsize=7)
    for j in range(n, nrow * ncol):
        ax[j // ncol][j % ncol].axis("off")
    save_path = save_path or str(Path(csv_path).with_name("results.png"))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_label_histogram(classes, names, save_path):
    """Plot a class-instance histogram (labels.png) from a list of class ids."""
    plt = _mpl()
    if plt is None or len(classes) == 0:
        return None
    classes = np.asarray(classes).astype(int)
    nc = int(classes.max()) + 1
    counts = np.bincount(classes, minlength=nc)
    labels = [names.get(i, str(i)) if isinstance(names, dict) else str(i) for i in range(nc)]
    fig, ax = plt.subplots(figsize=(max(6, nc * 0.5), 4), tight_layout=True)
    ax.bar(range(nc), counts, color="steelblue")
    ax.set_xticks(range(nc))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("instances")
    ax.set_title("Class distribution")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def feature_visualization(x, save_path, n=16):
    """Save a grid of the first ``n`` channels of a feature map tensor ``x`` (1,C,H,W)."""
    plt = _mpl()
    if plt is None:
        return None
    if x.ndim != 4:
        return None
    c = min(n, x.shape[1])
    blocks = x[0, :c].detach().cpu().float()
    cols = min(c, 8)
    rows = (c + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows), tight_layout=True, squeeze=False)
    for i in range(rows * cols):
        a = ax[i // cols][i % cols]
        a.axis("off")
        if i < c:
            a.imshow(blocks[i], cmap="viridis")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


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

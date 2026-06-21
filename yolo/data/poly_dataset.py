"""Polygon (+ optional distance) datasets in YOLO format.

Label files contain one object per line::

    class_label x1 y1 x2 y2 ... xn yn            # polygon only  (V8ParserExtended)
    class_label x1 y1 x2 y2 ... xn yn distance   # polygon + distance (V8DistanceParser)

Polygon points are normalised to ``[0, 1]``; ``n`` (the vertex count) may vary per
object.  There are no box coordinates — the bounding box is derived from the
polygon.  The two parsers can be merged into a single dataloader (with a separate
batch size per dataset) via :func:`build_merged_dataloader`; polygon-only samples
carry a sentinel distance of ``-10.0``.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..utils import LOGGER
from ..utils.poly_ops import (
    DEFAULT_ANGLE_STEP,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_NUM_ANGLES,
    NO_DISTANCE,
    bbox_from_polygon,
    encode_distance,
    poly_vector_len,
    polygon_to_star,
)
from .augment import (
    DEFAULT_HYP,
    Albumentations,
    Compose,
    CopyPaste,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
    xyxy2xywhn,
)
from .dataset import IMG_EXTS, img2label_path


class PolyFormat:
    """Final transform: derive bbox from polygon, build the star target, normalise & to-tensor.

    Output keys: ``img`` (CHW float tensor), ``cls`` (M,1), ``bboxes`` (M,4 norm xywh),
    ``poly`` (M, 2 + 3N normalised star), ``distance`` (M,).
    """

    def __init__(self, num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP, min_vertices=3):
        self.num_angles = num_angles
        self.angle_step = angle_step
        self.min_vertices = min_vertices

    def __call__(self, labels):
        img = labels["img"]
        h, w = img.shape[:2]
        polygons = labels.get("polygons") or []
        cls = np.asarray(labels.get("cls", np.zeros((0, 1), np.float32)), np.float32).reshape(-1, 1)
        distance = np.asarray(labels.get("distance", np.full(cls.shape[0], NO_DISTANCE, np.float32)), np.float32).reshape(-1)

        keep_cls, keep_box, keep_poly, keep_dist = [], [], [], []
        for i, poly in enumerate(polygons):
            poly = np.asarray(poly, np.float32).reshape(-1, 2)
            if poly.shape[0] < self.min_vertices:
                continue
            box = bbox_from_polygon(poly)  # pixel xyxy
            if (box[2] - box[0]) <= 1 or (box[3] - box[1]) <= 1:
                continue
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
            star = polygon_to_star(poly, center, self.num_angles, self.angle_step)  # pixel
            # Normalise origin + vertex coords by image size (conf columns left untouched).
            star = star.copy()
            star[0] /= w
            star[1] /= h
            verts = star[2:].reshape(self.num_angles, 3)
            verts[:, 0] /= w
            verts[:, 1] /= h
            star[2:] = verts.reshape(-1)

            keep_cls.append(cls[i] if i < cls.shape[0] else np.zeros(1, np.float32))
            keep_box.append(box)
            keep_poly.append(star)
            keep_dist.append(distance[i] if i < distance.shape[0] else NO_DISTANCE)

        n = len(keep_box)
        plen = poly_vector_len(self.num_angles)
        if n:
            boxes_xyxy = np.stack(keep_box, 0)
            bboxes = xyxy2xywhn(boxes_xyxy, w, h, clip=True, eps=1e-3)
            cls_out = np.stack(keep_cls, 0).astype(np.float32)
            poly_out = np.stack(keep_poly, 0).astype(np.float32)
            dist_out = np.asarray(keep_dist, np.float32).reshape(-1)
        else:
            bboxes = np.zeros((0, 4), np.float32)
            cls_out = np.zeros((0, 1), np.float32)
            poly_out = np.zeros((0, plen), np.float32)
            dist_out = np.zeros((0,), np.float32)

        chw = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        labels["img"] = torch.from_numpy(chw).float() / 255.0
        labels["cls"] = torch.from_numpy(cls_out)
        labels["bboxes"] = torch.from_numpy(bboxes)
        labels["poly"] = torch.from_numpy(poly_out)
        labels["distance"] = torch.from_numpy(dist_out)
        labels.pop("polygons", None)
        return labels


def v8_poly_transforms(dataset, imgsz, hyp, num_angles, angle_step):
    """Build the polygon training pipeline (standard geometric augs + PolyFormat)."""
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp["mosaic"])
    affine = RandomPerspective(
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        shear=hyp["shear"],
        perspective=hyp["perspective"],
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, CopyPaste(p=hyp["copy_paste"]), affine])
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp["mixup"]),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"]),
            RandomFlip(direction="vertical", p=hyp["flipud"]),
            RandomFlip(direction="horizontal", p=hyp["fliplr"]),
            PolyFormat(num_angles=num_angles, angle_step=angle_step),
        ]
    )


class PolygonDataset(Dataset):
    """Base polygon dataset. ``has_distance`` toggles parsing of a trailing distance token."""

    has_distance = False  # overridden by subclasses

    def __init__(
        self,
        path,
        imgsz=640,
        augment=False,
        hyp=None,
        num_angles=DEFAULT_NUM_ANGLES,
        angle_step=DEFAULT_ANGLE_STEP,
        min_distance=DEFAULT_MIN_DISTANCE,
        max_distance=DEFAULT_MAX_DISTANCE,
    ):
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = {**DEFAULT_HYP, **(hyp or {})}
        self.num_angles = num_angles
        self.angle_step = angle_step
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.im_files = self._gather_images(path)
        if not self.im_files:
            raise FileNotFoundError(f"No images found under {path!r}")
        self.label_files = [img2label_path(f) for f in self.im_files]
        self.transforms = self.build_transforms()
        kind = "polygon+distance" if self.has_distance else "polygon"
        LOGGER.info(f"{type(self).__name__}: {len(self.im_files)} {kind} images from {path}")

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _gather_images(path):
        path = Path(path)
        files = []
        if path.is_dir():
            files = [str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in IMG_EXTS]
        elif path.is_file() and path.suffix == ".txt":
            base = path.parent
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    p = (base / line) if not Path(line).is_absolute() else Path(line)
                    files.append(str(p))
        elif path.suffix.lower() in IMG_EXTS:
            files = [str(path)]
        return files

    def build_transforms(self):
        if self.augment:
            return v8_poly_transforms(self, self.imgsz, self.hyp, self.num_angles, self.angle_step)
        return Compose([LetterBox((self.imgsz, self.imgsz), scaleup=False), PolyFormat(self.num_angles, self.angle_step)])

    def close_mosaic(self):
        self.hyp["mosaic"] = 0.0
        self.hyp["mixup"] = 0.0
        self.hyp["copy_paste"] = 0.0
        self.transforms = self.build_transforms()

    # ------------------------------------------------------------------ parsing
    def __len__(self):
        return len(self.im_files)

    def parse_label_file(self, idx):
        """Return ``(list[poly_norm (k,2)], cls (n,), distance (n,))`` for image ``idx``."""
        lf = self.label_files[idx]
        polys, classes, dists = [], [], []
        if Path(lf).is_file():
            for line in Path(lf).read_text().strip().splitlines():
                toks = line.split()
                if not toks:
                    continue
                cls_id = float(toks[0])
                coords = toks[1:]
                dist = None
                if self.has_distance:
                    dist = float(coords[-1])
                    coords = coords[:-1]
                if len(coords) < 6 or len(coords) % 2 != 0:  # need >= 3 vertices
                    continue
                poly = np.array(coords, np.float32).reshape(-1, 2)
                polys.append(poly)
                classes.append(cls_id)
                dists.append(encode_distance(dist, self.min_distance, self.max_distance) if self.has_distance else NO_DISTANCE)
        cls = np.array(classes, np.float32) if classes else np.zeros((0,), np.float32)
        distance = np.array(dists, np.float32) if dists else np.zeros((0,), np.float32)
        return polys, cls, distance

    def get_image_and_label(self, idx):
        im = cv2.imread(self.im_files[idx])
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[idx]}")
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (round(w0 * r), round(h0 * r)), interpolation=interp)
        h, w = im.shape[:2]

        polys_norm, cls, distance = self.parse_label_file(idx)
        polygons, boxes = [], []
        for p in polys_norm:
            pp = p.copy()
            pp[:, 0] *= w
            pp[:, 1] *= h
            polygons.append(pp)
            boxes.append(bbox_from_polygon(pp))
        bboxes = np.stack(boxes, 0) if boxes else np.zeros((0, 4), np.float32)
        return {
            "img": im,
            "cls": cls.reshape(-1, 1),
            "bboxes": bboxes,
            "polygons": polygons,
            "distance": distance,
            "ori_shape": (h0, w0),
            "im_file": self.im_files[idx],
        }

    def __getitem__(self, idx):
        labels = self.transforms(self.get_image_and_label(idx))
        return {
            "img": labels["img"],
            "cls": labels["cls"],
            "bboxes": labels["bboxes"],
            "poly": labels["poly"],
            "distance": labels["distance"],
            "im_file": labels.get("im_file", self.im_files[idx]),
            "ori_shape": labels.get("ori_shape", (self.imgsz, self.imgsz)),
        }

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([b["img"] for b in batch], 0)
        cls = torch.cat([b["cls"] for b in batch], 0)
        bboxes = torch.cat([b["bboxes"] for b in batch], 0)
        poly = torch.cat([b["poly"] for b in batch], 0)
        distance = torch.cat([b["distance"] for b in batch], 0)
        batch_idx = torch.cat([torch.full((b["cls"].shape[0],), i) for i, b in enumerate(batch)], 0)
        return {
            "img": imgs,
            "cls": cls,
            "bboxes": bboxes,
            "poly": poly,
            "distance": distance,
            "batch_idx": batch_idx,
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
        }


class V8ParserExtended(PolygonDataset):
    """Polygon-only dataset (no distance). Every object's distance is the ``-10.0`` sentinel."""

    has_distance = False


class V8DistanceParser(PolygonDataset):
    """Polygon + distance dataset. Distance is stored as ``log(clip(distance))`` (or sentinel)."""

    has_distance = True


class MixedBatchSampler:
    """Yield batches each drawn entirely from one sub-dataset, with per-dataset batch sizes."""

    def __init__(self, lengths, batch_sizes, shuffle=True, drop_last=False):
        self.lengths = list(lengths)
        self.batch_sizes = list(batch_sizes)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.offsets = np.cumsum([0] + self.lengths[:-1]).tolist()

    def _make_batches(self):
        all_batches = []
        for length, bs, off in zip(self.lengths, self.batch_sizes, self.offsets):
            order = list(range(length))
            if self.shuffle:
                random.shuffle(order)
            for i in range(0, length, bs):
                chunk = order[i : i + bs]
                if self.drop_last and len(chunk) < bs:
                    continue
                all_batches.append([off + j for j in chunk])
        if self.shuffle:
            random.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self):
        return len(self._make_batches())


def build_merged_dataloader(
    poly_path,
    dist_path,
    imgsz=640,
    poly_batch=8,
    dist_batch=8,
    augment=True,
    workers=4,
    hyp=None,
    num_angles=DEFAULT_NUM_ANGLES,
    angle_step=DEFAULT_ANGLE_STEP,
    min_distance=DEFAULT_MIN_DISTANCE,
    max_distance=DEFAULT_MAX_DISTANCE,
    shuffle=True,
):
    """Concatenate a polygon-only dataset and a polygon+distance dataset into one loader.

    Each sub-dataset keeps its own batch size; every yielded batch is drawn entirely
    from one sub-dataset (so distance validity is consistent within a batch).
    """
    common = dict(
        imgsz=imgsz,
        augment=augment,
        hyp=hyp,
        num_angles=num_angles,
        angle_step=angle_step,
        min_distance=min_distance,
        max_distance=max_distance,
    )
    ds_poly = V8ParserExtended(poly_path, **common)
    ds_dist = V8DistanceParser(dist_path, **common)
    concat = ConcatDataset([ds_poly, ds_dist])
    concat.datasets_list = [ds_poly, ds_dist]  # keep references (e.g. for close_mosaic)
    sampler = MixedBatchSampler(
        lengths=[len(ds_poly), len(ds_dist)],
        batch_sizes=[poly_batch, dist_batch],
        shuffle=shuffle,
    )
    loader = DataLoader(
        concat,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=False,
        collate_fn=PolygonDataset.collate_fn,
    )
    return loader, concat


def build_poly_dataloader(path, has_distance=False, imgsz=640, batch=8, augment=False, workers=4, hyp=None,
                          num_angles=DEFAULT_NUM_ANGLES, angle_step=DEFAULT_ANGLE_STEP,
                          min_distance=DEFAULT_MIN_DISTANCE, max_distance=DEFAULT_MAX_DISTANCE, shuffle=None):
    """Single-dataset polygon dataloader (used for validation / single-source training)."""
    cls = V8DistanceParser if has_distance else V8ParserExtended
    ds = cls(path, imgsz=imgsz, augment=augment, hyp=hyp, num_angles=num_angles, angle_step=angle_step,
             min_distance=min_distance, max_distance=max_distance)
    if shuffle is None:
        shuffle = augment
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=False,
                      collate_fn=PolygonDataset.collate_fn)

# Polygon + Distance extension

This extends the standard YOLOv8s detector to additionally predict, **per object**:

* a **polygon** (star-shaped representation around the box centre), and
* a scalar **distance** (e.g. depth),

while keeping the bounding-box and class branches exactly as in stock YOLOv8 (so
official `yolov8s.pt` weights still load into them).

Model output format: **`bbox, cls, conf, distance, polygon`**.

## 1. Datasets

Two YOLO-format datasets, polygon points normalised to `[0, 1]`, **no box
coordinates** (the box is derived from the polygon). Each `.txt` line is one object:

```
# polygon only            -> V8ParserExtended
cls x1 y1 x2 y2 ... xn yn
# polygon + distance       -> V8DistanceParser
cls x1 y1 x2 y2 ... xn yn distance
```

`n` (vertex count) may vary per object. The two datasets are concatenated into one
loader with **a separate batch size each** (`build_merged_dataloader`); every batch
is drawn entirely from one sub-dataset, and polygon-only objects carry a sentinel
distance of `-10.0`.

## 2. Parsing

* **Bounding box** is the axis-aligned min/max of the polygon, then used exactly
  like a standard YOLO box.
* **Star polygon** (`yolo/utils/poly_ops.py::polygon_to_star`): with the box centre
  as origin and `num_angles = 360 // angle_step` (default `angle_step=15` → 24 bins),
  each vertex's distance & angle from the centre give a bin index
  `floor(angle / angle_step)`. Per bin we keep the **max-distance** vertex
  (`conf = 1`) or store `(0, 0, 0)`. The target is
  `[origin_x, origin_y, x0, y0, conf0, ..., x_{N-1}, y_{N-1}, conf_{N-1}]`
  (length `2 + 3N`), in normalised image coordinates.
* **Distance** is stored as `log(clip(distance, min, max))`; values `<= 0` or
  missing become the `-10.0` sentinel (`encode_distance`).
* **Parsers**: `V8ParserExtended` (polygon-only, distance `= -10`) and
  `V8DistanceParser` (polygon + distance). Merge via `build_merged_dataloader`.
* **Augmentation** is the standard `v8_transforms` chain — every geometric
  transform (mosaic, perspective, mixup, flips, letterbox) also transforms the
  polygon vertices, and the star target is built **after** augmentation.

## 3. Architecture (`DetectPolygon`)

* Box (`cv2`) and class (`cv3`) heads: **unchanged**.
* **Polygon branch** — 3 heads fed from the **2nd-last layer of the box branch**
  (`cv2[i][:2]`), each a 1×1 conv-BN with `num_angles` output channels:
  `poly_conf`, `poly_angle`, `poly_dist` → each `(B, N, A)`.
* **Distance head** — same input as the box/class heads (the level feature):
  `num_dist_blocks` 3×3 conv-BN + a 1×1 conv-BN with 1 channel → `(B, 1, A)`.

Inference channel layout: `[ box(4) | cls(nc) | dist(1) | poly_conf(N) | poly_angle(N) | poly_dist(N) ]`.

## 4. Loss (`v8PolygonDistanceLoss`)

Standard YOLOv8 loss (box CIoU + DFL + class BCE) plus:

* **Polygon loss** (foreground anchors only; distance & angle additionally masked by
  per-bin target presence):
  * *distance*: `MSE(softplus(pred), target_dist)`, target from `dx,dy`;
  * *angle*: `BCE(sigmoid(pred), frac_angle)`, `frac = (angle - idx*step)/step`;
  * *confidence*: `BCE_with_logits(pred, target_conf)`;
  * each normalised by `num_vertices × num_objects`.
  * `polygon_loss = 2.0·dist + 0.2·conf + 0.5·angle`.
* **Distance loss**: `L1(pred, target_log)` over foreground anchors **with valid
  distance** (`!= -10`), normalised by the number of such objects.
* **Total**: `… + 0.1·polygon_loss + 0.1·distance_loss`, then `× batch_size`.

Targets for a foreground anchor come from its assigned GT object (Task-Aligned
Assigner `target_gt_idx`). The polygon origin in the target is the GT box centre;
the predicted polygon origin (used at inference) is the **predicted** box centre.

## 5. Inference / post-processing

* Standard box + class decode.
* **Polygon decode** (`decode_polygons`): `dist = softplus`, `frac = sigmoid(angle)`,
  `conf = sigmoid(conf)`; absolute bin angle `= (frac + offset)/N·360°`; polar→cartesian
  offsets from the predicted box centre → vertices; scaled back to the original image
  (vertices with `conf < 0.5` are dropped).
* **Distance decode**: `clip(exp(pred), min, max)`.
* **NMS** runs on the boxes and carries the polygon + distance as attributes.

Validation metric: **bounding-box F1** (`PolygonValidator`).

## Quick start

```python
from yolo.poly_model import YOLOPolygon

model = YOLOPolygon("yolov8s.pt", nc=3, num_angles=24)   # box/cls seeded from official weights
model.train(poly_train="poly/images/train", dist_train="polyd/images/train",
            val_data="polyd/images/train", val_has_distance=True,
            epochs=100, poly_batch=8, dist_batch=8)
results = model.predict("img.jpg")                       # bbox, cls, conf, distance, polygon
print(results[0].summary())
results[0].save("out.jpg")
```

CLI: `examples/polygon_train.py`, `examples/polygon_predict.py`.

## Coordinate convention

Polygon geometry (origins, vertices, distances) is kept in **normalised image
coordinates** `[0, 1]` throughout training and decode, so the loss targets and the
inference decode share one space; the predictor then scales vertices back to the
original image. For a fixed square input this normalisation is equivalent to the
per-level stride denormalisation described in the task spec.

## Files

```
yolo/utils/poly_ops.py       star conversion, distance encode/decode, polygon decode
yolo/data/poly_dataset.py    V8ParserExtended, V8DistanceParser, PolyFormat, merged loader
yolo/nn/poly_modules.py      DetectPolygon head
yolo/nn/poly_tasks.py        build_polygon_model
yolo/utils/poly_loss.py      v8PolygonDistanceLoss
yolo/engine/poly_predictor.py / poly_validator.py / poly_trainer.py
yolo/poly_model.py           YOLOPolygon high-level API
examples/polygon_train.py, examples/polygon_predict.py
tests/test_polygon.py
```

# Models

Weights are **not** committed. YOLOE-26 is **AGPL-3.0** (Ultralytics lineage),
which is a real constraint on anything you distribute — check it against your
plans before shipping.

## Export

```bash
pip install ultralytics onnx onnxslim
yolo export model=yoloe26-s-seg.pt format=onnx half=True imgsz=640 \
            simplify=True opset=12
mv yoloe26-s-seg.onnx models/
```

Sizes run `n` → `x` (`yoloe26-n-seg` is 4.8M params). `s` or `m` is the sweet
spot in a browser: `n` is fast but misses small objects, `l`/`x` will not stay
under the 1 GB ceiling alongside a 45 MP photo.

Point `MODEL.url` in `js/yoloe-engine.js` elsewhere if you use another size.

## vocabulary.json

Optional, sits next to the `.onnx`. A JSON array of label strings, in the
class order the export was re-parameterized with:

```json
["person", "bicycle", "car", "..."]
```

YOLOE folds text embeddings into the network at export time, so **there is no
text encoder at runtime** — a phrase is matched against these labels, which
costs nothing. Without this file, click / box / lasso still work and text
search reports that it has no vocabulary rather than silently matching
nothing.

Use the prompt-free RAM++ tag set (4,585 categories) for open-world coverage,
or a short custom list if you only care about specific classes — a smaller
vocabulary is both faster and more accurate on the classes you kept.

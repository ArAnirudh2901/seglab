# SEGLAB — on-device segmentation

Import a photo, then select anything — click it, drag a box, draw a lasso, or
just describe it. One model, running entirely in your browser. Zero cloud:
nothing is uploaded, ever.

## One analysis, many selections

**YOLOE-26** runs **once per photo** and finds every instance in it. Each mode
is then a query over that one set:

| mode | query |
|---|---|
| click | the instance under the point (smallest wins, so nested objects stay reachable) |
| box | the instances the region genuinely covers |
| lasso | the same, against the polygon |
| text | the instances whose label matches your phrase |

Two consequences worth knowing. First, everything after the initial analysis
is pure arithmetic over typed arrays — **no model in the loop**, so clicking
around an analyzed photo is instant. Second, the modes always agree: a click
and the phrase for the same object return the *same pixels*, because they
return the same instance.

A phrase that isn't in the photo returns **nothing**. That's the correct
answer, not a failure, and it's a tested gate.

## Run

```bash
python3 -m http.server 8788
# open http://localhost:8788
```

You need to export the model first — it isn't committed. See
[`models/README.md`](models/README.md). Short version:

```bash
pip install ultralytics onnx onnxslim
yolo export model=yoloe26-s-seg.pt format=onnx half=True imgsz=640 simplify=True opset=12
mv yoloe26-s-seg.onnx models/
```

**YOLOE-26 is AGPL-3.0** (Ultralytics lineage). That's a real constraint on
anything you distribute — check it before shipping.

## Controls

- **Click** — tap an object; right/Alt-click (or the ＋/− toggle) removes one
- **Box** / **Lasso** — drag or draw around what you want
- **Text** — "all the zebras", "the red bicycle". Plural or "all …" selects
  every match; definite singular selects the best one
- `Z` undo · `R` reset · `A` select everything · `E` raw/refined ·
  **Cutout PNG** downloads the selection with transparency

## The honest trade-off

YOLO-family masks come from 32 prototype basis functions at ~160×160, so their
*boundaries* are materially weaker than SAM's — published comparisons show
YOLO-class masks losing ~48–50 points of AP as the IoU threshold tightens,
against ~4 for SAM3. Detection and recall are excellent; the last few pixels
are not.

That's exactly what `edge-refine.js` repairs: a guided filter re-derives the
boundary from the photo itself. It's model-agnostic and cheap enough to run on
every instance of every selection. **The pairing is the point** — YOLOE-26
finds things fast, the post pipeline makes the edges good.

## Memory

The whole app is budgeted under **1 GB**. Running one model instead of a stack
is most of why that's reachable; the other half is that a 45 MP photo is ~340 MB
of decoded RGBA on its own, so `memory-governor.js` tracks the image, not just
the weights, and refuses a load that would break the ceiling rather than
OOM-ing the tab.

Instance masks are stored **bbox-cropped**, never full-frame — a 6000×4000
mask is ~24 MB each, so "select everything" would otherwise be gigabytes.

## Verify

```bash
bun verify.mjs
```

Two tiers. The **pure gates** always run — no browser, no network, no weights —
and are hard failures:

```bash
bun test-yoloe-core.mjs      # letterbox geometry, tensor layouts, mask assembly
bun test-select-core.mjs     # click/region/phrase rules, absent-phrase rejection
bun test-post-pipeline.mjs   # optimized hygiene + edge refinement vs a naive reference (exact)
bun test-governor.mjs        # the 1 GB ceiling under 45 MP load
```

The **browser gates** need Chromium *and* an exported model; without either
they are reported as **skipped**, never passed. They assert plumbing, not
detection accuracy — the demo scene is flat synthetic geometry and YOLOE-26 is
trained on photographs, so asserting it finds "a red circle" there would test
the fixture, not the app.

## Benchmark

```bash
bun bench.mjs                       # scans ~/Desktop, biggest files first
bun bench.mjs --images ~/Pictures/x --phrases "all the birds,the red door"
bun bench.mjs --synthetic           # DSLR-scale fixtures, no photos needed
```

Reports analyze vs select vs post latency (P50/P95, cold vs warm), peak browser
RSS and JS heap, and the absent-phrase false-positive rate. **Peak RSS above
1 GB fails the run.** Output lands in `bench-out/`.

Camera raw (`.CR2`/`.NEF`/`.ARW`/`.DNG`) and HEIC can't be decoded by Chromium —
those are listed as skipped, so export to JPEG to include them.

## Architecture

```
app.js                 UI: modes, interaction, overlay, cutout
 └─ engine-client.js   main-thread API; sticky inline fallback if the worker dies
     └─ engine-worker.js
         └─ engine.js  one contract for every mode: analyze → select → polish
             ├─ yoloe-engine.js   THE model (onnxruntime-web, WebGPU→WASM)
             ├─ yoloe-core.js     pure: letterbox, tensor decode, mask assembly
             ├─ select-core.js    pure: click/region/phrase → which instances
             ├─ mask-core.js      pure: hygiene, components, hole fill
             ├─ edge-refine.js    pure: guided-filter boundary repair
             └─ memory-governor.js  the 1 GB ceiling
```

Everything below `engine.js` except `yoloe-engine.js` is pure and dependency-
free, which is why the accuracy rules are unit-testable without a browser.

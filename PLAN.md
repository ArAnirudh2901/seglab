# SEGLAB — Full Plan

**Goal:** browser-only, zero-cloud, any-device segmentation that matches top-tier
product quality (Samsung AI-Eraser / Meta demo class). Free to build and run.
Select anything in a photo via **clicks (±) / box / lasso / text**, get a
precise, clean-edged mask, in real time.

**Hard constraints (non-negotiable):**
- Zero cloud — no image ever leaves the device. No server fallback, ever.
- **Under 1 GB RAM for the whole app**, on a normal laptop, with a DSLR file open.
- **One model.** Every mode is a query over the same instance set — no lanes,
  no per-capability network, no way for two modes to disagree.
- Any device — one model means one quality tier; weak devices pay in time, not
  in mask quality.
- Every phase ships with a headless verify gate (`bun verify.mjs`) before it counts as done.

---

## Architecture (as built)

One reference frame: photo → ≤1024 canonical canvas. All prompts, masks, and
exports live there; display scaling is CSS.

```
app.js (UI: modes, interaction, overlay, cutout)
  └─ engine-client.js (worker transport, sticky inline fallback)
      └─ engine-worker.js (dedicated worker — UI never blocks)
          └─ engine.js (ONE contract: analyze → select → polish)
              ├─ yoloe-engine.js  THE model — YOLOE-26-seg via onnxruntime-web
              │                   (WebGPU → WASM), ONE forward per image
              ├─ yoloe-core.js    pure: letterbox, tensor decode, mask assembly
              ├─ select-core.js   pure: click / box / lasso / text → which instances
              └─ post pipeline (model-agnostic, every instance):
                   seeded component cleanup + hole fill (mask-core.js)
                   → guided-filter edge-band refinement (edge-refine.js)
```

**Why one model:** YOLOE-26 is end-to-end and NMS-free, so a single forward
yields boxes, labels AND instance masks. That is the difference between an
encoder/decoder architecture — where every interaction is another decode and
every capability is another network — and this one, where the model runs once
per photo and interaction never touches it again. Click, box, lasso and text
resolve to the same instance set, so they cannot disagree about the same object.

**Why post-pipeline is now load-bearing, not a nicety:** YOLO-family masks come
from 32 prototype basis functions at ~160×160. Detection is excellent; the
boundary is the weak part (~48–50 AP lost as IoU tightens, vs ~4 for SAM3).
Guided-filter refinement re-derives that boundary from the photo itself. The
model finds things; the post pipeline makes the edges good. Neither half is
optional.

---

## Phase log — DONE (all gated by verify.mjs, 10/10 green)

| # | What | Proof |
|---|------|-------|
| 0 | Standalone app, worker engine, click ± / box, encode-once cache | disc 6.2% vs 6.2% analytic; cached decode 75 ms |
| 1 | Lasso = prompt generator (bbox + centroid point) + spatial clamp | lasso bbox = target near-pixel-perfect; clamp held |
| 2 | Minute-object selection | 9 px dot selected, 0.05% vs 0.046% expected |
| 3 | Mask hygiene: keep clicked components + ≥1%-of-largest, fill pinholes | components = 1 on disc (crumbs gone) |
| 4 | Edge-band refinement: gray guided filter, ±6 px band, soft output, E toggle | 3131 soft boundary px; post ~110 ms |
| 5 | SAM3 flagship lane: background download, hot-swap, prompt replay, sticky demote | lane=sam3 confirmed headless; encode 5581 ms / decode 630 ms |
| 6 | Post-pipeline rewrite (guide hoisting, bbox confinement, buffer pooling) | 487/487 exact vs naive reference; 12-instance query 2795→97 ms (28.7×) |
| 7 | **Unification on YOLOE-26** — one model replaces the whole SAM stack; click/box/lasso/text become queries over one instance set; 1 GB governor | 138 pure assertions green; SlimSAM + SAM3 + OWLv2 removed |

### P7 — why one model, and why this one

The plan below staged text prompts as OWLv2 (stage 1) then SAM3 concepts
(stage 2). Both are gone. Two constraints killed them:

- **1 GB whole-app ceiling.** SAM3 promptable concept segmentation has a
  browser-ready INT8 export at quality parity, but it is ~900 MB of weights
  before any ONNX Runtime arena. OWLv2-q was 155 MB *plus* SlimSAM *plus*
  SAM3-tracker — ~470 MB of encoders for three capabilities.
- **One unified workflow.** Three models meant three code paths, three failure
  modes, and modes that could disagree with each other about the same object.

**YOLOE-26** (YOLO26 + YOLOE, arXiv 2602.00168) collapses all of it: one
end-to-end, NMS-free network giving boxes, labels and instance masks in a
single forward. `yoloe26-n-seg` is 4.8M params; `s`/`m` are the browser sweet
spot. Open-vocabulary modules are **re-parameterized into the network at export
time**, so there is no text encoder at runtime — an arbitrary phrase costs
nothing but a string comparison against the exported vocabulary.

The architectural payoff is that interaction leaves the model entirely. Analyze
once, then click / box / lasso / text are pure arithmetic over the cached
instance set. The modes cannot disagree, because they resolve to the same
instances.

**The honest cost:** YOLO-family masks are prototype-based and their boundaries
are materially weaker than SAM's — ~48–50 points of AP lost as IoU tightens,
against ~4 for SAM3. The guided-filter refinement in `edge-refine.js` is the
counterweight, and the P6 optimization is what makes it affordable on every
instance of every selection. **This pairing is the whole design.**

**Licence:** YOLOE-26 is AGPL-3.0. Weights are not committed; see
`models/README.md`.

## Remaining phases (in order)

### P8 — Zoom-crop re-analysis (the DSLR equalizer)
- **Step:** when a selected instance's bbox diagonal is <~15% of the frame
  diagonal, re-run the model on a padded NATIVE-RESOLUTION crop around it and
  replace the instance with the sharper result. `select-core.js` already ships
  `needsZoomRefine` / `zoomCropRect`; the wiring is what is missing.
- **Why:** a 200 px bird in a 6000 px frame is 34 px after the canonical
  downscale and ~21 px at the model's 640 input. No network segments that well.
  This is worth more accuracy on DSLR files than any model swap.
- **Gate:** synthetic 6000×4000 scene with a 60 px object — boundary-F improves
  vs no-crop, and peak RSS stays under the ceiling.

### P9 — Analysis persistence (OPFS)
- **Step:** persist the instance set (compact planes + labels) to OPFS keyed by
  content hash; analyze at import rather than at first interaction.
- **Why:** kills the last wait. Reopening a photo is instant selection with no
  model run at all. Instance planes are small, unlike SAM embeddings — this is
  much cheaper here than it would have been in the old architecture.
- **Gate:** second page-load of the same image selects with `analyzed=false`.

### P10 — Granularity
- **Step:** YOLOE returns whole objects from its vocabulary. Clicking a shirt
  selects the person. Add Tab-cycling through containing instances (shirt →
  person → group) using the nesting already computed by `pickByPoint`.
- **Why:** the honest capability gap vs SAM, which offers part/whole/sub-part
  candidates per click. Nesting recovers most of it without another model.
- **Gate:** Tab on an overlapping region changes the rendered mask.

### P11 — Erase / edit actions on the mask (deferred by design)
- **Step:** consume the mask: erase (LaMa-class inpainting ONNX in-browser),
  cutout compositing, background swap.
- **Why:** deferred earlier — the segmentation contract (mask ImageData + soft
  edges) is already the right input.
- **Gate:** erase the demo disc → background continuity metric on the hole region.

### P12 — Platform slots (when the web catches up)
- **COOP/COEP headers** if a hosted deployment wants multi-threaded WASM.
- **WebNN execution provider** the day it ships stable — one line in the EP
  ladder in `yoloe-engine.js`, unlocks phone NPUs. Do not build on it before then.

## Device tier matrix (end state)

| Device | All four modes | Feel |
|---|---|---|
| WebGPU (desktops, M-Macs, iOS 26 Safari, Chrome/Android 12+) | YOLOE-26-s/m fp16 | analyze once, then instant |
| WASM-only (old phones, exotic browsers) | same model, same masks | slower first analysis, identical afterwards |
| Any, revisited photo | cached analysis (P9) | instant, no model run |

There is no quality tier any more. One model means every device gets the same
masks; weak devices only pay in the one-time analysis.

## Risks (with odds)

| Risk | Odds | Mitigation |
|---|---|---|
| YOLOE-26 boundaries too coarse for cutout-grade work | ~35% | guided-filter refinement is the counterweight; P8 zoom-crop for small objects; measure with `bun bench.mjs` before believing either way |
| AGPL-3.0 blocks a distribution plan | real, not probabilistic | decide before building on it; the engine boundary is one file (`yoloe-engine.js`) if a swap is ever needed |
| ONNX export shape differs from what `yoloe-core` expects | ~30% | both tensor layouts handled and unit-tested; mismatched dims throw loudly rather than decode garbage |
| Vocabulary too narrow for open-ended phrases | ~40% | ship the RAM++ 4,585-tag set; text honestly reports "no vocabulary" rather than matching nothing |
| No part/sub-part granularity vs SAM | certain | P10 nesting recovers most of it; a real capability loss otherwise |

## Working rules
- Every phase lands with a `verify.mjs` check before it's called done.
- Post-pipeline stays model-agnostic — anything added must upgrade all lanes.
- Never trade the zero-cloud constraint for quality; trade download seconds instead.

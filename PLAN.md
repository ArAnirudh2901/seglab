# SEGLAB — Full Plan

**Goal:** browser-only, zero-cloud, any-device segmentation that matches top-tier
product quality (Samsung AI-Eraser / Meta demo class). Free to build and run.
Select anything in a photo via **clicks (±) / box / lasso / text**, get a
precise, clean-edged mask, in real time.

**Hard constraints (non-negotiable):**
- Zero cloud — no image ever leaves the device. No server fallback, ever.
- Any device — weak phones get a working tool, strong devices get flagship quality.
- Free — open weights, free CDN delivery, user's own compute.
- Every phase ships with a headless verify gate (`bun verify.mjs`) before it counts as done.

---

## Architecture (as built)

One reference frame: photo → ≤1024 canonical canvas. All prompts, masks, and
exports live there; display scaling is CSS.

```
app.js (UI: modes, prompts, overlay, replay)
  └─ sam-client.js (worker transport, sticky inline fallback)
      └─ sam-worker.js (dedicated worker — UI never blocks)
          └─ sam-engine.js (TWO LANES, one segment() contract)
              ├─ draft:    SlimSAM-77 quantized (~14 MB, Apache-2.0) — everywhere, loads in seconds
              ├─ flagship: SAM3-tracker q4f16 (297+5.4 MB, SAM License) — WebGPU, background download, hot-swap + prompt replay
              └─ post pipeline (model-agnostic, every decode):
                   lasso clamp → seeded component cleanup + hole fill (sam-core.js)
                   → guided-filter edge-band refinement (edge-refine.js)
```

**Why two lanes:** encoders are heavy, decoders are tiny. The draft lane makes the
tool instantly usable; the flagship encodes once per image (~5.6 s, cached) and
then every click is a ~630 ms decode. Weak devices pay in background seconds,
not in quality ceiling. The user's clicks are *prompts*, so they replay
losslessly when a better lane arrives.

**Why post-pipeline instead of bigger models only:** the 256² mask grid is a
model-family limit. Hygiene + image-guided edge refinement fix artifacts that
NO encoder size fixes, and they upgrade every current and future lane.

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

---

## Remaining phases (in order)

### P6 — Text prompts, stage 1: OWLv2 → boxes → same decoder
- **Step:** add `Xenova/owlv2-base-patch16-ensemble` (Apache-2.0, in transformers.js)
  as a `detect(text)` op in the engine; each detected box feeds the existing
  SAM decoder; union masks for multi-instance ("all zebras").
- **Why:** works on EVERY device today (no new licensing, no export work), and
  reuses the whole existing mask pipeline. Enables the disabled Text button.
- **Gate:** demo scene gains labeled objects; text "red circle" → same bbox/coverage
  bars as the click test; multi-instance returns N components.
- **Honest limit:** weaker than true SAM3 concepts on rare/tiny objects — stage 2 fixes that.

### P7 — Candidate cycling + first-click granularity
- **Step:** keep all 3 decoder candidates; Tab cycles part → whole → sub-part;
  on a single first click, prefer the largest high-scoring candidate.
- **Why:** the #1 cause of wasted corrective clicks is the model picking "part"
  when the user meant "whole" (bike test: cost ~3 clicks).
- **Gate:** verify asserts the 3 candidates differ and cycling changes the rendered mask.

### P8 — Zoom-crop re-encode (minute-object equalizer)
- **Step:** when the active selection bbox < ~15% of frame diagonal, re-encode a
  padded crop around it at full 1024 and decode there; paste refined result back.
- **Why:** a 100 px object in a 4000 px photo lands on ~25 px of encoder input —
  no model segments that well. Cropping is up to ~10× effective resolution on
  exactly the thing being selected. This, not model choice, is the real
  "minute thing" solution for large photos.
- **Gate:** synthetic scene at 4000 px with a 60 px object: boundary-F improves vs no-crop.

### P9 — Embedding persistence + encode-at-import (OPFS)
- **Step:** persist flagship embeddings (~33 MB/image) to OPFS keyed by content
  hash; start encoding at photo import (not first click); load embeddings on
  revisit instead of re-encoding.
- **Why:** kills the last wait. Reopening a photo = flagship-quality clicks with
  ZERO encode wait, even on weak devices. Nobody in browser segmentation does this.
- **Gate:** second page-load of same image: first click decodes with `encoded=false`.

### P10 — Text prompts, stage 2: true concept segmentation on-device (the flag-plant)
- **Step:** quantize the SAM3 text stack (354M text encoder + ~30M detector) to
  4-bit via the calibrated-WOQ recipe (per-block sensitivity scan, mixed
  precision), targeting <300 MB for the full text lane. Primary vehicle:
  **EfficientSAM3 / SAM3-LiteText** (Apache-2.0, weights released, ONNX export
  still an upstream TODO — we do the export = first in a browser). Fallback
  vehicle: samexporter on facebook/sam3 + custom quantization.
- **Why:** "every zebra in one prompt" at SAM3 quality, fully local — no shipped
  product has this in a browser. Odds: 60–75% the export+quant clears the
  gates; if it misses, OWLv2 (P6) remains the text path and nothing regresses.
- **Gate (hard, from the approved plan):** vs fp16 reference — mean mIoU ≥ 0.98
  (min 0.95/fixture) AND boundary-F(2 px) ≥ 0.95, offline eval BEFORE any UI work.

### P11 — Erase / edit actions on the mask (deferred by design)
- **Step:** consume the mask: erase (LaMa-class inpainting ONNX in-browser),
  cutout compositing, background swap.
- **Why:** deferred earlier ("forget about erasing, we will do it later") — the
  segmentation contract (mask ImageData + soft edges) is already the right input.
- **Gate:** erase the demo disc → background continuity metric on the hole region.

### P12 — Platform slots (when the web catches up)
- **COOP/COEP headers** if a hosted deployment wants multi-threaded WASM (faster
  draft lane on weak devices). Local `http.server` skips this.
- **WebNN execution provider** the day it ships stable (verified: Chrome origin
  trial only, Android excluded, realistic 2027) — one config line in the EP
  ladder, unlocks phone NPUs. Do not build on it before then.

---

## Device tier matrix (end state)

| Device | Click/box/lasso | Text | Feel |
|---|---|---|---|
| WebGPU (desktops, M-Macs, iOS 26 Safari, Chrome/Android 12+) | SAM3 q4f16 | LiteText (P10) or OWLv2 | flagship |
| WASM-only (old phones, exotic browsers) | SlimSAM + full post pipeline | OWLv2 | draft lane, same UX, softer masks |
| Any, revisited photo | cached embeddings (P9) | same | instant |

## Risks (with odds)

| Risk | Odds | Mitigation |
|---|---|---|
| EfficientSAM3 ONNX export fights back | ~30% | fallback: SAM2.1 ONNX (proven in-browser) for the mid lane; OWLv2 keeps text alive |
| q4 text-stack quality below gates | ~25–40% | gates decide — ship OWLv2 as text default, keep flagship clicks |
| WebGPU driver flakiness in the wild | ongoing | already handled: sticky lane demote + WASM fallback, user never sees a dead click |
| SAM License (flagship lane) vs product plans | low | tracker lane is commercial-OK; Apache alternatives (SlimSAM/EfficientSAM3) exist for every lane |

## Working rules
- Every phase lands with a `verify.mjs` check before it's called done.
- Post-pipeline stays model-agnostic — anything added must upgrade all lanes.
- Never trade the zero-cloud constraint for quality; trade download seconds instead.

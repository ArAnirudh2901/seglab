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

## Architecture (as built — post memory-safety rewrite, 2026-07)

One reference frame: photo → **≤768 px proxy** (lite policy, js/policy.js).
The original exists only as its compressed Blob; all prompts, masks, and the
overlay live in the proxy frame; display scaling is CSS.

```
Compressed original Blob (asset custody in app.js `doc`)
  → decode-worker (bounded header parse via image-io.js → resize-during-decode)
  → ≤768px proxy canvas
  → heavy-job-queue.js — ONE lane, concurrency 1, revision-cancelled:
       proxy-decode → model-warm → encode/prompt-decode → cv-refine → export
  → sam-worker (SlimSAM only by default; ONE embedding slot, tensors disposed)
  → cv-refine-worker (C++/wasm cleanup + guided-filter edges; JS fallback)
  → one-channel Uint8 alpha masks everywhere; RGBA only at render/export
SAM3 flagship: DORMANT — explicit confirmed opt-in only, never automatic.
```

**Why one lane + one embedding:** on an 8 GB machine the crash mode was
concurrent peaks (full-res decode × model warm × background SAM3 download).
Strict serialization plus a single disposed embedding slot makes peak memory
predictable; quality is preserved by the post pipeline, not by residency.

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

## Memory-safety rewrite — DONE (2026-07, all gated by verify.mjs, 60+ checks green)

| M | What | Proof |
|---|------|-------|
| M0 | Lite policy forced on ≤8GB/unknown; URL flags clamped; SAM3 de-automated (explicit `startFlagship` only); proxy 1024→768 | `?flagship=1&profile=ultra&proxy=max` inert; zero sam3 requests incl. bypass flags |
| M1 | All runtime assets vendored (`scripts/download-models.mjs`: transformers.min.js, 4 ORT wasm, SlimSAM fp32+q8) | full selection with huggingface.co + jsdelivr route-aborted in a fresh cache-less context |
| M2 | Shared heavy-job queue, concurrency 1, priorities, revision cancellation | log-proven: max 1 active ever; rejected job no deadlock; interaction preempts speculative |
| M3 | Bounded decode pipeline: image-io header parsers (EXIF/ISOBMFF/TIFF+RAW preview), decode-worker, Blob-only custody | 6000×4000 → 768×512 proxy; import race: second wins; no main-thread full decode |
| M4 | Revision protocol end-to-end; ONE embedding slot with `Tensor.dispose()`; one-channel alpha mask contract | resident=1 across N imports; import alone never encodes; stale results never commit |
| M5 | C++/wasm cv_refine (seeded cleanup, holes, open/close; ≤768; fixed 16 MB heap) + cv worker with JS fallback | wasm suite green (dims reject, hole fill, min-area, transfer-not-copy); route-abort → JS fallback keeps selection |
| M6 | Bounded export (≤4096px/8MP from Blob), pressure L1–L3, `[seglab][memory]` telemetry, SAM3 opt-in UX | 24 MP → 3464×2309 reduced=true; L2 kills wasm refine, selection survives; opt-in confirmed warn-only |
| M7 | Service worker (cache-on-fetch, versioned), docs, static memory-contract suite | getImageData whitelist, no toDataURL/pyodide/opencv/WORKING_MAX_SIDE, sw precache-free |

Deferred by design: crop escalation (P8) stays OFF under lite; guided filter
in C++ (spec M6-optional) and ROI GrabCut (M7-optional) remain unimplemented
until measurements demand them — the JS guided filter already runs in the cv
worker off-main-thread.

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

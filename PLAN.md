# SEGLAB — Full Plan (merged with Phosmith Offline Pro)

**Goal:** browser-only, zero-cloud, any-device segmentation that matches top-tier
product quality (Samsung AI-Eraser / Meta demo class). Free to build and run.
Select anything in a photo via **clicks (±) / box / lasso / text**, get a
precise, clean-edged mask, and export it at the photo's **native resolution**.

This plan merges the original SEGLAB roadmap with the *Phosmith Offline Pro
Implementation Plan* (2026-07). Its cloud-fallback sibling spec was rejected:
zero-cloud stays absolute.

**Hard constraints (non-negotiable):**
- Zero cloud — no image, prompt, or mask ever leaves the device. No server fallback, ever.
- Any device — equal FEATURES everywhere; hardware buys seconds and proxy size, never capability.
  Target device for the "no loss" bar: M2 MacBook Air base (8 GB) = the dev machine.
- Free — open weights, free CDN delivery, user's own compute.
- Every phase ships with a headless verify gate (`bun verify.mjs`) before it counts as done.

---

## Architecture (target)

```
Original asset (native res — the export truth; asset-store.js)
  ├─ interaction proxy ≤1024 (#view canvas — clicks/box/lasso/overlay)
  ├─ detector canvas (OWLv2 model input FIXED 960² — canvas 960 Std/Pro, 640 Lite)
  └─ refinement crops (original-res, ≤1024 model input — small objects + HD export)

app.js / text-ui.js / export-hd.js (UI, candidates, HD export orchestration)
  └─ sam-client.js  (worker transport + inline fallback; revisioned ops)
      └─ sam-worker.js (serialize queue: ONE heavy GPU job at a time)
          └─ sam-engine.js — lanes + provider registry, budget-aware residency
              ├─ draft:    SlimSAM-77 quantized (~14 MB, Apache-2.0) — everywhere
              ├─ flagship: SAM3-tracker q4f16 (297+5.4 MB) — WebGPU, hot-swap + replay
              ├─ textLite: OWLv2 (Apache-2.0) via detect-engine.js — boxes only
              └─ post pipeline (model-agnostic, every decode):
                   lasso clamp → seeded cleanup + hole fill (sam-core.js)
                   → guided-filter edge-band refinement (edge-refine.js)
policy.js (budgets: Lite/Standard/Pro — static presets now, capability probe in M4)
```

**Why two segmentation lanes:** encoders are heavy, decoders are tiny. Draft
makes the tool instantly usable; flagship encodes once (~5.6 s, cached) and
every click is a ~630 ms decode. Prompts replay losslessly when a better lane
arrives.

**Why a text DETECTOR (not a text segmenter):** interaction-aware routing — a
click already contains its own localization, so clicks never touch the
detector; text is the only route that needs text-conditioned boxes, and boxes
feed the SAME decoder every other prompt uses.

**Why the post pipeline:** the 256² mask grid is a model-family limit. Hygiene
+ image-guided edge refinement fix artifacts no encoder size fixes, and they
upgrade every current and future lane.

---

## Phase log — DONE (all gated by verify.mjs)

| # | What | Proof |
|---|------|-------|
| 0 | Standalone app, worker engine, click ± / box, encode-once cache | disc 6.2% vs 6.2% analytic; cached decode 75 ms |
| 1 | Lasso = prompt generator (bbox + centroid point) + spatial clamp | lasso bbox = target near-pixel-perfect; clamp held |
| 2 | Minute-object selection | 9 px dot selected, 0.05% vs 0.046% expected |
| 3 | Mask hygiene: keep clicked components + ≥1%-of-largest, fill pinholes | components = 1 on disc |
| 4 | Edge-band refinement: gray guided filter, ±6 px band, soft output, E toggle | 3131 soft boundary px; post ~110 ms |
| 5 | SAM3 flagship lane: background download, hot-swap, prompt replay, sticky demote | lane=sam3 headless; encode 5581 ms / decode 630 ms |

---

## Merged roadmap — SHIPPED (in order; each phase verify-green, then committed)

All seven phases are on `main`, each with its verify gate passing at commit
time. Full run: `bun verify.mjs` (Phases T, A–A8, B).

| Phase | Commit | Gate proof |
|-------|--------|------------|
| ENV | `798e2d3` | local playwright+chromium; missing playwright = hard fail |
| M0  | `e217f9e` | overlapping selections → only the newest commits |
| M1  | `d0a86e6` | 2400 px export dims exact; radial err 1.5 px ≤ 3 px |
| M2  | `73117e6` | box → mask same bars as clicks; "all" → 2 components |
| M3  | `92bb517` | 4000 px scene: escalated 1.0 px vs 5.3 px control |
| M4  | `f6d48b3` | dev machine probes `pro`; `?force=wasm` runs full pipeline |
| M5  | `8b62f07` | offline: click+text+export, 0 net fetches; revisit `encoded=false` |

### ENV — verify gate must be real on this machine ✅ `798e2d3`
Local playwright dev-dependency + chromium; drop the foreign hard-coded path;
missing playwright = hard fail (CI_SKIP_BROWSER=1 is the only skip).

### M0 — Contracts: revision, cancel, serialize, policy stub ✅ `e217f9e`
`document.revision` carried through client → worker → engine; `cancel` op
(ONNX kernels can't be interrupted → in-flight jobs skip the post pipeline and
return `{stale:true}`; queued jobs drop at dequeue; stale can never commit).
`serialize()` queue = one heavy GPU job at a time (also caps 8 GB transient
peaks). `policy.js` static Standard budget + URL overrides. Cache key scheme
`doc:${assetHash}` / `crop:${assetHash}:${rect}`.
**Gate:** overlapping selections → only the newest commits; older reported stale.

### M1 — Original asset + cropSegment + HD export (the "no loss" keystone) ✅ `d0a86e6`
Today the original bitmap is CLOSED after building the ≤1024 proxy and the
cutout exports from the proxy — native pixels are discarded. `asset-store.js`
takes custody (Bitmap ≤24 MP, else Blob) + AssetTransform. New engine
primitive `cropSegment` (padded original-res crop → encode under `crop:` key →
prompts transformed → decode → post pipeline) — shared by HD export, M3
escalation, and future "Improve detail". `edge-refine.js` gains a tiled
entry (2048² tiles, 64 px overlap — window-local filter ⇒ seamless; never
full-frame at 12 MP). Export: crop decode when meaningfully sub-frame +
band-tiled guided refine at `band = clamp(round(6·scale), 6, 16)` → alpha
composite → native-resolution PNG.
**Gate:** 2400×1600 demo — export dims exact; radial boundary error ≤ 3 px vs
the analytic disc.

### M2 — Text select, local on every device (OWLv2) ✅ `73117e6`
`Xenova/owlv2-base-patch16-ensemble` (verified present in the pinned
transformers.js 4.2.0 build). `text-core.js` (pure: phrase normalization with
the "a photo of a X" template, all/every → multi-intent, NMS, mapping),
`detect-engine.js` (pipeline-first adapter, fp16 WebGPU / q8 WASM, ALL phrases
in one call, detection cache), `text-ui.js` (debounced input → numbered
candidate chips → pick one / All N / re-phrase / Esc; state machine
IDLE→DETECTING→CANDIDATES→SEGMENTING→READY). Chosen boxes → the SAME segment
path as lasso (box + center point) → union via seeded hygiene. Residency:
Lite disposes the detector after boxes (one-heavy rule); Standard idles it
out ~10 s. Demo scene gains embedded photographic thumbnails (OWLv2 is
out-of-distribution on flat synthetic shapes — photo patches carry the gates).
**Gate:** text prompt lands a candidate on the photo patch → selecting it
passes the same bbox/coverage bars as clicks; shape prompts WARN-only.

### M3 — Crop pyramid: interactive small-object escalation ✅ `92bb517`
After post pipeline: bbox diag < 15% of proxy diag OR near-empty/solid →
exactly ONE auto `cropSegment` escalation (original-res padded crop), merged
back to the proxy mask + kept as an original-res `hdPatch` that HD export
reuses. Auto on Standard/Pro; Lite = manual action (later).
**Gate:** 4000 px scene, 60 px object — radial error/coverage materially
better than a `?escalate=0` control run.

### M4 — Capability probe fills policy.js ✅ `f6d48b3`
Probe at boot (WebGPU adapter, deviceMemory, 512 px warm-up encode timing) →
Lite/Standard/Pro budgets (proxy 768/1024/1024; detector canvas 640/960/960;
flagship off/on/on; maxResidentHeavy 1/2/2; original Blob/Bitmap/Bitmap;
HD export filter-only/full/full; detector dispose now/idle/resident).
Pressure ladder: dispose OWLv2 → drop flagship embeddings → drop flagship
session (extends the existing sticky demote). Memory check (Pro, 12 MP):
~1.0 GB steady, ~2.6 GB transient peak, serialized — fits 8 GB.
**Gate:** `?force=wasm` completes click + text + export (the "entire pipeline
on weak devices" proof); default run on the dev machine reports `pro`.

### M5 — Kill the waits + prove zero-cloud ✅ `8b62f07`
Encode-at-import (eager background `encode` op — first click hits cache).
OPFS persistence of document embeddings keyed by `lane:contentHash`
(write-then-move, ~500 MB byte-capped LRU; read failure ⇒ re-encode). Crop
keys stay memory-only. **Gates:** (A) warmed profile + `setOffline(true)` +
request log ⇒ click/text/export still pass with zero successful fetches —
zero-cloud as a tested property, not a promise (route-interception would
bypass the HTTP cache and false-fail; models live in CacheStorage which serves
offline); (B) revisit the same image ⇒ import encode + first click both
`encoded=false`, restored tensors decode a correct disc.

Note: the plan named *flagship* embeddings; shipped persisting whichever lane
is active (draft in headless, flagship on WebGPU) since both are `doc:`-keyed
and the pack format is lane-agnostic — a revisit skips the encode either way.

---

## Future (after this tranche)

- **Candidate cycling (old P7):** keep all 3 decoder candidates; Tab cycles
  part → whole → sub-part; first click prefers the largest high-scoring one.
- **SAM3 text stack (old P10):** quantized 354M text encoder + detector as a
  drop-in behind the same `detect()` contract; hard offline gates (mIoU ≥ 0.98,
  boundary-F ≥ 0.95 vs fp16) BEFORE any UI work. OWLv2 remains the floor.
- **Detail pack (optional):** local trimap → ViTMatte-class matting for
  hair/glass, Pro-tier, explicit user action — roadmap-only by decision.
- **Erase / edit (old P11):** LaMa-class inpainting consuming the mask.
- **Platform slots (old P12):** COOP/COEP for threaded WASM; WebNN when it
  actually ships (realistic 2027).
- **Lite "Improve detail" button:** manual crop escalation for devices where
  auto is off.

## Acceptance targets (engineering gates, not product claims)

| Operation (warm) | Lite | Standard | Pro (M2 Air) |
|---|---|---|---|
| Correction click | ≤2 s | ≤700 ms | ≤700 ms flagship / ≤300 ms draft |
| Text → boxes | ≤10 s | ≤5 s | ≤2.5 s |
| HD export (12 MP) | ≤10 s filter-only | ≤6 s | ≤6 s |
| Cached mask reuse | <100 ms | <100 ms | <100 ms |

## Risks (with mitigations)

| Risk | Mitigation |
|---|---|
| Pinned transformers.js 4.2.0 OWLv2 API surface differs | M2 step-0 smoke test; adapter isolates it; raw-model fallback |
| OWLv2 weak on synthetic demo shapes | photo thumbnails carry the text gates; shape prompts WARN-only |
| WASM detect latency on weak devices (8–20 s cold) | progress UI + phrase cache; contingency: owlvit-patch32 Lite pack |
| HD quality where crop decode is skipped (frame-filling objects) | radial-error gate decides; ≤2 crop decodes before filter-only |
| 8 GB pressure with three resident sessions | serialize queue; OWLv2 idle-dispose; embeddings evict first |
| WebGPU driver flakiness in the wild | already handled: sticky lane demote + WASM fallback |

## Working rules
- Every phase lands with a `verify.mjs` check before it's called done; commit per phase.
- Post-pipeline stays model-agnostic — anything added must upgrade all lanes.
- Never trade the zero-cloud constraint for quality; trade download seconds instead.
- Deterministic gates ride the draft lane; flagship checks are WARN-only headless,
  confirmed manually in real Chrome on the dev machine.

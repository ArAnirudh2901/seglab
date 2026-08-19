# SEGLAB — Full Plan (merged with Phosmith Offline Pro)

> **2026-08-18 — the open-vocabulary text lane was removed.** "AI Text" (the
> YOLOE text-prompt detector + MobileCLIP2 text tower, its taxonomy /
> autocomplete / refine-chip UI, the disposable detect worker, and every budget
> knob that sized it) is gone from the app, the gate and the model downloader.
> Everything below that describes it is history, kept for what it measured.

## 2026-07-23 — measured-headroom climb: high-end devices auto-reach `standard` (SHIPPED)

The auto-tier ceiling was standard8 for every plain browser — a 64 GB gaming
tower reads `deviceMemory: 8` (privacy cap), so boot signals can never prove
more. The governor's `onHeadroom` (4 clean 2.5 s cycles, measured bytes
< 0.5×budget, drift < 500 ms) was telemetry-only; it now drives
`policy.climbBudget`: a live standard8 **auto** tier climbs ONE step to
`standard` (`profileSource: 'auto-climb'`), unlocking native-res export, auto
escalation and 24 MP caps. Fields headroom can't vouch for are masked to the
standard8 stance: `samIdleMs` 300 s, `embedPersist` false, `draftCacheMax` 1,
`memBudgetMB` 1900 (the climb never raises the governor ceiling that
authorized it), `flagship` false. Escalation on a climbed budget additionally
requires headroom measured in the last 30 s. Any pressure event demotes the
climbed tier back to standard8 BEFORE the ratchet applies (down always wins)
and poisons re-climbing for the session; the climb state is page-lifetime and
re-earned every session. All budget re-resolutions (probe, Phosmith update,
profile/scale toggles) funnel through one `recomputeBudget()` that re-applies
or correctly drops a latched climb (manual override and trust flips win).

Also in this change, from the same review:
- **Export predicate** (the ultrareview finding): keeping the detect worker
  warm across export now needs POSITIVE proof — not-wasm AND pressure 0 AND
  (on memory-locked budgets) headroom measured within 30 s. Trusted hosts
  (vouched memory, often no measure API) keep absence-of-pressure semantics.
  A new import zeroes the freshness stamp.
- **Visibility shed no longer ratchets**: backgrounding a tab called
  `shedMemory(1)` whose pressure ratchet is one-way — a single tab switch
  permanently disabled eager-encode + the GPU detector lane. It now disposes
  reloadables (`relievePressure(1)`) without touching the budget.
- **Detector scale fallback**: yoloe/yolo-world lanes retry at scale 's' when
  a larger scale's files aren't deployed (the footer switcher can't strand
  search on a partial deploy; vocab follows the actually-loaded scale).

Gates: pure climb matrix (eligibility × masks × URL-lowering survival) + live
climb/demote/poison gate driven through `__seglabGovernor.feedMeasurement` +
`cycleNow` with a stubbed non-fallback adapter (headless SwiftShader is a
fallback → real path ineligible in CI) + manual-override negative + visibility
no-ratchet gate. Full suite green.

## 2026-07-23 — GPU-first detect lanes, int8 wasm floor, 8 GB import/search hardening (SHIPPED)

Models are now fully materialized locally (previously the YOLO/CLIP lanes 404'd
on a fresh checkout): YOLOE-26 s/m/l exported (46/99/117 MB fp32 + 4585-label
vocabs), YOLO-World s/m/l (51/114/187 MB fp32), the CLIP ViT-B/32 vocab table
(4585×512 fp32, 9.4 MB — transformers 5.2's `position_ids` UNEXPECTED warning
is benign, table validated end-to-end: 'cat' 0.925 / 'television' 0.919 /
'banana' → ∅ on COCO fixtures), and `lib/ort-web/` vendored via
`download-models --detector`.

**GPU-first everywhere (user requirement):** both detect lanes take an EP
preference and default `['webgpu','wasm']` — Apple/NVIDIA/AMD and Intel/AMD
iGPUs all ride WebGPU; wasm is the floor, and the pressure ratchet's
`detectorWebGPU=false` now actually pins it (the lanes previously ignored it).
YOLO-World's old wasm-first ordering (unverified GPU parity) is replaced by
runtime trust: a flooded score field (≥2000/8400 anchors past threshold) or a
first-ever zero-det GPU result is cross-checked once on wasm; disagreement
sticky-demotes that worker to wasm, agreement trusts the GPU thereafter. Cost:
≤1 extra wasm inference per worker spawn.

**int8 wasm floor (QDQ):** `quantize_lanes` static-quantizes backbone+neck
(QDQ QUInt8/QInt8 per-channel, MinMax over 36 COCO frames through the exact
browser preprocessing; detect head excluded — whole-graph collapsed to zero
dets, head-included MinMax/Entropy both hallucinated 0.88 'stormy'/'night sky'
labels). Shipped: YOLO-World s/m/l `.int8.onnx` (~20 MB vs 51 fp32 at s; drift
≤0.05, IoU ≥0.99, 'cat' 0.974) — wasm loads int8-first, fp32 fallback; WebGPU
keeps fp32. YOLOE int8 was REJECTED (same hallucinations persisted with head
fp32) — YOLOE stays fp32 on both EPs; correctness beats the 18% CPU win in the
lane whose labels drive selection.

**Search latency:** one detector frame per document (`text-ui` caches the 640²
letterbox keyed by transform identity — repeat searches skip the bounded
re-decode + readback) and the worker now receives a disposable copy, fixing a
real bug: the transfer detached `frame.data` before `focusRequestedColor`/
`dominantColorForBox` read it, so colour-qualified ranking and the colour facet
were silently dead on the worker path.

**8 GB import spike (Z 8 NEF, red memory pressure):** the import path was
already guarded (eager encode is calm-gated, pressure-checked, and `gpuOnly`
on unverified budgets — sam-engine returns `{skipped:'wasm-lane'}`), so the
observed freeze is the FIRST CLICK paying the wasm encode arena (or a stale
deploy). Two residual fixes: (1) the wasm SlimSAM session sets
`enableCpuMemArena:false` + `enableMemPattern:false` — ORT's BFC arena
over-reserves in big chunks, and its padding was permanent heap growth on the
exact lane that is the memory floor (measured resident after first encode:
~3 GB → 1.83 GB). (2) **worker recycle**: the engine's arena release frees
ORT's view but the worker's grown wasm Memory stays mapped forever (wasm
memory never shrinks) — so idle hibernate and pressure ≥ 3 on the wasm lane
now `recycleWorker()` (sam-client): deliberate termination, doesn't count
toward crash restarts, next selection rebuilds from cached weights. Measured:
resident 1771 MB → **20 MB** after release. GPU sessions keep ORT defaults and
are never recycled. (The 1024-edge encode transient itself is irreducible:
SlimSAM's processor resizes every input to 1024.)

**Measured (real Chrome 138+, 51 MP JPEG fixture, `scripts/profile-chrome.mjs`):**

| | GPU (WebGPU) | wasm floor (`--disable-gpu` + `?force=wasm`) |
|---|---|---|
| import 51 MP | 222 ms · 3 MB | 215 ms · 3 MB (eager encode skipped) |
| first click | 336 ms | 1.57 s (warm+encode) |
| second click | 376 ms | 333 ms |
| YOLOE "cat" cold/warm | 682 / 211 ms | 889 / 367 ms |
| YOLO-World "cat" | 350 ms fp32 | 540 ms **int8 QDQ loads on ort-web wasm** |
| peak measured bytes | 367 MB | ~1.9–2.6 GB transient (encode+detect) |
| resident after release | 206 MB | **20 MB** (`freed=[embedding,arena,worker]`) |

Verify: now **118 ok / exit 0** (was 114 ok + 4 pre-existing failures from the
YOLOE merge, confirmed identical on a clean HEAD baseline). Triage outcome:
ONE real app regression — the merge moved detectors into the disposable
detect-worker but nothing terminated it under memory pressure anymore; fixed
in sam-client.relievePressure (level ≥ 1 disposes the detect worker first and
reports 'detector' — guaranteed non-resident after the call). The THREE export
gates asserted the retired full-frame-PNG contract; export-hd's deliverable is
deliberately the PADDED CROP RECT at native res (bbox + max(24, 6%·diag) —
verified against observations: disc@2400→655, disc@4200→1147, dot@5000→148).
Gates rewritten to that contract with real teeth: std8-vs-lite now runs a
5600 px scene where cropMaxSide genuinely bites (lite 1280-capped composite vs
std8 1530 HD-decoded — measured exactly that), and pro/working-copy assert the
native-res rect ±slack plus the existing decode/alpha/radial checks.
Tools: `scripts/quantize-detect-lanes.py` (CALIB_DIR of photos → int8),
`scripts/profile-chrome.mjs [gpu|wasm]` (real-Chrome memory/latency drive,
fixture `models/test-fixtures/dslr-45mp.jpg`).

## 2026-07-17 — RAW develop fixture: first real sensor through LibRaw in the suite (CLOSED)

The fixture-gated develop phase had never decoded a real RAW. Closed with
`d750-lossless.nef` (repo root, untracked via `*.NEF`): Nikon D750 14-bit
Lossless sample, CC0 from raw.pixls.us, sha256 `f90deb28…7665` verified against
the index. Full suite green — 101 checks — including both fixture phases:
- parser lifts the embedded preview (6016×4016, 975 KB JPEG + 746 KB proxy);
- develop demosaics the sensor on-device: 6016×4016 → half-size 3016×2016 JPEG
  (551 KB), decode-verified end to end.
Rerun: `RAW_FIXTURE=d750-lossless.nef bun verify.mjs`.

No preview-stripping (exiftool) needed — the parser phase REQUIRES previews
present, and the develop hook (`developRawUrl`) drives LibRaw directly, so one
ordinary lossless RAW satisfies both.

Finding: the repo's Z 8 NEF (`2680558334.nef`) is High Efficiency (TicoRAW).
LibRaw can't decode it ("Unsupported file format"), so HE/HE* can never use the
develop fallback. By design this doesn't matter: HE NEFs always embed full-size
previews (the Z 8 file carries three; parser phase passes on it), so the fast
path covers them — the fallback exists for older preview-less files, which
predate HE. A Z 8 set to Lossless compression produces develop-eligible files.

## 2026-07-16 — on-device RAW develop: LibRaw wasm fallback for preview-less RAW (SHIPPED)

Closes the last gap in RAW support. The fast path (`image-raw.js`) still lifts
the camera's embedded JPEG preview for essentially every modern RAW with no
sensor decode. The **rare RAW with no usable embedded preview** — which used to
hard-fail with "export a JPEG/TIFF" — now falls back to an on-device develop:
**LibRaw 0.22.2** compiled to WebAssembly with Emscripten (`cpp/raw_develop.cpp`
→ `public/wasm/raw-develop.wasm`, ~1.0 MB). The worker demosaics the sensor and
**libjpeg re-encodes in-worker**, so only a compact JPEG Blob rejoins the normal
decode-to-proxy path — a full-res RGBA frame never crosses threads.

Memory-safe by construction, respecting every locked-budget rule:
- `half_size` demosaic (a ≤1024 px masking proxy never needs more; ~4× smaller).
- Megapixel cap checked **before `unpack()`** (lite 40 · std 60 · pro 80 · ultra
  100), so an image-bomb sensor is refused before LibRaw allocates it.
- Lazy (~1 MB module fetched only when a preview-less RAW opens; never at boot,
  never precached in `sw.js`) and the worker is **terminated after each develop**
  because wasm memory only grows — termination returns it to the OS.
- Off at memory pressure ≥ 2; SIMD-only build; failure surfaces the old error.

Library choice, fact-checked against the source: RawSpeed (decoder only, no
demosaic; speed-first, OOM-prone) and rawpy/imagecodecs (Python — no runtime,
backend rejected twice) don't fit a zero-cloud browser. LibRaw is the only
option that does and is itself hardened against giant/malicious allocations.

Gates: static (artifacts exist; lazy+disposed+pressure-gated wiring; JPEG-Blob
handoff, no RGBA across threads; `half_size` present) run always; a fixture-gated
browser phase (`RAW_FIXTURE=…`) drives the real worker→wasm→LibRaw→libjpeg chain
and asserts a decodable JPEG at the sensor's developed dimensions.

## 2026-07-16 — crisp preview: proxy → 1024 + display decoupled from the model (SHIPPED)

Two changes, both verify-green, that make a 45 MP DSLR photo look sharp on an
unproven-memory browser without weakening the memory contract:

1. **Interaction proxy 768 → 1024.** 1024 is SlimSAM's own internal encode
   edge, so the model now sees its exact native frame (below it the encoder
   upscales for no memory saving). cv-refine cap 768 → 1024, wasm heap
   16 → 32 MiB (rebuilt). The memory-pressure floor still drops the proxy to
   768. No device size is hard-coded anywhere — budgets are framed by trust,
   not by a RAM number, and the device chip no longer prints one.

2. **Display decoupled from the model buffer.** The on-screen photo (`#photo`)
   is a separate crisp GPU-resident frame layered over the hidden ≤1024 model
   buffer (`#view`, which still drives stage layout); the mask overlay stays on
   the ≤1024 buffer. Display is sized bounded-safe per host: Chromium
   scale-decodes the full frame cheaply; Safari uses the largest safe embedded/
   working blob. `?display=native` opts into one full-res decode for a true
   ~4 K preview (the user's chosen tradeoff over the RAW tab-kill risk);
   `?display=off`/`<px>` tune it. Display-only, so honored even when locked.

Gates: policy `?display` parsing; live `#photo` out-resolves `#view` while
`#overlay` tracks `#view`; all lite proxy assertions updated 768 → 1024.
Manually confirmed on real Google Chrome with the 45 MP NEF (2048 default /
4096 native, mask aligned).

## 2026-07-15 — client-side memory-safe architecture rewrite (SHIPPED)

The upload→selection path is now hard-bounded for any machine whose memory
headroom cannot be verified — which is every plain browser tab. Default
architecture: compressed Blob → dedicated bounded decode
worker → ≤768 px proxy → SlimSAM only → single segmentation worker → optional
C++/wasm SIMD refinement → 1-channel Uint8 mask → UI render. All heavy
operations serialize through one shared queue (`js/heavy-job-queue.js`).

Invariant: uploading and selecting from a DSLR image never causes concurrent
full-size decoding, heavyweight model init, model upgrades, detector loading,
full-resolution crop escalation, or unbounded wasm allocation.

Phased delivery (all landed together, each gated in `verify.mjs`):

| Phase | Content | Status |
|---|---|---|
| M0 | Locked lite policy: browser-reported/unknown memory ⇒ lite; unsafe URL flags refused; SlimSAM-only segmentation | ✅ |
| M1 | Shared heavy-job queue (concurrency 1, priorities, revision cancel) + serialized decode/model lifecycle; lazy model warm (post-proxy) | ✅ |
| M2 | Dedicated bounded decode worker (header parse + `ImageDecoder`/`createImageBitmap` resize + working copy), transferables | ✅ |
| M3 | One-embedding SlimSAM lifecycle: single resident entry with disposal, `releaseDocument` on import, no OPFS persistence and no pre-selection encode in lite | ✅ |
| M4 | C++/wasm seeded cleanup, bounded hole fill, min-area removal, open/close (SIMD, fixed 16 MiB heap, 768 px cap, own worker) | ✅ |
| M5 | Benchmarks (structured queue/decode logs) + memory-contract verify suites (policy, sizing, queue, embedding, wasm, static scans) | ✅ |
| M6 | Optional edge-band refinement in wasm — only after profiling shows the JS guided filter is hot | ⏸ deferred |
| M7 | Optional ROI-only GrabCut — only if measurements show need | ⏸ deferred |

Notable policy pivots vs the earlier roadmap (superseding the 2026-07-12/15
amendments below): browser-reported memory now resolves **lite** (was
standard); SAM3/flagship has been retired from the interactive editor, so no
host hint or URL can cause a second segmentation-model download, compile,
cache, or upgrade; OPFS embedding persistence and encode-at-import are
trusted-host features (lite forbids both); lite export is capped 4096 px/8 MP;
the working copy is budget-sized (lite 1280 px). Higher trusted profiles may
raise bounded proxy/export limits but still use SlimSAM only.

Worker note: workers isolate lifecycle and keep the UI thread responsive —
they do NOT multiply the machine's memory budget. WebAssembly here is
browser-native sandboxed code with a fixed 16 MiB linear memory, not an
arbitrary native executable.

---

## Archived pre-rewrite roadmap (superseded)

The remaining record preserves earlier project decisions and benchmark history.
It is not the current architecture: any references below to SAM3/flagship,
multiple segmentation lanes, automatic upgrades, or unrestricted native export
were superseded by the 2026-07-15 8 GB-safe rewrite above.

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
`onnx-community/owlv2-base-patch16-ensemble-ONNX` (the `-ONNX` suffix is part of
the id; the Xenova export's class_head won't place on ORT-web v4). `text-core.js`
(pure: phrase normalization with the "a photo of a X" template, all/every →
multi-intent, NMS, mapping), `detect-engine.js` (pipeline-first adapter, q8 WASM
— fp16 WebGPU only where `detectorWebGPU` allows it, since OWLv2's fixed 960²
wedges 8 GB unified memory), `text-ui.js` (debounced input → numbered
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

> **Amended 2026-07-12:** flagship is now **opt-in** (`?flagship=1`), off in every
> preset. `navigator.deviceMemory` caps at 8, so an 8 GB Air and a 64 GB Studio
> both read 8 — the probe can't confirm the headroom a co-resident 300 MB SAM3
> needs, and auto-loading it thrashed the 8 GB target into swap (120 s decode
> stalls). Profile still probes `pro` (sizing tier); the draft lane keeps full
> capability. Also added `cropMaxSide` (1536/2048/4096) to bound native crop memory.
>
> **Amended 2026-07-12 (residency / 8K):** custody is now **blob-only** — a file
> upload is never held as full-res RGBA. `importOriginal` reads dims from the
> header (`image-io.js`) and decodes **straight to the ≤proxyMax proxy** via
> `createImageBitmap` resize (JPEG scaled-DCT decode ⇒ an 8K/48 MP frame never
> materializes); the compressed Blob is kept and escalation/export/detector
> re-decode only the bounded region they need (`cropMaxSide` / `exportMaxSide` /
> `DETECTOR_INPUT`). Rotated/exotic uploads fall back to the held decode (rare).
> HD composite dropped to one full-frame buffer. Retires `bitmapMaxMP`. This
> supersedes the M4 line "original Blob/Bitmap/Bitmap" — it is Blob everywhere now.
>
> **Amended 2026-07-12 (multi-select tab-kill):** a 2nd-object selection on a
> DSLR photo froze ~35 s then Chrome killed the tab (screen recording). Fixes:
> worker death now RESTARTS a fresh worker (≤2, memory reclaimed) instead of
> falling inline — inline is boot-construction-failure only; escalation fires
> once per SETTLED selection (skip while input pending + 250 ms settle);
> overlay mask layers cached per mask object + rAF-coalesced painting (was 3
> canvases + readbacks per pointermove); hdRefine passes its crop pixels/gray
> into the encode (no duplicate crop-sized buffers); pro `cropMaxSide` 4096→2048
> (2× model input; tensor peak quadruples per doubling); Chrome heap watchdog
> sheds escalation → residents before the OS sheds the tab.
>
> **Amended 2026-07-12 (RAW + adaptive proxy):** camera RAW (.nef/.cr2/.arw/
> .dng/…) is now supported by **lifting the embedded full-res JPEG preview**
> (`image-raw.js`): a DSLR RAW is a container of sensor mosaic + metadata + one
> or more developed JPEG previews — for a cutout tool the preview IS the photo,
> so TIFF-IFD pointers (or a scan fallback for CR3/RAF) locate the largest one
> and we slice out just its bytes (a 33 MB NEF → a 3 MB JPEG, reading ~3 MB of
> the file, zero-cloud, no demosaic). Orientation baked in for portrait RAW; the
> JPEG then rides the existing decode-to-proxy path. On-device demosaic (WASM
> LibRaw) SHIPPED 2026-07-16 as the preview-less fallback — see the top entry.
> Proxy is now **capability-adaptive**: probe →
> `proxyMax` (wasm 768 / GPU 1024 / strong-GPU+f16 1280), `?proxy=<px>|off`
> override; honest cap — the encoder ingests 1024² so bigger only buys click/
> overlay precision. No server side exists to prune (static app; dev-only Playwright).
>
> **Amended 2026-07-12 (Safari RAW tab-kill — CapCut proxy):** Safari reloaded
> the page on RAW selection — its `createImageBitmap` doesn't scaled-decode a
> JPEG, so "decode to proxy" still materialized the full 45 MP (~182 MB), twice
> (import + escalation). Fix, matching the CapCut proxy-edit model: interact on a
> lightweight proxy, relink the mask to full-res at export. `extractRawPreview`
> now returns the full-res preview + a small `proxyBlob` (the RAW's own smaller
> embedded preview); `importOriginal({proxyBlob})` builds the proxy from it (a
> ~7 MB decode, 26× less), full-res kept only for export. `escalateMaxMP`
> (12/24/30) skips interactive native-crop escalation on huge originals — the
> Chrome-only `performance.memory` watchdog can't help Safari, so the fix is
> structural (never decode the big frame during interaction).
>
> **Amended 2026-07-15 (working copy — Safari big-JPEG / detector tab-kill):**
> Safari never shipped `ImageDecoder` (MDN BCD: `version_added:false` — the
> bounded WebCodecs path was dead code there), so a big plain JPEG (no RAW
> preview to dodge into) full-raster-decoded at import AND on every
> original-res re-decode: the detector frame and each escalation crop cost a
> ~180–240 MB transient — the reload banner. Fix: on unbounded-decode hosts
> the ONE unavoidable import decode also re-encodes a ≤4096 `workingBlob`
> (RAW retains its small embedded preview instead — zero extra decodes);
> detector frames + escalation crops decode from it (~45 MB transients),
> escalation re-enables for big files/RAW (gates evaluate the working source),
> and export alone bypasses it (`native:true`) — native-res export truth
> unchanged (a bounded escalation patch is interaction-only; export re-decodes
> natively). Crop prompts now scale by the real bitmap/rect ratio (latent bug:
> a cropMaxSide-downscaled export crop had misplaced prompts, silently saved
> by the IoU gate); working vs native crops key separately (`:w` suffix).
> `?working=1|0` override; gate: Phase A4b (5000 px blob upload →
> `workingActive`, escalation `decoded=true`, export dims exact).

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

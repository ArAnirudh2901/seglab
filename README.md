# SEGLAB — on-device segmentation, safe on 8 GB machines

Select anything in a photo with clicks (±), a box, or a rough lasso and get a
precise, clean-edged cutout — **fully client-side**. No server, no upload
endpoint, no cloud inference: images, prompts, masks, and model weights never
leave the device. That is the product's core privacy guarantee and it is
structural, not a setting.

## Quick start

```bash
node scripts/download-models.mjs   # one-time: vendors ~87 MB of weights/wasm locally
python3 -m http.server 8788        # any static server works
# open http://localhost:8788
```

Without the download step the app still works online (pinned CDN + Hugging
Face fallback). With it, everything serves from `lib/` + `models/` and the
app is fully offline after first load (a service worker caches static assets
on fetch — nothing is ever prefetched).

Verify end-to-end (static contracts + real headless-Chromium E2E):

```bash
bun verify.mjs    # or: node verify.mjs
```

## Controls

- **Click** — left/tap = include, right/Alt-click = exclude (touch: the ＋/− toggle)
- **Box** — drag around the object
- **Lasso** — draw a rough loop; it snaps to the object and never bleeds outside the loop
- `Z` undo · `R` reset · `E` raw/refined toggle · **Cutout PNG** exports with transparency

## The 8 GB Safe Mode (lite policy)

SEGLAB assumes a memory-constrained machine unless proven otherwise. Browsers
report at most 8 GB via `navigator.deviceMemory` (Safari reports nothing), so
in practice **every device runs the `lite` profile** (`js/policy.js`):

| Limit | Value |
|---|---|
| Interaction proxy | **≤ 768 px** long edge |
| Export | ≤ 4096 px long edge AND ≤ 8 MP |
| Model | SlimSAM only, one lane |
| Embeddings resident | exactly 1 (disposed on every image change) |
| Auto-escalation / HD re-decode / speculative work | off |

Safety precedence is strict: `device limit > memory pressure > URL flag >
feature request`. Flags like `?flagship=1`, `?proxy=max`, or `?profile=ultra`
are inert on ≤ 8 GB/unknown devices — verified by tests. Only downgrades
(`?flagship=0`, `?wasm=1`, `?proxy=512`, `?debug=1`) are always honored.

## How a DSLR photo is handled

1. The original file is kept **only as its compressed Blob** — full-resolution
   pixels are never retained by the app (no full-size ImageData, canvas,
   dataURL, or lingering ImageBitmap).
2. A dedicated decode worker parses dimensions + EXIF orientation from a
   bounded header window (JPEG, PNG, WebP, AVIF/HEIF, TIFF, GIF, BMP, and
   RAW containers via their embedded JPEG preview), then asks the browser to
   **resize during decode** (`createImageBitmap` with resize;
   `ImageDecoder` scaled decode as a JPEG fast path) into a ≤ 768 px proxy.
3. Every heavy operation — proxy decode, model warm-up, image encode, prompt
   decode, wasm refinement, export — runs through **one shared queue with
   concurrency 1** (`js/heavy-job-queue.js`). Decode and model work can never
   overlap; peaks never stack. A new import cancels the previous image's jobs
   and every worker reply is revision-checked so stale results never commit.
4. Interaction happens on the proxy. On the first click SlimSAM encodes the
   proxy **once** into a single embedding slot (tensors disposed on
   replacement); further clicks are decoder-only (tens of ms). Nothing is
   persisted to OPFS/IndexedDB.
5. Mask cleanup runs in a compact custom **C++/WebAssembly** module
   (`cpp/cv_refine.cpp` → `public/wasm/cv-refine.wasm`, ~13 KB): seeded
   component cleanup, hole filling, binary open/close — fixed 16 MB heap, no
   threads, no SharedArrayBuffer, inputs capped at 768 px. A JS fallback
   keeps selections working if wasm is unavailable; edge softness comes from
   a guided filter against the grayscale proxy.
6. Masks travel between workers as **one-channel Uint8 alpha buffers**
   (transferred, never copied); RGBA exists only at the final render/export
   edge.

Honest notes: web workers improve responsiveness and lifecycle isolation but
do **not** add system memory budget — the queue's serialization is the actual
memory guarantee. For non-JPEG formats the browser may still perform one
transient full decode internally; it is never retained by JS and nothing else
heavy runs while it happens. WebAssembly here is sandboxed browser-native
compiled code, not an arbitrary native executable.

## Models

- **SlimSAM-77 uniform** (Apache-2.0) — the only automatic model. fp32 on
  WebGPU, q8 on WASM fallback (deterministic file names, both vendored).
- **SAM3 tracker** (~300 MB, WebGPU only) — **dormant**. It never loads,
  downloads, or warms automatically. The only path in is the small "SAM3…"
  button (footer) → explicit confirmation dialog, for users with plenty of
  RAM/storage. Memory pressure blocks the opt-in; failures demote back to
  SlimSAM. On an 8 GB machine: leave it off.
- Text mode is disabled: *"Text selection is unavailable on this device's
  safe memory profile."* Any future detector is policy-gated to WASM/q8,
  load-on-query, dispose-immediately, never concurrent with other heavy work.

## Export

Export is explicit (Cutout PNG button). The original Blob is re-decoded
bounded to **4096 px / 8 MP** inside the export job, the proxy mask is
upscaled into that frame, and the result downloads as a transparent PNG. If
the source exceeds the caps the export is reduced and the status bar says so.
There is no "best-effort" full-resolution path.

## Memory pressure

`performance.memory` (Chrome-only, JS-heap-only) is treated as a hint;
allocation/inference failures escalate a monotonic pressure latch:
L1 stops speculative work · L2 disables wasm refinement and disposes its
worker · L3 clears the embedding after each interaction, caps exports at
4 MP, and shows *"Memory pressure detected — running in safe mode."*
Selection on SlimSAM keeps working at every level. `?debug=1` enables
structured `[seglab][memory]` telemetry and the `__seglab.releaseMemory()`
debug action.

## Repository

```
js/app.js               UI, prompts, overlay, document custody, policy boot
js/policy.js            lite profile + URL-flag clamping + pressure reductions
js/capability.js        device detection + pressure latch
js/heavy-job-queue.js   THE single lane for memory-heavy work
js/image-io.js          sizing formula, bounded header parsers, bounded decode
js/decode-worker/client dedicated bounded decode + export compositing
js/sam-engine/worker/client   SlimSAM inference, one embedding slot, revisions
js/cv-refine-worker/client    wasm hygiene + guided-filter edge refinement
js/sam-core.js          prompt/mask math (one-channel alpha contract)
js/export-hd.js         explicit bounded export
cpp/ + scripts/build-cv-wasm.sh   C++ source + build (artifacts committed)
scripts/download-models.mjs       vendors weights/ort/transformers locally
sw.js                   cache-on-fetch service worker (never prefetches)
verify.mjs              static contracts + headless-Chromium E2E gate
```

Browser support: Chromium (WebGPU) is the primary target; Safari runs the
WASM/q8 lane (the non-asyncify ORT build is vendored for it); anything
without workers falls back inline, still bounded. Rebuilding the wasm module
needs Emscripten (`brew install emscripten`, then
`scripts/build-cv-wasm.sh`) — end users never need it, the artifacts are
committed.

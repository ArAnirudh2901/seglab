# SEGLAB — on-device segmentation

Import a photo, then select anything — clicks (+/−), a box, or a rough lasso
that snaps to the object. Everything runs in the browser (SlimSAM plus a
resource-gated Grounding DINO / OWLv2 detector via transformers.js, WebGPU with
WASM fallback) — no server, no upload, zero cloud.

**Privacy:** images, prompts, masks and all inference stay on this device.
There is no upload endpoint and the offline verify phase proves selection and
export work with the network physically cut.

## Run

```bash
cd ~/seglab
bun run dev          # any static server works; this is python3 -m http.server
# open http://127.0.0.1:8788
```

Nothing downloads at startup. The first image import warms SlimSAM (~14 MB,
one-time, browser-cached). The first **Text** search uses Grounding DINO Tiny
(~151 MB q4f16) on a confirmed accelerated WebGPU device; otherwise it uses
the OWLv2 WASM fallback (~163 MB q8). Both run on-device and are never
uploaded.

## 8 GB Safe Mode (the default)

Browsers privacy-round `navigator.deviceMemory` and cap it at 8 GB, so a
plain browser tab can never prove more than 8 GB. SEGLAB therefore treats
every browser-reported or unknown memory budget as an 8 GB machine and locks
the **lite** profile:

| Setting | lite value |
| --- | --- |
| Interaction proxy | **≤ 768 px long edge** — every import path (JPEG/PNG/WebP/AVIF/HEIF/TIFF/GIF/BMP/RAW preview/paste/drop) |
| Model | **SlimSAM only** — the SAM3 flagship lane never loads, downloads, or upgrades |
| Embeddings | exactly **one** resident, current image only, no OPFS/IndexedDB persistence, no speculative encode before the first selection |
| Detector | Grounding DINO q4f16 only when accelerated WebGPU is confirmed; otherwise OWLv2 WASM/q8. Loads on a real Text query and is disposed immediately after boxes |
| Escalation | no automatic native-crop escalation, no HD crop re-decode |
| Export | explicit user action, capped at **4096 px / 8 MP** (a larger source exports reduced, with a status note) |
| Working copy | ≤ 1280 px bounded re-decode source on unbounded-decode hosts |

Safety precedence is: **hard device limit > memory pressure > URL parameter >
feature request**. `?flagship=1` has no effect on any device: the retired
SAM3 path cannot be re-enabled. On a locked budget, `?profile=ultra`,
`?proxy=max`, `?proxy=off` and `?working=1` are also refused; parameters may
only lower limits (`?proxy=512`, `?escalate=0`, `?force=wasm`).

## DSLR upload handling

The original photo is retained as its **compressed Blob only** — never as a
full-resolution `ImageData`, canvas, RGBA buffer or data URL. A dedicated
decode worker parses the header (dimensions + EXIF from the first 512 KB) and
decodes **straight to the bounded proxy**: `ImageDecoder` with desired output
dimensions where the browser supports it, `createImageBitmap` resize
otherwise. On hosts that can only full-raster-decode (Safari has no
`ImageDecoder`), that one unavoidable decode also re-encodes a small bounded
working copy, then the full raster is closed immediately. Crops and exports
re-decode only the bounded region they need.

## One heavy job at a time

`js/heavy-job-queue.js` owns every memory-heavy operation — proxy decode,
model warm, image encode, prompt decode, detector runs, wasm refinement,
export re-decodes — at concurrency 1, so their peak allocations can never
stack. Import decode outranks model work; user interaction outranks
speculative prewarm; a new upload or click invalidates queued stale jobs; a
rejected job releases ownership in `finally`.

Model loading is **lazy**: nothing warms until the proxy is on screen, and
the first selection encodes exactly one embedding for the current document.

## C++/WebAssembly mask refinement

`cpp/cv_refine.cpp` compiles to a compact SIMD wasm module (~12 KB, fixed
16 MiB heap, no growth, no threads/SharedArrayBuffer) that runs seeded
connected-component cleanup, bounded hole fill, min-area small-object removal
and binary open/close on the **one-channel ≤768 px mask** inside its own
worker. It lazy-loads on the first refinement, transfers buffers (never
copies), rejects dimensions above 768 px, and on any failure the model's own
mask is kept — no retries, no fallback CV library. WebAssembly is
browser-native sandboxed code, not an arbitrary executable; and a worker
isolates lifecycle/responsiveness — it does **not** add system memory budget.

Build it from a clean checkout with Emscripten on PATH:

```bash
./scripts/build-cv-wasm.sh    # emits public/wasm/cv-refine.{js,wasm}
```

Browsers without wasm SIMD (or with the module unavailable) keep the existing
JavaScript post-pipeline — feature loss is quality-polish only.

## Trusted-host budgets (Phosmith)

Only a trusted native host can raise the profile past lite, by injecting a
usable editor budget before `app.js` loads:

```js
window.__PHOSMITH_DEVICE_RESOURCES__ = {
  memoryBudgetGB: 24,       // editor headroom: 4 / 8 / 16 / 24 …
  vramGB: 24,               // optional
  gpuName: 'RTX 4090',      // optional label
  mode: 'balanced',         // 'conservative' | 'balanced' | 'performance'
}
```

12–23 GB earns **pro** (1280 px proxy, native HD export, auto escalation,
OPFS embedding persistence), 24 GB+ earns **ultra**. These budgets can expand
bounded proxy/export settings only; segmentation remains SlimSAM-only and
`?flagship=1` is ignored. Live updates via the
`phosmithresourceschange` event only shrink or grow **future** allocations;
memory pressure is a one-way ratchet.

## About “256 × 256” masks

`256 × 256` is SlimSAM's internal decoder-mask resolution, not the image
resolution. The model sees the ≤768 px interaction frame; its mask is
restored to that frame and snapped to real image edges before display/export.
A larger proxy cannot invent model detail because SlimSAM encodes at a
1024 px long edge internally.

## Controls

- **Click** — left/tap = include, right/Alt-click = exclude (touch: the ＋/− toggle)
- **Box** — drag around the object
- **Lasso** — draw a rough loop; it snaps to the object and can never bleed outside the loop
- **Region** — draw an exact freehand mask; it selects the drawn area, not the object inside it
- **Rect / Ellipse / Polygon** — direct marquee masks; polygon closes with double-click or `Enter`
- **Magic / Color** — select a contiguous colour region or all matching colours. Tolerance is the allowed RGB colour distance: start at **36**; lower it (16–28) for a crisp edge, raise it (45–60) only to include shadows/highlights. It never affects Text mode.
- **Brush** — paint a mask; right-click or Alt paints an erase stroke
- **Text** — describe an object; the first search downloads a ~151 MB accelerated detector or ~163 MB portable fallback.
  Grounding DINO/OWLv2 return candidates, so results are not a
  guarantee that every instance in a crowded scene is found. Colour-qualified
  prompts also rank boxes using local colour evidence. If it cannot load within
  the safe memory profile, text
  selection reports itself unavailable instead of retrying heavier backends
- `Z` undo · `R` reset · **Cutout PNG** downloads the selection with transparency

## Verify (headless)

```bash
bun verify.mjs   # policy/sizing/queue/embedding/wasm/static suites + real app in headless Chromium
```

## Browser compatibility

- **Chromium**: `ImageDecoder` bounded decode, WebGPU where exposed, wasm SIMD.
- **Safari 16.4+**: no `ImageDecoder` → `createImageBitmap` + one-time bounded
  working copy; wasm SIMD supported; WebGPU per OS version, WASM fallback.
- **No SIMD / old browsers**: wasm refinement silently off; JS pipeline serves.
- No COOP/COEP, `SharedArrayBuffer` or threads are required.

## Architecture

- `js/heavy-job-queue.js` — ONE memory-heavy job at a time (priorities, revisions, cancellation)
- `js/proxy-plan.js` — the single authoritative ≤768 px proxy-sizing function
- `js/decode-worker.js` / `decode-client.js` / `decode-core.js` — bounded image decode off the UI thread
- `js/asset-store.js` — blob-only custody of the original; bounded crop/export re-decodes
- `js/sam-engine.js` / `sam-worker.js` / `sam-client.js` — SlimSAM inference; encode-once/decode-per-click; one-embedding lite lifecycle
- `js/cv-refine-worker.js` / `cv-refine-client.js` + `cpp/cv_refine.cpp` — wasm mask cleanup
- `js/detect-engine.js` / `detect-worker.js` — disposable, resource-gated text detector
- `js/policy.js` / `capability.js` — locked lite budgets, trusted-host tiers, pressure ladder
- `js/app.js` — UI, prompts, overlay, export orchestration

A future experimental scripting/plugin panel is documentation-only; no Python
runtime (Pyodide/PyScript/etc.) and no full OpenCV.js ship in this app.

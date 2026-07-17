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
bun run models       # one-time: vendors ~90 MB (transformers.js, ORT wasm, SlimSAM)
bun run dev          # any static server works; this is python3 -m http.server
# open http://127.0.0.1:8788
```

`bun run models` is what makes the app **work with no internet**. It vendors
the transformers.js bundle, the ORT wasm pair and the SlimSAM weights under
`lib/` and `models/` (both gitignored); the app prefers those and only falls
back to the pinned CDNs for anything unvendored. After it has run, import,
click/box/lasso selection and export need no network at all — a fresh,
cache-less browser profile with huggingface.co and jsdelivr blocked still
works, and `bun verify.mjs` gates exactly that.

Text-select (OWLv2, 163 MB) is **not** vendored by default because the feature
is optional and disposable at runtime. To make it offline too:

```bash
bun run models:all   # core + OWLv2 detector (~253 MB total)
```

Skipping `bun run models` is still fine — the app then loads from the pinned
CDNs on first use and caches into Cache Storage, which survives reloads but is
evictable and empty in a fresh profile. Vendoring is the durable answer.

Nothing downloads at startup. Unvendored, the first image import warms SlimSAM
(~14 MB, one-time, browser-cached). The first **Text** search uses Grounding DINO Tiny
(~151 MB q4f16) on a confirmed accelerated WebGPU device; otherwise it uses
the OWLv2 WASM fallback (~163 MB q8). Both run on-device and are never
uploaded.

## Memory-bounded by default

A browser tab cannot verify its real memory headroom —
`navigator.deviceMemory` is privacy-rounded and capped, so a report proves
nothing about actual free RAM. SEGLAB therefore never sizes budgets from a
browser memory report: every unverified budget gets the **lite** profile.
Lite is not a degraded mode; it is the fully optimised baseline every feature
is engineered to fit, so a DSLR photo works smoothly on any laptop:

| Setting | lite value |
| --- | --- |
| Interaction proxy | **≤ 1024 px long edge** — every import path (JPEG/PNG/WebP/AVIF/HEIF/TIFF/GIF/BMP/RAW preview or on-device develop/paste/drop). 1024 is SlimSAM's own internal encode edge, so the model sees its exact native frame |
| Display preview | **decoupled from the model** — a crisp GPU-resident display frame (≤ 2048 px, bounded-safe) so a DSLR photo looks sharp while the model keeps its bounded ≤ 1024 buffer (see *Crisp preview* below) |
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

## Crisp preview (display decoupled from the model)

The on-screen photo and the pixels the model reads are **two different
frames**. SlimSAM (and magic-wand / colour / brush) work on the bounded
≤ 1024 px interaction buffer — that is the memory contract. What you *see* is a
separate, higher-resolution **display frame** painted over it: a GPU-resident
bitmap (a texture, never a full CPU RGBA buffer or a model tensor), so a 45 MP
DSLR photo looks sharp without ever materialising 45 MP of RAM. The mask
overlay is drawn on the ≤ 1024 buffer and scaled onto the display; export is
unaffected (it always re-decodes natively).

The display frame is sized **bounded-safe per host**:

- **Bounded-decode hosts** (Chromium's `ImageDecoder`) scale-decode the full
  frame down to the display size cheaply → a near-native preview.
- **Unbounded-decode hosts** (Safari has no `ImageDecoder`) use the largest
  source that is safe to fully decode — a RAW's own smaller embedded preview or
  the bounded working copy — never the full 45 MP frame (the tab-kill path).

`?display=native` opts into one full-resolution decode for a true ~4 K preview
even on Safari (accepting a large one-time transient); `?display=off` pins the
preview to the model proxy; `?display=<px>` caps the display long edge lower.
Because this is display-only, it never touches the segmentation memory contract
and is honoured even on a locked budget.

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

## Camera RAW (preview first, then on-device develop)

A camera RAW (`.nef/.cr2/.cr3/.arw/.dng/.raf/…`) is a container, not a picture.
The **fast path** (`js/image-raw.js`) never demosaics: it parses the container's
IFD pointers, lifts the camera's own **embedded JPEG preview** by byte range
(a 33 MB NEF → a ~3 MB JPEG), and hands that to the normal decode-to-proxy path.
For a select/cutout tool the preview *is* the photo, and this covers essentially
every modern RAW with no sensor decode at all.

The rare RAW that carries **no usable embedded preview** falls back to an
on-device develop (`js/raw-develop-client.js` → `public/wasm/raw-develop.wasm`):
[LibRaw](https://github.com/libraw/libraw) 0.22.2, compiled to WebAssembly with
Emscripten, demosaics the sensor and libjpeg re-encodes the result **inside the
worker**, so only a compact JPEG Blob crosses back — a full-res RGBA frame never
leaves that heap. It is memory-safe by construction and by policy:

- **Half-resolution demosaic** (`half_size`) — a ≤1024 px masking proxy never
  needs more, and it cuts the develop footprint ~4×.
- **Megapixel cap before unpack** — sensors above the profile's `rawDevelopMaxMP`
  (lite 40 · standard 60 · pro 80 · ultra 100) are refused before LibRaw
  allocates the full sensor buffer, so the peak is bounded and predictable.
- **Lazy + disposed** — the ~2 MB module is never fetched until a preview-less
  RAW is actually opened, and the worker is **terminated after every develop**
  because a full-sensor demosaic grows the wasm heap and wasm memory never
  shrinks; terminating returns it to the OS.
- **Gated** — off at memory pressure ≥ 2; on failure the import surfaces the
  existing "export a JPEG/TIFF" error. Still no server, no Python, zero cloud.

RawSpeed (decoder only, no demosaic; speed-first, OOM-prone on huge files) and
rawpy/imagecodecs (Python) were evaluated and rejected — LibRaw is the only
option that fits a zero-cloud browser and is itself hardened against image-bomb
allocations.

Build it from a clean checkout with Emscripten on PATH:

```bash
./scripts/build-libraw-wasm.sh   # fetches pinned LibRaw, emits public/wasm/raw-develop.{js,wasm}
```

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
32 MiB heap, no growth, no threads/SharedArrayBuffer) that runs seeded
connected-component cleanup, bounded hole fill, min-area small-object removal
and binary open/close on the **one-channel ≤1024 px mask** inside its own
worker. It lazy-loads on the first refinement, transfers buffers (never
copies), rejects dimensions above 1024 px, and on any failure the model's own
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
resolution. The model sees the ≤1024 px interaction frame; its mask is
restored to that frame and snapped to real image edges before display/export.
A proxy larger than 1024 px cannot invent model detail because SlimSAM encodes
at a 1024 px long edge internally — which is exactly why the interaction proxy
is capped there and the crisp on-screen preview is a separate display frame.

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
- `js/proxy-plan.js` — the single authoritative ≤1024 px proxy-sizing function
- `js/decode-worker.js` / `decode-client.js` / `decode-core.js` — bounded image decode off the UI thread
- `js/asset-store.js` — blob-only custody of the original; bounded crop/export re-decodes
- `js/sam-engine.js` / `sam-worker.js` / `sam-client.js` — SlimSAM inference; encode-once/decode-per-click; one-embedding lite lifecycle
- `js/cv-refine-worker.js` / `cv-refine-client.js` + `cpp/cv_refine.cpp` — wasm mask cleanup
- `js/image-raw.js` — RAW embedded-preview extractor (fast path, no demosaic)
- `js/raw-develop-worker.js` / `raw-develop-client.js` + `cpp/raw_develop.cpp` — LibRaw wasm develop (preview-less fallback, disposed after use)
- `js/detect-engine.js` / `detect-worker.js` — disposable, resource-gated text detector
- `js/policy.js` / `capability.js` — locked lite budgets, trusted-host tiers, pressure ladder
- `js/app.js` — UI, prompts, overlay, export orchestration

A future experimental scripting/plugin panel is documentation-only; no Python
runtime (Pyodide/PyScript/etc.) and no full OpenCV.js ship in this app.

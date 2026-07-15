# SEGLAB — on-device segmentation

Import a photo, then select anything — clicks (+/−), a box, or a rough lasso
that snaps to the object. Everything runs in the browser (SlimSAM and OWLv2 via
transformers.js, WebGPU with WASM fallback) — no server, no upload, zero cloud.

## Run

```bash
cd ~/seglab
bun run dev          # any static server works; this is python3 -m http.server
# open http://127.0.0.1:8788
```

First selection downloads ~14 MB of model files once (browser-cached after).
The first **Text** search downloads the OWLv2 detector (~163 MB, q8) once; it
runs on-device and is never uploaded.

> On 8 GB machines the detector deliberately runs on WASM, not WebGPU. OWLv2
> infers at a fixed 960² whatever you feed it, and the fp16 WebGPU session pins
> enough unified memory to wedge the whole host — see `detectorWebGPU` in
> `js/policy.js`. Searches are slower there; the machine stays usable.

## Device-adaptive images and GPU use

SEGLAB separates memory budgeting from GPU acceleration. It asks WebGPU for a
high-performance adapter, so a browser that exposes an RTX-class GPU or Apple
GPU runs inference there; it falls back to WASM when WebGPU is unavailable.
Small images use a native interaction frame (no proxy). Large images retain
only compressed source bytes and use bounded proxy, crop, and export canvases,
so a full DSLR frame is never held resident as decoded RGBA just for editing.

The standalone web build uses these conservative budgets:

| Usable memory evidence | Profile | Proxy | Bounded export |
| --- | --- | --- | --- |
| Confirmed under 8 GB (including 4 GB) | Lite | 768 px | 8 MP |
| Browser-reported 8 GB | Standard | 1024 px | 24 MP |
| Memory unavailable to the browser | Standard-safe | 1024 px | 16 MP |
| Trusted Phosmith 12–23 GB budget | Pro | 1280 px | 36 MP |
| Trusted Phosmith 24 GB+ budget | Ultra | 1536 px | 64 MP |

Browsers intentionally round `navigator.deviceMemory` and cap it at 8 GB, and
do not provide dependable VRAM. Therefore an ordinary tab cannot safely tell a
16 GB M1 from a 24 GB M-series device, or confirm an RTX 4090's VRAM. It still
uses WebGPU when available, but only a trusted host is allowed to increase
memory allocations beyond the Standard profile.

### About “256 × 256” masks

The normal browser profile already uses a **1024 px interaction frame**; Lite
uses 768 px, while a trusted larger-memory host can use 1280 or 1536 px. The
`256 × 256` figure is SlimSAM's internal decoder-mask resolution, not the
uploaded-photo or encoder resolution. The model sees the interaction frame,
then its mask is restored to that frame and snapped to real image edges before
display/export. `?proxy=512` is available for a deliberately lighter preview;
increasing it beyond 1024 improves click and edge-guide precision, but cannot
invent model detail because SlimSAM itself encodes at a 1024 px long edge.

For an upright large JPEG on a browser with WebCodecs, import now asks
`ImageDecoder` to decode directly to that bounded interaction size. The source
remains compressed bytes until a bounded crop/export operation needs it, which
avoids the otherwise common full-DSLR-raster spike during import.

### Phosmith host contract

Before loading `app.js`, a Phosmith WebView can inject a usable editor budget
(not merely installed RAM). This lets the host account for other active apps:

```js
window.__PHOSMITH_DEVICE_RESOURCES__ = {
  memoryBudgetGB: 24,       // editor headroom: 4 / 8 / 16 / 24 …
  vramGB: 24,               // optional; useful for host telemetry
  gpuName: 'RTX 4090',      // optional display/diagnostics label
  mode: 'balanced',         // 'conservative' | 'balanced' | 'performance'
  allowFlagship: true,      // explicit consent to preload the optional SAM3 lane
}
```

To react to an OS memory warning, replace the value and dispatch an event. The
new budget governs subsequent imports, native re-decodes, and exports; it never
enlarges already-allocated canvases in place. A lower budget immediately retires
the optional flagship GPU session and its embeddings.

```js
window.__PHOSMITH_DEVICE_RESOURCES__ = { memoryBudgetGB: 8, mode: 'conservative' }
window.dispatchEvent(new CustomEvent('phosmithresourceschange', {
  detail: window.__PHOSMITH_DEVICE_RESOURCES__,
}))
```

`allowFlagship` is deliberately explicit: hardware detection alone never starts
a ~300 MB model download or reserves that GPU memory. JPEG, PNG, WebP, AVIF/
HEIF, TIFF, GIF, BMP, and common camera RAW containers are accepted; RAW files
use their embedded developed JPEG preview for on-device masking.

`?proxy=off` requests a native frame, but it is deliberately refused for large
files and falls back to the selected device-safe cap instead of risking a tab
crash. A runtime pressure ladder releases the detector, flagship embeddings,
and then the optional flagship session before lowering future canvas limits.

## Controls

- **Click** — left/tap = include, right/Alt-click = exclude (touch: the ＋/− toggle)
- **Box** — drag around the object
- **Lasso** — draw a rough loop; it snaps to the object and can never bleed outside the loop
- **Region** — draw an exact freehand mask; it selects the drawn area, not the object inside it
- **Rect / Ellipse / Polygon** — direct marquee masks; polygon closes with double-click or `Enter`
- **Magic / Color** — select a contiguous colour region or all matching colours; use Tolerance to tune the match
- **Brush** — paint a mask; right-click or Alt paints an erase stroke
- **Text** — describe an object; the first search downloads the detector (~163 MB)
  and says so before it starts
- `Z` undo · `R` reset · **Cutout PNG** downloads the selection with transparency

## Verify (headless)

```bash
bun verify.mjs   # drives the real app in headless Chromium, asserts on known answers
```

## Architecture

- `js/sam-core.js` — pure prompt/mask math (no DOM, no ML deps)
- `js/sam-engine.js` — SlimSAM inference; encode-once/decode-per-click embedding cache; WebGPU→WASM sticky fallback
- `js/sam-worker.js` — dedicated worker so inference never blocks the UI
- `js/sam-client.js` — main-thread API; sticky inline fallback if the worker dies
- `js/app.js` — interface: prompts, lasso→prompt conversion + clamp, overlay, cutout
- `js/detect-engine.js` — on-device OWLv2 text detector; disposable, device ladder from policy

Next lanes (same `segment()` contract): SAM3/EfficientSAM3 flagship tier, text prompts.

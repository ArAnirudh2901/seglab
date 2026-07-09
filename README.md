# SEGLAB — on-device segmentation

Import a photo, then select anything — clicks (+/−), a box, or a rough lasso
that snaps to the object. All inference runs in the browser (SlimSAM via
transformers.js, WebGPU with WASM fallback). Zero cloud: nothing is uploaded.

## Run

```bash
cd ~/seglab
python3 -m http.server 8788
# open http://localhost:8788
```

First selection downloads ~14 MB of model files once (browser-cached after).

## Controls

- **Click** — left/tap = include, right/Alt-click = exclude (touch: the ＋/− toggle)
- **Box** — drag around the object
- **Lasso** — draw a rough loop; it snaps to the object and can never bleed outside the loop
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

Next lanes (same `segment()` contract): SAM3/EfficientSAM3 flagship tier, text prompts.

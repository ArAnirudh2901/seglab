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
- `js/gpu-post.js` — post pipeline as **WebGPU compute shaders (WGSL)** — the default path
- `js/edge-refine.js` — the same guided-filter math as an O(N) CPU reference and fallback
- `js/sam-worker.js` — dedicated worker so inference never blocks the UI
- `js/sam-client.js` — main-thread API; sticky inline fallback if the worker dies
- `js/app.js` — interface: prompts, lasso→prompt conversion + clamp, overlay, cutout

### Why the hot path is WGSL and not C/C++

A browser page cannot execute C or C++ on a GPU. C/C++ through Emscripten
becomes WASM, which is still a CPU target — single-threaded and scalar. The
only language that runs *directly on the GPU* from a web page is a shader, and
for WebGPU that is WGSL. `gpu-post.js` therefore carries the guided-filter
refinement as C-style compute kernels (one thread per pixel), which is exactly
what a C++/Dawn build of this pipeline would hand the driver anyway.

Both backends are verified to produce byte-identical masks. `?gpu=0` forces the
CPU path; the timing chip reports which one served the last decode.

### Post-pipeline cost

The band that edge refinement can change is a few percent of a frame, and mask
components cannot exist outside the mask's own bounding box. Both stages
therefore run on that box plus a margin rather than the whole frame — see
CROP MARGIN in `edge-refine.js` for why the shell is provably sufficient. The
saving scales with how much of the frame the subject occupies, so it is
largest exactly where the old code was most wasteful: small objects.

CPU path at the canonical 1024² frame, by subject size:

| subject | frame covered | refinement | hygiene |
|---|---|---|---|
| huge (60% of width) | 43% | 271 → 88 ms | 57 → 18 ms |
| medium (30%) | 13% | 399 → 29 ms | 41 → 7.5 ms |
| small (10%) | 2.5% | 241 → 8.6 ms | 44 → 6.0 ms |
| minute (r=9px) | 0.6% | 220 → 4.2 ms | 41 → 5.6 ms |

Output is byte-identical before and after, verified across 31 scenes covering
border-touching, multi-component, nested-hole, odd-dimension and full-frame
masks.

Next lanes (same `segment()` contract): SAM3/EfficientSAM3 flagship tier, text prompts.

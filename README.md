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
- **Text** — describe what to select ("all the zebras", "the red bicycle") and get
  masks for every match. Plural or "all …" phrasing selects every instance;
  definite singular ("the red bicycle") selects just the best one. When several
  match, each gets a chip in the footer you can toggle on or off.
  A phrase that isn't in the photo returns **nothing** — that's the correct
  answer, not a failure.
- `Z` undo · `R` reset · `E` raw/refined · **Cutout PNG** downloads the selection with transparency

## Verify (headless)

```bash
bun verify.mjs   # drives the real app in headless Chromium, asserts on known answers
```

Phases A/B/C need a browser (`npx playwright install chromium`); without one they
are reported as **skipped**, not passed. The pure-module gates always run:

```bash
bun test-post-pipeline.mjs   # optimized hygiene + edge refinement vs a naive reference (exact)
bun test-text-core.mjs       # query parsing, NMS, absent-phrase rejection
bun test-governor.mjs        # the 2.2 GB ceiling policy under 45 MP load
```

## Benchmark (real photos, real numbers)

```bash
bun bench.mjs                       # scans ~/Desktop, biggest files first
bun bench.mjs --images ~/Pictures/x --phrases "all the birds,the red door"
bun bench.mjs --synthetic           # DSLR-scale fixtures, no photos needed
```

Reports per-stage latency (detect / decode / post, P50 + P95, cold vs warm),
peak browser RSS and JS heap, and the absent-phrase false-positive rate.
**Peak RSS above 2.2 GB fails the run.** Output lands in `bench-out/`.

Camera raw (`.CR2`/`.NEF`/`.ARW`/`.DNG`) and HEIC can't be decoded by Chromium —
those files are listed as skipped, so export to JPEG to include them.

## Architecture

- `js/sam-core.js` — pure prompt/mask math (no DOM, no ML deps)
- `js/text-core.js` — pure query→prompt rules: phrase parsing, mask NMS, adaptive detection thresholds
- `js/sam-engine.js` — SlimSAM inference; encode-once/decode-per-click embedding cache; WebGPU→WASM sticky fallback
- `js/text-engine.js` — open-vocabulary detection (OWLv2 quantized, 155 MB) for the text lane
- `js/memory-governor.js` — enforces the whole-app RAM ceiling; one heavy encoder resident at a time
- `js/edge-refine.js` — guided-filter edge refinement; per-image terms hoisted, bbox-confined, pooled buffers
- `js/sam-worker.js` — dedicated worker so inference never blocks the UI
- `js/sam-client.js` — main-thread API; sticky inline fallback if the worker dies
- `js/app.js` — interface: prompts, lasso→prompt conversion + clamp, text bar, overlay, cutout

**Text is masks, not boxes.** A phrase produces boxes internally, those boxes
drive the same SAM decoder a click does, and every mask goes through the same
post pipeline — so a described selection is as clean-edged as a clicked one.
Instance masks are stored bbox-cropped, never full-frame (a 6000×4000 RGBA mask
is ~96 MB *each*).

Memory: the text detector and the SAM3 flagship encoder are never resident
together. The governor evicts one to load the other, and skips the flagship
entirely when the budget can't take it — a working draft lane beats an OOM tab.

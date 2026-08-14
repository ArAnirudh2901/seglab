# SEGLAB — Mask Lane Design

Single-configuration architecture for the interactive mask lane: SAM 2.1 from
`darktable-org/darktable-ai`, YOLOE-first routing, and a memory contract that
holds on an 8 GB MacBook Air with a 45 MP RAW.

Status: **implemented and measured.** Written 2026-08-06; revised the same day
against measurements on the target device (M2 / 8 GB, Chrome 150, ORT-Web 1.27).

> **Memory contract met.** fp16 on ORT-Web 1.27 with `graphOptimizationLevel:
> 'all'` holds the whole app at **1782–1850 MB** total Chrome footprint
> (GPU process 1095–1132 MB) with a 45 MP RAW imported, clicked, and a second
> RAW loaded on top — under the 2 GB ceiling, and stable across imports.

> Sections marked **[MEASURED]** were corrected after the conversion pipeline and
> the SAM 2.1 lane were built and profiled. Several original claims turned out to
> be wrong in ways that silently damage mask quality or memory, so read the
> correction before trusting the surrounding prose. Full evidence, with numbers,
> in `spikes/sam21/FINDINGS.md`.
>
> Implemented: `scripts/export-sam21.py`, `js/sam21-lane.js`, `js/sam21-host.js`,
> `js/sam21-client.js`, `js/sam21-adapter.js`, `js/mask-refine.js` (§10),
> the §11 config collapse, and the harnesses under `spikes/sam21/`.
> Not implemented: §5 YOLOE-first routing.

---

## 1. Why

Three problems, one root cause.

**Tiering has outgrown its usefulness.** Five presets (`lite` / `standard8` /
`standard` / `pro` / `ultra`) plus a capability auto-tier ladder plus a manual
override plus four pressure levels. Every new model lane must be wired into all
of them, and the combinations are untestable in practice. ~100 references across
10 files.

**A preset guarantees nothing.** It picks numbers. It cannot promise that peak
memory stays bounded, because nothing enforces the numbers at allocation time.
The measured NEF import+click peak of ~2.1 GB happened *inside* a configuration
that was supposed to prevent it.

**SlimSAM is the quality ceiling.** It is a pruned SAM 1. `darktable-ai` now
publishes SAM 2.1 as ONNX, which is a genuine upgrade and is architecturally
compatible with the encode-once/decode-per-click design already in place.

The fix is to stop describing the system as a *configuration* and start
describing it as a set of *invariants* enforced by admission control.

---

## 2. Goals / non-goals

**Goals**
- One configuration. No tiers, no auto-climb, no manual override.
- No memory spikes, with a 45 MP RAW, on 8 GB, under any interaction order.
- Click → mask paint feels instant.
- Highest mask quality achievable inside the budget.

**Non-goals**
- Supporting devices without WebGPU. See §4.
- Running darktable's denoise/upscale models. Deferred; see §16.
- Matching darktable's *output* pipeline. We use its models, not its rendering.

---

## 3. Invariants

These replace the preset table. They are assertions, testable in `verify.mjs`.

```
I1  per-tab peak resident never exceeds 900 MB
I2  click → first mask paint stays inside the device's click budget
    (`postBudgetMs`, 220 ms; see js/hardware-fit.js)
I3  no two heavy jobs are ever in flight ON THE MACHINE
```

`I3` is what makes `I1` meaningful — it turns peak from `sum()` into `max()`.
`I1` is set at roughly half the ~1.9 GB usable budget so the OS, the browser's
own processes, and the user's other tabs are not starved.

### Why I1 is per-tab and I3 is machine-wide

Each tab is its own JS context with its own ORT sessions, arenas, WebGPU device,
and workers. Three tabs hold roughly three times the memory, and each one
believes it is behaving. `measureUserAgentSpecificMemory()` only reports the
calling agent cluster, so a tab is **structurally blind** to the others — the
governor cannot detect the aggregate problem at all.

Strictly bounding total memory across tabs would require full cross-tab
coordination (shared ledger, leader election, crash recovery). That is more
machinery than this earns. It is also unnecessary, because the two cases differ
enormously:

| Case | Total | Outcome |
|---|---|---|
| 3 tabs at steady state (270 MB) | ~810 MB | survivable |
| 3 tabs encoding simultaneously (650 MB) | ~2 GB | fatal |

**So bound the spikes, not the steady state.** `I3` is enforced machine-wide via
`navigator.locks` (§9), which converts the fatal case into a merely slow one.
`I1` stays per-tab because that is the only thing a tab can actually observe and
control.

**The memory governor stays**, but its role changes. It is no longer the primary
defense (reacting to pressure); it is the backstop for when a projection in §9
turns out wrong.

---

## 4. Device contract

**WebGPU + `shader-f16` are hard requirements.** A device without them is
refused with a clear message.

This is not a compromise, it is the only honest position. WASM SlimSAM held
~3 GB resident against ~0.5 GB on WebGPU (measured). A silent CPU fallback is
not a safety net — it is a second, hidden product that OOMs. Half-precision is
required because the fp16 model variants are what fit the budget (§7), and a
machine lacking `shader-f16` would be unusable on speed regardless.

Consequence: `js/sam-engine.js:644-648` currently demotes to WASM after
`WEBGPU_FAILURE_LIMIT` consecutive failures. That must become a hard error.
`?force=wasm` survives only as a development flag, never a runtime path.

---

## 5. Model strategy — two paths, routed by intent

The encoder is the only expensive component in the entire system. Everything
else is rounding error. So the design question is not "how do we make the
encoder fast" but **"how do we make the encoder rare."**

```
import → YOLOE-26s prompt-free pass        ~200 ms, 44 MB, already resident
       → candidate instance masks on screen
       │
       ├─ click lands inside a YOLOE mask  → INSTANT. no SAM. no encode.
       │
       └─ click misses every YOLOE mask    → NOW load SAM, encode, decode
```

| Path | Model | When | Cost |
|---|---|---|---|
| Fast | YOLOE-26s (already shipped) | every import | ~200 ms, 44 MB resident |
| Precision | SAM 2.1 small | on demand | ~650 MB transient, 1–3 s |

**Why this is the highest-leverage decision in the document:** in a large
fraction of sessions the SAM encoder never loads, so the 650 MB transient never
occurs and the 1–3 s wait never occurs. It also de-risks the project — if SAM
2.1 measures badly on the Air, a working product still ships.

**Speculative warm:** when the cursor dwells >400 ms outside every YOLOE mask,
begin the SAM encode in the background. The fallback path then feels pre-warmed
too. Admission-controlled (§9), so it yields to any real job.

---

## 6. Model facts (measured, not estimated)

Pulled from `darktable-org/darktable-ai` release `release-5.6.0`. `.dtmodel` is
a plain DEFLATE zip (`darktable_ai/package.py`).

| Package | Download | Unpacked | Encoder | Decoder |
|---|---|---|---|---|
| `mask-object-sam21-tiny` | 137 MB | 147.7 MB | 128.0 MB | 19.6 MB |
| **`mask-object-sam21-small`** | 162 MB | 174.8 MB | **155.1 MB** | **19.6 MB** |
| `mask-object-sam21-base-plus` | 294 MB | 343.7 MB | 324.0 MB | 19.6 MB |
| `mask-object-segnext-b2hq` | 403 MB | 442.2 MB | 338.8 MB | 103.4 MB |

**Choice: `small`.** Rationale in §6.1. Rejected: `base-plus` (324 MB encoder
widens `embed_dim`, so unlike `small` it scales the high-resolution early stages
and the arena grows with it) and `segnext-b2hq` (its 103 MB decoder makes every
*interaction* expensive, not just loading).

### 6.1 Why `small` costs almost nothing over `tiny`

Hiera-T and Hiera-S both use `embed_dim: 96` and identical stage-1/stage-2
configuration. The only difference is stage-3 depth — `(1,2,7,2)` vs
`(1,2,11,2)` — and both carry exactly three global-attention blocks.

Peak arena is set by the stages they share:

| Stage | Spatial | Per-tensor fp16 | tiny | small |
|---|---|---|---|---|
| 1 | 256×256×96 | ~12.6 MB | 1 blk | 1 blk |
| 2 | 128×128×192 | ~6.3 MB | 2 blk | 2 blk |
| 3 | 64×64×384 | ~3.1 MB | 7 blk | **11 blk** |
| global attn | 4096²×4 heads | ~134 MB | 3 blk | 3 blk |

The four extra blocks sit at the *lowest* spatial resolution of the deep stages.
`small` costs ~14 MB more weights at fp16 and ~25–35 % more encode latency —
paid once per image and then cached. It barely moves the high-water mark.

### 6.2 The encoder is swappable

The decoder is 19.6 MB for **all three** variants, so the embedding contract is
identical. Keep the encoder path a config value: swapping `small` → `tiny` is a
one-line change with no cache migration and nothing for the persistence layer to
relearn.

### 6.3 Real graph shapes

Read from the actual ONNX, not assumed:

```
encoder  in : image[1,3,1024,1024]                        FLOAT   static ✓
         out: image_embed[1,256,64,64]                    FLOAT   static ✓
              high_res_feats_0[<4× symbolic>]             FLOAT   DYNAMIC ✗
              high_res_feats_1[<4× symbolic>]             FLOAT   DYNAMIC ✗

decoder  in : image_embed[1,256,64,64]
              high_res_feats_0[1,32,256,256]
              high_res_feats_1[1,64,128,128]
              point_coords[1,num_points,2]                        DYNAMIC ✗
              point_labels[1,num_points]                          DYNAMIC ✗
              mask_input[1,1,256,256]
              has_mask_input[1]
         out: masks[1,<Clip dim>,256,256]                         DYNAMIC ✗
              iou_predictions[<dim>,3]
```

**Embedding size:** 4 + 8 + 4 = **16 MB fp32 → 8 MB fp16.**

**Critical finding:** the encoder emits `high_res_feats_0/1` with *symbolic*
dims even though the decoder declares them static. Three planned optimizations
require static shapes, so shape freezing is step one of the conversion, not an
afterthought.

---

## 7. Conversion pipeline (offline, one time)

Target: `scripts/export-sam21.py`. Runs against the toolchain already installed
(onnx 1.22, onnxruntime 1.26, onnxconverter-common 1.16).

### Step 1 — fetch, verify, unpack

`.dtmodel` is a plain DEFLATE zip. Cache under `spikes/sam21/`.

**Pull `versions.json` from the same release** (1.34 KB) — it carries a version
string and a sha256 per model:

```json
"mask-object-sam21-small": {
  "version": "1.0",
  "sha256": "sha256:06a629ac1adf40daec47d2191cbdbdac609d93769584a05f65f4276bf784663f"
}
```

Three uses:

- **Pin the version.** Record `1.0` so a nightly build never silently changes
  what ships. `release-5.6.0` is the pinned release; `nightly-5.7.0` exists and
  carries the same digests today, but that is not a guarantee.
- **Verify before unpacking.** A truncated or corrupted fetch must fail loudly
  here, not produce a broken ORT session hours later. *(Verified 2026-08-06: the
  downloaded `small` package matches its published digest.)*
- **Cheap update checks.** Poll 1.34 KB instead of 162 MB.

This maps onto existing machinery — `models/manifest.json`, `js/model-assets.js`,
`js/model-registry.js`. Record the darktable release tag and model version in the
manifest so a cached model's provenance is inspectable.

### Step 2 — freeze shapes *(partly mandatory)* **[MEASURED]**
- encoder `high_res_feats_0` → `[1,32,256,256]` ✔ freeze
- encoder `high_res_feats_1` → `[1,64,128,128]` ✔ freeze
- decoder mask count → `3` ✔ freeze
- decoder `num_points` → **leave dynamic. Do NOT pad.**

Freezing the encoder outputs is right — they are genuinely static and ORT would
otherwise size its arena for an unbounded worst case.

**`num_points` padding was wrong and is actively harmful.** Label `-1` is not
"semantically a no-op": the graph already appends its own padding point, and
every extra `-1` slot adds another `not_a_point_embed` token to the prompt
attention. Quality degrades monotonically with each one:

| real clicks | slots | predicted IoU | mask IoU vs unpadded |
|---|---|---|---|
| 1 | 1 | **0.8995** | 1.000 |
| 1 | 4 | 0.5382 | 0.852 |
| 1 | 8 | 0.0001 | 0.571 |
| 2 | 2 | **0.7696** | 1.000 |
| 2 | 8 | 0.0001 | **0.227** |

Pass exactly the clicks you have. The consequence is that `enableGraphCapture`
cannot be used on the decoder — which costs nothing, since the decoder is ~23 ms
and the encoder is where all the time and memory live.

### Step 3 — mixed-precision fp16
```
convert_float_to_float16(model, keep_io_types=False, op_block_list=[])
```

**fp16 ships — but only on ORT-Web ≥ 1.24.3. [MEASURED]**

The pinned ORT-Web **1.22** build's WebGPU fp16 kernels produce a materially
wrong embedding for this model. That is a runtime bug, fixed upstream:

| ORT-Web | posFrac (fp32 ref 0.2915) | verdict |
|---|---|---|
| 1.22.0 | 0.9962 | wrong |
| **1.24.3 / 1.26.0 / 1.27.0** | **0.2910** | correct |

seglab vendors **1.27.0** and ships fp16 — 78 MB encoder, 8 MB embedding, and
resident memory under 2 GB (FINDINGS §5). The failure below is what the 1.22
build did, and is why a WebGPU lane can never be validated on CPU alone. The error compounds with depth through Hiera — `high_res_feats_0` (stage
1) is close, `image_embed` (deepest) is far off — and the resulting mask covers
99.6 % of the frame where the correct answer is 29 %. It fails *quietly*: the
model still reports predicted IoU 0.987 for that mask.

| variant | posFrac | logit range |
|---|---|---|
| CPU fp16 (reference) | 0.2919 | −13.52 … 13.91 |
| **WebGPU fp32** | **0.2915** | −13.43 … 13.81 |
| WebGPU fp16, blanket | 0.9962 | −1.08 … 10.76 |
| WebGPU fp16, block LN+Softmax+Resize | 0.9961 | −1.08 … 10.74 |

Two things follow. **A block list does not help** — the error is not in
LayerNorm or Softmax, so the reasoning about `Cast` nodes above is moot. And
**CPU validation cannot catch this**: on CPU, fp16 matches fp32 at cosine
0.999994. Only comparing the browser's actual embedding finds it
(`spikes/sam21/precision.html`, which is now the regression gate for any ORT-Web
upgrade).

Shipping artifacts are `encoder.fp16.onnx` / `decoder.fp16.onnx` on ORT-Web
1.27, and they are the ONLY artifacts. The fp32 pair used to sit in `models/`
as a fallback for a runtime whose fp16 could not be trusted, selected by a
`precision` switch in `js/sam21-lane.js`. Both are gone: the switch was a
mutable config field nothing ever wrote (so the fallback was unreachable), and
keeping 175 MB of weights for an unreachable path is not a safety net. If a
future ORT-Web regresses fp16 the answer is to pin the runtime, not to flip a
field at runtime — `spikes/sam21/ortver.html` is the gate to re-run on any
runtime upgrade, and re-export with `scripts/export-sam21.py` if it ever fails.

`keep_io_types=False` remains correct *for the fp16 build* — the encoder emits
fp16 so the fp16 decoder needs no boundary `Cast`.

**Two `onnxconverter-common` 1.16 bugs to know about.** It retypes a pre-existing
`Cast` node's declared output to fp16 but leaves the node's `to` attribute at
FLOAT, so ORT rejects the model outright (11 nodes in the decoder;
`repair_casts()` fixes it). And with `op_block_list=['LayerNormalization',
'Softmax']` it hands `Resize` an fp16 constant where fp32 is required.

### Step 4 — validate numerically *(mandatory)*
`onnxconverter-common` has mistyped a head in this repo before: fp16 broke the
YOLOE **LRPC** head and that export had to stay fp32. (The script that hit it,
`scripts/export-yoloe.py`, is gone — it built the prompt-free detector the text
lane replaced, and the current `export-yoloe-text.py` has no LRPC head to
mistype.) Never ship an unvalidated conversion.

- encoder: cosine similarity + max-abs delta, fp32 vs fp16, per output
- decoder: mask IoU after threshold, on real click points
- gate: IoU ≥ 0.999 on the fixture set, else populate the block list and repeat

### Step 5 — `.ort` conversion *(removed — it breaks the model)* **[MEASURED]**

The reverse of the original claim holds on the pinned runtime:

| artifact | result |
|---|---|
| `encoder.onnx`, `graphOptimizationLevel: 'disabled'` | **loads, 1739 ms** |
| `encoder.onnx`, `'all'` | loads, 1830 ms |
| `.ort`, Runtime style | fails — `[Transpose] blocks.0/attn/Transpose_1 … no GPU data for input` |
| `.ort`, Fixed style | fails, and inflates 78 MB → 156 MB by undoing fp16 |

The plain `.onnx` is what ships. `graphOptimizationLevel: 'disabled'` stays, but
because it is *faster* (1739 vs 1830 ms), not because the model needs
pre-optimizing.

### Step 6 — emit
`models/sam21/` + regenerate `models/manifest.json`.

### Expected output

| Artifact | Size |
|---|---|
| `encoder.fp16.ort` | ~78 MB |
| `decoder.fp16.onnx` | ~10 MB |
| embedding at runtime | 8 MB |

---

## 8. Runtime architecture

### 8.1 Import — preview-first, never full-res

```
file → embedded JPEG preview (image-raw.js already lifts this)
     → createImageBitmap(blob, {resizeWidth: 2560, resizeQuality: 'high'})
     → display bitmap  ~26 MB
     → proxy 1536×1024 ~6 MB   (per-axis — see 8.1a)
     → YOLOE pass
LibRaw develop: EXPORT ONLY, strip-wise
```

`createImageBitmap` with `resizeWidth` decodes and downscales *inside the
browser*. The full-size RGBA buffer never enters your heap — for a 45 MP frame
that is the difference between ~180 MB and ~26 MB.

**This is not merely an optimization.** `2680558334.nef` (Z 8, High Efficiency /
TicoRAW) **cannot be decoded by LibRaw at all** — see PLAN.md 2026-07-17. HE
NEFs always embed full-size previews, so preview-first is the *only* viable path
for the canonical test image. The develop fallback exists solely for older
preview-less RAW.

### 8.1a The proxy is sized per AXIS, not by the long edge **[MEASURED]**

The encoder resizes its input to a **1024×1024 square**
(`sam21-lane.js`, `drawImage(bitmap, 0, 0, SIDE, SIDE)`), so usable detail is
capped **per axis**. A long-edge cap starves the short one: a 3:2 frame became
1024×683, stretching 683 real rows to fill 1024.

Sizing so the **short** edge reaches 1024 (1536×1024 for 3:2), scored against
ground truth at native resolution on the canonical NEF:

| case | proxy 1023×682 | proxy 1400×933 | Δ boundary IoU |
|---|---|---|---|
| tulip | 0.8198 | 0.9020 | **+8.21** |
| tulip 2 | 0.8395 | 0.9208 | **+8.13** |

~**10×** what the whole refinement stage buys on real logits (§10c), at no model
cost: same 1024² input tensor, same session, same arena. Cold-start peak over
the process tree (`scripts/mem-peak.sh`, five clicks, canonical NEF):

| | peak |
|---|---|
| long-edge 1024×683 | 2541 MB |
| per-axis 1536×1024 | 2374 MB |

**Memory-neutral** — the cold encode dominates and swamps the extra few MB.
`postMs` rises ~12–48 → ~19–58 ms, still under `decodeMs` (78–205 ms).

Both peaks breach the 2.2 GB contract; that is pre-existing (the encoder
session, §9), not caused by the proxy. Weights are 78 MB of a ~2400 MB peak, so
int8/mixed precision addresses ~3 % of it — the cost is ORT's WebGPU activation
pool, not the weights.

Three guards, tested in `verify.mjs`:

- **`proxyLongMax` 2048 / `proxyPixelMax` 2.1 M** — an 8:1 pano clamps to
  2048×256, not 8192×1024.
- **Manual `?proxy=` opts out** — an explicit number is the user's.
- **Pressure L3 sets `proxyShortMax = 0`** — the boost goes first.

### 8.1b What the proxy costs, and what pays for it **[MEASURED]**

The proxy is the only knob that moves click latency at all: encode and decode
are **flat** in proxy size (the encoder squashes to 1024², the decoder emits
256²). Everything the proxy scales is CPU post-processing, and post-processing
is 60–70 % of a warm click. `js/hardware-fit.js` spends that cost against
`postBudgetMs` and walks the short edge down a ladder when a device cannot pay
— llmfit's technique (class-table estimate → utilisation band → quantisation
walk → real benchmark replaces the estimate), adapted to what a page can see.

The cost has **two** terms, not one. Measured 2026-08-14 (Apple metal-3,
headless Chromium, a 1536×896 proxy, a 12-point click grid × 3 repeats,
per-stage timings from `js/mask-refine.js`):

| stage | scales with | cost |
|---|---|---|
| `upsampleLogits` | proxy | ~6.0 ms |
| `bandAlpha` (full frame) | proxy | ~2.2 ms |
| raw-mask copy | proxy | ~0.3 ms |
| `refineField` (guided filter) | refined rect | ~24.6 ms/MP |
| `bandAlphaRect` | refined rect | ~4.8 ms/MP |

⇒ **8.6 ms per proxy-MP fixed + ~30 ms per rect-MP**. On the *same* 1.376 MP
proxy that is 12.2 ms for a 9.7 kpx band and 52.6 ms for a frame-spanning one —
a 4.3× spread that a single ms/MP figure charges to the machine. So the model
carries the scene explicitly:

```
postMs = proxyMP × rate × (0.22 + 0.78 × bandFraction)
```

Normalised through that shape, eleven of the twelve clicks return **35.6–43.3
ms/MP** across a 140× range of band area — a genuine device constant.
`sam21-adapter` reports `bandPixels` per click so `app.js` can feed the rate
and the fraction back separately; a click that reports no band is dropped
rather than normalised by a guess. The stored figure expires after 14 days
(`?post=reset` forces it) — it describes the machine as it was that day, on
battery or thermally throttled, and the fit only ever ratchets downward.

Two levers that look like they should help and do not:

- **Narrowing `BAND = 6`.** Frame-spanning rects on `streetlight.jpg` are
  genuine object extent (coverage 17–19 %), not a fat band: thresholding at 3
  logit units saves ~2.8 ms on the worst real case and nothing at all on a
  spanning object. Block-skipping prototypes (128 px and 192 px blocks with a
  24 px halo) were **slower** than one rect refine.
- **Trading guided-filter scale.** 4 → 6 → 8 moved the refine stage
  3.53 → 3.27 → 3.13 ms — ~11 % of one stage, ~0.4 ms per click — because
  extraction, downsample and `compositeUp` are full-resolution regardless.
  r=8/s=4 is measured-optimal for quality (§10b) and stays.

The same judgement runs on the text lane's tile grid: `detectorMaxCells` is
clamped by measured per-cell inference (`detectorBudgetMs`, 2500 ms) on top of
the class-derived memory cap, which stays — an ORT arena cannot be timed.

### 8.2 SAM encode — on demand, in a worker

```
create encoder session
    URL-sourced (streamed) + external-data weights   → no pinned ArrayBuffer
    static shapes, graphOptimizationLevel: 'disabled'
  → run with preferredOutputLocation: 'gpu-buffer'
  → copy 3 outputs into OWN GPUBuffers               ← ORT's die with the session
  → encoderSession.release()                         ← peak collapses here
  → persist embedding to OPFS, keyed by image hash
  → one dummy decode to warm the WebGPU pipeline
```

**Why the copy matters.** ORT's output buffers are allocated from the session
arena. Releasing the encoder frees them. Copying into buffers you own (a
GPU→GPU copy of 8 MB, sub-millisecond) is what allows zero-copy decode *and*
encoder disposal simultaneously.

**Session lifetime: ref-counted with drain-release**, not release-immediately. A
batch import should not pay repeated shader compilation. Release when the encode
queue drains, or on `samIdleMs`.

**Load-time transient.** Constructing an `InferenceSession` from an
`ArrayBuffer` briefly holds the buffer *plus* the parsed graph *plus* the GPU
buffers — roughly 2–2.5× weights. Passing a URL and using external-data format
avoids ~150–300 MB of avoidable spike.

### 8.3 Click — the responsiveness path

```
decoder: warm, fp16, static-shaped   (graph capture was tried and rejected — §12)
inputs:  already GPU-resident  → zero uploads, zero casts, zero dispatch setup
paint:   composite mask as a GPU texture
refine:  guided filter (§10) runs AFTER first paint, repaints if better
```

The decoder is 19.6 MB running against a 64×64 embedding. **It is not
compute-bound.** Essentially all click latency is overhead — dispatch, uploads,
casts, dynamic-shape re-planning. Graph capture + GPU residency + static shapes
removes nearly all of it. This is why "fastest" and "no quality compromise" are
not in tension: none of these touch numerics.

Share **one** `GPUDevice` between ORT and the compositor so buffers pass without
cross-device copies.

### 8.3a What the selection looks like — fill *and* border

A 32%-alpha tint alone answers "something is selected" but not "which pixels".
That distinction is the whole product: §10c establishes that single-click mask
quality is finished and that what is left is SAM choosing the wrong *object*
(§10a — three candidates at 1.9 / 6.3 / 13.2 % coverage on one measured click).
A user cannot cycle with `C` toward the right interpretation without seeing
exactly where the current one ends, and a translucent wash over a busy photo
does not show that.

The border is built once per committed mask, next to the tinted fill, and cached
on the same key:

1. threshold the mask to a **hard core** at alpha ≥ 128 — dilating the feathered
   alpha directly gives a wide soft double band instead of one line;
2. draw that core eight times at ±`r`, which *is* its dilation, then
   `destination-out` the core itself: what survives is a band hugging the
   outside of the selection;
3. tint it with the accent, and paint it under a `shadowBlur` halo — the accent
   alone disappears against a light subject.

Eight `drawImage` calls, not a JS dilation pass, and never on `pointermove`: a
brush stroke keeps drawing its own transient preview and picks the border up at
commit. `r` is relative to the frame (`0.003 × min(w,h)`, floor 2 px), so the
line reads the same on a 1024 px proxy and a 4000 px native frame, and it is
built at mask resolution — display scaling is CSS, so the border stays exactly
on the boundary at any zoom.

### 8.4 Export

Strip-wise LibRaw develop, admission-controlled. Mask upscaling per §10.

### 8.5 Tab lifecycle

Multiple tabs are a legitimate workflow for an editing tool — two photos open
side by side. A single-instance lock ("seglab is already open elsewhere") would
be simpler and is **rejected**: it would feel broken. Three mechanisms instead.

**Visibility-driven dormancy.** On `document.visibilityState === 'hidden'`:

```
release  encoder session   (the multi-GB arena)
KEEP     decoder session   [MEASURED] — see below
release  YOLOE session
KEEP     embedding (8 MB) + proxy (4 MB)
```

**[MEASURED] The decoder cannot be released here.** Releasing the *last* ORT
session destroys the WebGPU device (ORT gives it up when nothing holds it), and
a device loss clears every embedding — so "release both sessions, KEEP the
embedding" is not a reachable state. Measured: release encoder, then decoder,
and `deviceLost` goes true ~400 ms later with the embedding map empty. It fails
silently, because `reason: 'destroyed'` is correctly treated as our own
teardown. The 9.9 MB decoder is therefore kept as the device anchor, which is
what makes keeping the 8 MB embedding possible at all. Full write-up in
`spikes/sam21/FINDINGS.md` §5c.

A hidden tab drops to ~30 MB plus page overhead. On becoming visible it rebuilds
the decoder — a second or two of shader compilation, hidden behind the same
warm-up dummy decode as §8.2. Since users look at one tab at a time, this alone
means one tab holds heavy resources in the common case. Side-by-side windows are
the exception, and the Web Lock covers them.

**Cross-tab heavy-job lock.** See §9 — `navigator.locks` promotes `I3` to
machine-wide.

**Shared OPFS cache turns a second tab into a benefit.** OPFS is shared
same-origin, so a second tab opening the same photo gets its embedding for free
with no encode at all. Guard writes with a Web Lock keyed on the image hash so
two tabs do not race the same file.

**`GPUDevice.lost` handling is mandatory, not optional.** Several WebGPU devices
across tabs makes Chrome far more likely to drop one under pressure. Mark
sessions invalid, rebuild on next use, never crash. Good practice regardless;
multi-tab moves it from theoretical to likely.

---

## 9. Memory contract

### Projected peaks

| Phase | Peak | vs I1 (900 MB) |
|---|---|---|
| Import + YOLOE | ~110 MB | 12 % |
| Steady state (clicking) | ~270 MB | 30 % |
| SAM encode (transient, often never) | ~650 MB | 72 % |
| Export (strip-wise develop) | ~200 MB | 22 % |

Steady state breakdown: decoder 10 + embedding 8 + proxy 4 + display 26 +
YOLOE 44 + app/JS/DOM ~180.

### 9.1 The peak is ORT's buffer cache, and it is a setting **[MEASURED]**

The ~1 GB attributed to "the encoder session" is neither the weights (fp16
encoder = 78 MB) nor session construction. Staged with a fresh profile and a
per-PID footprint sampler:

| stage | gpu-process | renderer |
|---|---|---|
| device only | 119 MB | ~50 MB |
| encoder session built | 226 MB | 792 MB |
| after one encode | **1117 MB** | 709 MB |
| after decode ×3 | 1125 MB | 763 MB |

Building the session costs 226 MB. The first **run** adds ~890 MB and holds it.
The WASM heap tops out at 260 MB and never grows again, so the renderer term is
session build plus staging, not a leak.

That ~890 MB is ORT's WebGPU storage-buffer cache in `Bucket` mode — every
intermediate rounded up to a bucket and never returned. `lazyRelease` returns
them at end of run. Isolated on the encoder:

| mode | gpu | all-Chrome | encode |
|---|---|---|---|
| Bucket (default) | 1128 MB | 1924 MB | 885 ms |
| **lazyRelease** | **290 MB** | **1157 MB** | **665 ms** |
| Simple | 1444 MB | 2224 MB | — |
| Disabled | crashes: "used in submit while destroyed" | | |

Logits are identical (pos 0.898148, sum 279177.323, iou 0.993164) and the encode
is not slower. On the whole app (NEF, 1536×1024 proxy, five clicks), three runs
per arm: Bucket 2369/2336/2351 MB, lazyRelease 2035/2036/2034 MB — **~315 MB,
and the crossing of the 2.2 GB ceiling**.

**Mask lane only.** The detector/text worker measures the other way: it builds,
runs once and terminates per phrase, so the churn costs more in renderer staging
than the cache retains — worker renderer 1945/1957/2138 MB on Bucket against
2930/2775/2782 MB on lazyRelease. One long held encode and a short repeated one
want opposite policies.

`epConfig` is not upstream API. ORT-Web forwards a fixed key list to the native
EP and the cache modes are not on it, so `scripts/download-models.mjs` patches
the vendored bundle (anchored; it throws if an ORT bump moves the anchor) and
verify.mjs gates the artifact, the patcher and the caller. The CDN fallback in
`ort-loader.js` is unpatched — it runs, without the reclaim.

Two things this rules out, both by direct measurement:

- **Quantization cannot fix the ceiling.** The fp16 encoder is 78 MB of a
  ~2350 MB peak; int8 would save ~39 MB. The cost was never weights.
- **The image barely matters.** A 1200 px demo with no RAW decode peaks within
  ~40 MB of a 45 MP NEF.

### Admission control

`js/heavy-job-queue.js` becomes an **admission controller**, not just a
serializer — and its scope widens from this tab to the whole machine:

```
admit(job):
    await navigator.locks.request('seglab.heavy', async () => {
        projected = current_resident + job.projected_peak
        if projected < I1:        run
        else:                     evict, recheck
        if still over:            defer, surface status
    })
```

**The Web Lock is what makes `I3` machine-wide.** It is same-origin, cross-tab,
browser-managed, and released automatically if a tab crashes or is closed — so
there is no stale-lock recovery to write. Wrapping the existing queue in it is a
small change for a large guarantee: only one encode, develop, or export runs
anywhere on the machine at a time, regardless of tab count.

Surface the wait. A tab blocked on another tab's encode should say so ("waiting
on another seglab tab") rather than appear frozen.

Eviction order: encoder session → embedding (to OPFS, recoverable) → decoder
session → YOLOE session.

Projections are **arithmetic on known tensor shapes**, not guesses:
`1×256×64×64×2 bytes = 2 MB`, and so on. That is what makes I1 enforceable
rather than aspirational.

### 9.2 The ledger has to be complete, because on WebKit it is the only input

`decidePressure` reads three signals: measured agent-cluster bytes
(`measureUserAgentSpecificMemory`), JS heap, and the app's own allocation
ledger. The first two are Chromium-only. On WebKit the ledger is not a
cross-check — it *is* the governor, and anything missing from it is not merely
under-counted, it is invisible.

The text lane was missing from it. It runs in a **separate worker with its own
ORT arena**, so none of it appears in the SAM lane's status object the ledger was
built from. Measured in that worker's process (DESIGN-TEXT-LANE §3, staged
per-PID table): 326 MB idle → 962 MB with the session built → 1034 MB after the
first run. Disposing the *session* returns only the GPU share (854 MB floor);
the arena comes back when the **worker** is terminated. So residency is keyed on
the worker, and the app charges itself 700 MB while one is up — 1030 MB on
WebKit, which is where the heavier figure was observed before a process reap.

Two consequences, both required for the entry to mean anything:

- **The shed can act on it.** `shedMemory(≥1)` now terminates an idle detect
  worker. It refuses while a detection is in flight — terminating mid-search
  rejects the call, which the user reads as a failed search, not as relief.
- **It must not fire during normal operation.** On the product path the lanes
  never peak together (`detectorEvictOnEncode` drops the embedding first), so a
  text search sits at roughly base + worker-warm + detector ≈ 1.6 GB, which is
  the 1607 MB actually measured. A search *with* an encoder live projects past
  the budget — and that case measures 2558 MB, so the shed is correct there.

What this does **not** do is bound the lane's own peak. That is still policy's
job (`detectorMaxCells` demotes to 5 wherever `deviceMemory` is unreadable —
WebKit and Gecko), and §"detectorDispose" in DESIGN-TEXT-LANE records why
terminating after *every* search makes the peak worse rather than better.

---

## 10. Mask quality — where the math matters more than the model

**SAM outputs 256×256. A Z 8 export is 8256×5504. That is a ~32× enlargement.**
How this gap is bridged determines perceived quality far more than `tiny` vs
`small` does.

**Wrong:** threshold at 256×256, then upscale the binary mask. Result is blocky,
stair-stepped edges that no encoder upgrade can fix.

**Right — three steps, all closed-form, all GPU:**

1. **Upsample logits, not the mask.** Bicubic on the continuous score field,
   threshold at full resolution. Sub-pixel boundaries for free.
2. **Guided filter against the full-res photo in COLOUR**, using the photograph
   itself as the guide image. Snaps the mask boundary onto the true object edge.
   O(n) with box filters, runs as a WGSL compute pass, no readback.
3. **Narrow-band matting.** Solve alpha only in a strip around the boundary, so
   hair and foliage get soft edges at negligible cost.

`js/edge-refine.js` and the cv-refine WASM lane already provide part of this.
The change is feeding them **logits instead of a thresholded mask**, and moving
the hot part to a compute shader.

### 10b. Why the guide is colour, and why it is Y/Cb/Cr

Step 2 originally guided on Rec.601 luma. That is blind to an **isoluminant**
boundary, and those are ordinary, not exotic — pink petals on green foliage,
orange paint on grey road. Measured on a rose-against-foliage crop:

- **33 % of the true boundary carries under 2 % luma contrast**;
- the mean boundary luma step (0.066) is only **2.9×** the mean step of the
  petal texture *inside* the object, so the guide can barely tell an edge from
  shading — and where it cannot, it follows the shading.

Full RGB carries 2.09× the contrast of luma there. Moving to He et al.'s
3-channel guided filter over seven real crops (`scratchpad/bench.mjs` — rose,
two tulips, a leaf, a tram, a lamp pole, overhead wires):

| guide | mean IoU | mean boundary IoU |
| --- | --- | --- |
| luma (was) | 0.902 | 0.735 |
| **colour Y/Cb/Cr, chroma ε ×3 (now)** | **0.935** | **0.864** |

Per crop the boundary-IoU gain runs **+3.2 to +33.6 points** on chromatic
subjects. Two details are load-bearing:

- **The ε matrix is diagonal, not He's scalar εU, and the guide is Y/Cb/Cr
  rather than raw RGB.** On a grey subject the RGB channels are collinear, Σ is
  near-singular, and the solve chases noise — measured as **−3.0 pt** boundary
  IoU on 1–2 px wires against sky. A decorrelated basis with a heavier ε on the
  two chroma axes degenerates cleanly to the luma filter when there is no chroma
  to use. Verified: on a greyscale guide the colour path and the luma path put
  the crossing in the *same* place to 0.00 px.
- **The ×3 chroma ratio is a constant on purpose.** Two adaptive rules were
  built and measured, and both lose — see the note in `js/mask-refine.js`.

**Canny edge snapping was tried and rejected.** Detecting edges (Di Zenzo colour
structure tensor → NMS → hysteresis) and pulling the zero crossing onto the
nearest edge loses on **all seven crops**: −0.6 pt on the rose, −9 to −18 pt on
the thin pole and the wires, where it snaps the boundary onto neighbouring
texture. An edge *detector* discards the magnitude information that makes the
guided filter work; it is a strictly weaker tool here.

---

### 10c. The refinement stage is finished — the remaining error is upstream **[MEASURED]**

Three further families were built and measured against the shipped colour
guided filter. All are recorded here so they are not retried.

| candidate | synthetic bench (7 crops) | real SAM logits |
| --- | --- | --- |
| **WGIF** — ε scaled by local variance (Li 2015) | −0.69 … −2.52 pt | not reached |
| **Di Zenzo tensor ε** — colour structure tensor as a continuous weight | −1.05 … −4.93 pt | not reached |
| **Fast Bilateral Solver** (Barron & Poole 2016) | **+0.23 pt**, +20.7 on the tram | **−0.54 pt** |
| FBS + thickness selector (`2·area/perimeter`) | **+4.43 pt** | not reproduced |

Both ε-weighting schemes fail like Canny, from the opposite direction:
sharpening ε *at* an edge makes the fit chase interior texture, destroying thin
subjects (−9.9 pt on a 5 px pole, −20.0 on 1–2 px wires). Both converge to the
shipped constant ε as λ→∞ — the sanity check that the comparison is fair.

The bilateral solver is a *global* colour-affinity solve, not a local window
fit. On the synthetic bench it is dramatic (tram +20.7, leaf +8.6) but fails on
thin structures (pole −25.4): a 1–2 px wire owns no grid vertex and averages
into the background. A thickness statistic separates those cleanly (5.4 / 7.1 px
vs ≥22.6 px), giving a hybrid worth +4.43 pt.

**None of it survives real SAM logits.** On fields captured through `clickAt`,
the solver loses at every confidence scale (1–16 px) and λ (8–128), barely
moving the boundary across a 16× sweep — not mistuned, nothing to correct. Raw
0.9248 → shipped filter 0.9323: the whole refinement stage is worth **+0.76 pt**
on a real field.

The synthetic bench manufactures headroom by coarsening the field 3.5×. It is a
fair model of *"recover a boundary from a coarse field"* and it justified the
colour guide, but it **overstates** what any refinement can buy.

Act on this: **the loss is in the field, not the filter.** §8.1a improved the
field by +8.2 pt — an order of magnitude more than this stage has left. Do not
port the solver to C++/WebGPU: 20× slower, worse on real data.

*Caveat:* only 2 of 6 attempted cases were valid — elsewhere SAM selected a
different object than the bench's colour rule describes (leaf raw IoU 0.17,
tram 0.33), making those rules circular for real logits. Thin evidence, but the
two agree and the parameter-insensitivity is the stronger signal.

## 10a. Which of the three masks — arbitration, not argmax

Step 10 fixes the *boundary*. It cannot fix picking the wrong **object**, and
that is a separate, larger source of "the mask is wrong".

Every decode returns **three** masks (roughly subpart / part / whole) and three
predicted IoUs. This lane took `argmax(iou_predictions)`. But predicted IoU
answers *"how well would this mask score against its own target"*, which is not
the question a click asks, and it is silent on every failure below:

| failure | what argmax does | what `js/mask-select.js` does |
| --- | --- | --- |
| nested object (a rose cluster, a person in a coat) | a confident sub-part outscores the whole, so the top score **is** the mask that leaves parts out | offers the other candidates; the user cycles with `C` |
| exclude click | a candidate still covering the point can win | hard-disqualified — an instruction, not a preference |
| box prompt (all text search) | a candidate spilling far outside the detector's box can win | ranked down by in-box fraction |
| no real boundary | a mushy field can score high | ranked down by SAM's own stability score |
| hierarchy drift between clicks | click 2 may switch level, so a point on a sleeve reshapes the person into a sleeve | continuity: prefer the candidate agreeing with what the user was already shown |
| speckle / enclosed gaps | shipped as-is | upstream SAM's `remove_small_regions`, both modes |

The arbitration order is **prompt consistency → continuity → rank**, and the
composition matters: an exclude click disqualifies the too-large candidate
*before* continuity runs, so continuity can never fight an explicit shrink.

**Cycling is the honest part.** One point on a rose cluster makes "this bloom"
and "the cluster" *both correct*; no scoring rule resolves that, only the user
can. The planes are already in memory, so `C` / `shift+C` is an upsample and a
guided filter — a repaint, never a decode — ordered by area so forward always
grows the selection.

### Region hygiene is gated, and the gate is measured

Removing small components is the classic cleanup and it is **actively harmful
here** if applied naively. Measured on the streetlight crop with real pixels
through the real pipeline (`scratchpad/quality.mjs`):

| rule | IoU | boundary IoU |
| --- | --- | --- |
| shipped pipeline, no hygiene | 0.846 | 0.842 |
| remove components < 10% of the largest | 0.796 | 0.791 |
| shipped gates (dominance + absolute floor) | **0.846** | **0.842** |

A 256² grid shatters wires and thin arms into dozens of small components, and
every one of them looks like speckle by size — that mask scores **0.568**
dominance against ~0.9 for a compact subject with real speckle. Severing a wire
*is* the "parts left out" failure, so four conditions must hold together before
anything is removed: dominance ≥ 0.85, ≤ 16 cells absolute, < 2% of the mask,
and no include click in it.

Hole filling needs no gate: 0.000 IoU cost on the streetlight, and on the
reported rose (`scratchpad/rose.mjs`) it adds 513 px, all of them shadowed
crevices *inside* the bloom — background to a colour ground truth, part of the
flower to a human.

---

## 11. Configuration collapse

`standard8` is already the correct shape — `js/policy.js:78-86` documents it as
built for "the worst device that reaches it (an 8-core / 8 GB laptop that may be
heavily loaded)." The work is deletion, not design.

### Single config

```
proxyMax          1024      # SlimSAM/SAM2 native encode edge; lower only
                            #   makes the model upscale, saving nothing
displayMax        2560
exportMaxMP       12
exportMaxSide     5120
hdExportDecode    true
memBudgetMB       900       # now I1, not a soft hint
draftCacheMax     1
maxResidentHeavy  1
samWebGPU         true      # unconditional
autoEscalate      false     # on-demand user action only
```

### Delete

| What | Where |
|---|---|
| four other `PRESETS` | `js/policy.js:16-230` |
| `TIER_MIN_GB`, `profileForMemory`, `autoTierFor`, `lowerProfile` | `js/capability.js:56-104` |
| `isMemoryLocked` + locked-budget branching in `resolveBudget` | `js/policy.js:233-302` |
| `PROFILE_RANK`, `profileOverride`, `setProfileOverride` | `js/app.js:36,236,269-293` |
| `#profile-select` | `index.html:397-404` |
| `?profile=` parameter | `js/policy.js` |

### Keep

- `gpuTierFor` (`js/capability.js:74-79`) — repurposed as the hard gate
- memory governor + `applyMemoryPressure` — backstop, one-way ratchet.
  Flatten its `profile === 'lite'` branches at `js/policy.js:398-400`.
- YOLOE scale toggle — a feature choice, not a tier

### Convert

- `js/sam-engine.js:644-648` WASM demotion → hard error
- escalation → explicit per-image user action

---

## 12. Optimization catalog

| # | Technique | Wins | Costs |
|---|---|---|---|
| 1 | YOLOE-first routing | encoder often never loads | routing logic |
| 2 | Preview-first import | 180 MB → 26 MB | required for HE NEF anyway |
| 3 | Admission control | makes I1 enforceable | queue complexity |
| 4 | Mixed fp16, `keep_io_types=False` | ½ weights, ½ arena, no casts | needs validation |
| 5 | Static shapes | unlocks 6, stops over-allocation | conversion step |
| 6 | ~~`enableGraphCapture`~~ | **ruled out** — §1.1 rejects it on quality, and dispatch is 23–57 ms of a ~1.1 s encode (FINDINGS §8), so there is nothing left to win |
| 7 | GPU-resident embedding | zero uploads per click | own-buffer copy |
| 8 | Encoder release after encode | peak collapses to steady state | rebuild on new image |
| 9 | Ref-counted drain-release | batch imports skip recompiles | lifetime tracking |
| 10 | OPFS embedding cache | fastest encode is a skipped one | 8 MB/image storage |
| 11 | Streamed weights + external data | −150–300 MB load spike | conversion step |
| 12 | Logit upsampling + guided filter | the actual quality win | compute shader |
| 13 | Async refine after first paint | mask appears immediately | double repaint |
| 14 | Shared `GPUDevice` | no cross-device copies | init ordering |
| 15 | Warm-up dummy decode | first real click is warm | one wasted run |
| 16 | Cross-tab Web Lock | `I3` holds machine-wide | tabs may wait on each other |
| 17 | Visibility dormancy | hidden tab → ~30 MB | rebuild on re-focus |
| 18 | Cross-tab OPFS reuse | 2nd tab on same photo encodes nothing | write lock per hash |

---

## 13. Where WebAssembly belongs

**Not on the click path.** The architecture keeps data GPU-resident; WASM cannot
read a `GPUBuffer`. Putting WASM on the hot path forces a GPU→CPU readback on
every interaction — destroying the zero-copy design to accelerate code that is
not the bottleneck. It would be measurably slower.

**Keep WASM where it already is:** LibRaw (`scripts/build-libraw-wasm.sh`,
export-time, CPU-bound) and cv-refine (`scripts/build-cv-wasm.sh`, off the click
path). To speed those up, enable **SIMD** in the Emscripten build rather than
rewriting the language.

**The modern equivalent of "drop to assembly" here is a WGSL compute shader** —
same instinct, but on the processor where the data already lives.

*Note:* AssemblyScript and Emscripten are alternative toolchains that both
target WASM, not stages of one pipeline. AssemblyScript carries a GC and a less
mature optimizer; for numeric work, C/C++ via the Emscripten toolchain already
in this repo is the better choice.

---

## 14. Work phases

**All seven phases are done.** The table is kept because the *ordering* is the
part worth remembering, and because several sections below still refer to a
phase by number.

| Phase | Work | Gate | State |
|---|---|---|---|
| 0 | Config collapse + WebGPU/`shader-f16` hard gate | one code path exists | shipped — §11 |
| 1 | Preview-first import + admission controller + **cross-tab Web Lock** | **I1, I3 assertable** | shipped — §8.1, §9 |
| 2 | YOLOE-first routing + visibility dormancy | ships value with no new model | shipped, then superseded: dormancy never fired on visibility and was replaced by the idle-exit rung (§8.5) |
| 3 | Conversion pipeline + **measure encoder arena on the Air** | go/no-go for `small` | shipped — arena measured, `small` kept (§6.1) |
| 4 | SAM2 lane: split lifecycle, GPU-resident, `device.lost` | **I2 assertable** | shipped, minus graph capture (§12 row 6 rules it out) |
| 5 | Logit upsampling + guided filter | quality target | shipped and **finished** — §10c |
| 6 | OPFS embedding persistence (cross-tab shared) → Phosmith | — | shipped — §8.5 |

**The ordering was deliberate.** Phases 1–2 delivered the responsiveness and
memory guarantees *independently of SAM*, so the riskiest work (3–4) sat behind
a product that already worked. Phase 3 gated phase 4: had the measured arena
blown I1, the fallback was `tiny` — it did not, and §6.1 later showed `tiny`
would have bought nothing anyway.

**The Web Lock belonged in phase 1**, not later. It is the same file and the same
abstraction as the admission controller, so building it then cost almost
nothing; retrofitting cross-tab safety afterwards would have meant revisiting
every heavy call site.

---

## 15. Verification

| Invariant | How |
|---|---|
| I1 | `measureUserAgentSpecificMemory()` sampled across import → encode → click → export, on the Air, with `2680558334.nef` |
| I2 | phase A drives five real clicks over objects of deliberately different band size and asserts the median `lastRun.postMs` against `postBudgetMs` (max gets 2x slack — one CI scheduling hiccup is not a broken invariant); a second check asserts every warm click reported `bandPixels`, without which the figure cannot be attributed to the device |
| I3 | queue assertion: in-flight heavy count never exceeds 1 **per tab** |
| I3 cross-tab | three tabs, encode triggered in all three within one second; assert the lock serialises them and total peak stays under ~1 GB |
| dormancy | hidden tab settles to ~30 MB within the shed window; re-focus restores a working decoder |
| integrity | corrupt a cached `.dtmodel` byte; assert the sha256 check rejects it before unpack |
| quality | fp32 vs fp16 mask IoU ≥ 0.999 on the fixture set |

Multi-tab cases need a real browser with several tabs, which `verify.mjs` cannot
drive on its own. Run them through the dev-browser harness against
`scripts/dev-server.mjs` on :8788 — COI headers and model caching both depend on
that server, so a plain static server will not reproduce the conditions.

`verify.mjs` now carries 200 `check()` assertions across 3060 lines, including
both RAW fixture phases. `--fast` runs the three that need no browser (pure
logic, heavy-job queue, static source scans — 116 assertions) and stops before
the browser phases; those need Playwright's Chromium and the dev server on
:8788. The profile-related assertions phase 0 was meant to rewrite are gone with
the presets — what is left refers to the single config.

---

## 16. Risks and open questions

Three rows that used to head this table are settled and have been removed:
`enableGraphCapture` (**ruled out**, §12 row 6 — it is not a risk because it is
not used), Hiera op coverage on ORT-Web WebGPU (**proven** by the shipped lane),
and the unmeasured encoder arena (**measured** — §9.1, and §6.1 shows `tiny`
would not have helped).

| Risk | Impact | Mitigation |
|---|---|---|
| fp16 conversion mistypes a head | silent quality loss | §7 step 4 is mandatory and gating |
| WebKit exposes no byte-level memory API | the governor cannot measure, only estimate | the allocation ledger is the sole input there and must therefore stay complete — it now counts the text lane's worker, which it did not (§9) |
| OPFS quota under many cached embeddings | eviction churn | LRU cap; embeddings are recoverable by re-encode |
| `GPUDevice.lost` under multi-tab GPU pressure | session dies mid-use | mandatory handler (§8.5): invalidate, rebuild on next use |
| Many tabs at steady state still sum (5 × 270 MB) | aggregate pressure no tab can see | dormancy (§8.5) bounds hidden tabs to ~30 MB; accepted residual risk |
| Tab closed mid-encode | partial OPFS write | Web Lock auto-releases; write embeddings atomically (temp + rename) |

**Deferred, not rejected:** darktable's tiled restoration models
(`upscale-realplksr` 52 MB, `denoise-nind` 52 MB, `rawdenoise-nind` 55 MB). All
declare `tiling: true` with fixed tile inputs, so their peak is bounded
independent of source resolution — architecturally the cheapest lanes available.
They fit at fp32; note darktable sets `fp16: false` explicitly for several, so
validate before converting. Phase 6 is done, so this is now simply open — it is
the next lane if there is one, not a scheduled item.

---

## 17. Appendix

### A. On-disk inventory *(nothing owed)*
The cleanup this section used to demand is done. Current state, for the next
person who wonders whether something on disk is dead:

- `models/` — **204 MB, all shipping**: `sam21/` 88 MB (the mask lane),
  `yoloe/` 53 MB and `clip-text/` 64 MB (the text lane). Grounding DINO and
  OWLv2 are gone from disk and from the manifest.
- `models/manifest.json` describes the **mask-lane** bundle only (ORT + the two
  SAM graphs) and carries `"detector": null`. That is the offline/vendoring
  gate's scope, not a claim that the text lane's models do not exist.
- `spikes/` — 6.7 MB (`sam21/` 1.9 MB, `debug/` 4.6 MB, `yoloe/` 264 KB): the
  graph inspection behind §6.3 and the measurement scripts behind §9.1. Kept
  deliberately; it is scripts and JSON, not weights.
- `scripts/export-sam21.py`, `export-yoloe-text.py`, `export-clip-text.py` are
  the reproducible conversion path for the three shipped models (§7). Keep.

### B. Test images
- `2680558334.nef` — Nikon Z 8, High Efficiency (TicoRAW). **LibRaw cannot
  decode.** Three embedded previews. Canonical test image; exercises the
  preview-only path.
- `d750-lossless.nef` — Nikon D750, 14-bit lossless, 6016×4016. CC0 from
  raw.pixls.us. Develop-eligible; exercises the LibRaw fallback.

### C. Sources
- [darktable-ai](https://github.com/darktable-org/darktable-ai) — models, `model.yaml`, `package.py`
- [`versions.json` @ release-5.6.0](https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/versions.json) — pinned versions + sha256 per model
- [MDN: Web Locks API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Locks_API) — cross-tab mutual exclusion for `I3`
- [webgpu-sam2](https://github.com/lucasgelfond/webgpu-sam2) — SAM2 in-browser precedent
- [SharpAI SAM2 ONNX](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx) — byte-identical exports, `.ort` variants
- [transformers.js #874](https://github.com/xenova/transformers.js/issues/874) — no SAM2 support; raw ORT-Web required
- [ORT Web + WebGPU](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)

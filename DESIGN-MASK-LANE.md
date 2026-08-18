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
| a repeated subject (rivets, grapes) drags its neighbours in | shipped as-is — too large for any size rule | separated, solid components with no click in them are dropped |

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

### Outliers are separated by space, not by size

The size rule cannot reach the other half of the problem. A click on one of a
*repeated* subject — one rivet in a plate of rivets, one grape — comes back with
the neighbours attached, whole: far above 16 cells, and large enough that
dominance drops under 0.85 on the way in, which switches the size rule off
entirely. That is exactly the reported screenshot: one click, one rivet, two
extra blobs.

Those blobs differ from a severed wire by **distance and shape**, so that is
what is tested. A component is removed when all of these hold:

| condition | why |
| --- | --- |
| gap ≥ 15% of the anchor's long side, min 2 cells | Chebyshev gap between bounding boxes; the floor keeps a grid-severed neck out of it |
| `area / long-side² ≥ 0.3` on **both** the anchor and the candidate | a disc scores ~0.8, a pole-with-wires ~0.1, a 2×300 wire 0.007 — so the streetlight turns the rule **off**, and a detached wire is never the thing removed. Bbox fill will not do this: an axis-aligned wire fills its own box perfectly |
| no include click in it | as everywhere else, the user pointed at it |
| an include click exists at all | the click is the anchor. At export scale the prompt is history and a box prompt never named a component, so with no click the rule does not run — otherwise it would delete the second blob of a deliberate two-click selection |

Hole filling needs no gate: 0.000 IoU cost on the streetlight, and on the
reported rose (`scratchpad/rose.mjs`) it adds 513 px, all of them shadowed
crevices *inside* the bloom — background to a colour ground truth, part of the
flower to a human.

### Hygiene has to run where the mask is thresholded

The gate above cleaned SAM's 256² grid, which is not where the mask is decided.
Two stages run after it, and both were measured manufacturing detached
components out of a field that reached them as a single one:

- the **bicubic upsample** (Catmull-Rom, negative lobes) rings beside the
  `-FILL` value hygiene itself writes, and severs a neck that is one cell wide
  at 256² but sub-pixel at proxy scale;
- the **colour guided filter** flips a near-zero plateau wherever a strong
  colour edge runs just outside the boundary — a luma guide on a clean disc
  produced a second component with nothing planted at all.

That is the speckle in the reported screenshots: dots, dashes and one-pixel
rings outside the contour, all of them born after the only pass that could
have removed them. So the same rule runs again at proxy resolution, on the
already-matted field, inside the mask's own bounding box (`upsampleLogits`
returns that window from the pass it was already making). Every caller that
thresholds a field now runs it — including `cropDecode`, which had none at all
because it does not go through the lane's arbitration.

The 16-cell floor is tuned on 256². The same speckle covers the same *fraction*
of a finer field, so the threshold scales by area (171 px on a 1024×683 proxy)
instead of being re-tuned per resolution, with 16 kept as the floor because on a
coarser grid speckle still lands in single cells.

Running per click at proxy resolution is a cost, so the labeller is built for
it. Cells are read once in address order as **runs** (scanline run-length +
union-find), not per-pixel flood fill: no `Int32Array` label plane (2.8 MB at
proxy), no recursion stack, and the only scattered access is a union-find sized
by run count — thousands — rather than by pixels. Dropped components are
rewritten with `TypedArray.fill`, the scratch arrays live at module scope, and
the pass returns the dirty rect so the RGBA repaint stays as small as the edit.

| measured, per click | ms |
| --- | --- |
| `refineField` (guided filter), for scale | 12.8 – 14.5 |
| proxy hygiene, typical subject | **1.04** |
| proxy hygiene, mask spanning the whole 1024×683 frame | 3.3 – 5.0 |

Equivalence with the flood fill it replaces is proven over 108 cases
(side ∈ {16, 64, 256} × blob, blob+speckle, wires, noise, empty, full) — every
one identical. On the planted case, 25/9/4 px speckle goes, a 600 px detached
wire survives, a 36 px pinhole is filled, and a speckle the user clicked is
never removed (`verify.mjs` phase T).

### What real DSLR frames found that planted fields could not

Planted cases prove the rules; they cannot find the places the rules are never
asked. `.leak-probe.mjs` drives real frames through the real pipeline — four
DSLR RAWs plus the tram crop, 27 clicks chosen to be hard (a rivet in a plate
of rivets, one tulip beside its twin, a person occluded by a chair, a black
shirt against a black neighbour, thin cable on dark wood, subjects cropped by
the frame, and correction sequences with negatives) — and reports, per click,
every component that holds no click: its area, its separation from the nearest
clicked one, and its solidity. That is the rules' own vocabulary, so a leak
arrives as a row rather than a screenshot. Two showed up.

**The scan window was smaller than the stage that dirties it.** Hygiene was
handed the mask box. The guided filter works on the *band* box plus a pad of
`radius·2 + scale·2` — 24 px on the proxy path, up to 80 px on the native one —
so it writes OUTSIDE the window that polices it, and a pixel it lifts over zero
out there is a component nothing can ever see. One rivet click on
`d750-lossless.nef` shipped exactly one such pixel, 3 px past the box. Every
call site now scans `box ∪ refineField`'s returned rect, which costs nothing:
both rects were already in hand, and `cropDecode` was discarding the second.

**One honest second part switched the whole size rule off.** Dominance stands
in for "the subject is legitimately fragmented", and a single real second
component fakes it. A click on a seated person, whose leg the chair cuts off,
scored 0.838 — under the 0.85 gate — so 26 specks of 40-80 px shipped with it,
28 components for one person. Lowering the gate is the wrong lever: it is
measured against the streetlight at 0.568, and the distance between them is the
whole safety margin.

What a fragmented subject cannot fake is the *risk of the removal itself*, so
below the gate the rule now caps that instead of guessing: components under a
quarter of `islandCells`, removed only while they add up to under 0.5 % of the
mask, all-or-nothing. The person drops 26 specks worth 0.25 % and keeps the leg
(28 components → 2). The streetlight's speckle is percent-of-mask, an order of
magnitude over the budget, so it is refused whole — which is the behaviour the
gate was built to give.

Three findings in the same table were **not** leaks, and saying so is the point
of measuring: the 13 535 px beside the seated person is her leg, the 9 281 px
under the black shirt is its hem, and the 172 px near the white chair is more
chair. The probe's first cut called all three catastrophic because it took only
the *first* clicked component as the anchor — so a deliberate two-object
selection read as a 4.6 % anchor. Every clicked component is an anchor now, and
a click outside the live mask commits a new object, so the row reports that too
(`+2obj`). A measurement that cannot tell a second object from a leak will
happily prove a leak that is not there.

### 10f. The union of several objects is regularised, not the objects **[MEASURED]**

Per-click hygiene works inside one decode. Selecting six adjacent cubes one
after the other produces a defect neither pass can see, because it exists only
*between* two decodes: each object stops about a cell short of the edge it
shares with its neighbour, so the union keeps a hairline slit along every
internal boundary. The outline is a dilation of the hard core, so a slit two
pixels wide is painted as a **border through the middle of what the user
selected as one thing** — that is the internal seams and the notches along the
floor line in the reported screenshots. Measured on that union: one component,
but 26 enclosed holes worth 461 px, plus open slits; the gap histogram is
201 px of 1-px runs, 238 of 2-px and 141 of 3-px before the tail of genuine
background at ≥16 px.

So `recomposeMask` regularises the *composed* mask on every recompose
(`bridgeGaps` + `smoothBoundary`, js/sam-core.js). It is re-derived, never
baked into an op, so undo and subtract stay exact.

- **`bridgeGaps`** — a morphological closing of the ≥128 core, add-only by
  construction, so no object can be lost to it. Radius is `max(1, long/512)` =
  **half a decoder cell**: a gap that narrow is two decodes disagreeing about
  one cell, not background anyone chose to keep. Measured on the sweep: r=1
  leaves 2 holes, **r=2 leaves 3 (+0.78 % area)**, r=3 leaves 0 but starts
  closing real pockets between objects, r=4 adds nothing further. Outside the
  frame reads as foreground, so a slit that runs off the frame edge closes to
  the edge instead of leaving a notch there.
- **`smoothBoundary`** — the mean of each (2r+1)² neighbourhood. The mask is a
  soft band around the decoder's level set and that band is near-linear across
  a straight edge, and a box mean of a ramp is the same ramp: straight and
  diagonal edges come back unmoved and only pixel-scale wobble averages out.
  That is curvature smoothing of the level set; a hard 0/255 median would
  instead re-quantise the edge it is meant to soften. Radius is
  `max(1, long/1024)` = 1: r=1 costs −2.3 % perimeter at IoU 0.9991, r=2
  −4.2 % at 0.9979, and on a 153-px object r=2 already costs 2.6 % of the area.
  Manual geometry skips it — a drawn rectangle keeps its corners by contract.

The one thing a mean cannot be trusted with is a thin structure: a 1-px wire
averages below the decision level along its whole length. A core cell is
therefore never cleared unless its 8 neighbours form a single arc (crossing
number 1). The guard has to be evaluated against the **running** core, not a
snapshot — the first version used a snapshot, and two adjacent cells that were
each simple on their own severed the streetlight into two components. Union
hole *filling* is deliberately absent: a brush erase leaves an enclosed hole,
and refilling it would undo the user's own action. Bridging at r=2 cannot
reach one, because brush strokes are ≥20 px wide.

End to end on the reported case: holes **26 → 3** (461 → 121 px), perimeter
**2090 → 1397** (−33 %), roughness 1.919 → 1.278, still one component, +0.79 %
area. The three survivors are genuine background pockets.

Both passes are linear and bounded by the selection, not the frame. Chebyshev
dilation is separable and each axis is two sweeps carrying the distance since
the last set cell, so **cost does not depend on the radius**: `dilateChannel`
went from 17.1 / 52.8 / 91.3 ms at r=2/6/12 to **4.2 / 5.8 / 4.9**, identical
output over 40 random fields × 5 radii. The mean uses sliding windows for the
same reason. Working inside the bbox grown by the kernel cuts a full-frame
close from 12.4 to 7.4 ms; the whole regulariser is **8.7 ms** on a 900×675
composed mask.

### 10d. Showing the three, instead of hiding them behind a key

Arbitration picks the best *default*; it cannot pick the right answer, because
on an ambiguous click there is no single right answer to pick (§10a). The
remaining failure is therefore not a math failure — it is that the user has no
way to know a second reading exists. `C` cycled them, and `C` is a key nobody
discovers.

The three candidates are already parked in memory, ordered small→large, and
picking one costs a repaint (upsample + guided filter over a plane that is
already decoded), not a decode. So the cost of *showing* them is a row of pills:
`#scope`, one button per candidate, `aria-pressed` on the active one. Measured
on the demo scene: default 6%, and clicking the smallest pill takes the mask
from 0.0638 to 0.0304 coverage with no decode.

Two details are what make it usable rather than merely present:

- **Hover previews without taking.** `sam21CandidateShape` thresholds the parked
  256² field into an alpha plane — no upsample, no guided filter — and the
  overlay draws it as an amber wash under the live selection. The point is to
  *compare* before committing, and a preview that paid the post pipeline would
  cost more than the mistake it prevents.
- **The bar sits off the mask.** `getMaskLayers` already walks every pixel to
  build the border core, so it records the core's extent in the same pass; the
  control is placed below that extent, not below the click. A bar parked on the
  subject hides the evidence the user opened it to look at.

`C` still works — the control does not replace the shortcut, it makes it
unnecessary.

### 10e. Showing the readings themselves

Showing the three was not enough, and every attempt at *naming* them was worse
than the numbers. `1.9% · 6.3% · 13.2%` names a measurement rather than a
choice. `Part · Object · Whole` — SAM's own hierarchy — names a **meaning the
model does not give**: what comes back is three sizes, and when the biggest
reading is a pole plus a wire plus a patch of sky, calling it "Whole" makes the
control less trustworthy, not more. A `− Less ○ ● ○ More +` row named a
direction but never a destination — the dots counted the readings without
showing one, so the only way to learn what "more" meant was to spend a repaint
and look.

What survives is the choice itself: one swatch per reading, painted from that
candidate's own mask — the lamp head, the lamp with its arm, the pole with the
wire. Same bargain as Photoshop's object finder: show the shape, take it on
click, with the ranking left to the eye because no ranking rule works here
(§10a). Hovering ghosts the shape on the canvas (`sam21CandidateShape`, a 256²
threshold — no upsample, no guided filter), so the answer to "what is this
button?" is the shape itself. Exact percentages stay in the tooltip, the
`aria-label` and the status line, where a number belongs.

Two things about the drawing were found by measuring, and both are the whole
reason it reads:

- **Own crop, not a shared one.** The first cut drew all three inside one union
  crop so the areas were literally comparable. On a real click — readings of
  0.2 / 0.7 / 17 % of the frame — that gives two invisible specks beside one
  filled blob, and the swatch that matters most is the one you are about to
  reject as too small. A ghost layer behind each swatch did not rescue it. Each
  shape now gets its own crop and **size carries only the order**: areas spread
  on a log ladder between the smallest and largest reading of *that* click,
  floored at 42 % of the cell (`fills`). Measured ink per swatch: 0.045 /
  0.174 / 0.626 — monotone and distinct.
- **The field is a square resize, so the silhouette has to be un-squashed.**
  The decoder returns `MASK_SIDE²`, a square resize of the frame, not
  longest-side-plus-pad. Drawn straight, a tram measuring 0.735:1 in mask space
  is 1.26:1 on the canvas — a different shape. `paint()` multiplies the width by
  the displayed **frame's** aspect (`geometry().aspect`, fed from the overlay
  rect — the stage box only matches it while the stage hugs the photo).

At an end nothing greys out, because there is no direction button left to grey:
every swatch is a destination, and the stepping paths simply clamp.

Two gestures step the same thing for people who never look at a bar:

| gesture | why it was free to take |
| --- | --- |
| **drag up and down over the selection** | in click mode a moved tap is already discarded, so a vertical scrub was dead input. 34 px per step, live ghost preview, commits on release. |
| **scroll over the selection** | the canvas has no zoom, so the wheel was unbound. 40 delta per step. |

The scrub starts only on ≥ 14 px of *vertical* travel that beats the horizontal
travel — above both tap slops (5 px pointer, 12 px touch) — so a shaky hand
never steps by accident and no other tool's drag is taken over. One mental
model across the drag, the wheel and `C`: **up is more**.

Two things were tried and cut. A third gesture, *tap the same spot again*, was
free in the same sense — a second prompt on the same pixel is a no-op SAM
already has — but it collides with the one rule every user has already learned,
that **clicks add points**, and it was the only stepper that wrapped. Wrapping
went with it. Every path now steps by one through the same clamped `stepScope`:
buttons, dots, wheel, drag, and the keyboard alike.

Driven through a real browser against `streetlight.jpg` (`.scope-probe.mjs`,
Playwright, real mouse — synthetic events would skip pointer capture and the
tap/scrub arbitration, which is the part worth proving):

```
first click:    {count:3, index:1, shapes:3, drawn:3, pressed:1}
last swatch:    index 1 → 2   "Selected 17.1% of frame (3 of 3)"
wheel down:     2 → 1         ×3 more: 0, and it stays 0 (clamped, not wrapped)
drag up 80px:   0 → 2         (34 px per step)
tap same spot:  clicks 1 → 2  (still a prompt, never a step)
ink per swatch: 0.045 / 0.174 / 0.626        pill 110×36, inside the stage
```

The whole control is `js/scope-control.js`: a mount element, a surface to
listen on, and callbacks. No app state, no imports, no framework — Mask Studio
and Phosmith take the file as-is, and `verify.mjs` asserts that it stays that
way.

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

### 12a. Every per-pixel pass is bounded by what it can touch **[MEASURED]**

The lane's own cost was already band-limited (§10). What was not is the half that
runs *after* it: on a click the app composed the op stack, unioned it with the
live object, summarised the result twice, re-derived the tinted and ringed
layers, and uploaded three canvases — each of them walking the whole proxy frame
regardless of how much of it the selection occupied. A selection is typically a
few percent of the frame, so nearly all of that work was reading zeros.

Each of those passes is now bounded by a rect that something already knew:

- The **lane** returns `maskRect` — the union of the mask box and the rect the
  guided filter and region hygiene rewrote. Every non-zero alpha is inside it by
  construction, so the client's summary scans it instead of the frame.
- Every **committed op** carries the bounding box of its own channel, measured
  once when it is pushed. Replaying the stack costs the sum of the objects' areas
  instead of (stack depth × frame), and the composed mask hands its union rect
  back as the scan window for everything derived from it.
- **cv-refine** copies and re-expands only the mask's soft extent, padded by the
  morphology radius that could grow it (zero today, so the pad is zero).
- **`getMaskLayers`** skips zero pixels outright, tracks the tinted layer's own
  extent alongside the core's, and uploads and shifts only those rects — eight
  ring draws of a 200 px object on a 4000 px frame is a hundredth of the fill
  rate it used to be.

Every bound is optional and a missing one falls back to the whole frame, so an op
stack restored from an older session still composes correctly. The rect passed to
`summarizeMaskRGBA` is a promise, not a crop: the caller asserts nothing selected
lies outside it, and what comes back is still the whole mask's coverage and bbox.
Where a mask is composited rather than measured the rect comes from the soft
extent (every non-zero pixel), never the ≥128 core, so the matted edge is never
clipped.

Measured, six clicks on the canonical NEF at its 1536×1024 proxy, same build with
the bounds handed in versus forced to null: compose **30.5 → 19.8 ms** per click
(−35%), wall **277 → 245 ms** (−12%). In isolation the gap widens with the stack:
at 1600×1067 a twelve-op compose is **40.9 → 1.8 ms**, and its summary
**4.4 → 2.1 ms**; at 4000×3000 (the escalated/export grid) it is **306 → 13 ms**.
Outputs are byte-identical either way — the coexist probe's coverages and the
region grids are unchanged.

**What was measured out.** The same idea applied to `upsampleLogits` — bound the
bicubic upsample to the rect that can produce a visible band, derived from the
2D Catmull-Rom negative-tap bound — is worthless on real data and was reverted.
Real SAM 2.1 logit floors are only −16 to −24, not the −30 a synthetic field
suggests, which puts the safe threshold at about −8 to −10; and on a real photo
12–23% of the 256² cells already sit above −6, scattered across the frame as
distractor objects. The work rect came out at 100% of the frame on every click
measured (four on cubes.webp, four on the NEF), so the pre-scan was pure cost.
The upsample is bounded by the *field*, and the field is not sparse.

### 12b. The debounce fires on the leading edge **[MEASURED]**

With the per-pixel passes bounded, the largest single item left in a click was
not computation at all: `scheduleRun` held every prompt for a trailing 80 ms
before starting the run. A trailing debounce exists to absorb a stream — but all
five call sites are one discrete gesture (a tap, an exclude click, a box or lasso
*pointerup*, an undo, a re-run the queue asked for). None of them streams, and
`runNow` already coalesces anything arriving mid-run through `state.runQueued`.
So the window was doing nothing for an isolated click except delaying it.

It now runs on the leading edge with a trailing coalesce: a prompt more than
80 ms after the last one starts immediately, and one inside that window still
collapses onto the trailing timer. `debounceArmed` still marks a pending trailing
run, which is what the escalation gate and the headless `waitForRun` read.

Measured, six clicks on the canonical NEF, same build and the same machine, with
the leading edge on versus off: app time (wall minus the lane's own total)
**133.8 → 53.0 ms** — the 80 ms, exactly — and wall **266.3 → 194.3 ms** (−27%).
The cost is one extra decode when a genuine double-click lands outside the
window, which is the already-existing `runQueued` path and not a new one.

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

`verify.mjs` now carries 227 `check()` calls across 3574 lines (231 assertions
in a full run), including both RAW fixture phases. `--fast` runs the three that
need no browser (pure logic, heavy-job queue, static source scans — 134
assertions) and stops before the browser phases; those need Playwright and the
dev server on :8788. The profile-related assertions phase 0 was meant to rewrite
are gone with the presets — what is left refers to the single config.

The browser phases drive the **installed Google Chrome**, not Playwright's
bundled Chrome-for-Testing: `scripts/harness/browser.mjs` resolves the channel
for verify and both probes, `--cft` opts back out, and the profile path carries
the channel because stable Chrome refuses a profile Chrome-for-Testing wrote.

Phase **G** is the only one that drives a real pointer. Every other phase clicks
through `clickAt`, which takes canvas coordinates and so never exercises the
pointer→canvas mapping — and that mapping is measured against a `#frame` the
zoom transforms, where a stale rect selects a different object without ever
throwing. G proves the mapping under zoom and pan, the pan clamp, the 1×–8×
range, the keys, and a CDP pinch (touch emulation on the same page, so it costs
no second browser). `.zoom-probe.mjs` is its exploratory sibling and adds a real
touch context plus the wheel-vs-scope arbitration.

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

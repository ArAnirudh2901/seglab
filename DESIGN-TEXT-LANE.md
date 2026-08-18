# Open-vocabulary text search

Companion to `DESIGN-MASK-LANE.md`. Built 2026-08-07.

## 1. What was wrong

Text search claimed to be open-vocabulary and was not. `js/clip-text.js` looked
phrases up in a 4585-word table precomputed offline and returned `null` for
anything else, which short-circuited the detector. "Muscari", "a rusty bicycle
leaning on a blue wall", any compound description — all silently found nothing.

Behind it sat two detectors for one job: YOLOE prompt-free (a vocabulary baked
into an LRPC head) with YOLO-World-v2 as a fallback. That is the duplication that
got Grounding DINO and OWLv2 deleted, reintroduced.

## 2. Shape

One lane. The phrase conditions the detector directly, so there is no vocabulary
to miss and nothing to fall back to.

```
phrase ─► CLIP BPE ─► MobileCLIP2-B text tower ─► k×512 vectors
                          (4-bit, transient)          │
                                                      ▼
   full frame + 2×2 tiles ────────► YOLOE-26L text-prompt ─► boxes per cell
   (one bounded decode, 640² each)   (RepRTA in-graph, nc=k)      │
                                                                  ▼
                                        merge to original px + NMS
                                                                  │
                                             SAM 2.1 box prompt ─► mask
```

All cells go to the detector in ONE worker message so they share a single
session. Exactly `k` class slots are fed — one per phrase, never padded (§6).

| artifact | size | built by |
|---|---|---|
| `models/yoloe/yoloe-26l-text.fp16.onnx` | 55.4 MB | `scripts/export-yoloe-text.py` |
| `models/clip-text/mclip2-text.q4.onnx` | 40.9 MB | `scripts/export-clip-text.py` |
| `models/clip-text/mclip2-embed.i8` + `.scale.f32` | 25.5 MB | ″ |
| `models/clip-text/merges.txt` | 0.5 MB | ″ |

RepRTA (YOLOE's 1.58M-parameter text adapter) stays **inside** the detector
graph: `get_tpe(t) = normalize(reprta(t))`. The runtime feeds raw MobileCLIP2
vectors, so the adapter can never drift out of sync with the detector it feeds.

## 3. Memory is the design

The app already measures 2281–2358 MB with a SAM encoder session live
(`seglab-encoder-session-is-the-ceiling`), so a text lane that lingers is not
affordable. Three rules, in order of how much they buy:

1. **Phrase vectors persist** (`js/text-embed-store.js`, OPFS). An all-hit query
   never builds the text encoder at all — checked on the main thread before the
   worker is even spawned.
2. **The encoder is load → encode → release.** Inside `encodePhrases` the token
   table is fetched, gathered from, and dropped *before* the ONNX session is
   built, and the session is released the moment the vectors are out. Peak is
   `max(25, 41)` MB, never the sum. The detector session is built only after all
   of that is gone.
3. **Eviction runs both ways.** `detectText` drops the SAM embedding before a
   search; `segment`/`encodeImage` terminate the detect worker before SAM
   allocates. It must fire *inside* the queued task — `enqueueHeavy` serialises
   execution, not residency, so it gives `max()` on compute and `sum()` on
   memory, and disposing outside the task kills a detect still in flight.

Measured (M2/8 GB, real Chrome 151, canonical NEF, `scripts/mem-probe.sh`):

| state | all Chrome |
|---|---|
| NEF imported, no encoder | 1383 MB |
| text search (product path) | **1607 MB** |
| SAM encoder live | 2281–2358 MB |
| product search on a live encoder | **1607 MB** (GPU proc 1216 → 301) |
| same search, eviction bypassed | 2558 MB |

Those are point probes. The ceiling question needs a *peak*, so
`scripts/mem-peak.sh` samples the whole process tree continuously across the
full first-visit scenario — boot → NEF import → SAM click → two phrases:

| scenario | peak, three runs |
|---|---|
| cold profile (nothing cached, models downloaded) | 1953 / 2018 / 1896 MB |
| warm revisit (HTTP + OPFS warm, session restored) | 1580 / 1545 / 1563 MB |

Both clear the 2.2 GB bar, and the warm case is the *cheaper* one — the earlier
2917–3610 MB cold readings were two separate artifacts, not one problem. Half
were the `lazyRelease` arm, which this lane must not use (see `js/ort-loader.js`);
the rest were baseline contamination — a previous run's Chrome still tearing down
matches the same `--user-data-dir` and lands in the probe as phantom memory, up
to ~700 MB of it. Any harness that relaunches on one profile has to wait for
`pgrep -f user-data-dir=…` to reach zero before it probes.

**Both text models run on WebGPU, not WASM.** Staged per-PID on a cold profile
with no image loaded, `text-encode.js` and `yoloe-detect.js` each report
`backend: 'webgpu'` — so there is no large wasm arena in this lane and nothing
here that a wasm-side lever would reach. Staged deltas, page renderer / GPU:

| stage | renderer | gpu | all Chrome |
|---|---|---|---|
| boot, no image | 150 | 62 | 326 MB |
| CLIP text encoded, session released | 472 | 106 | 791 MB |
| YOLOE session built, not run | 581 | 168 | 962 MB |
| YOLOE first run | 532 | 293 | 1034 MB |
| YOLOE disposed | 535 | 115 | 854 MB |

Disposal returns the GPU side (293 → 115 MB) and almost nothing on the renderer
side; the renderer only comes back when the worker terminates, which is why
`detect-worker-terminated` after every phrase is load-bearing.

**This table is also the ledger entry.** Because the lane lives in its own
worker, none of it reached the memory governor's allocation ledger — and on
WebKit that ledger is the governor's *only* input, so the app's heaviest lane
was invisible on the one engine that cannot measure. `detectorResidentMB`
(`js/sam-client.js`) charges 700 MB, 1030 MB on WebKit, for as long as a detect
worker is up — keyed on the worker, not the session, for the reason in the
paragraph above. See DESIGN-MASK-LANE §9.2 for the shed side of it.

## 4. Quantization

One scheme per op family, chosen by kernel support — not per device, and not per
model.

- **Text tower → 4-bit block-wise `MatMulNBits`**, block 16, asymmetric,
  `accuracy_level` 0. Measured min cosine **0.991** against the fp32 tower.
  Sweep: block 64 sym 0.981 / 33 MB · block 32 sym 0.982 / 35 MB · block 16 sym
  0.986 / 40 MB · **block 16 asym 0.991 / 41 MB**.
- **Token table → int8 per row.** It is a `Gather`, not a `MatMul`, so
  `MatMulNBits` cannot touch it. Ablated: the table contributes 0.9998 cosine on
  its own; all the loss is in the tower. Lifting it out of the graph also keeps
  25 MB out of the ORT arena.
- **Detector → blanket fp16**, max score delta 0.00097. No mixed precision
  needed: the `-pf` export's documented fp16 blocker was its LRPC head, and this
  variant has none.

`accuracy_level: 4` requests int8 *arithmetic* — the activation quantization that
collapsed CLIP alignment 0.9 → 0.03 in July. Never use it here.

## 5. Traps

- **`text_model` is `mobileclip2:b`, not `blt`.** Read the attribute off the
  checkpoint. A different CLIP feeds RepRTA out-of-distribution vectors.
- **The shipped tower is frozen TorchScript and will not export** ("transpose for
  tensor of unknown rank"). Weights come from `timm/MobileCLIP2-B-OpenCLIP`
  instead, and the export *asserts* cosine ≥ 0.9999 against the shipped blob
  (measured 1.000000) before proceeding. Never swap weights without that gate.
- **CLIP's first 512 vocabulary ids follow `bs` insertion order**, not byte order
  0–255. Byte order gives the same set with every id shifted — silently
  mistokenizing punctuation and non-ASCII. `verify.mjs` pins it with
  known-answer tests against the reference BPE.
- **SAM 2.1 does support box prompts.** `js/sam21-adapter.js` used to collapse a
  box to a centre click, throwing away the extent the detector had just
  localised — which degraded *every* text-driven selection, since text search
  selects by box. The decoder carries `prompt_encoder.point_embeddings.2/.3` and
  branches on labels 2/3 (verified against `models/sam21/decoder.fp16.onnx`).

## 6. Two things that were wrong, and the measurements that caught them

Both were found by instrumenting output rather than reading code, and both were
invisible in a passing test suite.

**Duplicate detections (fixed).** The class axis was static at 32 with short
phrase lists padded by repetition. Padding looks free — duplicate slots score
identically, so a max-over-classes is unaffected — but the head emits each anchor
once *per class* into a fixed top-300. A one-phrase query therefore produced
**10 unique boxes out of 300 rows**, capping recall at ~10 instances whatever was
in frame. The axis is now dynamic and exactly one slot is fed per phrase:

| query | unique boxes before | after |
|---|---|---|
| `"muscari"` | 10 / 300 | **197 / 197** |
| `"orange tulip"` | 14 / 300 | **371 / 481** |

**Resolution floor (fixed).** A 45 MP frame squeezed into 640² is a 12.9×
shrink, so a 200 px subject lands under 16 px. That is why muscari scored 0.09
while a tulip in the *same* frame scored 0.60 — a resolution problem wearing the
costume of a vocabulary problem. `tilePlans` now adds a 2×2 overlapping grid
(§2) whenever the shrink would exceed 2.5×, giving 7.4× instead of 12.9×:

| query | top score before | after |
|---|---|---|
| `"muscari"` | 0.091 | **0.171** |
| `"orange tulip"` | 0.596 | **0.702** |

Cost is 5 inferences per search instead of 1 — all on one session, in one worker
message, because a worker respawn per tile would dwarf the saving. Measured
1.6–3.3 s per search, and only on images large enough to need it.

### 6.1 The setting is not the subject — both halves of it

The detector scores a phrase as a **bag of words**, so a phrase whose *setting*
is in the photo matches whether or not its subject is. Measured on the canonical
NEF: `"the dog sitting among the flowers"` returned 5 boxes and `"snow covering
the flowers"` 6 — every one a flower, in an image with no dog and no snow.

`normalizePhrase` splits a post-modifier off the subject (`headCore`), and the
subject gets its own detector slots plus its taxonomy expansion. That yields two
separate rules, and shipping only the first left the bug half-fixed:

1. **No subject hit anywhere → return nothing.** The compound match is the
   setting bleeding through, and the honest answer is "not here".
2. **A subject hit → select the subject *only*.** This is the half that was
   missing. In a photo that does contain a dog, the flower boxes still scored,
   still survived NMS, and still arrived as candidates — so the user asked for a
   dog and got a dog plus five flowers. The setting exists to condition the
   score. It was never something the user asked to select.

Both live in `filterToSubject` (`js/text-core.js`), which is pure and pinned by
`verify.mjs` in both directions, including the identity case: a phrase with no
post-modifier (`"orange tulip"`, `"a cluster of tightly packed blue florets"`)
has no setting to separate and passes through untouched. A gate that fired on
ordinary phrases would reject every real search.

**Escalation asks about the subject too.** The 2×2 → 3×3 retry fires when
nothing clears the rank threshold. Judged over *all* labels, a well-scoring
setting satisfied that bar and skipped the finer pass — so a small subject in a
rich setting was answered "nothing here" without ever looking harder, which is
exactly the case §6's resolution floor exists for. The check is now
subject-scoped.

## 7. The detector proxy, and its cap

The lane has its own proxy, separate from the interaction frame: `buildFrames`
re-decodes the ORIGINAL so each tile reaches the detector square at real
resolution. That decode used to be sized from tile geometry alone — no budget,
no ceiling, no response to pressure. The one lane that scales with image size
was the one lane nothing capped.

`detectorPlan` (`js/proxy-plan.js`, beside every other sizing decision) now owns
it, with **one knob**: `detectorMaxCells`, the number of 640² inferences a search
may run. Cells are what cost memory, so the grid comes from that budget and the
source resolution is DERIVED from the grid it bought — two independent numbers
could disagree and starve an axis. `detectorMaxSide`/`detectorMaxMP` are
guardrails behind it, not the working limit.

That knob has **two** ceilings, because a cell costs two things. Memory stays
with the class signals in `policy.js` (mobile, GPU tier, integrated GPU,
reported RAM) — an ORT arena only grows and cannot be timed. **Latency** is
measured: `yoloe-detect` reports `inferMs` per cell (from after the session
exists, so a cold build is not charged to the device), `text-ui` accumulates it
across both passes of a search, and `hardware-fit` divides by cells and clamps
`detectorMaxCells` against `detectorBudgetMs` (2500 ms). On the measured
~140 ms/cell reference that reproduces today's 10 cells; ~420 ms/cell demotes
to the 2×2 pass, past ~900 to full frame only. It only ever lowers the cap.
`?detect=0` turns the latency clamp off.

The derived size lands exactly on the geometry it is sizing for, so on a capable
device the cap never binds:

| grid | cells | derived source | measured `need` |
|---|---|---|---|
| 1 (full frame only) | 1 | 640 | 640 |
| 2 | 5 | 1114 | 1113 |
| 3 | 10 | 1670 | 1669 |

Corner tiles carry padding on one side only, so they are the smallest cell and
they set the number: `side × grid / (1 + overlap)`. Below it a tile is *upscaled*
into its square, which is the resolution floor tiling exists to lift (§6).

Demotion uses signals that cannot be spoofed upward. `memoryGB` 0 means the
browser would not even guess — WebKit and Gecko ship no `navigator.deviceMemory`,
and that is the same engine where the governor has no byte API to read
(`webkit-has-no-byte-api-governor-is-blind`), so nothing downstream can catch
this lane climbing either.

| device | cells | grid |
|---|---|---|
| Chrome desktop, ≥8 GB reported | 10 | 3 |
| Safari / Firefox (no `deviceMemory`), mobile, ≤4 GB, `gpuTier: basic` | 5 | 2 |
| ≤2 GB, or no WebGPU (WASM detector, arena only grows) | 1 | full frame |
| pressure 1–2 / pressure 3 | 5 / 1 | 2 / full frame |

Measured on a 7680×4320 PNG (33 MP), real Chrome, production path:

| stage | before | after (10 cells) | after (5 cells) |
|---|---|---|---|
| import | 1342 MB | 1330 MB | 1341 MB |
| `"car"` peak | 1951 MB | 1939 MB | 1956 MB |
| absent phrase (escalates) | 2902 ms | 2878 ms | **1233 ms** |

Two things that measurement contradicted, both worth keeping written down:

- **Cells do not drive PEAK memory in the production path.** They drive work.
  With the worker held across the idle window the session is built once and
  ORT's pool saturates, so the 3×3 pass reuses the 2×2 pass's buffers. Cell
  count only showed up as memory under `detectorDispose: 'now'` (+872 MB for
  grid 3 over grid 2), where every pass rebuilds. The peak is the first search's
  fixed cost — model parse, session build, CLIP encoder — not the proxy.
- **`detectorDispose: 'now'` is the wrong companion to a demoted rung.** It does
  drop the settled floor (1894 → 1407 MB), and terminating the worker is the only
  true free of the ORT arena. But it makes every later search rebuild the YOLOE
  session, which raised the peak (1956 → 2069, then 2284 MB) and took 2.9 s →
  6.2 s. The failure being defended against is an OOM kill, and a kill is decided
  by the peak.

`detectCandidates` holds the detect worker across the two passes of one search
(`keepAlive` → `disposeDetector`). Under `dispose: 'now'` an escalated search
otherwise paid a full session build between its own halves.

## 8. Cache generation

`/models/` is served `immutable, max-age=31536000` *and* held by the service
worker, so **a model file cannot be updated in place**. Changing
`yoloe-26l-text.fp16.onnx` from a static 32-class input to a dynamic one under
the same URL made every search fail with `Got: 1 Expected: 32` in a browser that
already had the old copy — the app was correct and the bytes were stale.

Two things are required together, and the second is easy to miss: bump
`CACHE_NAME` in `sw.js` (drops the old generation) **and** refill with
`fetch(request, { cache: 'reload' })`, or the refetch is answered from the
immutable HTTP cache with the very copy the bump exists to discard.

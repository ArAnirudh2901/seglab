/**
 * sam21-lane — SAM 2.1 (darktable-ai release-5.6.0) encode/decode, raw ORT-Web.
 *
 * Split lifecycle: the encoder is the only expensive thing here, so it is built
 * on demand, run once per image, and RELEASED — peak collapses back to steady
 * state while the 8 MB embedding survives. The decoder (9.9 MB) stays warm; it
 * runs against a 64×64 embedding and is overhead-bound, not compute-bound.
 *
 * Embeddings stay in GPU buffers this module owns, not ORT's: ORT allocates
 * outputs from the session arena, so they die with the encoder. A GPU→GPU copy
 * of 8 MB is what buys encoder disposal AND zero-copy decode at the same time.
 *
 * Pure lane — no worker plumbing, no cross-tab policy. sam21-host.js owns the
 * single-instance guarantee; this file is agnostic about where it runs.
 */

import { loadOrt as loadOrtShared, webgpuEP } from './ort-loader.js'
import { chooseCandidate, cleanRegions } from './mask-select.js'

const DIR = new URL('../models/sam21/', import.meta.url).href

export const SIDE = 1024        // encoder input edge (model native)
export const MASK_SIDE = 256

// Encoder outputs, in the order the frozen graph declares them.
const EMBED = [
    { name: 'high_res_feats_0', dims: [1, 32, 256, 256] },
    { name: 'high_res_feats_1', dims: [1, 64, 128, 128] },
    { name: 'image_embed', dims: [1, 256, 64, 64] },
]
const count = (dims) => dims.reduce((a, b) => a * b, 1)
const EMBED_ELEMS = EMBED.reduce((n, e) => n + count(e.dims), 0)

const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]

/** fp32 → fp16 bits. Float16Array where available (Chrome 135+), else manual. */
const HAS_F16 = typeof Float16Array !== 'undefined'
const f32to16 = (src) => {
    if (HAS_F16) return new Uint16Array(new Float16Array(src).buffer)
    const out = new Uint16Array(src.length)
    const buf = new DataView(new ArrayBuffer(4))
    for (let i = 0; i < src.length; i += 1) {
        buf.setFloat32(0, src[i])
        const x = buf.getUint32(0)
        const sign = (x >>> 16) & 0x8000
        let exp = ((x >>> 23) & 0xff) - 112
        const man = x & 0x7fffff
        if (exp <= 0) { out[i] = sign; continue }
        if (exp >= 0x1f) { out[i] = sign | 0x7c00; continue }
        out[i] = sign | (exp << 10) | (man >>> 13)
    }
    return out
}
const f16to32 = (src) => {
    if (HAS_F16) return Float32Array.from(new Float16Array(src.buffer, src.byteOffset, src.length))
    const out = new Float32Array(src.length)
    for (let i = 0; i < src.length; i += 1) {
        const h = src[i]
        const sign = (h & 0x8000) ? -1 : 1
        const exp = (h >>> 10) & 0x1f
        const man = h & 0x3ff
        out[i] = exp === 0 ? sign * 2 ** -14 * (man / 1024)
            : exp === 0x1f ? (man ? NaN : sign * Infinity)
                : sign * 2 ** (exp - 15) * (1 + man / 1024)
    }
    return out
}

// Weight files are served `immutable, max-age=1y`, so a re-export is invisible
// until the URL changes. model.json carries the darktable release + sha256, so
// pin the weights to it — provenance and cache-busting in one. The pointer file
// itself is fetched uncached; it is ~800 bytes.
let versionPromise = null
const modelURL = (file) => {
    versionPromise ??= fetch(new URL('model.json', DIR), { cache: 'no-cache' })
        .then((r) => (r.ok ? r.json() : null))
        .then((m) => (m ? `${m.version}-${String(m.sha256).slice(0, 12)}` : 'dev'))
        .catch(() => 'dev')
    return versionPromise.then((v) => `${DIR}${file}?v=${v}`)
}

// WebGPU only here; there is no wasm fallback (§4), so the threaded build never
// runs and its pool would be pure overhead. Wrapped rather than imported bare so
// no call site below can pick up the default thread policy by accident.
const loadOrt = () => loadOrtShared({ threads: 1 })

/* ─── The lane, as constants ──────────────────────────────────────────────────
 * These were a mutable `config` object with a `configure()` patch entry point.
 * Nothing in the product ever wrote to it, but its existence meant the app had
 * no single answer to "which model, at what precision" — the host could be
 * re-pointed mid-session, and the OPFS cache key (which encodes precision) was
 * read once and memoised, so a flip silently mixed fp16 and fp32 embeddings
 * under one key. Freezing them is what makes the pipeline single-valued.
 */

// fp16, and there is no fp32 path. ORT-Web 1.22 (the previously pinned build)
// computes the WebGPU fp16 path WRONG for this model — the embedding error
// compounds through Hiera and the mask covers 99.6% of the frame against a
// correct 29%. That was a RUNTIME bug, fixed upstream: measured across versions
// (spikes/sam21/ortver.html), posFrac vs the 0.2915 fp32 reference —
//   1.22.0  0.9962  ✗      1.24.3  0.2910  ✓
//   1.26.0  0.2910  ✓      1.27.0  0.2910  ✓
// On 1.27 fp16 is correct and halves the encoder (155 -> 78 MB) and the
// embedding (16 -> 8 MB), which is what brings resident GPU memory under the
// 2 GB ceiling. There is no fp32 fallback because there is no device to fall
// back FOR: checkDevice() refuses an adapter without `shader-f16` up front, so
// the precondition is checked once and stated, never discovered mid-encode.
export const PRECISION = 'fp16'
export const LANE = 'sam2.1-small'

// Plain .onnx, MEASURED against both .ort styles (spikes/sam21/bench.html):
// .ort fails outright here — "[Transpose] blocks.0/attn/Transpose_1 … no GPU
// data for input" — for Runtime and Fixed style alike. The .onnx runs.
// The decoder is 19.6 MB for every variant, so the embedding contract is
// identical and swapping small→tiny stays a one-line edit with no migration.
const ENCODER_FILE = 'encoder.fp16.onnx'
const DECODER_FILE = 'decoder.fp16.onnx'

// 'all', but NOT for the reason recorded here before. The old ladder
// (disabled 138 · basic 100 · extended 62 · all 37 ms) timed `run()` on a
// gpu-buffer-output session, which resolves when the commands are QUEUED —
// it measured submission, not the encode. Re-measured with a fence
// (FINDINGS §8): at fp16 the level does not move GPU time at all, 1095–1202
// ms across all four with overlapping ranges. Keep 'all' for ORT's
// memory-reuse planner and the lower dispatch cost, not for a speed win.
const GRAPH_OPT = 'all'

// Grace before an idle lane is released. Rebuilding costs seconds of shader
// compilation (measured: 3 back-to-back encodes went 3.3s → 34s when released
// between each), so a hold is worth far more than the window of elevated
// memory it opens. Queued work extends it; see keepEncoder.
//
// 5 s, not the 1.5 s this was, because what happens at the end of it changed —
// see releaseIdle. Dropping the device is a much bigger reclaim and a much
// bigger rebuild, so the window has to outlast an ordinary pause between clicks
// rather than just an animation frame.
//
// ENGINE-AWARE, because the thing this window trades against costs wildly
// different amounts per engine. Rebuilding means recompiling the encoder's whole
// pipeline set, and Metal's `createShaderModule` is the documented hot spot on
// that path — measured session creation for this encoder on Safari 26.6 ranged
// 0.9 s to 19.9 s, against ~1.3 s on Chrome. With a 5 s window, an ordinary
// pause (look at the photo, then import the next one) made Safari pay it again:
// measured 11.0 s for the second image's encode, of which the run itself was
// under 1 s. Chromium keeps the tight window — its rebuild is cheap and the
// ~976 MB the release returns is worth more than the milliseconds. WebKit holds
// until just under the host's dedicated-mode idle exit (30 s), so the compile is
// paid once per working session instead of once per pause.
//
// This does NOT weaken the memory contract: the pressure governor still sheds
// the encoder at L1 and everything at L2 regardless of this timer, and the host
// still exits on its own idle rung.
const ENCODER_IDLE_MS = (typeof navigator !== 'undefined' && !navigator.userAgentData) ? 20_000 : 5000

let onDeviceLost = null
/** Host hook: notify tabs so they can re-encode rather than show a failure. */
export const setDeviceLostHandler = (fn) => { onDeviceLost = fn }

/** One resident embedding per connected tab (see state.embedCap). */
export const setEmbedCap = (n) => { state.embedCap = Math.max(1, n | 0); trimEmbeds() }
/** Read-only, for status/debug surfaces. There is no setter. */
export const laneConfig = () => ({
    precision: PRECISION, lane: LANE, encoder: ENCODER_FILE, decoder: DECODER_FILE,
})
// One precision, so these are values rather than switches. Every fp16 detail
// is still named here because the byte count is load-bearing: it sizes the GPU
// buffers, the OPFS records and the eviction accounting.
const ORT_TYPE = 'float16'
const BYTES_PER = 2
const EMBED_BYTES = EMBED_ELEMS * BYTES_PER
/** Float32Array → the tensor payload ORT wants. */
const pack = f32to16
/** ORT tensor payload → Float32Array. */
const unpack = f16to32

const state = {
    ort: null,
    device: null,
    encoder: null,
    encoderPromise: null,
    decoder: null,
    decoderPromise: null,
    // key → { buffers: GPUBuffer[], bytes, usedAt }. MUST be a cache, not one
    // slot: one instance serves every tab, so a single slot would let tab B's
    // encode silently steal tab A's embedding and A would decode the wrong
    // image. 8 MB each, so a handful is free — and a second tab opening the
    // same photo hits it and skips the encode entirely.
    embeds: new Map(),
    // draftCacheMax is 1 (one resident embedding). With ONE instance serving
    // every tab that has to mean one PER TAB, or a second tab's encode evicts
    // the first tab's image out from under it. The host sets this to the live
    // client count; 16 MB each makes the difference rounding error.
    embedCap: 1,
    rebuilds: 0,        // device recoveries; a SharedWorker's console is invisible to the page
    encoderRefs: 0,     // drain-release: batch imports skip shader recompiles
    // Decodes need one too now that the idle release destroys the device: it
    // frees the embedding's GPUBuffers, and a decode holding them in a submitted
    // pass dies with "Buffer used in submit while destroyed".
    decodeRefs: 0,
    idleTimer: null,
    lost: false,
}

/** Cap the embedding cache, evicting least-recently-used. */
const trimEmbeds = () => {
    while (state.embeds.size > state.embedCap) {
        let oldest = null
        for (const [k, v] of state.embeds) if (!oldest || v.usedAt < oldest[1].usedAt) oldest = [k, v]
        if (!oldest) break
        for (const b of oldest[1].buffers) { try { b.destroy() } catch { /* gone */ } }
        state.embeds.delete(oldest[0])
        forgetPick(oldest[0])
    }
}

/** Hard gate (§4): WebGPU + shader-f16 or nothing. A silent WASM fallback is a
 *  second, hidden product that OOMs — measured ~3 GB vs ~0.5 GB. */
export const checkDevice = async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
        return { ok: false, reason: 'WebGPU unavailable' }
    }
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return { ok: false, reason: 'no WebGPU adapter' }
    if (!adapter.features.has('shader-f16')) return { ok: false, reason: 'no shader-f16' }
    return { ok: true, vendor: adapter.info?.vendor || null, arch: adapter.info?.architecture || null }
}

// `outputs` is a SESSION option in ORT-Web, not a per-run one — omit it and
// results land on the CPU. Only the encoder wants them left on device.
// WebGPU is not a preference here: checkDevice() has already refused anything
// that cannot run it, so there is no wasm EP to select.
const sessionOpts = (outputs = null) => ({
    executionProviders: [webgpuEP()],
    graphOptimizationLevel: GRAPH_OPT,
    ...(outputs ? { preferredOutputLocation: outputs } : {}),
})

const captureDevice = () => {
    const dev = state.ort?.env?.webgpu?.device || null
    if (dev && dev !== state.device) {
        state.device = dev
        // Several WebGPU devices across tabs makes a drop likely, not theoretical.
        // A lost device leaves GPUBuffers that still LOOK valid and fault on
        // use, so drop every handle rather than trying to salvage any of it.
        // Sessions and embeddings are both rebuildable; a stale handle is not.
        dev.lost?.then((info) => {
            state.lost = true
            state.encoder = null
            state.encoderPromise = null
            state.decoder = null
            state.decoderPromise = null
            state.embeds.clear()          // buffers belong to the dead device
            forgetPick(null)
            state.device = null
            state.ort = null
            // NB: there was an `ortPromise = null` here, aimed at "ORT caches the
            // device on its module". It named a variable that lives in
            // ort-loader.js and is not in scope — an assignment to an undeclared
            // identifier, which in a module (always strict) throws ReferenceError
            // and killed the rest of THIS handler. So `onDeviceLost` never fired
            // and tabs were never told to re-encode: the one job the handler has.
            // It would not have worked either — that promise memoises the module
            // IMPORT, and the ES module cache hands back the same object with the
            // same cached device. `ensureDevice` is what actually re-arms ORT.
            // reason 'destroyed' means WE called destroy() (deep-idle teardown,
            // governor top rung). That is not a fault and needs no broadcast —
            // shutdown already told the tabs.
            if (info?.reason === 'destroyed') return
            console.warn('[sam21] GPUDevice lost:', info?.reason, info?.message)
            try { onDeviceLost?.(info) } catch { /* listener bug */ }
        })
    }
    return state.device
}

/**
 * Recover from a lost device. ORT caches its GPUDevice on its module object and
 * the ES module cache survives re-import, so once that device dies ORT will not
 * build another — every later encode fails "no WebGPU device after encode" and
 * the lane is dead until the worker exits (up to the 120 s deep-idle rung).
 * Handing it a fresh device is the supported seam: `env.webgpu.device` is
 * writable and ORT adopts it. Only runs after a loss, so the normal first-build
 * path is untouched.
 */
const ensureDevice = async (ort) => {
    if (!state.lost) return
    const adapter = await navigator.gpu?.requestAdapter()
    if (!adapter) throw new Error('sam21: no WebGPU adapter for recovery')
    // Carry the adapter's own limits over. ORT raises buffer/binding limits well
    // above the defaults for a model this size; a default device would build and
    // then fail on allocation.
    const requiredLimits = {}
    for (const k in adapter.limits) {
        const v = adapter.limits[k]
        if (typeof v === 'number') requiredLimits[k] = v
    }
    // GPUSupportedLimits exposes its values as prototype getters, so a for-in
    // that yields nothing would silently hand back a DEFAULT device — which
    // builds fine and then fails on the encoder's allocations. Name the ones
    // that decide that outcome explicitly.
    for (const k of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize']) {
        if (typeof adapter.limits[k] === 'number') requiredLimits[k] = adapter.limits[k]
    }
    const dev = await adapter.requestDevice({
        requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
        requiredLimits,
    })
    ort.env.webgpu.device = dev
    state.lost = false
    state.device = null   // captureDevice re-arms the lost handler on the new one
    state.rebuilds += 1
    console.warn('[sam21] rebuilt the GPU device after a loss')
}

export const buildEncoder = () => {
    state.encoderPromise ??= (async () => {
        const ort = state.ort ??= await loadOrt()
        await ensureDevice(ort)
        // URL-sourced, not ArrayBuffer: constructing from a buffer holds the
        // bytes + the parsed graph + the GPU copy at once (~2–2.5× weights).
        const s = await ort.InferenceSession.create(await modelURL(ENCODER_FILE), sessionOpts('gpu-buffer'))
        markWeightsCached()   // session build populated the cache; no prefetch owed
        captureDevice()
        state.encoder = s
        return s
    })()
    state.encoderPromise.catch(() => { state.encoderPromise = null })
    return state.encoderPromise
}

export const buildDecoder = () => {
    state.decoderPromise ??= (async () => {
        const ort = state.ort ??= await loadOrt()
        await ensureDevice(ort)
        const s = await ort.InferenceSession.create(await modelURL(DECODER_FILE), sessionOpts())
        captureDevice()
        state.decoder = s
        return s
    })()
    state.decoderPromise.catch(() => { state.decoderPromise = null })
    return state.decoderPromise
}

/**
 * Encoder weights into the model cache, no session: the 78 MB fetch is the slow
 * half of a cold first encode, and costs no GPU memory here. Body drained and
 * discarded — the caches keep the bytes. Abortable; the host runs it only while
 * idle, and an aborted attempt restarts from scratch next window.
 */
let prefetchPromise = null
let prefetched = false
export const weightsPrefetched = () => prefetched
export const markWeightsCached = () => { prefetched = true }
export const prefetchWeights = (signal) => {
    if (prefetched) return Promise.resolve(true)
    prefetchPromise ??= (async () => {
        const r = await fetch(await modelURL(ENCODER_FILE), { signal })
        if (!r.ok) throw new Error(`sam21: encoder prefetch failed (${r.status})`)
        const reader = r.body?.getReader()
        if (!reader) { await r.arrayBuffer(); return true }
        let bytes = 0
        for (;;) {
            const { done, value } = await reader.read()
            if (done) break
            bytes += value.byteLength
        }
        console.log('[sam21] encoder weights cached', `${(bytes / 1e6).toFixed(1)} MB`)
        return true
    })()
    prefetchPromise.then(() => { prefetched = true }, () => { prefetchPromise = null })
    return prefetchPromise
}

export const releaseEncoder = () => {
    clearTimeout(state.idleTimer)
    state.idleTimer = null
    const s = state.encoder
    state.encoder = null
    state.encoderPromise = null
    try { s?.release?.() } catch { /* already gone */ }
}

/**
 * Release once nothing has needed the lane for ENCODER_IDLE_MS.
 *
 * This drops EVERYTHING, not just the encoder, and that is the whole point.
 * Measured on the canonical NEF, all-Chrome footprint, one tab:
 *
 *   encoder live                            2096 MB   (gpu 1254)
 *   releaseEncoder() — decoder kept          2059 MB   (gpu 1209)   −37 MB
 *   releaseAll()     — device given up       1120 MB   (gpu  270)  −976 MB
 *
 * ORT pools GPU memory per DEVICE, and the device lives as long as any session
 * holds it. So keeping the 9.9 MB decoder as a "cheap anchor" to preserve the
 * embedding was not cheap at all — it pinned the encoder's whole ~1 GB pool for
 * as long as the anchor stood, and only the deep-idle worker exit (30 s+) ever
 * took it back. Releasing the last session destroys the device, which is the
 * event that actually returns the pool.
 *
 * The embedding dies with the device, but it was persisted to OPFS on the way
 * in, so the next click re-establishes it from disk rather than re-encoding.
 * Clicks inside the window are untouched (~32 ms); the first click after it
 * pays the rebuild.
 */
const releaseIdle = () => {
    if (state.encoderRefs > 0 || state.decodeRefs > 0) { scheduleIdleRelease(); return }
    releaseAll()
}

const scheduleIdleRelease = () => {
    clearTimeout(state.idleTimer)
    state.idleTimer = setTimeout(() => {
        state.idleTimer = null
        releaseIdle()
    }, ENCODER_IDLE_MS)
}

/** Releasing the LAST session destroys the WebGPU device — ORT drops it once no
 *  session holds it — and a device loss clears every embedding. Callers that
 *  want the embedding to survive must keep one session alive; the decoder is
 *  9.9 MB and is the cheap anchor. Only release both to give the device up. */
export const releaseDecoder = () => {
    const s = state.decoder
    state.decoder = null
    state.decoderPromise = null
    try { s?.release?.() } catch { /* already gone */ }
}

// WebKit crashes the content process after a few hundred WebGPU inferences on
// one ORT session — measured upstream at ~500 with ORT 1.24.3 on iOS 26
// (microsoft/onnxruntime#27584, still open and unfixed), and the same engine
// shows a runaway wasm-compile path on WebKit 26 (#26827). Chrome does not have
// it. A long editing session is hundreds of decodes, so this is reachable.
//
// Retire the DECODER on a run count well under that. It is the session doing
// nearly all the inferences (one encode per image, one decode per click), it is
// 9.9 MB, and rebuilding it costs ~1 s — against a crashed tab. Deliberately
// NOT the encoder: dropping the last session destroys the WebGPU device and
// every embedding with it, so this only fires while the encoder is holding the
// device open, and otherwise waits for a moment when it is.
const RUN_RECYCLE_AFTER = 300
// Chromium is the engine without the bug and the one where a needless rebuild
// costs real time, so it opts out; same signal as the SharedWorker gate.
const NEEDS_RUN_RECYCLE = typeof navigator !== 'undefined' && !navigator.userAgentData
let runsSinceRecycle = 0
/** Count one inference; retire the decoder before WebKit's crash threshold. */
const noteRun = () => {
    if (!NEEDS_RUN_RECYCLE) return
    runsSinceRecycle += 1
    if (runsSinceRecycle < RUN_RECYCLE_AFTER) return
    if (!state.decoder || !state.encoder) return   // never leave the device unheld
    console.warn(`[seglab][sam21] retiring the decoder after ${runsSinceRecycle} runs (WebKit inference-count crash guard)`)
    releaseDecoder()
    runsSinceRecycle = 0
}

export const releaseEmbedding = (key = null) => {
    for (const [k, v] of state.embeds) {
        if (key !== null && k !== key) continue
        for (const b of v.buffers) { try { b.destroy() } catch { /* gone */ } }
        state.embeds.delete(k)
        forgetPick(k)   // nothing to continue from once the embedding is gone
    }
}

/** Drop everything. Used by dormancy and by the memory governor's top rung. */
export const releaseAll = () => {
    clearTimeout(state.idleTimer)
    state.idleTimer = null
    releaseEncoder()
    releaseDecoder()
    releaseEmbedding()
    // Dropping the LAST session takes ORT's GPUDevice with it and ORT will not
    // build another by itself, so arm the recovery seam for the next encode.
    // Without this the governor's top rung left the lane dead until the worker
    // exited: every later click failed "no WebGPU device after encode".
    if (state.device) state.lost = true
}

/** Destroy the GPUDevice. TERMINAL for this JS context: ORT-Web caches its
 *  device on the module object and the ES module cache survives re-import, so
 *  it will not build a new one — measured, the next encode hangs. The caller
 *  must tear the whole worker down (sam21-host `shutdown`), which is also the
 *  only thing that returns ORT's buffer pool to the OS. */
export const destroyDevice = () => {
    releaseAll()
    dropPreproc()
    const dev = state.device
    state.device = null
    state.lost = true // ensureDevice re-arms ORT on next use
    try { dev?.destroy?.() } catch { /* already gone */ }
}

/** ImageBitmap → ImageNet-normalised NCHW at 1024². Downscale happens in
 *  createImageBitmap/drawImage, so no full-res RGBA ever lands in the heap. */
const toTensorCPU = (ort, bitmap) => {
    const canvas = new OffscreenCanvas(SIDE, SIDE)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(bitmap, 0, 0, SIDE, SIDE)
    const { data } = ctx.getImageData(0, 0, SIDE, SIDE)
    const plane = SIDE * SIDE
    const chw = new Float32Array(3 * plane)
    for (let c = 0; c < 3; c += 1) {
        const m = IMAGENET_MEAN[c]
        const s = IMAGENET_STD[c]
        for (let p = 0, i = c; p < plane; p += 1, i += 4) chw[c * plane + p] = (data[i] / 255 - m) / s
    }
    return new ort.Tensor(ORT_TYPE, pack(chw), [1, 3, SIDE, SIDE])
}

/* ─── GPU preprocessing ───────────────────────────────────────────────────
   Same work, without the round trip: the CPU path downloads 4 MB with
   getImageData and then walks 3.1 M elements twice (normalise, pack to fp16)
   to hand ORT a buffer it uploads straight back. Measured on a 768×1024 proxy
   (spikes/sam21/pretensor-scale.html): 12.8 ms → 4.7 ms, and the tensor is
   BIT-IDENTICAL — all 3 145 728 fp16 elements, zero differing.

   drawImage stays the scaler. A WGSL sampler would not match canvas
   resampling, and dropping `willReadFrequently` does not either: the flag
   selects a different 2D backend whose filter disagrees with the old one on
   40 % of pixels (max |Δ| 0.018). Same flag, same pixels. */
const PREPROC_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
const SIDE: u32 = ${SIDE}u;
const PLANE: u32 = ${SIDE * SIDE}u;
fn chan(t: vec4f, c: u32) -> f32 {
    if (c == 0u) { return t.r; }
    if (c == 1u) { return t.g; }
    return t.b;
}
fn mean_of(c: u32) -> f32 {
    if (c == 0u) { return ${IMAGENET_MEAN[0]}; }
    if (c == 1u) { return ${IMAGENET_MEAN[1]}; }
    return ${IMAGENET_MEAN[2]};
}
fn std_of(c: u32) -> f32 {
    if (c == 0u) { return ${IMAGENET_STD[0]}; }
    if (c == 1u) { return ${IMAGENET_STD[1]}; }
    return ${IMAGENET_STD[2]};
}
fn value(e: u32) -> f32 {
    let c = e / PLANE;
    let p = e % PLANE;
    // textureLoad on rgba8unorm is byte/255 exactly — the number getImageData
    // gave. A sampler, or an -srgb format, would not be.
    let t = textureLoad(src, vec2u(p % SIDE, p / SIDE), 0);
    return (chan(t, c) - mean_of(c)) / std_of(c);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let w = gid.x;
    if (w >= arrayLength(&dst)) { return; }
    // PLANE is even, so a word never straddles two channel planes.
    // pack2x16float is core WGSL — no shader-f16 feature to negotiate.
    dst[w] = pack2x16float(vec2f(value(w * 2u), value(w * 2u + 1u)));
}`

let preproc = null

const dropPreproc = () => {
    try { preproc?.tex.destroy() } catch { /* device already gone */ }
    preproc = null
}

const buildPreproc = (dev) => {
    if (preproc?.device === dev) return preproc
    dropPreproc()
    preproc = {
        device: dev,
        stage: new OffscreenCanvas(SIDE, SIDE),
        tex: dev.createTexture({
            size: [SIDE, SIDE],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
                | GPUTextureUsage.RENDER_ATTACHMENT,
        }),
        pipeline: dev.createComputePipeline({
            layout: 'auto',
            compute: { module: dev.createShaderModule({ code: PREPROC_WGSL }), entryPoint: 'main' },
        }),
    }
    preproc.ctx = preproc.stage.getContext('2d', { willReadFrequently: true })
    return preproc
}

/** Returns { tensor, buffer } — the buffer is OURS: ORT does not take
 *  ownership of a fromGpuBuffer input, and it must outlive the queued run. */
const toTensorGPU = (ort, bitmap, dev) => {
    const p = buildPreproc(dev)
    p.ctx.drawImage(bitmap, 0, 0, SIDE, SIDE)
    dev.queue.copyExternalImageToTexture(
        { source: p.stage, flipY: false },
        { texture: p.tex, premultipliedAlpha: false, colorSpace: 'srgb' },
        [SIDE, SIDE],
    )
    const bytes = 3 * SIDE * SIDE * 2
    const buffer = dev.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    const enc = dev.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(p.pipeline)
    pass.setBindGroup(0, dev.createBindGroup({
        layout: p.pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: p.tex.createView() },
            { binding: 1, resource: { buffer } },
        ],
    }))
    pass.dispatchWorkgroups(Math.ceil(bytes / 4 / 64))
    pass.end()
    dev.queue.submit([enc.finish()])
    return {
        buffer,
        tensor: ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float16', dims: [1, 3, SIDE, SIDE] }),
    }
}

/** Move an encoder output into a buffer WE own, so releasing the encoder does
 *  not take the embedding with it (ORT allocates outputs from the session
 *  arena). GPU→GPU where ORT kept it on device; upload otherwise. */
const ownCopy = (dev, tensor, bytes) => {
    const dst = dev.createBuffer({
        size: Math.ceil(bytes / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    const src = tensor.location === 'gpu-buffer' ? tensor.gpuBuffer : null
    if (src) {
        const enc = dev.createCommandEncoder()
        enc.copyBufferToBuffer(src, 0, dst, 0, dst.size)
        dev.queue.submit([enc.finish()])
    } else {
        const d = tensor.data
        dev.queue.writeBuffer(dst, 0, d.buffer, d.byteOffset, d.byteLength)
    }
    return dst
}

/**
 * Encode `bitmap` into the resident embedding. `key` identifies the image so a
 * repeat encode is skipped. Returns timings + the bytes now held.
 */
/** Adopt raw embedding bytes (from OPFS) into GPU buffers we own. */
const adoptBuffers = (dev, raw) => {
    const buffers = EMBED.map((e, i) => {
        const want = count(e.dims) * BYTES_PER
        if (!raw[i] || raw[i].byteLength < want) throw new Error(`sam21: short ${e.name}`)
        const dst = dev.createBuffer({
            size: Math.ceil(want / 4) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
        dev.queue.writeBuffer(dst, 0, raw[i], 0, want)
        return dst
    })
    return buffers
}

/**
 * OPFS lives in the TAB, not here. Measured: navigator.storage.getDirectory()
 * throws SecurityError inside a SharedWorker in Chrome (same-origin script, not
 * just blob:), while the main thread and dedicated workers are fine. So the
 * client reads/writes the cache and moves raw bytes across the port; these two
 * functions are that seam. 8 MB, transferable, zero-copy.
 */
export const adoptEmbedding = async (key, raw) => {
    if (!raw || raw.length !== EMBED.length) return false
    // Already resident — the common multi-tab case, since the second tab on a
    // photo reads the same OPFS file the first tab wrote. Re-uploading over a
    // live embedding was 8 MB of pure waste AND destroyed buffers that in-flight
    // decode work still referenced: measured, the second tab's first click died
    // with "Buffer used in submit while destroyed". Adopting an identical copy
    // has nothing to do.
    const resident = state.embeds.get(key)
    if (resident) { resident.usedAt = performance.now(); return true }
    // A cold tab has no session yet, so there is no ORT device to upload into.
    // The decoder is the cheap one (10 MB) and is needed for the next click
    // anyway — building it is how we get the device.
    let dev = captureDevice()
    if (!dev) { await buildDecoder(); dev = captureDevice() }
    // Still nothing (ORT populates the device lazily) — refuse rather than
    // throw. The caller falls back to a real encode, which is only slower.
    if (!dev) return false
    releaseEmbedding(key)
    state.embeds.set(key, { buffers: adoptBuffers(dev, raw), bytes: EMBED_BYTES, usedAt: performance.now() })
    trimEmbeds()
    return true
}

/** Mark `key` freshly used; reports whether it is resident. A tab riding
 *  another tab's embedding never encodes, so without this it never touches the
 *  LRU either and its image can be evicted out from under it. */
export const touchEmbedding = (key) => {
    const held = state.embeds.get(key)
    if (!held) return false
    held.usedAt = performance.now()
    return true
}

// Bytes the recycle below already pulled off the GPU, held for the tab's OPFS
// write so an import pays ONE 8 MB readback instead of two. Consumed once —
// the host transfers these buffers to the tab, which detaches them.
let lastReadback = null

/** Raw embedding bytes for `key`, for the tab to persist. */
export const exportEmbedding = async (key) => {
    if (lastReadback?.key === key) {
        const { raw } = lastReadback
        lastReadback = null
        return raw
    }
    const held = state.embeds.get(key)
    const dev = state.device
    if (!held || !dev) return null
    return readbackEmbedding(dev, held.buffers)
}

/**
 * Hand ORT's device pool back the moment the encode is done, instead of at the
 * idle timer.
 *
 * ORT pools GPU memory per DEVICE and the device is pinned by ANY live session,
 * so the encoder's workspace outlives the encoder itself. Measured (releaseIdle
 * above): releasing the encoder alone returns 37 MB; giving the device up
 * returns 976 MB. The idle timer does eventually do that — but every click
 * re-arms it, so across a real editing session the encoder's ~1 GB is resident
 * from the first import until the user walks away. That is the steady state, and
 * it is the one the tab gets reaped in.
 *
 * The encoder is needed for exactly one operation per image. The decoder runs
 * against a 64² embedding and needs a fraction of that pool. So: read the
 * embedding back (which the OPFS persist was going to pay for anyway — see
 * lastReadback), drop everything, and let the DECODER build the next device.
 * Same two sessions, never on the same device, so the click phase never inherits
 * the encoder's workspace.
 *
 * Deliberately explicit about the destroy rather than waiting on ORT to drop the
 * device on its own: that takes ~400 ms (measured, releaseIdle) and adopting
 * into a device that is halfway through dying is how you get "Buffer used in
 * submit while destroyed". `ensureDevice` is the supported seam for handing ORT
 * a fresh one and is already the spontaneous-loss path.
 *
 * Best-effort by construction. Any failure latches this off for the session and
 * leaves the previous behaviour (idle release) in charge — a memory optimisation
 * must not be able to take the lane down with it.
 */
let recycleBroken = false
export const recycleEncoderDevice = async (key) => {
    if (recycleBroken) return null
    const held = state.embeds.get(key)
    const dev = state.device
    // Never mid-flight: another tab's encode or a decode still holds buffers.
    if (!held || !dev || state.encoderRefs > 0 || state.decodeRefs > 0) return null
    // Nothing to reclaim if no encoder was ever built on this device.
    if (!state.encoder) return null
    try {
        const raw = await readbackEmbedding(dev, held.buffers)
        releaseAll()
        dropPreproc()
        state.device = null
        try { dev.destroy() } catch { /* already gone */ }
        // Drive the recovery state directly: the lost handler will also set
        // these, but it resolves a turn or more later and adoptEmbedding must
        // not race it.
        state.lost = true
        state.ort = null
        const ok = await adoptEmbedding(key, raw)
        if (!ok) throw new Error('sam21: could not re-adopt after recycle')
        lastReadback = { key, raw }
        return { key, bytes: EMBED_BYTES }
    } catch (err) {
        recycleBroken = true
        console.warn('[seglab][sam21] device recycle failed; falling back to idle release:', err?.message)
        return null
    }
}

/** Read the resident embedding back to CPU so it can be persisted. */
const readbackEmbedding = async (dev, buffers) => {
    const out = []
    for (const src of buffers) {
        const staging = dev.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
        const enc = dev.createCommandEncoder()
        enc.copyBufferToBuffer(src, 0, staging, 0, src.size)
        dev.queue.submit([enc.finish()])
        await staging.mapAsync(GPUMapMode.READ)
        out.push(staging.getMappedRange().slice(0))
        staging.unmap()
        staging.destroy()
    }
    return out
}

export const encode = async ({ bitmap, key, keepEncoder = false }) => {
    const hit = state.embeds.get(key)
    if (hit) {
        hit.usedAt = performance.now()
        return { cached: true, key, bytes: hit.bytes }
    }
    const t0 = performance.now()

    state.encoderRefs += 1
    try {
        const ort = state.ort ??= await loadOrt()
        const session = await buildEncoder()
        // THE fallback — the only one in the lane. The shader hands ORT a device
        // buffer, and there is no device until ORT has built one (null right
        // after a worker restart; see the capture below). Nothing else can send
        // us here now that the EP and precision are fixed. The CPU path is
        // correct, just slower: ~12 ms of preprocessing instead of ~1 ms, both
        // bit-identical, against a ~500 ms encode.
        const preDev = captureDevice()
        const built = preDev ? toTensorGPU(ort, bitmap, preDev) : null
        const input = built?.tensor || toTensorCPU(ort, bitmap)
        const tPre = performance.now()

        const out = await session.run({ [session.inputNames[0]]: input })
        if (!built) input.dispose?.()   // ours to destroy, after the fence below
        const tRun = performance.now()

        // Capture AFTER the run. ORT populates env.webgpu.device lazily, and
        // immediately after a worker restart (deep-idle shutdown, governor
        // teardown) it is still null at session-create time — reading it early
        // produced "Cannot read properties of null (reading 'createBuffer')".
        const dev = captureDevice()
        if (!dev) throw new Error('sam21: no WebGPU device after encode')

        const buffers = EMBED.map((e) => {
            const t = out[e.name]
            if (!t) throw new Error(`sam21: encoder emitted no ${e.name}`)
            return ownCopy(dev, t, count(e.dims) * BYTES_PER)
        })
        // ownCopy only SUBMITS the GPU→GPU copy. Disposing ORT's tensors frees
        // the source buffers, and freeing one that queued work still references
        // is what "used in submit while destroyed" means. Wait for the copy.
        await dev.queue.onSubmittedWorkDone()
        built?.buffer.destroy()   // safe only now: the run referenced it
        for (const e of EMBED) out[e.name]?.dispose?.()
        state.embeds.set(key, { buffers, bytes: EMBED_BYTES, usedAt: performance.now() })
        trimEmbeds()
        return {
            cached: false,
            key,
            bytes: EMBED_BYTES,
            preMs: +(tPre - t0).toFixed(1),
            runMs: +(tRun - tPre).toFixed(1),
            totalMs: +(performance.now() - t0).toFixed(1),
        }
    } finally {
        state.encoderRefs -= 1
        noteRun()   // WebKit inference-count guard; no-op on Chromium
        // Drain-release, deferred. Releasing the instant refs hit zero makes a
        // queue of encodes rebuild the session between every one; the caller
        // passes keepEncoder when more encodes are already queued, and the idle
        // timer covers the rest.
        if (state.encoderRefs === 0 && state.decodeRefs === 0 && !keepEncoder) scheduleIdleRelease()
    }
}

const gpuTensor = (ort, buffer, dims) =>
    ort.Tensor.fromGpuBuffer(buffer, { dataType: ORT_TYPE, dims })

/**
 * Decode a mask from the resident embedding. `clicks` are [{x, y, label}] in
 * 1024² space (label 1 include / 0 exclude).
 *
 * Pass EXACTLY the clicks you have — never pad to a fixed width. The graph
 * appends its own padding point, and each extra label -1 slot adds another
 * not_a_point_embed token to the prompt attention. Measured: 2 clicks padded to
 * 8 slots score predicted-IoU 0.0001 and land IoU 0.227 against the same clicks
 * unpadded. That is why num_points is left dynamic in the export.
 *
 * Returns the highest predicted-IoU candidate of the three SAM emits — picking
 * argmax rather than index 0 is free accuracy the model already computed.
 * Logits, not a thresholded mask: §10 upsamples the continuous field and runs a
 * guided filter, which is where boundary quality actually comes from.
 */
export const decode = async ({ clicks = [], key, sid = null }) => {
    const embed = state.embeds.get(key)
    // Explicit key, never "whatever is resident": with one instance across tabs
    // the latest encode is not necessarily this tab's image.
    if (!embed) throw new Error(`sam21: no embedding for ${key}`)
    // Hold the lane open for the whole decode, and re-arm the idle window on the
    // way out — a click is what "still in use" looks like once the encode is done.
    clearTimeout(state.idleTimer)
    state.idleTimer = null
    state.decodeRefs += 1
    try {
        return await runDecode(embed, clicks, key, sid)
    } finally {
        state.decodeRefs -= 1
        noteRun()   // WebKit inference-count guard; no-op on Chromium
        if (state.encoderRefs === 0 && state.decodeRefs === 0) scheduleIdleRelease()
    }
}

/**
 * Refine by re-prompting with the previous pass's own logits.
 *
 * `mask_input` / `has_mask_input` were being fed a zero mask forever, which
 * threw away a trained-in capability the export already carries: verified
 * against decoder.fp16.onnx, mask_input runs through
 * `prompt_encoder.mask_downscaling` and has_mask_input blends it against
 * no_mask_embed. That is SAM's own iterative-refinement path — the second call
 * a reference predictor makes — and it is what SAMRefiner (ICLR 2025)
 * generalises.
 *
 * Why it is the right lever HERE: the boundary loss on a thin subject is in the
 * FIELD, not in the post-filter. Measured on streetlight.jpg's lamp assembly
 * (a 4–10 px structure at the 1024 proxy), against a luma ground truth:
 *
 *   coarse 256-grid field, no filter    IoU 0.522   boundary IoU 0.410
 *   + guided filter, shipped r=8 s=4    IoU 0.824   boundary IoU 0.771
 *
 * and every other (radius, scale) tried was WORSE — r=8/s=1 fell to 0.575,
 * r=4/s=2 to 0.745. The filter is already at its ceiling; only a better field
 * moves the remaining 0.18. A refeed costs one decoder run against a resident
 * 64² embedding and 128 KB of tensor — no session, no model, no GPU memory.
 *
 * Gated on the first pass's own predicted IoU rather than run always: a decode
 * is ~32 ms and the click→paint budget is 50 ms, so a second pass has to be
 * bought where it pays. A confident mask is a clean blob that refinement cannot
 * improve; a low score is exactly the thin/ambiguous subject that it can. Set
 * REFINE_BELOW to 1 to refine unconditionally.
 */
const REFINE_BELOW = 0.95
// Below this the second pass answered a different question — keep pass one.
const REFINE_MIN_AGREE = 0.5

/** One decoder pass. `maskInput` = fp32 256² logits from a previous pass. */
const decodePass = async (ort, session, embed, coords, labels, n, maskInput) => {
    const feeds = {
        image_embed: gpuTensor(ort, embed.buffers[2], EMBED[2].dims),
        high_res_feats_0: gpuTensor(ort, embed.buffers[0], EMBED[0].dims),
        high_res_feats_1: gpuTensor(ort, embed.buffers[1], EMBED[1].dims),
        point_coords: new ort.Tensor(ORT_TYPE, pack(coords), [1, n, 2]),
        point_labels: new ort.Tensor(ORT_TYPE, pack(labels), [1, n]),
        mask_input: new ort.Tensor(ORT_TYPE, pack(maskInput || new Float32Array(MASK_SIDE * MASK_SIDE)), [1, 1, MASK_SIDE, MASK_SIDE]),
        has_mask_input: new ort.Tensor(ORT_TYPE, pack(Float32Array.of(maskInput ? 1 : 0)), [1]),
    }
    const out = await session.run(feeds)
    for (const k of ['point_coords', 'point_labels', 'mask_input', 'has_mask_input']) feeds[k].dispose?.()

    const iou = unpack(out.iou_predictions.data)
    const plane = MASK_SIDE * MASK_SIDE
    const all = unpack(out.masks.data)
    // Every candidate, not just the winner. SAM computes three (subpart / part /
    // whole) on every decode and this lane threw two away — 512 KB of already-
    // paid-for inference. The refine pass below needs them to pick by agreement
    // rather than by score, and "the mask took part of my object" is answerable
    // from them with no model call at all.
    const planes = [0, 1, 2].map((i) => Float32Array.from(all.subarray(i * plane, (i + 1) * plane)))
    out.masks.dispose?.()
    out.iou_predictions.dispose?.()
    return { planes, scores: Array.from(iou) }
}

/** Grid cells per unit of SAM's 1024² prompt space. */
const GRID = MASK_SIDE / SIDE

/**
 * One-slot continuity cache: the mask this caller was last shown for this
 * image. `sid` identifies the client, because one shared instance serves every
 * tab and "the previous mask" is per-tab, not per-image.
 *
 * A COPY, not a reference — the chosen plane is transferred to the client, so
 * holding the original would leave a detached buffer here.
 *
 * Only a STRICTLY LONGER click list counts as a refinement. A fresh first click
 * elsewhere in the frame must not be pinned to the previous object, and the
 * app's exclude-click path decodes a lone point that has nothing to continue.
 */
let lastPick = null
const priorMask = (key, sid, n) =>
    (lastPick && lastPick.key === key && lastPick.sid === sid && n > lastPick.n ? lastPick.plane : null)

/** Forget the continuity anchor (image swap, eviction, device loss). */
export const forgetPick = (key) => { if (!key || lastPick?.key === key) lastPick = null }

const runDecode = async (embed, clicks, key, sid) => {
    embed.usedAt = performance.now()
    const t0 = performance.now()
    const ort = state.ort ??= await loadOrt()
    const session = await buildDecoder()

    const pts = clicks.length ? clicks : [{ x: SIDE / 2, y: SIDE / 2, label: 1 }]
    const n = pts.length
    const coords = new Float32Array(n * 2)
    const labels = new Float32Array(n)
    for (let i = 0; i < n; i += 1) {
        coords[i * 2] = pts[i].x
        coords[i * 2 + 1] = pts[i].y
        labels[i] = pts[i].label ?? 1
    }

    const prior = priorMask(key, sid, n)
    const first = await decodePass(ort, session, embed, coords, labels, n, null)

    // Arbitration, not argmax — see mask-select.js for why predicted IoU alone
    // gets the nested / negative-click / box cases wrong.
    const sel = chooseCandidate({
        planes: first.planes, scores: first.scores, clicks: pts,
        side: MASK_SIDE, scale: GRID, previous: prior, minAgree: REFINE_MIN_AGREE,
    })
    let chosen = first.planes[sel.index]
    let index = sel.index
    let iou = first.scores[sel.index]
    let stability = sel.pick.stability
    let reason = sel.reason
    let refined = false

    if (iou < REFINE_BELOW) {
        const second = await decodePass(ort, session, embed, coords, labels, n, chosen)
        noteRun()   // the extra inference counts against the WebKit crash guard
        // The refined pass is judged against the mask we FED IT, by the same
        // arbitration. Agreement is the acceptance test: the reference flow
        // re-prompts with multimask_output=False and gets one consolidated
        // mask, but this export always emits three, so the second pass is just
        // as free to answer a DIFFERENT question — on the streetlight it can
        // walk from the lamp arm onto the trolley pole crossing behind it.
        // Scoring by predicted IoU instead would be actively wrong for nesting:
        // a confident sub-cluster of petals outscores a whole bloom, so the top
        // score is the mask that leaves parts out.
        const r = chooseCandidate({
            planes: second.planes, scores: second.scores, clicks: pts,
            side: MASK_SIDE, scale: GRID, previous: chosen, minAgree: REFINE_MIN_AGREE,
        })
        if (r.reason === 'agree') {
            chosen = second.planes[r.index]
            index = r.index
            iou = second.scores[r.index]
            stability = r.pick.stability
            reason = 'refined'
            refined = true
        }
    }

    // The candidates SAM already computed, so the app can offer "not that part
    // — the whole thing" without another decode. Model order (roughly subpart →
    // part → whole) is preserved; the app sorts by area for cycling.
    const alternates = first.planes.filter((_, i) => i !== sel.index)
    // Hygiene on every plane, so cycling never lands on an uncleaned mask.
    const regions = cleanRegions(chosen, MASK_SIDE, { clicks: pts, scale: GRID })
    for (const a of alternates) cleanRegions(a, MASK_SIDE, { clicks: pts, scale: GRID })

    lastPick = { key, sid, n, plane: Float32Array.from(chosen) }

    return {
        logits: chosen,                  // fp32 [256×256], threshold at >0 AFTER upsampling
        alternates,
        side: MASK_SIDE,
        index,
        iou,
        scores: first.scores,
        // Aligned with `alternates` — the chosen plane may come from the refine
        // pass, so its score cannot be recovered from `scores` by index.
        altScores: first.scores.filter((_, i) => i !== sel.index),
        stability: +stability.toFixed(3),
        reason,
        regions,
        refined,
        ms: +(performance.now() - t0).toFixed(1),
    }
}

/** One throwaway decode so the first real click pays no shader compilation. */
export const warmDecoder = async (key) => {
    if (!state.embeds.has(key)) return false
    try { await decode({ key, clicks: [{ x: SIDE / 2, y: SIDE / 2, label: 1 }] }); return true } catch { return false }
}

/** Debug: read an embedding tensor back to the CPU and summarise it, so the
 *  browser's encoder output can be diffed against the Python reference. */
export const embedStats = async (key, which = 'image_embed') => {
    const embed = state.embeds.get(key)
    if (!embed) throw new Error(`sam21: no embedding for ${key}`)
    const idx = EMBED.findIndex((e) => e.name === which)
    const src = embed.buffers[idx]
    const dev = state.device
    const staging = dev.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const enc = dev.createCommandEncoder()
    enc.copyBufferToBuffer(src, 0, staging, 0, src.size)
    dev.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const buf = staging.getMappedRange().slice(0)
    const v = unpack(new Uint16Array(buf))
    staging.unmap()
    staging.destroy()
    let mn = Infinity
    let mx = -Infinity
    let s = 0
    for (const x of v) { if (x < mn) mn = x; if (x > mx) mx = x; s += x }
    return { name: which, n: v.length, min: mn, max: mx, mean: s / v.length, head: Array.from(v.slice(0, 8)) }
}

export const laneState = () => ({
    encoder: !!state.encoder,
    decoder: !!state.decoder,
    embedKeys: [...state.embeds.keys()],
    embedBytes: state.embeds.size * EMBED_BYTES,
    encoderRefs: state.encoderRefs,
    decodeRefs: state.decodeRefs,
    deviceLost: state.lost,
    rebuilds: state.rebuilds,
    // Inferences since the last WebKit crash-guard recycle. A SharedWorker's
    // console never reaches the page, so this counter is the only way to observe
    // the guard from a tab (or from verify).
    runs: runsSinceRecycle,
    runRecycleAt: NEEDS_RUN_RECYCLE ? RUN_RECYCLE_AFTER : 0,
})

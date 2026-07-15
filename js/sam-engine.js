/**
 * sam-engine — SlimSAM on-device segmentation, worker/main agnostic
 * --------------------------------------------------------------------
 * The actual transformers.js inference. Environment-agnostic (no DOM
 * elements, no `window`): accepts ImageBitmap/canvas sources and uses
 * OffscreenCanvas, so the same code runs inside the dedicated worker
 * (sam-worker.js — the production path) or inline on the main thread
 * (sam-client.js's fallback when worker construction fails).
 *
 * SlimSAM-77 (Apache-2.0, ~14 MB) is the sole model lane. It is lazy-loaded
 * only after the bounded interaction proxy is visible, then encodes once per
 * document and prompt-decodes on every interaction. A second segmentation
 * model is deliberately absent: no upgrade/download/compile/cache may add a
 * competing heavyweight allocation to the upload-selection path.
 *
 * Every decoded mask then goes through the shared post pipeline:
 *   lasso clamp → seeded component cleanup + hole fill (sam-core) →
 *   guided-filter edge-band refinement against the photo (edge-refine).
 * The pipeline stays bounded to the active document and interaction proxy.
 */

import {
    buildBoxPrompt,
    buildPointPrompt,
    cleanupMaskRGBA,
    maskChannelToRGBA,
    maskIoU,
    pickBestMask,
} from './sam-core.js'
import { refineMaskEdges, refineMaskEdgesTiled } from './edge-refine.js'
import { loadEmbedding, saveEmbedding } from './embed-store.js'

// Pinned CDN build of transformers.js (ESM single file, CORS-enabled) —
// version-locked so a CDN-side major bump can never break the app.
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'

const LANES = {
    draft: {
        label: 'slimsam',
        model: 'Xenova/slimsam-77-uniform',
        cls: 'SamModel',
        options: {},
        cacheMax: 1,
    },
}

const LOAD_TIMEOUT_MS = 12 * 60 * 1000
const INFER_TIMEOUT_MS = 120 * 1000
const PROGRESS_THROTTLE_MS = 120

const withTimeout = (promise, ms, label) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms)
        promise.then(
            (v) => { clearTimeout(timer); resolve(v) },
            (e) => { clearTimeout(timer); reject(e) },
        )
    })

// A budget-less engine assumes the 8 GB safe profile; policy.js overrides
// bounded sizes and optional features through warm({ budget }).
const DEFAULT_BUDGET = {
    profile: 'lite',
    draftCacheMax: 1,
    forceWasm: false,
    detectorWebGPU: false,
    embedPersist: false,
}

const state = {
    device: null,            // 'webgpu' | 'wasm' once known
    forcedWasm: false,       // sticky downgrade after a WebGPU runtime failure
    ready: false,            // SlimSAM serving
    obsoleteBefore: 0,       // jobs with revision < this must not commit (cancel op)
    budget: { ...DEFAULT_BUDGET },
}
const trace = (event, detail = {}) => console.log(`[seglab][engine] ${event}`, detail)

/** Obsolete jobs older than `revision`: queued ones drop at dequeue,
 *  in-flight ones skip the post pipeline and report {stale:true}. */
export const cancelBefore = (revision) => {
    if (revision > state.obsoleteBefore) state.obsoleteBefore = revision
}

const isStale = (req) => req.revision !== undefined && req.revision < state.obsoleteBefore

// One heavy GPU job at a time (Offline Pro rule; also caps memory peaks).
let jobChain = Promise.resolve()
const serialize = (fn) => {
    const run = jobChain.then(() => fn())
    jobChain = run.then(() => {}, () => {})
    return run
}

let transformersPromise = null
const bundles = { draft: null }

/** `${lane}:${imageKey}` → { embeddings, original_sizes, reshaped_input_sizes, gray, w, h } */
const embedCache = new Map()

// Event sink — the worker shell points this at postMessage; the inline
// fallback points it at the client's emitter. Events: {type:'progress'}.
let eventSink = null
export const setEventSink = (fn) => { eventSink = fn }
let lastProgressAt = 0
const emitEvent = (event) => {
    if (!eventSink) return
    if (event.type === 'progress' && event.detail?.status === 'progress') {
        const now = Date.now()
        if (now - lastProgressAt < PROGRESS_THROTTLE_MS) return
        lastProgressAt = now
    }
    try { eventSink(event) } catch { /* sink gone */ }
}

const activeLaneKey = () => 'draft'

export const getBudget = () => ({ ...state.budget })

// Text-detector production lane. Grounding DINO Tiny is intentionally absent:
// its current ONNX export's Gather_11 expects a rank-3 phrase-token tensor,
// whereas transformers.js uses the standard rank-2 zero-shot representation.
// OWLv2/WASM is slower but compatible and never enters that broken graph.
export const detectorCandidates = () => {
    return [{ detector: 'owl', device: 'wasm', dtype: 'q8' }]
}

export const getEngineState = () => ({
    device: state.device,
    forcedWasm: state.forcedWasm,
    ready: state.ready,
    lane: LANES[activeLaneKey()].label,
    profile: state.budget.profile,
    cachedImages: embedCache.size,
})

export const loadTransformers = () => {
    transformersPromise ??= import(TRANSFORMERS_CDN).then((T) => {
        // Explicitly enable the browser Cache API so model ONNX blobs are
        // persisted in Cache Storage after the first download. The Service
        // Worker (sw.js) provides a second, independent caching layer that
        // also handles the ESM bundle import itself.
        if (T.env) {
            T.env.useBrowserCache = true
            // Allow loading from cache even when offline.
            T.env.allowLocalModels = false
            // WASM threads exist only under COOP/COEP (plain hosts run 1);
            // when isolated, leave a core for the OS instead of min(4, cores).
            const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4
            if (T.env.backends?.onnx?.wasm) T.env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, cores - 1))
        }
        return T
    })
    return transformersPromise
}

const pickDevice = async () => {
    if (state.forcedWasm) { state.device = 'wasm'; return 'wasm' }
    if (state.device) return state.device
    let device = 'wasm'
    try {
        // navigator.gpu exists in dedicated workers too (where supported).
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            // Prefer the user's performance GPU (including a discrete GPU) when
            // the browser permits it; fall back to its default adapter.
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
                || await navigator.gpu.requestAdapter()
            if (adapter) device = 'webgpu'
        }
    } catch { /* no WebGPU */ }
    state.device = device
    return device
}

const loadBundle = () => {
    if (bundles.draft) return bundles.draft
    const lane = LANES.draft
    bundles.draft = (async () => {
        const transformers = await loadTransformers()
        const Cls = transformers[lane.cls]
        if (!Cls) throw new Error(`${lane.cls} is missing from this transformers.js build`)
        const device = await pickDevice()
        const progress_callback = (info) => emitEvent({
            type: 'progress',
            detail: {
                lane: 'draft',
                status: info?.status,
                file: info?.file,
                progress: info?.progress,
                loaded: info?.loaded,
                total: info?.total,
            },
        })
        const [model, processor] = await withTimeout(
            Promise.all([
                Cls.from_pretrained(lane.model, { device, progress_callback, ...lane.options })
                    // SlimSAM may retry deviceless; the bounded WASM lane is
                    // the safe fallback when WebGPU initialization fails.
                    .catch(() => Cls.from_pretrained(lane.model, { progress_callback, ...lane.options })),
                transformers.AutoProcessor.from_pretrained(lane.model, { progress_callback }),
            ]),
            LOAD_TIMEOUT_MS,
            `${lane.label} model load`,
        )
        return { model, processor, transformers, laneKey: 'draft' }
    })()
    bundles.draft.catch(() => { bundles.draft = null })
    return bundles.draft
}

/** Warm SlimSAM only; all other segmentation-model paths are retired. */
export const warm = async ({ budget = null } = {}) => {
    trace('warm-start', { model: 'slimsam', profile: budget?.profile })
    if (budget) state.budget = { ...DEFAULT_BUDGET, ...budget }
    if (state.budget.forceWasm && !state.forcedWasm) {
        state.forcedWasm = true
        state.device = 'wasm'
    }
    await loadBundle()
    state.ready = true
    const engineState = getEngineState()
    trace('warm-ready', engineState)
    return engineState
}

const makeCanvas = (w, h) => {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
    if (typeof document === 'undefined') throw new Error('No canvas available for on-device selection')
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    return c
}

/**
 * Encode `source` once per lane and cache embeddings + a grayscale copy of
 * the photo (the guide image for edge refinement) under `lane:imageKey`.
 * `ephemeral` skips the cache insert — one-shot crop encodes (HD export)
 * must never evict a document embedding the user is actively clicking on.
 */
const ensureEmbeddings = async (bundle, imageKey, source, { ephemeral = false, pixels = null, gray = null } = {}) => {
    const cacheKey = `${bundle.laneKey}:${imageKey}`
    const hit = embedCache.get(cacheKey)
    if (hit) {
        trace('embedding-hit', { cacheKey })
        // Refresh LRU position.
        embedCache.delete(cacheKey)
        embedCache.set(cacheKey, hit)
        return { entry: hit, encoded: false }
    }
    // Document embeddings persist in OPFS across sessions on trusted budgets
    // only — the lite profile forbids persistence (packing duplicates tensor
    // buffers, an avoidable peak on the upload-selection path). Crop keys
    // stay memory-only.
    const persistable = !ephemeral && imageKey.startsWith('doc:') && state.budget.embedPersist === true
    if (persistable) {
        const persisted = await loadEmbedding(cacheKey, bundle.transformers.Tensor)
        if (persisted) {
            trace('embedding-persisted-hit', { cacheKey })
            insertWithCap(bundle, cacheKey, persisted)
            return { entry: persisted, encoded: false }
        }
    }
    if (!source) throw new Error('Embedding cache miss and no image source provided')
    trace('embedding-miss', { cacheKey, width: source.width, height: source.height, ephemeral })
    const { RawImage } = bundle.transformers
    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('Selection source has no usable dimensions')

    // One draw, one readback: pixels feed BOTH the model input and the
    // grayscale guide used by edge refinement. Callers that already hold the
    // crop's pixels/gray (hdRefine) pass them in — no second readback.
    if (!pixels) {
        const canvas = makeCanvas(w, h)
        canvas.getContext('2d').drawImage(source, 0, 0)
        pixels = canvas.getContext('2d').getImageData(0, 0, w, h)
    }
    const image = new RawImage(pixels.data, w, h, 4)
    if (!gray) {
        gray = new Float32Array(w * h)
        for (let i = 0; i < gray.length; i += 1) {
            const j = i * 4
            gray[i] = (0.299 * pixels.data[j] + 0.587 * pixels.data[j + 1] + 0.114 * pixels.data[j + 2]) / 255
        }
    }

    const inputs = await bundle.processor(image)
    const embeddings = await withTimeout(
        bundle.model.get_image_embeddings({ pixel_values: inputs.pixel_values }),
        INFER_TIMEOUT_MS,
        'image encode',
    )
    const entry = {
        embeddings,
        original_sizes: inputs.original_sizes,
        reshaped_input_sizes: inputs.reshaped_input_sizes,
        gray,
        w,
        h,
    }
    if (ephemeral) return { entry, encoded: true }
    insertWithCap(bundle, cacheKey, entry)
    // Delay the OPFS copy until the caller knows this prewarm is still useful.
    // Packing tensors duplicates their buffers briefly, so beginning that work
    // for a stale click would create exactly the avoidable memory peak this
    // pipeline is designed to prevent.
    return { entry, encoded: true, save: persistable ? () => saveEmbedding(cacheKey, entry) : null }
}

// Release an evicted entry's tensors where the runtime exposes disposal;
// otherwise dropping the JS references is the release.
const releaseEmbedding = (entry) => {
    if (!entry) return
    try {
        for (const t of Object.values(entry.embeddings || {})) t?.dispose?.()
    } catch { /* runtime without dispose */ }
    entry.gray = null
}

// Insert + LRU eviction, capped by the session budget. A cap of 1 (lite)
// makes holding two embeddings structurally impossible: the old entry is
// released before the new one is retained.
const insertWithCap = (bundle, cacheKey, entry) => {
    const cacheMax = Math.max(1, state.budget.draftCacheMax ?? LANES.draft.cacheMax)
    if (cacheMax === 1) {
        for (const key of [...embedCache.keys()]) {
            if (key.startsWith(`${bundle.laneKey}:`)) {
                releaseEmbedding(embedCache.get(key))
                embedCache.delete(key)
            }
        }
        embedCache.set(cacheKey, entry)
        return
    }
    embedCache.set(cacheKey, entry)
    let laneCount = 0
    for (const key of embedCache.keys()) if (key.startsWith(`${bundle.laneKey}:`)) laneCount += 1
    if (laneCount > cacheMax) {
        for (const key of embedCache.keys()) {
            if (key.startsWith(`${bundle.laneKey}:`)) {
                releaseEmbedding(embedCache.get(key))
                embedCache.delete(key)
                break
            }
        }
    }
}

/** New document: drop its one embedding; model weights stay. */
export const releaseDocument = () => {
    for (const key of [...embedCache.keys()]) {
        releaseEmbedding(embedCache.get(key))
        embedCache.delete(key)
    }
}

/**
 * Memory-pressure ladder — dispose the detector first, then (at level 3)
 * release the current embedding. The detector lives in another module, so the
 * dynamic import avoids a static cycle.
 */
export const relievePressure = async (level = 1) => {
    const freed = []
    if (level >= 1) {
        try { (await import('./detect-engine.js')).disposeDetector(); freed.push('detector') } catch { /* no detector loaded */ }
    }
    if (level >= 3) {
        releaseDocument()
        freed.push('embedding')
    }
    return freed
}

/** Zero the mask outside `poly` dilated by `margin` — the lasso guarantee:
 *  the decoder snaps the boundary INSIDE the region, the clamp owns the
 *  outside, so a lasso can never bleed onto a neighbouring object. */
const clampRGBAToPolygon = (rgba, w, h, poly, margin) => {
    const canvas = makeCanvas(w, h)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.beginPath()
    ctx.moveTo(poly[0][0], poly[0][1])
    for (let i = 1; i < poly.length; i += 1) ctx.lineTo(poly[i][0], poly[i][1])
    ctx.closePath()
    ctx.fillStyle = '#fff'
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = Math.max(1, margin * 2)
    ctx.lineJoin = 'round'
    ctx.fill()
    ctx.stroke()
    const region = ctx.getImageData(0, 0, w, h).data
    for (let i = 0; i < region.length; i += 4) {
        if (region[i + 3] === 0) {
            rgba[i] = 0
            rgba[i + 1] = 0
            rgba[i + 2] = 0
        }
    }
}

const staleResult = (req) => ({ stale: true, revision: req.revision })

/**
 * Prompt build → decode → best-candidate RGBA → lasso clamp → hygiene.
 * The shared decode core: the proxy path (segmentOnce) and the crop paths
 * (hdRefine now, interactive escalation next) all run THIS against their
 * own embeddings entry — prompts are expressed in the entry's pixel space.
 * Returns null when the job went stale mid-kernel.
 */
const decodeMask = async (bundle, entry, req) => {
    const { Tensor } = bundle.transformers
    const { clicks, box } = req
    const reshaped = entry.reshaped_input_sizes[0]
    const boxPrompt = buildBoxPrompt(box, entry.w, entry.h, reshaped)
    // Box with no clicks: anchor with the box centre as a positive point —
    // an interior anchor improves whole-object box selection, and the SAM
    // export cannot run point-free (see buildBoxPrompt).
    const effectiveClicks = (clicks && clicks.length > 0)
        ? clicks
        : (boxPrompt ? [[boxPrompt.center[0], boxPrompt.center[1], 1]] : [])
    const pointPrompt = buildPointPrompt(effectiveClicks, entry.w, entry.h, reshaped)
    if (!pointPrompt) throw new Error('No clicks or box to select with')

    const decoderInputs = {
        ...entry.embeddings,
        input_points: new Tensor('float32', pointPrompt.points, pointPrompt.pointDims),
        input_labels: new Tensor('int64', pointPrompt.labels, pointPrompt.labelDims),
    }
    if (boxPrompt) {
        decoderInputs.input_boxes = new Tensor('float32', boxPrompt.box, boxPrompt.boxDims)
    }

    let outputs
    try {
        outputs = await withTimeout(bundle.model(decoderInputs), INFER_TIMEOUT_MS, 'mask decode')
    } catch (err) {
        // SlimSAM's decoder export accepts the box tensor on some ORT builds
        // but fails inside Gather_11 on others (rather than reporting an
        // unknown input). The box's centre is already a positive point, so a
        // points-only retry is both safe and substantially better than
        // failing the user's selection outright. Keep unrelated decoder
        // errors visible: only box-input validation and Gather failures take
        // this compatibility path.
        const message = String(err?.message || err)
        if (!decoderInputs.input_boxes || !/\b(?:input|invalid|unknown|gather)\b/i.test(message)) throw err
        trace('box-decoder-fallback', { reason: message.slice(0, 180) })
        delete decoderInputs.input_boxes
        outputs = await withTimeout(bundle.model(decoderInputs), INFER_TIMEOUT_MS, 'mask decode (points only)')
    }

    // The kernel above was uninterruptible; if this job was cancelled while
    // it ran, skip the post pipeline — the encode is cached, nothing wasted.
    if (isStale(req)) return null

    const masks = await bundle.processor.post_process_masks(
        outputs.pred_masks,
        entry.original_sizes,
        entry.reshaped_input_sizes,
    )
    const scores = outputs.iou_scores.data
    const best = pickBestMask(scores)
    const [, , mh, mw] = masks[0].dims
    if (!mw || !mh) throw new Error('Decoder returned a malformed mask')
    const rawRgba = maskChannelToRGBA(masks[0].data, mw, mh, best)
    const decodedAt = Date.now()

    // Clamp → hygiene. `rawRgba` survives untouched for the UI's
    // raw-vs-refined comparison toggle.
    const rgba = rawRgba.slice()
    if (Array.isArray(req.clampPoly) && req.clampPoly.length >= 3) {
        clampRGBAToPolygon(rgba, mw, mh, req.clampPoly, req.clampMargin || 8)
    }
    const seeds = effectiveClicks.filter((c) => c[2]).map((c) => [c[0], c[1]])
    const hygiene = cleanupMaskRGBA(rgba, mw, mh, seeds)

    return {
        rgba,
        rawRgba,
        width: mw,
        height: mh,
        score: Number(scores[best]) || 0,
        hygiene,
        decodedAt,
    }
}

const segmentOnce = async (req) => {
    // Dropped at dequeue: a queued job whose revision was cancelled while it
    // waited must not touch the model at all.
    if (isStale(req)) return staleResult(req)
    const t0 = Date.now()
    const laneKey = 'draft'
    trace('segment-start', { laneKey, revision: req.revision, imageKey: req.imageKey })
    const bundle = await loadBundle()
    const { entry, encoded } = await ensureEmbeddings(bundle, req.imageKey, req.source)
    const tEncoded = Date.now()

    const dec = await decodeMask(bundle, entry, req)
    if (!dec) return staleResult(req)

    const refined = refineMaskEdges(dec.rgba, dec.width, dec.height, entry.gray)

    const result = {
        rgba: dec.rgba,
        rawRgba: dec.rawRgba,
        width: dec.width,
        height: dec.height,
        score: dec.score,
        device: state.device,
        lane: LANES[laneKey].label,
        revision: req.revision,
        encoded,
        encodeMs: tEncoded - t0,
        decodeMs: dec.decodedAt - tEncoded,
        postMs: Date.now() - dec.decodedAt,
        hygiene: dec.hygiene,
        bandPixels: refined.bandPixels,
    }
    trace('segment-result', { lane: result.lane, revision: result.revision, encoded, score: result.score })
    return result
}

/**
 * Segment with a single safe retry path: a WebGPU failure falls back to
 * SlimSAM on WASM after dropping the poisoned session and embedding.
 *
 * @param {{ imageKey: string, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement|null,
 *           clicks: Array<[number, number, 0|1]>, box: number[]|null,
 *           clampPoly?: Array<[number, number]>, clampMargin?: number }} req
 */
export const segment = async (req) => {
    // A click can beat the queued warm: adopt the caller's budget so the
    // implicit lazy load still runs under the session policy, never defaults.
    if (req.budget && !state.ready) state.budget = { ...DEFAULT_BUDGET, ...req.budget }
    const laneKey = activeLaneKey()
    trace('segment-route', { laneKey, revision: req.revision })
    try {
        return await serialize(() => segmentOnce({ ...req, lane: laneKey }))
    } catch (err) {
        console.error('[seglab][engine] segment-failed', { laneKey, revision: req.revision, err })
        if (isStale(req)) return staleResult(req) // don't demote lanes over a cancelled job
        // Out-of-memory: free the detector before rebuilding SlimSAM on WASM.
        if (/out of memory|allocation failed|\boom\b|out of device memory/i.test(String(err?.message))) {
            await relievePressure(2)
        }
        if (state.device !== 'webgpu' || state.forcedWasm) throw err
        console.warn('[seglab] segment failed on WebGPU; retrying on WASM:', err?.message)
        state.forcedWasm = true
        state.device = 'wasm'
        state.ready = false
        bundles.draft = null
        releaseDocument()
        return serialize(() => segmentOnce({ ...req, lane: 'draft' }))
    }
}

/**
 * M5 encode-at-import: warm SlimSAM's embedding for `imageKey` in the
 * background so the first click skips straight to decode. The model encode is
 * serialized; the optional OPFS copy deliberately happens after that queue is
 * released, so a real selection never waits behind a disk write.
 */
export const encodeImage = async (req) => {
    const prepared = await serialize(async () => {
        if (isStale(req)) return { stale: true }
        const laneKey = 'draft'
        const bundle = await loadBundle()
        const { encoded, save } = await ensureEmbeddings(bundle, req.imageKey, req.source)
        if (isStale(req)) return { stale: true }
        return { encoded, save, lane: LANES[laneKey].label, device: state.device }
    })
    if (prepared.stale || isStale(req)) return staleResult(req)
    // `await` yields the worker event loop, so segments can enter serialize()
    // while OPFS writes this best-effort revisit cache in the background.
    if (prepared.save) await prepared.save()
    if (isStale(req)) return staleResult(req)
    return { encoded: prepared.encoded, lane: prepared.lane, device: prepared.device }
}

/**
 * Encode a crop (cached under `cacheKey`, or `ephemeral` for one-shot export
 * crops that must not evict a live document embedding) and decode prompts
 * expressed in the crop's OWN pixel space. The shared crop primitive behind
 * both HD export (M1) and interactive escalation (M3). Returns the hygiene'd
 * crop mask + the crop's gray guide so the caller runs edge refinement with
 * whatever band it needs; null when the job went stale mid-kernel.
 */
const cropSegment = async (req) => {
    const laneKey = 'draft'
    const bundle = await loadBundle()
    const { entry, encoded } = await ensureEmbeddings(bundle, req.cacheKey, req.source, {
        ephemeral: req.ephemeral, pixels: req.pixels, gray: req.gray,
    })
    const dec = await decodeMask(bundle, entry, req)
    if (!dec) return null
    return { ...dec, gray: entry.gray, encoded, lane: LANES[laneKey].label }
}

/* ─── HD export refinement (original-resolution crop) ────────────────────── */

const hdRefineOnce = async (req) => {
    if (isStale(req)) return staleResult(req)
    const t0 = Date.now()
    const { source, proxyMask, proxySubrect, prompts, doDecode, cropKey } = req
    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('HD crop has no usable dimensions')

    // The crop's own pixels are both the (optional) model input and the
    // guide image for edge refinement.
    const canvas = makeCanvas(w, h)
    const cctx = canvas.getContext('2d', { willReadFrequently: true })
    cctx.drawImage(source, 0, 0)
    const pixels = cctx.getImageData(0, 0, w, h)
    const gray = new Float32Array(w * h)
    for (let i = 0; i < gray.length; i += 1) {
        const j = i * 4
        gray[i] = (0.299 * pixels.data[j] + 0.587 * pixels.data[j + 1] + 0.114 * pixels.data[j + 2]) / 255
    }

    // Base alpha = the user-approved proxy mask, bilinearly upscaled. The
    // export may sharpen this, but never contradict it.
    const pm = makeCanvas(proxyMask.width, proxyMask.height)
    pm.getContext('2d').putImageData(
        new ImageData(new Uint8ClampedArray(proxyMask.data), proxyMask.width, proxyMask.height), 0, 0,
    )
    const up = makeCanvas(w, h)
    const uctx = up.getContext('2d', { willReadFrequently: true })
    uctx.imageSmoothingEnabled = true
    uctx.imageSmoothingQuality = 'high'
    uctx.drawImage(pm, proxySubrect.sx, proxySubrect.sy, proxySubrect.sw, proxySubrect.sh, 0, 0, w, h)
    let rgba = uctx.getImageData(0, 0, w, h).data

    // One crop re-decode at native resolution (the real detail recovery),
    // IoU-gated: if the decoder grabbed a different object than the preview,
    // keep the base — the export must match what the user saw.
    let decoded = false
    let iou = 0
    if (doDecode && cropKey) {
        try {
            const dec = await cropSegment({
                source: canvas,
                cacheKey: cropKey,
                ephemeral: true,
                pixels, // crop readback + gray already built above — reuse
                gray,
                revision: req.revision,
                ...prompts,
            })
            if (dec === null) return staleResult(req)
            iou = maskIoU(dec.rgba, rgba)
            if (iou >= 0.5) {
                rgba = dec.rgba
                decoded = true
            }
        } catch (err) {
            console.warn('[seglab] HD crop decode failed; filter-only refinement:', err?.message)
        }
    }
    if (isStale(req)) return staleResult(req)

    // Band scales with the upsampling factor (a 1-proxy-pixel staircase is
    // upFactor original pixels wide), capped so the edge never goes mushy.
    const band = Math.max(6, Math.min(16, Math.round(6 * (req.upFactor || 1))))
    const refined = refineMaskEdgesTiled(rgba, w, h, gray, { band })

    return {
        alpha: rgba,
        width: w,
        height: h,
        decoded,
        iou,
        revision: req.revision,
        bandPixels: refined.bandPixels,
        lane: decoded ? LANES.draft.label : null,
        ms: Date.now() - t0,
    }
}

/**
 * Build the original-resolution alpha for an export crop: upscaled proxy
 * mask base → optional one crop re-decode (IoU-gated) → band-tiled guided
 * refinement against the crop's own gray. Serialized like every heavy job.
 *
 * @param {{ revision?: number, cropKey: string|null, source: ImageBitmap,
 *           proxyMask: { data: Uint8ClampedArray|ArrayBuffer, width: number, height: number },
 *           proxySubrect: { sx: number, sy: number, sw: number, sh: number },
 *           prompts: { clicks: Array, box: number[]|null, clampPoly: Array|null, clampMargin: number },
 *           doDecode: boolean, upFactor: number }} req
 */
export const hdRefine = (req) => serialize(() => hdRefineOnce(req))

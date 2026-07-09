/**
 * sam-engine — two-lane on-device segmentation, worker/main agnostic
 * --------------------------------------------------------------------
 * The actual transformers.js inference. Environment-agnostic (no DOM
 * elements, no `window`): accepts ImageBitmap/canvas sources and uses
 * OffscreenCanvas, so the same code runs inside the dedicated worker
 * (sam-worker.js — the production path) or inline on the main thread
 * (sam-client.js's fallback when worker construction fails).
 *
 * TWO LANES, one contract:
 *   draft    — SlimSAM-77 (Apache-2.0, ~14 MB): loads in seconds, runs
 *              everywhere down to plain WASM. Always available first.
 *   flagship — SAM3-tracker q4f16 (SAM License, ~300 MB): Meta-demo-grade
 *              masks. WebGPU only; downloaded IN THE BACKGROUND after the
 *              draft lane is serving, hot-swapped in when ready (the 'lane'
 *              event lets the app replay current prompts at the higher
 *              quality). Browser-cached after the first download.
 *
 * Architecture — encode once, decode per interaction:
 * Both lanes split into a heavy image encoder (run ONCE per image, cached
 * per content key per lane) and a small prompt decoder (run per
 * interaction, tens of ms) — that's what makes live refinement instant.
 *
 * Every decoded mask then goes through the shared post pipeline:
 *   lasso clamp → seeded component cleanup + hole fill (sam-core) →
 *   guided-filter edge-band refinement against the photo (edge-refine).
 * The pipeline is model-agnostic: it upgrades both lanes.
 */

import {
    buildBoxPrompt,
    buildPointPrompt,
    cleanupMaskRGBA,
    maskChannelToRGBA,
    pickBestMask,
} from './sam-core.js'
import { refineMaskEdges } from './edge-refine.js'

// Pinned CDN build of transformers.js (ESM single file, CORS-enabled) —
// version-locked so a CDN-side major bump can never break the app.
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'

const LANES = {
    draft: {
        label: 'slimsam',
        model: 'Xenova/slimsam-77-uniform',
        cls: 'SamModel',
        options: {},
        // Embeddings ~8.4 MB/image → 6 keeps a session decoder-only in ~50 MB.
        cacheMax: 6,
    },
    flagship: {
        label: 'sam3',
        model: 'onnx-community/sam3-tracker-ONNX',
        cls: 'Sam3TrackerModel',
        // q4f16 = the 297 MB + 5.4 MB variant (quality-gated in the plan).
        options: { dtype: 'q4f16' },
        // Multi-level embeddings ~33 MB/image → keep 2 (~66 MB).
        cacheMax: 2,
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

const state = {
    device: null,            // 'webgpu' | 'wasm' once known
    forcedWasm: false,       // sticky downgrade after a WebGPU runtime failure
    ready: false,            // draft lane serving
    flagship: 'idle',        // 'idle' | 'loading' | 'ready' | 'failed' | 'unavailable'
}

let transformersPromise = null
const bundles = { draft: null, flagship: null }

/** `${lane}:${imageKey}` → { embeddings, original_sizes, reshaped_input_sizes, gray, w, h } */
const embedCache = new Map()

// Event sink — the worker shell points this at postMessage; the inline
// fallback points it at the client's emitter. Events: {type:'progress'|'lane'}.
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

const activeLaneKey = () => (state.flagship === 'ready' ? 'flagship' : 'draft')

export const getEngineState = () => ({
    device: state.device,
    forcedWasm: state.forcedWasm,
    ready: state.ready,
    flagship: state.flagship,
    lane: LANES[activeLaneKey()].label,
    cachedImages: embedCache.size,
})

const loadTransformers = () => {
    transformersPromise ??= import(TRANSFORMERS_CDN)
    return transformersPromise
}

const pickDevice = async () => {
    if (state.forcedWasm) { state.device = 'wasm'; return 'wasm' }
    if (state.device) return state.device
    let device = 'wasm'
    try {
        // navigator.gpu exists in dedicated workers too (where supported).
        if (typeof navigator !== 'undefined' && navigator.gpu && await navigator.gpu.requestAdapter()) {
            device = 'webgpu'
        }
    } catch { /* no WebGPU */ }
    state.device = device
    return device
}

const loadBundle = (laneKey) => {
    if (bundles[laneKey]) return bundles[laneKey]
    const lane = LANES[laneKey]
    bundles[laneKey] = (async () => {
        const transformers = await loadTransformers()
        const Cls = transformers[lane.cls]
        if (!Cls) throw new Error(`${lane.cls} is missing from this transformers.js build`)
        const device = await pickDevice()
        if (laneKey === 'flagship' && device !== 'webgpu') {
            throw new Error('flagship lane requires WebGPU')
        }
        const progress_callback = (info) => emitEvent({
            type: 'progress',
            detail: {
                lane: laneKey,
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
                    // Draft may retry deviceless (tiny model, WASM is fine);
                    // flagship never silently falls to WASM — 300 MB there
                    // would be an unusable lane, not a fallback.
                    .catch((err) => {
                        if (laneKey === 'flagship') throw err
                        return Cls.from_pretrained(lane.model, { progress_callback, ...lane.options })
                    }),
                transformers.AutoProcessor.from_pretrained(lane.model, { progress_callback }),
            ]),
            LOAD_TIMEOUT_MS,
            `${lane.label} model load`,
        )
        return { model, processor, transformers, laneKey }
    })()
    bundles[laneKey].catch(() => {
        // Draft load failures are retryable (transient network); flagship
        // failure handling is owned by maybeStartFlagship / segment.
        if (laneKey === 'draft') bundles.draft = null
    })
    return bundles[laneKey]
}

/** Kick the background flagship download once the draft lane is serving. */
const maybeStartFlagship = () => {
    if (state.flagship !== 'idle') return
    state.flagship = 'loading';
    (async () => {
        try {
            const device = await pickDevice()
            if (device !== 'webgpu' || state.forcedWasm) {
                state.flagship = 'unavailable'
                return
            }
            await loadBundle('flagship')
            state.flagship = 'ready'
            emitEvent({ type: 'lane', lane: 'flagship', label: LANES.flagship.label })
        } catch (err) {
            console.warn('[seglab] flagship lane unavailable:', err?.message)
            state.flagship = 'failed'
            bundles.flagship = null
        }
    })()
}

/**
 * Warm the draft lane (download + compile), then start the flagship
 * download in the background (pass flagship:false to skip — used by tests
 * and as a data-saver escape hatch).
 */
export const warm = async ({ flagship = true } = {}) => {
    await loadBundle('draft')
    state.ready = true
    if (flagship) maybeStartFlagship()
    else if (state.flagship === 'idle') state.flagship = 'unavailable'
    return getEngineState()
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
 */
const ensureEmbeddings = async (bundle, imageKey, source) => {
    const lane = LANES[bundle.laneKey]
    const cacheKey = `${bundle.laneKey}:${imageKey}`
    const hit = embedCache.get(cacheKey)
    if (hit) {
        // Refresh LRU position.
        embedCache.delete(cacheKey)
        embedCache.set(cacheKey, hit)
        return { entry: hit, encoded: false }
    }
    if (!source) throw new Error('Embedding cache miss and no image source provided')
    const { RawImage } = bundle.transformers
    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('Selection source has no usable dimensions')

    // One draw, one readback: pixels feed BOTH the model input and the
    // grayscale guide used by edge refinement.
    const canvas = makeCanvas(w, h)
    canvas.getContext('2d').drawImage(source, 0, 0)
    const pixels = canvas.getContext('2d').getImageData(0, 0, w, h)
    const image = new RawImage(pixels.data, w, h, 4)
    const gray = new Float32Array(w * h)
    for (let i = 0; i < gray.length; i += 1) {
        const j = i * 4
        gray[i] = (0.299 * pixels.data[j] + 0.587 * pixels.data[j + 1] + 0.114 * pixels.data[j + 2]) / 255
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
    embedCache.set(cacheKey, entry)
    // Per-lane LRU eviction.
    let laneCount = 0
    for (const key of embedCache.keys()) if (key.startsWith(`${bundle.laneKey}:`)) laneCount += 1
    if (laneCount > lane.cacheMax) {
        for (const key of embedCache.keys()) {
            if (key.startsWith(`${bundle.laneKey}:`)) { embedCache.delete(key); break }
        }
    }
    return { entry, encoded: true }
}

const purgeLane = (laneKey) => {
    for (const key of [...embedCache.keys()]) {
        if (key.startsWith(`${laneKey}:`)) embedCache.delete(key)
    }
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

const segmentOnce = async (req) => {
    const t0 = Date.now()
    const laneKey = req.lane || activeLaneKey()
    const bundle = await loadBundle(laneKey)
    const { Tensor } = bundle.transformers
    const { entry, encoded } = await ensureEmbeddings(bundle, req.imageKey, req.source)
    const tEncoded = Date.now()

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
        // Some SAM ONNX exports lack the input_boxes input. The box's centre
        // point is already in the prompt, so points-only is a usable retry.
        if (!decoderInputs.input_boxes || !/input|invalid|unknown/i.test(String(err?.message))) throw err
        delete decoderInputs.input_boxes
        outputs = await withTimeout(bundle.model(decoderInputs), INFER_TIMEOUT_MS, 'mask decode (points only)')
    }

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
    const tDecoded = Date.now()

    // Shared post pipeline: clamp → hygiene → edge refinement. `rawRgba`
    // survives untouched for the UI's raw-vs-refined comparison toggle.
    const rgba = rawRgba.slice()
    if (Array.isArray(req.clampPoly) && req.clampPoly.length >= 3) {
        clampRGBAToPolygon(rgba, mw, mh, req.clampPoly, req.clampMargin || 8)
    }
    const seeds = effectiveClicks.filter((c) => c[2]).map((c) => [c[0], c[1]])
    const hygiene = cleanupMaskRGBA(rgba, mw, mh, seeds)
    const refined = refineMaskEdges(rgba, mw, mh, entry.gray)

    return {
        rgba,
        rawRgba,
        width: mw,
        height: mh,
        score: Number(scores[best]) || 0,
        device: state.device,
        lane: LANES[laneKey].label,
        encoded,
        encodeMs: tEncoded - t0,
        decodeMs: tDecoded - tEncoded,
        postMs: Date.now() - tDecoded,
        hygiene,
        bandPixels: refined.bandPixels,
    }
}

/**
 * segmentOnce with two-level degradation, never a dead click:
 *   flagship runtime failure → sticky-demote to draft, retry;
 *   draft WebGPU failure     → sticky WASM, drop poisoned caches, retry.
 *
 * @param {{ imageKey: string, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement|null,
 *           clicks: Array<[number, number, 0|1]>, box: number[]|null,
 *           clampPoly?: Array<[number, number]>, clampMargin?: number }} req
 */
export const segment = async (req) => {
    const laneKey = activeLaneKey()
    try {
        return await segmentOnce({ ...req, lane: laneKey })
    } catch (err) {
        if (laneKey === 'flagship') {
            console.warn('[seglab] flagship decode failed; demoting to draft lane:', err?.message)
            state.flagship = 'failed'
            bundles.flagship = null
            purgeLane('flagship')
            emitEvent({ type: 'lane', lane: 'draft', label: LANES.draft.label })
            return segment(req) // re-enters on the draft lane (bounded: flagship is now sticky-failed)
        }
        if (state.device !== 'webgpu' || state.forcedWasm) throw err
        console.warn('[seglab] segment failed on WebGPU; retrying on WASM:', err?.message)
        state.forcedWasm = true
        state.device = 'wasm'
        state.ready = false
        bundles.draft = null
        if (state.flagship === 'ready' || state.flagship === 'loading') {
            state.flagship = 'failed' // flagship is WebGPU-only
            bundles.flagship = null
        }
        embedCache.clear()
        return segmentOnce({ ...req, lane: 'draft' })
    }
}

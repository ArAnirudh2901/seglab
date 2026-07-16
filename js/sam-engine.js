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
 *   draft    — SlimSAM-77 (Apache-2.0): loads in seconds, runs everywhere
 *              down to plain WASM. THE default and only automatic lane.
 *   flagship — SAM3-tracker q4f16 (SAM License, ~300 MB): Meta-demo-grade
 *              masks. WebGPU only, and DORMANT by default: it never loads,
 *              warms, or downloads automatically. The only path in is
 *              startFlagship(), reached from an explicit, confirmed user
 *              gesture (8 GB memory-safety contract — see js/policy.js).
 *              When it does arrive, the 'lane' event lets the app replay
 *              current prompts at the higher quality.
 *
 * Architecture — encode once, decode per interaction:
 * Both lanes split into a heavy image encoder (run ONCE per image) and a
 * small prompt decoder (run per interaction, tens of ms). The 8 GB
 * residency contract allows exactly ONE embedding: a single slot keyed by
 * (lane, document revision), tensors disposed on every replacement.
 *
 * Every decoded mask then goes through the shared post pipeline:
 *   lasso clamp → seeded component cleanup + hole fill (sam-core) →
 *   guided-filter edge-band refinement against the photo (edge-refine).
 * The pipeline is model-agnostic: it upgrades both lanes.
 */

import {
    buildBoxPrompt,
    buildPointPrompt,
    cleanupMaskAlpha,
    maskChannelToAlpha,
    pickBestMask,
    summarizeMaskAlpha,
    validateClickMask,
} from './sam-core.js'
import { refineMaskEdges } from './edge-refine.js'
import { loadTransformersModule } from './asset-store.js'

const LANES = {
    draft: {
        label: 'slimsam',
        model: 'Xenova/slimsam-77-uniform',
        cls: 'SamModel',
        options: {},
        // Pinned per device so weight filenames stay deterministic (the
        // vendoring script mirrors exactly these): fp32 on WebGPU
        // (vision_encoder.onnx), q8 on WASM (vision_encoder_quantized.onnx).
        dtypeByDevice: { webgpu: 'fp32', wasm: 'q8' },
    },
    flagship: {
        label: 'sam3',
        model: 'onnx-community/sam3-tracker-ONNX',
        cls: 'Sam3TrackerModel',
        // q4f16 = the 297 MB + 5.4 MB variant (quality-gated in the plan).
        options: { dtype: 'q4f16' },
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

// ONE embedding slot for the whole engine — the 8 GB residency contract.
// entry = { embeddings, original_sizes, reshaped_input_sizes, gray, w, h }.
let activeEmbedding = null // { laneKey, revision, entry }
let currentRevision = 0
let encodeCount = 0

const disposeEmbeddingEntry = (entry) => {
    if (!entry?.embeddings) return
    for (const t of Object.values(entry.embeddings)) {
        try { t?.dispose?.() } catch { /* released with the session */ }
    }
}

const releaseEmbedding = () => {
    const previous = activeEmbedding
    activeEmbedding = null
    disposeEmbeddingEntry(previous?.entry)
}

/** New document: raise the revision floor and free the embedding slot. */
export const resetImage = (revision = 0) => {
    if (revision) currentRevision = Math.max(currentRevision, revision)
    releaseEmbedding()
    return getEngineState()
}

const staleError = (label) => {
    const err = new Error(`stale ${label} (image replaced)`)
    err.stale = true
    return err
}

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
    residentEmbeddings: activeEmbedding ? 1 : 0,
    encodeCount,
})

const loadTransformers = () => {
    transformersPromise ??= loadTransformersModule()
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
        const dtypeOpts = lane.dtypeByDevice ? { dtype: lane.dtypeByDevice[device] || 'q8' } : {}
        const [model, processor] = await withTimeout(
            Promise.all([
                Cls.from_pretrained(lane.model, { device, progress_callback, ...dtypeOpts, ...lane.options })
                    // Draft may retry deviceless (tiny model, WASM is fine);
                    // flagship never silently falls to WASM — 300 MB there
                    // would be an unusable lane, not a fallback.
                    .catch((err) => {
                        if (laneKey === 'flagship') throw err
                        return Cls.from_pretrained(lane.model, { progress_callback, dtype: 'q8', ...lane.options })
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
        // failure handling is owned by startFlagship / segment.
        if (laneKey === 'draft') bundles.draft = null
    })
    return bundles[laneKey]
}

/**
 * Warm the draft lane only (download + compile). The flagship lane is NEVER
 * touched here — no background download, no upgrade, no retry. That is the
 * 8 GB memory-safety contract: one model resident, chosen for the device.
 */
export const warm = async () => {
    await loadBundle('draft')
    state.ready = true
    return getEngineState()
}

/**
 * EXPLICIT OPT-IN ONLY. Loads the SAM3 flagship lane and hot-swaps it in.
 * Reached exclusively from a confirmed user gesture routed through the
 * heavy-job queue — never from warm(), never from an import, never from a
 * URL flag. Requires WebGPU; failure is sticky for the session.
 */
export const startFlagship = async () => {
    if (state.flagship === 'ready' || state.flagship === 'loading') return getEngineState()
    const device = await pickDevice()
    if (device !== 'webgpu' || state.forcedWasm) {
        state.flagship = 'unavailable'
        return getEngineState()
    }
    state.flagship = 'loading'
    try {
        await loadBundle('flagship')
        state.flagship = 'ready'
        emitEvent({ type: 'lane', lane: 'flagship', label: LANES.flagship.label })
    } catch (err) {
        console.warn('[seglab] flagship lane unavailable:', err?.message)
        state.flagship = 'failed'
        bundles.flagship = null
    }
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
 * Encode `source` once into the single embedding slot (keyed lane+revision)
 * plus a grayscale guide for edge refinement. A slot hit skips the encoder;
 * a miss disposes the previous tensors, encodes, then re-checks currency —
 * an encode that finished for a replaced image is disposed, never stored.
 */
const ensureEmbeddings = async (bundle, revision, source) => {
    if (
        activeEmbedding
        && activeEmbedding.laneKey === bundle.laneKey
        && activeEmbedding.revision === revision
    ) {
        return { entry: activeEmbedding.entry, encoded: false }
    }
    if (!source) throw new Error('Embedding cache miss and no image source provided')
    const { RawImage } = bundle.transformers
    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('Selection source has no usable dimensions')

    // One draw, one readback (proxy-sized, ≤768): pixels feed BOTH the model
    // input and the grayscale guide used by edge refinement.
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
    encodeCount += 1
    const entry = {
        embeddings,
        original_sizes: inputs.original_sizes,
        reshaped_input_sizes: inputs.reshaped_input_sizes,
        gray,
        w,
        h,
    }
    if (revision && revision < currentRevision) {
        disposeEmbeddingEntry(entry)
        throw staleError('image encode')
    }
    releaseEmbedding()
    activeEmbedding = { laneKey: bundle.laneKey, revision, entry }
    return { entry, encoded: true }
}

/** Zero the mask outside `poly` dilated by `margin` — the lasso guarantee:
 *  the decoder snaps the boundary INSIDE the region, the clamp owns the
 *  outside, so a lasso can never bleed onto a neighbouring object. */
const clampAlphaToPolygon = (alpha, w, h, poly, margin) => {
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
    for (let i = 0; i < alpha.length; i += 1) {
        if (region[i * 4 + 3] === 0) alpha[i] = 0
    }
}

const segmentOnce = async (req) => {
    const t0 = Date.now()
    const laneKey = req.lane || activeLaneKey()
    const bundle = await loadBundle(laneKey)
    const { Tensor } = bundle.transformers
    const { entry, encoded } = await ensureEmbeddings(bundle, req.revision || 0, req.source)
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
    if (req.revision && req.revision < currentRevision) throw staleError('mask decode')
    const alphaRaw = maskChannelToAlpha(masks[0].data, mw, mh, best)
    const tDecoded = Date.now()

    // Post pipeline: clamp → hygiene → edge refinement. `alphaRaw` survives
    // untouched for the UI's raw-vs-refined comparison toggle. post:
    // 'clamp-only' hands hygiene/refinement to the cv-refine worker (the
    // caller composes the two inside one heavy job); 'js' runs it all here.
    const alpha = alphaRaw.slice()
    if (Array.isArray(req.clampPoly) && req.clampPoly.length >= 3) {
        clampAlphaToPolygon(alpha, mw, mh, req.clampPoly, req.clampMargin || 8)
    }
    let hygiene = null
    let refined = { bandPixels: 0 }
    let summary = null
    let verdict = { usable: true, reason: null }
    if (req.post !== 'clamp-only') {
        const seeds = effectiveClicks.filter((c) => c[2]).map((c) => [c[0], c[1]])
        hygiene = cleanupMaskAlpha(alpha, mw, mh, seeds)
        refined = refineMaskEdges(alpha, mw, mh, entry.gray)
        summary = summarizeMaskAlpha(alpha, mw, mh)
        verdict = validateClickMask(summary)
    }

    return {
        alpha,
        alphaRaw,
        width: mw,
        height: mh,
        summary,
        usable: verdict.usable,
        reason: verdict.reason,
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
 * @param {{ revision: number, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement|null,
 *           clicks: Array<[number, number, 0|1]>, box: number[]|null,
 *           clampPoly?: Array<[number, number]>, clampMargin?: number }} req
 */
export const segment = async (req) => {
    const laneKey = activeLaneKey()
    try {
        return await segmentOnce({ ...req, lane: laneKey })
    } catch (err) {
        if (err?.stale) throw err // superseded, not broken — no demotion
        if (laneKey === 'flagship') {
            console.warn('[seglab] flagship decode failed; demoting to draft lane:', err?.message)
            state.flagship = 'failed'
            bundles.flagship = null
            releaseEmbedding()
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
        releaseEmbedding()
        return segmentOnce({ ...req, lane: 'draft' })
    }
}

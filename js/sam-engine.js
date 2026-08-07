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
    makeMaskScratch,
    maskChannelToRGBA,
    pickBestMask,
} from './sam-core.js'
import { buildGuide, makeScratch, refineMaskEdges } from './edge-refine.js'
import {
    cropMaskPlane,
    detectionToPrompt,
    maskNMS,
    parseQuery,
    selectDetections,
} from './text-core.js'
import { detect as detectText, disposeText, warmText } from './text-engine.js'
import * as governor from './memory-governor.js'

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

/** `${lane}:${imageKey}` → { embeddings, original_sizes, reshaped_input_sizes, gray, w, h, guide } */
const embedCache = new Map()

/**
 * Post-pipeline working buffers, reused across every mask of every query.
 * A text prompt decodes N masks at once, so allocating per mask was the
 * pipeline's real memory cost — these are allocated once and grown on demand.
 */
const postScratch = { edge: makeScratch(), mask: null, w: 0, h: 0 }
const maskScratchFor = (w, h) => {
    if (!postScratch.mask || postScratch.w !== w || postScratch.h !== h) {
        postScratch.mask = makeMaskScratch(w, h)
        postScratch.w = w
        postScratch.h = h
    }
    return postScratch.mask
}

/**
 * The guided filter's image-only terms (mean/var of the grayscale guide).
 * They depend on the photo alone, so they are built once per cached image and
 * every mask decoded on it — click, box, lasso, or any of a text prompt's N
 * instances — reuses them.
 */
const ensureGuide = (entry) => {
    entry.guide ??= buildGuide(entry.gray, entry.w, entry.h)
    return entry.guide
}

/** Shared post pipeline: hygiene → edge refinement, bbox-threaded. */
const runPostPipeline = (rgba, w, h, entry, seeds) => {
    const hygiene = cleanupMaskRGBA(rgba, w, h, seeds, maskScratchFor(w, h))
    const refined = refineMaskEdges(rgba, w, h, ensureGuide(entry), {
        scratch: postScratch.edge,
        bbox: hygiene.bbox,
    })
    return { hygiene, bandPixels: refined.bandPixels }
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
            // 300 MB of weights plus its arena is the single biggest thing
            // the app can load. On a machine (or a photo) that cannot spare
            // the room, the draft lane stays — a working tool beats an OOM.
            if (!governor.canAfford('flagship')) {
                console.warn('[seglab] flagship lane skipped — not enough memory budget:', JSON.stringify(governor.budget()))
                state.flagship = 'unavailable'
                return
            }
            await loadBundle('flagship')
            governor.register('flagship', async () => {
                try { await (await bundles.flagship)?.model?.dispose?.() } catch { /* already gone */ }
                bundles.flagship = null
                purgeLane('flagship')
                state.flagship = 'idle'
                emitEvent({ type: 'lane', lane: 'draft', label: LANES.draft.label })
            })
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
    governor.adoptDeviceCeiling()
    await loadBundle('draft')
    state.ready = true
    governor.register('draft', async () => {
        try { await (await bundles.draft)?.model?.dispose?.() } catch { /* already gone */ }
        bundles.draft = null
        purgeLane('draft')
        state.ready = false
    })
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
    // Tell the governor what this photo costs before anything heavy loads —
    // a 45 MP file is ~340 MB of the ceiling on its own.
    governor.setImageFootprint(w, h)

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

/**
 * Run the prompt decoder once against cached embeddings. Shared by the click
 * lane and by every instance a text prompt produces, so the two lanes can
 * never drift apart in mask quality — text search is the same decoder, just
 * driven by a detector instead of a finger.
 */
const decodeMask = async (bundle, entry, clicks, box) => {
    const { Tensor } = bundle.transformers
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
    return { rawRgba, mw, mh, score: Number(scores[best]) || 0, effectiveClicks }
}

const segmentOnce = async (req) => {
    const t0 = Date.now()
    const laneKey = req.lane || activeLaneKey()
    const bundle = await loadBundle(laneKey)
    const { entry, encoded } = await ensureEmbeddings(bundle, req.imageKey, req.source)
    const tEncoded = Date.now()

    const { rawRgba, mw, mh, score, effectiveClicks } = await decodeMask(bundle, entry, req.clicks, req.box)
    const tDecoded = Date.now()

    // Shared post pipeline: clamp → hygiene → edge refinement. `rawRgba`
    // survives untouched for the UI's raw-vs-refined comparison toggle.
    const rgba = rawRgba.slice()
    if (Array.isArray(req.clampPoly) && req.clampPoly.length >= 3) {
        clampRGBAToPolygon(rgba, mw, mh, req.clampPoly, req.clampMargin || 8)
    }
    const seeds = effectiveClicks.filter((c) => c[2]).map((c) => [c[0], c[1]])
    const { hygiene, bandPixels } = runPostPipeline(rgba, mw, mh, entry, seeds)

    return {
        rgba,
        rawRgba,
        width: mw,
        height: mh,
        score,
        device: state.device,
        lane: LANES[laneKey].label,
        encoded,
        encodeMs: tEncoded - t0,
        decodeMs: tDecoded - tEncoded,
        postMs: Date.now() - tDecoded,
        hygiene,
        bandPixels,
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

/* ─── Text lane ──────────────────────────────────────────────────────────── */

/**
 * Select every object matching a free-form phrase, as MASKS.
 *
 * The phrase produces boxes (text-engine), the boxes drive the same SAM
 * decoder the click lane uses, and every mask goes through the same post
 * pipeline — so a text selection is exactly as clean-edged as a clicked one.
 * Boxes never surface: the result is a union mask plus one compact plane per
 * instance.
 *
 * Memory: instance masks are bbox-cropped `Uint8` planes, not full-frame RGBA.
 * At DSLR scale a full-frame RGBA mask is ~96 MB EACH, so "all the birds"
 * would be gigabytes; the union buffer is the only full-frame allocation.
 *
 * @param {{ imageKey: string, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement,
 *           query: string, maxInstances?: number, maskNmsIoU?: number }} req
 */
export const segmentText = async (req) => {
    const t0 = Date.now()
    const concepts = parseQuery(req.query)
    if (concepts.length === 0) {
        return { instances: [], rgba: null, width: 0, height: 0, reason: 'no concept in that phrase', query: req.query }
    }

    // The detector and the flagship encoder must not be resident together.
    await governor.makeRoomFor('text', { protect: ['draft'] })

    const laneKey = activeLaneKey()
    const bundle = await loadBundle(laneKey)
    const { entry, encoded } = await ensureEmbeddings(bundle, req.imageKey, req.source)
    const tEncoded = Date.now()

    const det = await detectText({
        imageKey: req.imageKey,
        source: req.source,
        phrases: concepts.map((c) => c.phrase),
        emit: emitEvent,
    })
    governor.register('text', disposeText)
    const tDetected = Date.now()

    let mw = 0
    let mh = 0
    let decodeMs = 0
    let postMs = 0
    const found = []
    for (const concept of concepts) {
        const chosen = selectDetections(det.byPhrase[concept.phrase] || [], {
            wantsAll: concept.wantsAll,
            maxInstances: req.maxInstances || 24,
        })
        for (const d of chosen) {
            const prompt = detectionToPrompt(d.box)
            const tD = Date.now()
            const dec = await decodeMask(bundle, entry, prompt.clicks, prompt.box)
            decodeMs += Date.now() - tD

            const tP = Date.now()
            const rgba = dec.rawRgba
            const seeds = prompt.clicks.map((c) => [c[0], c[1]])
            const { hygiene } = runPostPipeline(rgba, dec.mw, dec.mh, entry, seeds)
            postMs += Date.now() - tP

            const plane = cropMaskPlane(rgba, dec.mw, dec.mh, hygiene.bbox)
            if (!plane || plane.area === 0) continue
            mw = dec.mw
            mh = dec.mh
            found.push({ phrase: concept.phrase, score: d.score, maskScore: dec.score, box: d.box, plane })
        }
    }

    // Two phrases can land on the same object ("dog" and "puppy"); mask-level
    // NMS is the honest de-duplication, box overlap is not.
    const kept = maskNMS(found, req.maskNmsIoU ?? 0.7)
    kept.sort((a, b) => b.score - a.score)

    // One full-frame buffer for the whole selection.
    let union = null
    if (kept.length && mw && mh) {
        union = new Uint8ClampedArray(mw * mh * 4)
        for (let i = 0; i < mw * mh; i += 1) union[i * 4 + 3] = 255
        for (const inst of kept) {
            const p = inst.plane
            for (let y = 0; y < p.h; y += 1) {
                const src = y * p.w
                const dst = (p.y0 + y) * mw + p.x0
                for (let x = 0; x < p.w; x += 1) {
                    if (!p.plane[src + x]) continue
                    const j = (dst + x) * 4
                    union[j] = 255; union[j + 1] = 255; union[j + 2] = 255
                }
            }
        }
    }

    return {
        rgba: union,
        width: mw,
        height: mh,
        query: req.query,
        concepts: concepts.map((c) => ({ phrase: c.phrase, wantsAll: c.wantsAll })),
        instances: kept.map((i) => ({
            phrase: i.phrase,
            score: i.score,
            maskScore: i.maskScore,
            box: i.box,
            plane: i.plane.plane,
            x0: i.plane.x0,
            y0: i.plane.y0,
            w: i.plane.w,
            h: i.plane.h,
            area: i.plane.area,
        })),
        // An empty result for a phrase that genuinely is not in the photo is
        // the CORRECT answer, not a failure — say so plainly.
        reason: kept.length ? null : 'no confident match for that phrase in this photo',
        device: state.device,
        lane: LANES[laneKey].label,
        encoded,
        encodeMs: tEncoded - t0,
        detectMs: tDetected - tEncoded,
        detectCached: det.cached,
        decodeMs,
        postMs,
        totalMs: Date.now() - t0,
    }
}

/** Warm the text detector (background download), independent of the SAM lanes. */
export const warmTextLane = async () => {
    await governor.makeRoomFor('text', { protect: ['draft'] })
    const s = await warmText(emitEvent)
    if (s.status === 'ready') governor.register('text', disposeText)
    return s
}

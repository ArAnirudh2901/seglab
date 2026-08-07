/**
 * yoloe-engine — the ONE model
 * -------------------------------
 * A single YOLOE-26 segmentation network serves every selection mode. There
 * is no draft lane, no flagship lane, no second detector: the model runs ONCE
 * per image, produces every instance in the frame, and click / box / lasso /
 * text are then pure queries over that set (see select-core.js).
 *
 * WHY THIS SHAPE
 *   SAM-family architectures split into a heavy image encoder plus a prompt
 *   decoder, so every interaction is another decoder pass and every new
 *   capability is another model. YOLOE-26 is end-to-end and NMS-free: one
 *   forward gives boxes, labels and instance masks together. That collapses
 *   three models (~470 MB) and two lanes into one network of tens of
 *   megabytes, and makes interaction free — clicking around an analyzed photo
 *   never touches ONNX again.
 *
 * THE HONEST TRADE-OFF
 *   YOLO-family masks come from 32 prototype basis functions at ~160×160, so
 *   their BOUNDARIES are materially weaker than SAM's: published comparisons
 *   show YOLO-class masks losing ~48–50 points of AP as the IoU threshold
 *   tightens, against ~4 for SAM3. Detection and recall are excellent; the
 *   last few pixels are not. That is exactly what the guided-filter edge
 *   refinement in edge-refine.js repairs — it re-derives the boundary from
 *   the photo itself, is model-agnostic, and is now cheap enough to run on
 *   every instance of every query. The pairing is the point: YOLOE-26 finds
 *   things fast, the post pipeline makes the edges good.
 *
 * WEIGHTS
 *   YOLOE-26 is AGPL-3.0 (Ultralytics lineage) — a real licensing constraint
 *   for anything distributed. Weights are NOT bundled. Export your own:
 *
 *     pip install ultralytics onnx onnxslim
 *     yolo export model=yoloe26-s-seg.pt format=onnx half=True imgsz=640 \
 *                 simplify=True opset=12
 *
 *   then drop the .onnx in models/ (see MODEL below). Sizes run from
 *   yoloe26-n-seg (4.8M params) to -x; s or m is the sweet spot in a browser.
 */

import { assembleMask, decodeDetections, preprocess } from './yoloe-core.js'
import { maskNMS } from './select-core.js'
import * as governor from './memory-governor.js'

/** onnxruntime-web, pinned so a CDN major bump can never break the app. */
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/esm/ort.webgpu.min.mjs'

export const MODEL = {
    url: './models/yoloe26-s-seg.onnx',
    imgsz: 640,
    numMasks: 32,
    // Filled from models/vocabulary.json — the label list the export was
    // re-parameterized with. YOLOE folds text embeddings into the network at
    // export time, so at runtime there is NO text encoder: an arbitrary
    // phrase is matched against these labels, costing nothing.
    labels: [],
    // Weights + ORT arena at 640², in MB. Used by the memory governor.
    costMB: 220,
}

const INFER_TIMEOUT_MS = 120 * 1000
const LOAD_TIMEOUT_MS = 10 * 60 * 1000

const withTimeout = (promise, ms, label) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms)
        promise.then((v) => { clearTimeout(timer); resolve(v) }, (e) => { clearTimeout(timer); reject(e) })
    })

const state = {
    status: 'idle',    // 'idle' | 'loading' | 'ready' | 'failed'
    device: null,      // 'webgpu' | 'wasm'
    lastError: null,
}

let sessionPromise = null
let ortPromise = null

/**
 * imageKey → { instances, w, h, gray, guide, analyzeMs }
 * ONE entry is enough: selection is instant against the analyzed frame, and
 * holding several DSLR instance sets is exactly what breaks a 1 GB ceiling.
 */
const analysisCache = new Map()
const ANALYSIS_CACHE_MAX = 2

export const getEngineState = () => ({
    status: state.status,
    device: state.device,
    lastError: state.lastError,
    ready: state.status === 'ready',
    model: MODEL.url,
    labels: MODEL.labels.length,
    cachedImages: analysisCache.size,
    budget: governor.budget(),
})

const loadOrt = () => {
    ortPromise ??= import(/* @vite-ignore */ ORT_CDN)
    return ortPromise
}

/* ─── Session ────────────────────────────────────────────────────────────── */

const loadSession = (emit) => {
    if (sessionPromise) return sessionPromise
    state.status = 'loading'
    sessionPromise = (async () => {
        const ort = (await loadOrt()).default ?? await loadOrt()
        ort.env.wasm.numThreads = Math.min(4, globalThis.navigator?.hardwareConcurrency || 1)
        ort.env.wasm.simd = true

        // Try WebGPU, fall back to WASM. A ~50 MB network is perfectly usable
        // on WASM, so unlike a 300 MB encoder this fallback is a real lane.
        let session = null
        let device = 'wasm'
        emit?.({ type: 'progress', detail: { lane: 'yoloe', status: 'initiate', file: MODEL.url } })
        const bytes = await withTimeout(
            fetch(MODEL.url).then((r) => {
                if (!r.ok) throw new Error(`${MODEL.url} → HTTP ${r.status}. Export the model first (see yoloe-engine.js header).`)
                return r.arrayBuffer()
            }),
            LOAD_TIMEOUT_MS,
            'model download',
        )
        emit?.({ type: 'progress', detail: { lane: 'yoloe', status: 'done', file: MODEL.url, loaded: bytes.byteLength, total: bytes.byteLength } })

        try {
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all',
            })
            device = 'webgpu'
        } catch {
            session = await ort.InferenceSession.create(bytes, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            })
            device = 'wasm'
        }

        // Vocabulary is optional: without it, text queries have no labels to
        // match and the app says so instead of silently matching nothing.
        try {
            const res = await fetch(new URL('vocabulary.json', new URL(MODEL.url, self.location?.href || 'http://localhost/')))
            if (res.ok) MODEL.labels = await res.json()
        } catch { /* no vocabulary shipped */ }

        state.device = device
        state.status = 'ready'
        governor.register('yoloe', dispose, MODEL.costMB)
        return { session, ort }
    })()
    sessionPromise.catch((err) => {
        state.status = 'failed'
        state.lastError = String(err?.message || err)
        sessionPromise = null
    })
    return sessionPromise
}

export const warm = async (emit) => {
    governor.adoptDeviceCeiling()
    await loadSession(emit)
    return getEngineState()
}

export const dispose = async () => {
    const held = sessionPromise
    sessionPromise = null
    state.status = 'idle'
    analysisCache.clear()
    if (!held) return
    try { (await held).session?.release?.() } catch { /* already gone */ }
}

/* ─── Analysis: one forward pass per image ───────────────────────────────── */

const makeCanvas = (w, h) => {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
    if (typeof document === 'undefined') throw new Error('No canvas available')
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    return c
}

/**
 * Run the model once and cache every instance it found, plus the grayscale
 * guide the edge refinement needs. Everything downstream — clicks, regions,
 * phrases — reads this and never runs a model again.
 *
 * @param {{ imageKey: string, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement,
 *           confThreshold?: number, emit?: Function }} req
 */
export const analyze = async ({ imageKey, source, confThreshold = 0.25, emit } = {}) => {
    const hit = analysisCache.get(imageKey)
    if (hit) {
        analysisCache.delete(imageKey)
        analysisCache.set(imageKey, hit)
        return { entry: hit, analyzed: false }
    }

    const t0 = Date.now()
    const { session, ort } = await loadSession(emit)
    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('Selection source has no usable dimensions')
    governor.setImageFootprint(w, h)

    // One draw, one readback: the pixels feed BOTH the network input and the
    // grayscale guide used by edge refinement.
    const canvas = makeCanvas(w, h)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(source, 0, 0)
    const pixels = ctx.getImageData(0, 0, w, h).data
    const gray = new Float32Array(w * h)
    for (let i = 0; i < gray.length; i += 1) {
        const j = i * 4
        gray[i] = (0.299 * pixels[j] + 0.587 * pixels[j + 1] + 0.114 * pixels[j + 2]) / 255
    }

    const { tensor, lb, dims } = preprocess(pixels, w, h, MODEL.imgsz)
    const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', tensor, dims) }
    const out = await withTimeout(session.run(feeds), INFER_TIMEOUT_MS, 'detection')
    const tInfer = Date.now()

    // output0 = detections, output1 = mask prototypes. Identify by rank:
    // prototypes are the 4-D tensor.
    const outputs = session.outputNames.map((n) => out[n])
    const protoT = outputs.find((t) => t.dims.length === 4)
    const detT = outputs.find((t) => t !== protoT)
    if (!detT) throw new Error('Model produced no detection output')
    if (!protoT) throw new Error('Model produced no mask prototypes — is this a -seg export?')

    const [, pm, ph, pw] = protoT.dims
    const numMasks = pm
    const stride = detT.dims.length === 3 ? Math.max(detT.dims[1], detT.dims[2]) : 0
    const attrLen = detT.dims[1] < detT.dims[2] ? detT.dims[1] : detT.dims[2]
    const numClasses = Math.max(1, attrLen - 4 - numMasks)

    const dets = decodeDetections(detT.data, detT.dims, {
        numClasses, numMasks, lb, srcW: w, srcH: h, confThreshold,
        // YOLO26 is end-to-end; NMS would delete legitimately overlapping
        // instances. Left off unless an older export needs it.
        nmsIoU: null,
        maxDetections: 100,
    })

    const instances = []
    for (const d of dets) {
        const m = assembleMask(d.coeffs, protoT.data, { numMasks, ph, pw }, d.box, lb, w, h)
        if (!m) continue
        instances.push({
            ...m,
            box: d.box,
            score: d.score,
            classIdx: d.classIdx,
            label: MODEL.labels[d.classIdx] || `class ${d.classIdx}`,
        })
    }
    // Guard against duplicate instances from non-end-to-end exports.
    const kept = maskNMS(instances, 0.9)

    const entry = {
        instances: kept,
        w,
        h,
        gray,
        guide: null, // built lazily by the post pipeline, cached here
        analyzeMs: tInfer - t0,
        maskMs: Date.now() - tInfer,
        numClasses,
        stride,
    }
    analysisCache.set(imageKey, entry)
    if (analysisCache.size > ANALYSIS_CACHE_MAX) {
        analysisCache.delete(analysisCache.keys().next().value)
    }
    return { entry, analyzed: true }
}

/** Drop cached analysis for one image (its pixels changed). */
export const invalidate = (imageKey) => analysisCache.delete(imageKey)

/** Test hook. */
export const clearCache = () => analysisCache.clear()

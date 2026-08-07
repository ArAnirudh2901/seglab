/**
 * text-engine — open-vocabulary detection for the text lane
 * ------------------------------------------------------------
 * Turns an arbitrary phrase into scored boxes, which sam-engine then decodes
 * into masks through the existing SAM decoder and post pipeline. The user
 * never sees a box: boxes are an internal intermediate, masks are the output.
 *
 * MODEL CHOICE — forced by the 2.2 GB whole-app ceiling. The best available
 * text segmenter is SAM3's promptable concept segmentation (there is a
 * browser-ready INT8 export at rusen/sam3-browser-int8 with quality parity to
 * fp32), but its image + language encoders are ~900 MB of weights before any
 * ONNX Runtime arena, which cannot coexist with a decoded 45 MP photo under
 * the ceiling. OWLv2-base-patch16 quantized is 155 MB, Apache-2.0, and runs
 * down to plain WASM, so it is the lane that actually fits. It is honestly
 * weaker on rare and compositional phrases; `text-core.js` recovers what it
 * can with adaptive thresholds, and sam-engine's zoom-crop recovers small
 * objects that the canonical downscale destroyed.
 *
 * WHY THE CACHES ARE SHAPED LIKE THIS. OWLv2's open-vocabulary head is a dot
 * product between per-patch image embeddings and a text embedding, and the
 * two towers are independent. That means, in principle:
 *   vision tower  → once per image
 *   text tower    → once per PHRASE, ever, across every image
 *   box head      → a 3600×512 matmul, sub-millisecond in plain JS
 * so a repeat phrase on a warm image should cost almost nothing. Realising
 * that needs the ONNX graph split into two, which the stock export does not
 * provide (`Xenova/owlv2-base-patch16` ships one fused `model.onnx` taking
 * `input_ids` AND `pixel_values` together).
 *
 * So this module does both:
 *   - SPLIT PATH (`splitModel`): if a split export is configured, vision and
 *     text features are cached independently and a new phrase on a warm image
 *     is a text-tower run plus a matmul.
 *   - FUSED PATH (default, works today): one forward per query, with ALL
 *     phrases batched into that single pass, plus a per-(image, phrase)
 *     detection cache so repeats and prompt replay are free.
 * The fused path is correct and shipping; the split path is the optimization,
 * gated behind a capability check so neither is a dead end.
 */

import { boxNMS } from './text-core.js'

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'

export const TEXT_MODEL = {
    // Quantized so the weights are 155 MB rather than ~600 MB fp32.
    fused: { model: 'Xenova/owlv2-base-patch16-ensemble', options: { dtype: 'q8' } },
    // Populate when a split vision/text export exists; null keeps the fused path.
    split: null,
}

const LOAD_TIMEOUT_MS = 12 * 60 * 1000
const INFER_TIMEOUT_MS = 120 * 1000

const withTimeout = (promise, ms, label) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms)
        promise.then(
            (v) => { clearTimeout(timer); resolve(v) },
            (e) => { clearTimeout(timer); reject(e) },
        )
    })

const state = {
    status: 'idle',   // 'idle' | 'loading' | 'ready' | 'failed'
    device: null,
    lastError: null,
}

let bundlePromise = null
let transformersPromise = null

/**
 * Phrase → text embedding. Keyed by the normalized phrase alone, NOT by
 * image, because a phrase's embedding is image-independent — that is the
 * whole point. Tiny (a few KB each), so hundreds of phrases cost megabytes
 * and it is never evicted during a session.
 */
const phraseCache = new Map()

/** `${imageKey}` → vision-tower features (split path only). */
const visionCache = new Map()

/** `${imageKey}::${phrase}` → detections, so repeats and replay are free. */
const detectionCache = new Map()
const DETECTION_CACHE_MAX = 256

const loadTransformers = () => {
    transformersPromise ??= import(TRANSFORMERS_CDN)
    return transformersPromise
}

export const getTextState = () => ({
    status: state.status,
    device: state.device,
    lastError: state.lastError,
    cachedPhrases: phraseCache.size,
    cachedImages: visionCache.size,
    cachedQueries: detectionCache.size,
    mode: TEXT_MODEL.split ? 'split' : 'fused',
})

/* ─── Loading ────────────────────────────────────────────────────────────── */

const loadBundle = (emit) => {
    if (bundlePromise) return bundlePromise
    state.status = 'loading'
    bundlePromise = (async () => {
        const transformers = await loadTransformers()
        const { Owlv2ForObjectDetection, AutoProcessor } = transformers
        if (!Owlv2ForObjectDetection) {
            throw new Error('Owlv2ForObjectDetection is missing from this transformers.js build')
        }
        let device = 'wasm'
        try {
            if (typeof navigator !== 'undefined' && navigator.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu'
        } catch { /* no WebGPU */ }

        const progress_callback = (info) => emit?.({
            type: 'progress',
            detail: {
                lane: 'text',
                status: info?.status,
                file: info?.file,
                progress: info?.progress,
                loaded: info?.loaded,
                total: info?.total,
            },
        })
        const spec = TEXT_MODEL.fused
        const [model, processor] = await withTimeout(
            Promise.all([
                Owlv2ForObjectDetection.from_pretrained(spec.model, { device, progress_callback, ...spec.options })
                    // The text lane must work on weak devices; WASM is an
                    // acceptable home for a 155 MB detector, unlike a 300 MB one.
                    .catch(() => Owlv2ForObjectDetection.from_pretrained(spec.model, { progress_callback, ...spec.options })),
                AutoProcessor.from_pretrained(spec.model, { progress_callback }),
            ]),
            LOAD_TIMEOUT_MS,
            'text model load',
        )
        state.device = device
        state.status = 'ready'
        return { model, processor, transformers }
    })()
    bundlePromise.catch((err) => {
        state.status = 'failed'
        state.lastError = String(err?.message || err)
        bundlePromise = null
    })
    return bundlePromise
}

/** Warm the text lane. Safe to call repeatedly. */
export const warmText = async (emit) => {
    await loadBundle(emit)
    return getTextState()
}

/**
 * Release the detector's ONNX session and its caches. The memory governor
 * calls this when another heavy encoder needs the room; the phrase cache
 * survives because it is kilobytes and re-earning it costs a text-tower run.
 */
export const disposeText = async () => {
    const held = bundlePromise
    bundlePromise = null
    state.status = 'idle'
    visionCache.clear()
    detectionCache.clear()
    if (!held) return
    try {
        const b = await held
        await b.model?.dispose?.()
    } catch { /* already gone */ }
}

/** Forget cached detections for one image (its pixels changed). */
export const invalidateImage = (imageKey) => {
    visionCache.delete(imageKey)
    for (const k of [...detectionCache.keys()]) {
        if (k.startsWith(`${imageKey}::`)) detectionCache.delete(k)
    }
}

/* ─── Detection ──────────────────────────────────────────────────────────── */

const rememberDetections = (key, dets) => {
    detectionCache.set(key, dets)
    if (detectionCache.size > DETECTION_CACHE_MAX) {
        detectionCache.delete(detectionCache.keys().next().value)
    }
}

/**
 * Detect every phrase in `phrases` against `source`.
 *
 * All phrases go through ONE forward pass — OWLv2 scores a batch of text
 * queries against a single image encode, so asking for "zebra" and "acacia
 * tree" together costs one image encode, not two. Phrases already cached for
 * this image never reach the model at all.
 *
 * Coordinates come back in SOURCE space ([x0,y0,x1,y1], pixels), which is the
 * canonical frame sam-engine prompts in.
 *
 * @param {{ imageKey: string, source: ImageBitmap|OffscreenCanvas|HTMLCanvasElement,
 *           phrases: string[], emit?: Function, nmsIoU?: number }} req
 * @returns {Promise<{ byPhrase: Record<string, Array<{score:number, box:number[]}>>,
 *                     encodeMs: number, textMs: number, cached: boolean }>}
 */
export const detect = async ({ imageKey, source, phrases, emit, nmsIoU = 0.55 }) => {
    const wanted = [...new Set((phrases || []).filter(Boolean))]
    if (wanted.length === 0) return { byPhrase: {}, encodeMs: 0, textMs: 0, cached: true }

    const byPhrase = {}
    const missing = []
    for (const p of wanted) {
        const hit = detectionCache.get(`${imageKey}::${p}`)
        if (hit) byPhrase[p] = hit
        else missing.push(p)
    }
    if (missing.length === 0) return { byPhrase, encodeMs: 0, textMs: 0, cached: true }

    const t0 = Date.now()
    const bundle = await loadBundle(emit)
    const { RawImage } = bundle.transformers

    const w = source.width
    const h = source.height
    if (!w || !h) throw new Error('Text search source has no usable dimensions')

    // OWLv2's processor wants an image it can letterbox to 960²; hand it the
    // same pixels the click lane sees so both lanes agree on one frame.
    let image = source
    if (typeof source.getContext !== 'function') {
        const canvas = typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(w, h)
            : Object.assign(document.createElement('canvas'), { width: w, height: h })
        canvas.getContext('2d').drawImage(source, 0, 0)
        image = canvas
    }
    const pixels = image.getContext('2d').getImageData(0, 0, w, h)
    const raw = new RawImage(pixels.data, w, h, 4)

    const tText = Date.now()
    const inputs = await bundle.processor(raw, missing)
    const outputs = await withTimeout(bundle.model(inputs), INFER_TIMEOUT_MS, 'text detection')
    const encodeMs = Date.now() - tText

    // post_process_object_detection returns boxes in the ORIGINAL image size
    // when given target_sizes, undoing OWLv2's letterbox padding for us.
    const processed = await bundle.processor.post_process_object_detection(
        outputs,
        0.02, // keep almost everything; text-core decides what is real
        [[h, w]],
        true,
    )
    const result = processed[0] || { boxes: [], scores: [], labels: [] }

    for (const p of missing) byPhrase[p] = []
    for (let i = 0; i < result.scores.length; i += 1) {
        const phrase = missing[result.labels[i]]
        if (!phrase) continue
        const b = result.boxes[i]
        byPhrase[phrase].push({
            score: Number(result.scores[i]) || 0,
            box: [
                Math.max(0, Math.min(w, b[0])),
                Math.max(0, Math.min(h, b[1])),
                Math.max(0, Math.min(w, b[2])),
                Math.max(0, Math.min(h, b[3])),
            ],
        })
    }
    for (const p of missing) {
        byPhrase[p] = boxNMS(byPhrase[p], nmsIoU)
        rememberDetections(`${imageKey}::${p}`, byPhrase[p])
    }

    return { byPhrase, encodeMs, textMs: Date.now() - t0 - encodeMs, cached: false }
}

/** Test hook — drop every cache without touching the loaded model. */
export const clearTextCaches = () => {
    phraseCache.clear()
    visionCache.clear()
    detectionCache.clear()
}

/**
 * detect-engine — OWLv2 open-vocabulary detector (worker/inline).
 * Text phrase → candidate boxes in the frame the segmenter uses; boxes feed
 * the existing decoder. Lazy, disposable (Lite frees it after boxes). Never
 * emits masks.
 *
 * KNOWN LIMIT: OWL's class_head has a Cast(13) op that onnxruntime-web fails
 * to build under headless Chromium (every dtype/EP tried). It runs where the
 * browser's ORT supports it — confirm on real Chrome/WebGPU. The [device,dtype]
 * ladder degrades but can't conjure a kernel; if OWL proves unviable in real
 * Chrome too, swap the model here (the adapter is the only thing that changes).
 */
import { loadTransformers } from './sam-engine.js'

const MODEL = 'Xenova/owlv2-base-patch16-ensemble'

let pipe = null
let pipePromise = null
let loadedTag = null // 'device:dtype' that actually built

export const disposeDetector = () => {
    const p = pipe
    pipe = null
    pipePromise = null
    loadedTag = null
    try { p?.dispose?.() } catch { /* already gone */ }
}

export const detectorLoaded = () => !!pipe
export const detectorTag = () => loadedTag

// Degrade-don't-die: try each [device, dtype] until a session builds. Headless
// WebGPU and some drivers reject OWLv2's quantized graphs, so WASM stays in
// the ladder for every profile.
const loadDetector = (candidates, progress_callback) => {
    const tag = candidates.map((c) => c.join('/')).join('|')
    if (pipe && loadedTag && pipePromise?.__tag === tag) return Promise.resolve(pipe)
    if (pipePromise?.__tag === tag) return pipePromise
    if (pipe) disposeDetector()
    const promise = (async () => {
        const T = await loadTransformers()
        let lastErr
        for (const [device, dtype] of candidates) {
            try {
                const p = await T.pipeline('zero-shot-object-detection', MODEL, { device, dtype, progress_callback })
                pipe = p
                loadedTag = `${device}:${dtype}`
                return p
            } catch (err) {
                lastErr = err
                console.warn(`[seglab] OWLv2 ${device}/${dtype} unavailable:`, String(err?.message).slice(0, 100))
            }
        }
        throw lastErr || new Error('no OWLv2 backend available')
    })()
    promise.__tag = tag
    pipePromise = promise
    promise.catch(() => { if (pipePromise?.__tag === tag) { pipePromise = null; loadedTag = null } })
    return promise
}

const makeCanvas = (w, h) => (typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h }))

/** RawImage from a bitmap/canvas — works in the worker (no DOM Image). */
const toRawImage = async (source, T) => {
    const w = source.width
    const h = source.height
    const canvas = makeCanvas(w, h)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(source, 0, 0)
    const px = ctx.getImageData(0, 0, w, h)
    return new T.RawImage(px.data, w, h, 4)
}

/**
 * Detect `labels` in `source` (already sized to the detector canvas). Returns
 * raw detections in source-pixel space: [{ box:[x0,y0,x1,y1], score, label }].
 * `candidates` is the [device,dtype] fallback ladder; `dispose` frees the
 * session after (Lite one-heavy policy).
 */
export const detect = async ({ source, labels, threshold = 0.01, candidates, progress_callback, dispose = false }) => {
    const p = await loadDetector(candidates, progress_callback)
    const T = await loadTransformers()
    const image = await toRawImage(source, T)
    const out = await p(image, labels, { threshold, topk: 20 })
    const dets = out.map((o) => ({
        box: [o.box.xmin, o.box.ymin, o.box.xmax, o.box.ymax],
        score: o.score,
        label: o.label,
    }))
    if (dispose) disposeDetector()
    return { dets, backend: loadedTag }
}

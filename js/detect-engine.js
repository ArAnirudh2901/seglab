/**
 * detect-engine — OWLv2 open-vocabulary detector (worker/inline).
 * Text phrase → candidate boxes in the frame the segmenter uses; boxes feed
 * the existing decoder. Lazy, disposable (Lite frees it after boxes). Never
 * emits masks. Runs fully on-device — no sidecar, no upload.
 *
 * Export: onnx-community/…-ONNX at q8 (163 MB). The Xenova export's fp16 on
 * WebGPU (307 MB) pinned unified memory and hung the 8 GB target; policy's
 * ladder now keeps that off the small profiles entirely.
 */
import { loadTransformers } from './sam-engine.js'

const MODEL = 'onnx-community/owlv2-base-patch16-ensemble-ONNX'

/** First-search download, for the UI's warning. q8 = model_quantized.onnx. */
export const DETECTOR_DOWNLOAD_MB = 163

/** Whether the weights are already on disk. The pipe itself lives in the worker,
 *  so the main thread can't just ask `detectorLoaded()`; Cache Storage is the
 *  shared fact, and it survives reloads. */
export const detectorCached = async () => {
    if (typeof caches === 'undefined') return false
    try {
        for (const name of await caches.keys()) {
            const hits = await (await caches.open(name)).keys()
            if (hits.some((r) => r.url.includes(MODEL) && r.url.includes('model_quantized'))) return true
        }
    } catch { /* cache unavailable — assume a download */ }
    return false
}

/** Boot prefetch: pull the model files through the Service Worker cache with
 *  plain fetches — no tokenizer, no session, no wasm arena. The first search
 *  then pays only the session build. detectorCached() is the receipt: it only
 *  reports true once the .onnx itself landed. */
const HUB_FILES = ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx']
export const prefetchDetectorWeights = async () => {
    if (await detectorCached()) return true
    await Promise.allSettled(HUB_FILES.map((f) => fetch(`https://huggingface.co/${MODEL}/resolve/main/${f}`)))
    return detectorCached()
}

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

// Degrade-don't-die: try each [device, dtype] until a session builds. `candidates`
// comes from policy — small profiles never offer webgpu, so the heavy path can't
// be reached on the machines it hangs.
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

/**
 * Detect `labels` in `frame` — { data, width, height } RGB bytes the caller has
 * already letterboxed into the detector's square (see text-core). Returns
 * detections whose boxes are normalized [0,1] against that square, for the
 * caller to un-letterbox: [{ box:[x0,y0,x1,y1], score, label }].
 * `candidates` is the [device,dtype] fallback ladder; `dispose` frees the
 * session after (Lite one-heavy policy).
 */
export const detect = async ({ frame, labels, threshold = 0.05, candidates, progress_callback, dispose = false }) => {
    const p = await loadDetector(candidates, progress_callback)
    const T = await loadTransformers()
    // 3-channel and already at the model's exact input size, so rgb(), resize()
    // and pad() inside transformers.js are all no-ops: the frame is never copied
    // again before the tensor.
    const image = new T.RawImage(frame.data, frame.width, frame.height, 3)
    const out = await p(image, labels, { threshold, percentage: true, top_k: 64 })
    const dets = out.map((o) => ({
        box: [o.box.xmin, o.box.ymin, o.box.xmax, o.box.ymax],
        score: o.score,
        label: o.label,
    }))
    if (dispose) disposeDetector()
    return { dets, backend: loadedTag }
}

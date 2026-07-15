/**
 * detect-engine — resource-gated open-vocabulary detector (worker/inline).
 * Text phrase → candidate boxes in the frame the segmenter uses; boxes feed
 * the existing decoder. Lazy and disposable (Lite frees it after boxes).
 * Everything runs on-device — no sidecar or image upload.
 *
 * OWLv2 q8/WASM is the compatible production lane. Grounding DINO Tiny's
 * current ONNX export expects a three-dimensional phrase-token tensor, while
 * transformers.js supplies the standard two-dimensional zero-shot tensor;
 * its Gather_11 node consequently fails at runtime. Keep it out of the
 * production ladder until the upstream export and pipeline agree.
 */
import { loadTransformers } from './sam-engine.js'

const DETECTORS = Object.freeze({
    owl: {
        id: 'owl',
        label: 'OWLv2',
        model: 'onnx-community/owlv2-base-patch16-ensemble-ONNX',
        weightFile: 'model_quantized.onnx',
        downloadMB: 163,
    },
})

const detectorFor = (id) => DETECTORS[id] || DETECTORS.owl
const normalizeCandidate = (candidate) => Array.isArray(candidate)
    ? { detector: 'owl', device: candidate[0], dtype: candidate[1] }
    : { detector: candidate?.detector || 'owl', device: candidate?.device, dtype: candidate?.dtype }

/** Grounding DINO is deliberately disabled until its ONNX input contract is
 * fixed upstream; callers use this to keep download/cache UI truthful. */
export const acceleratedDetectorAvailable = false

/** First-search download for the UI: the compatible OWLv2 q8 model. */
export const detectorDownloadMB = () => DETECTORS.owl.downloadMB
export const DETECTOR_DOWNLOAD_MB = detectorDownloadMB()

/** Whether the weights are already on disk. The pipe itself lives in the worker,
 *  so the main thread can't just ask `detectorLoaded()`; Cache Storage is the
 *  shared fact, and it survives reloads. */
export const detectorCached = async ({ accelerated = false } = {}) => {
    if (typeof caches === 'undefined') return false
    const detector = DETECTORS.owl
    try {
        for (const name of await caches.keys()) {
            const hits = await (await caches.open(name)).keys()
            if (hits.some((r) => r.url.includes(detector.model) && r.url.includes(detector.weightFile))) return true
        }
    } catch { /* cache unavailable — assume a download */ }
    return false
}

/** Boot prefetch: pull the model files through the Service Worker cache with
 *  plain fetches — no tokenizer, no session, no wasm arena. The first search
 *  then pays only the session build. detectorCached() is the receipt: it only
 *  reports true once the .onnx itself landed. */
const HUB_FILES = ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json']
export const prefetchDetectorWeights = async ({ accelerated = false } = {}) => {
    const detector = DETECTORS.owl
    if (await detectorCached({ accelerated })) return true
    await Promise.allSettled([...HUB_FILES, `onnx/${detector.weightFile}`].map((f) => fetch(`https://huggingface.co/${detector.model}/resolve/main/${f}`)))
    return detectorCached({ accelerated })
}

let pipe = null
let pipePromise = null
let loadedTag = null // 'device:dtype' that actually built
let loadedOption = null
// A session can construct successfully yet still reject a real input (for
// example when a browser's ORT build cannot execute an exported Gather node).
// Remember that outcome for this worker so every subsequent phrase goes
// straight to the fallback instead of failing once per search.
const inferenceFailures = new Set()

const optionTag = (option) => `${option.detector}/${option.device}/${option.dtype}`

export const disposeDetector = () => {
    const p = pipe
    pipe = null
    pipePromise = null
    loadedTag = null
    loadedOption = null
    try { p?.dispose?.() } catch { /* already gone */ }
}

export const detectorLoaded = () => !!pipe
export const detectorTag = () => loadedTag

// Degrade-don't-die: try each model/backend option until a session builds.
// `candidates` currently contains only OWLv2/WASM; the generic ladder remains
// in place so a future compatible accelerator can fall back on run failures.
const loadDetector = (candidates, progress_callback) => {
    const options = (candidates || []).map(normalizeCandidate)
    const tag = options.map(optionTag).join('|')
    if (pipe && loadedTag && pipePromise?.__tag === tag) return Promise.resolve({ pipe, option: loadedOption })
    if (pipePromise?.__tag === tag) return pipePromise
    if (pipe) disposeDetector()
    const promise = (async () => {
        const T = await loadTransformers()
        let lastErr
        for (const option of options) {
            const detector = detectorFor(option.detector)
            try {
                const p = await T.pipeline('zero-shot-object-detection', detector.model, {
                    device: option.device,
                    dtype: option.dtype,
                    progress_callback,
                })
                pipe = p
                loadedTag = `${detector.id}:${option.device}:${option.dtype}`
                loadedOption = option
                return { pipe: p, option }
            } catch (err) {
                lastErr = err
                console.warn(`[seglab] ${detector.label} ${option.device}/${option.dtype} unavailable:`, String(err?.message).slice(0, 120))
            }
        }
        throw lastErr || new Error('no text-detection backend available')
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
    const T = await loadTransformers()
    // 3-channel and already at the model's exact input size, so rgb(), resize()
    // and pad() inside transformers.js are all no-ops: the frame is never copied
    // again before the tensor.
    const image = new T.RawImage(frame.data, frame.width, frame.height, 3)
    const requested = (candidates || []).map(normalizeCandidate)
    // Keep the configured order, but don't retry a model that already proved
    // incompatible with this browser's ONNX Runtime during this worker life.
    let remaining = requested.filter((option) => !inferenceFailures.has(optionTag(option)))
    if (remaining.length === 0) remaining = requested
    let lastErr
    try {
        while (remaining.length) {
            const { pipe: activePipe, option } = await loadDetector(remaining, progress_callback)
            try {
                const out = await activePipe(image, labels, { threshold, percentage: true, top_k: 64 })
                const dets = out.map((o) => ({
                    box: [o.box.xmin, o.box.ymin, o.box.xmax, o.box.ymax],
                    score: o.score,
                    label: o.label,
                }))
                return { dets, backend: loadedTag }
            } catch (err) {
                lastErr = err
                const failed = optionTag(option)
                inferenceFailures.add(failed)
                console.warn(`[seglab] ${detectorFor(option.detector).label} inference failed; falling back:`, String(err?.message || err).slice(0, 160))
                disposeDetector()
                remaining = remaining.filter((candidate) => optionTag(candidate) !== failed)
            }
        }
        throw lastErr || new Error('no text-detection backend could run this image')
    } finally {
        // Lite intentionally owns no detector session after a query, including
        // a failed query. A rejected GPU allocation must not pin the next one.
        if (dispose) disposeDetector()
    }
}

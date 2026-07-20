/**
 * detect-engine — resource-gated open-vocabulary detector (worker/inline).
 * Text phrase → candidate boxes in the frame the segmenter uses; boxes feed
 * the existing decoder. Lazy and disposable (Lite frees it after boxes).
 * Everything runs on-device — no sidecar or image upload.
 *
 * Two lanes, always ending on the universal floor:
 *   • Grounding DINO Tiny q4f16 on WebGPU — accelerated, 151 MB, best phrase
 *     grounding. Only attempted on an f16-capable accelerator (gpuTier gate).
 *     transformers.js v4's ONNX export fixed the earlier rank-3 token-tensor
 *     mismatch, so it now runs the standard zero-shot graph.
 *   • OWLv2 q8 on WASM — the compatible fallback that runs on every browser,
 *     no GPU, no f16. The ladder degrades here on any accelerated-lane failure.
 */
import { loadTransformers } from './sam-engine.js'
import { isVendored, ortWebGpuSupported } from './model-assets.js'
import { bareLabel, degenerateScores } from './text-core.js'

const DETECTORS = Object.freeze({
    grounding: {
        id: 'grounding',
        label: 'Grounding DINO',
        model: 'onnx-community/grounding-dino-tiny-ONNX',
        weightFile: 'model_q4f16.onnx',
        downloadMB: 151,
    },
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

// The accelerated lane downloads Grounding DINO; the fallback lane, OWLv2.
const laneDetector = ({ accelerated = false } = {}) => (accelerated ? DETECTORS.grounding : DETECTORS.owl)

// Grounding DINO's grounded head keys off dot-terminated lowercase noun
// phrases — the CLIP template dilutes it (see bareLabel); OWLv2 takes the
// templates as-is. Format per lane so either can run the same normalizePhrase
// output.
const formatLabels = (labels, detectorId) => (detectorId === 'grounding'
    ? [...new Set(labels.map((label) => `${bareLabel(label)}.`))]
    : labels)

/** Grounding DINO now runs on transformers.js v4; the accelerated lane is live
 *  wherever the gpuTier gate clears AND the ORT build has a WebGPU EP (pre-26
 *  Safari's doesn't). Callers still show OWLv2's size when it doesn't. */
export const acceleratedDetectorAvailable = ortWebGpuSupported()

/** First-search download for the UI: the lane the caller's device will use. */
export const detectorDownloadMB = ({ accelerated = false } = {}) => laneDetector({ accelerated }).downloadMB
export const DETECTOR_DOWNLOAD_MB = detectorDownloadMB()

/** Whether the weights are already on disk. The pipe itself lives in the worker,
 *  so the main thread can't just ask `detectorLoaded()`; Cache Storage is the
 *  shared fact, and it survives reloads. */
export const detectorCached = async ({ accelerated = false } = {}) => {
    const detector = laneDetector({ accelerated })
    // Vendored (--detector) serves from disk: no download, true even cache-less.
    if (await isVendored(detector.model)) return true
    if (typeof caches === 'undefined') return false
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
    const detector = laneDetector({ accelerated })
    // detectorCached() short-circuits on the manifest, so a --detector
    // checkout never touches huggingface.co.
    if (await detectorCached({ accelerated })) return true
    await Promise.allSettled([...HUB_FILES, `onnx/${detector.weightFile}`].map((f) => fetch(`https://huggingface.co/${detector.model}/resolve/main/${f}`)))
    return detectorCached({ accelerated })
}

let pipe = null
let pipePromise = null
let loadedTag = null // 'device:dtype' that actually built
let loadedOption = null
// A lane can fail at session build (no usable backend for its device) or
// construct successfully yet still reject a real input (for example when a
// browser's ORT build cannot execute an exported Gather node). Remember either
// outcome for this worker so every subsequent phrase goes straight to the
// fallback instead of failing once per search.
const laneFailures = new Set()

const optionTag = (option) => `${option.detector}/${option.device}/${option.dtype}`

// Idle TTL: a warm session saves the per-search rebuild, but its weights sit
// in (unified) GPU memory the whole time. After detectorIdleMs without a
// search the session is dropped, returning the memory; the next search pays
// one rebuild. Never fires mid-detect — detect() cancels it on entry.
let idleTimer = null
const cancelIdleDispose = () => {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
}
const scheduleIdleDispose = (ms) => {
    cancelIdleDispose()
    if (!(ms > 0) || !pipe) return
    idleTimer = setTimeout(() => {
        idleTimer = null
        console.log(`[seglab] detector idle ${Math.round(ms / 1000)}s — session dropped, memory returned`)
        disposeDetector()
    }, ms)
}

export const disposeDetector = () => {
    cancelIdleDispose()
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
// The ladder runs Grounding DINO/WebGPU first (when the gpuTier gate cleared)
// and falls back to OWLv2/WASM, which builds on every browser.
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
                laneFailures.add(optionTag(option))
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
 * session after (Lite one-heavy policy); `idleMs` frees it after that much
 * quiet time instead (0 = keep warm until the next encode evicts it).
 */
export const detect = async ({ frame, labels, threshold = 0.05, candidates, progress_callback, dispose = false, idleMs = 0 }) => {
    cancelIdleDispose()
    const T = await loadTransformers()
    // 3-channel and already at the model's exact input size, so rgb(), resize()
    // and pad() inside transformers.js are all no-ops: the frame is never copied
    // again before the tensor.
    const image = new T.RawImage(frame.data, frame.width, frame.height, 3)
    const requested = (candidates || []).map(normalizeCandidate)
    // Keep the configured order, but don't retry a model that already proved
    // incompatible with this browser's ONNX Runtime during this worker life.
    let remaining = requested.filter((option) => !laneFailures.has(optionTag(option)))
    if (remaining.length === 0) remaining = requested
    let lastErr
    try {
        while (remaining.length) {
            const { pipe: activePipe, option } = await loadDetector(remaining, progress_callback)
            try {
                const out = await activePipe(image, formatLabels(labels, option.detector), { threshold, percentage: true, top_k: 64 })
                // `percentage` yields [0,1]; normalize defensively so a lane that
                // returns pixels (some grounded heads ignore the flag) still maps
                // through unletterboxBox, which expects square-normalized boxes.
                const nx = (v) => (Math.abs(v) > 1.5 ? v / frame.width : v)
                const ny = (v) => (Math.abs(v) > 1.5 ? v / frame.height : v)
                const dets = out.map((o) => ({
                    box: [nx(o.box.xmin), ny(o.box.ymin), nx(o.box.xmax), ny(o.box.ymax)],
                    score: o.score,
                    label: o.label,
                }))
                // A lane can run yet return garbage: q4f16 grounding on some
                // GPUs fills top_k with a flat score band, so ranking picks
                // arbitrary boxes. Demote to the next lane while one exists.
                if (degenerateScores(dets) && remaining.length > 1) {
                    throw new Error(`degenerate score distribution (${dets.length} boxes, flat scores)`)
                }
                return { dets, backend: loadedTag }
            } catch (err) {
                lastErr = err
                const failed = optionTag(option)
                laneFailures.add(failed)
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
        else scheduleIdleDispose(idleMs)
    }
}

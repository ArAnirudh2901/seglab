/**
 * yoloe-detect — YOLOE-26 prompt-free vision lane (raw onnxruntime-web).
 *
 * The fast baked-vocabulary detector. Prompt-free (LRPC) bakes a ~4585-class
 * vocabulary into the head, so this needs NO text encoder: it detects everything
 * with class labels and the MAIN thread matches the user's phrase to those labels
 * (see text-ui + text-core phraseMatchesLabel). Arbitrary phrases that hit no
 * baked class fall through to the YOLO-World open-vocab lane (yolo-world-detect.js).
 *
 * Hosted in the disposable detect-worker so its ORT wasm arena (which only grows)
 * stays out of the segmentation worker. Runs on WebGPU where available, WASM
 * otherwise. Output is NMS-free — output0 [1,300,38] = xyxy, score, class, 32 mask
 * coeffs (masks ignored; SlimSAM owns masks). Boxes are returned normalized [0,1]
 * against the square so the caller un-letterboxes with the existing contract.
 *
 * onnxruntime-web is resolved vendored-first, CDN fallback (mirrors
 * model-assets.js). Until scripts vendor it, the CDN copy serves the plain dev
 * server; the COEP production path REQUIRES the vendored copy (2a.6).
 */

import { YOLOE_INPUT } from './text-core.js'

const SCALES = new Set(['n', 's', 'm', 'l', 'x'])

const ORT_VERSION = '1.22.0'
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
const ORT_LOCAL = new URL('../lib/ort-web/', import.meta.url).href
const ORT_FILE = 'ort.webgpu.bundle.min.mjs'
const modelURL = (scale) => new URL(`../models/yoloe/yoloe-26${scale}-seg-pf.onnx`, import.meta.url).href
const vocabURL = (scale) => new URL(`../models/yoloe/yoloe-26${scale}-pf.vocab.json`, import.meta.url).href

let ortPromise = null
/** Import onnxruntime-web (vendored first) and point its wasm loader at the same
 *  directory, so the WASM EP fallback resolves offline too. */
const loadOrt = () => {
    ortPromise ??= (async () => {
        for (const base of [ORT_LOCAL, ORT_CDN]) {
            try {
                const ort = await import(/* @vite-ignore */ base + ORT_FILE)
                ort.env.wasm.wasmPaths = base
                ort.env.wasm.numThreads = (typeof self !== 'undefined' && self.crossOriginIsolated)
                    ? Math.max(1, Math.min(4, ((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4) - 1))
                    : 1
                return ort
            } catch { /* try next source */ }
        }
        throw new Error('onnxruntime-web unavailable (no vendored copy and CDN blocked)')
    })()
    return ortPromise
}

const vocabCache = new Map() // scale → { [id]: label }
const loadVocab = async (scale) => {
    if (!vocabCache.has(scale)) vocabCache.set(scale, await (await fetch(vocabURL(scale))).json())
    return vocabCache.get(scale)
}

let session = null
let sessionPromise = null
let loadedScale = null
let backend = null // 'webgpu' | 'wasm' — the EP that actually built

/** Build the session for `scale`, trying WebGPU then WASM (ORT falls back
 *  silently inside a multi-EP list, so probe one at a time to record the EP). */
const buildSession = async (ort, scale) => {
    let lastErr
    for (const ep of ['webgpu', 'wasm']) {
        try {
            const s = await ort.InferenceSession.create(modelURL(scale), {
                executionProviders: [ep],
                graphOptimizationLevel: 'all',
            })
            backend = ep
            return s
        } catch (err) { lastErr = err }
    }
    throw lastErr || new Error('yoloe: no execution provider available')
}

export const loadYoloe = (scale = 's') => {
    if (!SCALES.has(scale)) scale = 's'
    if (session && loadedScale === scale) return Promise.resolve(session)
    if (sessionPromise && loadedScale === scale) return sessionPromise
    if (session) disposeYoloe() // scale switch — release the old session first
    loadedScale = scale
    sessionPromise = (async () => {
        const ort = await loadOrt()
        await loadVocab(scale)
        session = await buildSession(ort, scale)
        return session
    })()
    sessionPromise.catch(() => { if (loadedScale === scale) { sessionPromise = null; loadedScale = null } })
    return sessionPromise
}

let idleTimer = null
const cancelIdle = () => { if (idleTimer) { clearTimeout(idleTimer); idleTimer = null } }
const scheduleIdle = (ms) => {
    cancelIdle()
    if (!(ms > 0) || !session) return
    idleTimer = setTimeout(() => { idleTimer = null; disposeYoloe() }, ms)
}

export const disposeYoloe = () => {
    cancelIdle()
    const s = session
    session = null
    sessionPromise = null
    loadedScale = null
    backend = null
    try { s?.release?.() } catch { /* already gone */ }
}

export const yoloeLoaded = () => !!session
export const yoloeBackend = () => backend

/**
 * Detect over `frame` — { data: RGB bytes, width, height } already letterboxed
 * into the 640² square (top-left) by the caller. Returns
 * { dets: [{ box:[x0,y0,x1,y1] normalized [0,1] to the square, score, label }],
 *   backend }. `scale` picks the model; `dispose`/`idleMs` free the session after.
 */
export const detectYoloe = async ({ frame, threshold = 0.25, scale = 's', dispose = false, idleMs = 0 }) => {
    cancelIdle()
    const ort = await loadOrt()
    const s = await loadYoloe(scale)
    const vocab = await loadVocab(scale)
    const side = YOLOE_INPUT
    try {
        // RGB bytes → NCHW float32 [0,1]. frame is exactly side², 3-channel.
        const d = frame.data
        const plane = side * side
        const chw = new Float32Array(3 * plane)
        for (let i = 0, p = 0; p < plane; i += 3, p += 1) {
            chw[p] = d[i] / 255
            chw[plane + p] = d[i + 1] / 255
            chw[2 * plane + p] = d[i + 2] / 255
        }
        const input = new ort.Tensor('float32', chw, [1, 3, side, side])
        const out = await s.run({ [s.inputNames[0]]: input })
        const o0 = out[s.outputNames[0]] // [1, 300, 38]
        const [, n, ch] = o0.dims
        const data = o0.data
        const dets = []
        for (let i = 0; i < n; i += 1) {
            const b = i * ch
            const score = data[b + 4]
            if (score < threshold) continue
            let x1 = data[b]; let y1 = data[b + 1]; let x2 = data[b + 2]; let y2 = data[b + 3]
            if (Math.max(x1, y1, x2, y2) <= 1.5) { x1 *= side; y1 *= side; x2 *= side; y2 *= side } // normalized guard
            const cls = Math.round(data[b + 5])
            dets.push({ box: [x1 / side, y1 / side, x2 / side, y2 / side], score, label: vocab[cls] ?? `#${cls}` })
        }
        return { dets, backend }
    } finally {
        if (dispose) disposeYoloe()
        else scheduleIdle(idleMs)
    }
}

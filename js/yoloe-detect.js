/**
 * yoloe-detect — YOLOE-26L TEXT-PROMPT vision lane (raw onnxruntime-web).
 *
 * ONE open-vocabulary detector. It replaces both predecessors: the prompt-free
 * lane (a 4585-class vocabulary baked into an LRPC head, so arbitrary phrases
 * were impossible) and YOLO-World (a second model for the same job, reached only
 * when the first failed). Class embeddings arrive as a live `txt_feats` input,
 * so any phrase works and there is nothing to fall back to.
 *
 * RepRTA lives INSIDE the graph (scripts/export-yoloe-text.py), so this takes
 * raw MobileCLIP2-B vectors from js/text-encode.js and the adapter can never
 * drift out of sync with the detector it feeds.
 *
 * Contract: images[1,3,640,640] f32 + txt_feats[1,nc,512] f32 → output0[1,300,38]
 * = xyxy, score, slot index, 32 mask coefficients. NMS-free, like the prompt-free
 * export before it, so the caller does not run NMS for dedup — only for merging
 * across scales. Boxes are normalized [0,1] against the square.
 *
 * fp16 (55.4 MB, blanket conversion, max score delta 0.00097 vs fp32). The -pf
 * export could not take fp16 because onnxconverter-common mistyped its LRPC
 * head; this variant has no LRPC head, so the blocker left with it.
 */

import { YOLOE_INPUT } from './text-core.js'
import { loadOrt } from './ort-loader.js'

/**
 * Upper bound on class slots. The export traces at 32, but the class axis is
 * DYNAMIC, so that width is a tracing detail and never a runtime limit — swept
 * live on this graph, nc = 1·2·22·32·45·64·128 all return [1,300,38] and cost
 * 128·132·134·134·137·138·146 ms. Essentially flat, so the cap exists only to
 * bound the tensor, not to ration anything. 48 clears the largest taxonomy
 * expansion ("animal" → 45 labels + the phrase + its object form); 32 silently
 * truncated that one, and which 13 kinds got dropped was list order.
 */
export const MAX_SLOTS = 48
export const DIM = 512

const modelURL = new URL('../models/yoloe/yoloe-26l-text.fp16.onnx', import.meta.url).href

let session = null
let sessionPromise = null
let backend = null // 'webgpu' | 'wasm' — the EP that actually built

/** Build the session, trying WebGPU then WASM (ORT falls back silently inside a
 *  multi-EP list, so probe one at a time to record the EP). */
const buildSession = async (ort) => {
    let lastErr
    for (const ep of ['webgpu', 'wasm']) {
        try {
            const s = await ort.InferenceSession.create(modelURL, {
                executionProviders: [ep],
                graphOptimizationLevel: 'all',
            })
            backend = ep
            return s
        } catch (err) { lastErr = err }
    }
    throw lastErr || new Error('yoloe: no execution provider available')
}

export const loadYoloe = () => {
    if (session) return Promise.resolve(session)
    if (sessionPromise) return sessionPromise
    sessionPromise = (async () => {
        const ort = await loadOrt()
        session = await buildSession(ort)
        return session
    })()
    sessionPromise.catch(() => { sessionPromise = null })
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
    backend = null
    try { s?.release?.() } catch { /* already gone */ }
}

export const yoloeLoaded = () => !!session
export const yoloeBackend = () => backend

/**
 * Detect over `frame` — { data: RGB bytes, width, height } already letterboxed
 * into the 640² square (top-left) by the caller — conditioned on `txtFeats`, a
 * Float32Array of k×DIM L2-normalized MobileCLIP2 vectors (k = the caller's
 * phrase count, up to MAX_SLOTS). Returns
 * { dets: [{ box:[x0,y0,x1,y1] normalized [0,1] to the square, score, classIdx }],
 *   backend }. classIdx indexes the caller's phrase list.
 *
 * Exactly k classes are fed, never a padded 32. The head emits each anchor once
 * per class into a fixed top-300, so padding by repetition spent the whole
 * budget on duplicates — measured 10 unique boxes out of 300 for a one-phrase
 * query, which capped recall at ~10 instances regardless of the scene.
 */
export const detectYoloe = async ({ frame, txtFeats, threshold = 0.25, dispose = false, idleMs = 0 }) => {
    cancelIdle()
    const ort = await loadOrt()
    const s = await loadYoloe()
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
        const nc = Math.floor(txtFeats.length / DIM)
        if (nc < 1) throw new Error('yoloe: no class embeddings')
        const out = await s.run({
            images: new ort.Tensor('float32', chw, [1, 3, side, side]),
            txt_feats: new ort.Tensor('float32', txtFeats, [1, nc, DIM]),
        })
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
            dets.push({
                box: [x1 / side, y1 / side, x2 / side, y2 / side],
                score,
                classIdx: Math.round(data[b + 5]),
            })
        }
        return { dets, backend }
    } finally {
        if (dispose) disposeYoloe()
        else scheduleIdle(idleMs)
    }
}

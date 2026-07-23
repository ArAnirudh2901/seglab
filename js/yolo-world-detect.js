/**
 * yolo-world-detect — YOLO-World-v2 OPEN-VOCABULARY vision lane (raw onnxruntime-web).
 *
 * The arbitrary-phrase detector: unlike prompt-free YOLOE (fixed baked vocab),
 * this takes the class text features as a LIVE input (txt_feats), so any phrase
 * works. The phrase is encoded to CLIP ViT-B/32 vectors on the main side
 * (clip-text.js) and the 32 class slots are filled with the phrase + taxonomy
 * synonyms. Runs on WebGPU (fp32) or pure WASM (int8, ~13 MB) — no GPU/f16
 * needed, which is why it replaces the OWLv2 WASM floor.
 *
 * Contract (scripts/export-yolo-world.py): images[1,3,640,640] f32 +
 * txt_feats[1,32,512] f32 → output0[1,36,8400] = (cx,cy,w,h box at 640 +
 * 32 sigmoid class scores) per anchor. YOLOv8 head — NOT NMS-free, so the
 * caller thresholds, maps xywh→xyxy, and runs NMS (text-core). Boxes are
 * returned normalized [0,1] against the 640 square, matching the YOLOE lane.
 *
 * onnxruntime-web is resolved vendored-first, CDN fallback (mirrors yoloe-detect).
 */

import { YOLOE_INPUT } from './text-core.js'

const SCALES = new Set(['s', 'm', 'l', 'x'])
export const YW_CLASS_SLOTS = 32 // fixed head width (scripts/export-yolo-world.py NC)

const ORT_VERSION = '1.22.0'
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
const ORT_LOCAL = new URL('../lib/ort-web/', import.meta.url).href
const ORT_FILE = 'ort.webgpu.bundle.min.mjs'
// fp32 (~51 MB) on both EPs. The int8 build is smaller (~13 MB) but its
// ConvInteger nodes have no onnxruntime-web WASM kernel (ERROR_CODE 9), so it
// can't load there; a wasm-compatible requantize (QDQ / MatMulInteger) is TODO.
const modelURL = (scale) => new URL(`../models/yolo-world/yolo-world-${scale}.onnx`, import.meta.url).href

let ortPromise = null
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

let session = null
let sessionPromise = null
let loadedScale = null
let backend = null // 'webgpu' | 'wasm'

const buildSession = async (ort, scale) => {
    let lastErr
    // WASM first: it's the proven-correct path (validated maxScore 0.909). WebGPU
    // numerical parity for this model's contrastive matmul is not yet verified —
    // flip to ['webgpu','wasm'] once a GPU parity check passes.
    for (const ep of ['wasm', 'webgpu']) {
        try {
            const s = await ort.InferenceSession.create(modelURL(scale), {
                executionProviders: [ep],
                graphOptimizationLevel: 'all',
            })
            backend = ep
            return s
        } catch (err) { lastErr = err }
    }
    throw lastErr || new Error('yolo-world: no execution provider available')
}

export const loadYoloWorld = (scale = 's') => {
    if (!SCALES.has(scale)) scale = 's'
    if (session && loadedScale === scale) return Promise.resolve(session)
    if (sessionPromise && loadedScale === scale) return sessionPromise
    if (session) disposeYoloWorld() // scale switch — release first
    loadedScale = scale
    sessionPromise = (async () => {
        const ort = await loadOrt()
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
    idleTimer = setTimeout(() => { idleTimer = null; disposeYoloWorld() }, ms)
}

export const disposeYoloWorld = () => {
    cancelIdle()
    const s = session
    session = null
    sessionPromise = null
    loadedScale = null
    backend = null
    try { s?.release?.() } catch { /* already gone */ }
}

export const yoloWorldLoaded = () => !!session
export const yoloWorldBackend = () => backend

/**
 * Detect over `frame` — { data: RGB bytes, width, height } letterboxed into the
 * 640² square (top-left) by the caller — conditioned on `txtFeats`: a Float32Array
 * of YW_CLASS_SLOTS×512 L2-normalized CLIP vectors (clip-text.js). Returns
 * { dets: [{ box:[x0,y0,x1,y1] normalized [0,1] to the square, score, classIdx }],
 *   backend }. classIdx indexes the slot list the caller supplied. No NMS here.
 */
export const detectYoloWorld = async ({ frame, txtFeats, threshold = 0.25, scale = 's', dispose = false, idleMs = 0 }) => {
    cancelIdle()
    const ort = await loadOrt()
    const s = await loadYoloWorld(scale)
    const side = YOLOE_INPUT // 640, shared square
    try {
        // RGB bytes → NCHW float32 [0,1].
        const d = frame.data
        const plane = side * side
        const chw = new Float32Array(3 * plane)
        for (let i = 0, p = 0; p < plane; i += 3, p += 1) {
            chw[p] = d[i] / 255
            chw[plane + p] = d[i + 1] / 255
            chw[2 * plane + p] = d[i + 2] / 255
        }
        const images = new ort.Tensor('float32', chw, [1, 3, side, side])
        const txt = new ort.Tensor('float32', txtFeats, [1, YW_CLASS_SLOTS, 512])
        const out = await s.run({ images, txt_feats: txt })
        const o0 = out[s.outputNames[0]] // [1, 4+NC, 8400]
        const [, ch, n] = o0.dims
        const data = o0.data
        const nc = ch - 4
        const dets = []
        for (let a = 0; a < n; a += 1) {
            // Best class for this anchor (scores are already sigmoid).
            let best = 0
            let cls = 0
            for (let c = 0; c < nc; c += 1) {
                const v = data[(4 + c) * n + a]
                if (v > best) { best = v; cls = c }
            }
            if (best < threshold) continue
            const cx = data[a]; const cy = data[n + a]
            const w = data[2 * n + a]; const h = data[3 * n + a]
            const x0 = (cx - w / 2) / side
            const y0 = (cy - h / 2) / side
            const x1 = (cx + w / 2) / side
            const y1 = (cy + h / 2) / side
            dets.push({ box: [x0, y0, x1, y1], score: best, classIdx: cls })
        }
        return { dets, backend }
    } finally {
        if (dispose) disposeYoloWorld()
        else scheduleIdle(idleMs)
    }
}

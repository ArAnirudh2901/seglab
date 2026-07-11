/**
 * sam-client — main-thread API over the segmentation worker
 * -----------------------------------------------------------
 * Owns the worker lifecycle and exposes one call: segment(canvas, prompts).
 * If worker construction or the worker script itself fails (file:// pages,
 * exotic browsers), the engine runs inline on the main thread instead —
 * identical module, sticky choice, degrade-don't-die.
 *
 * Forwards the engine's broadcasts to subscribers:
 *   {type:'progress', detail:{lane, file, loaded, total, ...}}  downloads
 *   {type:'lane', label}   flagship hot-swap (app replays prompts on this)
 *   {type:'state'}         device/lane/timing chips should re-render
 */

import { summarizeMaskRGBA, validateClickMask } from './sam-core.js'

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

export const clientState = {
    device: null,   // 'webgpu' | 'wasm' once known (as reported by the engine)
    mode: null,     // 'worker' | 'inline' once decided
    lane: null,     // 'slimsam' | 'sam3' — which model served the last result
    ready: false,
    lastRun: null,  // { encodeMs, decodeMs, postMs, encoded, score, ms, lane }
}

const listeners = new Set()
export const subscribe = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }
const emit = (event) => { for (const cb of listeners) { try { cb(event) } catch { /* listener bug */ } } }

/** Engine broadcast → client event (shared by worker + inline paths). */
const onEngineEvent = (event) => {
    if (event?.type === 'lane') {
        clientState.lane = event.label
        emit({ type: 'lane', label: event.label })
        return
    }
    emit(event)
}

/* ─── Worker transport (with sticky inline fallback) ─────────────────────── */

let worker = null
let workerBroken = false
let seq = 0
const pending = new Map() // id → {resolve, reject}

const failAllPending = (reason) => {
    for (const [, entry] of pending) entry.reject(new Error(reason))
    pending.clear()
}

const getWorker = () => {
    if (workerBroken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./sam-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            if (data.type) {
                onEngineEvent(data)
                return
            }
            const entry = pending.get(data.id)
            if (!entry) return
            pending.delete(data.id)
            if (data.ok) entry.resolve(data.result)
            else entry.reject(new Error(data.error || 'on-device selection failed'))
        }
        worker.onerror = (event) => {
            // A worker-level error (script load failure, unhandled throw) is
            // not recoverable per-request: fail everything in flight and go
            // inline for the rest of the session.
            console.warn('[seglab] worker error; switching to inline engine:', event?.message)
            failAllPending(event?.message || 'selection worker crashed')
            try { worker.terminate() } catch { /* already dead */ }
            worker = null
            workerBroken = true
            clientState.mode = 'inline'
            emit({ type: 'state' })
        }
        clientState.mode = 'worker'
        return worker
    } catch (err) {
        console.warn('[seglab] worker construction failed; using inline engine:', err?.message)
        workerBroken = true
        clientState.mode = 'inline'
        return null
    }
}

let inlineSinkSet = false
let inlineEngine = null
const getInlineEngine = async () => {
    const engine = await import('./sam-engine.js')
    if (!inlineSinkSet) {
        engine.setEventSink(onEngineEvent)
        inlineSinkSet = true
    }
    inlineEngine = engine
    return engine
}

/**
 * Obsolete every in-flight/queued job older than `revision` (fire-and-forget
 * — cancellation must never queue behind the job it cancels). Call on any
 * prompt or document change; the engine skips stale post-processing and a
 * stale result can never commit.
 */
export const cancelBefore = (revision) => {
    if (worker) {
        try { worker.postMessage({ op: 'cancel', payload: { before: revision } }) } catch { /* dying worker */ }
    }
    // If neither transport has started, nothing is in flight to cancel.
    if (inlineEngine) inlineEngine.cancelBefore(revision)
}

/** One request over the worker, or inline when the worker is unavailable. */
const call = async (op, payload, transfer, timeoutMs, label) => {
    const w = getWorker()
    if (w) {
        const id = `${op}-${++seq}`
        const roundtrip = new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject })
            try {
                w.postMessage({ id, op, payload }, transfer || [])
            } catch (err) {
                pending.delete(id)
                reject(err)
            }
        })
        try {
            return await withTimeout(roundtrip, timeoutMs, label)
        } finally {
            // A timeout leaves the entry behind; a late worker reply to a
            // ghost id must not accumulate.
            pending.delete(id)
        }
    }
    const engine = await getInlineEngine()
    if (op === 'warm') return withTimeout(engine.warm(payload || {}), timeoutMs, label)
    if (op === 'segment' || op === 'hdExport') {
        const fn = op === 'segment' ? engine.segment : engine.hdRefine
        try {
            return await withTimeout(fn(payload), timeoutMs, label)
        } finally {
            // The worker shell closes transferred bitmaps; inline, that's ours.
            try { payload?.source?.close?.() } catch { /* already closed */ }
        }
    }
    if (op === 'detect') {
        const det = await import('./detect-engine.js')
        try {
            return await withTimeout(det.detect({
                source: payload.source,
                labels: payload.labels,
                threshold: payload.threshold,
                candidates: engine.detectorCandidates(),
                dispose: engine.getBudget().detectorDispose === 'now',
                progress_callback: (info) => onEngineEvent({ type: 'progress', detail: { lane: 'text', status: info?.status, file: info?.file, progress: info?.progress, loaded: info?.loaded, total: info?.total } }),
            }), timeoutMs, label)
        } finally {
            try { payload?.source?.close?.() } catch { /* already closed */ }
        }
    }
    throw new Error(`Unknown op: ${op}`)
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

let warmPromise = null

/** Download + compile the draft lane, then the flagship in the background
 *  (idempotent). `budget` is policy.js's resolved profile budget;
 *  `flagship:false` skips the background upgrade. */
export const warmUp = ({ flagship = true, budget = null } = {}) => {
    if (warmPromise) return warmPromise
    warmPromise = call('warm', { flagship, budget }, null, LOAD_TIMEOUT_MS, 'model load')
        .then((engineState) => {
            clientState.device = engineState?.device || clientState.device
            clientState.lane = engineState?.lane || clientState.lane
            clientState.ready = true
            emit({ type: 'state' })
            return clientState
        })
    warmPromise.catch(() => { warmPromise = null })
    return warmPromise
}

/**
 * Content key for the embedding cache: dims + FNV-1a over a 16×16
 * downsample. Content-addressed so the cache never serves stale embeddings
 * for new pixels. ~1 ms — negligible next to even a cached decode.
 */
const contentKey = (canvas) => {
    const c = document.createElement('canvas')
    c.width = 16
    c.height = 16
    const ctx = c.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(canvas, 0, 0, 16, 16)
    const px = ctx.getImageData(0, 0, 16, 16).data
    let h = 0x811c9dc5
    for (let i = 0; i < px.length; i += 1) {
        h ^= px[i]
        h = Math.imul(h, 0x01000193) >>> 0
    }
    // `doc:` namespaces whole-document embeddings; crop re-encodes will live
    // under `crop:${hash}:${rect}` and must never collide with these.
    return `doc:${canvas.width}x${canvas.height}:${h.toString(16)}`
}

/**
 * Run click/box/lasso selection fully on-device against `canvas` (the
 * canonical ≤1024 frame). All coordinates are canvas coordinates. The
 * engine returns the polished mask (clamp → hygiene → edge refinement) plus
 * the raw decoder mask for the UI's comparison toggle.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ clicks?: Array<[number, number, 0|1]>, box?: number[]|null,
 *           clampPoly?: Array<[number, number]>|null, clampMargin?: number,
 *           revision?: number }} prompts
 */
export const segment = async (canvas, { clicks = [], box = null, clampPoly = null, clampMargin = 0, revision } = {}) => {
    const startedAt = Date.now()
    if (!canvas?.width || !canvas?.height) throw new Error('Selection source has no usable dimensions')
    if ((!clicks || clicks.length === 0) && !box) throw new Error('No clicks or box to select with')

    const imageKey = contentKey(canvas)
    const buildPayload = async () => {
        // The bitmap transfers zero-copy into the worker; the worker closes it.
        const source = await createImageBitmap(canvas)
        return { payload: { imageKey, source, clicks, box, clampPoly, clampMargin, revision }, transfer: [source] }
    }

    let result
    try {
        const { payload, transfer } = await buildPayload()
        result = await call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection')
    } catch (err) {
        // The request that DISCOVERS a broken worker must not fail the
        // user's click: the bitmap transferred into the dead worker is gone,
        // so rebuild it and retry — call() now routes inline.
        if (!workerBroken) throw err
        console.warn('[seglab] retrying selection inline after worker failure')
        const { payload, transfer } = await buildPayload()
        result = await call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection (inline retry)')
    }
    // A cancelled job: no mask, no state churn — the caller just drops it.
    if (result?.stale) return { stale: true, revision: result.revision }
    clientState.device = result.device || clientState.device
    clientState.lane = result.lane || clientState.lane
    clientState.ready = true

    const toImageData = (buf) => new ImageData(
        buf instanceof Uint8ClampedArray ? buf : new Uint8ClampedArray(buf),
        result.width,
        result.height,
    )
    const imageData = toImageData(result.rgba)
    const rawImageData = toImageData(result.rawRgba)
    const summary = summarizeMaskRGBA(imageData.data, result.width, result.height)
    const verdict = validateClickMask(summary)

    clientState.lastRun = {
        encodeMs: result.encodeMs,
        decodeMs: result.decodeMs,
        postMs: result.postMs,
        encoded: result.encoded,
        score: result.score,
        lane: result.lane,
        ms: Date.now() - startedAt,
    }
    emit({ type: 'state' })
    return {
        imageData,
        rawImageData,
        width: result.width,
        height: result.height,
        score: result.score,
        summary,
        usable: verdict.usable,
        reason: verdict.reason,
        device: result.device,
        lane: result.lane,
        revision: result.revision,
        encoded: result.encoded,
        hygiene: result.hygiene,
        bandPixels: result.bandPixels,
        ms: Date.now() - startedAt,
    }
}

/**
 * Original-resolution alpha for one export crop. Thin transport over the
 * worker's `hdExport` op; export-hd.js owns bbox/padding/crop/composite. The
 * crop bitmap transfers zero-copy (worker closes it); the proxy-mask buffer
 * transfers too. Returns { alpha: Uint8ClampedArray, width, height, decoded }
 * or { stale:true }.
 */
export const hdExport = async (payload, transfer) => {
    const result = await call('hdExport', payload, transfer, INFER_TIMEOUT_MS, 'HD export')
    if (result?.stale) return { stale: true }
    const alpha = result.alpha instanceof Uint8ClampedArray ? result.alpha : new Uint8ClampedArray(result.alpha)
    return { alpha, width: result.width, height: result.height, decoded: result.decoded, lane: result.lane }
}

/** OWLv2 detection over `canvas` (already sized to the detector canvas).
 *  Returns { dets:[{box,score,label}] (canvas-pixel space), backend }. */
export const detectText = async (canvas, labels, { threshold = 0.02 } = {}) => {
    const source = await createImageBitmap(canvas)
    return call('detect', { source, labels, threshold }, [source], INFER_TIMEOUT_MS, 'Text detection')
}

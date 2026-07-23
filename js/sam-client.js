/**
 * sam-client — main-thread API over the segmentation worker
 * -----------------------------------------------------------
 * Owns the worker lifecycle and exposes one call: segment(canvas, prompts).
 * If worker CONSTRUCTION fails (file:// pages, exotic browsers), the engine
 * runs inline on the main thread — identical module, sticky choice. A worker
 * that dies MID-SESSION (usually memory) is restarted fresh instead (its
 * arenas are reclaimed); inline is the last resort after repeated deaths.
 *
 * Forwards the engine's broadcasts to subscribers:
 *   {type:'progress', detail:{lane, file, loaded, total, ...}}  downloads
 *   {type:'lane', label}   retained protocol field; SlimSAM is the only lane
 *   {type:'state'}         device/lane/timing chips should re-render
 */

import { summarizeMaskRGBA, validateClickMask } from './sam-core.js'
import { enqueueHeavy, cancelHeavyBefore, STALE } from './heavy-job-queue.js'

const LOAD_TIMEOUT_MS = 12 * 60 * 1000
const INFER_TIMEOUT_MS = 120 * 1000
// First text search may build a ~163 MB detector session and run its first
// wasm inference in one call — minutes on a weak machine, not a hang.
const DETECT_TIMEOUT_MS = 6 * 60 * 1000

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
    gpuInfo: null,  // { vendor, architecture, … } of the adapter actually in use
    mode: null,     // 'worker' | 'inline' once decided
    lane: null,     // 'slimsam' — the only interactive segmentation lane
    ready: false,
    lastRun: null,  // { encodeMs, decodeMs, postMs, encoded, score, ms, lane }
}

const listeners = new Set()
export const subscribe = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }
const emit = (event) => { for (const cb of listeners) { try { cb(event) } catch { /* listener bug */ } } }
const trace = (event, detail = {}) => console.log(`[seglab][client] ${event}`, detail)

/** Engine broadcast → client event (shared by worker + inline paths). */
const onEngineEvent = (event) => {
    if (event?.type !== 'progress') trace('engine-event', event)
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
let workerGeneration = 0 // bumped on every worker death (callers retry on change)
const MAX_WORKER_RESTARTS = 2
let seq = 0
const pending = new Map() // id → {resolve, reject}

const failAllPending = (reason) => {
    trace('pending-failed', { count: pending.size, reason })
    for (const [, entry] of pending) entry.reject(new Error(reason))
    pending.clear()
}

const getWorker = () => {
    if (workerBroken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./sam-worker.js', import.meta.url), { type: 'module' })
        trace('worker-created')
        worker.onmessage = (event) => {
            const data = event.data || {}
            if (data.type !== 'progress') {
                trace('worker-message', { id: data.id, type: data.type, ok: data.ok, error: data.error })
            }
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
            console.error('[seglab][client] worker-error', event?.message || event)
            // Worker died (usually memory). Terminating it is the memory
            // reset — the OS reclaims its WASM/GPU arenas — so RESTART a
            // fresh worker. Never fall inline mid-session: the full engine
            // on the main thread freezes the page and doubles renderer
            // memory (the proven tab-kill path). Inline stays boot-only.
            failAllPending(event?.message || 'selection worker crashed')
            try { worker.terminate() } catch { /* already dead */ }
            worker = null
            if (workerGeneration < MAX_WORKER_RESTARTS) {
                workerGeneration += 1
                console.warn('[seglab] worker crashed; restarting it fresh:', event?.message)
                warmPromise = null
                if (lastWarm) warmUp(lastWarm).catch(() => {})
            } else {
                console.warn('[seglab] worker crashed repeatedly; switching to inline engine:', event?.message)
                workerGeneration += 1
                workerBroken = true
                clientState.mode = 'inline'
            }
            emit({ type: 'state' })
        }
        clientState.mode = 'worker'
        return worker
    } catch (err) {
        console.error('[seglab][client] worker-create-failed', err)
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
    cancelHeavyBefore(revision) // queued-but-unstarted heavy jobs drop first
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
        seq += 1
        const id = `${op}-${seq}`
        trace('request', { id, op, label, transferCount: transfer?.length || 0 })
        const roundtrip = new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject })
            try {
                w.postMessage({ id, op, payload }, transfer || [])
            } catch (err) {
                pending.delete(id)
                console.error('[seglab][client] post-failed', { id, op, err })
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
    trace('inline-request', { op, label })
    const engine = await getInlineEngine()
    if (op === 'warm') return withTimeout(engine.warm(payload || {}), timeoutMs, label)
    if (op === 'segment' || op === 'hdExport' || op === 'encode') {
        const fn = op === 'segment' ? engine.segment : op === 'encode' ? engine.encodeImage : engine.hdRefine
        try {
            return await withTimeout(fn(payload), timeoutMs, label)
        } finally {
            // The worker shell closes transferred bitmaps; inline, that's ours.
            try { payload?.source?.close?.() } catch { /* already closed */ }
        }
    }
    if (op === 'pressure') return { freed: await engine.relievePressure(payload?.level || 1) }
    if (op === 'releaseDocument') { engine.releaseDocument(); return engine.getEngineState() }
    if (op === 'state') return engine.getEngineState()
    throw new Error(`Unknown op: ${op}`)
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

let warmPromise = null
let lastWarm = null // remembered so a restarted worker re-warms with the same budget

const applyWarmState = (engineState) => {
    if (!engineState || engineState.stale) {
        warmPromise = null // cancelled before it ran; the next caller retries
        return clientState
    }
    trace('warm-ready', engineState)
    clientState.device = engineState?.device || clientState.device
    clientState.gpuInfo = engineState?.gpuInfo || clientState.gpuInfo
    clientState.lane = engineState?.lane || clientState.lane
    clientState.ready = true
    emit({ type: 'state' })
    return clientState
}

/** Download and compile SlimSAM after an interaction proxy is visible.
 *  There is intentionally no model-upgrade branch: a second segmentation
 *  model would violate the bounded interaction-memory contract. */
export const warmUp = ({ budget = null } = {}) => {
    if (warmPromise) return warmPromise
    trace('warm-start', { model: 'slimsam', profile: budget?.profile })
    lastWarm = { budget }
    warmPromise = enqueueHeavy('model-warm', () => call('warm', { budget }, null, LOAD_TIMEOUT_MS, 'SlimSAM model load'))
        .then(applyWarmState)
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
export const segment = async (canvas, { clicks = [], box = null, clampPoly = null, clampMargin = 0, revision, budget = null } = {}) => {
    const startedAt = Date.now()
    if (!canvas?.width || !canvas?.height) throw new Error('Selection source has no usable dimensions')
    if ((!clicks || clicks.length === 0) && !box) throw new Error('No clicks or box to select with')

    const imageKey = contentKey(canvas)
    trace('segment-start', { imageKey, clicks: clicks.length, box: Boolean(box), revision })
    const buildPayload = async () => {
        // The bitmap transfers zero-copy into the worker; the worker closes it.
        const source = await createImageBitmap(canvas)
        return { payload: { imageKey, source, clicks, box, clampPoly, clampMargin, revision, budget }, transfer: [source] }
    }

    // One heavy job: a selection can never overlap a decode/refine/export.
    const result = await enqueueHeavy('segment', async () => {
        const gen = workerGeneration
        try {
            const { payload, transfer } = await buildPayload()
            return await call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection')
        } catch (err) {
            console.error('[seglab][client] segment-request-failed', err)
            // The click that DISCOVERS a dead worker must not fail: its bitmap
            // died with the worker, so rebuild and retry once — call() routes
            // to the restarted worker (or inline after repeated deaths).
            if (workerGeneration === gen) throw err
            console.warn('[seglab] retrying selection after worker crash')
            const { payload, transfer } = await buildPayload()
            return call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection (retry)')
        }
    }, { priority: 'high', revision: revision ?? null })
    // A cancelled job: no mask, no state churn — the caller just drops it.
    if (result === STALE || result?.stale) return { stale: true, revision: result?.revision ?? revision }
    trace('segment-result', { revision, lane: result?.lane, encoded: result?.encoded, ms: Date.now() - startedAt })
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
 * Idle-time image encode. `revision` lets an input event obsolete a queued
 * prewarm before it begins; an already-running kernel still completes safely
 * but reports stale and can never affect the UI. `encoded:false` means the
 * embedding came from memory or OPFS.
 */
export const encodeImage = async (canvas, { revision, prime = false, gpuOnly = false } = {}) => {
    if (!canvas?.width || !canvas?.height) return null
    const imageKey = contentKey(canvas)
    const result = await enqueueHeavy('encode-prewarm', async () => {
        const source = await createImageBitmap(canvas)
        return call('encode', { imageKey, source, revision, prime, gpuOnly }, [source], INFER_TIMEOUT_MS, 'Idle image encode')
    }, { priority: 'idle', revision: revision ?? null })
    return result === STALE ? { stale: true, revision } : result
}

/**
 * Original-resolution alpha for one export crop. Thin transport over the
 * worker's `hdExport` op; export-hd.js owns bbox/padding/crop/composite. The
 * crop bitmap transfers zero-copy (worker closes it); the proxy-mask buffer
 * transfers too. Export (compose) returns a Blob when emitBlob is set (encoded
 * in the worker, main thread stays flat) or a cutout buffer otherwise;
 * escalation returns the mask alpha. Plus width, height and decoded, or
 * { stale:true }.
 */
export const hdExport = async (payload, transfer) => {
    const result = await enqueueHeavy(
        'export-refine',
        () => call('hdExport', payload, transfer, INFER_TIMEOUT_MS, 'HD export'),
        { priority: 'high', revision: payload?.revision ?? null },
    )
    if (result === STALE || result?.stale) return { stale: true }
    const toU8C = (b) => (b instanceof Uint8ClampedArray ? b : new Uint8ClampedArray(b))
    const out = { width: result.width, height: result.height, decoded: result.decoded, lane: result.lane }
    if (result.blob) out.blob = result.blob
    if (result.cutout) out.cutout = toU8C(result.cutout)
    if (result.alpha) out.alpha = toU8C(result.alpha)
    return out
}

/* ── Disposable YOLO detect worker ───────────────────────────────────────────
 * Separate from sam-worker so its ORT wasm arena (which only grows) is freed by
 * TERMINATING the worker after the dispose window — the only true free. */
let detectWorker = null
let detectSeq = 0
const detectPending = new Map()
let detectIdleTimer = null

const disposeDetectWorker = () => {
    if (detectIdleTimer) { clearTimeout(detectIdleTimer); detectIdleTimer = null }
    const w = detectWorker
    detectWorker = null
    if (!w) return
    for (const [, e] of detectPending) e.reject(new Error('detect worker disposed'))
    detectPending.clear()
    try { w.terminate() } catch { /* already gone */ }
    trace('detect-worker-terminated')
}

const getDetectWorker = () => {
    if (detectWorker) return detectWorker
    const w = new Worker(new URL('./detect-worker.js', import.meta.url), { type: 'module' })
    w.onmessage = (event) => {
        const data = event.data || {}
        if (data.type === 'progress') { onEngineEvent(data); return }
        const entry = detectPending.get(data.id)
        if (!entry) return
        detectPending.delete(data.id)
        if (data.ok) entry.resolve(data.result)
        else entry.reject(new Error(data.error || 'text detection failed'))
    }
    w.onerror = (event) => {
        for (const [, e] of detectPending) e.reject(new Error(event?.message || 'detect worker crashed'))
        detectPending.clear()
        detectWorker = null
        try { w.terminate() } catch { /* dead */ }
    }
    detectWorker = w
    return w
}

/** One detect on the disposable worker; terminates it after `idleMs` (0 = now —
 *  the true wasm-arena free). Inline fallback when a worker can't be built. */
const callDetectWorker = async (payload, transfer, timeoutMs, label, idleMs) => {
    let w = null
    try { w = getDetectWorker() } catch { /* inline below */ }
    if (!w) {
        // No worker (e.g. Safari nested-worker limits): run the lane inline. The
        // ORT arena then lives on this thread, so dispose:true frees it now.
        if (payload.lane === 'yoloworld') {
            const [{ detectYoloWorld }, { embedSlots }] = await Promise.all([import('./yolo-world-detect.js'), import('./clip-text.js')])
            const emb = await embedSlots(payload.phrases)
            if (!emb) return { dets: [], slotNames: [], backend: null }
            const r = await withTimeout(detectYoloWorld({ frame: payload.frame, txtFeats: emb.txtFeats, threshold: payload.threshold, scale: payload.scale, dispose: true }), timeoutMs, label)
            return { ...r, slotNames: emb.slotNames }
        }
        const yoloe = await import('./yoloe-detect.js')
        return withTimeout(yoloe.detectYoloe({ ...payload, dispose: true }), timeoutMs, label)
    }
    if (detectIdleTimer) { clearTimeout(detectIdleTimer); detectIdleTimer = null }
    detectSeq += 1
    const id = `yoloe-${detectSeq}`
    const roundtrip = new Promise((resolve, reject) => {
        detectPending.set(id, { resolve, reject })
        try { w.postMessage({ id, payload }, transfer || []) } catch (err) { detectPending.delete(id); reject(err) }
    })
    try {
        return await withTimeout(roundtrip, timeoutMs, label)
    } finally {
        detectPending.delete(id)
        if (idleMs > 0) detectIdleTimer = setTimeout(disposeDetectWorker, idleMs)
        else disposeDetectWorker()
    }
}

/** YOLOE baked-vocab detection over a 640² letterboxed RGB frame (`frame.data`
 *  transfers). Runs in the disposable detect worker; `evict` drops the SAM
 *  embedding first so they never peak together; `idleMs` (0 = now) sets when the
 *  worker is terminated to reclaim its wasm arena. */
export const detectTextYoloe = async (frame, { scale = 's', threshold = 0.25, revision = null, idleMs = 0, evict = false } = {}) => {
    if (evict) await releaseDocument()
    const result = await enqueueHeavy(
        'detect',
        () => callDetectWorker({ lane: 'yoloe', frame, scale, threshold }, [frame.data.buffer], DETECT_TIMEOUT_MS, 'YOLOE detection', idleMs),
        { priority: 'normal', revision },
    )
    if (result === STALE) return { dets: [], backend: null, stale: true }
    return result
}

/** YOLO-World open-vocab detection over a 640² letterboxed RGB frame. `phrases`
 *  (the phrase + taxonomy synonyms) are CLIP-encoded in the worker to condition
 *  the vision head; returns { dets:[{box,score,classIdx}], slotNames, backend }
 *  where slotNames[classIdx] is the phrase a detection matched. */
export const detectTextYoloWorld = async (frame, phrases, { scale = 's', threshold = 0.25, revision = null, idleMs = 0, evict = false } = {}) => {
    if (evict) await releaseDocument()
    const result = await enqueueHeavy(
        'detect',
        () => callDetectWorker({ lane: 'yoloworld', frame, phrases, scale, threshold }, [frame.data.buffer], DETECT_TIMEOUT_MS, 'Open-vocab detection', idleMs),
        { priority: 'normal', revision },
    )
    if (result === STALE) return { dets: [], slotNames: [], backend: null, stale: true }
    return result
}

/** Drop every embedding for the outgoing document (model weights stay). */
export const releaseDocument = () => call('releaseDocument', {}, null, 30_000, 'release document').catch(() => null)

/** Engine residency snapshot (verify/debug): { cachedImages, lane, … }. */
export const engineState = () => call('state', {}, null, 30_000, 'engine state')

/** Free reloadable residents under memory pressure. */
export const relievePressure = async (level = 1) => {
    const res = await call('pressure', { level }, null, INFER_TIMEOUT_MS, 'relieve pressure')
    return res?.freed || []
}

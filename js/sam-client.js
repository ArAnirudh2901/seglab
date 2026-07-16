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
 *   {type:'lane', label}   opt-in flagship hot-swap (app replays prompts)
 *   {type:'state'}         device/lane/timing chips should re-render
 */

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
            else {
                const err = new Error(data.error || 'on-device selection failed')
                if (data.stale) err.stale = true
                entry.reject(err)
            }
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
const getInlineEngine = async () => {
    const engine = await import('./sam-engine.js')
    if (!inlineSinkSet) {
        engine.setEventSink(onEngineEvent)
        inlineSinkSet = true
    }
    return engine
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
    if (op === 'warm') return withTimeout(engine.warm(), timeoutMs, label)
    if (op === 'flagship-start') return withTimeout(engine.startFlagship(), timeoutMs, label)
    if (op === 'reset') return engine.resetImage(payload?.revision || 0)
    if (op === 'state') return engine.getEngineState()
    if (op === 'segment') {
        try {
            return await withTimeout(engine.segment(payload), timeoutMs, label)
        } finally {
            // The worker shell closes transferred bitmaps; inline, that's ours.
            try { payload?.source?.close?.() } catch { /* already closed */ }
        }
    }
    throw new Error(`Unknown op: ${op}`)
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

let warmPromise = null

/** Download + compile the draft (SlimSAM) lane — idempotent. The flagship
 *  lane is never touched here; see startFlagship (explicit opt-in only). */
export const warmUp = () => {
    if (warmPromise) return warmPromise
    warmPromise = call('warm', {}, null, LOAD_TIMEOUT_MS, 'model load')
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
 * EXPLICIT OPT-IN: load the SAM3 flagship lane. Only ever called from a
 * confirmed user gesture in the UI — never automatically, never from a URL
 * flag. Resolves to the engine state (flagship: 'ready'|'failed'|...).
 */
export const startFlagship = () => call('flagship-start', {}, null, LOAD_TIMEOUT_MS, 'SAM3 opt-in load')
    .then((engineState) => {
        clientState.device = engineState?.device || clientState.device
        clientState.lane = engineState?.lane || clientState.lane
        emit({ type: 'state' })
        return engineState
    })

/** New document: raise the engine's revision floor, free the embedding slot. */
export const resetImage = (revision) => call('reset', { revision }, null, 30_000, 'engine reset')
    .catch(() => { /* worker may be cold — nothing to release */ })

/** Engine internals snapshot (residentEmbeddings, encodeCount — verify). */
export const getEngineSnapshot = () => call('state', {}, null, 30_000, 'engine state')

/**
 * Run click/box/lasso selection fully on-device against `canvas` (the
 * bounded proxy frame). All coordinates are canvas coordinates. The engine
 * returns the polished one-channel mask (clamp → hygiene → edge refinement)
 * plus the raw decoder mask for the UI's comparison toggle.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ clicks?: Array<[number, number, 0|1]>, box?: number[]|null,
 *           clampPoly?: Array<[number, number]>|null, clampMargin?: number }} prompts
 * @param {{ revision?: number, post?: 'js'|'clamp-only' }} opts  staleness +
 *   post-pipeline mode ('clamp-only' when a cv-refine stage follows)
 */
export const segment = async (canvas, { clicks = [], box = null, clampPoly = null, clampMargin = 0 } = {}, { revision = 0, post = 'js' } = {}) => {
    const startedAt = Date.now()
    if (!canvas?.width || !canvas?.height) throw new Error('Selection source has no usable dimensions')
    if ((!clicks || clicks.length === 0) && !box) throw new Error('No clicks or box to select with')

    const buildPayload = async () => {
        // The bitmap transfers zero-copy into the worker; the worker closes it.
        const source = await createImageBitmap(canvas)
        return { payload: { revision, post, source, clicks, box, clampPoly, clampMargin }, transfer: [source] }
    }

    let result
    try {
        const { payload, transfer } = await buildPayload()
        result = await call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection')
    } catch (err) {
        // The request that DISCOVERS a broken worker must not fail the
        // user's click: the bitmap transferred into the dead worker is gone,
        // so rebuild it and retry — call() now routes inline.
        if (!workerBroken || err?.stale) throw err
        console.warn('[seglab] retrying selection inline after worker failure')
        const { payload, transfer } = await buildPayload()
        result = await call('segment', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection (inline retry)')
    }
    clientState.device = result.device || clientState.device
    clientState.lane = result.lane || clientState.lane
    clientState.ready = true

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
        alpha: result.alpha,
        alphaRaw: result.alphaRaw,
        width: result.width,
        height: result.height,
        score: result.score,
        summary: result.summary,
        usable: result.usable,
        reason: result.reason,
        device: result.device,
        lane: result.lane,
        encoded: result.encoded,
        hygiene: result.hygiene,
        bandPixels: result.bandPixels,
        ms: Date.now() - startedAt,
    }
}

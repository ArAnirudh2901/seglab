/**
 * engine-client — main-thread API over the segmentation worker
 * ---------------------------------------------------------------
 * Owns the worker lifecycle and exposes one call: select(canvas, request).
 * If worker construction or the worker script itself fails (file:// pages,
 * exotic browsers), the engine runs inline on the main thread instead —
 * identical module, sticky choice, degrade-don't-die.
 *
 * Forwards the engine's broadcasts to subscribers:
 *   {type:'progress', detail:{lane, file, loaded, total, ...}}  model download
 *   {type:'state'}   device/timing chips should re-render
 */

import { summarizeMaskRGBA, validateClickMask } from './mask-core.js'

const LOAD_TIMEOUT_MS = 10 * 60 * 1000
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
    device: null,   // 'webgpu' | 'wasm' once known
    mode: null,     // 'worker' | 'inline' once decided
    ready: false,
    detected: 0,    // instances the model found in the current photo
    lastRun: null,  // { analyzeMs, selectMs, postMs, analyzed, ms }
}

const listeners = new Set()
export const subscribe = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }
const emit = (event) => { for (const cb of listeners) { try { cb(event) } catch { /* listener bug */ } } }

/* ─── Worker transport (with sticky inline fallback) ─────────────────────── */

let worker = null
let workerBroken = false
let seq = 0
const pending = new Map()

const failAllPending = (reason) => {
    for (const [, entry] of pending) entry.reject(new Error(reason))
    pending.clear()
}

const getWorker = () => {
    if (workerBroken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./engine-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            if (data.type) { emit(data); return }
            const entry = pending.get(data.id)
            if (!entry) return
            pending.delete(data.id)
            if (data.ok) entry.resolve(data.result)
            else entry.reject(new Error(data.error || 'on-device selection failed'))
        }
        worker.onerror = (event) => {
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
    const engine = await import('./engine.js')
    if (!inlineSinkSet) {
        engine.setEventSink((e) => emit(e))
        inlineSinkSet = true
    }
    return engine
}

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
            pending.delete(id)
        }
    }
    const engine = await getInlineEngine()
    if (op === 'warm') return withTimeout(engine.warm(), timeoutMs, label)
    if (op === 'select' || op === 'list') {
        const run = op === 'select' ? engine.select : engine.listInstances
        try {
            return await withTimeout(run(payload), timeoutMs, label)
        } finally {
            // The worker shell closes transferred bitmaps; inline, that's ours.
            try { payload?.source?.close?.() } catch { /* already closed */ }
        }
    }
    throw new Error(`Unknown op: ${op}`)
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

let warmPromise = null

/** Download + compile the model (idempotent). */
export const warmUp = () => {
    warmPromise ??= call('warm', null, null, LOAD_TIMEOUT_MS, 'model load')
        .then((s) => {
            clientState.device = s?.device || clientState.device
            clientState.ready = s?.ready ?? true
            emit({ type: 'state' })
            return clientState
        })
    warmPromise.catch(() => { warmPromise = null })
    return warmPromise
}

/**
 * Content key for the analysis cache: dims + FNV-1a over a 16×16 downsample.
 * Content-addressed so the cache never serves stale instances for new pixels.
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
    return `${canvas.width}x${canvas.height}:${h.toString(16)}`
}

/**
 * Select on-device against `canvas` (the canonical ≤1024 frame). All
 * coordinates are canvas coordinates.
 *
 * The FIRST call on a photo runs the model; every later call — whatever the
 * mode — is a pure query against the cached instance set, so switching
 * between clicking, dragging and describing costs nothing.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ mode: 'click'|'box'|'lasso'|'text'|'all', point?: number[],
 *           box?: number[], poly?: Array<number[]>, query?: string,
 *           exclude?: number[] }} request
 */
export const select = async (canvas, request) => {
    const startedAt = Date.now()
    if (!canvas?.width || !canvas?.height) throw new Error('Selection source has no usable dimensions')
    if (!request?.mode) throw new Error('No selection mode given')

    const imageKey = contentKey(canvas)
    const buildPayload = async () => {
        // The bitmap transfers zero-copy into the worker; the worker closes it.
        const source = await createImageBitmap(canvas)
        return { payload: { ...request, imageKey, source }, transfer: [source] }
    }

    let result
    try {
        const { payload, transfer } = await buildPayload()
        result = await call('select', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection')
    } catch (err) {
        // The request that DISCOVERS a broken worker must not fail the user's
        // interaction: the bitmap transferred into the dead worker is gone,
        // so rebuild it and retry — call() now routes inline.
        if (!workerBroken) throw err
        console.warn('[seglab] retrying selection inline after worker failure')
        const { payload, transfer } = await buildPayload()
        result = await call('select', payload, transfer, INFER_TIMEOUT_MS, 'On-device selection (inline retry)')
    }

    clientState.device = result.device || clientState.device
    clientState.ready = true
    clientState.detected = result.detected
    clientState.lastRun = {
        analyzeMs: result.analyzeMs,
        maskMs: result.maskMs,
        selectMs: result.selectMs,
        postMs: result.postMs,
        analyzed: result.analyzed,
        instances: result.instances.length,
        detected: result.detected,
        ms: Date.now() - startedAt,
    }

    const toImageData = (buf) => (buf
        ? new ImageData(buf instanceof Uint8ClampedArray ? buf : new Uint8ClampedArray(buf), result.width, result.height)
        : null)
    const imageData = toImageData(result.rgba)
    const rawImageData = toImageData(result.rawRgba)
    const summary = imageData ? summarizeMaskRGBA(imageData.data, result.width, result.height) : null
    const verdict = summary ? validateClickMask(summary) : { usable: false, reason: result.reason }

    emit({ type: 'state' })
    return {
        imageData,
        rawImageData,
        width: result.width,
        height: result.height,
        summary,
        instances: result.instances,
        indices: result.indices,
        detected: result.detected,
        concepts: result.concepts,
        mode: result.mode,
        query: result.query,
        usable: result.instances.length > 0 && verdict.usable,
        reason: result.reason || verdict.reason,
        device: result.device,
        analyzed: result.analyzed,
        analyzeMs: result.analyzeMs,
        selectMs: result.selectMs,
        postMs: result.postMs,
        ms: Date.now() - startedAt,
    }
}

/** Everything the model found — for the instance browser / "select all". */
export const listInstances = async (canvas) => {
    const imageKey = contentKey(canvas)
    const source = await createImageBitmap(canvas)
    return call('list', { imageKey, source }, [source], INFER_TIMEOUT_MS, 'Analyze')
}

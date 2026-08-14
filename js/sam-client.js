/**
 * sam-client — main-thread API over the segmentation lane
 * -----------------------------------------------------------
 * ONE path: every selection, prewarm and export goes to the SAM 2.1 host
 * (js/sam21-host.js), a SharedWorker holding a single ORT session for all tabs.
 * There is no second engine, no transport choice and no lane switch — the
 * routing that used to live here (worker vs inline vs SlimSAM) is gone, and
 * with it the states where the app could not say which model produced a mask.
 *
 * Everything here is transport and post-processing. The lane's own fallback
 * (no WebGPU device at encode time -> CPU tensor) is internal to sam21-lane.js
 * and needs nothing from this layer.
 *
 * Forwards broadcasts to subscribers:
 *   {type:'progress', detail:{lane, file, loaded, total, ...}}  downloads
 *   {type:'state'}         device/lane/timing chips should re-render
 */

import { summarizeMaskRGBA, validateClickMask } from './sam-core.js'
import { enqueueHeavy, cancelHeavyBefore, onHeavyActivity, STALE } from './heavy-job-queue.js'
import { sam21Candidates, sam21Cycle, sam21HdCompose, sam21Segment } from './sam21-adapter.js'
import { LANE } from './sam21-lane.js'
import { noteModel } from './model-registry.js'

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

// device/mode stay null until something has actually been built. The lane is
// WebGPU-only and there is one transport, but "what we will use" and "what we
// have connected" are different claims, and boot must not assert the second.
export const clientState = {
    device: null,   // 'webgpu' once a host is up
    gpuInfo: null,  // { vendor, architecture, … } of the adapter actually in use
    mode: null,     // 'shared-worker' | 'dedicated-worker' once connected
    lane: LANE,     // the only interactive segmentation lane
    ready: false,
    lastRun: null,  // { encodeMs, decodeMs, postMs, encoded, score, ms, lane }
}

const listeners = new Set()
export const subscribe = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }
const emit = (event) => { for (const cb of listeners) { try { cb(event) } catch { /* listener bug */ } } }
const trace = (event, detail = {}) => console.log(`[seglab][client] ${event}`, detail)

/** Worker broadcast → client event. Download progress only; the lane label is
 *  a constant now, so there is nothing left to negotiate at runtime. */
const onWorkerEvent = (event) => {
    if (event?.type !== 'progress') trace('worker-event', event)
    emit(event)
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

/**
 * Obsolete every in-flight/queued job older than `revision` (fire-and-forget —
 * cancellation must never queue behind the job it cancels). Call on any prompt
 * or document change; a stale result can never commit.
 */
export const cancelBefore = (revision) => cancelHeavyBefore(revision)

// Detector runs, RAW decodes and exports never touch the SAM host, so the host
// cannot see them. Forward the tab's busy edge; it holds its weight prefetch.
onHeavyActivity((busy) => {
    import('./sam21-client.js').then((c) => c.noteBusy(busy)).catch(() => null)
})

let warmPromise = null

/** Build the decoder once an interaction proxy is visible. The encoder is the
 *  expensive half and stays unbuilt until there is an actual image to encode.
 *  There is intentionally no model-upgrade branch: a second segmentation model
 *  would violate the bounded interaction-memory contract. */
export const warmUp = ({ speculative = false } = {}) => {
    if (warmPromise) return warmPromise
    trace('warm-start', { model: LANE, speculative })
    // A speculative (boot) warm yields to user work: lowest rank, and an
    // `isCurrent` so clearHeavyQueue can cancel it while still queued rather
    // than making an import decode wait behind a warm nobody asked for.
    warmPromise = enqueueHeavy('model-warm', async () => {
        const c = await import('./sam21-client.js')
        await c.hello('seglab')
        await c.warm()
        clientState.mode = c.hostMode() === 'shared' ? 'shared-worker' : 'dedicated-worker'
        clientState.device = 'webgpu'
        clientState.ready = true
        noteModel('sam21', { device: 'webgpu', scale: 'small', release: 'darktable-5.6.0' })
        emit({ type: 'state' })
        return clientState
    }, speculative ? { priority: 'idle', isCurrent: () => true } : {})
    // STALE means cancelled before it ran. Drop the memo or every later warmUp
    // hands back a resolved promise that never built anything.
    warmPromise = warmPromise.then((r) => {
        if (r === STALE) warmPromise = null
        return r === STALE ? clientState : r
    })
    warmPromise.catch(() => { warmPromise = null })
    return warmPromise
}

let encoderPromise = null
let encoderBuilt = false

/** True once the encoder SESSION exists, so an encode costs only its forward
 *  pass. The eager path skips its settle gate on this — the gate guards session
 *  create, and that is already paid. */
export const encoderReady = () => encoderBuilt

/**
 * Build the encoder session ahead of any image. The ~1 GB half and the shader
 * compile that otherwise land on the first click. Speculative callers yield to
 * user work; the lane holds the session until the first encode arms its idle
 * release, so it is still standing when a photo arrives.
 */
let trackingLane = false
export const warmEncoder = ({ speculative = false } = {}) => {
    if (encoderPromise) return encoderPromise
    trace('encoder-warm-start', { speculative })
    encoderPromise = enqueueHeavy('encoder-warm', async () => {
        const c = await import('./sam21-client.js')
        await c.hello('seglab')
        // The lane drops the session on its own — post-encode idle release,
        // governor shed, host exit — and only its broadcasts see that. Track
        // them or encoderReady() goes on claiming a session that is gone.
        if (!trackingLane) {
            trackingLane = true
            c.subscribe((s) => { if (s?.lane) encoderBuilt = Boolean(s.lane.encoder) })
        }
        const r = await c.buildEncoder()
        encoderBuilt = Boolean(r?.encoder ?? true)
        emit({ type: 'state' })
        return r
    }, speculative ? { priority: 'idle', isCurrent: () => true } : {})
    encoderPromise = encoderPromise.then((r) => {
        if (r === STALE) encoderPromise = null
        return r
    })
    encoderPromise.catch(() => { encoderPromise = null })
    return encoderPromise
}

/** The governor and the lane's idle ladder both drop the session behind our
 *  back; re-arm so a later boot-style warm can rebuild it. */
export const forgetEncoder = () => { encoderBuilt = false; encoderPromise = null }

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
 * canonical ≤1024 frame). All coordinates are canvas coordinates. The lane
 * returns the polished mask (clamp → hygiene → edge refinement) plus the raw
 * decoder mask for the UI's comparison toggle.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ clicks?: Array<[number, number, 0|1]>, box?: number[]|null,
 *           revision?: number }} prompts
 */
export const segment = async (canvas, { clicks = [], box = null, revision } = {}) => {
    const startedAt = Date.now()
    if (!canvas?.width || !canvas?.height) throw new Error('Selection source has no usable dimensions')
    if ((!clicks || clicks.length === 0) && !box) throw new Error('No clicks or box to select with')

    const imageKey = contentKey(canvas)
    trace('segment-start', { imageKey, clicks: clicks.length, box: Boolean(box), revision })

    // The lane's own host serialises across tabs, but a selection still takes a
    // slot here so it cannot overlap an export/refine in THIS tab.
    const r = await enqueueHeavy('segment', () => {
        // Eviction has to run BOTH ways. detectText drops the SAM embedding
        // before a search, but nothing dropped the detector before a selection —
        // so in the ordinary search → click → segment flow the detector stayed
        // resident (120 s idle window) through SAM's ~1 GB encoder build.
        // enqueueHeavy serialises execution, not residency: max() on compute,
        // sum() on memory. It must fire INSIDE the queued task, or it would
        // terminate a detect that is still in flight.
        disposeDetectWorker()
        return sam21Segment({
            canvas, imageKey, clicks, box, onWait: (w) => emit({ type: 'waiting', ...w }),
        })
    }, { priority: 'high', revision: revision ?? null }).catch(async (err) => {
        // The queue's watchdog fired, so the host is wedged (or its GPU work is)
        // and every later click would ride the same corpse. Tear it down: the
        // next selection connects to a fresh instance and re-encodes from OPFS.
        // Fire-and-forget — recovery must never become the thing that hangs.
        if (/timed out/i.test(String(err?.message))) {
            console.warn('[seglab][client] selection timed out; recycling the host')
            import('./sam21-client.js').then((c) => c.op('shutdown')).catch(() => null)
        }
        throw err
    })
    // A cancelled job: no mask, no state churn — the caller just drops it.
    if (r === STALE || r?.stale) return { stale: true, revision: r?.revision ?? revision }
    return finishSegment(r, revision, startedAt)
}

/** Mask buffers → ImageData, summarise, validate, publish state. */
function finishSegment(result, revision, startedAt) {
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
        // What postMs was spent ON — the refined band, not the proxy. hardware-fit
        // normalises by this, so it has to travel with the timing that produced it.
        bandPixels: result.bandPixels || 0,
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
        candidates: result.candidates,
        pick: result.pick,
        cycled: result.cycled,
        ms: Date.now() - startedAt,
    }
}

/**
 * Step through SAM's other candidates for the selection already on screen.
 *
 * Deliberately NOT queued and NOT async: the planes are in memory, so this is
 * an upsample + guided filter, and routing it through enqueueHeavy would make a
 * repaint wait behind the decode it exists to replace. Returns the same shape
 * as `segment`, or null when there is nothing to cycle.
 */
export const cycleCandidate = (delta = 1) => {
    const r = sam21Cycle(delta)
    return r ? finishSegment(r, null, Date.now()) : null
}

/**
 * Idle-time image encode. `revision` lets an input event obsolete a queued
 * prewarm before it begins; an already-running kernel still completes safely
 * but reports stale and can never affect the UI. `encoded:false` means the
 * embedding came from memory or OPFS.
 */
export const encodeImage = async (canvas, { revision } = {}) => {
    if (!canvas?.width || !canvas?.height) return null
    const imageKey = contentKey(canvas)
    return enqueueHeavy('encode-prewarm', async () => {
        disposeDetectWorker() // the encoder is the app's largest allocation
        const { encodeImage: enc, hello } = await import('./sam21-client.js')
        await hello('seglab')
        const bitmap = await createImageBitmap(canvas)
        const r = await enc(bitmap, imageKey)
        return { encoded: !r.cached, imageKey, device: 'webgpu', lane: LANE }
    }, { priority: 'idle', revision: revision ?? null }).catch(() => null)
}

/**
 * Original-resolution alpha for one export crop. export-hd.js owns
 * bbox/padding/crop/composite; the alpha comes from the lane's continuous field
 * at native resolution and the composite is plain canvas work. Returns a Blob
 * when emitBlob is set, a cutout buffer otherwise; escalation returns the mask
 * alpha. Plus width, height and decoded, or { stale:true }.
 *
 * No fallback path: this used to drop through to the retired engine's worker,
 * which meant building a whole transformers + ORT stack just to multiply pixels
 * — wasteful, and it crashed doing it. A failure here is a real failure.
 */
export const hdExport = async (payload) => {
    if (!payload?.source || !payload?.proxySubrect) {
        throw new Error('hdExport needs a source bitmap and a proxy subrect')
    }
    const out = await enqueueHeavy('export-refine', () => sam21HdCompose({
        bitmap: payload.source,
        imageKey: payload.imageKey || null,
        subrect: payload.proxySubrect,
        compose: payload.compose,
        emitBlob: payload.emitBlob,
        doDecode: payload.doDecode,
        cropKey: payload.cropKey,
        prompts: payload.prompts,
    }), { priority: 'high', revision: payload?.revision ?? null })
    if (out === STALE || out?.stale) return { stale: true }
    try { payload.source.close?.() } catch { /* already gone */ }
    return out
}

/* ── Disposable YOLO detect worker ───────────────────────────────────────────
 * Separate from sam-worker so its ORT wasm arena (which only grows) is freed by
 * TERMINATING the worker after the dispose window — the only true free. */
let detectWorker = null
let detectSeq = 0
const detectPending = new Map()
let detectIdleTimer = null

// A worker killed for exceeding the host's per-process memory ceiling usually
// dies WITHOUT firing onerror, so the roundtrip promise never settles. Measured
// on Safari 26.6: this lane climbs to ~1.03 GB, the web process is killed, and
// the tab sits on "Selecting…" until DETECT_TIMEOUT_MS — six minutes, which is
// indistinguishable from a hang. That ceiling has to stay generous because a
// first-run model pull legitimately takes minutes on a slow link, so the timeout
// is the wrong instrument. Progress events tick while bytes are moving, so
// SILENCE is what separates "still downloading" from "the worker is gone".
const DETECT_SILENCE_MS = 45 * 1000
let detectLastSignal = 0
let detectWatchdog = null

const noteDetectSignal = () => { detectLastSignal = Date.now() }

const failDetect = (message, code) => {
    if (!detectPending.size) return
    const err = new Error(message)
    err.code = code
    for (const [, e] of detectPending) e.reject(err)
    detectPending.clear()
}

const stopDetectWatchdog = () => {
    if (detectWatchdog) { clearInterval(detectWatchdog); detectWatchdog = null }
}

const startDetectWatchdog = () => {
    if (detectWatchdog) return
    noteDetectSignal()
    detectWatchdog = setInterval(() => {
        if (!detectPending.size || Date.now() - detectLastSignal < DETECT_SILENCE_MS) return
        trace('detect-worker-silent')
        failDetect('the detection worker stopped responding — it was most likely killed for memory', 'detect-worker-died')
        disposeDetectWorker()
    }, 5000)
}

const disposeDetectWorker = () => {
    if (detectIdleTimer) { clearTimeout(detectIdleTimer); detectIdleTimer = null }
    stopDetectWatchdog()
    const w = detectWorker
    detectWorker = null
    if (!w) return
    failDetect('detect worker disposed')
    try { w.terminate() } catch { /* already gone */ }
    trace('detect-worker-terminated')
}

const getDetectWorker = () => {
    if (detectWorker) return detectWorker
    const w = new Worker(new URL('./detect-worker.js', import.meta.url), { type: 'module' })
    w.onmessage = (event) => {
        noteDetectSignal()
        const data = event.data || {}
        if (data.type === 'progress') { onWorkerEvent(data); return }
        const entry = detectPending.get(data.id)
        if (!entry) return
        detectPending.delete(data.id)
        if (data.ok) entry.resolve(data.result)
        else entry.reject(new Error(data.error || 'text detection failed'))
    }
    w.onerror = (event) => {
        failDetect(event?.message || 'detect worker crashed', 'detect-worker-died')
        stopDetectWatchdog()
        detectWorker = null
        try { w.terminate() } catch { /* dead */ }
    }
    detectWorker = w
    return w
}

/** One detect on the disposable worker; terminates it after `idleMs` (0 = now —
 *  the true wasm-arena free). Inline fallback when a worker can't be built. */
const callDetectWorker = async (payload, transfer, timeoutMs, label, idleMs, keepAlive = false) => {
    let w = null
    try { w = getDetectWorker() } catch { /* inline below */ }
    if (!w) {
        // No worker (e.g. Safari nested-worker limits): run the lane inline. The
        // ORT arena then lives on THIS thread and nothing can terminate it, so
        // both sessions are disposed explicitly and the encoder still runs
        // before the detector is built.
        const [{ detectYoloe, DIM }, { encodePhrases }] = await Promise.all([
            import('./yoloe-detect.js'), import('./text-encode.js'),
        ])
        const phrases = payload.phrases || []
        if (!phrases.length) return { results: [], slotNames: [], backend: null, learned: [] }
        const known = new Map(payload.known || [])
        const misses = phrases.filter((p) => !known.has(p))
        const learned = []
        if (misses.length) {
            const { vectors } = await encodePhrases(misses)
            misses.forEach((p, i) => {
                const vec = vectors.slice(i * DIM, (i + 1) * DIM)
                known.set(p, vec)
                learned.push([p, vec])
            })
        }
        const txtFeats = new Float32Array(phrases.length * DIM)
        phrases.forEach((p, i) => { const v = known.get(p); if (v) txtFeats.set(v, i * DIM) })
        const results = []
        let backend = null
        let inferMs = 0
        const all = payload.frames || []
        for (let i = 0; i < all.length; i += 1) {
            const r = await withTimeout(detectYoloe({
                frame: all[i], txtFeats, threshold: payload.threshold,
                dispose: i === all.length - 1, // only the last one frees the arena
            }), timeoutMs, label)
            results.push(r.dets)
            backend = r.backend
            inferMs += r.inferMs || 0
        }
        return { results, slotNames: phrases, backend, learned, inferMs, cells: results.length }
    }
    if (detectIdleTimer) { clearTimeout(detectIdleTimer); detectIdleTimer = null }
    detectSeq += 1
    const id = `yoloe-${detectSeq}`
    const roundtrip = new Promise((resolve, reject) => {
        detectPending.set(id, { resolve, reject })
        startDetectWatchdog()
        try { w.postMessage({ id, payload }, transfer || []) } catch (err) { detectPending.delete(id); reject(err) }
    })
    try {
        return await withTimeout(roundtrip, timeoutMs, label)
    } finally {
        detectPending.delete(id)
        // keepAlive: the caller has another pass of the SAME search coming and
        // will dispose itself. Without it a two-pass (escalated) search under
        // 'dispose now' rebuilds the whole YOLOE session between its own halves.
        if (keepAlive) return
        if (idleMs > 0) detectIdleTimer = setTimeout(disposeDetectWorker, idleMs)
        else disposeDetectWorker()
    }
}

/** Terminate the detect worker now — the only true free of its ORT arena.
 *  Exported for a caller holding it across the passes of one search. */
export const disposeDetector = () => disposeDetectWorker()

/* What a live detect worker costs the app, for the governor's ledger — on
 * WebKit the ledger is the ONLY input, and the text lane was entirely absent
 * from it, so a lane that peaks near a gigabyte was invisible to the one
 * engine with no byte API. Measured in the detect process (DESIGN-TEXT-LANE
 * §"YOLOE lane, isolated"): 326 MB idle → 962 MB session built → 1034 MB first
 * run, and disposing the SESSION only returns the GPU share (854 MB floor) —
 * the arena goes back when the WORKER is terminated, which is why residency is
 * keyed on the worker, not on the session. Safari runs the same lane heavier
 * (~1.03 GB observed before a process reap), so it gets the pessimistic figure.
 */
// Same WebKit signal the lane uses (sam21-lane.js §IS_WEBKIT): vendor is
// 'Apple Computer, Inc.' on every WebKit browser including iOS Chrome, and ''
// on Gecko — a UA sniff for 'safari' would miss the first and catch neither.
const IS_WEBKIT = typeof navigator !== 'undefined'
    && !navigator.userAgentData
    && /apple/i.test(navigator.vendor || '')
const DETECT_RESIDENT_MB = IS_WEBKIT ? 1030 : 700

/** MB the live detect worker is holding (0 when it is not up). Ledger input. */
export const detectorResidentMB = () => (detectWorker ? DETECT_RESIDENT_MB : 0)

/** Free the detect worker for a shed, but never out from under a running
 *  search — terminating mid-detection rejects the in-flight call, which the
 *  user sees as a failed search rather than as memory relief. Returns whether
 *  it actually freed anything. */
export const disposeDetectorIfIdle = () => {
    if (!detectWorker || detectPending.size) return false
    disposeDetectWorker()
    return true
}

/**
 * Open-vocabulary detection over a 640² letterboxed RGB frame (`frame.data`
 * transfers). `phrases` fill the detector's class slots in order, so
 * `slotNames[classIdx]` is the phrase a detection matched.
 *
 * Phrase vectors are looked up in the durable cache FIRST and passed down as
 * `known`; when every phrase hits, the worker never builds the text encoder at
 * all. Newly encoded vectors come back in `learned` and are persisted here.
 *
 * `evict` drops the SAM embedding first so the two lanes never peak together;
 * `idleMs` (0 = now) sets when the worker is terminated.
 */
export const detectText = async (frames, phrases, {
    threshold = 0.25, revision = null, idleMs = 0, evict = false, keepAlive = false,
} = {}) => {
    if (evict) await releaseDocument()
    const { lookupPhrases, rememberPhrases } = await import('./text-embed-store.js')
    const { hits } = await lookupPhrases(phrases)
    const known = [...hits.entries()]
    const result = await enqueueHeavy(
        'detect',
        () => callDetectWorker(
            { lane: 'text', frames, phrases, known, threshold },
            frames.map((f) => f.data.buffer), DETECT_TIMEOUT_MS, 'Open-vocab detection', idleMs, keepAlive,
        ),
        { priority: 'normal', revision },
    )
    if (result === STALE) return { results: [], slotNames: [], backend: null, stale: true }
    if (result?.learned?.length) {
        const words = result.learned.map(([p]) => p)
        const flat = new Float32Array(words.length * 512)
        result.learned.forEach(([, v], i) => flat.set(v, i * 512))
        rememberPhrases(words, flat).catch(() => { /* cache is best-effort */ })
    }
    return result
}

/** Drop every embedding for the outgoing document (model weights stay). */
export const releaseDocument = () =>
    import('./sam21-client.js').then((c) => c.releaseAll()).catch(() => null)

/** Same intent as releaseDocument, but keeps the sessions and the GPU device.
 *  For an image SWAP, where an encode of the new document follows immediately:
 *  releaseAll drops the last session, which destroys the device (lane
 *  releaseAll), so the next click paid a device rebuild + a ~1 GB session build
 *  + the shader compile — measured 457 ms against a 215 ms steady state. */
export const releaseEmbeddings = () =>
    import('./sam21-client.js').then((c) => c.releaseEmbedding()).catch(() => null)

/** Lane residency snapshot (verify/debug): { cachedImages, lane, … }. */
export const engineState = () =>
    import('./sam21-client.js').then((c) => c.refreshStatus())

/** Free reloadable residents under memory pressure. */
export const relievePressure = async (level = 1) => {
    const c = await import('./sam21-client.js')
    // A real ladder, cheapest first. L1 keeps the embedding because losing
    // it buys a ~1 s re-encode to free 8 MB; L2 gives up the device (and
    // with it the embeddings) but keeps the worker warm; only L3 returns
    // ORT's WebGPU pool to the OS (FINDINGS §1.4, measured 2112 → 78 MB).
    if (level >= 3) { await c.op('shutdown').catch(() => null); return ['sam21:worker'] }
    // Encoder only: it is the session with the multi-GB arena, and dropping
    // the LAST session destroys the WebGPU device — which clears every
    // embedding. The 9.9 MB decoder stays as the device anchor.
    await c.op('releaseEncoder').catch(() => null)
    if (level < 2) return ['sam21:encoder']
    await c.op('releaseAll').catch(() => null)
    return ['sam21:sessions']
}

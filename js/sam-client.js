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

import { IS_WEBKIT } from './engine.js'
import { summarizeMaskRGBA, validateClickMask } from './sam-core.js'
import { enqueueHeavy, cancelHeavyBefore, onHeavyActivity, STALE } from './heavy-job-queue.js'
import {
    sam21Candidates, sam21CandidateShape, sam21Cycle, sam21HdCompose, sam21PickCandidate, sam21Segment,
} from './sam21-adapter.js'
import { LANE } from './sam21-lane.js'
import { noteModel } from './model-registry.js'

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
    // A speculative (boot) warm yields to user work by RANK — an import decode
    // is priority 0 and jumps it in the queue. It is not cancelled by an
    // import: the weights it is fetching are exactly what that import's first
    // click needs. `speculative` keeps clearHeavyQueue's abandon path for the
    // case that actually blocks, a warm already stuck holding the slot.
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
    }, speculative ? { priority: 'idle', speculative: true } : {})
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
    }, speculative ? { priority: 'idle', speculative: true } : {})
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
    // The lane knows where it painted, so the summary scans that rect instead of
    // the frame. A lane that reports no rect (or an older one) still gets the
    // whole-frame scan.
    const summary = summarizeMaskRGBA(imageData.data, result.width, result.height, result.maskRect || null)
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
        // Why this candidate won and what region hygiene removed. Nothing in the
        // UI reads it; it is the only way a check can tell "the mask is wrong"
        // from "which of the six mechanisms produced it".
        pick: result.pick,
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

/** Same repaint, addressed absolutely — what a pointer on the scope control does. */
export const pickCandidate = (index) => {
    const r = sam21PickCandidate(index)
    return r ? finishSegment(r, null, Date.now()) : null
}

/** A candidate's coarse shape (256² alpha) for a hover preview. */
export const candidateShape = (index) => sam21CandidateShape(index)

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
        const { encodeImage: enc, hello } = await import('./sam21-client.js')
        await hello('seglab')
        const bitmap = await createImageBitmap(canvas)
        const r = await enc(bitmap, imageKey)
        return { encoded: !r.cached, imageKey, device: 'webgpu', lane: LANE }
    }, { priority: 'idle', revision: revision ?? null }).catch((err) => {
        // Swallowing this silently is what hid a prewarm that never ran: the
        // caller cannot tell "no embedding, by design" from "the encode threw".
        console.warn('[seglab] encode-prewarm failed:', err?.message || err)
        return null
    })
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

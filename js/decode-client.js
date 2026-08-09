/**
 * decode-client — main-thread transport to decode-worker, queued through the
 * shared heavy-job scheduler so a decode can never overlap model work. If the
 * worker cannot be constructed (file:// pages), the same decode-core code
 * runs inline — identical bounds, sticky choice.
 */

import { enqueueHeavy, STALE } from './heavy-job-queue.js'
import { readImageMeta } from './image-io.js'
import { decodeBoundedBitmap, decodeWithWorkingCopy, decodeOpaqueBounded } from './decode-core.js'
import { interactionPlan } from './proxy-plan.js'

export { STALE }

let worker = null
let workerBroken = false
let seq = 0
const pending = new Map()

const getWorker = () => {
    if (workerBroken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./decode-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            const entry = pending.get(data.requestId)
            if (!entry) {
                // Late reply for a timed-out/cancelled request: release its bitmaps.
                try { data.bitmap?.close?.() } catch { /* not a bitmap */ }
                try { data.display?.close?.() } catch { /* not a bitmap */ }
                return
            }
            pending.delete(data.requestId)
            if (data.type === 'error') entry.reject(new Error(data.error))
            else entry.resolve(data)
        }
        worker.onerror = (event) => {
            console.warn('[seglab][decode] worker error; restarting fresh:', event?.message)
            for (const [, entry] of pending) entry.reject(new Error(event?.message || 'decode worker crashed'))
            pending.clear()
            try { worker.terminate() } catch { /* dead */ }
            worker = null
        }
        return worker
    } catch (err) {
        console.warn('[seglab][decode] worker construction failed; decoding inline:', err?.message)
        workerBroken = true
        return null
    }
}

/** A worker the OS killed under memory pressure does NOT always fire `onerror`
 *  — it simply stops answering, and an un-timed round-trip then never settles.
 *  That wedged the whole heavy queue (one job at a time), so the next import sat
 *  on its spinner forever. Bound every call and rebuild the worker on expiry;
 *  the next decode gets a live one instead of piling onto a corpse. */
const TIMEOUT_MS = { meta: 20_000, default: 90_000 }

const killWorker = (why) => {
    console.warn('[seglab][decode] worker unresponsive; rebuilding:', why)
    const w = worker
    worker = null
    for (const [, entry] of pending) entry.reject(new Error('decode worker unresponsive'))
    pending.clear()
    try { w?.terminate() } catch { /* already dead */ }
}

const post = (msg, transfer = []) => {
    const w = getWorker()
    if (!w) return null
    seq += 1
    const requestId = `d${seq}`
    return new Promise((resolve, reject) => {
        const ms = TIMEOUT_MS[msg.type] ?? TIMEOUT_MS.default
        const timer = setTimeout(() => {
            if (!pending.has(requestId)) return
            pending.delete(requestId)
            reject(new Error(`decode (${msg.type}) timed out after ${Math.round(ms / 1000)}s`))
            killWorker(msg.type)
        }, ms)
        const done = (fn) => (v) => { clearTimeout(timer); fn(v) }
        pending.set(requestId, { resolve: done(resolve), reject: done(reject) })
        try {
            w.postMessage({ ...msg, requestId }, transfer)
        } catch (err) {
            clearTimeout(timer)
            pending.delete(requestId)
            reject(err)
        }
    })
}

/** Header meta (dims + EXIF orientation) — cheap, not queued. Falls back to the
 *  inline parser when the worker is missing OR unresponsive: the same code path,
 *  and an import must not die because a worker went away. */
export const readMeta = async (blob) => {
    const roundtrip = post({ type: 'meta', blob })
    if (!roundtrip) return readImageMeta(blob)
    try {
        return (await roundtrip).meta
    } catch (err) {
        console.warn('[seglab][decode] meta via worker failed; parsing inline:', err?.message)
        return readImageMeta(blob)
    }
}

/**
 * Bounded proxy decode as ONE heavy job. Resolves { bitmap, working } or the
 * STALE sentinel. The caller owns (and must close) the returned bitmap.
 */
export const decodeProxy = ({
    blob, decodeW, decodeH, scale, orientation = 1,
    wantWorking = false, workingMaxSide = 1280, displaySide = 0,
    revision = null, isCurrent = null,
}) => enqueueHeavy('decode-proxy', async () => {
    const roundtrip = post({
        type: 'decode-proxy', revision, blob, decodeW, decodeH, scale, orientation, wantWorking, workingMaxSide, displaySide,
    })
    if (roundtrip) {
        try {
            const res = await roundtrip
            return { bitmap: res.bitmap, working: res.working, display: res.display || null }
        } catch (err) {
            // Worker gone or unresponsive: decode inline rather than failing the
            // import. Same bounds, so the memory ceiling is unchanged — it just
            // costs this thread's time instead of the worker's.
            console.warn('[seglab][decode] proxy via worker failed; decoding inline:', err?.message)
        }
    }
    if (wantWorking) return decodeWithWorkingCopy(blob, { w: decodeW, h: decodeH }, scale, workingMaxSide, displaySide)
    return { bitmap: await decodeBoundedBitmap(blob, decodeW, decodeH, scale, orientation), working: null, display: null }
}, { priority: 'import', revision, isCurrent })

/** Opaque-format decode (no parseable header): bounded inside the worker. */
export const decodeOpaque = ({ blob, budget, revision = null, isCurrent = null }) => enqueueHeavy(
    'decode-proxy',
    async () => {
        const slim = {
            proxyMax: budget.proxyMax,
            proxyMode: budget.proxyMode,
            directMaxMP: budget.directMaxMP,
            directMaxSide: budget.directMaxSide,
            safeProxyMax: budget.safeProxyMax,
        }
        const roundtrip = post({ type: 'decode-opaque', revision, blob, budget: slim })
        if (roundtrip) {
            const res = await roundtrip
            return { bitmap: res.bitmap, original: res.original, proxyActive: res.proxyActive }
        }
        const res = await decodeOpaqueBounded(blob, (w, h) => interactionPlan(w, h, slim))
        return { bitmap: res.bitmap, original: { width: res.width, height: res.height }, proxyActive: res.proxyActive }
    },
    { priority: 'import', revision, isCurrent },
)

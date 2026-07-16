/**
 * decode-client — main-thread API over decode-worker
 * ----------------------------------------------------
 * decodeProxy(blob, maxLongSide, revision) → { bitmap, original, proxy }.
 * Worker failure is sticky-inline (same degrade-don't-die contract as
 * sam-client): the fallback still uses the bounded decode path — an async
 * browser API, not a main-thread pixel loop.
 */

import { compositeCutout, decodeBoundedBitmap } from './image-io.js'

const DECODE_TIMEOUT_MS = 60 * 1000
const EXPORT_TIMEOUT_MS = 120 * 1000

const withTimeout = (promise, ms, label) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms)
        promise.then(
            (v) => { clearTimeout(timer); resolve(v) },
            (e) => { clearTimeout(timer); reject(e) },
        )
    })

let worker = null
let workerBroken = false
let seq = 0
const pending = new Map() // id → {resolve, reject}

const getWorker = () => {
    if (workerBroken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./decode-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            const entry = pending.get(data.id)
            if (!entry) {
                // Late reply to a timed-out request — release its bitmap.
                try { data.result?.bitmap?.close?.() } catch { /* detached */ }
                return
            }
            pending.delete(data.id)
            if (data.ok) entry.resolve(data.result)
            else entry.reject(new Error(data.error || 'image decode failed'))
        }
        worker.onerror = (event) => {
            console.warn('[seglab] decode worker error; switching inline:', event?.message)
            for (const [, entry] of pending) entry.reject(new Error(event?.message || 'decode worker crashed'))
            pending.clear()
            try { worker.terminate() } catch { /* already dead */ }
            worker = null
            workerBroken = true
        }
        return worker
    } catch (err) {
        console.warn('[seglab] decode worker construction failed; using inline decode:', err?.message)
        workerBroken = true
        return null
    }
}

const roundtrip = (message, transfer, timeoutMs, label) => {
    const w = getWorker()
    if (!w) return null
    const id = `decode-${++seq}`
    const reply = new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject })
        try {
            w.postMessage({ id, ...message }, transfer || [])
        } catch (err) {
            pending.delete(id)
            reject(err)
        }
    })
    return withTimeout(reply, timeoutMs, label).finally(() => pending.delete(id))
}

export const decodeProxy = async (blob, maxLongSide, revision = 0) => {
    const viaWorker = roundtrip({ op: 'decode-proxy', revision, blob, maxLongSide }, null, DECODE_TIMEOUT_MS, 'Image decode')
    if (viaWorker) {
        try {
            return await viaWorker
        } catch (err) {
            if (!workerBroken) throw err
            console.warn('[seglab] retrying decode inline after worker failure')
        }
    }
    return decodeBoundedBitmap(blob, maxLongSide)
}

/** Explicit-export composite; mask alpha (and any bitmap) transfer in. */
export const exportComposite = async ({ blob = null, bitmap = null, mask, caps }, revision = 0) => {
    const transfer = [mask.alpha.buffer]
    if (bitmap) transfer.push(bitmap)
    const viaWorker = roundtrip({ op: 'export-composite', revision, blob, bitmap, mask, caps }, transfer, EXPORT_TIMEOUT_MS, 'Cutout export')
    if (viaWorker) {
        try {
            return await viaWorker
        } catch (err) {
            if (!workerBroken) throw err
            console.warn('[seglab] retrying export inline after worker failure')
            throw err // mask buffer transferred into the dead worker — caller retries with a fresh copy
        }
    }
    return compositeCutout({ blob, bitmap, mask, caps })
}

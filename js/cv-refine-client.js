/**
 * cv-refine-client — main-thread API over cv-refine-worker, queued through
 * the shared heavy scheduler. Lazy: the worker/wasm load only when a first
 * real refinement is requested. Failure is sticky and silent — the caller
 * keeps the model's own mask; there is no retry loop and no fallback library.
 */

import { enqueueHeavy, STALE } from './heavy-job-queue.js'

const MAX_SIDE = 1024
const REFINE_TIMEOUT_MS = 15_000

// Wasm SIMD feature probe (the build is SIMD-only; unsupported hosts keep
// the JS post pipeline instead).
const SIMD_PROBE = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
    10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
])
let simdOk = null
export const simdSupported = () => {
    simdOk ??= (() => {
        try { return WebAssembly.validate(SIMD_PROBE) } catch { return false }
    })()
    return simdOk
}

let worker = null
let broken = false
let seq = 0
const pending = new Map()

const getWorker = () => {
    if (broken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./cv-refine-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            const entry = pending.get(data.requestId)
            if (!entry) return
            pending.delete(data.requestId)
            if (data.type === 'result') entry.resolve(data)
            else entry.reject(new Error(data.error || 'refine failed'))
        }
        worker.onerror = (event) => {
            console.warn('[seglab][cv] refine worker failed; keeping model masks:', event?.message)
            for (const [, entry] of pending) entry.reject(new Error(event?.message || 'refine worker crashed'))
            pending.clear()
            try { worker.terminate() } catch { /* dead */ }
            worker = null
            broken = true // no repeated retries
        }
        return worker
    } catch (err) {
        console.warn('[seglab][cv] refine worker unavailable:', err?.message)
        broken = true
        return null
    }
}

export const cvRefineAvailable = () => !broken && simdSupported()
export const cvRefineLoaded = () => !!worker

/** Terminate the worker (memory pressure ≥ 2, new document, before export). */
export const disposeCvRefine = () => {
    if (!worker) return
    try { worker.postMessage({ type: 'dispose' }) } catch { /* gone */ }
    try { worker.terminate() } catch { /* gone */ }
    worker = null
    for (const [, entry] of pending) entry.reject(new Error('refine worker disposed'))
    pending.clear()
}

/**
 * Refine a one-channel mask. `alpha` is a Uint8Array (TRANSFERRED — detached
 * after this call); returns a new Uint8Array, or null when refinement is
 * unavailable/stale/failed (the caller keeps its original mask — pass a copy).
 */
export const refineAlpha = async ({
    alpha, width, height, rgb = null, seeds = [], options = {}, revision = null, budget = {},
}) => {
    if (!cvRefineAvailable()) return null
    if (budget.cvRefine === false || (budget.pressureLevel || 0) >= 2) return null
    if (!alpha || Math.max(width, height) > MAX_SIDE || alpha.length !== width * height) return null

    const result = await enqueueHeavy('cv-refine', () => {
        const w = getWorker()
        if (!w) return null
        seq += 1
        const requestId = `cv${seq}`
        return new Promise((resolve) => {
            const timer = setTimeout(() => {
                pending.delete(requestId)
                console.warn('[seglab][cv] refine timed out; keeping model mask')
                resolve(null)
            }, REFINE_TIMEOUT_MS)
            pending.set(requestId, {
                resolve: (data) => { clearTimeout(timer); resolve(data) },
                reject: (err) => {
                    clearTimeout(timer)
                    console.warn('[seglab][cv] refine failed; keeping model mask:', err?.message)
                    resolve(null)
                },
            })
            const transfer = [alpha.buffer]
            if (rgb) transfer.push(rgb.buffer)
            try {
                w.postMessage({
                    type: 'refine-mask',
                    requestId,
                    revision,
                    width,
                    height,
                    rgb: rgb ? rgb.buffer : null,
                    alpha: alpha.buffer,
                    seeds,
                    options,
                }, transfer)
            } catch (err) {
                clearTimeout(timer)
                pending.delete(requestId)
                console.warn('[seglab][cv] refine post failed:', err?.message)
                resolve(null)
            }
        })
    }, { priority: 'high', revision })

    if (result === STALE || !result || result.stale) return null
    return new Uint8Array(result.alpha)
}

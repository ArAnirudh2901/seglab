/**
 * cv-refine-client — main-thread API over cv-refine-worker
 * ----------------------------------------------------------
 * refine() transfers the mask buffers both ways (never copies). Worker
 * construction/crash is sticky-broken: callers then keep the engine's JS
 * pipeline — a lost refiner never costs a selection. dispose() releases the
 * worker under memory pressure; it can be recreated later.
 */

const CV_TIMEOUT_MS = 30 * 1000

const withTimeout = (promise, ms, label) =>
    new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)), ms)
        promise.then(
            (v) => { clearTimeout(timer); resolve(v) },
            (e) => { clearTimeout(timer); reject(e) },
        )
    })

let worker = null
let broken = false
let seq = 0
let guideRevision = 0
const pending = new Map()

const getWorker = () => {
    if (broken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./cv-refine-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            const entry = pending.get(data.id)
            if (!entry) return
            pending.delete(data.id)
            if (data.ok) entry.resolve(data.result)
            else entry.reject(new Error(data.error || 'cv refinement failed'))
        }
        worker.onerror = (event) => {
            console.warn('[seglab] cv worker error; refinement disabled:', event?.message)
            for (const [, entry] of pending) entry.reject(new Error(event?.message || 'cv worker crashed'))
            pending.clear()
            try { worker.terminate() } catch { /* already dead */ }
            worker = null
            broken = true
        }
        return worker
    } catch (err) {
        console.warn('[seglab] cv worker construction failed:', err?.message)
        broken = true
        return null
    }
}

const post = (op, payload, transfer, label) => {
    const w = getWorker()
    if (!w) return Promise.reject(new Error('cv worker unavailable'))
    const id = `cv-${++seq}`
    const roundtrip = new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject })
        try {
            w.postMessage({ id, op, ...payload }, transfer || [])
        } catch (err) {
            pending.delete(id)
            reject(err)
        }
    })
    return withTimeout(roundtrip, CV_TIMEOUT_MS, label).finally(() => pending.delete(id))
}

export const isCvUsable = () => !broken

/** Ship the ≤768 proxy once per document as the guided-filter guide. */
export const ensureGuide = async (revision, canvas) => {
    if (guideRevision === revision) return
    const bitmap = await createImageBitmap(canvas)
    try {
        await post('set-guide', { revision, bitmap }, [bitmap], 'cv guide')
        guideRevision = revision
    } catch (err) {
        try { bitmap.close() } catch { /* transferred or closed */ }
        throw err
    }
}

/** Hygiene + edge refinement; alpha/alphaRaw buffers transfer both ways. */
export const refine = ({ revision, width, height, alpha, alphaRaw, seeds = [], options = {} }) =>
    post('refine', { revision, width, height, alpha, alphaRaw, seeds, options },
        [alpha.buffer, alphaRaw.buffer], 'cv refine')

export const resetCv = (revision) => {
    guideRevision = 0
    if (worker && !broken) post('reset', { revision }, null, 'cv reset').catch(() => { /* cold */ })
}

/** Memory pressure: drop the wasm heap + guide and stop the worker. */
export const disposeCv = () => {
    guideRevision = 0
    if (!worker) return
    post('dispose', {}, null, 'cv dispose')
        .catch(() => { /* dying anyway */ })
        .finally(() => {
            try { worker?.terminate() } catch { /* already dead */ }
            worker = null
        })
}

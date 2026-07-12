/**
 * sam-worker — dedicated-worker shell around sam-engine
 * -------------------------------------------------------
 * All inference runs here so encoder and decoder passes never block the UI
 * thread. Plain request/response protocol with correlation ids:
 *
 *   in : { id, op: 'warm', payload: { flagship } }
 *   in : { id, op: 'segment', payload: { imageKey, source: ImageBitmap
 *          (transferred), clicks, box, clampPoly, clampMargin } }
 *   out: { id, ok: true, result }   — mask buffers transferred back, zero-copy
 *   out: { id, ok: false, error }
 *   out: { type: 'progress'|'lane', ... }  — engine broadcasts (download
 *         progress, flagship hot-swap)
 *
 * All engine logic (lanes, embedding cache, post pipeline, fallbacks) lives
 * in sam-engine.js so the exact same code can run inline on the main thread
 * when worker construction fails (see sam-client.js).
 */

import {
    cancelBefore, detectorCandidates, encodeImage, getBudget, getEngineState, hdRefine, relievePressure, segment, setEventSink, warm,
} from './sam-engine.js'
import { detect } from './detect-engine.js'

setEventSink((event) => {
    try { self.postMessage(event) } catch { /* non-cloneable — best-effort */ }
})

self.onmessage = async (event) => {
    const { id, op, payload } = event.data || {}
    if (!op) return
    // Cancel is fire-and-forget (no id, no reply): it must take effect
    // immediately, never queue behind the job it is trying to obsolete.
    if (op === 'cancel') {
        cancelBefore(payload?.before || 0)
        return
    }
    if (!id) return
    try {
        if (op === 'warm') {
            const result = await warm(payload || {})
            self.postMessage({ id, ok: true, result })
            return
        }
        if (op === 'segment') {
            const { source } = payload || {}
            try {
                const result = await segment(payload || {})
                // Stale (cancelled) results carry no mask buffers.
                const transfer = result?.rgba ? [result.rgba.buffer, result.rawRgba.buffer] : []
                self.postMessage({ id, ok: true, result }, transfer)
            } finally {
                // The transferred bitmap is this side's to release.
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'encode') {
            const { source } = payload || {}
            try {
                const result = await encodeImage(payload || {})
                self.postMessage({ id, ok: true, result })
            } finally {
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'detect') {
            const { source } = payload || {}
            try {
                const progress_callback = (info) => self.postMessage({
                    type: 'progress',
                    detail: { lane: 'text', status: info?.status, file: info?.file, progress: info?.progress, loaded: info?.loaded, total: info?.total },
                })
                const res = await detect({
                    source,
                    labels: payload.labels,
                    threshold: payload.threshold,
                    candidates: detectorCandidates(),
                    dispose: getBudget().detectorDispose === 'now',
                    progress_callback,
                })
                self.postMessage({ id, ok: true, result: res })
            } finally {
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'hdExport') {
            const { source } = payload || {}
            try {
                const result = await hdRefine(payload || {})
                const transfer = result?.alpha ? [result.alpha.buffer] : []
                self.postMessage({ id, ok: true, result }, transfer)
            } finally {
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'state') {
            self.postMessage({ id, ok: true, result: getEngineState() })
            return
        }
        if (op === 'pressure') {
            const freed = await relievePressure(payload?.level || 1)
            self.postMessage({ id, ok: true, result: { freed } })
            return
        }
        throw new Error(`Unknown op: ${op}`)
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

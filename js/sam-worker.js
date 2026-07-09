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

import { getEngineState, segment, setEventSink, warm } from './sam-engine.js'

setEventSink((event) => {
    try { self.postMessage(event) } catch { /* non-cloneable — best-effort */ }
})

self.onmessage = async (event) => {
    const { id, op, payload } = event.data || {}
    if (!id || !op) return
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
                self.postMessage({ id, ok: true, result }, [result.rgba.buffer, result.rawRgba.buffer])
            } finally {
                // The transferred bitmap is this side's to release.
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'state') {
            self.postMessage({ id, ok: true, result: getEngineState() })
            return
        }
        throw new Error(`Unknown op: ${op}`)
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

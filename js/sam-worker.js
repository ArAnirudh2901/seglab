/**
 * sam-worker — dedicated-worker shell around sam-engine
 * -------------------------------------------------------
 * All inference runs here so encoder and decoder passes never block the UI
 * thread. Plain request/response protocol with correlation ids:
 *
 *   in : { id, op: 'warm', payload: {} }               — draft lane only
 *   in : { id, op: 'flagship-start', payload: {} }     — EXPLICIT opt-in only
 *   in : { id, op: 'reset', payload: { revision } }    — new document
 *   in : { id, op: 'segment', payload: { revision, source: ImageBitmap
 *          (transferred), clicks, box, clampPoly, clampMargin } }
 *   out: { id, ok: true, result }   — one-channel alpha buffers transferred
 *   out: { id, ok: false, error, stale? }
 *   out: { type: 'progress'|'lane', ... }  — engine broadcasts (download
 *         progress, opt-in flagship hot-swap)
 *
 * All engine logic (lanes, embedding cache, post pipeline, fallbacks) lives
 * in sam-engine.js so the exact same code can run inline on the main thread
 * when worker construction fails (see sam-client.js).
 */

import { getEngineState, resetImage, segment, setEventSink, startFlagship, warm } from './sam-engine.js'

setEventSink((event) => {
    try { self.postMessage(event) } catch { /* non-cloneable — best-effort */ }
})

self.onmessage = async (event) => {
    const { id, op, payload } = event.data || {}
    if (!id || !op) return
    try {
        if (op === 'warm') {
            const result = await warm()
            self.postMessage({ id, ok: true, result })
            return
        }
        if (op === 'flagship-start') {
            const result = await startFlagship()
            self.postMessage({ id, ok: true, result })
            return
        }
        if (op === 'reset') {
            self.postMessage({ id, ok: true, result: resetImage(payload?.revision || 0) })
            return
        }
        if (op === 'segment') {
            const { source } = payload || {}
            try {
                const result = await segment(payload || {})
                self.postMessage({ id, ok: true, result }, [result.alpha.buffer, result.alphaRaw.buffer])
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
        self.postMessage({ id, ok: false, error: String(err?.message || err), stale: !!err?.stale })
    }
}

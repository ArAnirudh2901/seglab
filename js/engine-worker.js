/**
 * engine-worker — dedicated-worker shell around the unified engine
 * -------------------------------------------------------------------
 * All inference runs here so the model pass never blocks the UI. Plain
 * request/response protocol with correlation ids:
 *
 *   in : { id, op: 'warm' }
 *   in : { id, op: 'select', payload: { imageKey, source (transferred),
 *          mode, point|box|poly|query, exclude } }
 *   in : { id, op: 'list',  payload: { imageKey, source } }
 *   out: { id, ok: true, result }   — mask buffers transferred back, zero-copy
 *   out: { id, ok: false, error }
 *   out: { type: 'progress'|'state', ... }  — engine broadcasts
 *
 * All engine logic lives in engine.js so the identical code can run inline on
 * the main thread when worker construction fails (see engine-client.js).
 */

import { getEngineState, listInstances, select, setEventSink, warm } from './engine.js'

setEventSink((event) => {
    try { self.postMessage(event) } catch { /* non-cloneable — best-effort */ }
})

self.onmessage = async (event) => {
    const { id, op, payload } = event.data || {}
    if (!id || !op) return
    try {
        if (op === 'warm') {
            self.postMessage({ id, ok: true, result: await warm() })
            return
        }
        if (op === 'select') {
            const { source } = payload || {}
            try {
                const result = await select(payload || {})
                const transfer = []
                if (result.rgba) transfer.push(result.rgba.buffer)
                if (result.rawRgba) transfer.push(result.rawRgba.buffer)
                self.postMessage({ id, ok: true, result }, transfer)
            } finally {
                // The transferred bitmap is this side's to release.
                try { source?.close?.() } catch { /* already closed */ }
            }
            return
        }
        if (op === 'list') {
            const { source } = payload || {}
            try {
                self.postMessage({ id, ok: true, result: await listInstances(payload || {}) })
            } finally {
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

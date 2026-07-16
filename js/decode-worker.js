/**
 * decode-worker — dedicated worker shell around image-io's bounded decode
 * -------------------------------------------------------------------------
 * Owns header parsing + proxy decoding so the UI thread never touches an
 * original frame. The worker does NOT add memory budget (workers share the
 * page's allocation space) — it exists for responsiveness and explicit
 * transfer ownership. Only ≤ maxLongSide bitmaps ever cross the boundary.
 *
 *   in : { id, op: 'decode-proxy', revision, blob, maxLongSide }
 *   out: { id, ok: true, revision, result: { bitmap, original, proxy } }
 *         — bitmap transferred, never cloned
 *   in : { id, op: 'export-composite', revision, blob|bitmap,
 *          mask: { alpha (transferred), width, height },
 *          caps: { maxSide, maxMP } }
 *   out: { id, ok: true, revision, result: { blob, width, height, reduced } }
 *   out: { id, ok: false, revision, error }
 */

import { compositeCutout, decodeBoundedBitmap } from './image-io.js'

self.onmessage = async (event) => {
    const { id, op, revision, blob, bitmap, mask, caps, maxLongSide } = event.data || {}
    if (!id || !op) return
    try {
        if (op === 'decode-proxy') {
            const result = await decodeBoundedBitmap(blob, maxLongSide)
            self.postMessage({ id, ok: true, revision, result }, [result.bitmap])
            return
        }
        if (op === 'export-composite') {
            const result = await compositeCutout({ blob, bitmap, mask, caps })
            self.postMessage({ id, ok: true, revision, result })
            return
        }
        throw new Error(`Unknown op: ${op}`)
    } catch (err) {
        self.postMessage({ id, ok: false, revision, error: String(err?.message || err) })
    }
}

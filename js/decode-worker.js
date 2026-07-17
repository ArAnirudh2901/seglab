/**
 * decode-worker — dedicated image-decode worker.
 * Owns header parsing and bounded proxy decoding so the UI thread never
 * blocks on a DSLR decode. Bitmaps return via transfer (zero-copy). This
 * worker shares the page's memory budget — it isolates lifecycle and keeps
 * the main thread responsive, nothing more.
 *
 * in : { type:'meta',         requestId, blob }
 * in : { type:'decode-proxy', requestId, revision, blob, decodeW, decodeH,
 *        scale, orientation, wantWorking, workingMaxSide, displaySide }
 * in : { type:'decode-opaque', requestId, revision, blob,
 *        budget:{proxyMax,proxyMode,directMaxMP,directMaxSide,safeProxyMax} }
 * out: { type:'meta-result', requestId, meta }
 * out: { type:'decode-proxy-result', requestId, revision, bitmap(transfer),
 *        working: {blob,w,h}|null, display: bitmap(transfer)|null,
 *        original:{width,height}, proxyActive? }
 * out: { type:'error', requestId, error }
 */

import { readImageMeta } from './image-io.js'
import { decodeBoundedBitmap, decodeWithWorkingCopy, decodeOpaqueBounded } from './decode-core.js'
import { interactionPlan } from './proxy-plan.js'

self.onmessage = async (event) => {
    const msg = event.data || {}
    const { type, requestId } = msg
    if (!requestId) return
    try {
        if (type === 'meta') {
            const meta = await readImageMeta(msg.blob)
            self.postMessage({ type: 'meta-result', requestId, meta })
            return
        }
        if (type === 'decode-proxy') {
            let bitmap = null
            let working = null
            let display = null
            if (msg.wantWorking) {
                ({ bitmap, working, display } = await decodeWithWorkingCopy(
                    msg.blob, { w: msg.decodeW, h: msg.decodeH }, msg.scale, msg.workingMaxSide, msg.displaySide || 0,
                ))
            } else {
                bitmap = await decodeBoundedBitmap(msg.blob, msg.decodeW, msg.decodeH, msg.scale, msg.orientation)
            }
            self.postMessage({
                type: 'decode-proxy-result',
                requestId,
                revision: msg.revision,
                bitmap,
                working,
                display: display || null,
                original: { width: msg.decodeW, height: msg.decodeH },
            }, display ? [bitmap, display] : [bitmap])
            return
        }
        if (type === 'decode-opaque') {
            const res = await decodeOpaqueBounded(msg.blob, (w, h) => interactionPlan(w, h, msg.budget))
            self.postMessage({
                type: 'decode-proxy-result',
                requestId,
                revision: msg.revision,
                bitmap: res.bitmap,
                working: null,
                original: { width: res.width, height: res.height },
                proxyActive: res.proxyActive,
            }, [res.bitmap])
            return
        }
        throw new Error(`Unknown decode op: ${type}`)
    } catch (err) {
        self.postMessage({ type: 'error', requestId, error: String(err?.message || err) })
    }
}

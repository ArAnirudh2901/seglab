/**
 * export-hd — native-resolution cutout compositing (main thread).
 * Maps the approved ≤1024 proxy mask back onto the original pixels:
 * mask bbox → pad → crop rect → (Std/Pro) one IoU-gated crop re-decode +
 * band-tiled refine in the worker → composite into a full-frame cutout.
 * Outside the padded crop the mask is empty, so the frame is transparent —
 * no seams.
 *
 * M3 shares the crop decode with interactive escalation: a small selection
 * re-decodes its crop at interaction time and caches the native alpha as an
 * `hdPatch`; export then just composites that patch (no second decode).
 */

import { summarizeMaskRGBA, mapPromptsToCrop } from './sam-core.js'
import {
    getTransform, getCropBitmap, cropKeyFor, getOriginalForExport, hasOriginal,
} from './asset-store.js'
import { hdExport } from './sam-client.js'

const makeCanvas = (w, h) => {
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    return c
}

// The native patch escalation produced, keyed by the revision it decoded for.
// Export reuses it; a prompt change (revision bump) retires it.
let hdPatch = null
export const setHdPatch = (patch) => { hdPatch = patch }
export const clearHdPatch = () => { hdPatch = null }
export const getHdPatch = (revision) => (hdPatch && hdPatch.revision === revision ? hdPatch : null)

// Proxy-mask bbox → padded integer crop rect in ORIGINAL px.
const cropRectFromBBox = (bbox, tf) => {
    const [minX, minY, maxX, maxY] = bbox
    const p2o = tf.originalW / tf.proxyW
    const ox0 = minX * p2o
    const oy0 = minY * p2o
    const ox1 = (maxX + 1) * p2o
    const oy1 = (maxY + 1) * p2o
    const pad = Math.max(24, 0.06 * Math.hypot(ox1 - ox0, oy1 - oy0))
    const x = Math.max(0, Math.floor(ox0 - pad))
    const y = Math.max(0, Math.floor(oy0 - pad))
    const x1 = Math.min(tf.originalW, Math.ceil(ox1 + pad))
    const y1 = Math.min(tf.originalH, Math.ceil(oy1 + pad))
    return { x, y, w: Math.max(1, x1 - x), h: Math.max(1, y1 - y) }
}

// Crop rect → native-res alpha via the worker's hdExport op. `forceDecode`
// (escalation) re-decodes regardless of the budget flag; a promptless union
// stays filter-only. A frame-filling crop is already at the proxy ceiling, so
// it skips the decode. Returns null when the job went stale.
const decodeCropAlpha = async (proxyMask, prompts, rect, tf, { budget, revision, forceDecode = false }) => {
    const p2o = tf.originalW / tf.proxyW
    const frameArea = tf.originalW * tf.originalH
    const cropArea = rect.w * rect.h
    const cropPrompts = mapPromptsToCrop(prompts, p2o, rect)
    // Re-decode needs a prompt; a union ("all bottles") has none → filter-only.
    const hasPrompt = cropPrompts.clicks.length > 0 || !!cropPrompts.box
    const doDecode = (forceDecode || !!budget?.hdExportDecode) && hasPrompt && cropArea / frameArea < 0.85

    const cropBitmap = await getCropBitmap(rect) // may be downscaled if huge
    const cropW = cropBitmap.width

    // Proxy-mask region for this crop + true proxy→crop upsampling (band).
    const proxySubrect = {
        sx: rect.x * tf.scale,
        sy: rect.y * tf.scale,
        sw: rect.w * tf.scale,
        sh: rect.h * tf.scale,
    }
    const upFactor = cropW / proxySubrect.sw

    const proxyBuf = new Uint8ClampedArray(proxyMask.data) // clone for transfer
    const res = await hdExport({
        revision,
        cropKey: doDecode ? cropKeyFor(rect) : null,
        source: cropBitmap,
        proxyMask: { data: proxyBuf.buffer, width: proxyMask.width, height: proxyMask.height },
        proxySubrect,
        prompts: cropPrompts,
        doDecode,
        upFactor,
    }, [cropBitmap, proxyBuf.buffer])
    if (res.stale) return null
    return { rect, proxySubrect, upFactor, alpha: res.alpha, width: res.width, height: res.height, decoded: res.decoded }
}

// Original photo kept only where the crop alpha is on; outside the rect stays
// clear (no seams). `alpha` is cloned for the copy, never mutated (a reused
// patch must survive a second export).
const compositeCropAlpha = async (alpha, width, height, rect, tf, decoded) => {
    const { source: original, owned } = await getOriginalForExport()
    try {
        const out = makeCanvas(tf.originalW, tf.originalH)
        const octx = out.getContext('2d')
        octx.drawImage(original, 0, 0, tf.originalW, tf.originalH)

        const cropMask = makeCanvas(width, height)
        const cm = new ImageData(new Uint8ClampedArray(alpha), width, height)
        for (let i = 0; i < cm.data.length; i += 4) cm.data[i + 3] = cm.data[i] // luma → alpha
        cropMask.getContext('2d').putImageData(cm, 0, 0)

        const maskFrame = makeCanvas(tf.originalW, tf.originalH) // transparent outside the crop
        const mctx = maskFrame.getContext('2d')
        mctx.imageSmoothingEnabled = true
        mctx.imageSmoothingQuality = 'high'
        mctx.drawImage(cropMask, 0, 0, width, height, rect.x, rect.y, rect.w, rect.h)

        octx.globalCompositeOperation = 'destination-in'
        octx.drawImage(maskFrame, 0, 0)

        return { canvas: out, width: tf.originalW, height: tf.originalH, decoded }
    } finally {
        if (owned) { try { original.close() } catch { /* closed */ } }
    }
}

/**
 * M3 interactive escalation: one native-res crop re-decode for a small
 * selection. On success caches the native patch (HD export reuses it) and
 * returns { rect, proxySubrect, alpha, width, height, decoded } so the caller
 * can merge it back into the proxy mask. null = no-op (no original, empty
 * mask, or the re-decode was IoU-rejected / went stale) — the proxy stands.
 */
export const escalateCrop = async (proxyMask, prompts, { budget, revision } = {}) => {
    if (!hasOriginal()) return null
    const tf = getTransform()
    if (!tf) return null
    const summary = summarizeMaskRGBA(proxyMask.data, proxyMask.width, proxyMask.height)
    if (!summary.bbox) return null
    const rect = cropRectFromBBox(summary.bbox, tf)
    const crop = await decodeCropAlpha(proxyMask, prompts, rect, tf, { budget, revision, forceDecode: true })
    if (!crop || !crop.decoded) return null
    setHdPatch({ revision, rect, alpha: crop.alpha, width: crop.width, height: crop.height, decoded: true })
    return crop
}

/** Full-res cutout for `proxyMask` selected by `prompts` (both proxy-space).
 *  → { canvas (original RGBA, masked alpha), width, height, decoded }, or
 *  null when no original is held (caller falls back to proxy export). */
export const buildCutout = async (proxyMask, prompts, { budget, revision } = {}) => {
    if (!hasOriginal()) return null
    const tf = getTransform()
    if (!tf) return null

    // Escalation already decoded this selection's crop at native res —
    // composite that patch instead of decoding again.
    const patch = getHdPatch(revision)
    if (patch) return compositeCropAlpha(patch.alpha, patch.width, patch.height, patch.rect, tf, patch.decoded)

    const summary = summarizeMaskRGBA(proxyMask.data, proxyMask.width, proxyMask.height)
    if (!summary.bbox) return null
    const rect = cropRectFromBBox(summary.bbox, tf)
    const crop = await decodeCropAlpha(proxyMask, prompts, rect, tf, { budget, revision })
    if (!crop) return null
    return compositeCropAlpha(crop.alpha, crop.width, crop.height, rect, tf, crop.decoded)
}

/** buildCutout → PNG Blob (or null when there's no original to export). */
export const exportCutoutBlob = async (proxyMask, prompts, opts) => {
    const res = await buildCutout(proxyMask, prompts, opts)
    if (!res) return null
    const blob = await new Promise((r) => res.canvas.toBlob(r, 'image/png'))
    return { blob, width: res.width, height: res.height, decoded: res.decoded }
}

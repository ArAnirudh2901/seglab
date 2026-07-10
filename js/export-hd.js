/**
 * export-hd — native-resolution cutout compositing (main thread).
 * Maps the approved ≤1024 proxy mask back onto the original pixels:
 * mask bbox → pad → crop rect → (Std/Pro) one IoU-gated crop re-decode +
 * band-tiled refine in the worker → composite into a full-frame cutout.
 * Outside the padded crop the mask is empty, so the frame is transparent —
 * no seams.
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

/** Full-res cutout for `proxyMask` selected by `prompts` (both proxy-space).
 *  → { canvas (original RGBA, masked alpha), width, height, decoded }, or
 *  null when no original is held (caller falls back to proxy export). */
export const buildCutout = async (proxyMask, prompts, { budget, revision } = {}) => {
    if (!hasOriginal()) return null
    const tf = getTransform()
    if (!tf) return null

    const summary = summarizeMaskRGBA(proxyMask.data, proxyMask.width, proxyMask.height)
    if (!summary.bbox) return null
    const [minX, minY, maxX, maxY] = summary.bbox

    // Pad the bbox in ORIGINAL space and clamp to the frame → integer crop.
    const p2o = tf.originalW / tf.proxyW      // proxy px → original px
    const ox0 = minX * p2o
    const oy0 = minY * p2o
    const ox1 = (maxX + 1) * p2o
    const oy1 = (maxY + 1) * p2o
    const pad = Math.max(24, 0.06 * Math.hypot(ox1 - ox0, oy1 - oy0))
    const x = Math.max(0, Math.floor(ox0 - pad))
    const y = Math.max(0, Math.floor(oy0 - pad))
    const x1 = Math.min(tf.originalW, Math.ceil(ox1 + pad))
    const y1 = Math.min(tf.originalH, Math.ceil(oy1 + pad))
    const rect = { x, y, w: Math.max(1, x1 - x), h: Math.max(1, y1 - y) }

    // Re-decode only when the object is sub-frame (a frame-filling object is
    // already at the proxy ceiling) and the profile allows it.
    const frameArea = tf.originalW * tf.originalH
    const cropArea = rect.w * rect.h
    const doDecode = !!budget?.hdExportDecode && cropArea / frameArea < 0.85

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

    const cropPrompts = mapPromptsToCrop(prompts, p2o, rect)
    const proxyBuf = new Uint8ClampedArray(proxyMask.data) // clone for transfer
    const payload = {
        revision,
        cropKey: doDecode ? cropKeyFor(rect) : null,
        source: cropBitmap,
        proxyMask: { data: proxyBuf.buffer, width: proxyMask.width, height: proxyMask.height },
        proxySubrect,
        prompts: cropPrompts,
        doDecode,
        upFactor,
    }
    const res = await hdExport(payload, [cropBitmap, proxyBuf.buffer])
    if (res.stale) return null

    // Composite: original photo → keep only masked alpha (crop alpha's luma
    // → alpha channel, scaled to the rect; outside the rect stays clear).
    const { source: original, owned } = await getOriginalForExport()
    try {
        const out = makeCanvas(tf.originalW, tf.originalH)
        const octx = out.getContext('2d')
        octx.drawImage(original, 0, 0, tf.originalW, tf.originalH)

        const cropMask = makeCanvas(res.width, res.height)
        const cm = new ImageData(res.alpha, res.width, res.height)
        for (let i = 0; i < cm.data.length; i += 4) cm.data[i + 3] = cm.data[i] // luma → alpha
        cropMask.getContext('2d').putImageData(cm, 0, 0)

        const maskFrame = makeCanvas(tf.originalW, tf.originalH) // transparent outside the crop
        const mctx = maskFrame.getContext('2d')
        mctx.imageSmoothingEnabled = true
        mctx.imageSmoothingQuality = 'high'
        mctx.drawImage(cropMask, 0, 0, res.width, res.height, rect.x, rect.y, rect.w, rect.h)

        octx.globalCompositeOperation = 'destination-in'
        octx.drawImage(maskFrame, 0, 0)

        return { canvas: out, width: tf.originalW, height: tf.originalH, decoded: res.decoded }
    } finally {
        if (owned) { try { original.close() } catch { /* closed */ } }
    }
}

/** buildCutout → PNG Blob (or null when there's no original to export). */
export const exportCutoutBlob = async (proxyMask, prompts, opts) => {
    const res = await buildCutout(proxyMask, prompts, opts)
    if (!res) return null
    const blob = await new Promise((r) => res.canvas.toBlob(r, 'image/png'))
    return { blob, width: res.width, height: res.height, decoded: res.decoded }
}

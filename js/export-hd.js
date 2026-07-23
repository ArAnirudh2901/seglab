/**
 * export-hd — native-resolution cutout compositing (main thread).
 * Maps the approved ≤1024 proxy mask back onto the original pixels:
 * mask bbox → pad → crop rect → (Std/Pro) one IoU-gated crop re-decode +
 * band-tiled refine in the worker, which fuses the crop's own RGB with the
 * matte and hands back ONE straight-alpha cutout. The deliverable is the tight
 * crop at native resolution — never a full-frame canvas and never a second
 * full-original decode, so there is no export memory spike and no MP cap.
 *
 * M3 still caches the escalation patch (a mask) for the LIVE overlay, but
 * export always recomposes fresh at full res so quality is never bounded by an
 * interaction-time decode.
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

// Crop decode caps. Export at full res decodes the crop natively (no MP cap)
// when the tier allows and memory pressure has not ratcheted it back; otherwise
// the tier bounded caps apply. Escalation (interaction-time) always passes its
// own bounded caps.
const cropCaps = (budget, { fullRes = false } = {}) => {
    const pressured = (budget?.pressureLevel || 0) >= 2
    if (fullRes && budget?.exportFullRes && !pressured) return { maxSide: 0, maxMP: 0 }
    return { maxSide: budget?.cropMaxSide ?? 4096, maxMP: budget?.cropMaxMP ?? 0 }
}

// Crop rect to native-res cutout or mask via the worker hdExport op. Compose
// (export) fuses the crop RGB with the matte and returns one cutout buffer, or a
// blob when emitBlob is set; otherwise (escalation) returns the mask. forceDecode
// (escalation) re-decodes regardless of the budget flag; a promptless union stays
// filter-only. A frame-filling crop is already at the proxy ceiling, so it skips
// the decode. Returns null when the job went stale.
const decodeCropAlpha = async (proxyMask, prompts, rect, tf, {
    budget, revision, forceDecode = false, nativeCrop = false, compose = false, emitBlob = false, caps = null,
}) => {
    const p2o = tf.originalW / tf.proxyW
    const frameArea = tf.originalW * tf.originalH
    const cropArea = rect.w * rect.h

    const c = caps || cropCaps(budget, { fullRes: nativeCrop })
    const cropBitmap = await getCropBitmap(rect, { maxSide: c.maxSide, maxMP: c.maxMP, native: nativeCrop })
    const cropW = cropBitmap.width
    const bounded = !nativeCrop && !!tf.workingActive

    // Prompts live in the crop bitmap's OWN pixel space; the bitmap can be
    // smaller than the rect (working copy and/or cropMaxSide downscale).
    const cropPrompts = scaleCropPrompts(mapPromptsToCrop(prompts, p2o, rect), cropW / rect.w)
    // Re-decode needs a prompt; a union ("all bottles") has none → filter-only.
    const hasPrompt = cropPrompts.clicks.length > 0 || !!cropPrompts.box
    const doDecode = (forceDecode || !!budget?.hdExportDecode) && hasPrompt && cropArea / frameArea < 0.85

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
        // Working-res and native-res pixels for the same rect must never
        // share a crop embedding.
        cropKey: doDecode ? cropKeyFor(rect) + (bounded ? ':w' : '') : null,
        source: cropBitmap,
        proxyMask: { data: proxyBuf.buffer, width: proxyMask.width, height: proxyMask.height },
        proxySubrect,
        prompts: cropPrompts,
        doDecode,
        upFactor,
        compose,
        emitBlob,
    }, [cropBitmap, proxyBuf.buffer])
    if (res.stale) return null
    return {
        rect, proxySubrect, upFactor, bounded,
        blob: res.blob, cutout: res.cutout, alpha: res.alpha, width: res.width, height: res.height, decoded: res.decoded,
    }
}

// Rect-space prompts → the crop bitmap's own pixel space.
const scaleCropPrompts = (p, ds) => (Math.abs(ds - 1) < 1e-6 ? p : {
    clicks: p.clicks.map(([x, y, label]) => [x * ds, y * ds, label]),
    box: p.box ? p.box.map((v) => v * ds) : null,
    clampPoly: p.clampPoly ? p.clampPoly.map(([x, y]) => [x * ds, y * ds]) : null,
    clampMargin: (p.clampMargin || 0) * ds,
})

// The worker already fused the crop RGB with the matte (straight alpha), so the
// deliverable is just that crop-sized buffer painted to a canvas. No full-frame
// allocation, no full-original decode, no separate mask frame. The old
// full-frame path allocated roughly twice the original in RGBA and was the
// export memory spike. Used by the in-page test hook; the download path takes
// the worker-encoded Blob instead.
const cutoutCanvas = (cutout, width, height, decoded, rect) => {
    const c = makeCanvas(width, height)
    c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(cutout), width, height), 0, 0)
    return { canvas: c, width, height, decoded, rect }
}

const compositeProxyMask = async (proxyMask, tf, { exportMaxSide = 0, exportMaxMP = 0 } = {}) => {
    const sideScale = exportMaxSide ? exportMaxSide / Math.max(tf.originalW, tf.originalH) : 1
    const pixelScale = exportMaxMP ? Math.sqrt((exportMaxMP * 1e6) / (tf.originalW * tf.originalH)) : 1
    const es = Math.min(1, sideScale, pixelScale)
    const outW = Math.max(1, Math.round(tf.originalW * es))
    const outH = Math.max(1, Math.round(tf.originalH * es))
    const { source: original, owned } = await getOriginalForExport({ maxSide: exportMaxSide, maxMP: exportMaxMP })
    try {
        const out = makeCanvas(outW, outH)
        const octx = out.getContext('2d')
        octx.drawImage(original, 0, 0, outW, outH)
        const mask = makeCanvas(proxyMask.width, proxyMask.height)
        const data = new ImageData(new Uint8ClampedArray(proxyMask.data), proxyMask.width, proxyMask.height)
        for (let i = 0; i < data.data.length; i += 4) data.data[i + 3] = data.data[i]
        mask.getContext('2d').putImageData(data, 0, 0)
        octx.globalCompositeOperation = 'destination-in'
        octx.drawImage(mask, 0, 0, proxyMask.width, proxyMask.height, 0, 0, outW, outH)
        return { canvas: out, width: outW, height: outH, decoded: false }
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
    setHdPatch({ revision, rect, alpha: crop.alpha, width: crop.width, height: crop.height, decoded: true, bounded: crop.bounded })
    return crop
}

/** Full-res tight cutout for the proxy mask selected by prompts (both in
 *  proxy-space). Returns width, height, decoded, rect (crop offset in original
 *  px, single-object only) and bounded (a resolution cap applied), plus either a
 *  canvas (default, used by the test hook) or a worker-encoded blob (emitBlob,
 *  the download path). null when no original is held so the caller falls back to
 *  the proxy export. */
export const buildCutout = async (proxyMask, prompts, { budget, revision, preserveShape = false, emitBlob = false } = {}) => {
    if (!hasOriginal()) return null
    const tf = getTransform()
    if (!tf) return null

    // A manual union (multiple objects) is not a single crop, so it keeps the
    // proxy-res full-frame composite, bounded by exportMaxMP with no native decode.
    if (preserveShape) {
        return compositeProxyMask(proxyMask, tf, {
            exportMaxSide: budget?.exportMaxSide || 0,
            exportMaxMP: budget?.exportMaxMP || 0,
        })
    }

    const summary = summarizeMaskRGBA(proxyMask.data, proxyMask.width, proxyMask.height)
    if (!summary.bbox) return null
    const rect = cropRectFromBBox(summary.bbox, tf)
    // Always recompose fresh at full res: the worker fuses crop RGB and matte
    // into one straight-alpha cutout. The escalation patch stays for the live
    // overlay only, so export quality is never bounded by an interaction decode.
    // Crop-sized throughout, so there is no full-frame spike. On the download
    // path the worker also encodes the PNG and returns a blob, keeping the main
    // thread flat; the test hook takes the buffer and builds a canvas.
    const crop = await decodeCropAlpha(proxyMask, prompts, rect, tf, {
        budget, revision, nativeCrop: true, compose: true, emitBlob, caps: cropCaps(budget, { fullRes: true }),
    })
    if (!crop) return null
    // Decoded below the crop native width means a resolution cap applied (bounded
    // tier or memory pressure), not full sensor resolution.
    const bounded = crop.width < rect.w - 1
    if (emitBlob) {
        if (!crop.blob) return null
        return { blob: crop.blob, width: crop.width, height: crop.height, decoded: crop.decoded, rect, bounded }
    }
    if (!crop.cutout) return null
    const out = cutoutCanvas(crop.cutout, crop.width, crop.height, crop.decoded, rect)
    out.bounded = bounded
    return out
}

/** buildCutout to a PNG Blob (or null when there is no original to export). The
 *  native path is encoded in the worker so the main thread never holds the
 *  full-frame canvas; the preserveShape path returns a canvas encoded here. */
export const exportCutoutBlob = async (proxyMask, prompts, opts) => {
    const res = await buildCutout(proxyMask, prompts, { ...opts, emitBlob: true })
    if (!res) return null
    const blob = res.blob || await new Promise((r) => res.canvas.toBlob(r, 'image/png'))
    // bounded is undefined for the preserveShape full-frame path, true or false
    // only for the native tight-crop path (drives the export status wording).
    return { blob, width: res.width, height: res.height, decoded: res.decoded, bounded: res.bounded }
}

/**
 * decode-core — bounded image decoding shared by decode-worker (production)
 * and the main-thread inline fallback. No DOM elements; OffscreenCanvas only.
 * A worker does NOT add system memory — it exists for UI responsiveness,
 * explicit transfer ownership and lifecycle isolation.
 */

export const resizeOpts = (w, h, scale) => (scale < 1
    ? { resizeWidth: Math.max(1, Math.round(w * scale)), resizeHeight: Math.max(1, Math.round(h * scale)), resizeQuality: 'high' }
    : {})

/**
 * Decode `blob` to a ≤target bitmap. Prefers WebCodecs ImageDecoder (bounded
 * native decode); falls back to createImageBitmap's resize path. Orientation
 * ≠ 1 stays on createImageBitmap — this app applies EXIF transforms manually.
 */
export const decodeBoundedBitmap = async (blob, w, h, scale, orientation) => {
    const opts = resizeOpts(w, h, scale)
    const wantsResize = opts.resizeWidth && opts.resizeHeight
    if (wantsResize && orientation === 1 && typeof ImageDecoder !== 'undefined' && blob.type) {
        let decoder = null
        try {
            if (typeof ImageDecoder.isTypeSupported === 'function'
                && !(await ImageDecoder.isTypeSupported(blob.type))) {
                throw new Error(`ImageDecoder does not support ${blob.type}`)
            }
            decoder = new ImageDecoder({
                data: blob.stream(),
                type: blob.type,
                preferAnimation: false,
                desiredWidth: opts.resizeWidth,
                desiredHeight: opts.resizeHeight,
            })
            const { image } = await decoder.decode({ frameIndex: 0, completeFramesOnly: true })
            try {
                const atTarget = image.displayWidth === opts.resizeWidth
                    && image.displayHeight === opts.resizeHeight
                return await createImageBitmap(image, atTarget ? {} : opts)
            } finally {
                image.close()
            }
        } catch (err) {
            console.warn('[seglab][decode] ImageDecoder proxy fallback:', err?.message)
        } finally {
            try { decoder?.close() } catch { /* already closed */ }
        }
    }
    return createImageBitmap(blob, { imageOrientation: 'none', ...opts })
}

/**
 * Unbounded-decode hosts (no ImageDecoder / no scaled createImageBitmap) pay
 * one full-raster decode for an oversized upload no matter what. Pay it once:
 * also re-encode a ≤workingMaxSide copy, and (when `displaySide` asks) harvest
 * the ≤displaySide on-screen frame from the same raster so the display never
 * costs a second full decode. Returns
 * { bitmap, working: { blob, w, h } | null, display: ImageBitmap | null }.
 */
export const decodeWithWorkingCopy = async (blob, meta, scale, workingMaxSide, displaySide = 0) => {
    const full = await createImageBitmap(blob, { imageOrientation: 'none' })
    try {
        let working = null
        if (typeof OffscreenCanvas !== 'undefined') {
            const ws = workingMaxSide / Math.max(meta.w, meta.h)
            const wc = new OffscreenCanvas(
                Math.max(1, Math.round(meta.w * ws)),
                Math.max(1, Math.round(meta.h * ws)),
            )
            const wctx = wc.getContext('2d')
            wctx.imageSmoothingEnabled = true
            wctx.imageSmoothingQuality = 'high'
            wctx.drawImage(full, 0, 0, wc.width, wc.height)
            const type = blob.type === 'image/png' ? 'image/png' : 'image/jpeg'
            const encoded = await wc.convertToBlob({ type, quality: 0.92 }).catch(() => null)
            if (encoded) working = { blob: encoded, w: wc.width, h: wc.height }
        }
        let display = null
        if (displaySide > 0) {
            const ds = Math.min(1, displaySide / Math.max(meta.w, meta.h))
            display = await createImageBitmap(full, resizeOpts(meta.w, meta.h, ds)).catch(() => null)
        }
        const bitmap = await createImageBitmap(full, resizeOpts(meta.w, meta.h, scale))
        return { bitmap, working, display }
    } finally {
        full.close() // the full raster never outlives this call
    }
}

/** Opaque formats (no cheap header): one decode reveals dimensions, then the
 *  result is immediately bounded by `plan(w, h)` and the full bitmap closed. */
export const decodeOpaqueBounded = async (blob, plan) => {
    const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' })
    const p = plan(bitmap.width, bitmap.height)
    if (!p.proxyActive) return { bitmap, width: bitmap.width, height: bitmap.height, proxyActive: false }
    try {
        const bounded = await createImageBitmap(bitmap, resizeOpts(bitmap.width, bitmap.height, p.scale))
        return { bitmap: bounded, width: bitmap.width, height: bitmap.height, proxyActive: true }
    } finally {
        bitmap.close()
    }
}

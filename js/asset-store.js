/**
 * asset-store — custody of the original image without keeping a DSLR-sized
 * RGBA frame resident. Uploads retain compressed bytes only; the interaction
 * frame is a device-sized proxy (or a genuinely native small image). Native
 * crops and exports are decoded one bounded operation at a time.
 *
 * EXIF/raw-container orientation is applied manually to the bounded bitmap.
 * That keeps portrait DSLR files on the frugal Blob path too — we never need
 * to hold a full rotated canvas just to edit a portrait photo.
 */

import { readImageMeta } from './image-io.js'

const store = {
    blob: null,        // compressed original — never full-res RGBA between operations
    drawable: null,    // canvas/bitmap supplied by the demo or another in-memory caller
    transform: null,   // displayed + encoded dimensions and interaction-frame mapping
    assetKey: null,    // 'doc:WxH:hash' — same scheme the embedding cache uses
    workingBlob: null, // bounded re-decode source on unbounded-decode hosts (Safari)
    workingW: 0,       // encoded-orientation dims of workingBlob
    workingH: 0,
}

// Long-edge cap for the working copy: interaction-time re-decodes (detector
// frame, escalation crops) never cost more than an ~11 MP transient.
const WORKING_MAX_SIDE = 4096

// Safari never shipped ImageDecoder and its createImageBitmap cannot
// scaled-decode, so every blob decode there materializes the full raster.
const unboundedDecodeHost = (budget = {}) => (budget.workingMode === 'force'
    || (budget.workingMode !== 'off' && typeof ImageDecoder === 'undefined'))

const makeCanvas = (w, h) => {
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    return c
}

const validOrientation = (value) => (Number.isInteger(value) && value >= 1 && value <= 8 ? value : 1)
const swapsAxes = (orientation) => orientation >= 5 && orientation <= 8
const displayDims = (w, h, orientation) => (swapsAxes(orientation) ? { w: h, h: w } : { w, h })

// Canvas transforms for EXIF orientation. `w`/`h` describe the source's
// encoded pixels; target dimensions are supplied by displayDims(source.w,…).
const orientTransform = (orientation, w, h) => ({
    2: [-1, 0, 0, 1, w, 0],
    3: [-1, 0, 0, -1, w, h],
    4: [1, 0, 0, -1, 0, h],
    5: [0, 1, 1, 0, 0, 0],
    6: [0, 1, -1, 0, h, 0],
    7: [0, -1, -1, 0, h, w],
    8: [0, -1, 1, 0, 0, w],
}[orientation] || [1, 0, 0, 1, 0, 0])

const drawOriented = (target, source, orientation) => {
    const ctx = target.getContext('2d')
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.setTransform(...orientTransform(orientation, source.width, source.height))
    ctx.drawImage(source, 0, 0)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
}

/** Content hash: dims + FNV-1a over a 16×16 downsample (~1 ms). */
export const hashCanvas = (canvas) => {
    const c = makeCanvas(16, 16)
    const ctx = c.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(canvas, 0, 0, 16, 16)
    const px = ctx.getImageData(0, 0, 16, 16).data
    let h = 0x811c9dc5
    for (let i = 0; i < px.length; i += 1) {
        h ^= px[i]
        h = Math.imul(h, 0x01000193) >>> 0
    }
    return `${canvas.width}x${canvas.height}:${h.toString(16)}`
}

export const releaseAsset = () => {
    try { store.drawable?.close?.() } catch { /* canvas has no close */ }
    store.blob = null
    store.drawable = null
    store.transform = null
    store.assetKey = null
    store.workingBlob = null
    store.workingW = 0
    store.workingH = 0
}

/**
 * Decide whether this image actually needs a proxy. In auto mode a source at
 * or below the device cap is native-sized (`proxyActive:false`). `?proxy=off`
 * is honored only while the source is below a conservative direct-image
 * budget; larger sources immediately return to the device's safe proxy cap.
 */
const interactionPlan = (w, h, budget = {}) => {
    const longSide = Math.max(w, h)
    const directSafe = (w * h) <= (budget.directMaxMP || 4) * 1e6
        && longSide <= (budget.directMaxSide || 4096)
    const disabled = budget.proxyMode === 'disabled'
    const cap = disabled && directSafe
        ? longSide
        : (disabled ? (budget.safeProxyMax || 1024) : (budget.proxyMax || 1024))
    const scale = Math.min(1, cap / longSide)
    return {
        scale,
        proxyActive: scale < 1,
        proxyReason: disabled && !directSafe ? 'safety' : (scale < 1 ? 'device' : 'native'),
    }
}

const resizeOpts = (w, h, scale) => (scale < 1
    ? { resizeWidth: Math.max(1, Math.round(w * scale)), resizeHeight: Math.max(1, Math.round(h * scale)), resizeQuality: 'high' }
    : {})

/**
 * Decode an upload directly at the interaction size when the browser exposes
 * WebCodecs ImageDecoder. Safari's createImageBitmap resize path may still
 * transiently decode a full DSLR raster; desiredWidth/desiredHeight lets its
 * native decoder choose the bounded frame first. This path matters most for a
 * normal 24–60 MP DSLR JPEG: the prior 8 MP gate sent precisely those files
 * through `createImageBitmap`, whose Safari implementation may materialize a
 * full raster before shrinking it. EXIF orientations other than 1 stay on the
 * established createImageBitmap path because this app applies their transform
 * manually and must not risk a browser-specific double rotate.
 */
const decodeProxyBlob = async (blob, w, h, scale, orientation) => {
    const opts = resizeOpts(w, h, scale)
    const wantsResize = opts.resizeWidth && opts.resizeHeight
    if (wantsResize && orientation === 1 && typeof ImageDecoder !== 'undefined' && blob.type) {
        let decoder = null
        try {
            // Avoid invoking a partial WebCodecs implementation for a codec
            // it explicitly does not support. `isTypeSupported` is optional,
            // so old implementations still get the guarded constructor try.
            if (typeof ImageDecoder.isTypeSupported === 'function'
                && !(await ImageDecoder.isTypeSupported(blob.type))) {
                throw new Error(`ImageDecoder does not support ${blob.type}`)
            }
            // Chromium and current Safari accept a ReadableStream here but
            // reject Blob directly. Streaming keeps the compressed upload
            // out of a second ArrayBuffer while the decoder builds the proxy.
            decoder = new ImageDecoder({
                data: blob.stream(),
                type: blob.type,
                preferAnimation: false,
                // These are ImageDecoderInit members (not decode() options).
                // Supplying them there makes the native decoder emit the
                // bounded proxy frame instead of a full-size VideoFrame.
                desiredWidth: opts.resizeWidth,
                desiredHeight: opts.resizeHeight,
            })
            const { image } = await decoder.decode({
                frameIndex: 0,
                completeFramesOnly: true,
            })
            try {
                // A compliant decoder has already produced the target-sized
                // VideoFrame, so do not ask for another resample/copy. The
                // resize fallback handles an implementation that accepted but
                // ignored desiredWidth/desiredHeight without exposing that
                // large frame to the editor canvas.
                const atTarget = image.displayWidth === opts.resizeWidth
                    && image.displayHeight === opts.resizeHeight
                const bitmap = await createImageBitmap(image, atTarget ? {} : opts)
                console.log('[seglab][asset] proxy-decoder', {
                    mode: 'ImageDecoder',
                    decoded: `${image.displayWidth}x${image.displayHeight}`,
                    target: `${opts.resizeWidth}x${opts.resizeHeight}`,
                    width: bitmap.width,
                    height: bitmap.height,
                })
                return bitmap
            } finally {
                image.close()
            }
        } catch (err) {
            // WebCodecs support varies by codec and Safari version. The
            // createImageBitmap path below remains the compatible fallback.
            console.warn('[seglab][asset] ImageDecoder proxy fallback:', err?.message)
        } finally {
            try { decoder?.close() } catch { /* already closed */ }
        }
    }
    return createImageBitmap(blob, { imageOrientation: 'none', ...opts })
}

/**
 * Unbounded-decode hosts pay one full-raster decode for any oversized upload
 * no matter what. Pay it exactly once: keep a ≤WORKING_MAX_SIDE re-encoded
 * copy (encoded orientation, no EXIF) so detector frames and escalation crops
 * re-decode ~11 MP instead of the whole DSLR raster. Returns the proxy bitmap.
 */
const decodeWithWorking = async (blob, meta, scale) => {
    const full = await createImageBitmap(blob, { imageOrientation: 'none' })
    try {
        const ws = WORKING_MAX_SIDE / Math.max(meta.w, meta.h)
        const wc = makeCanvas(Math.max(1, Math.round(meta.w * ws)), Math.max(1, Math.round(meta.h * ws)))
        const wctx = wc.getContext('2d')
        wctx.imageSmoothingEnabled = true
        wctx.imageSmoothingQuality = 'high'
        wctx.drawImage(full, 0, 0, wc.width, wc.height)
        const type = blob.type === 'image/png' ? 'image/png' : 'image/jpeg'
        const encoded = await new Promise((resolve) => wc.toBlob(resolve, type, 0.92))
        if (encoded) {
            store.workingBlob = encoded
            store.workingW = wc.width
            store.workingH = wc.height
            console.log('[seglab][asset] working-blob', {
                from: `${meta.w}x${meta.h}`, to: `${wc.width}x${wc.height}`, bytes: encoded.size,
            })
        }
        return await createImageBitmap(full, resizeOpts(meta.w, meta.h, scale))
    } finally {
        full.close()
    }
}

const putInteractionFrame = (proxyCanvas, source, orientation) => {
    const dims = displayDims(source.width, source.height, orientation)
    proxyCanvas.width = dims.w
    proxyCanvas.height = dims.h
    drawOriented(proxyCanvas, source, orientation)
}

/**
 * Take custody of a new original and draw its interaction frame into
 * `proxyCanvas`. Blob inputs remain Blob-only even when portrait-oriented;
 * `orientation` lets RAW container metadata override a preview JPEG's EXIF.
 */
export const importOriginal = async (source, {
    budget = {}, proxyCanvas, proxyBlob = null, orientation: orientationOverride = null,
    sourceWasRaw = false, sourceBytes = source?.size || 0,
} = {}) => {
    releaseAsset()

    if (source instanceof Blob) {
        const meta = await readImageMeta(source)
        if (meta?.w && meta?.h) {
            const orientation = validOrientation(orientationOverride ?? meta.orientation)
            const displayed = displayDims(meta.w, meta.h, orientation)
            const plan = interactionPlan(displayed.w, displayed.h, budget)
            // For a real proxy, a RAW's small embedded preview is the safest
            // source. A native-sized frame always uses the full preview/image.
            const proxySource = plan.proxyActive && proxyBlob ? proxyBlob : source
            const proxyMeta = proxySource === source ? meta : await readImageMeta(proxySource)
            const targetW = Math.min(
                Math.max(1, Math.round(meta.w * plan.scale)),
                proxyMeta?.w || meta.w,
            )
            const targetH = Math.min(
                Math.max(1, Math.round(meta.h * plan.scale)),
                proxyMeta?.h || meta.h,
            )
            console.log('[seglab][asset] import-plan', {
                original: `${meta.w}x${meta.h}`,
                decode: `${proxyMeta?.w || meta.w}x${proxyMeta?.h || meta.h}`,
                target: `${targetW}x${targetH}`,
                proxy: proxySource !== source,
            })
            const decodeW = proxyMeta?.w || meta.w
            const decodeH = proxyMeta?.h || meta.h
            const decodeScale = Math.min(1, targetW / decodeW, targetH / decodeH)
            const wantWorking = plan.proxyActive && unboundedDecodeHost(budget)
            let bitmap
            if (wantWorking && proxySource === source && Math.max(meta.w, meta.h) > WORKING_MAX_SIDE) {
                bitmap = await decodeWithWorking(source, meta, decodeScale)
            } else {
                bitmap = await decodeProxyBlob(proxySource, decodeW, decodeH, decodeScale, orientation)
                // A RAW's small embedded preview is already a bounded decode
                // source — retain it instead of re-encoding anything.
                if (wantWorking && proxySource !== source && (proxyMeta?.w || 0) < meta.w) {
                    store.workingBlob = proxySource
                    store.workingW = proxyMeta.w
                    store.workingH = proxyMeta.h
                }
            }
            try {
                putInteractionFrame(proxyCanvas, bitmap, orientation)
            } finally {
                bitmap.close()
            }
            store.blob = source
            store.transform = {
                originalW: displayed.w,
                originalH: displayed.h,
                encodedW: meta.w,
                encodedH: meta.h,
                orientation,
                proxyW: proxyCanvas.width,
                proxyH: proxyCanvas.height,
                scale: proxyCanvas.width / displayed.w,
                proxyActive: plan.proxyActive,
                proxyReason: plan.proxyReason,
                workingActive: !!store.workingBlob,
                workingW: store.workingW,
                workingH: store.workingH,
                sourceWasRaw,
                sourceBytes,
            }
            store.assetKey = `doc:${hashCanvas(proxyCanvas)}`
            console.log('[seglab][asset] import-ready', { proxy: `${proxyCanvas.width}x${proxyCanvas.height}`, proxyActive: plan.proxyActive, working: !!store.workingBlob })
            return { ...store.transform }
        }

        // Uncommon formats without a cheap header parser still avoid resident
        // full-res pixels. The browser must decode once to reveal dimensions,
        // then the bitmap is immediately reduced and released. `from-image`
        // is the only reliable orientation signal for these opaque formats.
        const bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' })
        try {
            const plan = interactionPlan(bitmap.width, bitmap.height, budget)
            const target = plan.proxyActive
                ? await createImageBitmap(bitmap, resizeOpts(bitmap.width, bitmap.height, plan.scale))
                : bitmap
            try {
                proxyCanvas.width = target.width
                proxyCanvas.height = target.height
                proxyCanvas.getContext('2d').drawImage(target, 0, 0)
            } finally {
                if (target !== bitmap) target.close()
            }
            store.blob = source
            store.transform = {
                originalW: bitmap.width,
                originalH: bitmap.height,
                encodedW: bitmap.width,
                encodedH: bitmap.height,
                orientation: 1,
                proxyW: proxyCanvas.width,
                proxyH: proxyCanvas.height,
                scale: proxyCanvas.width / bitmap.width,
                proxyActive: plan.proxyActive,
                proxyReason: plan.proxyReason,
                opaqueFormat: true,
                sourceWasRaw,
                sourceBytes,
            }
            store.assetKey = `doc:${hashCanvas(proxyCanvas)}`
            return { ...store.transform }
        } finally {
            bitmap.close()
        }
    }

    // Drawable source (demo canvas / in-memory caller). It is the only path
    // permitted to remain resident because the caller already owns pixels.
    const originalW = source?.width
    const originalH = source?.height
    if (!originalW || !originalH) throw new Error('Image has no usable dimensions')
    const plan = interactionPlan(originalW, originalH, budget)
    const frame = plan.proxyActive
        ? await createImageBitmap(source, resizeOpts(originalW, originalH, plan.scale))
        : source
    try {
        proxyCanvas.width = frame.width
        proxyCanvas.height = frame.height
        proxyCanvas.getContext('2d').drawImage(frame, 0, 0)
    } finally {
        if (frame !== source) frame.close()
    }
    store.drawable = source
    store.transform = {
        originalW,
        originalH,
        encodedW: originalW,
        encodedH: originalH,
        orientation: 1,
        proxyW: proxyCanvas.width,
        proxyH: proxyCanvas.height,
        scale: proxyCanvas.width / originalW,
        proxyActive: plan.proxyActive,
        proxyReason: plan.proxyReason,
        sourceWasRaw: false,
        sourceBytes: 0,
    }
    store.assetKey = `doc:${hashCanvas(proxyCanvas)}`
    return { ...store.transform }
}

export const hasOriginal = () => !!(store.blob || store.drawable)
export const getTransform = () => (store.transform ? { ...store.transform } : null)
export const getAssetKey = () => store.assetKey

/** Crop embedding key — never collides with `doc:` keys. */
export const cropKeyFor = (rect) => (store.assetKey
    ? `crop:${store.assetKey.slice(4)}:${rect.x},${rect.y},${rect.w},${rect.h}`
    : null)

/** Resize options bounded by both long edge and total pixels. */
const downOpts = (w, h, { maxSide = 0, maxMP = 0 } = {}) => {
    const sideScale = maxSide ? maxSide / Math.max(w, h) : 1
    const pixelScale = maxMP ? Math.sqrt((maxMP * 1e6) / (w * h)) : 1
    const down = Math.min(1, sideScale, pixelScale)
    return resizeOpts(w, h, down)
}

const orientToCanvas = (bitmap, orientation) => {
    if (orientation === 1) return bitmap
    const dims = displayDims(bitmap.width, bitmap.height, orientation)
    const canvas = makeCanvas(dims.w, dims.h)
    drawOriented(canvas, bitmap, orientation)
    return canvas
}

// Display-space point → encoded-space point. Applying this to all four crop
// corners gives an axis-aligned encoded crop for every EXIF orientation.
const displayToEncoded = (x, y, t) => {
    const { encodedW: w, encodedH: h, orientation } = t
    switch (orientation) {
    case 2: return [w - x, y]
    case 3: return [w - x, h - y]
    case 4: return [x, h - y]
    case 5: return [y, x]
    case 6: return [y, h - x]
    case 7: return [w - y, h - x]
    case 8: return [w - y, x]
    default: return [x, y]
    }
}

const displayRectToEncoded = (rect, t) => {
    const corners = [
        displayToEncoded(rect.x, rect.y, t),
        displayToEncoded(rect.x + rect.w, rect.y, t),
        displayToEncoded(rect.x, rect.y + rect.h, t),
        displayToEncoded(rect.x + rect.w, rect.y + rect.h, t),
    ]
    const xs = corners.map(([x]) => x)
    const ys = corners.map(([, y]) => y)
    const x0 = Math.max(0, Math.floor(Math.min(...xs)))
    const y0 = Math.max(0, Math.floor(Math.min(...ys)))
    const x1 = Math.min(t.encodedW, Math.ceil(Math.max(...xs)))
    const y1 = Math.min(t.encodedH, Math.ceil(Math.max(...ys)))
    return { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) }
}

/** Decode the original bounded by output limits; `native` bypasses the
 *  working copy (export truth — everything else takes the bounded source). */
const getDecodedOriginal = async ({ maxSide = 0, maxMP = 0, native = false } = {}) => {
    const t = store.transform
    if (!t) return { source: null, owned: false }
    if (store.drawable) {
        const opts = downOpts(t.originalW, t.originalH, { maxSide, maxMP })
        if (!Object.keys(opts).length) return { source: store.drawable, owned: false }
        return { source: await createImageBitmap(store.drawable, opts), owned: true }
    }
    if (!native && store.workingBlob) {
        const bitmap = await createImageBitmap(store.workingBlob, {
            imageOrientation: 'none',
            ...downOpts(store.workingW, store.workingH, { maxSide, maxMP }),
        })
        if (t.orientation === 1) return { source: bitmap, owned: true }
        try {
            return { source: orientToCanvas(bitmap, t.orientation), owned: true }
        } finally {
            bitmap.close()
        }
    }
    if (!store.blob) return { source: null, owned: false }
    const bitmap = await createImageBitmap(store.blob, {
        imageOrientation: t.opaqueFormat ? 'from-image' : 'none',
        ...downOpts(t.encodedW, t.encodedH, { maxSide, maxMP }),
    })
    if (t.orientation === 1) return { source: bitmap, owned: true }
    try {
        return { source: orientToCanvas(bitmap, t.orientation), owned: true }
    } finally {
        bitmap.close()
    }
}

/**
 * Display-space original crop as an ImageBitmap, bounded before transfer to
 * the worker. Portrait/mirrored photos map the crop into encoded coordinates,
 * then orient only that bounded crop — never the whole original.
 */
export const getCropBitmap = async (rect, { maxSide = 4096, maxMP = 0, native = false } = {}) => {
    const t = store.transform
    if (!t) throw new Error('No original asset held')
    if (store.drawable) {
        return createImageBitmap(store.drawable, rect.x, rect.y, rect.w, rect.h, downOpts(rect.w, rect.h, { maxSide, maxMP }))
    }
    if (!native && store.workingBlob) {
        const enc = displayRectToEncoded(rect, t)
        const ws = store.workingW / t.encodedW
        const wx = Math.max(0, Math.floor(enc.x * ws))
        const wy = Math.max(0, Math.floor(enc.y * ws))
        const ww = Math.max(1, Math.min(store.workingW - wx, Math.ceil(enc.w * ws)))
        const wh = Math.max(1, Math.min(store.workingH - wy, Math.ceil(enc.h * ws)))
        const bitmap = await createImageBitmap(store.workingBlob, wx, wy, ww, wh, {
            imageOrientation: 'none',
            ...downOpts(ww, wh, { maxSide, maxMP }),
        })
        if (t.orientation === 1) return bitmap
        try {
            const canvas = orientToCanvas(bitmap, t.orientation)
            return await createImageBitmap(canvas)
        } finally {
            bitmap.close()
        }
    }
    if (!store.blob) throw new Error('No original asset held')
    const encoded = displayRectToEncoded(rect, t)
    const bitmap = await createImageBitmap(store.blob, encoded.x, encoded.y, encoded.w, encoded.h, {
        imageOrientation: 'none',
        ...downOpts(encoded.w, encoded.h, { maxSide, maxMP }),
    })
    if (t.orientation === 1) return bitmap
    try {
        const canvas = orientToCanvas(bitmap, t.orientation)
        return await createImageBitmap(canvas)
    } finally {
        bitmap.close()
    }
}

/** Full-resolution-or-bounded original for export compositing; close if owned. */
export const getOriginalForExport = (opts = {}) => getDecodedOriginal({ ...opts, native: true })

/** Interaction-time frame (detector, previews): prefers the working copy. */
export const getBoundedOriginal = (opts = {}) => getDecodedOriginal(opts)

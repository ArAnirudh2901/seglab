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

import { readMeta, decodeProxy, decodeOpaque, STALE } from './decode-client.js'
import { resizeOpts } from './decode-core.js'
import { interactionPlan, displayPlan } from './proxy-plan.js'

const store = {
    blob: null,        // compressed original — never full-res RGBA between operations
    drawable: null,    // canvas/bitmap supplied by the demo or another in-memory caller
    transform: null,   // displayed + encoded dimensions and interaction-frame mapping
    assetKey: null,    // 'doc:WxH:hash' — same scheme the embedding cache uses
    workingBlob: null, // bounded re-decode source on unbounded-decode hosts (Safari)
    workingW: 0,       // encoded-orientation dims of workingBlob
    workingH: 0,
}

// Long-edge cap for the working copy comes from the budget (lite 1280): the
// bounded re-decode source for detector frames and escalation crops.
const workingMaxSide = (budget = {}) => budget.workingMaxSide || 1280

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

const putInteractionFrame = (proxyCanvas, source, orientation) => {
    const dims = displayDims(source.width, source.height, orientation)
    proxyCanvas.width = dims.w
    proxyCanvas.height = dims.h
    drawOriented(proxyCanvas, source, orientation)
}

const copyCanvas = (dst, src) => {
    dst.width = src.width
    dst.height = src.height
    dst.getContext('2d').drawImage(src, 0, 0)
}

// Viewport signal for the display formula: what the screen can physically
// show. displayPlan clamps DPR and adds the zoom slack.
const currentViewport = () => {
    if (typeof window === 'undefined') return { w: 0, h: 0, dpr: 1 }
    const vv = window.visualViewport
    return {
        w: Math.round(vv?.width || window.innerWidth || 0),
        h: Math.round(vv?.height || window.innerHeight || 0),
        dpr: window.devicePixelRatio || 1,
    }
}

const displayPlanFor = (srcW, srcH, budget = {}) => displayPlan({
    srcW,
    srcH,
    budget,
    viewport: currentViewport(),
    textureLimit: budget.textureLimit || 0,
})

/**
 * Draw a crisp display bitmap into `displayCanvas`, staying bounded-safe. A
 * bounded-decode host (ImageDecoder) scale-decodes the full frame cheaply; an
 * unbounded host decodes the largest source whose one-shot full raster fits
 * the decode budget (displayPlan.allowFullDecode) — preferring the ORIGINAL —
 * or, with displayMode 'native', pays one uncapped full-res decode. On any
 * miss the model proxy (`proxyCanvas`, already oriented) is shown, so the
 * preview never regresses.
 */
const renderDisplayFrame = async (displayCanvas, proxyCanvas, {
    budget, fullBlob, fullW, fullH, safeBlob, safeW, safeH, orientation, revision, isCurrent,
}) => {
    let src = fullBlob
    let sW = fullW
    let sH = fullH
    let plan = displayPlanFor(sW, sH, budget)
    if (!plan.side) { copyCanvas(displayCanvas, proxyCanvas); return }
    const boundedHost = typeof ImageDecoder !== 'undefined'

    if (!boundedHost && !plan.allowFullDecode) {
        // The original's one-shot raster exceeds the decode budget here: fall
        // back to the largest bounded source (RAW preview / working copy) that
        // fits; otherwise the proxy stays the preview.
        const safePlan = (safeBlob && safeW) ? displayPlanFor(safeW, safeH, budget) : null
        if (safePlan?.allowFullDecode && safePlan.side) {
            src = safeBlob; sW = safeW; sH = safeH; plan = safePlan
        } else { copyCanvas(displayCanvas, proxyCanvas); return }
    }
    if (!src) { copyCanvas(displayCanvas, proxyCanvas); return }

    const scale = Math.min(1, plan.side / Math.max(sW, sH))
    // Skip the decode when the display would not out-resolve the proxy already shown.
    if (Math.round(Math.max(sW, sH) * scale) <= Math.max(proxyCanvas.width, proxyCanvas.height)) {
        copyCanvas(displayCanvas, proxyCanvas)
        return
    }
    try {
        const res = await decodeProxy({ blob: src, decodeW: sW, decodeH: sH, scale, orientation, revision, isCurrent })
        if (res === STALE) return // a newer import owns the canvases now
        if (!res?.bitmap) { copyCanvas(displayCanvas, proxyCanvas); return }
        try { putInteractionFrame(displayCanvas, res.bitmap, orientation) } finally { res.bitmap.close() }
    } catch (err) {
        console.warn('[seglab][asset] display decode fell back to proxy:', err?.message)
        copyCanvas(displayCanvas, proxyCanvas)
    }
}

/**
 * Take custody of a new original and draw its interaction frame into
 * `proxyCanvas`. Blob inputs remain Blob-only even when portrait-oriented;
 * `orientation` lets RAW container metadata override a preview JPEG's EXIF.
 */
export const importOriginal = async (source, {
    budget = {}, proxyCanvas, displayCanvas = null, proxyBlob = null, orientation: orientationOverride = null,
    sourceWasRaw = false, sourceBytes = source?.size || 0,
    revision = null, isCurrent = null,
} = {}) => {
    releaseAsset()

    if (source instanceof Blob) {
        const meta = await readMeta(source)
        if (meta?.w && meta?.h) {
            const orientation = validOrientation(orientationOverride ?? meta.orientation)
            const displayed = displayDims(meta.w, meta.h, orientation)
            const plan = interactionPlan(displayed.w, displayed.h, budget)
            // For a real proxy, a RAW's small embedded preview is the safest
            // source. A native-sized frame always uses the full preview/image.
            const proxySource = plan.proxyActive && proxyBlob ? proxyBlob : source
            const proxyMeta = proxySource === source ? meta : await readMeta(proxySource)
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
            const wMax = workingMaxSide(budget)
            const wantWorking = plan.proxyActive && unboundedDecodeHost(budget)
            // A RAW's small embedded preview is already a bounded decode
            // source — retain it instead of re-encoding anything.
            const rawWorking = wantWorking && proxySource !== source && (proxyMeta?.w || 0) < meta.w
            const payFullDecode = wantWorking && proxySource === source && Math.max(meta.w, meta.h) > wMax
            // The import is about to pay the one full-raster decode anyway —
            // harvest the on-screen frame from it so the display never costs a
            // second one.
            const displayTarget = displayCanvas ? displayPlanFor(displayed.w, displayed.h, budget) : { side: 0 }
            const harvestSide = payFullDecode && displayTarget.side > Math.max(targetW, targetH)
                ? displayTarget.side
                : 0
            const res = await decodeProxy({
                blob: proxySource,
                decodeW,
                decodeH,
                scale: decodeScale,
                orientation,
                wantWorking: payFullDecode,
                workingMaxSide: wMax,
                displaySide: harvestSide,
                revision,
                isCurrent,
            })
            if (res === STALE) return null
            const { bitmap, working, display } = res
            if (working) {
                store.workingBlob = working.blob
                store.workingW = working.w
                store.workingH = working.h
                console.log('[seglab][asset] working-blob', {
                    from: `${meta.w}x${meta.h}`, to: `${working.w}x${working.h}`, bytes: working.blob.size,
                })
            } else if (rawWorking) {
                store.workingBlob = proxySource
                store.workingW = proxyMeta.w
                store.workingH = proxyMeta.h
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
            if (display && displayCanvas) {
                // Harvested from the single import decode — no second decode.
                try { putInteractionFrame(displayCanvas, display, orientation) } finally { display.close() }
                console.log('[seglab][asset] display-ready', { display: `${displayCanvas.width}x${displayCanvas.height}`, mode: budget.displayMode || 'auto', harvested: true })
            } else if (displayCanvas) {
                if (display) display.close()
                // Safe display source on unbounded hosts: the RAW's own smaller
                // embedded preview, else the bounded working copy.
                const safe = (proxySource !== source)
                    ? { blob: proxySource, w: proxyMeta?.w || 0, h: proxyMeta?.h || 0 }
                    : (store.workingBlob ? { blob: store.workingBlob, w: store.workingW, h: store.workingH } : { blob: null, w: 0, h: 0 })
                await renderDisplayFrame(displayCanvas, proxyCanvas, {
                    budget, fullBlob: source, fullW: meta.w, fullH: meta.h,
                    safeBlob: safe.blob, safeW: safe.w, safeH: safe.h,
                    orientation, revision, isCurrent,
                })
                console.log('[seglab][asset] display-ready', { display: `${displayCanvas.width}x${displayCanvas.height}`, mode: budget.displayMode || 'auto' })
            } else if (display) display.close()
            return { ...store.transform }
        }

        // Uncommon formats without a cheap header parser still avoid resident
        // full-res pixels: the worker decodes once to reveal dimensions, then
        // immediately bounds and releases the full bitmap. `from-image` is the
        // only reliable orientation signal for these opaque formats.
        const res = await decodeOpaque({ blob: source, budget, revision, isCurrent })
        if (res === STALE) return null
        const { bitmap: target, original, proxyActive } = res
        try {
            proxyCanvas.width = target.width
            proxyCanvas.height = target.height
            proxyCanvas.getContext('2d').drawImage(target, 0, 0)
        } finally {
            target.close()
        }
        store.blob = source
        store.transform = {
            originalW: original.width,
            originalH: original.height,
            encodedW: original.width,
            encodedH: original.height,
            orientation: 1,
            proxyW: proxyCanvas.width,
            proxyH: proxyCanvas.height,
            scale: proxyCanvas.width / original.width,
            proxyActive,
            proxyReason: proxyActive ? 'device' : 'native',
            opaqueFormat: true,
            sourceWasRaw,
            sourceBytes,
        }
        store.assetKey = `doc:${hashCanvas(proxyCanvas)}`
        if (displayCanvas) copyCanvas(displayCanvas, proxyCanvas)
        return { ...store.transform }
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
    if (displayCanvas) {
        // The caller already owns full-res pixels — draw them into the display
        // at a bounded size (no decode). Falls back to the proxy when disabled.
        const dPlan = displayPlanFor(originalW, originalH, budget)
        if (!dPlan.side) {
            copyCanvas(displayCanvas, proxyCanvas)
        } else if (dPlan.side >= Math.max(originalW, originalH)) {
            copyCanvas(displayCanvas, source)
        } else {
            const s = Math.min(1, dPlan.side / Math.max(originalW, originalH))
            displayCanvas.width = Math.max(1, Math.round(originalW * s))
            displayCanvas.height = Math.max(1, Math.round(originalH * s))
            const dctx = displayCanvas.getContext('2d')
            dctx.imageSmoothingEnabled = true
            dctx.imageSmoothingQuality = 'high'
            dctx.drawImage(source, 0, 0, displayCanvas.width, displayCanvas.height)
        }
    }
    return { ...store.transform }
}

export const hasOriginal = () => !!(store.blob || store.drawable)

/** Compressed original bytes for session persistence; null for demo drawables. */
export const getOriginalBlob = () => store.blob
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

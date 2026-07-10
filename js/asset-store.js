/**
 * asset-store — custody of the ORIGINAL image (main thread).
 * Original pixels are the export truth; the ≤1024 #view is a disposable
 * interaction proxy. Export/crop re-encodes read native res from here.
 * Custody is budget-aware: resident ImageBitmap ≤ bitmapMaxMP, else Blob +
 * re-decode on demand (trades memory for decode seconds, never capability).
 */

const store = {
    bitmap: null,      // ImageBitmap | HTMLCanvasElement | OffscreenCanvas | null
    blob: null,        // source Blob/File when bitmap custody is not allowed
    transform: null,   // { originalW, originalH, proxyW, proxyH, scale }
    assetKey: null,    // 'doc:WxH:hash' — same scheme the embedding cache uses
}

/** Content hash: dims + FNV-1a over a 16×16 downsample (~1 ms). */
export const hashCanvas = (canvas) => {
    const c = document.createElement('canvas')
    c.width = 16
    c.height = 16
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
    try { store.bitmap?.close?.() } catch { /* already closed */ }
    store.bitmap = null
    store.blob = null
    store.transform = null
    store.assetKey = null
}

/** Take custody of a new original and draw the ≤budget.proxyMax proxy into
 *  `proxyCanvas`. Returns the transform. */
export const importOriginal = (source, { blob = null, budget, proxyCanvas }) => {
    releaseAsset()
    const originalW = source.width
    const originalH = source.height
    if (!originalW || !originalH) throw new Error('Image has no usable dimensions')

    const scale = Math.min(1, (budget.proxyMax || 1024) / Math.max(originalW, originalH))
    proxyCanvas.width = Math.max(1, Math.round(originalW * scale))
    proxyCanvas.height = Math.max(1, Math.round(originalH * scale))
    proxyCanvas.getContext('2d').drawImage(source, 0, 0, proxyCanvas.width, proxyCanvas.height)

    store.transform = {
        originalW,
        originalH,
        proxyW: proxyCanvas.width,
        proxyH: proxyCanvas.height,
        scale: proxyCanvas.width / originalW,
    }
    store.assetKey = `doc:${hashCanvas(proxyCanvas)}`

    const mp = (originalW * originalH) / 1e6
    if (mp <= (budget.bitmapMaxMP ?? 24) || !blob) {
        store.bitmap = source // custody transferred — caller must NOT close it
    } else {
        store.blob = blob     // over budget: keep only the blob, re-decode later
        try { source.close?.() } catch { /* canvas source */ }
    }
    return { ...store.transform }
}

export const hasOriginal = () => !!(store.bitmap || store.blob)
export const getTransform = () => (store.transform ? { ...store.transform } : null)
export const getAssetKey = () => store.assetKey

/** Crop embedding key — never collides with `doc:` keys. */
export const cropKeyFor = (rect) => (store.assetKey
    ? `crop:${store.assetKey.slice(4)}:${rect.x},${rect.y},${rect.w},${rect.h}`
    : null)

/** Decode the original; caller closes the result only when `owned`. */
const getDecodedOriginal = async () => {
    if (store.bitmap) return { source: store.bitmap, owned: false }
    if (!store.blob) return { source: null, owned: false }
    const bmp = await createImageBitmap(store.blob, { imageOrientation: 'from-image' })
    return { source: bmp, owned: true }
}

/** Original-res crop {x,y,w,h} as an ImageBitmap, downscaled if > maxSide. */
export const getCropBitmap = async (rect, { maxSide = 4096 } = {}) => {
    const { source, owned } = await getDecodedOriginal()
    if (!source) throw new Error('No original asset held')
    try {
        const down = Math.min(1, maxSide / Math.max(rect.w, rect.h))
        const opts = down < 1
            ? { resizeWidth: Math.round(rect.w * down), resizeHeight: Math.round(rect.h * down), resizeQuality: 'high' }
            : {}
        return await createImageBitmap(source, rect.x, rect.y, rect.w, rect.h, opts)
    } finally {
        if (owned) { try { source.close() } catch { /* closed */ } }
    }
}

/** Full-res original for export compositing; close only when `owned`. */
export const getOriginalForExport = () => getDecodedOriginal()

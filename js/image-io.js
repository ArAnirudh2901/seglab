/**
 * image-io — bounded image sizing, header parsing, and proxy decoding
 * ---------------------------------------------------------------------
 * The custody contract for original images: only a compressed Blob and a
 * ≤ maxLongSide ImageBitmap may exist. Dimensions and EXIF orientation are
 * parsed from a bounded byte window (never the whole file), and decoding
 * asks the browser to resize DURING decode. JS never retains a
 * full-resolution frame; the one unavoidable-full-decode fallback (unknown
 * container) closes the full bitmap before returning.
 *
 * Pure ESM: header parsing runs under node (verify.mjs) as well as in the
 * decode worker and, as a fallback, on the main thread.
 */

/** Single authoritative proxy-size formula. Throws on invalid dimensions. */
export function getBoundedProxySize(sourceWidth, sourceHeight, maxLongSide = 768) {
    if (
        !Number.isFinite(sourceWidth)
        || !Number.isFinite(sourceHeight)
        || sourceWidth <= 0
        || sourceHeight <= 0
    ) {
        throw new Error('Invalid image dimensions')
    }
    const longSide = Math.max(sourceWidth, sourceHeight)
    const scale = Math.min(1, maxLongSide / longSide)
    return {
        width: Math.max(1, Math.round(sourceWidth * scale)),
        height: Math.max(1, Math.round(sourceHeight * scale)),
        scale,
        proxyActive: scale < 1,
    }
}

/** EXIF orientations 5–8 transpose the displayed frame. */
export const orientedSize = (width, height, orientation = 1) => (
    orientation >= 5 && orientation <= 8 ? { width: height, height: width } : { width, height }
)

/** One bounded read window covers every parser (TIFF IFD tables included). */
export const HEADER_READ_BYTES = 4 * 1024 * 1024

/* ─── Header parsers ─────────────────────────────────────────────────────── */

const ascii = (view, ofs, len) => {
    let s = ''
    for (let i = 0; i < len; i += 1) s += String.fromCharCode(view.getUint8(ofs + i))
    return s
}

const u16 = (view, ofs, le) => (ofs + 2 <= view.byteLength ? view.getUint16(ofs, le) : null)
const u32 = (view, ofs, le) => (ofs + 4 <= view.byteLength ? view.getUint32(ofs, le) : null)

/**
 * TIFF/EXIF IFD walk. `base` = offset of the TIFF header ("II"/"MM") inside
 * the window. Collects IFD0 dims + orientation and every JPEG-preview span
 * (tags 0x0201/0x0202, incl. SubIFDs) — how CR2/NEF/ARW/DNG expose their
 * embedded previews. Offsets in `previews` are relative to `base`.
 */
const parseTiffIfds = (view, base) => {
    const end = view.byteLength
    const b0 = u16(view, base, false)
    const le = b0 === 0x4949
    if (!le && b0 !== 0x4d4d) return null
    if (u16(view, base + 2, le) !== 42) return null

    const out = { width: null, height: null, orientation: 1, previews: [] }
    const visited = new Set()
    let orientationSet = false

    const readIfd = (rel, depth) => {
        if (depth > 4 || !rel || visited.has(rel)) return
        visited.add(rel)
        const ofs = base + rel
        const n = u16(view, ofs, le)
        if (n === null) return
        const count = Math.min(n, 512)
        let jpegOfs = null
        let jpegLen = null
        let width = null
        let height = null
        const subIfds = []
        for (let i = 0; i < count; i += 1) {
            const e = ofs + 2 + i * 12
            if (e + 12 > end) break
            const tag = view.getUint16(e, le)
            const type = view.getUint16(e + 2, le)
            const cnt = view.getUint32(e + 4, le)
            const val = type === 3 ? view.getUint16(e + 8, le) : view.getUint32(e + 8, le)
            if (tag === 0x0100) width = val
            else if (tag === 0x0101) height = val
            else if (tag === 0x0112 && !orientationSet && val >= 1 && val <= 8) {
                out.orientation = val
                orientationSet = true
            } else if (tag === 0x0201) jpegOfs = val
            else if (tag === 0x0202) jpegLen = val
            else if (tag === 0x014a) {
                if (cnt === 1) subIfds.push(val)
                else {
                    for (let k = 0; k < Math.min(cnt, 4); k += 1) {
                        const p = base + view.getUint32(e + 8, le) + k * 4
                        const sub = u32(view, p, le)
                        if (sub !== null) subIfds.push(sub)
                    }
                }
            }
        }
        if (width && height && !out.width) { out.width = width; out.height = height }
        if (jpegOfs !== null && jpegLen !== null && jpegLen > 0) {
            out.previews.push({ offset: jpegOfs, length: jpegLen })
        }
        for (const sub of subIfds) readIfd(sub, depth + 1)
        const next = u32(view, ofs + 2 + count * 12, le)
        if (next) readIfd(next, depth + 1)
    }

    const ifd0 = u32(view, base + 4, le)
    if (ifd0 === null) return null
    readIfd(ifd0, 0)
    return out
}

const parseJpeg = (view) => {
    const end = view.byteLength
    let ofs = 2
    let width = null
    let height = null
    let orientation = 1
    while (ofs + 4 <= end) {
        if (view.getUint8(ofs) !== 0xff) break
        const marker = view.getUint8(ofs + 1)
        if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) { ofs += 2; continue }
        const len = u16(view, ofs + 2, false)
        if (len === null || len < 2) break
        if (marker === 0xe1 && ofs + 10 <= end && ascii(view, ofs + 4, 6) === 'Exif\0\0') {
            const tiff = parseTiffIfds(view, ofs + 10)
            if (tiff) orientation = tiff.orientation
        }
        const isSof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc
        if (isSof && ofs + 9 <= end) {
            height = view.getUint16(ofs + 5, false)
            width = view.getUint16(ofs + 7, false)
            break
        }
        if (marker === 0xda) break // entropy-coded data — no SOF found before it
        ofs += 2 + len
    }
    if (!width || !height) return null
    return { format: 'jpeg', width, height, orientation }
}

const parsePng = (view) => {
    if (ascii(view, 12, 4) !== 'IHDR') return null
    const width = u32(view, 16, false)
    const height = u32(view, 20, false)
    if (!width || !height) return null
    return { format: 'png', width, height, orientation: 1 }
}

const parseGif = (view) => {
    const width = u16(view, 6, true)
    const height = u16(view, 8, true)
    if (!width || !height) return null
    return { format: 'gif', width, height, orientation: 1 }
}

const parseBmp = (view) => {
    const width = u32(view, 18, true)
    let height = view.byteLength >= 26 ? view.getInt32(22, true) : 0
    height = Math.abs(height)
    if (!width || !height) return null
    return { format: 'bmp', width, height, orientation: 1 }
}

const parseWebp = (view) => {
    const end = view.byteLength
    let ofs = 12
    while (ofs + 8 <= end) {
        const fourcc = ascii(view, ofs, 4)
        const size = u32(view, ofs + 4, true)
        if (size === null) break
        const data = ofs + 8
        if (fourcc === 'VP8X' && data + 10 <= end) {
            const w = 1 + (view.getUint8(data + 4) | (view.getUint8(data + 5) << 8) | (view.getUint8(data + 6) << 16))
            const h = 1 + (view.getUint8(data + 7) | (view.getUint8(data + 8) << 8) | (view.getUint8(data + 9) << 16))
            return { format: 'webp', width: w, height: h, orientation: 1 }
        }
        if (fourcc === 'VP8 ' && data + 10 <= end) {
            const w = view.getUint16(data + 6, true) & 0x3fff
            const h = view.getUint16(data + 8, true) & 0x3fff
            if (w && h) return { format: 'webp', width: w, height: h, orientation: 1 }
        }
        if (fourcc === 'VP8L' && data + 5 <= end && view.getUint8(data) === 0x2f) {
            const b = view.getUint32(data + 1, true)
            return { format: 'webp', width: (b & 0x3fff) + 1, height: ((b >> 14) & 0x3fff) + 1, orientation: 1 }
        }
        ofs = data + size + (size & 1)
    }
    return null
}

/** AVIF/HEIF: box walk to the largest `ispe` (item spatial extent); `irot`
 *  maps to a dims-swapping orientation (decode applies the real transform). */
const parseIsobmff = (view) => {
    const end = view.byteLength
    let best = null
    let rotation = 0

    const walk = (ofs, limit, depth) => {
        if (depth > 8) return
        while (ofs + 8 <= limit) {
            let size = u32(view, ofs, false)
            const type = ascii(view, ofs + 4, 4)
            let hdr = 8
            if (size === 1) {
                if (ofs + 16 > limit) return
                const hi = view.getUint32(ofs + 8, false)
                const lo = view.getUint32(ofs + 12, false)
                size = hi * 4294967296 + lo
                hdr = 16
            } else if (size === 0) {
                size = limit - ofs
            }
            if (size < hdr || ofs + size > limit + 0) {
                // Truncated window: still descend into what we can see.
                size = Math.min(size, limit - ofs)
                if (size < hdr) return
            }
            const body = ofs + hdr
            if (type === 'meta') walk(body + 4, ofs + size, depth + 1) // fullbox
            else if (type === 'iprp' || type === 'ipco' || type === 'moov' || type === 'trak') walk(body, ofs + size, depth + 1)
            else if (type === 'ispe' && body + 12 <= limit) {
                const w = view.getUint32(body + 4, false)
                const h = view.getUint32(body + 8, false)
                if (w && h && (!best || w * h > best.width * best.height)) best = { width: w, height: h }
            } else if (type === 'irot' && body + 1 <= limit) {
                rotation = view.getUint8(body) & 3
            }
            ofs += size
        }
    }

    walk(0, end, 0)
    if (!best) return null
    // irot counts 90° anti-clockwise turns; 1↔8 and 3↔6 swap dims like EXIF.
    const orientation = [1, 8, 3, 6][rotation]
    return { format: 'isobmff', width: best.width, height: best.height, orientation }
}

/**
 * Parse dimensions + orientation from a bounded header window.
 * Returns null for unknown/undecodable containers (callers fall back to a
 * decode-then-downscale path). TIFF results may carry `rawPreview`
 * ({offset,length} of an embedded JPEG) for RAW containers.
 */
export function parseImageHeader(bytes) {
    if (!bytes || bytes.length < 16) return null
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    const b0 = bytes[0]
    const b1 = bytes[1]
    if (b0 === 0xff && b1 === 0xd8) return parseJpeg(view)
    if (b0 === 0x89 && ascii(view, 1, 3) === 'PNG') return parsePng(view)
    if (ascii(view, 0, 3) === 'GIF') return parseGif(view)
    if (b0 === 0x42 && b1 === 0x4d) return parseBmp(view)
    if (ascii(view, 0, 4) === 'RIFF' && ascii(view, 8, 4) === 'WEBP') return parseWebp(view)
    if (bytes.length > 12 && ascii(view, 4, 4) === 'ftyp') return parseIsobmff(view)
    if ((b0 === 0x49 && b1 === 0x49) || (b0 === 0x4d && b1 === 0x4d)) {
        const tiff = parseTiffIfds(view, 0)
        if (!tiff) return null
        const largest = tiff.previews.sort((a, b) => b.length - a.length)[0] || null
        return {
            format: 'tiff',
            width: tiff.width || null,
            height: tiff.height || null,
            orientation: tiff.orientation,
            rawPreview: largest,
        }
    }
    return null
}

/* ─── Bounded proxy decode (browser-only entry point) ────────────────────── */

/** Rebound an oversized bitmap; closes the input when it gets replaced. */
const enforceCap = async (bitmap, maxLongSide) => {
    if (Math.max(bitmap.width, bitmap.height) <= maxLongSide) return bitmap
    const target = getBoundedProxySize(bitmap.width, bitmap.height, maxLongSide)
    try {
        return await createImageBitmap(bitmap, {
            resizeWidth: target.width,
            resizeHeight: target.height,
            resizeQuality: 'high',
        })
    } finally {
        bitmap.close()
    }
}

/** JPEG-only fast path: ImageDecoder can scale during decode (codec-gated).
 *  JPEG scaled decode snaps to 1/2·1/4·1/8, so ask ×2 the target — rounding
 *  then errs on the larger side and the exact-target resize owns final dims. */
const decodeViaImageDecoder = async (blob, target, oriented) => {
    const decoder = new ImageDecoder({
        type: blob.type || 'image/jpeg',
        data: blob.stream(),
        desiredWidth: Math.min(oriented.width, target.width * 2),
        desiredHeight: Math.min(oriented.height, target.height * 2),
        preferAnimation: false,
    })
    try {
        const { image } = await decoder.decode()
        try {
            if (image.displayWidth === target.width && image.displayHeight === target.height) {
                return await createImageBitmap(image)
            }
            return await createImageBitmap(image, {
                resizeWidth: target.width,
                resizeHeight: target.height,
                resizeQuality: 'high',
            })
        } finally {
            image.close()
        }
    } finally {
        try { decoder.close() } catch { /* already closed */ }
    }
}

/**
 * Decode `blob` into a ≤ maxLongSide ImageBitmap plus metadata.
 * Returns { bitmap, original: {width,height,orientation,format},
 *           proxy: {width,height,scale} }.
 */
export async function decodeBoundedBitmap(blob, maxLongSide, depth = 0) {
    const window = new Uint8Array(await blob.slice(0, Math.min(blob.size, HEADER_READ_BYTES)).arrayBuffer())
    const header = parseImageHeader(window)

    // RAW container → decode its embedded JPEG preview instead (best-effort).
    if (depth === 0 && header?.format === 'tiff' && header.rawPreview) {
        const { offset, length } = header.rawPreview
        if (offset > 0 && length > 1024 && offset + length <= blob.size) {
            try {
                const inner = await decodeBoundedBitmap(blob.slice(offset, offset + length, 'image/jpeg'), maxLongSide, depth + 1)
                inner.original.format = 'raw-preview'
                return inner
            } catch { /* fall through to the container itself */ }
        }
    }

    if (header?.width && header?.height) {
        const oriented = orientedSize(header.width, header.height, header.orientation)
        const target = getBoundedProxySize(oriented.width, oriented.height, maxLongSide)
        let bitmap = null
        if (header.format === 'jpeg' && header.orientation === 1 && typeof ImageDecoder !== 'undefined') {
            bitmap = await decodeViaImageDecoder(blob, target, oriented).catch(() => null)
        }
        if (!bitmap) {
            // Constrain only the oriented long edge — aspect ratio is then
            // codec truth, and enforceCap corrects any orientation surprise.
            const resize = oriented.width >= oriented.height
                ? { resizeWidth: target.width }
                : { resizeHeight: target.height }
            bitmap = await createImageBitmap(blob, {
                ...resize,
                resizeQuality: 'high',
                imageOrientation: 'from-image',
            })
        }
        bitmap = await enforceCap(bitmap, maxLongSide)
        return {
            bitmap,
            original: { width: header.width, height: header.height, orientation: header.orientation, format: header.format },
            proxy: { width: bitmap.width, height: bitmap.height, scale: bitmap.width / oriented.width },
        }
    }

    return decodeUnknownContainer(blob, maxLongSide)
}

/** Long edge that satisfies BOTH the side cap and the megapixel cap. */
export const exportLongSide = (width, height, { maxSide, maxMP }) => {
    const long = Math.max(width, height)
    const mpScale = Math.sqrt(Math.min(1, (maxMP * 1e6) / (width * height)))
    return Math.max(1, Math.min(maxSide, Math.floor(long * mpScale), long))
}

/**
 * Explicit-export composite: bounded decode of the original (≤ caps), mask
 * upscaled from the proxy, destination-in cutout → PNG blob. Everything
 * transient is closed before returning. Never called from selection flow.
 */
export async function compositeCutout({ blob = null, bitmap = null, mask, caps }) {
    let src = bitmap
    let reduced = false
    if (!src) {
        const window = new Uint8Array(await blob.slice(0, Math.min(blob.size, HEADER_READ_BYTES)).arrayBuffer())
        const header = parseImageHeader(window)
        const guess = header?.width && header?.height
            ? orientedSize(header.width, header.height, header.orientation)
            : null
        const effLong = guess ? exportLongSide(guess.width, guess.height, caps) : caps.maxSide
        const decoded = await decodeBoundedBitmap(blob, effLong)
        src = decoded.bitmap
        const srcLong = Math.max(decoded.original.width, decoded.original.height)
        reduced = Math.max(src.width, src.height) < srcLong
    }
    try {
        // MP cap can still bind when only the side cap was known pre-decode.
        const finalLong = exportLongSide(src.width, src.height, caps)
        let outW = src.width
        let outH = src.height
        if (finalLong < Math.max(src.width, src.height)) {
            const scale = finalLong / Math.max(src.width, src.height)
            outW = Math.max(1, Math.round(src.width * scale))
            outH = Math.max(1, Math.round(src.height * scale))
            reduced = true
        }
        const out = new OffscreenCanvas(outW, outH)
        const ctx = out.getContext('2d')
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(src, 0, 0, outW, outH)
        const maskCanvas = new OffscreenCanvas(mask.width, mask.height)
        const maskImage = new ImageData(mask.width, mask.height)
        for (let p = 0, i = 3; p < mask.alpha.length; p += 1, i += 4) {
            maskImage.data[i] = mask.alpha[p]
        }
        maskCanvas.getContext('2d').putImageData(maskImage, 0, 0)
        ctx.globalCompositeOperation = 'destination-in'
        ctx.drawImage(maskCanvas, 0, 0, outW, outH)
        const png = await out.convertToBlob({ type: 'image/png' })
        return { blob: png, width: outW, height: outH, reduced }
    } finally {
        try { src.close?.() } catch { /* canvas source */ }
    }
}

// Unknown container — the unavoidable full decode. Bound immediately, close
// the full frame, yield one macrotask before resolving so the browser can
// release it before any model work starts.
async function decodeUnknownContainer(blob, maxLongSide) {
    const full = await createImageBitmap(blob, { imageOrientation: 'from-image' })
    const ow = full.width
    const oh = full.height
    let bitmap = full
    try {
        const target = getBoundedProxySize(ow, oh, maxLongSide)
        if (target.proxyActive) {
            bitmap = await createImageBitmap(full, {
                resizeWidth: target.width,
                resizeHeight: target.height,
                resizeQuality: 'high',
            })
        }
    } finally {
        if (bitmap !== full) full.close()
    }
    await new Promise((resolve) => setTimeout(resolve, 0))
    return {
        bitmap,
        original: { width: ow, height: oh, orientation: 1, format: 'unknown' },
        proxy: { width: bitmap.width, height: bitmap.height, scale: bitmap.width / ow },
    }
}

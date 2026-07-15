/**
 * image-io — cheap image-header parse (dims + EXIF orientation).
 * Lets the app decode a huge photo STRAIGHT to a bounded proxy via
 * createImageBitmap's resize path instead of materializing full-res RGBA
 * (an 8K frame is ~132 MB; a 48 MP shot ~192 MB). Reads only the first
 * 512 KB. JPEG/PNG/WebP/AVIF-HEIF/TIFF/GIF/BMP are covered; null means a
 * one-time browser decode is required for that format.
 */

const HEAD = 512 * 1024

const jpegMeta = (dv) => {
    if (dv.getUint16(0) !== 0xffd8) return null // SOI
    let o = 2
    let orientation = 1
    const len = dv.byteLength
    while (o + 4 < len) {
        if (dv.getUint8(o) !== 0xff) { o += 1; continue }
        const marker = dv.getUint8(o + 1)
        // Standalone markers (no length): padding, RSTn, SOI/EOI.
        if (marker === 0xff || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) { o += 2; continue }
        const seg = dv.getUint16(o + 2)
        if (marker === 0xe1 && o + 10 < len && dv.getUint32(o + 4) === 0x45786966) { // "Exif"
            orientation = exifOrientation(dv, o + 10) || orientation
        }
        // SOF0-3/5-7/9-11/13-15 carry the frame dimensions.
        const sof = (marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7)
            || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)
        if (sof && o + 9 < len) return { w: dv.getUint16(o + 7), h: dv.getUint16(o + 5), orientation }
        o += 2 + seg
    }
    return null
}

// TIFF header at `base` ('II'|'MM'); return the Orientation tag (0x0112) or 0.
const exifOrientation = (dv, base) => {
    try {
        const le = dv.getUint16(base) === 0x4949
        const ifd = base + dv.getUint32(base + 4, le)
        const n = dv.getUint16(ifd, le)
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            if (dv.getUint16(e, le) === 0x0112) return dv.getUint16(e + 8, le)
        }
    } catch { /* malformed EXIF → orientation 1 */ }
    return 0
}

const pngMeta = (dv) => {
    if (dv.getUint32(0) !== 0x89504e47 || dv.getUint32(4) !== 0x0d0a1a0a) return null
    if (dv.getUint32(12) !== 0x49484452) return null // IHDR
    return { w: dv.getUint32(16), h: dv.getUint32(20), orientation: 1 }
}

const webpMeta = (dv) => {
    if (dv.getUint32(0) !== 0x52494646 || dv.getUint32(8) !== 0x57454250) return null // RIFF…WEBP
    const cc = dv.getUint32(12)
    if (cc === 0x56503858) { // 'VP8X' — 24-bit width-1/height-1 (LE) at 24/27
        const w = 1 + (dv.getUint8(24) | (dv.getUint8(25) << 8) | (dv.getUint8(26) << 16))
        const h = 1 + (dv.getUint8(27) | (dv.getUint8(28) << 8) | (dv.getUint8(29) << 16))
        return { w, h, orientation: 1 }
    }
    if (cc === 0x56503820 && dv.getUint8(23) === 0x9d && dv.getUint8(24) === 0x01 && dv.getUint8(25) === 0x2a) {
        return { w: dv.getUint16(26, true) & 0x3fff, h: dv.getUint16(28, true) & 0x3fff, orientation: 1 } // 'VP8 ' keyframe
    }
    if (cc === 0x5650384c && dv.getUint8(20) === 0x2f) { // VP8L lossless
        const b1 = dv.getUint8(21)
        const b2 = dv.getUint8(22)
        const b3 = dv.getUint8(23)
        const b4 = dv.getUint8(24)
        return {
            w: 1 + (b1 | ((b2 & 0x3f) << 8)),
            h: 1 + ((b2 >> 6) | (b3 << 2) | ((b4 & 0x0f) << 10)),
            orientation: 1,
        }
    }
    return null
}

const gifMeta = (dv) => {
    const magic = dv.getUint32(0)
    if (magic !== 0x47494638 || (dv.getUint16(4) !== 0x3761 && dv.getUint16(4) !== 0x3961)) return null // GIF87a/GIF89a
    return { w: dv.getUint16(6, true), h: dv.getUint16(8, true), orientation: 1 }
}

const bmpMeta = (dv) => {
    if (dv.getUint16(0) !== 0x424d || dv.byteLength < 26) return null // BM
    const w = Math.abs(dv.getInt32(18, true))
    const h = Math.abs(dv.getInt32(22, true))
    return w && h ? { w, h, orientation: 1 } : null
}

// Baseline TIFF (also used by some DSLR-adjacent exported files). BigTIFF is
// deliberately left to the browser decoder: its 64-bit IFD layout needs a
// different parser and is uncommon for browser-openable image files.
const tiffMeta = (dv) => {
    try {
        const le = dv.getUint16(0) === 0x4949
        if (!le && dv.getUint16(0) !== 0x4d4d) return null
        if (dv.getUint16(2, le) !== 0x2a) return null
        const ifd = dv.getUint32(4, le)
        const n = dv.getUint16(ifd, le)
        let w = 0
        let h = 0
        let orientation = 1
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            const tag = dv.getUint16(e, le)
            const type = dv.getUint16(e + 2, le)
            const count = dv.getUint32(e + 4, le)
            if (count !== 1) continue
            const value = type === 3 ? dv.getUint16(e + 8, le) : (type === 4 ? dv.getUint32(e + 8, le) : 0)
            if (tag === 0x0100) w = value
            else if (tag === 0x0101) h = value
            else if (tag === 0x0112) orientation = value || orientation
        }
        return w && h ? { w, h, orientation } : null
    } catch { return null }
}

// AVIF/HEIF store dimensions in an `ispe` image-spatial-extents box. We scan
// only the bounded header; if a rotation/mirror box is present we decline the
// frugal path rather than guess its composition with EXIF orientation.
const isoImageMeta = (dv) => {
    try {
        if (dv.getUint32(4) !== 0x66747970) return null // ftyp
        let spatial = null
        let transformed = false
        for (let i = 0; i + 20 <= dv.byteLength; i += 1) {
            const type = dv.getUint32(i + 4)
            if (type === 0x69737065) { // ispe
                const w = dv.getUint32(i + 12)
                const h = dv.getUint32(i + 16)
                if (w && h) spatial = { w, h }
            } else if (type === 0x69726f74 || type === 0x696d6972) { // irot / imir
                transformed = true
            }
        }
        return spatial && !transformed ? { ...spatial, orientation: 1 } : null
    } catch { return null }
}

/** { w, h, orientation } in ENCODED pixels (pre-rotation), or null. */
export const readImageMeta = async (blob) => {
    try {
        const dv = new DataView(await blob.slice(0, HEAD).arrayBuffer())
        return jpegMeta(dv) || pngMeta(dv) || webpMeta(dv)
            || gifMeta(dv) || bmpMeta(dv) || tiffMeta(dv) || isoImageMeta(dv)
    } catch { return null }
}

/**
 * image-raw — extract the embedded JPEG preview from a camera RAW (main thread).
 * A DSLR RAW (.nef/.cr2/.arw/.dng/…) is a container: the sensor mosaic + heavy
 * metadata + one or more ALREADY-DEVELOPED JPEG previews, usually including one
 * at full sensor resolution. For a select/cutout tool the preview IS the photo,
 * so we lift it out and drop the rest — no demosaicing, no Python, no server:
 *   • strips sensor data + metadata (a 33 MB NEF → a ~3 MB JPEG),
 *   • reads only the header + the preview's byte range (never the whole file),
 *   • hands a plain image/jpeg Blob to the normal decode-to-proxy path.
 * TIFF-container RAW (Nikon/Canon-CR2/Sony/Pentax/Olympus/Panasonic/Adobe DNG…)
 * is parsed by its IFD preview pointers; other containers (CR3/RAF) fall back to
 * a bounded, validated JPEG scan.
 *
 * This is the FAST PATH and covers essentially every modern RAW: not a raw
 * developer — the preview is the camera's 8-bit rendering, not a fresh demosaic
 * — which is exactly right for masking. When a file carries NO usable embedded
 * preview, extractRawPreview returns null and the import falls back to on-device
 * LibRaw develop (raw-develop-client.js → raw-develop.wasm): still no Python, no
 * server, and only a compact developed JPEG rejoins this same decode path.
 */

import { readImageMeta } from './image-io.js'

const RAW_EXTS = new Set([
    'nef', 'nrw', 'cr2', 'cr3', 'crw', 'arw', 'sr2', 'srf', 'dng', 'orf', 'rw2',
    'raf', 'pef', 'srw', 'raw', 'rwl', 'iiq', '3fr', 'fff', 'erf', 'mrw', 'mef',
    'mos', 'dcr', 'kdc', 'x3f', 'nrf',
])

const HEADER_BYTES = 2 * 1024 * 1024   // IFD tables sit near the front
const SCAN_BYTES = 24 * 1024 * 1024    // fallback scan cap for non-TIFF containers
const MAX_EDGE = 30000                  // reject absurd (false-positive) dims
const PROXY_MIN_EDGE = 1024             // standard interaction cap; callers may raise it
const trace = (event, detail = {}) => console.log(`[seglab][raw] ${event}`, detail)

export const rawExtOf = (name = '') => (name.split('.').pop() || '').toLowerCase()
export const isRawFile = (file) => !!file && (RAW_EXTS.has(rawExtOf(file.name || '')) || /(^|\/)x-(nikon|canon|sony|adobe-dng|fuji)/i.test(file.type || ''))

/** Walk TIFF IFD0 + SubIFDs + the IFD chain; collect preview (offset,length)
 *  pairs from the JPEGInterchangeFormat tags and the IFD0 orientation. */
const parseTiff = (dv) => {
    if (dv.byteLength < 8) return null
    const le = dv.getUint16(0) === 0x4949
    if (!le && dv.getUint16(0) !== 0x4d4d) return null
    const u16 = (o) => dv.getUint16(o, le)
    const u32 = (o) => dv.getUint32(o, le)
    if (u16(2) !== 0x2a) return null
    const len = dv.byteLength
    const previews = []
    let orientation = 1
    const seen = new Set()
    const walk = (ifd, depth) => {
        if (ifd <= 0 || ifd + 2 > len || seen.has(ifd) || depth > 8) return
        seen.add(ifd)
        const n = u16(ifd)
        if (ifd + 2 + n * 12 + 4 > len) return
        let off = 0
        let jlen = 0
        const subs = []
        for (let i = 0; i < n; i += 1) {
            const e = ifd + 2 + i * 12
            const tag = u16(e)
            if (tag === 0x0201) off = u32(e + 8)                 // JPEGInterchangeFormat
            else if (tag === 0x0202) jlen = u32(e + 8)           // …Length
            else if (tag === 0x0112) orientation = u16(e + 8) || orientation
            else if (tag === 0x014a) {                            // SubIFDs
                const cnt = u32(e + 4)
                if (cnt === 1) subs.push(u32(e + 8))
                else { const p = u32(e + 8); for (let k = 0; k < cnt && p + 4 * k + 4 <= len; k += 1) subs.push(u32(p + 4 * k)) }
            }
        }
        if (off > 0 && jlen > 0) previews.push({ off, len: jlen })
        for (const s of subs) walk(s, depth + 1)
        walk(u32(ifd + 2 + n * 12), depth + 1)
    }
    walk(u32(4), 0)
    return { previews, orientation }
}

/** SOF dims of a JPEG at `start` (for validating scanned candidates). */
const jpegSof = (dv, start) => {
    let o = start + 2
    const len = dv.byteLength
    while (o + 9 < len) {
        if (dv.getUint8(o) !== 0xff) { o += 1; continue }
        const m = dv.getUint8(o + 1)
        if (m === 0xff || m === 0x01 || (m >= 0xd0 && m <= 0xd9)) { o += 2; continue }
        const sof = (m >= 0xc0 && m <= 0xc3) || (m >= 0xc5 && m <= 0xc7) || (m >= 0xc9 && m <= 0xcb) || (m >= 0xcd && m <= 0xcf)
        if (sof) return { w: dv.getUint16(o + 7), h: dv.getUint16(o + 5) }
        o += 2 + dv.getUint16(o + 2)
    }
    return null
}

/** All sane JPEGs in `bytes` (fallback for CR3/RAF and other containers). */
const scanForPreviews = (bytes) => {
    const out = []
    let i = 0
    while (i + 3 < bytes.length) {
        if (bytes[i] === 0xff && bytes[i + 1] === 0xd8 && bytes[i + 2] === 0xff) {
            const sof = jpegSof(new DataView(bytes.buffer, bytes.byteOffset + i, Math.min(bytes.length - i, 262144)), 0)
            if (sof && sof.w > 320 && sof.h > 320 && sof.w < MAX_EDGE && sof.h < MAX_EDGE) {
                let e = i + 2
                while (e + 1 < bytes.length && !(bytes[e] === 0xff && bytes[e + 1] === 0xd9)) e += 1
                if (e + 1 < bytes.length) { out.push({ off: i, len: e + 2 - i, w: sof.w, h: sof.h }); i = e + 2; continue }
            }
        }
        i += 1
    }
    return out
}

const validSoi = async (blob) => {
    const s = new Uint8Array(await blob.slice(0, 3).arrayBuffer())
    return s[0] === 0xff && s[1] === 0xd8 && s[2] === 0xff
}

// Fill in missing SOF dims for TIFF-pointer candidates (read a small window).
const withDims = async (file, cands) => {
    const out = []
    for (const c of cands) {
        if (c.w && c.h) { out.push(c); continue }
        if (c.off <= 0 || c.off + 3 > file.size) continue
        const meta = await readImageMeta(file.slice(c.off, c.off + Math.min(c.len, 262144), 'image/jpeg'))
        if (meta && meta.w > 0) out.push({ ...c, w: meta.w, h: meta.h })
    }
    return out
}

/**
 * Extract embedded JPEG previews from a RAW file. Returns the FULL-resolution
 * preview (the export truth) plus a small `proxyBlob` — the camera's own smaller
 * embedded preview — so the interaction proxy is built WITHOUT ever decoding the
 * full-res frame (critical on Safari, which materializes the whole 45 MP JPEG
 * rather than scaled-decoding it). proxyBlob is null when no suitable smaller
 * preview exists (caller decode-downscales the full one).
 * @returns {Promise<{ blob: Blob, width: number, height: number, proxyBlob: Blob|null, orientation: number } | null>}
 */
export const extractRawPreview = async (file, { proxyMinEdge = PROXY_MIN_EDGE } = {}) => {
    trace('extract-start', { name: file.name, bytes: file.size })
    let cands = []
    let orientation = 1
    try {
        const head = new DataView(await file.slice(0, HEADER_BYTES).arrayBuffer())
        const tiff = parseTiff(head)
        if (tiff) { cands = tiff.previews; orientation = tiff.orientation }
    } catch { /* not TIFF → scan */ }
    if (!cands.length) {
        try {
            const bytes = new Uint8Array(await file.slice(0, Math.min(file.size, SCAN_BYTES)).arrayBuffer())
            cands = scanForPreviews(bytes)
        } catch { /* give up below */ }
    }

    const sized = await withDims(file, cands)
    if (!sized.length) return null
    const full = sized.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a))
    if (full.off + full.len > file.size) return null

    // Proxy source = the smallest preview whose long edge still clears the proxy
    // target; falls back to the full one when nothing smaller is big enough.
    const minProxyEdge = Math.max(512, Math.min(MAX_EDGE, Math.round(Number(proxyMinEdge) || PROXY_MIN_EDGE)))
    const proxyPick = sized
        .filter((c) => Math.max(c.w, c.h) >= minProxyEdge)
        .sort((a, b) => a.w * a.h - b.w * b.h)[0] || full

    const slice = (c) => file.slice(c.off, c.off + c.len, 'image/jpeg')
    const blob = slice(full)
    if (!(await validSoi(blob))) return null
    const result = {
        blob,
        width: full.w,
        height: full.h,
        proxyBlob: proxyPick === full ? null : slice(proxyPick),
        orientation,
    }
    trace('extract-ready', {
        previews: sized.length,
        full: `${full.w}x${full.h}`,
        proxy: result.proxyBlob ? `${proxyPick.w}x${proxyPick.h}` : 'full',
        minProxyEdge,
        orientation,
    })
    return result
}

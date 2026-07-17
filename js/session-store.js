/**
 * session-store — OPFS persistence of the working session (document + the
 * committed selection stack), so a crash or power cut does not lose work.
 *
 * Writes are incremental and durable: a power cut fires no pagehide/beforeunload,
 * so saving on unload would save nothing. Every commit schedules a debounced
 * write-then-move instead.
 *
 * Holds the compressed original only — never a full-res RGBA frame — matching
 * asset-store's custody rule. Best-effort: any failure just means no restore.
 */

const DIR = 'seglab-session'
const FILE = 'session.bin'
const MAX_BYTES = 300 * 1024 * 1024 // refuse absurd originals rather than fill the disk

const opfsAvailable = () => typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory

let dirPromise = null
const getDir = () => {
    dirPromise ??= navigator.storage.getDirectory().then((root) => root.getDirectoryHandle(DIR, { create: true }))
    return dirPromise
}

// entry → ArrayBuffer: [u32 headerLen][JSON header][original bytes][mask chan?]
//
// The composed mask is stored, not the op stack: a single click's selection
// lives in state.liveMask and never reaches baseOps, so persisting ops alone
// loses the most common selection there is. Restore replays it as the base
// floor — the selection returns exactly; per-op undo history does not.
const pack = (entry, bytes) => {
    const blobs = []
    let offset = 0
    const push = (data) => {
        const view = new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength ?? data.length)
        blobs.push(view)
        const at = offset
        offset += view.byteLength
        return { at, length: view.byteLength }
    }
    const original = push(bytes)
    const mask = entry.mask ? push(entry.mask) : null
    const header = {
        v: 1,
        assetKey: entry.assetKey,
        w: entry.w,
        h: entry.h,
        mime: entry.mime,
        name: entry.name,
        savedAt: Date.now(),
        original,
        mask,
    }
    const headerBytes = new TextEncoder().encode(JSON.stringify(header))
    const out = new Uint8Array(4 + headerBytes.byteLength + offset)
    new DataView(out.buffer).setUint32(0, headerBytes.byteLength)
    out.set(headerBytes, 4)
    let p = 4 + headerBytes.byteLength
    for (const b of blobs) { out.set(b, p); p += b.byteLength }
    return out.buffer
}

const unpack = (buf) => {
    const headerLen = new DataView(buf).getUint32(0)
    const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)))
    if (h.v !== 1) throw new Error(`unknown session version ${h.v}`)
    const base = 4 + headerLen
    const slice = (r) => new Uint8Array(buf.slice(base + r.at, base + r.at + r.length))
    return {
        assetKey: h.assetKey,
        w: h.w,
        h: h.h,
        savedAt: h.savedAt,
        blob: new Blob([slice(h.original)], { type: h.mime || 'image/jpeg' }),
        name: h.name || 'restored',
        mask: h.mask ? slice(h.mask) : null,
    }
}

/** Persisted session, or null (absent / corrupt / no OPFS). */
export const loadSession = async () => {
    if (!opfsAvailable()) return null
    try {
        const dir = await getDir()
        const fh = await dir.getFileHandle(FILE) // throws if absent
        return unpack(await (await fh.getFile()).arrayBuffer())
    } catch { return null }
}

// Serialized: no live .tmp while another save runs.
let saveChain = Promise.resolve(false)
let pending = null
let timer = null

const saveOnce = async (getEntry) => {
    const entry = typeof getEntry === 'function' ? getEntry() : getEntry
    if (!opfsAvailable() || !entry?.blob) return false
    try {
        const bytes = new Uint8Array(await entry.blob.arrayBuffer())
        if (bytes.byteLength > MAX_BYTES) return false
        const dir = await getDir()
        const buf = pack(entry, bytes)
        const tmp = `${FILE}.tmp`
        const tfh = await dir.getFileHandle(tmp, { create: true })
        const w = await tfh.createWritable()
        await w.write(buf)
        await w.close()
        if (tfh.move) {
            await tfh.move(FILE) // atomic swap — a cut mid-write leaves the old session intact
        } else {
            const fh = await dir.getFileHandle(FILE, { create: true })
            const w2 = await fh.createWritable()
            await w2.write(buf)
            await w2.close()
            await dir.removeEntry(tmp).catch(() => {})
        }
        return true
    } catch { return false }
}

/**
 * Persist (write-then-move), coalescing bursts. `getEntry` is a thunk called
 * at write time, so a drag scheduling hundreds of saves serializes the mask
 * once, not per frame. `delay` 0 writes now — used on import, where the whole
 * document is at risk until the next commit.
 */
export const saveSession = (getEntry, { delay = 600 } = {}) => {
    pending = getEntry
    if (timer) clearTimeout(timer)
    return new Promise((resolve) => {
        const run = () => {
            timer = null
            const next = pending
            pending = null
            // Both handlers: a rejected chain would otherwise skip every later
            // save silently, losing work for the rest of the session.
            saveChain = saveChain.then(() => saveOnce(next), () => saveOnce(next))
            saveChain.then(resolve)
        }
        if (delay === 0) run()
        else timer = setTimeout(run, delay)
    })
}

/** Drop the session (explicit reset / restore declined). */
export const clearSession = async () => {
    if (timer) { clearTimeout(timer); timer = null }
    pending = null
    if (!opfsAvailable()) return
    try { await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }) } catch { /* absent */ }
    dirPromise = null
}

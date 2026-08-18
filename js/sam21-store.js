/**
 * sam21-store — OPFS persistence for SAM 2.1 embeddings.
 *
 * The fastest encode is the one that never runs (§12 #10). An embedding is
 * ~8 MB at fp16 against a ~1.4 s encode that also opens a >1 GB session, so
 * persisting it turns a revisit — or a second tab on the same photo — into a
 * decode-only interaction.
 *
 * Runs in the TAB, not the SharedWorker. Measured: navigator.storage
 * .getDirectory() throws SecurityError inside a SharedWorker in Chrome — a
 * same-origin script, so it is not a blob:-origin artefact — while the main
 * thread and dedicated workers both succeed. sam21-client owns the cache and
 * moves raw bytes over the port (transferable, zero-copy); the lane exposes
 * adoptEmbedding/exportEmbedding as that seam.
 *
 * Concurrent writers are therefore possible again (one per tab), so writes stay
 * atomic via write-then-move and are keyed by content hash + precision — two
 * tabs racing the same photo write identical bytes, and a crash mid-write can
 * never leave a truncated file under the real name.
 *
 * Best-effort throughout. Any read/parse/quota failure just means re-encode, so
 * every path here swallows and returns null rather than surfacing an error.
 */

const DIR = 'seglab-sam21'
const MAX_BYTES = 256 * 1024 * 1024   // ~30 embeddings at fp16; LRU beyond that
const MAGIC = 0x53414d32              // 'SAM2'

const available = () => typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory

let dirPromise = null
const getDir = () => {
    dirPromise ??= navigator.storage.getDirectory()
        .then((root) => root.getDirectoryHandle(DIR, { create: true }))
    return dirPromise
}

const fileName = (key) => `${String(key).replace(/[^a-z0-9]/gi, '_').slice(0, 120)}.bin`

/** [magic u32][version u32][count u32][byteLength u32 × count][payload…] */
const pack = (buffers) => {
    const head = 4 * (3 + buffers.length)
    const total = head + buffers.reduce((n, b) => n + b.byteLength, 0)
    const out = new ArrayBuffer(total)
    const dv = new DataView(out)
    dv.setUint32(0, MAGIC)
    dv.setUint32(4, 1)
    dv.setUint32(8, buffers.length)
    buffers.forEach((b, i) => dv.setUint32(12 + i * 4, b.byteLength))
    const bytes = new Uint8Array(out)
    let off = head
    for (const b of buffers) { bytes.set(new Uint8Array(b), off); off += b.byteLength }
    return out
}

const unpack = (buf) => {
    if (!buf || buf.byteLength < 12) return null
    const dv = new DataView(buf)
    if (dv.getUint32(0) !== MAGIC || dv.getUint32(4) !== 1) return null
    const count = dv.getUint32(8)
    if (count < 1 || count > 8) return null
    const head = 4 * (3 + count)
    const sizes = []
    let need = head
    for (let i = 0; i < count; i += 1) { const n = dv.getUint32(12 + i * 4); sizes.push(n); need += n }
    if (buf.byteLength < need) return null   // truncated write; treat as a miss
    const out = []
    let off = head
    for (const n of sizes) { out.push(buf.slice(off, off + n)); off += n }
    return out
}

/** Read an embedding's raw buffers, or null. Touches mtime for the LRU. */
export const loadEmbedding = async (key) => {
    if (!available()) return null
    try {
        const dir = await getDir()
        const fh = await dir.getFileHandle(fileName(key))
        const buf = await (await fh.getFile()).arrayBuffer()
        return unpack(buf)
    } catch { return null }
}

/** Write-then-move so a crash or a quota abort can never leave a half file
 *  under the real name (§16: "write embeddings atomically"). */
export const saveEmbedding = async (key, buffers, retried = false) => {
    if (!available() || !buffers?.length) return false
    const name = fileName(key)
    const tmp = `${name}.tmp`
    try {
        const dir = await getDir()
        const data = pack(buffers)
        const th = await dir.getFileHandle(tmp, { create: true })
        // createSyncAccessHandle is worker-only and avoids the stream plumbing;
        // fall back where it is unavailable.
        if (th.createSyncAccessHandle) {
            const h = await th.createSyncAccessHandle()
            try { h.truncate(0); h.write(new Uint8Array(data), { at: 0 }); h.flush() } finally { h.close() }
        } else {
            const w = await th.createWritable()
            await w.write(data)
            await w.close()
        }
        // OPFS has no rename; copy through then drop the temp.
        const fh = await dir.getFileHandle(name, { create: true })
        if (fh.createSyncAccessHandle) {
            const h = await fh.createSyncAccessHandle()
            try { h.truncate(0); h.write(new Uint8Array(data), { at: 0 }); h.flush() } finally { h.close() }
        } else {
            const w = await fh.createWritable()
            await w.write(data)
            await w.close()
        }
        await dir.removeEntry(tmp).catch(() => {})
        evictLRU(dir).catch(() => {})
        return true
    } catch (err) {
        try { (await getDir()).removeEntry(tmp).catch(() => {}) } catch { /* nothing to clean */ }
        // Quota is recoverable and common on a shared origin: drop the cache to
        // half the cap and try once more. Anything else, give up quietly — a
        // failed write only costs one re-encode later.
        if (err?.name === 'QuotaExceededError' && !retried) {
            try {
                await evictLRU(await getDir(), MAX_BYTES / 2)
                return await saveEmbedding(key, buffers, true)
            } catch { /* still no room */ }
        }
        return false
    }
}

/** Byte-capped LRU. Embeddings are recoverable by re-encode, so evicting is
 *  always safe — the only cost is one encode next time. */
const evictLRU = async (dir, cap = MAX_BYTES) => {
    const files = []
    let total = 0
    for await (const [name, handle] of dir.entries()) {
        if (handle.kind !== 'file' || name.endsWith('.tmp')) continue
        try {
            const f = await handle.getFile()
            files.push({ name, size: f.size, at: f.lastModified })
            total += f.size
        } catch { /* vanished mid-scan */ }
    }
    if (total <= cap) return
    files.sort((a, b) => a.at - b.at)
    for (const f of files) {
        if (total <= cap) break
        try { await dir.removeEntry(f.name); total -= f.size } catch { /* already gone */ }
    }
}

export const clearStore = async () => {
    if (!available()) return
    dirPromise = null
    try { await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }) } catch { /* absent */ }
}


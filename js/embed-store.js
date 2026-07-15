/**
 * embed-store — OPFS persistence of image embeddings (worker/main).
 * Embeddings are expensive to recompute (SlimSAM takes a model-encode pass);
 * persist them keyed by `lane:assetKey` so a revisit skips the encode. Binary
 * pack (JSON header + concatenated tensor/gray buffers), write-then-move for
 * atomicity, byte-capped LRU (~500 MB). Any read/parse/quota failure just
 * means re-encode — persistence is best-effort, never fatal, never networked.
 */

const DIR = 'seglab-embeds'
const MAX_BYTES = 500 * 1024 * 1024

const opfsAvailable = () => typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory

let dirPromise = null
const getDir = () => {
    dirPromise ??= navigator.storage.getDirectory().then((root) => root.getDirectoryHandle(DIR, { create: true }))
    return dirPromise
}

const fileName = (key) => `${key.replace(/[^a-z0-9]/gi, '_')}.bin`

// entry → ArrayBuffer: [u32 headerLen][JSON header][tensor blobs…][gray blob].
const packEntry = (entry) => {
    const blobs = []
    const tensors = []
    let offset = 0
    const push = (data) => {
        const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        blobs.push(view)
        const at = offset
        offset += view.byteLength
        return { at, length: view.byteLength }
    }
    for (const [name, t] of Object.entries(entry.embeddings)) {
        if (!t?.data) continue
        const { at, length } = push(t.data)
        tensors.push({ name, type: t.type, dims: t.dims, ctor: t.data.constructor.name, at, length })
    }
    if (!tensors.length) throw new Error('no tensor data to persist') // GPU-resident outputs
    const gray = push(entry.gray)
    const header = {
        v: 1,
        tensors,
        gray,
        original_sizes: entry.original_sizes,
        reshaped_input_sizes: entry.reshaped_input_sizes,
        w: entry.w,
        h: entry.h,
    }
    const headerBytes = new TextEncoder().encode(JSON.stringify(header))
    const out = new Uint8Array(4 + headerBytes.byteLength + offset)
    new DataView(out.buffer).setUint32(0, headerBytes.byteLength)
    out.set(headerBytes, 4)
    let p = 4 + headerBytes.byteLength
    for (const b of blobs) { out.set(b, p); p += b.byteLength }
    return out.buffer
}

const TYPED = {
    Float32Array, Float64Array, Int8Array, Uint8Array, Uint8ClampedArray,
    Int16Array, Uint16Array, Int32Array, Uint32Array, BigInt64Array, BigUint64Array,
}
if (typeof Float16Array !== 'undefined') TYPED.Float16Array = Float16Array

const unpackEntry = (buf, Tensor) => {
    const headerLen = new DataView(buf).getUint32(0)
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)))
    const base = 4 + headerLen
    const embeddings = {}
    for (const t of header.tensors) {
        const Ctor = TYPED[t.ctor]
        if (!Ctor) throw new Error(`unknown tensor ctor ${t.ctor}`) // → null → re-encode
        embeddings[t.name] = new Tensor(t.type, new Ctor(buf.slice(base + t.at, base + t.at + t.length)), t.dims)
    }
    const gray = new Float32Array(buf.slice(base + header.gray.at, base + header.gray.at + header.gray.length))
    return {
        embeddings,
        original_sizes: header.original_sizes,
        reshaped_input_sizes: header.reshaped_input_sizes,
        gray,
        w: header.w,
        h: header.h,
    }
}

/** Persisted embedding for `key`, or null (absent / corrupt / no OPFS). */
export const loadEmbedding = async (key, Tensor) => {
    if (!opfsAvailable()) return null
    try {
        const dir = await getDir()
        const fh = await dir.getFileHandle(fileName(key)) // throws if absent
        const buf = await (await fh.getFile()).arrayBuffer()
        return unpackEntry(buf, Tensor)
    } catch { return null }
}

// Saves are serialized: no live .tmp while evictLRU sweeps, no interleaved evictions.
let saveChain = Promise.resolve(false)

/** Persist `entry` under `key` (write-then-move), then LRU-evict. Best-effort. */
export const saveEmbedding = (key, entry) => {
    saveChain = saveChain.then(() => saveOnce(key, entry))
    return saveChain
}

const saveOnce = async (key, entry) => {
    if (!opfsAvailable()) return false
    try {
        const dir = await getDir()
        const buf = packEntry(entry)
        const tmp = `${fileName(key)}.tmp`
        const tfh = await dir.getFileHandle(tmp, { create: true })
        const w = await tfh.createWritable()
        await w.write(buf)
        await w.close()
        if (tfh.move) {
            await tfh.move(fileName(key)) // atomic swap
        } else {
            const fh = await dir.getFileHandle(fileName(key), { create: true })
            const w2 = await fh.createWritable()
            await w2.write(buf)
            await w2.close()
            await dir.removeEntry(tmp).catch(() => {})
        }
        await evictLRU(dir)
        return true
    } catch { return false }
}

// Byte-capped LRU by write time (a load doesn't retouch — writes approximate use).
const evictLRU = async (dir) => {
    const files = []
    let total = 0
    for await (const [name, handle] of dir.entries()) {
        if (handle.kind !== 'file') continue
        if (name.endsWith('.tmp')) { await dir.removeEntry(name).catch(() => {}); continue } // crashed write
        if (name.endsWith('.bin')) {
            try { const f = await handle.getFile(); files.push({ name, t: f.lastModified, size: f.size }); total += f.size } catch { /* skip */ }
        }
    }
    if (total <= MAX_BYTES) return
    files.sort((a, b) => a.t - b.t)
    for (const f of files) {
        if (total <= MAX_BYTES) break
        await dir.removeEntry(f.name).catch(() => {})
        total -= f.size
    }
}

/** Drop the whole store (test hook / user "clear cache"). */
export const clearStore = async () => {
    if (!opfsAvailable()) return
    try { await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }) } catch { /* absent */ }
    dirPromise = null
}

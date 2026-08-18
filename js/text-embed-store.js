/**
 * text-embed-store — OPFS cache of phrase → MobileCLIP2 vector (main thread).
 *
 * A vector is 2 KB and costs a 41 MB model build to produce, so caching them is
 * the difference between "the text encoder loads once ever" and "it loads on
 * every search". The main thread checks this BEFORE spawning the detect worker;
 * an all-hit query never builds the encoder at all.
 *
 * One packed file rather than a file per phrase: entries are tiny and the whole
 * set is wanted at once, so per-file overhead would dominate. Reads and writes
 * are best-effort — a miss only means re-encoding.
 *
 * Keyed by the encoder id, so regenerating the model invalidates the cache
 * instead of mixing vectors from two different embedding spaces.
 */

const DIR = 'seglab-text-embeds'
const FILE = 'mclip2-b.v1.bin'
const DIM = 512
const MAX_ENTRIES = 4000

const opfsAvailable = () => typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory

let dirPromise = null
const getDir = () => {
    dirPromise ??= navigator.storage.getDirectory().then((root) => root.getDirectoryHandle(DIR, { create: true }))
    return dirPromise
}

/** phrase → Float32Array(DIM); insertion order doubles as LRU recency. */
let cache = null
let loadPromise = null
let dirty = false
let writeTimer = null

const pack = () => {
    const phrases = [...cache.keys()]
    const header = new TextEncoder().encode(JSON.stringify({ dim: DIM, phrases }))
    const buf = new ArrayBuffer(4 + header.byteLength + phrases.length * DIM * 4)
    new DataView(buf).setUint32(0, header.byteLength, true)
    new Uint8Array(buf, 4, header.byteLength).set(header)
    const body = new Float32Array(buf, 4 + header.byteLength)
    let at = 0
    for (const vec of cache.values()) { body.set(vec, at); at += DIM }
    return buf
}

const unpack = (buf) => {
    const map = new Map()
    if (buf.byteLength < 4) return map
    const headerLen = new DataView(buf).getUint32(0, true)
    if (headerLen <= 0 || 4 + headerLen > buf.byteLength) return map
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)))
    if (meta.dim !== DIM) return map // different encoder — drop rather than mix spaces
    const body = new Float32Array(buf, 4 + headerLen)
    meta.phrases.forEach((p, i) => {
        const vec = body.subarray(i * DIM, (i + 1) * DIM)
        if (vec.length === DIM) map.set(p, Float32Array.from(vec))
    })
    return map
}

const load = () => {
    loadPromise ??= (async () => {
        cache = new Map()
        if (!opfsAvailable()) return cache
        try {
            const dir = await getDir()
            const file = await (await dir.getFileHandle(FILE)).getFile()
            cache = unpack(await file.arrayBuffer())
        } catch { /* absent or corrupt — start empty */ }
        return cache
    })()
    return loadPromise
}

const flush = async () => {
    if (!dirty || !opfsAvailable()) return
    dirty = false
    try {
        const dir = await getDir()
        // Write to a temp name then move, so a crash mid-write cannot leave a
        // truncated cache behind (same contract as embed-store).
        const tmp = await dir.getFileHandle(`${FILE}.tmp`, { create: true })
        const w = await tmp.createWritable()
        await w.write(pack())
        await w.close()
        await tmp.move(FILE)
    } catch { /* quota / unsupported — cache stays in memory only */ }
}

const scheduleFlush = () => {
    dirty = true
    if (writeTimer) clearTimeout(writeTimer)
    writeTimer = setTimeout(() => { writeTimer = null; flush() }, 800)
}

/** Look `phrases` up. → { hits: Map<phrase, Float32Array>, misses: string[] } */
export const lookupPhrases = async (phrases) => {
    const map = await load()
    const hits = new Map()
    const misses = []
    for (const p of phrases) {
        const vec = map.get(p)
        if (vec) {
            hits.set(p, vec)
            map.delete(p) // re-insert to refresh recency
            map.set(p, vec)
        } else misses.push(p)
    }
    return { hits, misses }
}

/** Persist newly encoded vectors. `vectors` is [phrases.length * DIM]. */
export const rememberPhrases = async (phrases, vectors) => {
    if (!phrases?.length || !vectors?.length) return
    const map = await load()
    phrases.forEach((p, i) => {
        const vec = vectors.subarray(i * DIM, (i + 1) * DIM)
        if (vec.length === DIM) {
            map.delete(p)
            map.set(p, Float32Array.from(vec))
        }
    })
    while (map.size > MAX_ENTRIES) map.delete(map.keys().next().value) // oldest first
    scheduleFlush()
}


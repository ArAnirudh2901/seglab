/**
 * clip-text — precomputed CLIP ViT-B/32 text embeddings for the YOLO-World lane.
 *
 * Runtime CLIP in the browser doesn't fit an 8 GB device (measured: q8 collapses
 * alignment 0.9→0.03, fp16 won't build, fp32 ~250 MB triggers memory pressure).
 * So the embeddings are computed ONCE offline at fp32 (scripts/build-clip-vocab.py)
 * over a fixed vocabulary and shipped as a table; here we just look vectors up —
 * no text model in the browser, fp32 quality, a ~9 MB one-time fetch.
 *
 * Vocabulary is YOLOE's 4585 object words + colours (already every taxonomy
 * kind). Phrases outside it are skipped; a search with no in-vocab phrase misses.
 * embedSlots keeps the same shape the vision lane expects: the phrase + synonyms
 * fill the 32 class slots, padded by repetition.
 */
import { YW_CLASS_SLOTS } from './yolo-world-detect.js'

const DIM = 512
const jsonURL = new URL('../models/clip-vocab/clip-vocab.json', import.meta.url).href
const binURL = new URL('../models/clip-vocab/clip-vocab.f32', import.meta.url).href

let wordIndex = null // Map<lowercased word, row>
let table = null // Float32Array [count*DIM], L2-normalized rows
let loadPromise = null

const load = (progress_callback) => {
    loadPromise ??= (async () => {
        progress_callback?.({ status: 'progress', name: 'clip-vocab', progress: 0 })
        const [meta, buf] = await Promise.all([
            fetch(jsonURL).then((r) => r.json()),
            fetch(binURL).then((r) => r.arrayBuffer()),
        ])
        table = new Float32Array(buf)
        wordIndex = new Map(meta.words.map((w, i) => [w, i]))
        progress_callback?.({ status: 'done', name: 'clip-vocab' })
        return true
    })()
    return loadPromise
}

export const clipTextLoaded = () => !!table
export const disposeClipText = () => { /* table is a small shared cache; the worker's termination frees it */ }

// One embed cache: re-running the same search skips the lookup pass.
let cacheKey = null
let cacheVal = null

/**
 * Look up `phrases` (phrase + taxonomy synonyms) in the precomputed table and
 * pack them into the fixed class slots. Returns { txtFeats, slotNames } (rows
 * are already L2-normalized) or null when none of the phrases are in-vocab.
 */
export const embedSlots = async (phrases, progress_callback) => {
    const raw = (phrases || []).map((p) => String(p || '').trim().toLowerCase()).filter(Boolean)
    if (raw.length === 0) return null
    const key = raw.join('|')
    if (key === cacheKey) return cacheVal

    await load(progress_callback)
    const found = []
    for (const w of raw) {
        const row = wordIndex.get(w)
        if (row !== undefined) found.push([w, row])
    }
    if (found.length === 0) return null

    const n = Math.min(found.length, YW_CLASS_SLOTS)
    const txtFeats = new Float32Array(YW_CLASS_SLOTS * DIM)
    const slotNames = new Array(YW_CLASS_SLOTS)
    for (let slot = 0; slot < YW_CLASS_SLOTS; slot += 1) {
        const [w, row] = found[slot % n] // pad-by-repeat over the in-vocab phrases
        txtFeats.set(table.subarray(row * DIM, row * DIM + DIM), slot * DIM)
        slotNames[slot] = w
    }
    cacheKey = key
    cacheVal = { txtFeats, slotNames }
    return cacheVal
}

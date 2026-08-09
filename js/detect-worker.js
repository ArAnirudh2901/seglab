/**
 * detect-worker — disposable open-vocabulary text-detection shell.
 * Detection runs HERE, not in the SAM worker: onnxruntime-web keeps ONE wasm
 * Memory per worker and it can only grow, so a detect in the segmentation worker
 * would permanently inflate its arena for the rest of the session. sam-client
 * terminates this worker after the dispose policy's window — termination is the
 * only true free. Weights persist in the SW cache, so a respawn pays a session
 * build, never a download.
 *
 * One lane now. The phrase is encoded to MobileCLIP2-B vectors, then the YOLOE
 * text-prompt head runs conditioned on them — no baked vocabulary, no second
 * model, no fallback path.
 *
 * The two sessions are strictly SEQUENTIAL: the text encoder is released before
 * the detector is built, so their footprints never sum. Phrases the main thread
 * already had cached are passed in via `known` and skip the encoder entirely,
 * which is the common case after first use.
 */
import { detectYoloe, DIM } from './yoloe-detect.js'

// Forward download progress to the main thread so the UI can show the one-time
// model pull, matching the SAM lane.
const progress = (info) => self.postMessage({
    type: 'progress',
    detail: {
        lane: 'text',
        name: info?.name,
        status: info?.status,
        file: info?.file,
        progress: info?.progress,
        loaded: info?.loaded,
        total: info?.total,
    },
})

const runText = async (payload) => {
    const phrases = payload.phrases || []
    if (!phrases.length) return { results: [], slotNames: [], backend: null, learned: [] }

    const known = new Map(payload.known || [])
    const misses = phrases.filter((p) => !known.has(p))
    const learned = []

    if (misses.length) {
        // Built and released inside encodePhrases — nothing survives into the
        // detector's allocation.
        const { encodePhrases } = await import('./text-encode.js')
        const { vectors } = await encodePhrases(misses, progress)
        misses.forEach((p, i) => {
            const vec = vectors.slice(i * DIM, (i + 1) * DIM)
            known.set(p, vec)
            learned.push([p, vec])
        })
    }

    // Exactly one slot per phrase — the class axis is dynamic, so classIdx maps
    // straight back to `phrases` and the head's top-300 is not spent on
    // duplicates.
    const txtFeats = new Float32Array(phrases.length * DIM)
    phrases.forEach((p, i) => {
        const vec = known.get(p)
        if (vec) txtFeats.set(vec, i * DIM)
    })

    // All frames share ONE session: the tiled pass would otherwise pay a session
    // build (and, with idleMs 0, a whole worker respawn) per tile.
    const results = []
    let backend = null
    for (const frame of payload.frames || []) {
        const r = await detectYoloe({ frame, txtFeats, threshold: payload.threshold, dispose: false })
        results.push(r.dets)
        backend = r.backend
    }
    return { results, slotNames: phrases, backend, learned }
}

self.onmessage = async (event) => {
    const { id, payload } = event.data || {}
    if (!id) return
    try {
        // dispose:false — this worker's termination IS the disposal.
        const result = await runText(payload)
        // Transfer the newly learned vectors rather than structured-cloning them.
        self.postMessage({ id, ok: true, result }, result.learned.map(([, v]) => v.buffer))
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

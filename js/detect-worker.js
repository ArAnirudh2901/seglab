/**
 * detect-worker — disposable text-detector shell (both YOLO lanes).
 * Text detection runs HERE, not in the SlimSAM worker: onnxruntime-web keeps ONE
 * wasm Memory per worker and it can only grow, so a detect in the segmentation
 * worker would permanently inflate its arena for the rest of the session.
 * sam-client terminates this worker after the dispose policy's window —
 * termination is the only true free. Weights persist in the SW cache, so a
 * respawn pays a session build, never a download.
 *
 * Two lanes, chosen by `payload.lane`:
 *   • 'yoloe'     — prompt-free baked vocab (fast, no text encoder)
 *   • 'yoloworld' — open vocab: CLIP-encode the phrases (clip-text) → run the
 *                   YOLO-World vision head conditioned on those embeddings.
 */
import { detectYoloe } from './yoloe-detect.js'

// Forward transformers.js download progress (CLIP text encoder) to the main
// thread so the UI can show the one-time model pull, matching the SAM lane.
const progress = (info) => self.postMessage({
    type: 'progress',
    detail: { lane: 'text', name: info?.name, status: info?.status, file: info?.file, progress: info?.progress, loaded: info?.loaded, total: info?.total },
})

const runYoloWorld = async (payload) => {
    const [{ detectYoloWorld }, { embedSlots }] = await Promise.all([
        import('./yolo-world-detect.js'),
        import('./clip-text.js'),
    ])
    const emb = await embedSlots(payload.phrases, progress)
    if (!emb) return { dets: [], slotNames: [], backend: null }
    const { dets, backend } = await detectYoloWorld({
        frame: payload.frame, txtFeats: emb.txtFeats, threshold: payload.threshold, scale: payload.scale, dispose: false,
    })
    return { dets, slotNames: emb.slotNames, backend }
}

self.onmessage = async (event) => {
    const { id, payload } = event.data || {}
    if (!id) return
    try {
        // dispose:false — this worker's termination IS the disposal.
        const result = payload?.lane === 'yoloworld'
            ? await runYoloWorld(payload)
            : await detectYoloe({ ...payload, dispose: false })
        self.postMessage({ id, ok: true, result })
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

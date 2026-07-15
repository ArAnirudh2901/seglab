/**
 * detect-worker — disposable text-detector shell.
 * Text search runs here, NOT in the segmentation worker: onnxruntime-web keeps
 * ONE wasm Memory per worker and it can only grow, so a single 960² detect
 * would permanently inflate the segmentation worker's arena past what the
 * 8 GB target survives. sam-client terminates this worker after the dispose
 * policy's window — termination is the only true free. Weights persist in the
 * SW cache, so a respawn pays a session build, never a download.
 */
import { detect } from './detect-engine.js'

self.onmessage = async (event) => {
    const { id, payload } = event.data || {}
    if (!id) return
    try {
        const progress_callback = (info) => self.postMessage({
            type: 'progress',
            detail: { lane: 'text', status: info?.status, file: info?.file, progress: info?.progress, loaded: info?.loaded, total: info?.total },
        })
        // dispose:false — this worker's lifetime IS the disposal.
        const result = await detect({ ...payload, dispose: false, progress_callback })
        self.postMessage({ id, ok: true, result })
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

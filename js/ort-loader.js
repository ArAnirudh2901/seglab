/**
 * ort-loader — the one place onnxruntime-web is imported and versioned.
 *
 * MUST stay >= 1.24.3. The 1.22 WebGPU fp16 kernels compute wrong results
 * SILENTLY (spikes/sam21/ortver.html — posFrac 0.9962 against a correct 0.2910),
 * and every model this app ships is fp16 or 4-bit. The version used to be a
 * literal in each lane, and two of the three had drifted back to 1.22 on the CDN
 * fallback path — a path that only fires when the vendored copy is missing, i.e.
 * exactly where nobody would go looking for wrong masks.
 *
 * Keep in step with scripts/download-models.mjs, which vendors this version into
 * lib/ort-web/; verify.mjs gates both against a stray `onnxruntime-web@` literal.
 */

export const ORT_VERSION = '1.27.0'

const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`
const ORT_LOCAL = new URL('../lib/ort-web/', import.meta.url).href
const ORT_FILE = 'ort.webgpu.bundle.min.mjs'

/** Threads for the wasm EP: capped at 4, never more than cores-1, and 1 without
 *  cross-origin isolation (the threaded build needs SharedArrayBuffer). */
export const autoThreads = () => (globalThis.crossOriginIsolated
    ? Math.max(1, Math.min(4, (globalThis.navigator?.hardwareConcurrency || 4) - 1))
    : 1)

/**
 * WebGPU EP entry, with the buffer cache mode ORT does not default to.
 *
 * ORT's WebGPU EP caches storage buffers in `Bucket` mode: every intermediate is
 * rounded up to a bucket and never returned. It is not the session that costs —
 * building the encoder is 226 MB of GPU process; the first RUN takes it to
 * 1117 MB and holds there. Isolated on the encoder alone (one encode + three
 * decodes, fresh profile, process-tree footprint):
 *   Bucket (default) gpu 1128 MB · all-Chrome 1924 MB · encode 885 ms
 *   lazyRelease      gpu  290 MB · all-Chrome 1157 MB · encode 665 ms
 *   Simple           gpu 1444 MB · all-Chrome 2224 MB
 *   Disabled         crashes — "used in submit while destroyed" on gpu-buffer outputs
 * Logits are identical (pos 0.898148, sum 279177.323, iou 0.993164), so this is
 * a pure reclaim, and the encode is not slower.
 *
 * On the whole app (NEF, 1536x1024 proxy, five clicks), three runs per arm:
 *   Bucket       peak 2369 / 2336 / 2351 MB · gpu 1335 / 1306 / 1311
 *   lazyRelease  peak 2035 / 2036 / 2034 MB · gpu 1016 / 1003 / 1005
 * ~315 MB, and it is what carries the app under the 2.2 GB ceiling. The app
 * keeps more than the isolated lane because the RAW decode, display frame and
 * proxy are live alongside the encoder.
 *
 * MASK LANE ONLY, and it has to stay that way if another lane is ever added:
 * measured on the short-lived worker of the (since removed) text lane the
 * result inverted — a session that builds, runs once and terminates pays more
 * in renderer staging than the cache retains (1945/1957/2138 MB on Bucket vs
 * 2930/2775/2782 MB on lazyRelease, three runs each). One long-held encode and
 * a short repeated one want opposite policies.
 *
 * The two session options that ARE on the forwarded key list were both tried on
 * this lane and both rejected. `enableGraphCapture: true` fails session
 * construction outright — the lane has no wasm fallback, so the click path comes
 * back ready:false / device:null / no mask. `preserveDevice: true` builds and
 * segments identically (score 0.8310546875) but costs rather than saves:
 * cold-scenario peaks 1887/2014/2157 MB against 1953/2018/1896 MB without it.
 *
 * `epConfig` is NOT upstream API: ORT-Web forwards a fixed key list to the
 * native EP, and the cache modes are not on it. scripts/download-models.mjs
 * patches the vendored bundle to pass this object through; verify.mjs gates the
 * patch. The CDN fallback below is UNPATCHED — it still runs, just without the
 * reclaim, which is why the vendored copy is the supported path.
 */
export const webgpuEP = () => ({ name: 'webgpu', epConfig: { storageBufferCacheMode: 'lazyRelease' } })

let ortPromise = null

/** 'local' | 'cdn' — 'cdn' means the epConfig patch is absent (see webgpuEP). */
export let ortSource = null

/**
 * Import ORT — vendored first, CDN second — and point its wasm loader at the
 * same base, so the WASM EP resolves from wherever the bundle came from.
 * Memoised per realm: ORT's `env` is global, so a second import would only
 * re-run the settings.
 */
export const loadOrt = ({ threads = autoThreads } = {}) => {
    ortPromise ??= (async () => {
        for (const base of [ORT_LOCAL, ORT_CDN]) {
            try {
                const ort = await import(/* @vite-ignore */ base + ORT_FILE)
                ortSource = base === ORT_LOCAL ? 'local' : 'cdn'
                if (ortSource === 'cdn') console.warn(
                    '[ort-loader] vendored bundle missing — running the UNPATCHED CDN copy: '
                    + 'storageBufferCacheMode is ignored and the mask lane costs ~315 MB more. '
                    + 'Run `node scripts/download-models.mjs` to restore lib/ort-web/.')
                ort.env.wasm.wasmPaths = base
                ort.env.wasm.numThreads = typeof threads === 'function' ? threads() : threads
                // ORT requests its OWN adapter (bundle: requestAdapter({power-
                // Preference, forceFallbackAdapter})), so without this the lanes
                // get the browser default while capability.js probed with
                // 'high-performance' — on a dual-GPU Mac, a different GPU. Only
                // read before the first session, and loadOrt is memoised.
                // Deprecated upstream for env.webgpu.device, not used: owning
                // the device would break the release-frees-976 MB contract.
                if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance'
                return ort
            } catch { /* next source */ }
        }
        throw new Error(`onnxruntime-web ${ORT_VERSION} unavailable (no vendored copy and CDN blocked)`)
    })()
    return ortPromise
}

/**
 * model-assets — offline-first loading of transformers.js, ORT wasm, weights.
 * Prefers what scripts/download-models.mjs vendored under lib/ + models/,
 * falling back to pinned CDNs per piece.
 *
 * Paths are import.meta.url-relative so workers resolve them identically.
 * Named model-assets, not asset-store: asset-store.js owns original-image
 * custody. Adapted from upstream 1943302.
 */

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'
const LOCAL_TRANSFORMERS = new URL('../lib/transformers/transformers.min.js', import.meta.url)
const LOCAL_ORT_DIR = new URL('../lib/ort/', import.meta.url)
const LOCAL_MODELS_DIR = new URL('../models/', import.meta.url)

const headOk = async (url) => {
    try { return (await fetch(url, { method: 'HEAD' })).ok } catch { return false }
}

// ORT build pick, mirroring transformers.js upstream (PR #1700): only Safari
// WITHOUT WebGPU (< 26) gets the plain CPU-only build — its wasm engine chokes
// on asyncify. Safari 26+ ships WebGPU and runs the asyncify build, which is
// the one carrying the WebGPU EP; all other browsers always get asyncify.
const isSafariUA = () => {
    const ua = typeof navigator !== 'undefined' ? String(navigator.userAgent || '') : ''
    return /Safari\//.test(ua) && !/Chrom(e|ium)|Edg\//.test(ua)
}
const hasWebGpuApi = () => typeof navigator !== 'undefined' && 'gpu' in navigator
const needsPlainOrtBuild = () => isSafariUA() && !hasWebGpuApi()
const ortName = (ext) => `ort-wasm-simd-threaded${needsPlainOrtBuild() ? '' : '.asyncify'}.${ext}`

/** Whether the ORT build this browser gets can run the WebGPU EP at all.
 *  The plain build has no webgpuInit export, so a 'webgpu' session there dies
 *  with "De().webgpuInit is not a function" even when navigator.gpu exists.
 *  Must stay in sync with ortName's pick. */
export const ortWebGpuSupported = () => hasWebGpuApi() && !needsPlainOrtBuild()

let manifestPromise = null

/** Vendored manifest, or null if nothing vendored. Cached — probe once. */
export const vendoredManifest = () => {
    manifestPromise ??= fetch(new URL('manifest.json', LOCAL_MODELS_DIR))
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null)
    return manifestPromise
}

/** Whether `model` (e.g. 'Xenova/slimsam-77-uniform') was vendored locally. */
export const isVendored = async (model) => {
    const manifest = await vendoredManifest()
    if (!manifest) return false
    return manifest.files.some((f) => f.path.includes(model))
}

const configureEnv = async (T, source) => {
    try {
        const [modelsOk, ortOk] = await Promise.all([
            headOk(new URL('manifest.json', LOCAL_MODELS_DIR)),
            headOk(new URL(ortName('wasm'), LOCAL_ORT_DIR)),
        ])
        if (modelsOk) {
            T.env.allowLocalModels = true // defaults false in browsers
            T.env.localModelPath = LOCAL_MODELS_DIR.href
            T.env.allowRemoteModels = true // per-file fallback for anything unvendored
        }
        if (ortOk) {
            // Object form keeps transformers' wasm Cache-API caching active.
            T.env.backends.onnx.wasm.wasmPaths = {
                mjs: new URL(ortName('mjs'), LOCAL_ORT_DIR).href,
                wasm: new URL(ortName('wasm'), LOCAL_ORT_DIR).href,
            }
        }
        // Cache Storage: caches weights on download so an unvendored
        // install survives reloads.
        T.env.useBrowserCache = true
        // Threads need COOP/COEP (plain hosts run 1); leave a core for the OS.
        const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4
        if (T.env.backends?.onnx?.wasm) T.env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, cores - 1))
        console.log(`[seglab] transformers: ${source}${modelsOk ? ' · local models' : ' · remote models'}${ortOk ? ' · local ort' : ''}`)
        return { source, modelsOk, ortOk }
    } catch (err) {
        console.warn('[seglab] asset wiring failed; using CDN defaults:', err?.message)
        return { source, modelsOk: false, ortOk: false }
    }
}

let transformersPromise = null

/** Import transformers.js (vendored copy first) with env configured. */
export const loadTransformersModule = () => {
    transformersPromise ??= (async () => {
        let mod
        let source = 'local'
        try {
            mod = await import(/* @vite-ignore */ LOCAL_TRANSFORMERS.href)
        } catch {
            source = 'cdn'
            mod = await import(/* @vite-ignore */ TRANSFORMERS_CDN)
        }
        await configureEnv(mod, source)
        return mod
    })()
    return transformersPromise
}

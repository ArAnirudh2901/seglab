/**
 * asset-store — offline-first loading of transformers.js, ORT wasm, weights
 * ---------------------------------------------------------------------------
 * scripts/download-models.mjs vendors assets under lib/ and models/
 * (gitignored). This module prefers those, falling back to the pinned CDNs
 * per piece, so a clean checkout still works online. Runs identically on the
 * main thread and inside workers (all paths are import.meta.url-relative).
 */

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0'
const LOCAL_TRANSFORMERS = new URL('../lib/transformers/transformers.min.js', import.meta.url)
const LOCAL_ORT_DIR = new URL('../lib/ort/', import.meta.url)
const LOCAL_MODELS_DIR = new URL('../models/', import.meta.url)

const headOk = async (url) => {
    try { return (await fetch(url, { method: 'HEAD' })).ok } catch { return false }
}

// Safari needs the non-asyncify ORT build (mirrors transformers.js's own pick).
const isSafariUA = () => {
    const ua = typeof navigator !== 'undefined' ? String(navigator.userAgent || '') : ''
    return /Safari\//.test(ua) && !/Chrom(e|ium)|Edg\//.test(ua)
}
const ortName = (ext) => `ort-wasm-simd-threaded${isSafariUA() ? '' : '.asyncify'}.${ext}`

const configureTransformersEnv = async (t, source) => {
    try {
        const [modelsOk, ortOk] = await Promise.all([
            headOk(new URL('manifest.json', LOCAL_MODELS_DIR)),
            headOk(new URL(ortName('wasm'), LOCAL_ORT_DIR)),
        ])
        if (modelsOk) {
            t.env.allowLocalModels = true // defaults false in browsers
            t.env.localModelPath = LOCAL_MODELS_DIR.href
            t.env.allowRemoteModels = true // per-file fallback for anything unvendored
        }
        if (ortOk) {
            // Object form keeps transformers' wasm Cache-API caching active.
            t.env.backends.onnx.wasm.wasmPaths = {
                mjs: new URL(ortName('mjs'), LOCAL_ORT_DIR).href,
                wasm: new URL(ortName('wasm'), LOCAL_ORT_DIR).href,
            }
        }
        console.log(`[seglab] transformers: ${source}${modelsOk ? ' · local models' : ' · remote models'}${ortOk ? ' · local ort' : ''}`)
    } catch (err) {
        console.warn('[seglab] asset wiring failed; using CDN defaults:', err?.message)
    }
}

let transformersPromise = null

/** Import transformers.js (vendored copy first) with env configured. */
export const loadTransformersModule = () => {
    transformersPromise ??= (async () => {
        let mod
        let source = 'local'
        try {
            mod = await import(LOCAL_TRANSFORMERS.href)
        } catch {
            source = 'cdn'
            mod = await import(TRANSFORMERS_CDN)
        }
        await configureTransformersEnv(mod, source)
        return mod
    })()
    return transformersPromise
}

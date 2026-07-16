/**
 * cv-refine-worker — owns the Wasm hygiene module + the grayscale guide
 * ------------------------------------------------------------------------
 * One job at a time (the heavy queue upstream already serializes). The wasm
 * module lazy-loads on the first refine; if it can't load or a call fails,
 * the worker falls back to the identical JS pipeline — a failed refinement
 * never fails the user's selection.
 *
 *   in : { id, op:'set-guide', revision, bitmap (transferred, ≤768) }
 *   in : { id, op:'refine', revision, width, height, alpha, alphaRaw
 *          (both transferred), seeds:[[x,y]…], options:{minArea,openRadius,
 *          closeRadius, wasm} }
 *   in : { id, op:'reset', revision } · { id, op:'dispose' }
 *   out: { id, ok:true, result:{ alpha, alphaRaw, summary, usable, reason,
 *          stats, bandPixels, refiner:'wasm'|'js' } } — buffers transferred
 *   out: { id, ok:false, error }
 */

import { cleanupMaskAlpha, summarizeMaskAlpha, validateClickMask } from './sam-core.js'
import { refineMaskEdges } from './edge-refine.js'

const MAX_SIDE = 768
const MAX_SEEDS = 64

let modPromise = null
let mod = null
let maskPtr = 0
let scratchPtr = 0
let seedsPtr = 0
let wasmBroken = false
let guide = null // { revision, gray: Float32Array, width, height }

const ensureModule = async () => {
    if (wasmBroken) return null
    if (mod) return mod
    // Dynamic import: the wasm module loads only when a mask needs refining.
    modPromise ??= import('../public/wasm/cv-refine.js')
        .then(({ default: createCvModule }) => createCvModule())
        .then((m) => {
            maskPtr = m._malloc(MAX_SIDE * MAX_SIDE)
            scratchPtr = m._malloc(MAX_SIDE * MAX_SIDE)
            seedsPtr = m._malloc(MAX_SEEDS * 8)
            if (!maskPtr || !scratchPtr || !seedsPtr) throw new Error('wasm workspace allocation failed')
            mod = m
            return m
        })
    try {
        return await modPromise
    } catch (err) {
        wasmBroken = true
        console.warn('[seglab] cv wasm unavailable; JS fallback:', err?.message)
        return null
    }
}

const runWasm = (m, alpha, width, height, seeds, options) => {
    m.HEAPU8.set(alpha, maskPtr)
    const nSeeds = Math.min(seeds.length, MAX_SEEDS)
    const seedView = new Int32Array(m.HEAPU8.buffer, seedsPtr, nSeeds * 2)
    for (let i = 0; i < nSeeds; i += 1) {
        seedView[i * 2] = Math.round(seeds[i][0])
        seedView[i * 2 + 1] = Math.round(seeds[i][1])
    }
    const kept = m._refine_mask_multi(
        maskPtr, scratchPtr, width, height, seedsPtr, nSeeds,
        options.minArea | 0, options.openRadius | 0, options.closeRadius | 0,
    )
    if (kept < 0) throw new Error(`cv_refine error ${kept}`)
    alpha.set(m.HEAPU8.subarray(maskPtr, maskPtr + width * height))
    return kept
}

self.onmessage = async (event) => {
    const { id, op, revision = 0, ...payload } = event.data || {}
    if (!id || !op) return
    try {
        if (op === 'set-guide') {
            const { bitmap } = payload
            try {
                if (Math.max(bitmap.width, bitmap.height) > MAX_SIDE) {
                    throw new Error(`guide exceeds ${MAX_SIDE}px`)
                }
                const c = new OffscreenCanvas(bitmap.width, bitmap.height)
                const ctx = c.getContext('2d', { willReadFrequently: true })
                ctx.drawImage(bitmap, 0, 0)
                const px = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data
                const gray = new Float32Array(bitmap.width * bitmap.height)
                for (let i = 0; i < gray.length; i += 1) {
                    const j = i * 4
                    gray[i] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255
                }
                guide = { revision, gray, width: bitmap.width, height: bitmap.height }
            } finally {
                try { bitmap.close() } catch { /* already closed */ }
            }
            self.postMessage({ id, ok: true, result: { revision } })
            return
        }
        if (op === 'refine') {
            const { width, height, alpha, alphaRaw, seeds = [], options = {} } = payload
            if (!Number.isInteger(width) || !Number.isInteger(height)
                || width <= 0 || height <= 0 || Math.max(width, height) > MAX_SIDE) {
                throw new Error(`refine rejects dimensions ${width}×${height} (max ${MAX_SIDE})`)
            }
            if (!(alpha instanceof Uint8Array) || alpha.length !== width * height) {
                throw new Error('refine: alpha buffer does not match dimensions')
            }
            const opts = { minArea: 48, openRadius: 0, closeRadius: 0, ...options }
            let refiner = 'js'
            let stats
            const m = opts.wasm === false ? null : await ensureModule()
            if (m) {
                try {
                    stats = { kept: runWasm(m, alpha, width, height, seeds, opts) }
                    refiner = 'wasm'
                } catch (err) {
                    wasmBroken = true
                    console.warn('[seglab] cv wasm call failed; JS fallback:', err?.message)
                }
            }
            if (refiner === 'js') stats = cleanupMaskAlpha(alpha, width, height, seeds)
            let bandPixels = 0
            if (guide && guide.revision === revision && guide.width === width && guide.height === height) {
                bandPixels = refineMaskEdges(alpha, width, height, guide.gray).bandPixels
            }
            const summary = summarizeMaskAlpha(alpha, width, height)
            const verdict = validateClickMask(summary)
            self.postMessage({
                id,
                ok: true,
                result: { alpha, alphaRaw, width, height, summary, usable: verdict.usable, reason: verdict.reason, stats, bandPixels, refiner },
            }, [alpha.buffer, alphaRaw.buffer])
            return
        }
        if (op === 'reset') {
            if (!guide || !revision || guide.revision < revision) guide = null
            self.postMessage({ id, ok: true, result: { revision } })
            return
        }
        if (op === 'dispose') {
            guide = null
            if (mod) {
                try {
                    mod._free(maskPtr)
                    mod._free(scratchPtr)
                    mod._free(seedsPtr)
                } catch { /* heap gone with the worker */ }
            }
            mod = null
            modPromise = null
            maskPtr = 0
            scratchPtr = 0
            seedsPtr = 0
            self.postMessage({ id, ok: true, result: {} })
            return
        }
        throw new Error(`Unknown op: ${op}`)
    } catch (err) {
        self.postMessage({ id, ok: false, error: String(err?.message || err) })
    }
}

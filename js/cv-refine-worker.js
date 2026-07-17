/**
 * cv-refine-worker — owns the cv-refine.wasm instance (one job at a time).
 * One-channel masks in, one-channel masks out, buffers transferred both ways.
 *
 * in : { type:'refine-mask', requestId, revision, width, height,
 *        rgb: ArrayBuffer|null, alpha: ArrayBuffer, seeds: [[x,y],...],
 *        options: { minArea, openRadius, closeRadius } }
 * in : { type:'dispose' }
 * out: { type:'result', requestId, revision, stale:false,
 *        alpha: ArrayBuffer (transfer), width, height, stats:{ms} }
 * out: { type:'error', requestId, revision, error }
 */

const MAX_SIDE = 1024

let modulePromise = null
const getModule = () => {
    modulePromise ??= import('../public/wasm/cv-refine.js')
        .then(({ default: createModule }) => createModule())
    return modulePromise
}

let busy = false

const validate = (msg) => {
    const { width, height, alpha, rgb } = msg
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        return 'invalid dimensions'
    }
    if (Math.max(width, height) > MAX_SIDE) return `dimensions exceed ${MAX_SIDE}px`
    const pixels = width * height
    if (pixels > MAX_SIDE * MAX_SIDE) return 'pixel count overflow'
    if (!(alpha instanceof ArrayBuffer) || alpha.byteLength !== pixels) return 'alpha buffer size mismatch'
    if (rgb != null && (!(rgb instanceof ArrayBuffer) || rgb.byteLength !== pixels * 3)) return 'rgb buffer size mismatch'
    return null
}

self.onmessage = async (event) => {
    const msg = event.data || {}
    if (msg.type === 'dispose') {
        modulePromise = null
        self.close()
        return
    }
    if (msg.type !== 'refine-mask' || !msg.requestId) return
    const reply = (payload, transfer = []) => self.postMessage({ requestId: msg.requestId, revision: msg.revision, ...payload }, transfer)
    if (busy) { reply({ type: 'error', error: 'refine worker is busy' }); return }
    const invalid = validate(msg)
    if (invalid) { reply({ type: 'error', error: invalid }); return }

    busy = true
    const t0 = Date.now()
    let maskPtr = 0
    let scratchPtr = 0
    let rgbPtr = 0
    let cv = null
    try {
        cv = await getModule()
        const { width, height } = msg
        const pixels = width * height
        const opts = msg.options || {}
        maskPtr = cv._malloc(pixels)
        scratchPtr = cv._malloc(pixels)
        if (!maskPtr || !scratchPtr) throw new Error('wasm allocation failed')
        cv.HEAPU8.set(new Uint8Array(msg.alpha), maskPtr)
        if (msg.rgb) {
            rgbPtr = cv._malloc(pixels * 3)
            if (!rgbPtr) throw new Error('wasm allocation failed')
            cv.HEAPU8.set(new Uint8Array(msg.rgb), rgbPtr)
        }
        const seed = Array.isArray(msg.seeds) && msg.seeds.length ? msg.seeds[0] : [-1, -1]
        cv._refine_mask(
            rgbPtr, maskPtr, scratchPtr, width, height,
            Math.round(seed[0]), Math.round(seed[1]),
            opts.minArea | 0, opts.openRadius | 0, opts.closeRadius | 0,
        )
        const out = new Uint8Array(pixels)
        out.set(cv.HEAPU8.subarray(maskPtr, maskPtr + pixels))
        reply({ type: 'result', stale: false, alpha: out.buffer, width, height, stats: { ms: Date.now() - t0 } }, [out.buffer])
    } catch (err) {
        reply({ type: 'error', error: String(err?.message || err) })
    } finally {
        // Free every allocation, including on failed/stale jobs.
        try {
            if (cv) {
                if (maskPtr) cv._free(maskPtr)
                if (scratchPtr) cv._free(scratchPtr)
                if (rgbPtr) cv._free(rgbPtr)
            }
        } catch { /* module gone */ }
        busy = false
    }
}

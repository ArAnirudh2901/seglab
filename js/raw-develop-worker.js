/**
 * raw-develop-worker — owns the raw-develop.wasm (LibRaw) instance, one job at
 * a time. RAW file bytes in, a developed image/jpeg buffer out (transferred).
 * The client disposes this worker after every job, so LibRaw's grown sensor
 * heap is returned to the OS rather than lingering (wasm memory never shrinks).
 *
 * in : { type:'develop', requestId, bytes: ArrayBuffer (transfer),
 *        options: { half, maxMP, quality } }
 * in : { type:'dispose' }
 * out: { type:'result', requestId, jpeg: ArrayBuffer (transfer),
 *        width, height, stats:{ ms } }
 * out: { type:'error', requestId, error }
 */

const MAX_BYTES = 512 * 1024 * 1024   // reject absurd inputs before allocating

let modulePromise = null
const getModule = () => {
    modulePromise ??= import('../public/wasm/raw-develop.js')
        .then(({ default: createModule }) => createModule())
    return modulePromise
}

let busy = false

self.onmessage = async (event) => {
    const msg = event.data || {}
    if (msg.type === 'dispose') {
        modulePromise = null
        self.close()
        return
    }
    if (msg.type !== 'develop' || !msg.requestId) return
    const reply = (payload, transfer = []) => self.postMessage({ requestId: msg.requestId, ...payload }, transfer)
    if (busy) { reply({ type: 'error', error: 'raw-develop worker is busy' }); return }
    if (!(msg.bytes instanceof ArrayBuffer) || msg.bytes.byteLength === 0) {
        reply({ type: 'error', error: 'empty raw buffer' }); return
    }
    if (msg.bytes.byteLength > MAX_BYTES) { reply({ type: 'error', error: 'raw file too large' }); return }

    busy = true
    const t0 = Date.now()
    let rd = null
    let inPtr = 0
    try {
        rd = await getModule()
        const opts = msg.options || {}
        const bytes = new Uint8Array(msg.bytes)
        inPtr = rd._malloc(bytes.length)
        if (!inPtr) throw new Error('wasm allocation failed')
        rd.HEAPU8.set(bytes, inPtr)

        const half = opts.half === false ? 0 : 1
        const maxMP = Math.max(0, Math.round(Number(opts.maxMP) || 0))
        const quality = Math.min(100, Math.max(1, Math.round(Number(opts.quality) || 90)))
        const rc = rd._rd_develop(inPtr, bytes.length, half, maxMP, quality)
        if (rc !== 0) throw new Error(rd.UTF8ToString(rd._rd_error()) || `develop failed (${rc})`)

        const jptr = rd._rd_jpeg()
        const jlen = rd._rd_jpeg_len()
        const width = rd._rd_width()
        const height = rd._rd_height()
        if (!jptr || jlen <= 0) throw new Error('develop produced no jpeg')
        const out = new Uint8Array(jlen)
        out.set(rd.HEAPU8.subarray(jptr, jptr + jlen))
        reply({ type: 'result', jpeg: out.buffer, width, height, stats: { ms: Date.now() - t0 } }, [out.buffer])
    } catch (err) {
        reply({ type: 'error', error: String(err?.message || err) })
    } finally {
        try {
            if (rd) {
                rd._rd_release()
                if (inPtr) rd._free(inPtr)
            }
        } catch { /* module gone */ }
        busy = false
    }
}

/**
 * text-encode — phrase → MobileCLIP2-B 512-d vector, in the disposable worker.
 *
 * This is what makes the search open-vocabulary. The old lane looked phrases up
 * in a 4585-word precomputed table and returned null for anything else, so any
 * description outside that list silently found nothing.
 *
 * MEMORY IS THE WHOLE DESIGN HERE. The app already measures 2634 MB when a SAM
 * encoder session is live, so a text lane that lingers is not affordable:
 *   • the token table is fetched, gathered from, and DROPPED before the ONNX
 *     session is built — peak is max(25, 41) MB, never the sum;
 *   • the session is released the moment the vectors are out;
 *   • the caller persists the results, so the encoder is only ever built for a
 *     phrase this browser has genuinely never seen (js/text-embed-store.js).
 * Everything here dies with the worker regardless — that termination is the only
 * true free of an ORT wasm arena.
 *
 * The 4-bit weights are weight-only MatMulNBits with the EP's native compute,
 * NOT the `quantize_dynamic`/QInt8 scheme that collapsed CLIP alignment
 * 0.9 -> 0.03 in July. Measured against the fp32 tower: min cosine 0.991.
 */

import { CONTEXT, EOT, loadTokenizer, tokenize } from './clip-tokenizer.js'
import { webgpuWorthTrying } from './gpu-adapter.js'
import { loadOrt } from './ort-loader.js'

export const DIM = 512
const VOCAB = 49408

const modelURL = new URL('../models/clip-text/mclip2-text.q4.onnx', import.meta.url).href
const tableURL = new URL('../models/clip-text/mclip2-embed.i8', import.meta.url).href
const scaleURL = new URL('../models/clip-text/mclip2-embed.scale.f32', import.meta.url).href

/**
 * Gather the rows this batch needs and dequantize just those. The table is
 * 25 MB on disk but a query touches a few hundred rows, so nothing large stays
 * referenced once this returns.
 */
const embedTokens = async (tokens, progress) => {
    progress?.({ status: 'progress', name: 'clip-text-table', progress: 0 })
    const [tableBuf, scaleBuf] = await Promise.all([
        fetch(tableURL).then((r) => r.arrayBuffer()),
        fetch(scaleURL).then((r) => r.arrayBuffer()),
    ])
    const q = new Int8Array(tableBuf)
    const scale = new Float32Array(scaleBuf)
    if (scale.length !== VOCAB) throw new Error(`clip-text: scale rows ${scale.length}, expected ${VOCAB}`)
    const out = new Float32Array(tokens.length * DIM)
    for (let i = 0; i < tokens.length; i += 1) {
        const row = tokens[i]
        if (row < 0 || row >= VOCAB) continue
        const s = scale[row]
        const src = row * DIM
        const dst = i * DIM
        for (let k = 0; k < DIM; k += 1) out[dst + k] = q[src + k] * s
    }
    progress?.({ status: 'done', name: 'clip-text-table' })
    return out // tableBuf/scaleBuf drop out of scope here, before the session builds
}

const buildSession = async (ort) => {
    let lastErr
    // Skip a software adapter: it builds and then runs slower than WASM.
    const eps = (await webgpuWorthTrying()) ? ['webgpu', 'wasm'] : ['wasm']
    for (const ep of eps) {
        try {
            return {
                session: await ort.InferenceSession.create(modelURL, {
                    executionProviders: [ep], graphOptimizationLevel: 'all',
                }),
                backend: ep,
            }
        } catch (err) { lastErr = err }
    }
    throw lastErr || new Error('clip-text: no execution provider available')
}

/**
 * `phrases` → { vectors: Float32Array [n*DIM] (L2-normalized), backend }.
 * The session is built and released inside this call: nothing survives it.
 */
export const encodePhrases = async (phrases, progress) => {
    if (!phrases?.length) return { vectors: new Float32Array(0), backend: null }
    await loadTokenizer(progress)
    const tokens = tokenize(phrases)
    const n = phrases.length

    // EOT position per phrase — the tower reads its features from that slot.
    const eot = new BigInt64Array(n)
    for (let i = 0; i < n; i += 1) {
        let at = 0
        for (let k = 0; k < CONTEXT; k += 1) if (tokens[i * CONTEXT + k] === EOT) { at = k; break }
        eot[i] = BigInt(at)
    }

    const embeds = await embedTokens(tokens, progress)
    const ort = await loadOrt()
    progress?.({ status: 'progress', name: 'clip-text', progress: 0 })
    const { session, backend } = await buildSession(ort)
    try {
        const out = await session.run({
            token_embeds: new ort.Tensor('float32', embeds, [n, CONTEXT, DIM]),
            eot_index: new ort.Tensor('int64', eot, [n]),
        })
        const vectors = Float32Array.from(out[session.outputNames[0]].data)
        progress?.({ status: 'done', name: 'clip-text' })
        return { vectors, backend }
    } finally {
        // Release immediately — the vectors are 2 KB each and the session is 41 MB.
        try { session.release?.() } catch { /* already gone */ }
    }
}

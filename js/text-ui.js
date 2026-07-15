/**
 * text-ui — text-select orchestration (main thread).
 * Phrase → letterboxed detector frame from the original → OWLv2 boxes → ranked
 * candidates in PROXY coords (ready to feed the existing box prompt). No app
 * state or DOM here; app.js owns the input, overlay, and selection glue.
 */
import {
    DETECTOR_INPUT, DETECTOR_PAD, letterboxPlan, normalizePhrase, rankDetections, scaleBox, unletterboxBox,
} from './text-core.js'
import { getTransform, getBoundedOriginal } from './asset-store.js'
import { detectText } from './sam-client.js'

/** Original → gray-padded square of RGB bytes at the detector's native size.
 *  Only the photo's own rows are read back and the padding is written straight
 *  into the destination, so no square of RGBA is ever materialized. */
const buildFrame = async (plan) => {
    const { side, dw, dh } = plan
    const { source, owned } = await getBoundedOriginal({ maxSide: side }) // bounded decode, never full-res
    if (!source) return null
    let px
    const canvas = document.createElement('canvas')
    canvas.width = dw
    canvas.height = dh
    try {
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(source, 0, 0, dw, dh)
        px = ctx.getImageData(0, 0, dw, dh).data
    } finally {
        if (owned) { try { source.close() } catch { /* closed */ } }
        canvas.width = canvas.height = 0 // drop the RGBA backing before the model allocates
    }
    const data = new Uint8ClampedArray(side * side * 3).fill(DETECTOR_PAD)
    for (let y = 0; y < dh; y++) {
        let s = y * dw * 4
        let d = y * side * 3
        for (let x = 0; x < dw; x++, s += 4, d += 3) {
            data[d] = px[s]
            data[d + 1] = px[s + 1]
            data[d + 2] = px[s + 2]
        }
    }
    return { data, width: side, height: side }
}

/** Detect `phrase` in the held original. Returns
 *  { candidates:[{ box:[x0,y0,x1,y1] proxy coords, score, label }], multi,
 *    backend, display } or null when there's no phrase/original. */
export const detectCandidates = async (phrase, { rankThreshold = 0.12 } = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null

    const plan = letterboxPlan(tf.originalW, tf.originalH, DETECTOR_INPUT)
    const frame = await buildFrame(plan)
    if (!frame) return null
    const { dets, backend } = await detectText(frame, norm.labels, { threshold: 0.05 })

    // Square-normalized → original px (dropping padding-only boxes), then proxy
    // px. Reject before ranking so a bogus box can't raise the relative floor.
    const mapped = []
    for (const d of dets) {
        const box = unletterboxBox(d.box, plan)
        if (box) mapped.push({ box, score: d.score, label: d.label })
    }
    const kx = tf.proxyW / tf.originalW
    const ky = tf.proxyH / tf.originalH
    const candidates = rankDetections(mapped, { threshold: rankThreshold, iou: 0.5, topK: 8, relative: 0.35 })
        .map((d) => ({ box: scaleBox(d.box, kx, ky), score: d.score, label: d.label }))
    return { candidates, multi: norm.multi, backend, display: norm.display }
}

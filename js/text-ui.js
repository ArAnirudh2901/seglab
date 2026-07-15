/**
 * text-ui — text-select orchestration (main thread).
 * Phrase → letterboxed detector frame from the original → grounded boxes → ranked
 * candidates in PROXY coords (ready to feed the existing box prompt). No app
 * state or DOM here; app.js owns the input, overlay, and selection glue.
 */
import {
    colorEvidenceForBox, DETECTOR_INPUT, DETECTOR_PAD, letterboxPlan, normalizePhrase, rankDetections, scaleBox, unletterboxBox,
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
    return { data, width: side, height: side, contentWidth: dw, contentHeight: dh }
}

/** Prefer candidates that visibly contain a colour named in the prompt. A
 * broad false-positive box may include a few red pixels; it should not beat a
 * tight red-flower box. If the image has no strong colour evidence, leave the
 * detector ranking untouched rather than inventing a rejection. */
const focusRequestedColor = (dets, frame, color) => {
    if (!color || dets.length === 0) return dets
    const withEvidence = dets.map((d) => ({ ...d, colorEvidence: colorEvidenceForBox(frame, d.rawBox, color) }))
    const best = withEvidence.reduce((max, d) => Math.max(max, d.colorEvidence), 0)
    if (best < 0.025) return dets
    const floor = Math.max(0.004, best * 0.2)
    return withEvidence
        .filter((d) => d.colorEvidence >= floor)
        .map((d) => ({
            ...d,
            modelScore: d.score,
            // Preserve some model confidence but make visible colour evidence
            // decisive when the user explicitly supplied a colour word.
            score: d.score * (0.15 + 0.85 * (d.colorEvidence / best)),
        }))
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
        if (box) mapped.push({ box, rawBox: d.box, score: d.score, label: d.label })
    }
    const kx = tf.proxyW / tf.originalW
    const ky = tf.proxyH / tf.originalH
    // The fallback detector can emit low-confidence, scene-sized guesses.
    // Start from its established floor, then demand a much closer score to
    // the best result; Grounding DINO benefits from the same duplicate guard.
    const eligible = mapped.filter((d) => d.score >= rankThreshold)
    const focused = focusRequestedColor(eligible, frame, norm.color)
    const candidates = rankDetections(focused, {
        threshold: norm.color && focused !== eligible ? 0 : rankThreshold,
        iou: 0.5,
        topK: 5,
        relative: 0.6,
    })
        .map((d) => ({ box: scaleBox(d.box, kx, ky), score: d.score, label: d.label }))
    return { candidates, multi: norm.multi, backend, display: norm.display }
}

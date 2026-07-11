/**
 * text-ui — text-select orchestration (main thread).
 * Phrase → detector canvas from the original → OWLv2 boxes → ranked
 * candidates in PROXY coords (ready to feed the existing box prompt). No app
 * state or DOM here; app.js owns the input, overlay, and selection glue.
 */
import { normalizePhrase, rankDetections, scaleBox } from './text-core.js'
import { getTransform, getOriginalForExport } from './asset-store.js'
import { detectText } from './sam-client.js'

/** Detect `phrase` in the held original. Returns
 *  { candidates:[{ box:[x0,y0,x1,y1] proxy coords, score, label }], multi,
 *    backend, display } or null when there's no phrase/original. */
export const detectCandidates = async (phrase, budget, { rankThreshold = 0.1 } = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null

    // Detector canvas: original scaled to budget.detectorCanvas long side.
    const side = budget.detectorCanvas || 960
    const scale = Math.min(1, side / Math.max(tf.originalW, tf.originalH))
    const dw = Math.max(1, Math.round(tf.originalW * scale))
    const dh = Math.max(1, Math.round(tf.originalH * scale))
    const canvas = document.createElement('canvas')
    canvas.width = dw
    canvas.height = dh
    const { source, owned } = await getOriginalForExport()
    try {
        canvas.getContext('2d').drawImage(source, 0, 0, dw, dh)
    } finally {
        if (owned) { try { source.close() } catch { /* closed */ } }
    }

    const { dets, backend } = await detectText(canvas, norm.labels, { threshold: 0.02 })
    // Detector-canvas px → proxy px (both are aspect-preserving scalings of the original).
    const kx = tf.proxyW / dw
    const ky = tf.proxyH / dh
    const candidates = rankDetections(dets, { threshold: rankThreshold, iou: 0.5, topK: 8 })
        .map((d) => ({ box: scaleBox(d.box, kx, ky), score: d.score, label: d.label }))
    return { candidates, multi: norm.multi, backend, display: norm.display }
}

/**
 * text-ui — text-select orchestration (main thread).
 * Phrase → letterboxed detector frame from the original → grounded boxes → ranked
 * candidates in PROXY coords (ready to feed the existing box prompt). No app
 * state or DOM here; app.js owns the input, overlay, and selection glue.
 */
import {
    collapseToObject, colorEvidenceForBox, dominantColorForBox, DETECTOR_PAD, letterboxPlan, normalizePhrase, rankDetections, scaleBox, unletterboxBox, YOLOE_INPUT,
} from './text-core.js'
import { labelMatchesQuery, expandQuery } from './search-taxonomy.js'
import { getTransform, getBoundedOriginal } from './asset-store.js'
import { detectTextYoloe, detectTextYoloWorld } from './sam-client.js'

/** A ranked detection → an app-facing candidate, tagged with the dominant
 *  colour bucket in its detector-frame box (the colour sub-class facet).
 *  `d.box` is still detector-square coords; `kx/ky` map it to proxy coords.
 *  `d.rawBox` is square-normalized against `frame` (colour is sampled there). */
const toCandidate = (d, frame, kx, ky) => ({
    box: scaleBox(d.box, kx, ky),
    score: d.score,
    label: d.label,
    color: (d.rawBox ? dominantColorForBox(frame, d.rawBox)?.color : null) || null,
})

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

/** YOLOE baked-vocab candidates: detect everything at 640, keep boxes whose
 *  label matches the phrase's object, then the shared rank/color/collapse
 *  pipeline. Returns null when nothing matches → caller falls back to YOLO-World. */
export const detectCandidatesYoloe = async (phrase, { scale = 's', rankThreshold = 0.25, idleMs = 0, evict = false } = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null
    const plan = letterboxPlan(tf.originalW, tf.originalH, YOLOE_INPUT)
    const frame = await buildFrame(plan)
    if (!frame) return null
    const { dets, backend } = await detectTextYoloe(frame, { scale, threshold: 0.2, idleMs, evict })
    // Taxonomy-aware: a main-class phrase ("flower") matches its vocab-child
    // kinds ("rose", "tulip"), which YOLOE labels as separate classes.
    const matched = (dets || []).filter((d) => labelMatchesQuery(norm.objectCore, d.label))
    if (matched.length === 0) return null
    const mapped = []
    for (const d of matched) {
        const box = unletterboxBox(d.box, plan)
        if (box) mapped.push({ box, rawBox: d.box, score: d.score, label: d.label })
    }
    if (mapped.length === 0) return null
    const kx = tf.proxyW / tf.originalW
    const ky = tf.proxyH / tf.originalH
    const focused = focusRequestedColor(mapped, frame, norm.color)
    const ranked = rankDetections(focused, {
        threshold: norm.color && focused !== mapped ? 0 : rankThreshold,
        iou: 0.5, topK: 5, relative: 0.5,
    }).map((d) => toCandidate(d, frame, kx, ky))
    const candidates = norm.multi ? ranked : collapseToObject(ranked)
    return { candidates, multi: norm.multi, backend: `yoloe:${backend}`, display: norm.display }
}

/** YOLO-World open-vocab candidates: CLIP-encode the phrase + taxonomy synonyms
 *  into the 32 class slots, run the text-conditioned vision head, then the shared
 *  rank/colour/collapse pipeline. Handles ARBITRARY phrases (unlike the baked
 *  YOLOE lane), on WASM or WebGPU. Returns null when nothing matches. */
export const detectCandidatesYoloWorld = async (phrase, { scale = 's', rankThreshold = 0.1, idleMs = 0, evict = false } = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null
    const plan = letterboxPlan(tf.originalW, tf.originalH, YOLOE_INPUT)
    const frame = await buildFrame(plan)
    if (!frame) return null
    // Fill the class slots with the full phrase (CLIP reads "red flower"
    // directly) plus any taxonomy synonyms for the object.
    const expanded = expandQuery(norm.objectCore)
    const phrases = [...new Set([norm.core, ...(expanded ? expanded.labels : [norm.objectCore])])]
    // YOLO-World-v2 scores run lower than YOLOE's baked head; 0.08 is its
    // usable floor (ref impl uses ~0.05), the shared ranker tightens from there.
    const { dets, slotNames, backend } = await detectTextYoloWorld(frame, phrases, { scale, threshold: 0.08, idleMs, evict })
    if (!dets || dets.length === 0) return null
    const mapped = []
    for (const d of dets) {
        const box = unletterboxBox(d.box, plan)
        if (box) mapped.push({ box, rawBox: d.box, score: d.score, label: slotNames?.[d.classIdx] || norm.objectCore })
    }
    if (mapped.length === 0) return null
    const kx = tf.proxyW / tf.originalW
    const ky = tf.proxyH / tf.originalH
    const focused = focusRequestedColor(mapped, frame, norm.color)
    const ranked = rankDetections(focused, {
        threshold: norm.color && focused !== mapped ? 0 : rankThreshold,
        iou: 0.5, topK: 8, relative: 0.5,
    }).map((d) => toCandidate(d, frame, kx, ky))
    const candidates = norm.multi ? ranked : collapseToObject(ranked)
    return { candidates, multi: norm.multi, backend: `yoloworld:${backend}`, display: norm.display }
}

/**
 * text-core (pure) — phrase normalization + box math for text select.
 * Dependency-free so it unit-tests headless like sam-core.
 */

/** OWLv2's input tensor is a fixed 960². transformers.js has no pad-to-square
 *  step (config `size` is {960,960} with no shortest_edge, so its resize
 *  stretches), which hands the model an aspect-distorted frame. Letterbox into
 *  the square ourselves and that stretch becomes the aspect-preserving resize
 *  the model was trained on. Gray bottom/right padding is what the reference
 *  OWLv2 pads with (0.5 in rescaled space). */
export const DETECTOR_INPUT = 960
export const DETECTOR_PAD = 128

const ARTICLES = /^(a|an|the|some|all|every|any)\s+/i
const COUNT_INTENT = /^(all|every|both|each|multiple|several)\b/i
const NEVER_PLURAL = /(ss|us|is)$/i // grass, bus, iris — a trailing s that isn't a plural

const depluralize = (w) => (w.length > 3 && w.endsWith('s') && !NEVER_PLURAL.test(w) ? w.slice(0, -1) : w)

/** Raw phrase → detector label(s) + selection intent.
 *  "all zebras" → { labels:['a photo of a zebra'], multi:true }. The
 *  "a photo of a X" template matches OWL's training distribution. */
export const normalizePhrase = (raw) => {
    const clean = String(raw || '').trim().toLowerCase().replace(/\s+/g, ' ')
    if (!clean) return null
    let core = clean
    while (ARTICLES.test(core)) core = core.replace(ARTICLES, '')
    if (!core) core = clean
    const words = core.split(' ')
    const head = words[words.length - 1]
    words[words.length - 1] = depluralize(head)
    core = words.join(' ')
    return {
        core,
        labels: [`a photo of a ${core}`],
        multi: COUNT_INTENT.test(clean) || words[words.length - 1] !== head,
        display: clean,
    }
}

/** IoU of two [x0,y0,x1,y1] boxes. */
export const boxIoU = (a, b) => {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
    const inter = ix * iy
    const areaA = (a[2] - a[0]) * (a[3] - a[1])
    const areaB = (b[2] - b[0]) * (b[3] - b[1])
    const union = areaA + areaB - inter
    return union > 0 ? inter / union : 0
}

/** Greedy NMS over scored detections ([{box, score}]) → kept, high score first. */
export const nms = (dets, iouThresh = 0.5) => {
    const order = [...dets].sort((p, q) => q.score - p.score)
    const kept = []
    for (const d of order) {
        if (kept.every((k) => boxIoU(d.box, k.box) < iouThresh)) kept.push(d)
    }
    return kept
}

/** Scale a box between coordinate spaces (per-axis factor). */
export const scaleBox = (box, sx, sy) => [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]

/** Where a w×h frame sits inside the padded square: scaled by `k` to dw×dh at
 *  the top-left, gray everywhere else. */
export const letterboxPlan = (w, h, side = DETECTOR_INPUT) => {
    const k = side / Math.max(w, h)
    return { side, k, dw: Math.max(1, Math.round(w * k)), dh: Math.max(1, Math.round(h * k)) }
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v)

/** Square-normalized ([0,1]) box → source px, clipped to the frame. Null when
 *  the box lies mostly on the gray padding — the model firing on nothing. */
export const unletterboxBox = (box, plan, { minOnImage = 0.4 } = {}) => {
    const { side, k, dw, dh } = plan
    const [x0, y0, x1, y1] = box.map((v) => v * side)
    const area = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
    if (area <= 0) return null
    const cx0 = clamp(x0, 0, dw)
    const cy0 = clamp(y0, 0, dh)
    const cx1 = clamp(x1, 0, dw)
    const cy1 = clamp(y1, 0, dh)
    if (Math.max(0, cx1 - cx0) * Math.max(0, cy1 - cy0) < area * minOnImage) return null
    if (cx1 - cx0 < 1 || cy1 - cy0 < 1) return null
    return [cx0 / k, cy0 / k, cx1 / k, cy1 / k]
}

/** Threshold → NMS → top-K, high score first. `relative` also drops boxes far
 *  below the best match: OWLv2 emits low-scoring whole-frame guesses alongside
 *  real hits, and they outlive a fixed floor. */
export const rankDetections = (dets, { threshold = 0.15, iou = 0.5, topK = 8, relative = 0 } = {}) => {
    const above = dets.filter((d) => d.score >= threshold)
    const floor = Math.max(threshold, relative * above.reduce((m, d) => Math.max(m, d.score), 0))
    return nms(above.filter((d) => d.score >= floor), iou).slice(0, topK)
}

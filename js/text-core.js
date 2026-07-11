/**
 * text-core (pure) — phrase normalization + box math for text select.
 * Dependency-free so it unit-tests headless like sam-core.
 */

const ARTICLES = /^(a|an|the|some|all|every|any)\s+/i
const PLURAL_INTENT = /^(all|every|both|each)\b|s$/i

/** Raw phrase → detector label(s) + selection intent.
 *  "all zebras" → { labels:['a photo of a zebra'], multi:true }. The
 *  "a photo of a X" template matches OWL's training distribution. */
export const normalizePhrase = (raw) => {
    const clean = String(raw || '').trim().toLowerCase().replace(/\s+/g, ' ')
    if (!clean) return null
    const multi = PLURAL_INTENT.test(clean)
    let core = clean
    while (ARTICLES.test(core)) core = core.replace(ARTICLES, '')
    core = core.replace(/s$/i, (m, i, s) => (s.length > 3 ? '' : m)) // depluralize, keep short words
    if (!core) core = clean
    return { core, labels: [`a photo of a ${core}`], multi, display: clean }
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

/** Threshold → NMS → top-K, high score first. */
export const rankDetections = (dets, { threshold = 0.15, iou = 0.5, topK = 8 } = {}) =>
    nms(dets.filter((d) => d.score >= threshold), iou).slice(0, topK)

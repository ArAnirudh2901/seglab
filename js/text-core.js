/**
 * text-core (pure — no DOM, no transformers.js)
 * ------------------------------------------------
 * Query→prompt math for the text lane: turning a free-form phrase into the
 * concepts a detector can score, turning raw detections into the instances
 * worth decoding, and de-duplicating the masks that come back.
 *
 * Kept dependency-free so every decision here is unit-testable headless —
 * which matters, because these are the rules that decide whether "giraffe"
 * on a photo with no giraffe returns nothing (correct) or returns the
 * highest-scoring piece of background (the failure mode open-vocabulary
 * detectors are notorious for, and the one nobody tests).
 *
 * Instance mask representation: a detection's mask is stored as a
 * BBOX-CROPPED Uint8 plane plus its offset, never a full-frame RGBA buffer.
 * At DSLR scale a full-frame RGBA mask is ~96 MB *per instance*; "all the
 * birds" would be gigabytes. Cropped planes are typically well under 1 MB.
 */

/* ─── Query parsing ──────────────────────────────────────────────────────── */

// Leading determiners/quantifiers carry intent ("all") but hurt the text
// encoder, which was trained on bare noun phrases.
const QUANT_ALL = /^(all|every|each|both)\s+(of\s+)?(the\s+|these\s+|those\s+)?/i
const LEADING_ARTICLE = /^(the|a|an|some|any|this|that|these|those)\s+/i

/** Irregular plurals worth handling; the regular rules cover the long tail. */
const IRREGULAR = new Map(Object.entries({
    people: 'person', men: 'man', women: 'woman', children: 'child',
    teeth: 'tooth', feet: 'foot', geese: 'goose', mice: 'mouse',
    leaves: 'leaf', wolves: 'wolf', knives: 'knife', lives: 'life',
    sheep: 'sheep', fish: 'fish', deer: 'deer', aircraft: 'aircraft',
}))

/**
 * A phrase that is nothing but determiners carries no concept. Without this
 * check a stray "the" survives as a query and the detector cheerfully scores
 * the whole frame against it.
 */
const STOPWORDS = new Set([
    'the', 'a', 'an', 'some', 'any', 'this', 'that', 'these', 'those',
    'all', 'every', 'each', 'both', 'of', 'it', 'them', 'there', 'here',
])

/** Words that look plural but are not. */
const NOT_PLURAL = new Set([
    'glass', 'grass', 'bus', 'lens', 'dress', 'cross', 'class', 'gas',
    'iris', 'moss', 'chess', 'press', 'brass', 'compass', 'canvas', 'trellis',
])

/**
 * Singularize the head noun of a phrase. Detectors score bare singular
 * concepts best ("zebra"), while the plural in the user's words is an intent
 * signal ("all of them") handled separately by `wantsAll`.
 */
export const singularize = (word) => {
    const lower = word.toLowerCase()
    if (IRREGULAR.has(lower)) return IRREGULAR.get(lower)
    if (NOT_PLURAL.has(lower) || lower.length <= 2) return lower
    if (/[^aeiou]ies$/.test(lower)) return `${lower.slice(0, -3)}y`
    if (/(ch|sh|ss|x|z|s)es$/.test(lower)) return lower.slice(0, -2)
    if (/[^s]s$/.test(lower)) return lower.slice(0, -1)
    return lower
}

const isPlural = (word) => singularize(word) !== word.toLowerCase()

/**
 * Split a free-form query into the concepts to detect.
 *
 * "all the zebras, and a red car" → two concepts: {zebra, wantsAll:true} and
 * {red car, wantsAll:false}. Definite-singular phrasing ("the red bicycle")
 * means the user has ONE thing in mind, so the selector keeps only the best
 * match; plural or "all" phrasing keeps every confident instance. That
 * distinction is the difference between a clean selection and a scattershot.
 *
 * @param {string} text
 * @returns {Array<{ phrase: string, wantsAll: boolean, raw: string }>}
 */
export const parseQuery = (text) => {
    if (typeof text !== 'string') return []
    const parts = text
        .split(/\s*(?:,|;|\n|\band\b|\+)\s*/i)
        .map((s) => s.trim())
        .filter(Boolean)

    const seen = new Set()
    const out = []
    for (const raw of parts) {
        let s = raw.toLowerCase().replace(/\s+/g, ' ').trim()
        let wantsAll = false
        if (QUANT_ALL.test(s)) {
            wantsAll = true
            s = s.replace(QUANT_ALL, '')
        }
        s = s.replace(LEADING_ARTICLE, '').trim()
        s = s.replace(/[.!?]+$/, '').trim()
        if (!s) continue

        // Plural head noun ⇒ the user means all of them.
        const words = s.split(' ')
        const head = words[words.length - 1]
        if (isPlural(head)) {
            wantsAll = true
            words[words.length - 1] = singularize(head)
        }
        const phrase = words.join(' ')
        if (!phrase || seen.has(phrase)) continue
        if (words.every((wd) => STOPWORDS.has(wd))) continue
        seen.add(phrase)
        out.push({ phrase, wantsAll, raw })
    }
    return out
}

/* ─── Geometry ───────────────────────────────────────────────────────────── */

/** Intersection-over-union of two [x0,y0,x1,y1] boxes. */
export const boxIoU = (a, b) => {
    const ix0 = Math.max(a[0], b[0])
    const iy0 = Math.max(a[1], b[1])
    const ix1 = Math.min(a[2], b[2])
    const iy1 = Math.min(a[3], b[3])
    const iw = ix1 - ix0
    const ih = iy1 - iy0
    if (iw <= 0 || ih <= 0) return 0
    const inter = iw * ih
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1])
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])
    const union = areaA + areaB - inter
    return union > 0 ? inter / union : 0
}

/**
 * IoU of two bbox-cropped binary planes. Only the overlapping rectangle is
 * visited, so disjoint instances cost nothing — which is what makes
 * mask-level NMS affordable at DSLR scale.
 *
 * @param {{plane: Uint8Array, x0: number, y0: number, w: number, h: number, area: number}} a
 */
export const maskIoU = (a, b) => {
    const ix0 = Math.max(a.x0, b.x0)
    const iy0 = Math.max(a.y0, b.y0)
    const ix1 = Math.min(a.x0 + a.w, b.x0 + b.w)
    const iy1 = Math.min(a.y0 + a.h, b.y0 + b.h)
    if (ix1 <= ix0 || iy1 <= iy0) return 0
    let inter = 0
    for (let y = iy0; y < iy1; y += 1) {
        const ra = (y - a.y0) * a.w - a.x0
        const rb = (y - b.y0) * b.w - b.x0
        for (let x = ix0; x < ix1; x += 1) {
            if (a.plane[ra + x] && b.plane[rb + x]) inter += 1
        }
    }
    const union = a.area + b.area - inter
    return union > 0 ? inter / union : 0
}

/** Crop a white-on-black RGBA mask to its bbox as a compact 0/1 plane. */
export const cropMaskPlane = (rgba, w, h, bbox) => {
    if (!bbox) return null
    const [x0, y0, x1, y1] = bbox
    const pw = x1 - x0 + 1
    const ph = y1 - y0 + 1
    const plane = new Uint8Array(pw * ph)
    let area = 0
    for (let y = 0; y < ph; y += 1) {
        const src = (y0 + y) * w + x0
        const dst = y * pw
        for (let x = 0; x < pw; x += 1) {
            if (rgba[(src + x) * 4] >= 128) { plane[dst + x] = 1; area += 1 }
        }
    }
    return { plane, x0, y0, w: pw, h: ph, area }
}

/* ─── Detection selection ────────────────────────────────────────────────── */

/**
 * Greedy NMS over boxes, highest score first. Runs BEFORE mask decoding, so
 * the expensive decoder never sees near-duplicate proposals.
 */
export const boxNMS = (dets, iouThreshold = 0.55) => {
    const order = [...dets].sort((a, b) => b.score - a.score)
    const kept = []
    for (const d of order) {
        if (kept.every((k) => boxIoU(k.box, d.box) < iouThreshold)) kept.push(d)
    }
    return kept
}

/** Greedy NMS over decoded mask planes — the real overlap test. */
export const maskNMS = (instances, iouThreshold = 0.7) => {
    const order = [...instances].sort((a, b) => b.score - a.score)
    const kept = []
    for (const inst of order) {
        if (!inst.plane) continue
        if (kept.every((k) => maskIoU(k.plane, inst.plane) < iouThreshold)) kept.push(inst)
    }
    return kept
}

/**
 * Decide which raw detections are real, and how many the user asked for.
 *
 * Two thresholds, because one is never enough:
 *   - `floor` is an absolute confidence gate. If the BEST detection is below
 *     it, the concept is simply not in this photo and we return NOTHING.
 *     This is the absent-phrase case ("giraffe" on a beach) and returning an
 *     empty selection is the correct, and rarely implemented, answer.
 *   - `relative` keeps detections within a fraction of the top score. It
 *     adapts to the scene: twelve equally-confident zebras all survive, while
 *     one confident car plus background noise yields just the car.
 *
 * `wantsAll:false` (definite singular, "the red bicycle") collapses to the
 * single best instance regardless — the user named one thing.
 *
 * @param {Array<{score:number, box:number[]}>} dets
 * @returns {Array<{score:number, box:number[]}>}
 */
export const selectDetections = (dets, {
    wantsAll = true,
    floor = 0.12,
    relative = 0.55,
    maxInstances = 24,
    nmsIoU = 0.55,
} = {}) => {
    if (!Array.isArray(dets) || dets.length === 0) return []
    let top = 0
    for (const d of dets) if (d.score > top) top = d.score
    if (top < floor) return [] // concept absent — an empty selection is the answer

    const cut = Math.max(floor, top * relative)
    const survivors = boxNMS(dets.filter((d) => d.score >= cut), nmsIoU)
    survivors.sort((a, b) => b.score - a.score)
    return wantsAll ? survivors.slice(0, maxInstances) : survivors.slice(0, 1)
}

/**
 * Detector box → SAM prompt. The box alone underconstrains the decoder, so
 * the box centre rides along as a positive point (the same trick the click
 * lane already uses for box-only prompts in sam-engine).
 */
export const detectionToPrompt = (box) => ({
    box: [box[0], box[1], box[2], box[3]],
    clicks: [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1]],
})

/**
 * Is this detection small enough that the canonical downscale destroyed it?
 * Below the threshold the engine re-encodes a native-resolution crop instead
 * of trusting the 1024-frame mask — on a 45 MP file that is the difference
 * between a blob and an outline.
 */
export const needsZoomRefine = (box, frameW, frameH, fraction = 0.15) => {
    const diag = Math.hypot(box[2] - box[0], box[3] - box[1])
    return diag < fraction * Math.hypot(frameW, frameH)
}

/**
 * Padded, square-ish native-resolution crop around a detection, clamped to
 * the frame. `scale` maps canonical coordinates to source pixels.
 */
export const zoomCropRect = (box, frameW, frameH, { pad = 0.35, minSide = 64 } = {}) => {
    const cx = (box[0] + box[2]) / 2
    const cy = (box[1] + box[3]) / 2
    const side = Math.max(minSide, Math.max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * pad))
    let x0 = Math.round(cx - side / 2)
    let y0 = Math.round(cy - side / 2)
    let x1 = Math.round(cx + side / 2)
    let y1 = Math.round(cy + side / 2)
    if (x0 < 0) { x1 -= x0; x0 = 0 }
    if (y0 < 0) { y1 -= y0; y0 = 0 }
    if (x1 > frameW) { x0 = Math.max(0, x0 - (x1 - frameW)); x1 = frameW }
    if (y1 > frameH) { y0 = Math.max(0, y0 - (y1 - frameH)); y1 = frameH }
    return [x0, y0, x1, y1]
}

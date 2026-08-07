/**
 * select-core (pure — no DOM, no model)
 * ----------------------------------------
 * The unified selection layer. YOLOE-26 runs ONCE per image and produces one
 * set of instances (mask + label + score). Every mode the user has is then a
 * QUERY OVER THAT SET, not a separate model invocation:
 *
 *   click  → the instance under the point (smallest wins, so overlapping
 *            objects stay reachable)
 *   box    → the instances the region actually covers
 *   lasso  → same, against the polygon
 *   text   → the instances whose label matches the phrase
 *
 * That is what makes the workflow single and unified: one forward pass, then
 * selection is pure arithmetic on typed arrays — microseconds, no model in
 * the loop. Clicking around an already-analyzed photo never touches ONNX again.
 *
 * Instances are BBOX-CROPPED Uint8 planes ({plane, x0, y0, w, h, area}), never
 * full-frame buffers. At DSLR scale a full-frame mask is ~24 MB per instance;
 * "segment everything" on a 45 MP file would be gigabytes otherwise.
 */

import { boxIoU } from './yoloe-core.js'

export { boxIoU }

/* ─── Query parsing ──────────────────────────────────────────────────────── */

const QUANT_ALL = /^(all|every|each|both)\s+(of\s+)?(the\s+|these\s+|those\s+)?/i
const LEADING_ARTICLE = /^(the|a|an|some|any|this|that|these|those)\s+/i

const IRREGULAR = new Map(Object.entries({
    people: 'person', men: 'man', women: 'woman', children: 'child',
    teeth: 'tooth', feet: 'foot', geese: 'goose', mice: 'mouse',
    leaves: 'leaf', wolves: 'wolf', knives: 'knife', lives: 'life',
    sheep: 'sheep', fish: 'fish', deer: 'deer', aircraft: 'aircraft',
}))

/**
 * A phrase that is nothing but determiners carries no concept. Without this
 * check a stray "the" survives as a query and matches everything.
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

/** Singularize a head noun; vocabularies are indexed by singular concepts. */
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
 * Split a free-form query into concepts.
 *
 * Definite-singular phrasing ("the red bicycle") means ONE thing, so the
 * selector keeps only the best match; plural or "all …" phrasing keeps every
 * confident instance. That distinction is the difference between a clean
 * selection and a scattershot one.
 *
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
        if (QUANT_ALL.test(s)) { wantsAll = true; s = s.replace(QUANT_ALL, '') }
        s = s.replace(LEADING_ARTICLE, '').trim().replace(/[.!?]+$/, '').trim()
        if (!s) continue

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

/* ─── Mask geometry ──────────────────────────────────────────────────────── */

/** IoU of two bbox-cropped planes; only the overlapping rect is visited. */
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

/** Is (x, y) inside this instance's mask? */
export const maskContains = (inst, x, y) => {
    const px = Math.round(x) - inst.x0
    const py = Math.round(y) - inst.y0
    if (px < 0 || py < 0 || px >= inst.w || py >= inst.h) return false
    return inst.plane[py * inst.w + px] === 1
}

/** Fraction of an instance's mask lying inside a [x0,y0,x1,y1] region. */
export const maskFractionInBox = (inst, box) => {
    const ix0 = Math.max(inst.x0, Math.floor(box[0]))
    const iy0 = Math.max(inst.y0, Math.floor(box[1]))
    const ix1 = Math.min(inst.x0 + inst.w, Math.ceil(box[2]) + 1)
    const iy1 = Math.min(inst.y0 + inst.h, Math.ceil(box[3]) + 1)
    if (ix1 <= ix0 || iy1 <= iy0 || inst.area === 0) return 0
    let inside = 0
    for (let y = iy0; y < iy1; y += 1) {
        const row = (y - inst.y0) * inst.w - inst.x0
        for (let x = ix0; x < ix1; x += 1) if (inst.plane[row + x]) inside += 1
    }
    return inside / inst.area
}

/** Even-odd point-in-polygon. */
export const pointInPolygon = (poly, x, y) => {
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
        const [xi, yi] = poly[i]
        const [xj, yj] = poly[j]
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
    }
    return inside
}

/** Fraction of an instance's mask inside a polygon (sampled on its own grid). */
export const maskFractionInPolygon = (inst, poly) => {
    if (inst.area === 0) return 0
    let inside = 0
    for (let y = 0; y < inst.h; y += 1) {
        const row = y * inst.w
        for (let x = 0; x < inst.w; x += 1) {
            if (!inst.plane[row + x]) continue
            if (pointInPolygon(poly, inst.x0 + x, inst.y0 + y)) inside += 1
        }
    }
    return inside / inst.area
}

/** Greedy NMS over mask planes — the honest overlap test for instances. */
export const maskNMS = (instances, iouThreshold = 0.7) => {
    const order = [...instances].sort((a, b) => b.score - a.score)
    const kept = []
    for (const inst of order) {
        if (!inst.plane) continue
        if (kept.every((k) => maskIoU(k, inst) < iouThreshold)) kept.push(inst)
    }
    return kept
}

/* ─── The four selection queries ─────────────────────────────────────────── */

/**
 * Click → the instance under the point. When several overlap (a face inside a
 * person inside a crowd), the SMALLEST wins: it is the most specific thing the
 * user could have meant, and repeated clicks stay able to reach the others.
 */
export const pickByPoint = (instances, x, y) => {
    let best = null
    for (const inst of instances) {
        if (!maskContains(inst, x, y)) continue
        if (!best || inst.area < best.area) best = inst
    }
    return best
}

/**
 * Box or lasso → the instances the region genuinely covers.
 *
 * `minCoverage` is a fraction of the INSTANCE that must fall inside the
 * region, so dragging a loose box around one object does not also grab the
 * neighbour it clipped. With nothing over the bar, the single best-covered
 * instance is returned rather than an empty selection — a drawn region is an
 * explicit request for something.
 */
export const pickByRegion = (instances, region, { minCoverage = 0.55, wantsAll = true } = {}) => {
    const isPoly = Array.isArray(region) && Array.isArray(region[0])
    const scored = instances
        .map((inst) => ({
            inst,
            coverage: isPoly ? maskFractionInPolygon(inst, region) : maskFractionInBox(inst, region),
        }))
        .filter((s) => s.coverage > 0)
        .sort((a, b) => b.coverage - a.coverage || b.inst.score - a.inst.score)

    if (scored.length === 0) return []
    const over = scored.filter((s) => s.coverage >= minCoverage)
    if (over.length === 0) return [scored[0].inst]
    return wantsAll ? over.map((s) => s.inst) : [over[0].inst]
}

/**
 * Phrase → matching instances.
 *
 * `score` here is the detector's confidence for that label. Two gates, because
 * one is never enough:
 *   - `floor` is absolute. If the best match is below it, the concept is NOT
 *     in this photo and the answer is an EMPTY selection. Returning the
 *     highest-scoring piece of background instead is the failure mode
 *     open-vocabulary models are notorious for, and it is rarely tested.
 *   - `relative` keeps matches within a fraction of the best, so twelve
 *     equally-confident zebras all survive while one car plus noise yields
 *     just the car.
 */
export const selectByPhrase = (instances, phrase, {
    wantsAll = true, floor = 0.25, relative = 0.55, maxInstances = 50, labelOf = (i) => i.label,
} = {}) => {
    const want = singularize(String(phrase || '').trim().toLowerCase())
    if (!want) return []

    const matches = instances.filter((inst) => {
        const label = String(labelOf(inst) || '').toLowerCase()
        if (!label) return false
        if (label === want) return true
        // Vocabulary entries are often multi-word ("sports car", "traffic
        // light"); a phrase matches when it is one of those words or the
        // label is contained in the phrase.
        const words = label.split(/[\s_/-]+/).map(singularize)
        return words.includes(want) || want.split(' ').every((w) => words.includes(singularize(w)))
    })
    if (matches.length === 0) return []

    let top = 0
    for (const m of matches) if (m.score > top) top = m.score
    if (top < floor) return []

    const cut = Math.max(floor, top * relative)
    const kept = matches.filter((m) => m.score >= cut).sort((a, b) => b.score - a.score)
    return wantsAll ? kept.slice(0, maxInstances) : kept.slice(0, 1)
}

/**
 * Run a whole free-form query against one instance set. Multi-concept queries
 * ("all the zebras and a red car") union their results, then mask-level NMS
 * removes instances two concepts both claimed.
 */
export const selectByQuery = (instances, query, opts = {}) => {
    const concepts = parseQuery(query)
    if (concepts.length === 0) return { instances: [], concepts, reason: 'no concept in that phrase' }
    const hits = []
    for (const c of concepts) {
        for (const inst of selectByPhrase(instances, c.phrase, { ...opts, wantsAll: c.wantsAll })) {
            if (!hits.includes(inst)) hits.push(inst)
        }
    }
    const kept = maskNMS(hits, opts.maskNmsIoU ?? 0.85)
    return {
        instances: kept.sort((a, b) => b.score - a.score),
        concepts,
        reason: kept.length ? null : 'no confident match for that phrase in this photo',
    }
}

/* ─── DSLR helpers ───────────────────────────────────────────────────────── */

/**
 * Is this instance small enough that the 640 detection input destroyed its
 * detail? Below the threshold the engine re-runs a native-resolution crop —
 * on a 45 MP file that is the difference between a blob and an outline.
 */
export const needsZoomRefine = (box, frameW, frameH, fraction = 0.15) =>
    Math.hypot(box[2] - box[0], box[3] - box[1]) < fraction * Math.hypot(frameW, frameH)

/** Padded, square-ish native-res crop around a detection, clamped to frame. */
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

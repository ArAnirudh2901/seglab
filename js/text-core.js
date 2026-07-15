/**
 * text-core (pure) — phrase normalization + box math for text select.
 * Dependency-free so it unit-tests headless like sam-core.
 */

/** The portable OWLv2 lane has a fixed 960² tensor. transformers.js has no
 *  pad-to-square step (config `size` is {960,960} with no shortest_edge, so
 *  its resize stretches), which hands the model an aspect-distorted frame.
 *  Letterbox into that square ourselves; Grounding DINO receives the same
 *  bounded frame, keeping the coordinate contract and memory cap identical.
 *  Gray bottom/right padding is what the reference OWLv2 path expects
 *  (0.5 in rescaled space). */
export const DETECTOR_INPUT = 960
export const DETECTOR_PAD = 128

const ARTICLES = /^(a|an|the|some|all|every|any)\s+/i
const COUNT_INTENT = /^(all|every|both|each|multiple|several)\b/i
const NEVER_PLURAL = /(ss|us|is)$/i // grass, bus, iris — a trailing s that isn't a plural
const COLOR_WORDS = new Map([
    ['red', 'red'], ['orange', 'orange'], ['yellow', 'yellow'], ['green', 'green'],
    ['blue', 'blue'], ['purple', 'purple'], ['violet', 'purple'], ['pink', 'pink'],
    ['brown', 'brown'], ['white', 'white'], ['black', 'black'], ['grey', 'gray'], ['gray', 'gray'],
])
const COLOR_FILLERS = new Set(['color', 'colour', 'colored', 'coloured'])

const depluralize = (w) => (w.length > 3 && w.endsWith('s') && !NEVER_PLURAL.test(w) ? w.slice(0, -1) : w)

/** Raw phrase → detector label(s) + selection intent.
 * "all red zebras" sends both the exact phrase and the object-only fallback.
 * The latter finds boxes more reliably; a colour check ranks its results. */
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
    const color = words.map((word) => COLOR_WORDS.get(word)).find(Boolean) || null
    const objectCore = color
        ? words.filter((word) => !COLOR_WORDS.has(word) && !COLOR_FILLERS.has(word)).join(' ')
        : core
    const labels = [`a photo of a ${core}`]
    if (color && objectCore && objectCore !== core) labels.push(`a photo of a ${objectCore}`)
    return {
        core,
        objectCore: objectCore || core,
        color,
        labels,
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

const rgbToHsv = (r, g, b) => {
    const rr = r / 255; const gg = g / 255; const bb = b / 255
    const hi = Math.max(rr, gg, bb); const lo = Math.min(rr, gg, bb)
    const delta = hi - lo
    let h = 0
    if (delta) {
        if (hi === rr) h = 60 * (((gg - bb) / delta) % 6)
        else if (hi === gg) h = 60 * (((bb - rr) / delta) + 2)
        else h = 60 * (((rr - gg) / delta) + 4)
    }
    return { h: h < 0 ? h + 360 : h, s: hi ? delta / hi : 0, v: hi }
}

const hueDistance = (a, b) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b))

const isRequestedColor = (r, g, b, color) => {
    const { h, s, v } = rgbToHsv(r, g, b)
    if (color === 'white') return s <= 0.22 && v >= 0.65
    if (color === 'black') return v <= 0.2
    if (color === 'gray') return s <= 0.16 && v > 0.18 && v < 0.8
    const rules = {
        red: [0, 28, 0.36, 0.16, 1], orange: [26, 24, 0.38, 0.2, 1],
        yellow: [55, 30, 0.3, 0.24, 1], green: [122, 54, 0.28, 0.14, 1],
        blue: [218, 44, 0.28, 0.15, 1], purple: [280, 43, 0.25, 0.12, 1],
        pink: [338, 40, 0.22, 0.35, 1], brown: [26, 32, 0.25, 0.1, 0.68],
    }
    const rule = rules[color]
    return !!rule && hueDistance(h, rule[0]) <= rule[1] && s >= rule[2] && v >= rule[3] && v <= rule[4]
}

/** Fraction of sampled pixels inside a normalized candidate box that match a
 * requested basic colour. contentWidth/contentHeight exclude letterbox padding.
 * This is deliberately a ranking signal, not a second detector. */
export const colorEvidenceForBox = (frame, box, color, { maxSamples = 4096 } = {}) => {
    if (!frame?.data || !color) return 0
    const { data, width, height } = frame
    const usableW = Math.min(width, frame.contentWidth || width)
    const usableH = Math.min(height, frame.contentHeight || height)
    const x0 = clamp(Math.floor(box[0] * width), 0, usableW)
    const y0 = clamp(Math.floor(box[1] * height), 0, usableH)
    const x1 = clamp(Math.ceil(box[2] * width), 0, usableW)
    const y1 = clamp(Math.ceil(box[3] * height), 0, usableH)
    const area = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
    if (!area) return 0
    const step = Math.max(1, Math.ceil(Math.sqrt(area / maxSamples)))
    let matches = 0
    let sampled = 0
    for (let y = y0; y < y1; y += step) {
        for (let x = x0; x < x1; x += step) {
            const i = (y * width + x) * 3
            if (isRequestedColor(data[i], data[i + 1], data[i + 2], color)) matches += 1
            sampled += 1
        }
    }
    return sampled ? matches / sampled : 0
}

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
 *  below the best match: an open-vocabulary detector can emit low-scoring,
 *  scene-sized guesses alongside real hits, and they outlive a fixed floor. */
export const rankDetections = (dets, { threshold = 0.15, iou = 0.5, topK = 8, relative = 0 } = {}) => {
    const above = dets.filter((d) => d.score >= threshold)
    const floor = Math.max(threshold, relative * above.reduce((m, d) => Math.max(m, d.score), 0))
    return nms(above.filter((d) => d.score >= floor), iou).slice(0, topK)
}

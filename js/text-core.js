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
export const YOLOE_INPUT = 640 // YOLOE-26 fixed square input

const ARTICLES = /^(a|an|the|some|all|every|any)\s+/i
const COUNT_INTENT = /^(all|every|both|each|multiple|several)\b/i
const NEVER_PLURAL = /(ss|us|is)$/i // grass, bus, iris — a trailing s that isn't a plural
const COLOR_WORDS = new Map([
    ['red', 'red'], ['orange', 'orange'], ['yellow', 'yellow'], ['green', 'green'],
    ['blue', 'blue'], ['purple', 'purple'], ['violet', 'purple'], ['pink', 'pink'],
    ['brown', 'brown'], ['white', 'white'], ['black', 'black'], ['grey', 'gray'], ['gray', 'gray'],
])
const COLOR_FILLERS = new Set(['color', 'colour', 'colored', 'coloured'])

/**
 * Words that start a post-modifier, i.e. everything after them describes the
 * SETTING rather than the subject. The head noun is whatever sits in front.
 *
 * The normalizer below otherwise takes the LAST word as the head, which is right
 * for "orange tulip" and badly wrong for "the dog sitting among the flowers" —
 * it heads on "flowers", and the detector, which scores a phrase as a bag of
 * words, then happily boxes flowers for a photo with no dog in it. Measured on
 * the canonical NEF: that phrase returned 5 boxes and "snow covering the
 * flowers" returned 6, all of them flowers.
 *
 * "of" is deliberately NOT here: "a cluster of blue florets" would head on
 * "cluster" and lose the subject entirely.
 */
const POST_MODIFIER = new Set([
    'in', 'on', 'at', 'among', 'amongst', 'behind', 'near', 'nearby', 'under',
    'underneath', 'over', 'above', 'below', 'beneath', 'beside', 'between',
    'against', 'inside', 'outside', 'atop', 'around', 'across', 'along',
    'that', 'which', 'who',
    'sitting', 'standing', 'lying', 'covering', 'covered', 'holding', 'held',
    'wearing', 'worn', 'resting', 'hanging', 'growing', 'floating', 'leaning',
    'walking', 'running', 'playing', 'parked', 'placed', 'surrounded',
])

// -ves and mutated plurals that strip-the-s mangles ("leaves" → "leave").
const IRREGULAR_PLURALS = new Map([
    ['leaves', 'leaf'], ['wolves', 'wolf'], ['shelves', 'shelf'], ['halves', 'half'],
    ['loaves', 'loaf'], ['scarves', 'scarf'], ['calves', 'calf'], ['hooves', 'hoof'],
    ['thieves', 'thief'], ['elves', 'elf'], ['knives', 'knife'], ['wives', 'wife'],
    ['people', 'person'], ['children', 'child'], ['men', 'man'], ['women', 'woman'],
    ['mice', 'mouse'], ['geese', 'goose'], ['feet', 'foot'], ['teeth', 'tooth'],
])

const depluralize = (w) => IRREGULAR_PLURALS.get(w)
    || (w.length > 3 && w.endsWith('s') && !NEVER_PLURAL.test(w) ? w.slice(0, -1) : w)

/** Raw phrase → detector cores + selection intent. Bare nouns only: YOLOE's text
 *  tower was trained on class names, and a "a photo of a …" wrapper only dilutes
 *  them. `objectCore` drops any colour word; a colour check ranks its results. */
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
    // The subject, with any "… sitting among the flowers" setting cut off. Null
    // when the phrase has no post-modifier, so callers can tell "no subject to
    // check separately" from "the subject IS the whole phrase".
    const cut = (objectCore || core).split(' ')
    const at = cut.findIndex((w, i) => i > 0 && POST_MODIFIER.has(w))
    const headWords = at > 0 ? cut.slice(0, at) : null
    if (headWords) headWords[headWords.length - 1] = depluralize(headWords[headWords.length - 1])
    const headCore = headWords?.length ? headWords.join(' ') : null
    return {
        core,
        objectCore: objectCore || core,
        headCore: headCore && headCore !== (objectCore || core) ? headCore : null,
        color,
        multi: COUNT_INTENT.test(clean) || words[words.length - 1] !== head,
        display: clean,
    }
}

/** True when a baked-vocab `label` satisfies the phrase's object `core`
 *  (normalizePhrase.objectCore). Single-word core: whole-word match incl. plural
 *  forms. Multi-word core: substring. Powers the YOLOE lane's label filter. */
export const phraseMatchesLabel = (core, label) => {
    if (!core || !label) return false
    const l = String(label).toLowerCase()
    if (l === core) return true
    if (core.includes(' ')) return l.includes(core)
    const forms = new Set([core, `${core}s`, `${core}es`])
    return l.split(/\s+/).some((w) => forms.has(w))
}

/** True when a detector filled its top-k cap with a near-uniform score band —
 *  a collapsed alignment head (seen on q4f16 Grounding DINO on some GPUs).
 *  Scores carry no ranking signal, so the output is unusable regardless of
 *  the boxes. */
export const degenerateScores = (dets, { minCount = 32, minSpread = 0.08 } = {}) => {
    if (dets.length < minCount) return false
    let lo = Infinity
    let hi = -Infinity
    for (const d of dets) {
        if (d.score < lo) lo = d.score
        if (d.score > hi) hi = d.score
    }
    return hi - lo < minSpread
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

/** Bounding box of two [x0,y0,x1,y1] boxes. */
export const boxUnion = (a, b) => [
    Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3]),
]

/** Two boxes are pieces of one object when they overlap, or sit colinear with
 *  only a small gap — the shape a detector makes when it splits one long
 *  subject (a train's front and rear) into separate boxes. Gap is measured
 *  against the smaller box, so a small far object never joins a large one. */
const sameObject = (a, b, gap) => {
    const ox = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
    const oy = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
    if (ox > 0 && oy > 0) return true
    const w = Math.min(a[2] - a[0], b[2] - b[0])
    const h = Math.min(a[3] - a[1], b[3] - b[1])
    if (oy > 0.6 * h && ox <= 0 && -ox <= gap * w) return true // horizontal gap
    if (ox > 0.6 * w && oy <= 0 && -oy <= gap * h) return true // vertical gap
    return false
}

/** Group detector fragments into distinct objects. Fragments of one thing merge
 *  transitively into a single box; a second object that stays apart is its own
 *  cluster and is KEPT — several instances of the class the phrase named are all
 *  answers, not rivals for one slot. Best score first. */
export const clusterObjects = (dets, { gap = 1.5 } = {}) => {
    if (dets.length <= 1) return dets
    const order = [...dets].sort((p, q) => q.score - p.score)
    const n = order.length
    // Still-unclaimed detections as a linked list in score order. A growing box
    // has to be re-tested against what is left, but not against what it already
    // absorbed: the flag array made every sweep walk the full list and re-read
    // members it had taken on the previous one. Unlinking is O(1) and keeps the
    // order, so the head is always the best-scoring survivor — the seed rule.
    const next = new Int32Array(n)
    for (let i = 0; i < n - 1; i += 1) next[i] = i + 1
    next[n - 1] = -1
    let head = 0
    const out = []
    while (head >= 0) {
        const seed = head
        head = next[seed]
        let box = order[seed].box.slice()
        for (let grew = true; grew;) {
            grew = false
            let prev = -1
            for (let i = head; i >= 0;) {
                if (!sameObject(box, order[i].box, gap)) { prev = i; i = next[i]; continue }
                box = boxUnion(box, order[i].box)
                const skip = next[i]
                if (prev < 0) head = skip
                else next[prev] = skip
                grew = true
                i = skip
            }
        }
        out.push({ ...order[seed], box })
    }
    return out
}

/** Where a w×h frame sits inside the padded square: scaled by `k` to dw×dh at
 *  the top-left, gray everywhere else. */
export const letterboxPlan = (w, h, side = DETECTOR_INPUT) => {
    const k = side / Math.max(w, h)
    return { side, k, dw: Math.max(1, Math.round(w * k)), dh: Math.max(1, Math.round(h * k)) }
}

/** How far the full-frame pass shrinks the photo to reach the detector's square.
 *  A 45 MP DSLR frame at 640 is ~13x, so anything under ~200 px in the original
 *  lands below 16 px and is effectively invisible to the detector. */
export const shrinkFactor = (w, h, side = YOLOE_INPUT) => Math.max(w, h) / side

/**
 * Overlapping detector tiles, in ORIGINAL pixel coordinates.
 *
 * The detector input is a fixed square, so a big photo is squeezed into it and
 * small subjects fall below the resolution floor — the reason a dense field of
 * muscari scored 0.09 while a tulip in the same frame scored 0.60. Each tile is
 * letterboxed into its own square, multiplying linear resolution by `grid`.
 *
 * Tiles overlap by `overlap` of a step so a subject on a seam is whole in at
 * least one tile; the caller de-duplicates with NMS after mapping back.
 * Returns [{ ox, oy, ow, oh, plan }] where plan is the cell's own letterbox.
 */
/** Tile overlap as a fraction of a step. Shared with proxy-plan's detectorPlan,
 *  which derives the source resolution from the SMALLEST (corner) cell — the
 *  two must agree or the cap starves the tiles it is sizing for. */
export const TILE_OVERLAP = 0.15

export const tilePlans = (w, h, side = YOLOE_INPUT, { grid = 2, overlap = TILE_OVERLAP } = {}) => {
    const out = []
    const stepX = w / grid
    const stepY = h / grid
    const padX = stepX * overlap
    const padY = stepY * overlap
    for (let gy = 0; gy < grid; gy += 1) {
        for (let gx = 0; gx < grid; gx += 1) {
            const x0 = Math.max(0, gx * stepX - padX)
            const y0 = Math.max(0, gy * stepY - padY)
            const x1 = Math.min(w, (gx + 1) * stepX + padX)
            const y1 = Math.min(h, (gy + 1) * stepY + padY)
            const ow = x1 - x0
            const oh = y1 - y0
            if (ow < 1 || oh < 1) continue
            out.push({ ox: x0, oy: y0, ow, oh, plan: letterboxPlan(ow, oh, side) })
        }
    }
    return out
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

// [hueCenter, hueTol, minS, minV, maxV] per chromatic colour.
const COLOR_RULES = {
    red: [0, 28, 0.36, 0.16, 1], orange: [26, 24, 0.38, 0.2, 1],
    yellow: [55, 30, 0.3, 0.24, 1], green: [122, 54, 0.28, 0.14, 1],
    blue: [218, 44, 0.28, 0.15, 1], purple: [280, 43, 0.25, 0.12, 1],
    pink: [338, 40, 0.22, 0.35, 1], brown: [26, 32, 0.25, 0.1, 0.68],
}

const isRequestedColor = (r, g, b, color) => {
    const { h, s, v } = rgbToHsv(r, g, b)
    if (color === 'white') return s <= 0.22 && v >= 0.65
    if (color === 'black') return v <= 0.2
    if (color === 'gray') return s <= 0.16 && v > 0.18 && v < 0.8
    const rule = COLOR_RULES[color]
    return !!rule && hueDistance(h, rule[0]) <= rule[1] && s >= rule[2] && v >= rule[3] && v <= rule[4]
}

/** Winner-take-all colour bucket for one pixel, or null if it matches none.
 *  Achromatic (white/black/gray) checked first; otherwise the chromatic rule
 *  with the closest hue that also clears its s/v gates. Unlike isRequestedColor
 *  (a per-colour tolerance test for ranking) this assigns each pixel ONE name,
 *  so a box can be tallied into dominant-colour buckets in a single pass. */
export const classifyPixelColor = (r, g, b) => {
    const { h, s, v } = rgbToHsv(r, g, b)
    if (v <= 0.2) return 'black'
    if (s <= 0.16 && v >= 0.65) return 'white'
    if (s <= 0.16) return 'gray'
    let best = null
    let bestDist = Infinity
    for (const name in COLOR_RULES) {
        const [hc, tol, minS, minV, maxV] = COLOR_RULES[name]
        if (s < minS || v < minV || v > maxV) continue
        const d = hueDistance(h, hc)
        if (d <= tol && d < bestDist) { best = name; bestDist = d }
    }
    return best
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

/**
 * Colour-region proposals from raw pixel evidence — the "stuff" fallback.
 * Both detector lanes are OBJECT detectors (grounding-data heads): they cannot
 * box amorphous material ("green leaves", "blue sky", "brown soil") no matter
 * how good the text embedding is — measured: YOLO-World with a perfect 'leaf'
 * embedding proposes zero anchors on a leaf-filled garden. When a query names
 * a colour, the pixels themselves are the evidence: threshold the frame with
 * the requested colour's rule, connected-components on a strided grid, and the
 * big components become candidate boxes for the normal ranking pipeline.
 * Pure and model-free; boxes come back normalized [0,1] to the square, same
 * contract as the detector lanes. `minFrac` is the component floor as a
 * fraction of the CONTENT area (not the padded square).
 */
export const colorRegionsFromFrame = (frame, color, { stride = 2, minFrac = 0.004, topK = 8 } = {}) => {
    if (!frame?.data || !color) return []
    const { data, width, height } = frame
    const usableW = Math.min(width, frame.contentWidth || width)
    const usableH = Math.min(height, frame.contentHeight || height)
    const gw = Math.ceil(usableW / stride)
    const gh = Math.ceil(usableH / stride)
    if (gw < 4 || gh < 4) return []
    const grid = new Uint8Array(gw * gh)
    for (let gy = 0; gy < gh; gy += 1) {
        const y = gy * stride
        for (let gx = 0; gx < gw; gx += 1) {
            const i = (y * width + gx * stride) * 3
            if (isRequestedColor(data[i], data[i + 1], data[i + 2], color)) grid[gy * gw + gx] = 1
        }
    }
    // 4-connected components, iterative stack (no recursion on a 320² grid).
    const label = new Int32Array(gw * gh)
    const stack = []
    const regions = []
    const minCells = Math.max(8, Math.floor(gw * gh * minFrac))
    let nextLabel = 0
    for (let start = 0; start < grid.length; start += 1) {
        if (!grid[start] || label[start]) continue
        nextLabel += 1
        label[start] = nextLabel
        stack.length = 0
        stack.push(start)
        let cells = 0
        let x0 = gw; let x1 = 0; let y0 = gh; let y1 = 0
        while (stack.length) {
            const p = stack.pop()
            cells += 1
            const px = p % gw
            const py = (p / gw) | 0
            if (px < x0) x0 = px
            if (px > x1) x1 = px
            if (py < y0) y0 = py
            if (py > y1) y1 = py
            if (px > 0 && grid[p - 1] && !label[p - 1]) { label[p - 1] = nextLabel; stack.push(p - 1) }
            if (px < gw - 1 && grid[p + 1] && !label[p + 1]) { label[p + 1] = nextLabel; stack.push(p + 1) }
            if (py > 0 && grid[p - gw] && !label[p - gw]) { label[p - gw] = nextLabel; stack.push(p - gw) }
            if (py < gh - 1 && grid[p + gw] && !label[p + gw]) { label[p + gw] = nextLabel; stack.push(p + gw) }
        }
        if (cells < minCells) continue
        const bw = (x1 - x0 + 1)
        const bh = (y1 - y0 + 1)
        regions.push({
            // Grid box → pixel box → normalized to the SQUARE (detector contract).
            box: [(x0 * stride) / width, (y0 * stride) / height, ((x1 + 1) * stride) / width, ((y1 + 1) * stride) / height],
            frac: cells / (gw * gh),
            // Confidence = how solidly the box is filled with the colour (density)
            // damped toward the mid-range: these are proposals, not detections.
            score: Math.min(0.9, 0.3 + 0.6 * (cells / (bw * bh))),
        })
    }
    return regions.sort((a, b) => b.frac - a.frac).slice(0, topK)
}

/** Dominant colour bucket inside a normalized box → { color, frac } or null.
 *  Tallies each sampled pixel into a single bucket (classifyPixelColor) and
 *  returns the winner when it covers at least `minFrac`. Drives the colour
 *  sub-class facet: label a detection's box without a second detector. */
export const dominantColorForBox = (frame, box, { maxSamples = 4096, minFrac = 0.15 } = {}) => {
    if (!frame?.data) return null
    const { data, width, height } = frame
    const usableW = Math.min(width, frame.contentWidth || width)
    const usableH = Math.min(height, frame.contentHeight || height)
    const x0 = clamp(Math.floor(box[0] * width), 0, usableW)
    const y0 = clamp(Math.floor(box[1] * height), 0, usableH)
    const x1 = clamp(Math.ceil(box[2] * width), 0, usableW)
    const y1 = clamp(Math.ceil(box[3] * height), 0, usableH)
    const area = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
    if (!area) return null
    const step = Math.max(1, Math.ceil(Math.sqrt(area / maxSamples)))
    const tally = new Map()
    let sampled = 0
    for (let y = y0; y < y1; y += step) {
        for (let x = x0; x < x1; x += step) {
            const i = (y * width + x) * 3
            const c = classifyPixelColor(data[i], data[i + 1], data[i + 2])
            if (c) tally.set(c, (tally.get(c) || 0) + 1)
            sampled += 1
        }
    }
    if (!sampled) return null
    let color = null
    let top = 0
    for (const [c, n] of tally) if (n > top) { top = n; color = c }
    const frac = top / sampled
    return color && frac >= minFrac ? { color, frac } : null
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

/** Fraction of `inner`'s area covered by `outer`. */
const containment = (outer, inner) => {
    const ix = Math.max(0, Math.min(outer[2], inner[2]) - Math.max(outer[0], inner[0]))
    const iy = Math.max(0, Math.min(outer[3], inner[3]) - Math.max(outer[1], inner[1]))
    const area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return area > 0 ? (ix * iy) / area : 0
}

/** Which edges of `box` sit on an interior edge of the tile `cell` it came from
 *  — i.e. the subject was cut by the tile, not by its own silhouette. An edge
 *  that coincides with the IMAGE border is not clipping: nothing continues past
 *  it. `tol` is in original px; letterbox rounding lands boxes a pixel or two
 *  inside the cell, so an exact compare would miss every real clip. */
export const clippedEdges = (box, cell, bounds, tol = 4) => {
    const out = []
    const at = (v, edge) => Math.abs(v - edge) <= tol
    if (at(box[0], cell.ox) && cell.ox > tol) out.push('x0')
    if (at(box[1], cell.oy) && cell.oy > tol) out.push('y0')
    if (at(box[2], cell.ox + cell.ow) && cell.ox + cell.ow < bounds.w - tol) out.push('x1')
    if (at(box[3], cell.oy + cell.oh) && cell.oy + cell.oh < bounds.h - tol) out.push('y1')
    return out
}

/**
 * Drop tile fragments that duplicate a whole-object detection.
 *
 * A tiled pass sees each subject once per tile it falls in, and every one of
 * those views is cut off at the tile border — the box stops where the tile
 * stops, not where the object does. Such a box is truncated EVIDENCE, not a
 * detection, yet nothing distinguished it before: it carried a full score and
 * competed on equal terms.
 *
 * Measured, both from the real detector:
 *   · a streetcar spanning the seam — the full-frame pass boxed all of it (0.547)
 *     and a tile fragment cut at two edges scored HIGHER (0.604), so plain NMS
 *     deleted the complete box and kept the truncated one.
 *   · a tulip — the top-scoring detection in the whole frame (0.702) was the
 *     bottom half of the flower, cut exactly at the tile edge.
 *
 * Score cannot arbitrate this: a tight crop of half an object genuinely looks
 * more like the phrase than the whole object does. Provenance can. So a clipped
 * box that is mostly inside an unclipped one is the same object seen worse, and
 * loses regardless of score. A clipped box with no unclipped rival survives
 * untouched — that is the small-object case tiling exists for.
 */
export const dropClippedDuplicates = (dets, { cover = 0.7 } = {}) => {
    const whole = dets.filter((d) => !d.clipped?.length)
    if (!whole.length) return dets
    return dets.filter((d) => !d.clipped?.length
        || !whole.some((w) => w !== d && containment(w.box, d.box) >= cover))
}

/** Drop group boxes. Shown several instances, a detector also emits a box
 *  around the whole cluster; it survives NMS (low IoU against each member) but
 *  selects far more than the phrase asked for. A box that mostly contains 2+
 *  other kept detections is the cluster, not an instance — keep the members. */
export const pruneContainers = (dets, { cover = 0.7 } = {}) => {
    const n = dets.length
    if (n < 3) return dets            // a cluster box needs 2 members to be one
    // Areas up front, then stop at 2: the old form built a whole array of
    // matches per detection just to read its length, so it did every pairwise
    // containment even after the answer was settled. `cover` also bounds the
    // pair geometrically — an inner box bigger than outer/cover cannot be
    // covered — which rejects most pairs on a subtraction.
    const area = new Float64Array(n)
    for (let i = 0; i < n; i += 1) {
        const b = dets[i].box
        area[i] = (b[2] - b[0]) * (b[3] - b[1])
    }
    return dets.filter((d, i) => {
        const cap = area[i] / cover     // no member bigger than this can be covered
        const o = d.box
        let held = 0
        for (let j = 0; j < n; j += 1) {
            if (area[j] > cap || j === i) continue
            const m = dets[j].box
            const ix = Math.min(o[2], m[2]) - Math.max(o[0], m[0])
            if (ix <= 0) continue
            const iy = Math.min(o[3], m[3]) - Math.max(o[1], m[1])
            if (iy <= 0) continue
            if (ix * iy >= cover * area[j] && (held += 1) === 2) return false
        }
        return true
    })
}

/**
 * Keep only what the phrase's SUBJECT matched; null when it matched nothing.
 *
 * The detector scores a phrase as a bag of words, so the setting keeps scoring
 * on its own: "the dog sitting among the flowers" boxed 5 flowers in a photo
 * with no dog. Answering "nothing" there is the gate. The other half is that a
 * photo WITH a dog returned the dog AND the flowers — the setting exists to
 * condition the score, and was never something the user asked to select.
 *
 * `subject` is null for a phrase with no post-modifier ("orange tulip"), where
 * the subject IS the whole phrase and there is nothing to separate.
 */
export const filterToSubject = (dets, subject) => {
    if (!subject) return dets
    const hits = dets.filter((d) => subject.has(d.label))
    return hits.length ? hits : null
}

/** Threshold → NMS → container prune → top-K, high score first. `relative`
 *  also drops boxes far below the best match: an open-vocabulary detector can
 *  emit low-scoring, scene-sized guesses alongside real hits, and they outlive
 *  a fixed floor. */
export const rankDetections = (dets, { threshold = 0.15, iou = 0.5, topK = 8, relative = 0 } = {}) => {
    const above = dets.filter((d) => d.score >= threshold)
    const floor = Math.max(threshold, relative * above.reduce((m, d) => Math.max(m, d.score), 0))
    return pruneContainers(nms(above.filter((d) => d.score >= floor), iou)).slice(0, topK)
}

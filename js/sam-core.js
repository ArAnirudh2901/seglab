/**
 * sam-core (pure — no DOM, no model runtime)
 * ----------------------------------------------
 * Prompt/mask math for the on-device segmentation engine: picking the best of
 * SAM's three candidate masks, mask channel composition, and mapping prompts
 * into crop space. Dependency-free so it can be unit-tested headless.
 *
 * Prompts reach the decoder in proxy space; sam21-lane owns the scaling into
 * model-input space, so nothing here needs the reshaped [h, w] pair.
 */

/**
 * Under an ambiguous prompt, a candidate covering ≥ this much of the frame is
 * SAM's whole-scene guess, not the clicked object. Whole-scene runaways reach
 * ~85%, so 0.9 let them through; 0.75 catches them. Only overrides when a
 * smaller candidate exists — a genuinely frame-filling subject has no smaller
 * granularity level to fall to, so its argmax winner still stands.
 */
export const RUNAWAY_COVERAGE = 0.75

/**
 * Index of the best of SAM's three candidate masks.
 *
 * SAM emits its candidates as granularity levels (subpart → part → whole) and
 * scores each one independently, so a plain argmax is only trustworthy when
 * the prompt says which level was meant. Under an ambiguous prompt — a single
 * positive click, no box — clicking one object in a field of near-identical
 * objects lets the whole-scene candidate outscore the object actually pointed
 * at, and the argmax winner comes back as the entire photo. In that case take
 * the best-scoring candidate that still leaves a background behind. If every
 * candidate is a runaway the subject may genuinely fill the frame, so the
 * argmax winner stands.
 */
export const pickBestMask = (scores, { coverages = null, ambiguous = false } = {}) => {
    const byScore = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a])
    if (!ambiguous || !coverages) return byScore[0]
    // An empty candidate is a miss, never an improvement on a runaway.
    const scoped = byScore.find((i) => coverages[i] > 0 && coverages[i] < RUNAWAY_COVERAGE)
    return scoped === undefined ? byScore[0] : scoped
}

/**
 * Fraction of each candidate channel that is selected, sharing the one pass
 * over the bool mask tensor ([1, C, H, W], Uint8 0/1) that every candidate
 * lives in. Feeds pickBestMask's runaway test before a channel is expanded
 * to RGBA — only the winner ever pays for that.
 */
export const maskChannelCoverages = (maskData, width, height, channels) => {
    const size = width * height
    const out = new Array(channels)
    for (let c = 0; c < channels; c += 1) {
        const offset = c * size
        let count = 0
        for (let i = 0; i < size; i += 1) if (maskData[offset + i]) count += 1
        out[c] = count / size
    }
    return out
}

/** Coverage + bbox of a white-on-black RGBA mask (reads the R channel). */
export const summarizeMaskRGBA = (rgba, width, height) => {
    let count = 0
    let minX = width
    let minY = height
    let maxX = -1
    let maxY = -1
    for (let y = 0; y < height; y += 1) {
        const row = y * width
        for (let x = 0; x < width; x += 1) {
            if (rgba[(row + x) * 4] >= 128) {
                count += 1
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }
    return {
        coverage: count / (width * height),
        bbox: maxX >= 0 ? [minX, minY, maxX, maxY] : null,
    }
}

/* ─── Committed-selection composition (click-union model) ─────────────────── */

/** 1-channel copy of a white-on-black RGBA mask. Keeps the full 8-bit value —
 *  the refined boundary band is soft alpha, and committing an object must not
 *  harden its edges. */
export const maskToChannel = (imageData) => {
    const { data, width, height } = imageData
    const chan = new Uint8Array(width * height)
    for (let i = 0; i < chan.length; i += 1) chan[i] = data[i * 4]
    return chan
}

/**
 * Replay an ordered op stack ({op:'add'|'sub', chan}) into a white-on-black
 * RGBA mask. Adds union per-pixel max (soft edges survive); subs zero where
 * the sub channel is selected. `floor` is an optional pre-flattened starting
 * channel. Returns null when nothing selected (callers keep the fast path).
 */
export const composeChannels = (ops, width, height, floor = null) => {
    const size = width * height
    const acc = floor ? Uint8Array.from(floor) : new Uint8Array(size)
    for (const { op, chan } of ops) {
        if (!chan || chan.length !== size) continue
        if (op === 'sub') { for (let i = 0; i < size; i += 1) if (chan[i] >= 128) acc[i] = 0 }
        else { for (let i = 0; i < size; i += 1) if (chan[i] > acc[i]) acc[i] = chan[i] }
    }
    let any = false
    const rgba = new Uint8ClampedArray(size * 4)
    for (let i = 0; i < size; i += 1) {
        const v = acc[i]
        if (v >= 128) any = true
        const j = i * 4
        rgba[j] = v; rgba[j + 1] = v; rgba[j + 2] = v; rgba[j + 3] = 255
    }
    return any ? { rgba, width, height } : null
}

/**
 * Dilate a 1-channel mask by `radius` px (chebyshev). Subtract ops grow by a
 * safety margin so removing an object never leaves a boundary-residue ring
 * where two decodes of the same object disagree by a pixel.
 */
export const dilateChannel = (chan, width, height, radius = 2) => {
    if (!radius) return chan
    const out = new Uint8Array(chan.length)
    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            if (!chan[y * width + x]) continue
            const x0 = Math.max(0, x - radius)
            const x1 = Math.min(width - 1, x + radius)
            const y0 = Math.max(0, y - radius)
            const y1 = Math.min(height - 1, y + radius)
            for (let yy = y0; yy <= y1; yy += 1) out.fill(255, yy * width + x0, yy * width + x1 + 1)
        }
    }
    return out
}

/** True when (x, y) — or any pixel within `tolerance` px — is selected. */
export const pointInMask = (imageData, x, y, tolerance = 3) => {
    if (!imageData) return false
    const { data, width, height } = imageData
    const x0 = Math.max(0, Math.round(x) - tolerance)
    const x1 = Math.min(width - 1, Math.round(x) + tolerance)
    const y0 = Math.max(0, Math.round(y) - tolerance)
    const y1 = Math.min(height - 1, Math.round(y) + tolerance)
    for (let yy = y0; yy <= y1; yy += 1) {
        for (let xx = x0; xx <= x1; xx += 1) {
            if (data[(yy * width + xx) * 4] >= 128) return true
        }
    }
    return false
}

/**
 * Accept/reject a selection mask. Empty means the prompts didn't land on
 * anything (a miss, not a crash); near-solid means the decoder failed to
 * separate the object (a real subject never fills the frame to within
 * 0.1%). Anything between — including very small masks — is legitimate:
 * minute-object selection is a first-class use case.
 */
export const validateClickMask = ({ coverage, bbox }) => {
    if (!bbox || coverage <= 0) return { usable: false, reason: 'empty mask (selection missed)' }
    if (coverage >= 0.9995) return { usable: false, reason: 'solid mask (object not separated)' }
    return { usable: true, reason: null }
}

/* ─── Mask hygiene ───────────────────────────────────────────────────────── */

/** 4-connected component labeling on a binary Uint8 map. Iterative stack
 *  flood fill (no recursion — 1024² masks would blow the call stack).
 *  Returns { labels: Int32Array (0 = background, 1..n), areas: number[] }
 *  where areas[k] is the pixel count of component k+1. */
const labelComponents = (bin, w, h) => {
    const labels = new Int32Array(w * h)
    const areas = []
    const stack = new Int32Array(w * h)
    let next = 0
    for (let start = 0; start < bin.length; start += 1) {
        if (!bin[start] || labels[start]) continue
        next += 1
        let area = 0
        let top = 0
        stack[top++] = start
        labels[start] = next
        while (top > 0) {
            const i = stack[--top]
            area += 1
            const x = i % w
            if (x > 0 && bin[i - 1] && !labels[i - 1]) { labels[i - 1] = next; stack[top++] = i - 1 }
            if (x < w - 1 && bin[i + 1] && !labels[i + 1]) { labels[i + 1] = next; stack[top++] = i + 1 }
            if (i >= w && bin[i - w] && !labels[i - w]) { labels[i - w] = next; stack[top++] = i - w }
            if (i < w * (h - 1) && bin[i + w] && !labels[i + w]) { labels[i + w] = next; stack[top++] = i + w }
        }
        areas.push(area)
    }
    return { labels, areas }
}

/** Component count of a mask (verify/debug hook). */
export const countMaskComponents = (rgba, w, h) => {
    const bin = new Uint8Array(w * h)
    for (let i = 0; i < bin.length; i += 1) bin[i] = rgba[i * 4] >= 128 ? 1 : 0
    return labelComponents(bin, w, h).areas.length
}

/* ─── Crop-space helpers (M1 crop pyramid / HD export) ──────────────────── */

/**
 * Map proxy-space prompts into a crop's own pixel space. `proxyToOriginal`
 * scales proxy → original coords; `rect` is the crop's origin+size in
 * original coords. Points/boxes are clamped into the crop so a prompt a few
 * pixels outside (padding round-off) still lands.
 *
 * @param {{ clicks?: Array<[number,number,0|1]>, box?: number[]|null,
 *           clampPoly?: Array<[number,number]>|null, clampMargin?: number }} prompts
 * @param {number} proxyToOriginal  originalW / proxyW
 * @param {{ x:number, y:number, w:number, h:number }} rect
 */
export const mapPromptsToCrop = (prompts, proxyToOriginal, rect) => {
    const px = (v) => Math.min(rect.w, Math.max(0, v * proxyToOriginal - rect.x))
    const py = (v) => Math.min(rect.h, Math.max(0, v * proxyToOriginal - rect.y))
    const clicks = (prompts.clicks || []).map(([x, y, label]) => [px(x), py(y), label])
    const box = Array.isArray(prompts.box) && prompts.box.length === 4
        ? [px(prompts.box[0]), py(prompts.box[1]), px(prompts.box[2]), py(prompts.box[3])]
        : null
    const clampPoly = Array.isArray(prompts.clampPoly) && prompts.clampPoly.length >= 3
        ? prompts.clampPoly.map(([x, y]) => [px(x), py(y)])
        : null
    return {
        clicks,
        box,
        clampPoly,
        clampMargin: (prompts.clampMargin || 0) * proxyToOriginal,
    }
}

/**
 * Lasso stroke → SAM prompts. The lasso is a PROMPT GENERATOR, not a
 * geometric cut: its bounding box becomes the box prompt and its centroid a
 * positive point, then the decoder snaps to the true object boundary inside.
 * The polygon itself is kept so the caller can clamp the result to
 * lasso ∪ margin (the "can never bleed onto the second zebra" guarantee).
 *
 * @param {Array<[number, number]>} poly  closed freehand stroke, source space
 * @returns {{ box: number[], point: [number, number, 1], margin: number } | null}
 */
export const lassoToPrompts = (poly) => {
    if (!Array.isArray(poly) || poly.length < 3) return null
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    // Polygon centroid (shoelace) — falls back to bbox centre for degenerate
    // (near-zero-area) scribbles.
    let areaSum = 0
    let cx = 0
    let cy = 0
    for (let i = 0; i < poly.length; i += 1) {
        const [x, y] = poly[i]
        const [nx, ny] = poly[(i + 1) % poly.length]
        const cross = x * ny - nx * y
        areaSum += cross
        cx += (x + nx) * cross
        cy += (y + ny) * cross
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
    }
    if (!(maxX > minX) || !(maxY > minY)) return null
    const area = areaSum / 2
    let centre
    if (Math.abs(area) > 1e-3) {
        centre = [cx / (6 * area), cy / (6 * area)]
    } else {
        centre = [(minX + maxX) / 2, (minY + maxY) / 2]
    }
    // A concave stroke can put the centroid outside the polygon; the bbox
    // centre is no better in general, but SAM tolerates near-boundary
    // anchors — clamp into the bbox to keep the prompt sane.
    centre = [
        Math.min(maxX, Math.max(minX, centre[0])),
        Math.min(maxY, Math.max(minY, centre[1])),
    ]
    const diag = Math.hypot(maxX - minX, maxY - minY)
    return {
        box: [minX, minY, maxX, maxY],
        point: [centre[0], centre[1], 1],
        // Clamp margin: forgiving of a sloppy stroke, but tight enough that
        // a neighbouring object outside the lasso stays out.
        margin: Math.max(8, diag * 0.04),
    }
}

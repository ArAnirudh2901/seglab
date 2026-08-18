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

/**
 * Coverage + bbox of a white-on-black RGBA mask (reads the R channel).
 *
 * `rect` (ends exclusive) restricts the scan. It is a promise, not a crop: the
 * caller is asserting no selected pixel lies outside it, so the coverage and
 * bbox that come back are the WHOLE mask's. Pass a rect only when something
 * already known bounds the mask — its own previous bbox, or the union of the
 * pieces it was composed from.
 */
export const summarizeMaskRGBA = (rgba, width, height, rect = null) => {
    let count = 0
    let minX = width
    let minY = height
    let maxX = -1
    let maxY = -1
    const [rx0, ry0, rx1, ry1] = rect
        ? [Math.max(0, rect[0]), Math.max(0, rect[1]), Math.min(width, rect[2]), Math.min(height, rect[3])]
        : [0, 0, width, height]
    // The soft extent is every non-zero pixel, which reaches past the >=128 core
    // by the width of the matted edge. Anything that COMPOSITES the mask needs
    // this one; anything that measures selected area needs the core.
    let sx0 = width; let sy0 = height; let sx1 = -1; let sy1 = -1
    for (let y = ry0; y < ry1; y += 1) {
        const row = y * width
        for (let x = rx0; x < rx1; x += 1) {
            const v = rgba[(row + x) * 4]
            if (!v) continue
            if (x < sx0) sx0 = x
            if (x > sx1) sx1 = x
            if (y < sy0) sy0 = y
            sy1 = y
            if (v >= 128) {
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
        // Ends exclusive — a scan window, not an inclusive bbox.
        softBox: sx1 < 0 ? null : [sx0, sy0, sx1 + 1, sy1 + 1],
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

// Grey level as one opaque RGBA word, built through a byte view so the packing
// follows the platform's endianness instead of assuming little-endian.
const packScratch = new Uint8Array(4)
const packWord = new Uint32Array(packScratch.buffer)
const opaque = (v) => {
    packScratch[0] = v; packScratch[1] = v; packScratch[2] = v; packScratch[3] = 255
    return packWord[0]
}

/**
 * Replay an ordered op stack ({op:'add'|'sub', chan, bounds}) into a
 * white-on-black RGBA mask. Adds union per-pixel max (soft edges survive); subs
 * zero where the sub channel is selected. `floor` is an optional pre-flattened
 * starting channel and `floorBounds` its extent. Returns null when nothing is
 * selected (callers keep the fast path).
 *
 * `bounds`/`floorBounds` are ends-exclusive rects from channelBounds. They are
 * an optimisation only: an op without one is replayed over the whole frame, so
 * a stack restored from an older session still composes correctly.
 */
export const composeChannels = (ops, width, height, floor = null, floorBounds = null) => {
    const size = width * height
    // slice(), not Uint8Array.from(): from() walks the iterator protocol a byte
    // at a time where slice() is a memcpy of the whole buffer.
    const acc = floor ? floor.slice() : new Uint8Array(size)
    // Every op knows where it lives, so replaying the stack costs the sum of the
    // objects' areas rather than (stack depth x frame). A missing bound means an
    // op from before this was tracked (a restored session) — scan it whole.
    let ux0 = width; let uy0 = height; let ux1 = 0; let uy1 = 0
    const widen = (b) => {
        if (!b) { ux0 = 0; uy0 = 0; ux1 = width; uy1 = height; return }
        if (b[0] < ux0) ux0 = b[0]
        if (b[1] < uy0) uy0 = b[1]
        if (b[2] > ux1) ux1 = b[2]
        if (b[3] > uy1) uy1 = b[3]
    }
    if (floor) widen(floorBounds)
    for (const { op, chan, bounds } of ops) {
        if (!chan || chan.length !== size) continue
        const [x0, y0, x1, y1] = bounds && bounds[2] <= width && bounds[3] <= height
            ? bounds : [0, 0, width, height]
        // A subtract only ever clears, so it cannot put the union anywhere new.
        if (op !== 'sub') widen(bounds)
        for (let y = y0; y < y1; y += 1) {
            const row = y * width
            if (op === 'sub') { for (let x = x0; x < x1; x += 1) if (chan[row + x] >= 128) acc[row + x] = 0 }
            else for (let x = x0; x < x1; x += 1) if (chan[row + x] > acc[row + x]) acc[row + x] = chan[row + x]
        }
    }
    if (ux1 <= ux0 || uy1 <= uy0) return null

    // Grey-on-opaque: alpha is 255 for the whole frame, so the background cannot
    // be left as the allocator's zeros. One 32-bit fill covers it, and only the
    // union rect is then written per pixel.
    const rgba = new Uint8ClampedArray(size * 4)
    const words = new Uint32Array(rgba.buffer)
    words.fill(opaque(0))
    let any = false
    for (let y = uy0; y < uy1; y += 1) {
        const row = y * width
        for (let x = ux0; x < ux1; x += 1) {
            const v = acc[row + x]
            if (!v) continue
            if (v >= 128) any = true
            words[row + x] = opaque(v)
        }
    }
    // The union rect bounds every non-zero pixel, so callers can hand it back as
    // the scan window for anything derived from this mask.
    return any ? { rgba, width, height, bounds: [ux0, uy0, ux1, uy1] } : null
}


/** Bounding box (ends exclusive) of the selected cells of a mask channel, or
 *  null when nothing is selected. One pass, so op stacks can carry it. */
export const channelBounds = (chan, width, height) => {
    let minX = width; let minY = height; let maxX = -1; let maxY = -1
    for (let y = 0; y < height; y += 1) {
        const row = y * width
        for (let x = 0; x < width; x += 1) {
            if (!chan[row + x]) continue
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            maxY = y
        }
    }
    return maxX < 0 ? null : [minX, minY, maxX + 1, maxY + 1]
}

/**
 * One axis of a chebyshev dilation: two sweeps carrying the distance since the
 * last set cell. Cost is one read/write per cell and does not depend on the
 * radius, which is what keeps the mask-wide morphology linear.
 */
const spread = (src, dst, n, step, base, r) => {
    let d = r + 1
    for (let i = 0; i < n; i += 1) {
        const p = base + i * step
        d = src[p] ? 0 : d + 1
        dst[p] = d <= r ? 1 : 0
    }
    d = r + 1
    for (let i = n - 1; i >= 0; i -= 1) {
        const p = base + i * step
        d = src[p] ? 0 : d + 1
        if (d <= r) dst[p] = 1
    }
}

/** Chebyshev dilation of a binary field, separably: along x, then along y. */
const dilateBinary = (src, dst, tmp, width, r, [x0, y0, x1, y1]) => {
    for (let y = y0; y < y1; y += 1) spread(src, tmp, x1 - x0, 1, y * width + x0, r)
    for (let x = x0; x < x1; x += 1) spread(tmp, dst, y1 - y0, width, y0 * width + x, r)
}

/**
 * Dilate a 1-channel mask by `radius` px (chebyshev). Subtract ops grow by a
 * safety margin so removing an object never leaves a boundary-residue ring
 * where two decodes of the same object disagree by a pixel.
 */
export const dilateChannel = (chan, width, height, radius = 2) => {
    if (!radius) return chan
    const n = chan.length
    const bin = new Uint8Array(n)
    for (let i = 0; i < n; i += 1) bin[i] = chan[i] ? 1 : 0
    const tmp = new Uint8Array(n)
    const out = new Uint8Array(n)
    dilateBinary(bin, out, tmp, width, radius, [0, 0, width, height])
    for (let i = 0; i < n; i += 1) out[i] = out[i] ? 255 : 0
    return out
}

/* ─── Boundary regularisation (the composed selection) ────────────────────── */

// Everything below works inside the selection's bounding box grown by the
// kernel. Nothing outside it can change, so a small object in a large frame
// costs its own area instead of the frame's. `rect` is an INCLUSIVE bbox, the
// shape summarizeMaskRGBA returns; the result is [x0, y0, x1, y1) exclusive.
const workRect = (rect, width, height, margin) => (rect
    ? [Math.max(0, Math.floor(rect[0]) - margin), Math.max(0, Math.floor(rect[1]) - margin),
        Math.min(width, Math.floor(rect[2]) + 1 + margin), Math.min(height, Math.floor(rect[3]) + 1 + margin)]
    : [0, 0, width, height])

/**
 * Close hairline gaps in the composed selection — a morphological closing of
 * the ≥128 core, written back at full value.
 *
 * Two objects selected one after the other each stop a pixel short of the edge
 * they share, so their union keeps a slit of background along it. The outline
 * is a dilation of the core, so a two-pixel slit gets drawn as a border THROUGH
 * the middle of what the user selected as one thing. A closing joins gaps
 * narrower than 2r and leaves every other boundary exactly where it was.
 *
 * Add-only: it can never drop a selected pixel, so no object is lost to it.
 * Outside the frame reads as foreground for the erosion, so a slit that runs
 * off the frame edge closes to the edge instead of leaving a notch there.
 * Returns the number of pixels filled.
 */
export const bridgeGaps = (rgba, width, height, { radius = 2, rect = null } = {}) => {
    const r = Math.max(1, Math.round(radius))
    const [x0, y0, x1, y1] = workRect(rect, width, height, r + 1)
    if (x1 <= x0 || y1 <= y0) return 0
    const n = width * height
    const core = new Uint8Array(n)
    for (let y = y0; y < y1; y += 1) {
        const row = y * width
        for (let x = x0; x < x1; x += 1) core[row + x] = rgba[(row + x) * 4] >= 128 ? 1 : 0
    }
    const tmp = new Uint8Array(n)
    const dil = new Uint8Array(n)
    dilateBinary(core, dil, tmp, width, r, [x0, y0, x1, y1])
    // Erosion is the complement of the dilation of the complement, so one
    // primitive covers both halves of the closing.
    for (let y = y0; y < y1; y += 1) {
        const row = y * width
        for (let x = x0; x < x1; x += 1) dil[row + x] = dil[row + x] ? 0 : 1
    }
    const back = new Uint8Array(n)
    dilateBinary(dil, back, tmp, width, r, [x0, y0, x1, y1])
    let filled = 0
    for (let y = y0; y < y1; y += 1) {
        const row = y * width
        for (let x = x0; x < x1; x += 1) {
            const p = row + x
            if (back[p] || core[p]) continue
            const j = p * 4
            rgba[j] = 255; rgba[j + 1] = 255; rgba[j + 2] = 255; rgba[j + 3] = 255
            filled += 1
        }
    }
    return filled
}

// Whether clearing a cell keeps the shape connected: with its 8 neighbours read
// as a ring, one 0→1 transition means they form a single arc, so the cell is on
// a boundary. Two or more means it is the LINK between separate parts — a 1 px
// wire is exactly that — and clearing it would sever them.
const SIMPLE = new Uint8Array(256)
for (let code = 0; code < 256; code += 1) {
    let arcs = 0
    for (let k = 0; k < 8; k += 1) {
        if (!((code >> k) & 1) && ((code >> ((k + 1) & 7)) & 1)) arcs += 1
    }
    SIMPLE[code] = arcs === 1 ? 1 : 0
}

const nbrCode = (core, p, x, y, w, box) => {
    const up = y > box[1]; const dn = y < box[3] - 1
    const lf = x > box[0]; const rt = x < box[2] - 1
    let c = 0
    if (up && core[p - w]) c |= 1
    if (up && rt && core[p - w + 1]) c |= 2
    if (rt && core[p + 1]) c |= 4
    if (dn && rt && core[p + w + 1]) c |= 8
    if (dn && core[p + w]) c |= 16
    if (dn && lf && core[p + w - 1]) c |= 32
    if (lf && core[p - 1]) c |= 64
    if (up && lf && core[p - w - 1]) c |= 128
    return c
}

/**
 * Smooth the selection boundary: replace each value by the mean of its
 * (2r+1)² neighbourhood.
 *
 * The mask is a soft band around the decoder's own level set, and across a
 * straight edge that band is close to linear — a box mean of a linear ramp is
 * the same ramp, so a straight or diagonal edge comes back unmoved and only
 * the pixel-scale wobble averages out. That is curvature smoothing of the
 * level set, which is what a jagged staircase needs; a hard 0/255 median
 * would instead re-quantise the edge it is meant to soften.
 *
 * The one thing a mean cannot be trusted with is a thin structure: a 1 px wire
 * averages below the decision level along its whole length and disappears. A
 * core cell is therefore never cleared unless its neighbours form a single arc
 * — the standard connectivity test — so wires and spokes keep their spine.
 *
 * Sliding windows: two passes, one add and one drop per cell, independent of r.
 * Returns the number of pixels changed.
 */
export const smoothBoundary = (rgba, width, height, { radius = 1, rect = null } = {}) => {
    const r = Math.max(1, Math.round(radius))
    const box = workRect(rect, width, height, r + 1)
    const [x0, y0, x1, y1] = box
    if (x1 <= x0 || y1 <= y0) return 0
    const n = width * height
    const core = new Uint8Array(n)
    const sums = new Int32Array(n)
    for (let y = y0; y < y1; y += 1) {
        const row = y * width
        let s = 0
        const seed = Math.min(x0 + r, x1 - 1)
        for (let x = x0; x <= seed; x += 1) s += rgba[(row + x) * 4]
        for (let x = x0; x < x1; x += 1) {
            const p = row + x
            sums[p] = s
            core[p] = rgba[p * 4] >= 128 ? 1 : 0
            const drop = x - r
            const add = x + r + 1
            if (drop >= x0) s -= rgba[(row + drop) * 4]
            if (add < x1) s += rgba[(row + add) * 4]
        }
    }
    let changed = 0
    for (let x = x0; x < x1; x += 1) {
        const wx = Math.min(x + r, x1 - 1) - Math.max(x - r, x0) + 1
        let s = 0
        const seed = Math.min(y0 + r, y1 - 1)
        for (let y = y0; y <= seed; y += 1) s += sums[y * width + x]
        for (let y = y0; y < y1; y += 1) {
            const p = y * width + x
            const wy = Math.min(y + r, y1 - 1) - Math.max(y - r, y0) + 1
            const v = rgba[p * 4]
            const m = Math.round(s / (wx * wy))
            const drop = y - r
            const add = y + r + 1
            if (drop >= y0) s -= sums[drop * width + x]
            if (add < y1) s += sums[add * width + x]
            if (m === v) continue
            if (v >= 128 && m < 128) {
                // Against the RUNNING core, not a snapshot: two neighbouring
                // cells can each be safe to clear on their own and sever the
                // shape between them if both go.
                if (!SIMPLE[nbrCode(core, p, x, y, width, box)]) continue
                core[p] = 0
            } else if (v < 128 && m >= 128) core[p] = 1
            const j = p * 4
            rgba[j] = m; rgba[j + 1] = m; rgba[j + 2] = m; rgba[j + 3] = 255
            changed += 1
        }
    }
    return changed
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

/**
 * Per-component geometry of a thresholded mask (verify/debug hook). A count
 * alone cannot tell speckle from a second object dragged in, which is the
 * whole question region hygiene answers — so this reports what the rules
 * themselves test: area, bbox, solidity, and whether a click landed in it.
 * Sorted largest first, capped at `limit`.
 */
export const maskRegions = (rgba, w, h, clicks = [], limit = 12) => {
    const bin = new Uint8Array(w * h)
    for (let i = 0; i < bin.length; i += 1) bin[i] = rgba[i * 4] >= 128 ? 1 : 0
    const { labels, areas } = labelComponents(bin, w, h)
    const rows = areas.map((area, k) => ({
        area, label: k + 1, x0: w, y0: h, x1: 0, y1: 0, clicked: false,
    }))
    for (let i = 0; i < labels.length; i += 1) {
        const l = labels[i]
        if (!l) continue
        const r = rows[l - 1]
        const y = (i / w) | 0
        const x = i - y * w
        if (x < r.x0) r.x0 = x
        if (x >= r.x1) r.x1 = x + 1
        if (y < r.y0) r.y0 = y
        if (y >= r.y1) r.y1 = y + 1
    }
    for (const [cx, cy, label] of clicks) {
        if (label !== 1) continue
        const i = Math.min(h - 1, Math.max(0, Math.round(cy))) * w
            + Math.min(w - 1, Math.max(0, Math.round(cx)))
        if (labels[i]) rows[labels[i] - 1].clicked = true
    }
    const total = areas.reduce((a, b) => a + b, 0)
    return rows.sort((a, b) => b.area - a.area).slice(0, limit).map((r) => ({
        area: r.area,
        share: total ? r.area / total : 0,
        box: [r.x0, r.y0, r.x1, r.y1],
        solidity: r.area / Math.max(1, Math.max(r.x1 - r.x0, r.y1 - r.y0) ** 2),
        clicked: r.clicked,
    }))
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

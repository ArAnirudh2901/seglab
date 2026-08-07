/**
 * sam-core (pure — no DOM, no transformers.js)
 * ----------------------------------------------
 * Prompt/mask math for the on-device segmentation engine: mapping click/box
 * coordinates into the model's reshaped input space, building prompt tensor
 * payloads, picking the best of SAM's three candidate masks, and converting
 * a mask tensor into RGBA. Dependency-free so it can be unit-tested headless.
 *
 * Coordinate model: SAM resizes the source so its longest side hits the
 * model input size (1024) — `reshaped_input_sizes` is that resized [h, w].
 * Prompts must be expressed in THAT space, while post_process_masks returns
 * masks back in the source's own size. All helpers here take the source
 * dims + reshaped pair so both ends agree on one reference frame.
 */

/** Scale one source-space point into reshaped-input space. */
export const scalePointToReshaped = (x, y, srcW, srcH, reshaped) => [
    (x * reshaped[1]) / srcW,
    (y * reshaped[0]) / srcH,
]

/**
 * Build the point-prompt payload from `[x, y, label]` clicks (label 1 =
 * include, 0 = exclude). Returns plain arrays + dims; the engine wraps them
 * in Tensors (this module stays transformers-free).
 *
 * @param {Array<[number, number, 0|1]>} clicks  source-space clicks
 * @param {number} srcW
 * @param {number} srcH
 * @param {[number, number]} reshaped  [h, w] model-input size
 */
export const buildPointPrompt = (clicks, srcW, srcH, reshaped) => {
    if (!Array.isArray(clicks) || clicks.length === 0) return null
    const n = clicks.length
    const points = new Float32Array(n * 2)
    const labels = new BigInt64Array(n)
    for (let i = 0; i < n; i += 1) {
        const [x, y, label] = clicks[i]
        const [rx, ry] = scalePointToReshaped(x, y, srcW, srcH, reshaped)
        points[i * 2] = rx
        points[i * 2 + 1] = ry
        labels[i] = BigInt(label ? 1 : 0)
    }
    return {
        points,
        pointDims: [1, 1, n, 2],
        labels,
        labelDims: [1, 1, n],
    }
}

/**
 * Build the box-prompt payload from a source-space `[x0, y0, x1, y1]` box.
 * Also reports the box centre (source space): when the prompt is ONLY a box,
 * the engine adds the centre as a positive click, both because whole-object
 * box selection benefits from an interior anchor and because the
 * transformers.js SAM forward() derives default labels from `input_points`
 * and cannot run point-free.
 */
export const buildBoxPrompt = (box, srcW, srcH, reshaped) => {
    if (!Array.isArray(box) || box.length !== 4) return null
    const [x0, y0, x1, y1] = box
    const [rx0, ry0] = scalePointToReshaped(Math.min(x0, x1), Math.min(y0, y1), srcW, srcH, reshaped)
    const [rx1, ry1] = scalePointToReshaped(Math.max(x0, x1), Math.max(y0, y1), srcW, srcH, reshaped)
    return {
        box: new Float32Array([rx0, ry0, rx1, ry1]),
        boxDims: [1, 1, 4],
        center: [(Math.min(x0, x1) + Math.max(x0, x1)) / 2, (Math.min(y0, y1) + Math.max(y0, y1)) / 2],
    }
}

/** Index of the best of SAM's three candidate masks (argmax IoU score). */
export const pickBestMask = (scores) => {
    let best = 0
    for (let i = 1; i < scores.length; i += 1) {
        if (scores[i] > scores[best]) best = i
    }
    return best
}

/**
 * Extract one channel of a post-processed bool mask tensor ([1, C, H, W],
 * Uint8 0/1 data) as opaque white-on-black RGBA.
 */
export const maskChannelToRGBA = (maskData, width, height, channel) => {
    const size = width * height
    const offset = channel * size
    const rgba = new Uint8ClampedArray(size * 4)
    for (let i = 0; i < size; i += 1) {
        const v = maskData[offset + i] ? 255 : 0
        const j = i * 4
        rgba[j] = v
        rgba[j + 1] = v
        rgba[j + 2] = v
        rgba[j + 3] = 255
    }
    return rgba
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

/**
 * Accept/reject a selection mask. Empty means the prompts didn't land on
 * anything (a miss, not a crash); near-solid means the decoder failed to
 * separate the object (a real subject never fills the frame to within
 * 0.1%). Anything between — including very small masks — is legitimate:
 * minute-object selection is a first-class use case.
 */
export const validateClickMask = ({ coverage, bbox }) => {
    if (!bbox || coverage <= 0) return { usable: false, reason: 'empty mask (selection missed)' }
    if (coverage >= 0.999) return { usable: false, reason: 'solid mask (object not separated)' }
    return { usable: true, reason: null }
}

/* ─── Mask hygiene ───────────────────────────────────────────────────────── */

/** 4-connected component labeling on a binary Uint8 map. Iterative stack
 *  flood fill (no recursion — 1024² masks would blow the call stack).
 *  Returns { labels: Int32Array (0 = background, 1..n), areas: number[] }
 *  where areas[k] is the pixel count of component k+1. */
const labelComponents = (bin, w, h, scratch = null) => {
    const size = w * h
    const labels = scratch?.labels?.length >= size ? scratch.labels.fill(0, 0, size) : new Int32Array(size)
    const areas = []
    // The flood-fill stack is the same size every time and is dead between
    // calls — hand one in and a multi-instance query allocates it once.
    const stack = scratch?.stack?.length >= size ? scratch.stack : new Int32Array(size)
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

/** Component label at (or within `radius` of) a seed point — a positive
 *  click can land a few pixels outside the mask the decoder returned. */
const labelNearSeed = (labels, w, h, x, y, radius = 8) => {
    const cx = Math.round(x)
    const cy = Math.round(y)
    for (let r = 0; r <= radius; r += 1) {
        for (let dy = -r; dy <= r; dy += 1) {
            for (let dx = -r; dx <= r; dx += 1) {
                if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue // ring only
                const px = cx + dx
                const py = cy + dy
                if (px < 0 || py < 0 || px >= w || py >= h) continue
                const l = labels[py * w + px]
                if (l) return l
            }
        }
    }
    return 0
}

/**
 * Clean a decoded mask in place:
 *   1. keep components containing (or near) a positive seed;
 *   2. keep unseeded components ≥ 1% of the largest kept one (an object
 *      split in two by an occluder — a bike behind a tree — must survive
 *      even though only one half was clicked);
 *   3. drop the rest (threshold crumbs/islands);
 *   4. fill interior holes below ~1% of the mask area (upsample pinholes) —
 *      big legitimate gaps (background seen through a frame) stay open.
 *
 * All work is confined to the mask's bounding box grown by a small margin.
 * Every foreground component lies inside that box by definition, and every
 * pixel OUTSIDE it is background connected to the image border — so a
 * background component that reaches the box border is provably not a hole,
 * which is exactly the full-frame test. A 1%-of-frame object therefore pays
 * ~1% of the old cost, and text queries run this N times per prompt.
 *
 * @param {Uint8ClampedArray} rgba  white-on-black mask, modified in place
 * @param {Array<[number, number]>} seeds  positive prompt points
 * @param {object|null} scratch  makeMaskScratch() pool, shared across instances
 * @param {number[]|null} bbox  known [x0,y0,x1,y1] of the foreground, if any
 * @returns {{ kept: number, dropped: number, holesFilled: number, bbox: number[]|null }}
 */
export const cleanupMaskRGBA = (rgba, w, h, seeds = [], scratch = null, bbox = null) => {
    const box = bbox || summarizeMaskRGBA(rgba, w, h).bbox
    if (!box) return { kept: 0, dropped: 0, holesFilled: 0, bbox: null }

    // ≥1 keeps the hole test valid; ≥8 keeps labelNearSeed's search radius
    // representable in rect coordinates.
    const M = 9
    const rx0 = Math.max(0, box[0] - M)
    const ry0 = Math.max(0, box[1] - M)
    const rx1 = Math.min(w - 1, box[2] + M)
    const ry1 = Math.min(h - 1, box[3] + M)
    const rw = rx1 - rx0 + 1
    const rh = ry1 - ry0 + 1
    const size = rw * rh

    const bin = scratch?.bin?.length >= size ? scratch.bin : new Uint8Array(size)
    let fgArea = 0
    for (let y = 0; y < rh; y += 1) {
        const src = (ry0 + y) * w + rx0
        const dst = y * rw
        for (let x = 0; x < rw; x += 1) {
            if (rgba[(src + x) * 4] >= 128) { bin[dst + x] = 1; fgArea += 1 } else bin[dst + x] = 0
        }
    }
    if (!fgArea) return { kept: 0, dropped: 0, holesFilled: 0, bbox: null }

    const { labels, areas } = labelComponents(bin, rw, rh, scratch && { labels: scratch.fgLabels, stack: scratch.stack })
    // Keep-set as a label-indexed LUT, not a Set: the decision is read once
    // per pixel below, and a Set lookup per pixel is ~1M hash probes a mask.
    const keep = new Uint8Array(areas.length + 1)
    let keptCount = 0
    for (const [sx, sy] of seeds) {
        const l = labelNearSeed(labels, rw, rh, sx - rx0, sy - ry0)
        if (l && !keep[l]) { keep[l] = 1; keptCount += 1 }
    }
    if (keptCount === 0) {
        // No seed hit anything (box/lasso edge cases) — keep the largest.
        let best = 1
        for (let k = 1; k < areas.length; k += 1) if (areas[k] > areas[best - 1]) best = k + 1
        keep[best] = 1
        keptCount = 1
    }
    let largestKept = 0
    for (let l = 1; l <= areas.length; l += 1) {
        if (keep[l] && areas[l - 1] > largestKept) largestKept = areas[l - 1]
    }
    const minUnseeded = Math.max(48, largestKept * 0.01)
    for (let k = 0; k < areas.length; k += 1) {
        if (!keep[k + 1] && areas[k] >= minUnseeded) { keep[k + 1] = 1; keptCount += 1 }
    }

    let dropped = 0
    let minX = rw
    let minY = rh
    let maxX = -1
    let maxY = -1
    for (let y = 0; y < rh; y += 1) {
        const dst = y * rw
        const src = (ry0 + y) * w + rx0
        for (let x = 0; x < rw; x += 1) {
            const i = dst + x
            if (!bin[i]) continue
            if (keep[labels[i]]) {
                // Track the surviving foreground so the caller (and the edge
                // refinement) can skip re-scanning for the bbox.
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                continue
            }
            bin[i] = 0
            dropped += 1
            const j = (src + x) * 4
            rgba[j] = 0
            rgba[j + 1] = 0
            rgba[j + 2] = 0
        }
    }

    // Hole fill: label the background; components that never touch the
    // rect border are enclosed by foreground — fill the small ones. (Every
    // pixel outside the mask's bbox is background reaching the image border,
    // so "touches the rect border" ≡ "reaches the image border".)
    const inv = scratch?.inv?.length >= size ? scratch.inv : new Uint8Array(size)
    for (let i = 0; i < size; i += 1) inv[i] = bin[i] ? 0 : 1
    const bg = labelComponents(inv, rw, rh, scratch && { labels: scratch.bgLabels, stack: scratch.stack })
    const fill = new Uint8Array(bg.areas.length + 1).fill(1)
    fill[0] = 0
    for (let x = 0; x < rw; x += 1) {
        fill[bg.labels[x]] = 0
        fill[bg.labels[(rh - 1) * rw + x]] = 0
    }
    for (let y = 0; y < rh; y += 1) {
        fill[bg.labels[y * rw]] = 0
        fill[bg.labels[y * rw + rw - 1]] = 0
    }
    const maxHole = Math.max(64, (fgArea - dropped) * 0.01)
    let anyFill = false
    for (let k = 0; k < bg.areas.length; k += 1) {
        if (fill[k + 1] && bg.areas[k] > maxHole) fill[k + 1] = 0
        if (fill[k + 1]) anyFill = true
    }
    let holesFilled = 0
    if (anyFill) {
        for (let y = 0; y < rh; y += 1) {
            const dst = y * rw
            const src = (ry0 + y) * w + rx0
            for (let x = 0; x < rw; x += 1) {
                if (!fill[bg.labels[dst + x]]) continue
                holesFilled += 1
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                const j = (src + x) * 4
                rgba[j] = 255
                rgba[j + 1] = 255
                rgba[j + 2] = 255
            }
        }
    }
    return {
        kept: keptCount,
        dropped,
        holesFilled,
        bbox: maxX >= 0 ? [minX + rx0, minY + ry0, maxX + rx0, maxY + ry0] : null,
    }
}

/**
 * Reusable buffers for `cleanupMaskRGBA`. Share one across every mask of a
 * multi-instance text query so hygiene allocates once, not N times.
 */
export const makeMaskScratch = (w, h) => {
    const size = w * h
    return {
        bin: new Uint8Array(size),
        inv: new Uint8Array(size),
        fgLabels: new Int32Array(size),
        bgLabels: new Int32Array(size),
        stack: new Int32Array(size),
    }
}

/** Component count of a mask (verify/debug hook). */
export const countMaskComponents = (rgba, w, h) => {
    const bin = new Uint8Array(w * h)
    for (let i = 0; i < bin.length; i += 1) bin[i] = rgba[i * 4] >= 128 ? 1 : 0
    return labelComponents(bin, w, h).areas.length
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

/**
 * mask-select — candidate arbitration and region hygiene on SAM's raw logit
 * field (DESIGN-MASK-LANE §10a).
 *
 * SAM emits three masks per decode (roughly subpart / part / whole) plus three
 * predicted IoUs, and the lane used to take argmax of that score. Predicted IoU
 * answers "how well would this mask score against ITS OWN target", which is not
 * the question a click asks. It is silent on every failure this module handles:
 *
 *   nested objects   a confident sub-cluster of petals outscores a whole bloom,
 *                    so the top score IS the mask that leaves parts out
 *   negative clicks  a candidate that still covers an exclude point can win
 *   box prompts      a candidate that spills far outside the drawn box can win
 *   mushy fields     a candidate with no real boundary can score high; low
 *                    stability is what exposes it
 *   hierarchy drift  the level can change between click 1 and click 2, so
 *                    adding a point to a person snaps the mask to their sleeve
 *   speckle / holes  islands far from the subject, and gaps inside it
 *
 * Everything here is pure math on Float32Array — no DOM, no GPU, no model — so
 * it runs in the worker, costs a fraction of a millisecond on the 256² grid,
 * and is testable in node (verify.mjs phase T).
 *
 * `stabilityScore` and `cleanRegions` are ports of upstream SAM's own
 * `calculate_stability_score` and `remove_small_regions`
 * (segment_anything/utils/amg.py), which this lane never had because it only
 * ever ran the interactive path, not the automatic mask generator.
 */

/** Logits, not probabilities: SAM's decision boundary is zero. */
const T = 0

/**
 * Upstream SAM's stability score: how much the mask changes when the threshold
 * is nudged either way. |field > +off| / |field > -off|.
 *
 * A crisp object has a steep field, so both thresholds cut in nearly the same
 * place and the ratio approaches 1. A candidate straddling an ambiguous
 * boundary — the exact thing predicted IoU cannot see — collapses toward 0.
 */
export const stabilityScore = (field, offset = 1) => {
    let hi = 0
    let lo = 0
    for (let i = 0; i < field.length; i += 1) {
        const v = field[i]
        if (v > T + offset) hi += 1
        if (v > T - offset) lo += 1
    }
    return lo ? hi / lo : 0
}

/** IoU of two logit fields at their zero crossing — "is this the same object". */
export const fieldIoU = (a, b) => {
    let inter = 0
    let union = 0
    for (let i = 0; i < a.length; i += 1) {
        const x = a[i] > T
        const y = b[i] > T
        if (x && y) inter += 1
        if (x || y) union += 1
    }
    return union ? inter / union : 0
}

/** Cells above threshold. */
export const fieldArea = (f) => {
    let n = 0
    for (let i = 0; i < f.length; i += 1) if (f[i] > T) n += 1
    return n
}

const clampi = (v, hi) => (v < 0 ? 0 : (v > hi ? hi : v))

/** Prompt coords live in the prompt's own space; `scale` maps them onto a
 *  w×h grid. Returns a flat index. */
const cellIndex = (c, w, h, scale) => {
    const gx = clampi(Math.round(c.x * scale), w - 1)
    const gy = clampi(Math.round(c.y * scale), h - 1)
    return gy * w + gx
}

/** The square-grid case: candidate arbitration only ever sees SAM's 256². */
const cellOf = (c, side, scale) => cellIndex(c, side, side, scale)

/** Any positive cell in the 3×3 around p. */
const near = (field, side, p) => {
    const x = p % side
    const y = (p - x) / side
    for (let dy = -1; dy <= 1; dy += 1) {
        const yy = y + dy
        if (yy < 0 || yy >= side) continue
        for (let dx = -1; dx <= 1; dx += 1) {
            const xx = x + dx
            if (xx < 0 || xx >= side) continue
            if (field[yy * side + xx] > T) return true
        }
    }
    return false
}

/**
 * How badly a candidate contradicts what the user actually asked for.
 *
 * Positives are checked leniently (3×3): one grid cell is 4 px at the 1024
 * proxy, so a click on a thin structure legitimately lands a cell off its
 * centre. Negatives are checked strictly at the cell — an exclude point that
 * is still inside the mask is not a rounding error, it is the wrong candidate.
 *
 * Labels 2/3 are box corners, not clicks, and are handled by `boxFraction`.
 */
export const promptFit = (field, side, clicks, scale) => {
    let miss = 0
    let leak = 0
    for (const c of clicks) {
        const label = c.label ?? 1
        if (label !== 0 && label !== 1) continue
        const p = cellOf(c, side, scale)
        if (label === 1) { if (!near(field, side, p)) miss += 1 } else if (field[p] > T) leak += 1
    }
    return { miss, leak }
}

/** The [x0,y0,x1,y1] box carried by label-2 / label-3 prompt points, if any. */
export const boxFromClicks = (clicks) => {
    const tl = clicks.find((c) => c.label === 2)
    const br = clicks.find((c) => c.label === 3)
    return tl && br ? [tl.x, tl.y, br.x, br.y] : null
}

/** Fraction of the candidate's area that falls inside the prompt box. */
export const boxFraction = (field, side, box, scale) =>
    planeStats(field, side, boxCells(box, side, scale), null).inBox

/** The prompt box in grid cells, inclusive — the form every per-cell test wants.
 *  Null box means "no box", which reads as full agreement. */
const boxCells = (box, side, scale) => (box ? [
    clampi(Math.round(box[0] * scale), side - 1),
    clampi(Math.round(box[1] * scale), side - 1),
    clampi(Math.round(box[2] * scale), side - 1),
    clampi(Math.round(box[3] * scale), side - 1),
] : null)

/**
 * Every per-plane statistic arbitration needs, in ONE traversal.
 *
 * stabilityScore, fieldArea, boxFraction and fieldIoU each walked the same 65536
 * cells — four passes per candidate, three candidates, up to twice per click, so
 * ~1.6 M reads where 200 k suffice. The exported single-purpose versions above
 * stay as they are: verify.mjs tests them one number at a time, and one of them
 * (stabilityScore) takes an offset this loop fixes at 1.
 *
 * Row-major, so the box test is a range check per row instead of a modulo and a
 * divide per cell.
 */
const planeStats = (field, side, cells, previous) => {
    const bx0 = cells ? cells[0] : 0
    const by0 = cells ? cells[1] : 0
    const bx1 = cells ? cells[2] : side - 1
    const by1 = cells ? cells[3] : side - 1
    let hi = 0
    let lo = 0
    let area = 0
    let inside = 0
    let inter = 0
    let union = 0
    for (let y = 0; y < side; y += 1) {
        const row = y * side
        const inRow = y >= by0 && y <= by1
        for (let x = 0; x < side; x += 1) {
            const v = field[row + x]
            if (v > T + 1) hi += 1
            if (v > T - 1) lo += 1
            const on = v > T
            if (on) {
                area += 1
                if (inRow && x >= bx0 && x <= bx1) inside += 1
            }
            if (previous) {
                const was = previous[row + x] > T
                if (on) { union += 1; if (was) inter += 1 } else if (was) union += 1
            }
        }
    }
    return {
        stability: lo ? hi / lo : 0,
        area,
        inBox: cells ? (area ? inside / area : 1) : 1,
        agree: previous ? (union ? inter / union : 0) : 0,
    }
}

/**
 * Rank among candidates that satisfy the prompt equally well.
 *
 * Predicted IoU stays the backbone — it is the only signal trained against real
 * masks — but it is discounted by two things it demonstrably cannot see:
 * stability (is there a boundary at all) and box agreement (did the detector
 * ask for THIS object). Both are multiplicative and floored, so neither can
 * veto a confident candidate on its own; they break ties and demote the
 * obviously-wrong.
 */
const rank = (s) => s.score * (0.5 + 0.5 * s.stability) * (0.3 + 0.7 * s.inBox)

/**
 * Pick one of SAM's candidates.
 *
 * Order of authority, strongest first:
 *   1. prompt consistency — never return a mask that covers an exclude click or
 *      misses an include click while a sibling does not. This is a hard filter
 *      because it encodes an explicit instruction, not a preference.
 *   2. continuity — when refining (a previous mask exists), prefer the survivor
 *      that most agrees with what the user was already looking at. This is what
 *      keeps a second click from switching hierarchy level. Below `minAgree`
 *      nothing genuinely continues the old mask, so fall through rather than
 *      pin to an unrelated blob.
 *   3. rank — predicted IoU tempered by stability and box fit.
 *
 * Note the composition: a negative click DISQUALIFIES the too-large candidate
 * in step 1, so step 2 can only choose among masks that already honour it.
 * Continuity therefore never fights an explicit shrink.
 */
export const chooseCandidate = ({
    planes, scores = [], clicks = [], side, scale, previous = null, minAgree = 0.5,
}) => {
    const cells = boxCells(boxFromClicks(clicks), side, scale)
    const stats = planes.map((p, i) => {
        const fit = promptFit(p, side, clicks, scale)
        const s = planeStats(p, side, cells, previous)
        return {
            i,
            violations: fit.miss + fit.leak,
            miss: fit.miss,
            leak: fit.leak,
            stability: s.stability,
            inBox: s.inBox,
            area: s.area,
            score: scores[i] ?? 0,
            agree: s.agree,
        }
    })

    let minV = Infinity
    for (const s of stats) if (s.area > 0 && s.violations < minV) minV = s.violations
    // An all-empty decode has nothing to filter; fall back to the full set so a
    // degenerate prompt still returns something rather than throwing.
    const pool = minV === Infinity ? stats : stats.filter((s) => s.area > 0 && s.violations === minV)

    if (previous) {
        let best = pool[0]
        for (const s of pool) if (s.agree > best.agree) best = s
        if (best.agree >= minAgree) return { index: best.i, reason: 'agree', stats, pick: best }
    }
    let best = pool[0]
    for (const s of pool) if (rank(s) > rank(best)) best = s
    return { index: best.i, reason: minV > 0 ? 'least-violation' : 'rank', stats, pick: best }
}

/* ── 4-connected components, as scanline runs ─────────────────────────────────
 * Runs plus union-find, not a flood fill.
 *
 * The fill this replaces allocated an Int32Array label plane the size of the
 * field and chased neighbours in every direction. On SAM's 256² grid that is
 * free. On the PROXY field — where the speckle the upsample and the guided
 * filter create actually lives — it is a 2.8 MB plane walked in random order,
 * which is the one access pattern a small cache cannot absorb, and it is paid
 * on every click.
 *
 * Runs read each cell exactly once, in address order, so the field streams. The
 * only scattered access is the union-find, and it is sized by RUN count
 * (thousands) rather than by pixels, so it stays resident on any machine.
 *
 * The runs are also the write plan: rewriting a component is one
 * TypedArray.fill per run — a memset — so the second pass never re-reads the
 * field at all.
 */

// Scratch, reused across calls. A click runs this up to eight times (chosen
// plus alternates, foreground then background) and the worker is single
// threaded, so allocating per call would only feed the collector.
let runs = new Int32Array(3 * 4096)   // [start, end, label] per run, absolute
let parent = new Int32Array(4096)
let csize = new Int32Array(4096)
let cedge = new Uint8Array(4096)
let cdrop = new Uint8Array(4096)
// Per-component bounds, inclusive. Carried because separation and shape — not
// size — are what tell a second object apart from a fragment of this one.
let cbx0 = new Int32Array(4096); let cby0 = new Int32Array(4096)
let cbx1 = new Int32Array(4096); let cby1 = new Int32Array(4096)
// Previous / current row's runs, as x coords. Sized by the widest possible row.
let aStart = new Int32Array(512); let aEnd = new Int32Array(512); let aLab = new Int32Array(512)
let bStart = new Int32Array(512); let bEnd = new Int32Array(512); let bLab = new Int32Array(512)

const growRuns = (need) => {
    if (need * 3 <= runs.length) return
    let cap = runs.length
    while (cap < need * 3) cap *= 2
    const n = new Int32Array(cap)
    n.set(runs)
    runs = n
}

const growLabels = (need) => {
    if (need <= parent.length) return
    let cap = parent.length
    while (cap < need) cap *= 2
    const p = new Int32Array(cap); p.set(parent); parent = p
    const s = new Int32Array(cap); s.set(csize); csize = s
    const e = new Uint8Array(cap); e.set(cedge); cedge = e
    const b0 = new Int32Array(cap); b0.set(cbx0); cbx0 = b0
    const b1 = new Int32Array(cap); b1.set(cby0); cby0 = b1
    const b2 = new Int32Array(cap); b2.set(cbx1); cbx1 = b2
    const b3 = new Int32Array(cap); b3.set(cby1); cby1 = b3
    cdrop = new Uint8Array(cap)          // rewritten per pass, never carried
}

const growRows = (need) => {
    if (need <= aStart.length) return
    let cap = aStart.length
    while (cap < need) cap *= 2
    aStart = new Int32Array(cap); aEnd = new Int32Array(cap); aLab = new Int32Array(cap)
    bStart = new Int32Array(cap); bEnd = new Int32Array(cap); bLab = new Int32Array(cap)
}

/** Union-find with path halving — the trees are shallow, so no rank needed. */
const find = (i) => {
    let r = i
    while (parent[r] !== r) { parent[r] = parent[parent[r]]; r = parent[r] }
    return r
}
const union = (a, b) => {
    const ra = find(a)
    const rb = find(b)
    if (ra === rb) return ra
    if (ra < rb) { parent[rb] = ra; return ra }   // root is always the lower id
    parent[ra] = rb
    return rb
}

/**
 * Label every run of `want` polarity inside `rect` (x0,y0,x1,y1, ends
 * exclusive) of a w-strided field. Results land in the module scratch above.
 *
 * `rect` is what makes this affordable on the proxy: the caller passes the
 * mask's own bounding box, so the scan covers the subject rather than the frame.
 */
const scanRuns = (field, w, rect, want) => {
    const rx0 = rect[0]; const ry0 = rect[1]; const rx1 = rect[2]; const ry1 = rect[3]
    growRows(((rx1 - rx0) >> 1) + 2)
    let pStart = aStart; let pEnd = aEnd; let pLab = aLab
    let cStart = bStart; let cEnd = bEnd; let cLab = bLab
    let nRun = 0
    let nLab = 0
    let total = 0
    let prevN = 0
    for (let y = ry0; y < ry1; y += 1) {
        const base = y * w
        const yEdge = y === ry0 || y === ry1 - 1
        let curN = 0
        let pi = 0
        let x = rx0
        while (x < rx1) {
            while (x < rx1 && (field[base + x] > T) !== want) x += 1
            if (x >= rx1) break
            const x0 = x
            do { x += 1 } while (x < rx1 && (field[base + x] > T) === want)
            const x1 = x
            // Both rows are sorted, so one shared pointer walks the previous
            // row: runs that end before this one starts can never touch a later
            // run either, and are retired for good.
            while (pi < prevN && pEnd[pi] <= x0) pi += 1
            let lab = -1
            for (let pj = pi; pj < prevN && pStart[pj] < x1; pj += 1) {
                const r = find(pLab[pj])
                lab = lab < 0 ? r : union(lab, r)
            }
            if (lab < 0) {
                growLabels(nLab + 1)
                lab = nLab
                parent[lab] = lab
                csize[lab] = 0
                cedge[lab] = 0
                cbx0[lab] = x0; cbx1[lab] = x1 - 1
                cby0[lab] = y; cby1[lab] = y
                nLab += 1
            }
            if (x0 < cbx0[lab]) cbx0[lab] = x0
            if (x1 - 1 > cbx1[lab]) cbx1[lab] = x1 - 1
            if (y < cby0[lab]) cby0[lab] = y
            if (y > cby1[lab]) cby1[lab] = y
            const len = x1 - x0
            csize[lab] += len
            total += len
            if (yEdge || x0 === rx0 || x1 === rx1) cedge[lab] = 1
            growRuns(nRun + 1)
            const k = nRun * 3
            runs[k] = base + x0
            runs[k + 1] = base + x1
            runs[k + 2] = lab
            nRun += 1
            cStart[curN] = x0; cEnd[curN] = x1; cLab[curN] = lab; curN += 1
        }
        let t = pStart; pStart = cStart; cStart = t
        t = pEnd; pEnd = cEnd; cEnd = t
        t = pLab; pLab = cLab; cLab = t
        prevN = curN
    }
    // Sizes and edge flags landed on whichever label a run held at the time;
    // fold them onto roots. Roots always carry the lower id, so one reverse
    // pass sees every child before its root.
    for (let i = nLab - 1; i >= 0; i -= 1) {
        const r = find(i)
        if (r !== i) {
            csize[r] += csize[i]
            cedge[r] |= cedge[i]
            if (cbx0[i] < cbx0[r]) cbx0[r] = cbx0[i]
            if (cby0[i] < cby0[r]) cby0[r] = cby0[i]
            if (cbx1[i] > cbx1[r]) cbx1[r] = cbx1[i]
            if (cby1[i] > cby1[r]) cby1[r] = cby1[i]
        }
    }
    return { nRun, nLab, total }
}

/** Root of the component covering flat index `idx`, or -1. Runs are emitted in
 *  address order, so this is a binary search — no label plane to look into. */
const rootAt = (idx, nRun) => {
    let lo = 0
    let hi = nRun - 1
    while (lo <= hi) {
        const mid = (lo + hi) >> 1
        const k = mid * 3
        if (idx < runs[k]) hi = mid - 1
        else if (idx >= runs[k + 1]) lo = mid + 1
        else return find(runs[k + 2])
    }
    return -1
}

/** Longest side of a component's bounding box — its own scale, which is what a
 *  separation has to be judged against. */
const extentOf = (i) => Math.max(cbx1[i] - cbx0[i] + 1, cby1[i] - cby0[i] + 1)

/**
 * Area against the SQUARE of the long side — "is this a blob or a sliver".
 *
 * Not the bbox fill: an axis-aligned 2×300 wire fills its own bounding box
 * completely and would read as perfectly compact. Dividing by the long side
 * squared measures mean thickness relative to length instead, so a disc lands
 * near 0.8, a pole with wires near 0.1, and that wire at 0.007.
 */
const solidity = (i) => {
    const e = extentOf(i)
    return e > 0 ? csize[i] / (e * e) : 0
}

/** Chebyshev gap between two components' bounding boxes, in cells. 0 = touching
 *  or overlapping. */
const boxGap = (i, j) => {
    const gx = Math.max(cbx0[i] - cbx1[j], cbx0[j] - cbx1[i]) - 1
    const gy = Math.max(cby0[i] - cby1[j], cby0[j] - cby1[i]) - 1
    return Math.max(gx > 0 ? gx : 0, gy > 0 ? gy : 0)
}

// Written into removed/filled cells. Deliberately modest: `bandWidth` in the
// adapter averages |∇field| only where |v| ≤ 1, so a huge step would distort
// the band it derives from the real boundary. 2.5 is safely outside the band
// and still gives the bicubic upsample a sane slope to interpolate across.
// Callers working on an ALREADY-matted field pass their own saturating value.
const FILL = 2.5

/**
 * Region hygiene — upstream SAM's `remove_small_regions`, both modes, in one
 * pass each. Rewrites `field` in place and reports what it did.
 *
 * Runs on any w×h field, twice per selection, because the mask is thresholded
 * at PROXY resolution, not on SAM's grid. Cleaning only the 256² logits leaves
 * the last two stages unpoliced, and both were measured creating detached
 * speckle out of a field that reached them with a single component: the bicubic
 * upsample severs a one-cell-wide neck and rings beside the -FILL it is handed,
 * and the colour guided filter flips a near-zero plateau wherever a strong
 * colour edge runs just outside the boundary. `rect` is what keeps the second
 * pass cheap — the caller hands over the mask's own bounding box, so the scan
 * covers the subject instead of the frame.
 *
 * Islands: a component holding no include click and far smaller than the main
 * one is speckle — but ONLY on a mask that is already essentially one blob.
 *
 * That gate is not caution, it is a measured correction. Without it, on the
 * streetlight crop (real pixels, real pipeline, scratchpad/quality.mjs) the
 * rule removed 21 components and cost 0.050 IoU / 0.051 boundary IoU against
 * the shipped pipeline — and it cost 0.041 even on a field with speckle
 * deliberately injected, because a 256² grid shatters wires and a thin arm into
 * many small components and every one of them looks like speckle by size.
 * Severing wires IS the "parts left out" failure this module exists to fix.
 *
 * `minDominance` is therefore the real safety property: if the largest
 * component is not almost the whole mask, the subject is legitimately
 * fragmented and nothing is removed. Leaving two blobs of speckle on a
 * streetlight is strictly better than cutting its wires. The margin is wide —
 * that crop's mask scores 0.568 (42 components, the two largest 739 and 342
 * cells), against a compact subject with speckle at ~0.9 — so 0.85 separates
 * them without sitting on either.
 *
 * But dominance is only a PROXY for "fragmented", and one honest second part
 * fakes it: a seated person whose leg is cut off by a chair scores 0.838 and
 * shipped 26 specks of 40-80 px with it. Below the gate the rule therefore does
 * not switch off, it switches to capping the RISK of the removal instead of
 * guessing at the subject: only components under a quarter of `islandCells` go,
 * and only while they add up to `tinyFrac` of the mask. On the streetlight that
 * speckle is percent-of-mask — orders of magnitude over the budget — so the
 * whole removal is refused rather than half-taken.
 *
 * Four conditions have to hold together before anything is removed, and every
 * one of them was added because the previous rule was measured deleting real
 * structure:
 *
 *   dominance ≥ 0.85   or the subject is legitimately fragmented (streetlight)
 *   ≤ islandCells      absolute, because speckle is a handful of cells whatever
 *                      the subject's size. Relative-only rules failed twice:
 *                      10%-of-largest deleted a detached wire on a 90%-dominant
 *                      mask, and 2%-of-total deleted petal fragments on the rose
 *                      cluster (real pixels: IoU 0.921 → 0.905, and the pixels
 *                      MISSED went up by 444)
 *   < islandFrac       so a small mask cannot have a meaningful share removed
 *   no include click   the user pointed at it, so it is the subject by definition
 *
 * 16 cells is tuned on the 256² grid — 0.02 % of it. A finer field scales it by
 * area rather than re-tuning, so the rule stays the same rule; 16 is also the
 * floor, because on a coarser grid speckle is still counted in single cells.
 *
 * Outliers are the other half of the islands problem, and the size rule cannot
 * reach them: a click on one of a repeated subject brings the neighbours back
 * whole, so they are large, and they push dominance under 0.85 on their way in.
 * They are separated by SPACE, not by size, so that is what is tested:
 *
 *   gap ≥ sepFrac · anchor extent   clear of every clicked component (min 2
 *                                   cells, so a grid-severed neck never counts)
 *   both sides solid                area / long-side² ≥ minSolidity on the anchor
 *                                   AND on the candidate. This is what keeps the
 *                                   streetlight safe: its parts are far apart
 *                                   too, but a pole-and-wires component scores
 *                                   ~0.1, which turns the rule off for that mask
 *                                   entirely — and a detached wire scores lower
 *                                   still, so it is never the thing removed
 *   no include click                as above, the user pointed at it. An include
 *                                   click is also what ANCHORS the rule: with no
 *                                   click there is nothing to be separated from,
 *                                   so the rule does not run at all (export
 *                                   refinement, box prompts)
 *
 * Holes need no such gate: measured on the same crop, filling cost 0.000 IoU
 * and moved the derived band width by 0.08%.
 *
 * Holes: a background component that does not reach the image border is
 * enclosed by the mask. Small ones with no exclude click in them are the "parts
 * left out" failure — gaps between petals, a dark gap inside a subject — not
 * deliberate cutouts. A hole the user explicitly clicked to exclude is left
 * alone, and so is anything touching the border, which is the background
 * itself.
 */
export const cleanRegions = (field, w, h, {
    clicks = [], scale = 1, rect = null, fill = FILL,
    islandCells = null, islandFrac = 0.02, holeFrac = 0.06, minDominance = 0.85,
    sepFrac = 0.15, minSolidity = 0.3, tinyFrac = 0.005,
} = {}) => {
    const box = rect || [0, 0, w, h]
    if (box[2] <= box[0] || box[3] <= box[1]) return { islands: 0, holes: 0, dirty: null }
    // 16 cells is the figure tuned on the 256² grid. The same speckle covers the
    // same FRACTION of a proxy or export field, not the same count, so the
    // threshold scales with the grid rather than being re-tuned per resolution.
    const minCells = islandCells ?? Math.max(16, Math.round((16 * w * h) / 65536))

    let islands = 0
    let holes = 0
    let dx0 = w; let dy0 = h; let dx1 = 0; let dy1 = 0
    const rewrite = (nRun, value) => {
        for (let i = 0; i < nRun; i += 1) {
            const k = i * 3
            if (!cdrop[find(runs[k + 2])]) continue
            const s = runs[k]
            const e = runs[k + 1]
            field.fill(value, s, e)
            const y = (s / w) | 0
            const x = s - y * w
            if (x < dx0) dx0 = x
            if (e - y * w > dx1) dx1 = e - y * w
            if (y < dy0) dy0 = y
            if (y >= dy1) dy1 = y + 1
        }
    }

    const fg = scanRuns(field, w, box, true)
    let removed = 0

    // What the click actually asked for. No include click, no anchor and the
    // outlier rule does not run: at export scale the prompt is history, and a
    // box prompt never named a component — guessing the anchor there would
    // delete the second blob of a deliberately two-click selection.
    const anchors = []
    if (fg.total > 0) {
        for (const c of clicks) {
            if ((c.label ?? 1) !== 1) continue
            const cell = cellIndex(c, w, h, scale)
            const r = rootAt(cell, fg.nRun)
            if (r < 0) continue
            if (!anchors.includes(r)) anchors.push(r)
        }
    }
    // Separation is off for a subject that is legitimately thin or fragmented —
    // the streetlight — and on only for solid ones.
    const separable = anchors.length > 0 && anchors.every((a) => solidity(a) >= minSolidity)

    if (fg.nLab > 1 && fg.total > 0) {
        cdrop.fill(0, 0, fg.nLab)
        let max = 0
        for (let i = 0; i < fg.nLab; i += 1) if (find(i) === i && csize[i] > max) max = csize[i]
        if (max / fg.total >= minDominance) {
            const limit = islandFrac * fg.total
            for (let i = 0; i < fg.nLab; i += 1) {
                if (find(i) === i && csize[i] <= minCells && csize[i] < limit) cdrop[i] = 1
            }
        } else {
            // Dominance is a proxy for "the subject is legitimately fragmented",
            // and ONE honest second part is enough to fake it: a click on a
            // seated person whose leg is cut off by a chair scored 0.838, which
            // switched the rule off and shipped 26 specks of 40-80 px with it
            // (DSC_0139.NEF, .leak-probe.mjs). What a fragmented subject cannot
            // fake is the risk of the removal itself, so that is what is capped
            // here instead: only the vanishingly small go, and only while they
            // add up to a rounding error. On the streetlight — the crop this
            // gate exists for — its speckle is ~5% of the mask, an order of
            // magnitude over the budget, so nothing is removed there either.
            const tiny = Math.max(16, minCells >> 2)
            const budget = tinyFrac * fg.total
            let sum = 0
            for (let i = 0; i < fg.nLab; i += 1) {
                if (find(i) === i && csize[i] <= tiny) { cdrop[i] = 1; sum += csize[i] }
            }
            if (sum > budget) cdrop.fill(0, 0, fg.nLab)
        }

        // Separated outliers — a second OBJECT rather than a piece of this one.
        // A click on a repeated subject (one rivet in a plate of them, one grape)
        // comes back with the neighbours attached: far too big for the size rule,
        // and big enough to drag dominance under its gate, so nothing above can
        // see them.
        //
        // Distance alone cannot be the test — the streetlight regression IS a
        // real subject whose parts are far apart — so both sides must be compact.
        // A fragmented or thin subject disables the rule outright, and a thin
        // fragment is never taken by it. What remains is two solid blobs with
        // clear space between them, and that is two objects.
        if (separable) {
            for (let i = 0; i < fg.nLab; i += 1) {
                if (find(i) !== i || cdrop[i] || anchors.includes(i)) continue
                if (solidity(i) < minSolidity) continue
                // Clear of EVERY anchor: a blob near any clicked one belongs to it.
                let far = true
                for (const a of anchors) {
                    if (boxGap(i, a) < Math.max(2, sepFrac * extentOf(a))) { far = false; break }
                }
                if (far) cdrop[i] = 1
            }
        }

        for (const a of anchors) cdrop[a] = 0    // the user pointed at it: it is the subject
        for (let i = 0; i < fg.nLab; i += 1) if (cdrop[i]) { islands += 1; removed += csize[i] }
        if (islands) rewrite(fg.nRun, -fill)
    }

    const area = fg.total - removed
    if (area > 0) {
        const bg = scanRuns(field, w, box, false)
        cdrop.fill(0, 0, bg.nLab)
        const limit = holeFrac * area
        for (let i = 0; i < bg.nLab; i += 1) {
            if (find(i) === i && !cedge[i] && csize[i] < limit) cdrop[i] = 1
        }
        for (const c of clicks) {               // an explicit cutout stays a cutout
            if (c.label !== 0) continue
            const r = rootAt(cellIndex(c, w, h, scale), bg.nRun)
            if (r >= 0) cdrop[r] = 0
        }
        for (let i = 0; i < bg.nLab; i += 1) if (cdrop[i]) holes += 1
        if (holes) rewrite(bg.nRun, fill)
    }

    return { islands, holes, dirty: dx1 > dx0 ? [dx0, dy0, dx1, dy1] : null }
}

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
 *   box prompts      a candidate that spills far outside the detector's box can
 *                    win, and text search prompts exclusively by box
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

/** Prompt coords live in SAM's 1024² space; `scale` maps them onto the grid. */
const cellOf = (c, side, scale) => {
    const gx = clampi(Math.round(c.x * scale), side - 1)
    const gy = clampi(Math.round(c.y * scale), side - 1)
    return gy * side + gx
}

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
export const boxFraction = (field, side, box, scale) => {
    const x0 = clampi(Math.round(box[0] * scale), side - 1)
    const y0 = clampi(Math.round(box[1] * scale), side - 1)
    const x1 = clampi(Math.round(box[2] * scale), side - 1)
    const y1 = clampi(Math.round(box[3] * scale), side - 1)
    let inside = 0
    let total = 0
    for (let i = 0; i < field.length; i += 1) {
        if (field[i] <= T) continue
        total += 1
        const x = i % side
        const y = (i - x) / side
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) inside += 1
    }
    return total ? inside / total : 1
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
    const box = boxFromClicks(clicks)
    const stats = planes.map((p, i) => {
        const fit = promptFit(p, side, clicks, scale)
        return {
            i,
            violations: fit.miss + fit.leak,
            miss: fit.miss,
            leak: fit.leak,
            stability: stabilityScore(p),
            inBox: box ? boxFraction(p, side, box, scale) : 1,
            area: fieldArea(p),
            score: scores[i] ?? 0,
            agree: previous ? fieldIoU(previous, p) : 0,
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

/** 4-connected components over cells satisfying `inside`. */
const components = (field, side, inside) => {
    const n = side * side
    const lab = new Int32Array(n).fill(-1)
    const sizes = []
    const edge = []
    const stack = new Int32Array(n)
    for (let seed = 0; seed < n; seed += 1) {
        if (lab[seed] !== -1 || !inside(field[seed])) continue
        const id = sizes.length
        sizes.push(0)
        edge.push(false)
        let sp = 0
        stack[sp] = seed
        sp += 1
        lab[seed] = id
        while (sp > 0) {
            sp -= 1
            const p = stack[sp]
            sizes[id] += 1
            const x = p % side
            const y = (p - x) / side
            if (x === 0 || y === 0 || x === side - 1 || y === side - 1) edge[id] = true
            if (x > 0 && lab[p - 1] === -1 && inside(field[p - 1])) { lab[p - 1] = id; stack[sp] = p - 1; sp += 1 }
            if (x < side - 1 && lab[p + 1] === -1 && inside(field[p + 1])) { lab[p + 1] = id; stack[sp] = p + 1; sp += 1 }
            if (y > 0 && lab[p - side] === -1 && inside(field[p - side])) { lab[p - side] = id; stack[sp] = p - side; sp += 1 }
            if (y < side - 1 && lab[p + side] === -1 && inside(field[p + side])) { lab[p + side] = id; stack[sp] = p + side; sp += 1 }
        }
    }
    return { lab, sizes, edge }
}

// Written into removed/filled cells. Deliberately modest: `bandWidth` in the
// adapter averages |∇field| only where |v| ≤ 1, so a huge step would distort
// the band it derives from the real boundary. 2.5 is safely outside the band
// and still gives the bicubic upsample a sane slope to interpolate across.
const FILL = 2.5

/**
 * Region hygiene — upstream SAM's `remove_small_regions`, both modes, in one
 * pass each. Rewrites `field` in place and reports what it did.
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
 * 16 cells is tuned for the 256² grid this lane always emits — about 64×64 px
 * at the 1024 proxy, or 0.02% of the mask grid.
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
export const cleanRegions = (field, side, {
    clicks = [], scale = 1, islandCells = 16, islandFrac = 0.02, holeFrac = 0.06, minDominance = 0.85,
} = {}) => {
    const pos = clicks.filter((c) => (c.label ?? 1) === 1).map((c) => cellOf(c, side, scale))
    const neg = clicks.filter((c) => c.label === 0).map((c) => cellOf(c, side, scale))

    let islands = 0
    let holes = 0

    const fg = components(field, side, (v) => v > T)
    let max = 0
    let total = 0
    for (const s of fg.sizes) { total += s; if (s > max) max = s }
    if (fg.sizes.length > 1 && total > 0 && max / total >= minDominance) {
        const keep = new Set(pos.map((p) => fg.lab[p]).filter((id) => id >= 0))
        const drop = fg.sizes.map((s, id) => s <= islandCells && s < islandFrac * total && !keep.has(id))
        if (drop.some(Boolean)) {
            for (let i = 0; i < field.length; i += 1) {
                const id = fg.lab[i]
                if (id >= 0 && drop[id]) field[i] = -FILL
            }
            islands = drop.filter(Boolean).length
        }
    }

    const area = fieldArea(field)
    if (area > 0) {
        const bg = components(field, side, (v) => v <= T)
        const spare = new Set(neg.map((p) => bg.lab[p]).filter((id) => id >= 0))
        const fill = bg.sizes.map((s, id) => !bg.edge[id] && s < holeFrac * area && !spare.has(id))
        if (fill.some(Boolean)) {
            for (let i = 0; i < field.length; i += 1) {
                const id = bg.lab[i]
                if (id >= 0 && fill[id]) field[i] = FILL
            }
            holes = fill.filter(Boolean).length
        }
    }

    return { islands, holes }
}

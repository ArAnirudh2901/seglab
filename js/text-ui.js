/**
 * text-ui — text-select orchestration (main thread).
 * Phrase → letterboxed detector frame from the original → grounded boxes → ranked
 * candidates in PROXY coords (ready to feed the existing box prompt). No app
 * state or DOM here; app.js owns the input, overlay, and selection glue.
 */
import {
    clippedEdges, clusterObjects, colorEvidenceForBox, dominantColorForBox, DETECTOR_PAD, dropClippedDuplicates, filterToSubject, letterboxPlan, nms, normalizePhrase, rankDetections, scaleBox, shrinkFactor, TILE_OVERLAP, tilePlans, unletterboxBox, YOLOE_INPUT,
} from './text-core.js'
import { expandQuery } from './search-taxonomy.js'
import { getTransform, getBoundedOriginal } from './asset-store.js'
import { detectText, disposeDetector } from './sam-client.js'
import { detectorPlan } from './proxy-plan.js'
import { MAX_SLOTS } from './yoloe-detect.js'

/** A ranked detection → an app-facing candidate, tagged with the dominant
 *  colour bucket in its detector-frame box (the colour sub-class facet).
 *  `d.box` is still detector-square coords; `kx/ky` map it to proxy coords.
 *  `d.rawBox` is square-normalized against `frame` (colour is sampled there). */
const toCandidate = (d, kx, ky) => ({
    box: scaleBox(d.box, kx, ky),
    score: d.score,
    label: d.label,
    // `d.frame` is the CELL this detection came from — with tiling, sampling the
    // full frame would read the wrong pixels for a tile-local box.
    color: (d.rawBox && d.frame ? dominantColorForBox(d.frame, d.rawBox)?.color : null) || null,
})

/** Tiling kicks in once the full-frame pass would shrink the photo by this much.
 *  Below it, a second pass buys nothing but latency. */
const TILE_MIN_SHRINK = 2.5
const TILE_GRID = 2

/** Second, finer grid, tried only when the first pass finds nothing. Measured on
 *  streetlight.jpg (2034², shrink 3.2), best score over all cells:
 *
 *    phrase           full   2x2    3x3
 *    street light     0.111  0.256  0.478
 *    streetlight      0.066  0.147  0.255
 *    person           0.826  0.826  0.826
 *    snowman (absent) 0.005  0.022  0.044
 *
 *  Small subjects sit under the 640² resolution floor at 2x2 and clear it at 3x3;
 *  absent objects stay quiet at every depth, so escalating cannot invent a match.
 *  Gated on shrink >= the grid: past that a tile holds no detail the last pass
 *  did not already have, only upscale. */
const ESCALATE_GRID = 3

/** Draw one source rect into a gray-padded side² RGB frame. `rect` is in the
 *  DECODED source's pixels; `plan` is the cell's own letterbox. */
const frameFromSource = (source, plan, rect) => {
    const { side, dw, dh } = plan
    const canvas = document.createElement('canvas')
    canvas.width = dw
    canvas.height = dh
    let px
    try {
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(source, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, dw, dh)
        px = ctx.getImageData(0, 0, dw, dh).data
    } finally {
        canvas.width = canvas.height = 0 // drop the RGBA backing before the model allocates
    }
    const data = new Uint8ClampedArray(side * side * 3).fill(DETECTOR_PAD)
    for (let y = 0; y < dh; y++) {
        let sp = y * dw * 4
        let d = y * side * 3
        for (let x = 0; x < dw; x++, sp += 4, d += 3) {
            data[d] = px[sp]
            data[d + 1] = px[sp + 1]
            data[d + 2] = px[sp + 2]
        }
    }
    return { data, width: side, height: side, contentWidth: dw, contentHeight: dh }
}

/** Frames for every cell in `cells` (each { ox, oy, ow, oh, plan } in ORIGINAL
 *  pixels). ONE bounded decode serves them all — sized so the smallest cell
 *  still fills the detector square, never the full-res original, and never
 *  above the budget's ceiling (`plan`, from detectorPlan). The cap is derived
 *  from the same grid these cells came from, so in the normal case it does not
 *  bind; it is what stops an unbounded source from reaching the decoder when
 *  the geometry, the device or the pressure level says otherwise. */
const buildFrames = async (cells, tf, plan) => {
    const need = Math.ceil(Math.max(...cells.map((c) => c.plan.side * (Math.max(tf.originalW, tf.originalH) / Math.max(c.ow, c.oh)))))
    const { source, owned } = await getBoundedOriginal({
        maxSide: Math.min(need, plan.maxSide),
        maxMP: plan.maxMP,
    })
    if (!source) return null
    try {
        const kx = source.width / tf.originalW
        const ky = source.height / tf.originalH
        return cells.map((c) => frameFromSource(source, c.plan, {
            sx: c.ox * kx, sy: c.oy * ky, sw: c.ow * kx, sh: c.oh * ky,
        }))
    } finally {
        if (owned) { try { source.close() } catch { /* closed */ } }
    }
}

/**
 * Inference cost of the last search — hardware-fit's detector benchmark, run
 * for free. Accumulated across BOTH passes of an escalated search, and readable
 * even when the search answered "nothing here": that answer is the two-pass
 * case, so dropping it would only ever measure the cheap half. Same channel
 * shape as the mask lane's `lastRun` (sam-client).
 */
let lastDetect = null
export const lastDetectCost = () => lastDetect

/** One detector sweep at `grid` → detections merged into ORIGINAL pixels.
 *  Null when the decode failed; `stale` when a newer request superseded this one. */
const runPass = async (tf, phrases, fallbackLabel, grid, budget, opts) => {
    const plan = planFor(tf, budget, grid)
    const cells = cellsFor(tf, YOLOE_INPUT, plan)
    const frames = await buildFrames(cells, tf, plan)
    if (!frames) return null
    const { results, slotNames, backend, stale, inferMs, cells: ran } = await detectText(frames, phrases, opts)
    if (inferMs > 0 && ran > 0) {
        lastDetect = { inferMs: (lastDetect?.inferMs || 0) + inferMs, cells: (lastDetect?.cells || 0) + ran }
    }
    if (stale) return { stale: true, mapped: [], backend }
    if (!results?.length) return { stale: false, mapped: [], backend }
    const bounds = { w: tf.originalW, h: tf.originalH }
    return { stale: false, backend, mapped: mergeCells(results, cells, frames, slotNames, fallbackLabel, bounds) }
}

/** Prefer candidates that visibly contain a colour named in the prompt. A
 * broad false-positive box may include a few red pixels; it should not beat a
 * tight red-flower box. If the image has no strong colour evidence, leave the
 * detector ranking untouched rather than inventing a rejection. */
const focusRequestedColor = (dets, color) => {
    if (!color || dets.length === 0) return dets
    const withEvidence = dets.map((d) => ({ ...d, colorEvidence: colorEvidenceForBox(d.frame, d.rawBox, color) }))
    const best = withEvidence.reduce((max, d) => Math.max(max, d.colorEvidence), 0)
    if (best < 0.025) return dets
    const floor = Math.max(0.004, best * 0.2)
    return withEvidence
        .filter((d) => d.colorEvidence >= floor)
        .map((d) => ({
            ...d,
            modelScore: d.score,
            // Preserve some model confidence but make visible colour evidence
            // decisive when the user explicitly supplied a colour word.
            score: d.score * (0.15 + 0.85 * (d.colorEvidence / best)),
        }))
}

/** The budget's ceiling for this image at the requested grid. */
const planFor = (tf, budget, grid) => detectorPlan(tf.originalW, tf.originalH, budget || {}, {
    side: YOLOE_INPUT, overlap: TILE_OVERLAP, grid,
})

/** Full frame, plus tiles when the photo is far larger than the detector square
 *  AND the budget affords them (`plan.grid`, which may be below the request). */
const cellsFor = (tf, side, plan) => {
    const full = {
        ox: 0, oy: 0, ow: tf.originalW, oh: tf.originalH,
        plan: letterboxPlan(tf.originalW, tf.originalH, side),
    }
    if (plan.grid < 2 || shrinkFactor(tf.originalW, tf.originalH, side) < TILE_MIN_SHRINK) return [full]
    // Keep the full frame: it is the only pass that can see a subject larger
    // than one tile, and it is what big-object phrases already matched on.
    return [full, ...tilePlans(tf.originalW, tf.originalH, side, { grid: plan.grid, overlap: TILE_OVERLAP })]
}

/** True when a finer grid still has original detail left to give AND the budget
 *  can pay for it — escalating to a grid the plan will clamp back down just
 *  re-runs the pass that already found nothing. */
const canEscalate = (tf, side, budget) => shrinkFactor(tf.originalW, tf.originalH, side) >= ESCALATE_GRID
    && planFor(tf, budget, ESCALATE_GRID).grid >= ESCALATE_GRID

/** Per-cell detections → one list in ORIGINAL pixels, de-duplicated.
 *  `rawBox` and `frame` stay the CELL's, so colour sampling reads the pixels the
 *  detector actually saw. */
const mergeCells = (results, cells, frames, slotNames, fallbackLabel, bounds) => {
    const merged = []
    // letterboxPlan rounds dw/dh to whole pixels, so unletterboxing divides a
    // rounded-up edge back by k and can land a couple of pixels past the frame.
    // Sub-pixel, but a box outside the image is still a box outside the image.
    const clampX = (v) => Math.min(Math.max(v, 0), bounds?.w ?? Infinity)
    const clampY = (v) => Math.min(Math.max(v, 0), bounds?.h ?? Infinity)
    results.forEach((dets, i) => {
        const cell = cells[i]
        if (!cell) return
        for (const d of dets || []) {
            const local = unletterboxBox(d.box, cell.plan)
            if (!local) continue
            const box = [
                clampX(local[0] + cell.ox), clampY(local[1] + cell.oy),
                clampX(local[2] + cell.ox), clampY(local[3] + cell.oy),
            ]
            merged.push({
                box,
                rawBox: d.box,
                frame: frames[i],
                score: d.score,
                // Which edges the TILE cut, so a truncated view of a subject
                // cannot outrank a whole one on score alone.
                clipped: bounds ? clippedEdges(box, cell, bounds) : [],
                label: slotNames?.[d.classIdx] || fallbackLabel,
            })
        }
    })
    // Tiles overlap, so one subject can arrive from several cells. Retire the
    // tile-truncated views of a subject the full frame already saw whole BEFORE
    // NMS — otherwise a fragment with the higher score deletes the complete box.
    return nms(dropClippedDuplicates(merged), 0.55)
}

/**
 * The slot list a phrase expands to. Slot 0 stays the user's own words; the
 * object-only form and taxonomy synonyms follow, so "flower" also reaches
 * "rose"/"tulip" — the expansion CONDITIONS the detector rather than filtering
 * its output, which is what lets an unlisted phrase work at all.
 *
 * DO NOT cut this back to one or two slots to "protect the top-300 budget".
 * That looks like the pad-to-32 pathology and is the opposite of it. Measured on
 * the canonical NEF, `testDetectRaw('flower', 0.05, slots)`:
 *
 *   slots  raw dets  unique boxes  after merge  top score  ms
 *       1        16            16           12      0.504  1826
 *       4        38            17           13      0.702  1656
 *      22       225            37           28      0.940  2713
 *
 * Padding hurt because it repeated ONE embedding, so the head spent its fixed
 * top-300 on exact duplicates of the same box. Distinct synonyms are 22 genuinely
 * different queries: each contributes boxes the others miss, the budget is nowhere
 * near saturated (225 of 1500 across 5 cells), and a specific kind scores far
 * higher on its own pixels than the generic parent does. nms() in mergeCells
 * collapses whatever the synonyms find in common.
 */
export const slotPhrases = (norm) => {
    // Expand the SUBJECT, not the whole phrase: "the dog sitting among the
    // flowers" should reach beagle and husky, not more kinds of flower.
    const expanded = expandQuery(norm.headCore || norm.objectCore)
    return [...new Set([
        norm.core,
        norm.objectCore,
        norm.headCore,
        ...(expanded ? expanded.labels : []),
    ].filter(Boolean))].slice(0, MAX_SLOTS)
}

/** The slots that stand for the SUBJECT of a post-modified phrase — the head
 *  noun and its taxonomy kinds. Null when the phrase is all subject. */
const subjectSlots = (norm) => {
    if (!norm.headCore) return null
    const expanded = expandQuery(norm.headCore)
    return new Set([norm.headCore, ...(expanded ? expanded.labels : [])])
}

/** Raw detections before ranking — diagnostics only. Distinguishes "the model
 *  does not know this phrase" (all scores ~0) from "the object is below the
 *  detector's 640² resolution floor" (real scores, just under threshold).
 *  `slots` overrides the slot list, which is how the slot budget above was
 *  measured rather than guessed. */
export const detectRaw = async (phrase, {
    threshold = 0.001, slots = null, grid = TILE_GRID, budget = {},
} = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null
    const plan = planFor(tf, budget, grid)
    const cells = cellsFor(tf, YOLOE_INPUT, plan)
    const frames = await buildFrames(cells, tf, plan)
    if (!frames) return null
    const phrases = slots?.length ? slots.slice(0, MAX_SLOTS) : slotPhrases(norm)
    const { results, slotNames, backend } = await detectText(frames, phrases, { threshold, idleMs: 0 })
    const flat = (results || []).flat()
    const scores = flat.map((d) => d.score).sort((a, b) => b - a)
    return {
        phrases,
        cells: cells.length,
        grid: plan.grid, // what the budget actually ran, not what was asked for
        sourceMax: plan.maxSide,
        n: scores.length,
        uniqueBoxes: new Set(flat.map((d) => d.box.map((v) => v.toFixed(4)).join(','))).size,
        afterMerge: mergeCells(results || [], cells, frames, slotNames, norm.objectCore, { w: tf.originalW, h: tf.originalH }).length,
        top: scores.slice(0, 4).map((s) => Number(s.toFixed(4))),
        backend,
    }
}

/** Every intermediate stage of one detection, in ORIGINAL pixels — diagnostics
 *  only. detectRaw answers "did the detector see it"; this answers "which
 *  ranking stage dropped or truncated the box it saw". */
export const detectStages = async (phrase, { rankThreshold = 0.08, budget = {} } = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null
    const plan = planFor(tf, budget, TILE_GRID)
    const cells = cellsFor(tf, YOLOE_INPUT, plan)
    const frames = await buildFrames(cells, tf, plan)
    if (!frames) return null
    const phrases = slotPhrases(norm)
    const { results, slotNames } = await detectText(frames, phrases, { threshold: 0.05, idleMs: 0 })
    const bounds = { w: tf.originalW, h: tf.originalH }
    const strip = (list) => list.map((d) => ({
        box: d.box.map((v) => Math.round(v)), score: +d.score.toFixed(3), label: d.label,
    }))
    // Per-cell, before the cross-cell merge.
    const perCell = results.map((dets, i) => ({
        cell: i === 0 ? 'full' : `tile${i}`,
        rect: [cells[i].ox, cells[i].oy, cells[i].ow, cells[i].oh].map((v) => Math.round(v)),
        n: (dets || []).length,
        top: strip((dets || []).map((d) => {
            const local = unletterboxBox(d.box, cells[i].plan)
            return local ? {
                box: [local[0] + cells[i].ox, local[1] + cells[i].oy, local[2] + cells[i].ox, local[3] + cells[i].oy],
                score: d.score, label: slotNames?.[d.classIdx] || norm.objectCore,
            } : null
        }).filter(Boolean)).sort((a, b) => b.score - a.score).slice(0, 6),
    }))
    const mapped = mergeCells(results, cells, frames, slotNames, norm.objectCore, bounds)
    const focused = focusRequestedColor(mapped, norm.color)
    const threshold = norm.color && focused !== mapped ? 0 : rankThreshold
    const above = focused.filter((d) => d.score >= threshold)
    const floor = Math.max(threshold, 0.5 * above.reduce((m, d) => Math.max(m, d.score), 0))
    const afterRelative = above.filter((d) => d.score >= floor)
    const afterNms = nms(afterRelative, 0.5)
    const ranked = rankDetections(focused, { threshold, iou: 0.5, topK: 8, relative: 0.5 })
    const candidates = norm.multi ? ranked : clusterObjects(ranked)
    return {
        norm: { core: norm.core, objectCore: norm.objectCore, headCore: norm.headCore, multi: norm.multi, color: norm.color },
        original: bounds,
        slots: phrases.length,
        perCell,
        afterMerge: strip(mapped).sort((a, b) => b.score - a.score),
        relativeFloor: +floor.toFixed(3),
        afterRelative: strip(afterRelative),
        afterNms: strip(afterNms),
        afterPrune: strip(ranked),
        final: strip(candidates),
    }
}

/**
 * Open-vocabulary candidates for ANY phrase.
 *
 * No label filter and no fallback lane: the detector only ever scores the
 * phrases it was given, so a returned box already IS a match.
 */
export const detectCandidates = async (phrase, {
    rankThreshold = 0.08, idleMs = 0, evict = false, budget = {},
} = {}) => {
    const norm = normalizePhrase(phrase)
    const tf = getTransform()
    if (!norm || !tf) return null
    lastDetect = null
    const phrases = slotPhrases(norm)
    // 0.05, not the 0.25 a closed-vocabulary detector wants. Measured on the
    // canonical NEF: an absent object ("a rusty bicycle") scores NOTHING even at
    // 0.001, while a real but tiny one ("muscari", florets well under the 640²
    // input's resolution) tops out at 0.091. The head is calibrated enough that
    // a low floor costs no false positives, and `relative` below is the real
    // guard — it drops anything far under the best match in THIS image.
    // An escalation is a SECOND detector pass. Under `detectorDispose: 'now'`
    // (idleMs 0) the worker is torn down after each one, so the two passes of a
    // single search would each pay a fresh YOLOE session build. Hold the worker
    // across the pair and dispose once, at the end.
    const mayEscalate = canEscalate(tf, YOLOE_INPUT, budget)
    const hold = mayEscalate && idleMs === 0
    const pass = (grid, keepAlive = false) => runPass(tf, phrases, norm.objectCore, grid, budget, {
        threshold: 0.05, idleMs, evict, keepAlive,
    })
    let first
    // Escalation asks about the SUBJECT only. A setting slot that scores well
    // ("flowers") would otherwise satisfy the bar and skip the finer pass, so a
    // small subject in a rich setting answered "nothing here" without ever
    // looking harder — the exact case the filter below drops.
    const subjectSet = subjectSlots(norm)
    const clearedBar = (r) => (r.mapped || []).some(
        (d) => d.score >= rankThreshold && (!subjectSet || subjectSet.has(d.label)),
    )
    try {
        first = await pass(TILE_GRID, hold)
        if (!first || first.stale) return null
        // Nothing cleared the bar — retry finer before answering "not here". A tiny
        // subject scores under the floor at 2x2 and well over it at 3x3 (ESCALATE_GRID).
        if (!clearedBar(first) && mayEscalate) {
            const deeper = await pass(ESCALATE_GRID)
            if (deeper?.stale) return null
            if (deeper?.mapped.length) first = deeper
        }
    } finally {
        if (hold) disposeDetector() // no-op once the escalated pass disposed it
    }
    const { mapped, backend } = first
    if (mapped.length === 0) return null
    // Setting bleed-through — both halves of it (text-core §filterToSubject).
    const hits = filterToSubject(mapped, subjectSet)
    if (!hits) return null
    const kx = tf.proxyW / tf.originalW
    const ky = tf.proxyH / tf.originalH
    const focused = focusRequestedColor(hits, norm.color)
    const ranked = rankDetections(focused, {
        threshold: norm.color && focused !== hits ? 0 : rankThreshold,
        iou: 0.5, topK: 8, relative: 0.5,
    }).map((d) => toCandidate(d, kx, ky))
    const candidates = norm.multi ? ranked : clusterObjects(ranked)
    return { candidates, multi: norm.multi, backend: `yoloe-text:${backend}`, display: norm.display }
}

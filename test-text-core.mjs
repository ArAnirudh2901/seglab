#!/usr/bin/env bun
/**
 * Unit tests for js/text-core.js — the pure query→prompt rules.
 *
 * These are the decisions that determine text-search accuracy: what counts
 * as a concept, how many instances the user meant, and when the honest
 * answer is "that isn't in this photo". No models, no network, no browser —
 * runs anywhere in milliseconds.
 *
 * Usage: bun test-text-core.mjs
 */
import {
    boxIoU, boxNMS, cropMaskPlane, detectionToPrompt, maskIoU, maskNMS,
    needsZoomRefine, parseQuery, selectDetections, singularize, zoomCropRect,
} from './js/text-core.js'

let pass = 0
let fail = 0
const eq = (label, got, want) => {
    const g = JSON.stringify(got)
    const w = JSON.stringify(want)
    if (g === w) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label}\n    got  ${g}\n    want ${w}`)
}
const ok = (label, cond, detail = '') => {
    if (cond) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label}${detail ? ` — ${detail}` : ''}`)
}

/* ─── singularize ────────────────────────────────────────────────────────── */

eq('singularize: regular', singularize('zebras'), 'zebra')
eq('singularize: -ies', singularize('berries'), 'berry')
eq('singularize: -shes', singularize('bushes'), 'bush')
eq('singularize: -xes', singularize('boxes'), 'box')
eq('singularize: irregular people', singularize('people'), 'person')
eq('singularize: irregular children', singularize('children'), 'child')
eq('singularize: unchanging sheep', singularize('sheep'), 'sheep')
eq('singularize: already singular', singularize('car'), 'car')
// The classic false positives — these are NOT plurals.
eq('singularize: bus stays bus', singularize('bus'), 'bus')
eq('singularize: glass stays glass', singularize('glass'), 'glass')
eq('singularize: lens stays lens', singularize('lens'), 'lens')
eq('singularize: grass stays grass', singularize('grass'), 'grass')

/* ─── parseQuery ─────────────────────────────────────────────────────────── */

eq('parse: bare singular is one instance',
    parseQuery('bicycle'), [{ phrase: 'bicycle', wantsAll: false, raw: 'bicycle' }])

eq('parse: definite singular stays one instance',
    parseQuery('the red bicycle'), [{ phrase: 'red bicycle', wantsAll: false, raw: 'the red bicycle' }])

eq('parse: plural means all',
    parseQuery('zebras'), [{ phrase: 'zebra', wantsAll: true, raw: 'zebras' }])

eq('parse: "all the X" means all, singularized',
    parseQuery('all the zebras'), [{ phrase: 'zebra', wantsAll: true, raw: 'all the zebras' }])

eq('parse: "every X" means all even when singular',
    parseQuery('every car'), [{ phrase: 'car', wantsAll: true, raw: 'every car' }])

eq('parse: comma splits concepts',
    parseQuery('a zebra, the red car').map((p) => p.phrase), ['zebra', 'red car'])

eq('parse: "and" splits concepts',
    parseQuery('all the birds and a tree').map((p) => [p.phrase, p.wantsAll]),
    [['bird', true], ['tree', false]])

eq('parse: attributes survive', parseQuery('dark blue coffee mug')[0].phrase, 'dark blue coffee mug')
eq('parse: compositional phrase survives', parseQuery('person on the left')[0].phrase, 'person on the left')
eq('parse: dedupes repeats', parseQuery('car, a car, the car').length, 1)
eq('parse: whitespace collapse', parseQuery('  red    car  ')[0].phrase, 'red car')
// Only LEADING articles are stripped — an article inside the phrase is part
// of the description ("person on the left") and must survive.
eq('parse: trailing punctuation', parseQuery('where is the dog?')[0].phrase, 'where is the dog')
eq('parse: empty input', parseQuery(''), [])
eq('parse: non-string input', parseQuery(null), [])
eq('parse: only noise', parseQuery('the, a, an'), [])

/* ─── boxIoU ─────────────────────────────────────────────────────────────── */

eq('boxIoU: identical', boxIoU([0, 0, 10, 10], [0, 0, 10, 10]), 1)
eq('boxIoU: disjoint', boxIoU([0, 0, 10, 10], [20, 20, 30, 30]), 0)
eq('boxIoU: touching edges is zero', boxIoU([0, 0, 10, 10], [10, 0, 20, 10]), 0)
eq('boxIoU: half overlap', boxIoU([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
eq('boxIoU: degenerate box', boxIoU([5, 5, 5, 5], [0, 0, 10, 10]), 0)

/* ─── boxNMS ─────────────────────────────────────────────────────────────── */

{
    const dets = [
        { score: 0.9, box: [0, 0, 10, 10] },
        { score: 0.8, box: [1, 1, 11, 11] },   // ~68% IoU with the first
        { score: 0.7, box: [50, 50, 60, 60] }, // disjoint
    ]
    const kept = boxNMS(dets, 0.55)
    eq('boxNMS: suppresses the near-duplicate', kept.map((k) => k.score), [0.9, 0.7])
    eq('boxNMS: keeps both when threshold is loose', boxNMS(dets, 0.9).map((k) => k.score), [0.9, 0.8, 0.7])
    ok('boxNMS: does not mutate input', dets[0].score === 0.9 && dets.length === 3)
}

/* ─── mask planes + maskIoU + maskNMS ────────────────────────────────────── */

/** Build a white-on-black RGBA mask with a filled rect. */
const rectMask = (w, h, x0, y0, x1, y1) => {
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i += 1) rgba[i * 4 + 3] = 255
    for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
            const j = (y * w + x) * 4
            rgba[j] = 255; rgba[j + 1] = 255; rgba[j + 2] = 255
        }
    }
    return rgba
}

{
    const a = cropMaskPlane(rectMask(64, 64, 10, 10, 19, 19), 64, 64, [10, 10, 19, 19])
    eq('cropMaskPlane: dims', [a.w, a.h, a.area, a.x0, a.y0], [10, 10, 100, 10, 10])
    eq('maskIoU: identical planes', maskIoU(a, a), 1)

    const b = cropMaskPlane(rectMask(64, 64, 40, 40, 49, 49), 64, 64, [40, 40, 49, 49])
    eq('maskIoU: disjoint planes', maskIoU(a, b), 0)

    // Half-overlapping: [10..19] vs [15..24] → inter 5×10=50, union 150.
    const c = cropMaskPlane(rectMask(64, 64, 15, 10, 24, 19), 64, 64, [15, 10, 24, 19])
    eq('maskIoU: half overlap', maskIoU(a, c), 50 / 150)

    const kept = maskNMS([
        { score: 0.9, plane: a }, { score: 0.5, plane: c }, { score: 0.8, plane: b },
    ], 0.3)
    eq('maskNMS: drops the overlapping lower score', kept.map((k) => k.score), [0.9, 0.8])
    eq('maskNMS: skips instances with no plane', maskNMS([{ score: 1, plane: null }], 0.5).length, 0)
}

/* ─── selectDetections — the accuracy rules ──────────────────────────────── */

{
    // Absent concept: everything is noise, so the answer is nothing at all.
    const noise = [{ score: 0.06, box: [0, 0, 5, 5] }, { score: 0.04, box: [9, 9, 14, 14] }]
    eq('select: absent concept returns EMPTY', selectDetections(noise, { floor: 0.12 }), [])

    // Twelve equally confident instances all survive.
    const many = Array.from({ length: 12 }, (_, i) => ({ score: 0.8 - i * 0.005, box: [i * 20, 0, i * 20 + 10, 10] }))
    eq('select: all confident instances survive', selectDetections(many, { wantsAll: true }).length, 12)

    // One real hit plus background noise → just the hit.
    const oneReal = [{ score: 0.85, box: [0, 0, 10, 10] }, { score: 0.14, box: [40, 40, 50, 50] }]
    eq('select: relative cut drops weak background', selectDetections(oneReal, { wantsAll: true }).length, 1)

    // Singular intent collapses to the best instance even with many hits.
    eq('select: wantsAll=false keeps only the best',
        selectDetections(many, { wantsAll: false }).map((d) => d.score), [0.8])

    // maxInstances caps runaway detections.
    const flood = Array.from({ length: 60 }, (_, i) => ({ score: 0.9, box: [i * 30, 0, i * 30 + 10, 10] }))
    eq('select: honours maxInstances', selectDetections(flood, { maxInstances: 24 }).length, 24)

    eq('select: empty input', selectDetections([]), [])
    eq('select: non-array input', selectDetections(null), [])

    // A top score exactly at the floor is present, not absent.
    eq('select: score exactly at floor is kept',
        selectDetections([{ score: 0.12, box: [0, 0, 10, 10] }], { floor: 0.12 }).length, 1)
}

/* ─── prompts + zoom crop ────────────────────────────────────────────────── */

eq('detectionToPrompt: centre point is positive',
    detectionToPrompt([10, 20, 30, 60]), { box: [10, 20, 30, 60], clicks: [[20, 40, 1]] })

ok('needsZoomRefine: tiny object in a big frame', needsZoomRefine([0, 0, 30, 30], 6000, 4000))
ok('needsZoomRefine: large subject does not', !needsZoomRefine([0, 0, 3000, 2500], 6000, 4000))

{
    const r = zoomCropRect([100, 100, 140, 140], 1000, 1000, { pad: 0.5 })
    ok('zoomCropRect: contains the box', r[0] <= 100 && r[1] <= 100 && r[2] >= 140 && r[3] >= 140, JSON.stringify(r))
    ok('zoomCropRect: pads outward', r[2] - r[0] > 40, JSON.stringify(r))

    // Clamped at a corner, and must still contain the detection.
    const c = zoomCropRect([0, 0, 20, 20], 1000, 1000, { pad: 1 })
    ok('zoomCropRect: clamps at the frame edge', c[0] === 0 && c[1] === 0, JSON.stringify(c))
    ok('zoomCropRect: stays inside the frame', c[2] <= 1000 && c[3] <= 1000, JSON.stringify(c))

    // A crop bigger than the frame collapses to the frame, not to negatives.
    const big = zoomCropRect([10, 10, 990, 990], 1000, 1000, { pad: 2 })
    ok('zoomCropRect: oversized crop stays in bounds',
        big[0] >= 0 && big[1] >= 0 && big[2] <= 1000 && big[3] <= 1000, JSON.stringify(big))
}

console.log(`\n${fail ? '✗ FAILED' : '✓ PASS'} — ${pass}/${pass + fail} assertions\n`)
process.exit(fail ? 1 : 0)

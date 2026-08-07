#!/usr/bin/env bun
/**
 * Unit tests for js/select-core.js — the unified selection layer.
 *
 * YOLOE-26 runs once and returns every instance; these rules decide which of
 * them a click, a region, or a phrase actually means. That makes this file
 * the accuracy spec for ALL FOUR modes at once — including the case that
 * matters most and is tested least: when the honest answer is "that isn't in
 * this photo". No model, no network, no browser.
 *
 * Usage: bun test-select-core.mjs
 */
import {
    maskContains, maskFractionInBox, maskFractionInPolygon, maskIoU, maskNMS,
    needsZoomRefine, parseQuery, pickByPoint, pickByRegion, pointInPolygon,
    selectByPhrase, selectByQuery, singularize, zoomCropRect,
} from './js/select-core.js'

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

/** A rectangular instance as the engine stores them: bbox-cropped plane. */
const rect = (x0, y0, w, h, { label = 'thing', score = 0.9 } = {}) => ({
    plane: new Uint8Array(w * h).fill(1),
    x0, y0, w, h, area: w * h, label, score,
    box: [x0, y0, x0 + w - 1, y0 + h - 1],
})

/* ─── singularize ────────────────────────────────────────────────────────── */

eq('singularize: regular', singularize('zebras'), 'zebra')
eq('singularize: -ies', singularize('berries'), 'berry')
eq('singularize: -shes', singularize('bushes'), 'bush')
eq('singularize: irregular people', singularize('people'), 'person')
eq('singularize: unchanging sheep', singularize('sheep'), 'sheep')
eq('singularize: already singular', singularize('car'), 'car')
// The classic false positives — these are NOT plurals.
eq('singularize: bus stays bus', singularize('bus'), 'bus')
eq('singularize: glass stays glass', singularize('glass'), 'glass')
eq('singularize: lens stays lens', singularize('lens'), 'lens')

/* ─── parseQuery ─────────────────────────────────────────────────────────── */

eq('parse: definite singular means ONE',
    parseQuery('the red bicycle'), [{ phrase: 'red bicycle', wantsAll: false, raw: 'the red bicycle' }])
eq('parse: plural means ALL', parseQuery('zebras'), [{ phrase: 'zebra', wantsAll: true, raw: 'zebras' }])
eq('parse: "all the X" means all, singularized',
    parseQuery('all the zebras'), [{ phrase: 'zebra', wantsAll: true, raw: 'all the zebras' }])
eq('parse: "every X" means all even when singular',
    parseQuery('every car'), [{ phrase: 'car', wantsAll: true, raw: 'every car' }])
eq('parse: comma splits concepts', parseQuery('a zebra, the red car').map((p) => p.phrase), ['zebra', 'red car'])
eq('parse: "and" splits concepts',
    parseQuery('all the birds and a tree').map((p) => [p.phrase, p.wantsAll]), [['bird', true], ['tree', false]])
eq('parse: attributes survive', parseQuery('dark blue coffee mug')[0].phrase, 'dark blue coffee mug')
eq('parse: dedupes repeats', parseQuery('car, a car, the car').length, 1)
eq('parse: empty input', parseQuery(''), [])
eq('parse: non-string input', parseQuery(null), [])
eq('parse: bare determiners are not concepts', parseQuery('the, a, an'), [])

/* ─── point / region geometry ────────────────────────────────────────────── */

{
    const r = rect(10, 10, 20, 20)
    ok('maskContains: inside', maskContains(r, 15, 15))
    ok('maskContains: outside', !maskContains(r, 5, 5))
    ok('maskContains: just past the far edge', !maskContains(r, 30, 30))
    ok('maskContains: on the near edge', maskContains(r, 10, 10))
}

ok('pointInPolygon: inside', pointInPolygon([[0, 0], [10, 0], [10, 10], [0, 10]], 5, 5))
ok('pointInPolygon: outside', !pointInPolygon([[0, 0], [10, 0], [10, 10], [0, 10]], 15, 5))
ok('pointInPolygon: concave notch is outside',
    !pointInPolygon([[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]], 5, 9))

{
    const r = rect(0, 0, 10, 10)
    eq('maskFractionInBox: fully inside', maskFractionInBox(r, [0, 0, 9, 9]), 1)
    eq('maskFractionInBox: half', maskFractionInBox(r, [0, 0, 4, 9]), 0.5)
    eq('maskFractionInBox: disjoint', maskFractionInBox(r, [50, 50, 60, 60]), 0)
    eq('maskFractionInPolygon: fully inside',
        maskFractionInPolygon(r, [[-1, -1], [20, -1], [20, 20], [-1, 20]]), 1)
    eq('maskFractionInPolygon: disjoint',
        maskFractionInPolygon(r, [[50, 50], [60, 50], [60, 60], [50, 60]]), 0)
}

/* ─── pickByPoint — the click rule ───────────────────────────────────────── */

{
    // A small object sitting inside a big one: clicking the overlap must give
    // the SMALL one, or nested objects become unreachable.
    const big = rect(0, 0, 100, 100, { label: 'person', score: 0.9 })
    const small = rect(40, 40, 10, 10, { label: 'watch', score: 0.7 })
    eq('pickByPoint: smallest containing instance wins',
        pickByPoint([big, small], 45, 45).label, 'watch')
    eq('pickByPoint: outside the small one gives the big one',
        pickByPoint([big, small], 5, 5).label, 'person')
    eq('pickByPoint: nothing there', pickByPoint([big, small], 500, 500), null)
    eq('pickByPoint: empty instance set', pickByPoint([], 5, 5), null)
}

/* ─── pickByRegion — the box/lasso rule ──────────────────────────────────── */

{
    const a = rect(10, 10, 20, 20, { label: 'a' })
    const b = rect(100, 100, 20, 20, { label: 'b' })
    // A box round `a` only.
    eq('pickByRegion: picks the covered instance',
        pickByRegion([a, b], [5, 5, 35, 35]).map((i) => i.label), ['a'])
    // A box covering both.
    eq('pickByRegion: picks every covered instance',
        pickByRegion([a, b], [0, 0, 200, 200]).map((i) => i.label).sort(), ['a', 'b'])
    // A box clipping a sliver of `a` is below minCoverage — but a drawn
    // region is an explicit request, so the best candidate still comes back.
    eq('pickByRegion: a clipping box still returns the best match',
        pickByRegion([a, b], [10, 10, 13, 13]).map((i) => i.label), ['a'])
    eq('pickByRegion: nothing overlapping returns empty',
        pickByRegion([a, b], [300, 300, 400, 400]), [])
    eq('pickByRegion: wantsAll=false keeps one',
        pickByRegion([a, b], [0, 0, 200, 200], { wantsAll: false }).length, 1)
    // Polygon path.
    eq('pickByRegion: polygon region works',
        pickByRegion([a, b], [[0, 0], [40, 0], [40, 40], [0, 40]]).map((i) => i.label), ['a'])
}

/* ─── selectByPhrase — the text rule ─────────────────────────────────────── */

{
    const zebras = Array.from({ length: 12 }, (_, i) =>
        rect(i * 30, 0, 20, 20, { label: 'zebra', score: 0.8 - i * 0.005 }))
    const car = rect(500, 0, 40, 40, { label: 'car', score: 0.85 })
    const noise = rect(600, 0, 10, 10, { label: 'traffic light', score: 0.05 })
    const all = [...zebras, car, noise]

    eq('phrase: absent concept returns EMPTY', selectByPhrase(all, 'giraffe'), [])
    eq('phrase: all confident instances survive', selectByPhrase(all, 'zebra').length, 12)
    eq('phrase: plural query via selectByQuery', selectByQuery(all, 'all the zebras').instances.length, 12)
    eq('phrase: definite singular keeps one', selectByQuery(all, 'the zebra').instances.length, 1)
    eq('phrase: singular concept picks its own instance',
        selectByPhrase(all, 'car').map((i) => i.label), ['car'])
    // Below the absolute floor ⇒ absent, even though the label matches.
    eq('phrase: label match below the floor is still absent',
        selectByPhrase(all, 'traffic light', { floor: 0.25 }), [])
    // Multi-word vocabulary labels are reachable by their head word.
    eq('phrase: multi-word label matched by one word',
        selectByPhrase([rect(0, 0, 10, 10, { label: 'sports car', score: 0.9 })], 'car').length, 1)
    eq('phrase: empty query', selectByPhrase(all, ''), [])

    // Multi-concept union.
    const multi = selectByQuery(all, 'all the zebras and a car')
    eq('query: multi-concept unions the results', multi.instances.length, 13)
    eq('query: concepts are reported', multi.concepts.map((c) => c.phrase), ['zebra', 'car'])
    eq('query: no concept in the phrase', selectByQuery(all, 'the').instances, [])
    ok('query: absent phrase explains itself', !!selectByQuery(all, 'giraffe').reason)
}

/* ─── mask NMS ───────────────────────────────────────────────────────────── */

{
    const a = rect(0, 0, 10, 10, { score: 0.9 })
    const dup = rect(0, 0, 10, 10, { score: 0.5 })
    const far = rect(100, 100, 10, 10, { score: 0.8 })
    eq('maskIoU: identical planes', maskIoU(a, dup), 1)
    eq('maskIoU: disjoint planes', maskIoU(a, far), 0)
    eq('maskNMS: drops the duplicate', maskNMS([a, dup, far], 0.7).map((i) => i.score), [0.9, 0.8])
    eq('maskNMS: skips planeless entries', maskNMS([{ score: 1, plane: null }], 0.5).length, 0)
}

/* ─── DSLR helpers ───────────────────────────────────────────────────────── */

ok('needsZoomRefine: tiny object in a big frame', needsZoomRefine([0, 0, 30, 30], 6000, 4000))
ok('needsZoomRefine: large subject does not', !needsZoomRefine([0, 0, 3000, 2500], 6000, 4000))

{
    const r = zoomCropRect([100, 100, 140, 140], 1000, 1000, { pad: 0.5 })
    ok('zoomCropRect: contains the box', r[0] <= 100 && r[1] <= 100 && r[2] >= 140 && r[3] >= 140, JSON.stringify(r))
    const c = zoomCropRect([0, 0, 20, 20], 1000, 1000, { pad: 1 })
    ok('zoomCropRect: clamps at the frame edge', c[0] === 0 && c[1] === 0, JSON.stringify(c))
    const big = zoomCropRect([10, 10, 990, 990], 1000, 1000, { pad: 2 })
    ok('zoomCropRect: oversized crop stays in bounds',
        big[0] >= 0 && big[1] >= 0 && big[2] <= 1000 && big[3] <= 1000, JSON.stringify(big))
}

console.log(`\n${fail ? '✗ FAILED' : '✓ PASS'} — ${pass}/${pass + fail} assertions\n`)
process.exit(fail ? 1 : 0)

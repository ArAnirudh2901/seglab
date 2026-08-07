#!/usr/bin/env bun
/**
 * Unit tests for js/yoloe-core.js — the raw-tensor decode path.
 *
 * This is the layer most likely to be silently wrong, because a mistake in
 * letterbox arithmetic or tensor layout does not throw; it just puts masks in
 * the wrong place. So the tests build synthetic tensors with known answers
 * and check the geometry exactly, including both output layouts and the
 * non-square DSLR aspect ratios that make padding asymmetric.
 *
 * Usage: bun test-yoloe-core.mjs
 */
import {
    assembleMask, boxIoU, decodeDetections, dedupe, letterbox,
    planeToRGBA, preprocess, unletterboxBox,
} from './js/yoloe-core.js'

let pass = 0
let fail = 0
const eq = (label, got, want) => {
    const g = JSON.stringify(got)
    const w = JSON.stringify(want)
    if (g === w) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label}\n    got  ${g}\n    want ${w}`)
}
const near = (label, got, want, tol = 1e-6) => {
    if (Math.abs(got - want) <= tol) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label} — got ${got}, want ${want} (±${tol})`)
}
const ok = (label, cond, detail = '') => {
    if (cond) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label}${detail ? ` — ${detail}` : ''}`)
}

/* ─── letterbox ──────────────────────────────────────────────────────────── */

{
    // Square source: no padding at all.
    const sq = letterbox(1000, 1000, 640)
    eq('letterbox: square has no padding', [sq.padX, sq.padY], [0, 0])
    near('letterbox: square scale', sq.scale, 0.64)

    // 3:2 DSLR landscape — pads top and bottom only.
    const lb = letterbox(6000, 4000, 640)
    near('letterbox: DSLR scale', lb.scale, 640 / 6000)
    eq('letterbox: DSLR width fills', lb.w, 640)
    eq('letterbox: DSLR height', lb.h, 427)
    eq('letterbox: DSLR pads vertically only', lb.padX, 0)
    ok('letterbox: DSLR vertical padding is centred', Math.abs(lb.padY - (640 - 427) / 2) < 1e-9)

    // Portrait — pads left and right.
    const pt = letterbox(4000, 6000, 640)
    eq('letterbox: portrait height fills', pt.h, 640)
    eq('letterbox: portrait pads horizontally only', pt.padY, 0)
}

/* ─── unletterboxBox — the round trip that must not drift ────────────────── */

{
    const srcW = 6000
    const srcH = 4000
    const lb = letterbox(srcW, srcH, 640)
    // Take a known source box, letterbox it forward by hand, then invert.
    const src = [1200, 800, 2400, 2000]
    const fwd = [
        src[0] * lb.scale + lb.padX, src[1] * lb.scale + lb.padY,
        src[2] * lb.scale + lb.padX, src[3] * lb.scale + lb.padY,
    ]
    const back = unletterboxBox(fwd, lb, srcW, srcH)
    for (let i = 0; i < 4; i += 1) near(`unletterbox: round trip [${i}]`, back[i], src[i], 1e-6)

    // Clamping: a box running off the frame comes back inside it.
    const off = unletterboxBox([-500, -500, 5000, 5000], lb, srcW, srcH)
    ok('unletterbox: clamps to the frame',
        off[0] >= 0 && off[1] >= 0 && off[2] <= srcW - 1 && off[3] <= srcH - 1, JSON.stringify(off))

    // A box entirely inside the PADDING maps outside the image and clamps.
    const pad = unletterboxBox([0, 0, 10, 10], lb, srcW, srcH)
    ok('unletterbox: padding-only box clamps to y=0', pad[1] === 0, JSON.stringify(pad))
}

/* ─── decodeDetections — both tensor layouts ─────────────────────────────── */

const buildOutput0 = ({ numClasses, numMasks, candidates, attrMajor }) => {
    const stride = 4 + numClasses + numMasks
    const n = candidates.length
    const data = new Float32Array(stride * n)
    const put = (attr, i, v) => {
        if (attrMajor) data[attr * n + i] = v
        else data[i * stride + attr] = v
    }
    candidates.forEach((c, i) => {
        put(0, i, c.cx); put(1, i, c.cy); put(2, i, c.w); put(3, i, c.h)
        for (let k = 0; k < numClasses; k += 1) put(4 + k, i, c.scores[k] ?? 0)
        for (let m = 0; m < numMasks; m += 1) put(4 + numClasses + m, i, c.coeffs?.[m] ?? 0)
    })
    return { data, dims: attrMajor ? [1, stride, n] : [1, n, stride] }
}

for (const attrMajor of [true, false]) {
    const tag = attrMajor ? 'attr-major [1,S,N]' : 'candidate-major [1,N,S]'
    const numClasses = 3
    const numMasks = 4
    const lb = letterbox(640, 640, 640) // identity: scale 1, no padding
    const { data, dims } = buildOutput0({
        numClasses,
        numMasks,
        attrMajor,
        candidates: [
            { cx: 100, cy: 100, w: 40, h: 40, scores: [0.9, 0.1, 0.0], coeffs: [1, 0, 0, 0] },
            { cx: 300, cy: 300, w: 60, h: 20, scores: [0.0, 0.8, 0.1], coeffs: [0, 1, 0, 0] },
            { cx: 500, cy: 500, w: 10, h: 10, scores: [0.05, 0.02, 0.01], coeffs: [0, 0, 1, 0] }, // below threshold
        ],
    })
    const dets = decodeDetections(data, dims, {
        numClasses, numMasks, lb, srcW: 640, srcH: 640, confThreshold: 0.25,
    })
    eq(`decode ${tag}: keeps only confident candidates`, dets.length, 2)
    // Scores round-trip through float32, so compare with tolerance.
    ok(`decode ${tag}: sorted by score`,
        Math.abs(dets[0].score - 0.9) < 1e-6 && Math.abs(dets[1].score - 0.8) < 1e-6,
        dets.map((d) => d.score).join(', '))
    eq(`decode ${tag}: best class index`, dets.map((d) => d.classIdx), [0, 1])
    eq(`decode ${tag}: xywh → xyxy`, dets[0].box, [80, 80, 120, 120])
    eq(`decode ${tag}: non-square box`, dets[1].box, [270, 290, 330, 310])
    eq(`decode ${tag}: coefficients extracted`, Array.from(dets[0].coeffs), [1, 0, 0, 0])
}

{
    // A dims triple matching neither axis must fail loudly, not silently
    // decode garbage.
    let threw = false
    try {
        decodeDetections(new Float32Array(10), [1, 7, 9], {
            numClasses: 80, numMasks: 32, lb: letterbox(640, 640, 640), srcW: 640, srcH: 640,
        })
    } catch { threw = true }
    ok('decode: mismatched dims throw rather than guess', threw)
}

{
    // maxDetections caps the flood.
    const numClasses = 1
    const numMasks = 1
    const lb = letterbox(640, 640, 640)
    const candidates = Array.from({ length: 50 }, (_, i) => ({
        cx: 10 + i * 10, cy: 20, w: 8, h: 8, scores: [0.9], coeffs: [1],
    }))
    const { data, dims } = buildOutput0({ numClasses, numMasks, candidates, attrMajor: true })
    const dets = decodeDetections(data, dims, {
        numClasses, numMasks, lb, srcW: 640, srcH: 640, confThreshold: 0.25, maxDetections: 10,
    })
    eq('decode: honours maxDetections', dets.length, 10)
}

/* ─── dedupe (only for non-end-to-end exports) ───────────────────────────── */

{
    const dets = [
        { score: 0.9, box: [0, 0, 10, 10] },
        { score: 0.8, box: [1, 1, 11, 11] },
        { score: 0.7, box: [50, 50, 60, 60] },
    ]
    eq('dedupe: suppresses the near-duplicate', dedupe(dets, 0.55).map((d) => d.score), [0.9, 0.7])
    eq('dedupe: loose threshold keeps all', dedupe(dets, 0.95).length, 3)
    ok('dedupe: does not mutate input', dets.length === 3 && dets[0].score === 0.9)
    eq('boxIoU: half overlap', boxIoU([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
}

/* ─── assembleMask ───────────────────────────────────────────────────────── */

{
    // One prototype that is 1 everywhere: sigmoid(coeff) decides the mask.
    const ph = 32
    const pw = 32
    const numMasks = 2
    const protos = new Float32Array(numMasks * ph * pw)
    protos.fill(1, 0, ph * pw)          // prototype 0 = all ones
    // prototype 1 stays all zeros

    const lb = letterbox(64, 64, 32)     // scale 0.5, no padding
    const box = [10, 10, 29, 29]

    const on = assembleMask(Float32Array.from([10, 0]), protos, { numMasks, ph, pw }, box, lb, 64, 64)
    ok('assembleMask: strong positive coefficient fills the box', !!on && on.area === on.w * on.h,
        on ? `area ${on.area} of ${on.w * on.h}` : 'null')
    eq('assembleMask: plane is cropped to the box', [on.x0, on.y0, on.w, on.h], [10, 10, 20, 20])

    const off = assembleMask(Float32Array.from([-10, 0]), protos, { numMasks, ph, pw }, box, lb, 64, 64)
    eq('assembleMask: strong negative yields nothing', off, null)

    // A degenerate box yields null rather than a zero-sized plane.
    eq('assembleMask: inverted box is rejected',
        assembleMask(Float32Array.from([10, 0]), protos, { numMasks, ph, pw }, [30, 30, 10, 10], lb, 64, 64), null)
}

{
    // Half-and-half prototype: the mask must follow the prototype's geometry,
    // which is what catches a flipped or transposed proto index.
    const ph = 40
    const pw = 40
    const numMasks = 1
    const protos = new Float32Array(ph * pw)
    for (let y = 0; y < ph; y += 1) {
        for (let x = 0; x < pw; x += 1) protos[y * pw + x] = x < pw / 2 ? 10 : -10
    }
    const lb = letterbox(40, 40, 40) // identity
    const m = assembleMask(Float32Array.from([1]), protos, { numMasks, ph, pw }, [0, 0, 39, 39], lb, 40, 40)
    ok('assembleMask: follows prototype geometry (left half on)',
        m && m.area > 0 && Math.abs(m.area - (ph * pw) / 2) <= pw, `area ${m?.area} of ${ph * pw}`)
    // Spot-check orientation explicitly.
    ok('assembleMask: left column is on', m.plane[0 * m.w + 2] === 1)
    ok('assembleMask: right column is off', m.plane[0 * m.w + (m.w - 3)] === 0)
}

/* ─── planeToRGBA ────────────────────────────────────────────────────────── */

{
    const inst = { plane: new Uint8Array(4).fill(1), x0: 2, y0: 3, w: 2, h: 2, area: 4 }
    const rgba = planeToRGBA(inst, 8, 8)
    const at = (x, y) => rgba[(y * 8 + x) * 4]
    eq('planeToRGBA: paints at the offset', [at(2, 3), at(3, 4)], [255, 255])
    eq('planeToRGBA: leaves elsewhere black', at(0, 0), 0)
    eq('planeToRGBA: alpha is opaque', rgba[3], 255)

    // Painting a second instance into the same buffer must not erase the first.
    const b = { plane: new Uint8Array(1).fill(1), x0: 7, y0: 7, w: 1, h: 1, area: 1 }
    planeToRGBA(b, 8, 8, rgba)
    eq('planeToRGBA: composites without erasing', [at(2, 3), at(7, 7)], [255, 255])
}

/* ─── preprocess ─────────────────────────────────────────────────────────── */

{
    const srcW = 4
    const srcH = 2
    const pixels = new Uint8ClampedArray(srcW * srcH * 4)
    for (let i = 0; i < srcW * srcH; i += 1) {
        pixels[i * 4] = 255      // R
        pixels[i * 4 + 1] = 0    // G
        pixels[i * 4 + 2] = 0    // B
        pixels[i * 4 + 3] = 255
    }
    const size = 8
    const { tensor, lb, dims } = preprocess(pixels, srcW, srcH, size)
    eq('preprocess: NCHW dims', dims, [1, 3, size, size])
    eq('preprocess: tensor length', tensor.length, 3 * size * size)

    const plane = size * size
    // Centre of the content area: pure red, normalized.
    const cx = Math.floor(size / 2)
    const cy = Math.floor(lb.padY + lb.h / 2)
    const idx = cy * size + cx
    near('preprocess: R channel normalized', tensor[idx], 1)
    near('preprocess: G channel normalized', tensor[plane + idx], 0)

    // A row inside the padding must be the 114/255 letterbox grey.
    const padIdx = 0 * size + 0
    near('preprocess: padding is 114/255', tensor[padIdx], 114 / 255, 1e-6)

    ok('preprocess: every value is in [0,1]', tensor.every((v) => v >= 0 && v <= 1))
}

console.log(`\n${fail ? '✗ FAILED' : '✓ PASS'} — ${pass}/${pass + fail} assertions\n`)
process.exit(fail ? 1 : 0)

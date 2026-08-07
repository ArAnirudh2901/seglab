#!/usr/bin/env bun
/**
 * Correctness gate for the optimized post pipeline.
 *
 * `edge-refine.js` and `mask-core.js` were rewritten for speed: the guided
 * filter's image-only terms are hoisted out per image, both stages are
 * confined to the mask's bounding box, morphology became a separable box
 * count, per-pixel `Set` lookups became label-indexed LUTs, and every working
 * buffer is pooled across a query's N instances. That is a lot of moving
 * parts for a ~15–29× speedup, so this file pins the result: the fast code
 * must agree EXACTLY with a naive, obviously-correct reference.
 *
 * The reference below is deliberately dumb — full-frame passes, sliding
 * min/max morphology, `Set`-based component bookkeeping. It is the spec.
 * Any disagreement is a bug in the fast path, not a tolerance to widen.
 *
 * Usage: bun test-post-pipeline.mjs
 */
import { buildGuide, makeScratch, refineMaskEdges } from './js/edge-refine.js'
import { cleanupMaskRGBA, countMaskComponents, makeMaskScratch } from './js/mask-core.js'

/* ─── Reference implementation (naive on purpose) ────────────────────────── */

const refMorph = (src, w, h, radius, isMax) => {
    const tmp = new Float32Array(src.length)
    const out = new Float32Array(src.length)
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            let v = isMax ? 0 : 1
            for (let i = Math.max(0, x - radius); i <= Math.min(w - 1, x + radius); i += 1) {
                const s = src[row + i]
                v = isMax ? (s > v ? s : v) : (s < v ? s : v)
            }
            tmp[row + x] = v
        }
    }
    for (let x = 0; x < w; x += 1) {
        for (let y = 0; y < h; y += 1) {
            let v = isMax ? 0 : 1
            for (let i = Math.max(0, y - radius); i <= Math.min(h - 1, y + radius); i += 1) {
                const s = tmp[i * w + x]
                v = isMax ? (s > v ? s : v) : (s < v ? s : v)
            }
            out[y * w + x] = v
        }
    }
    return out
}

const refIntegral = (src, w, h) => {
    const sat = new Float64Array((w + 1) * (h + 1))
    for (let y = 0; y < h; y += 1) {
        let rowSum = 0
        for (let x = 0; x < w; x += 1) {
            rowSum += src[y * w + x]
            sat[(y + 1) * (w + 1) + x + 1] = rowSum + sat[y * (w + 1) + x + 1]
        }
    }
    return sat
}

const refBoxMean = (sat, w, h, radius, out) => {
    const W = w + 1
    for (let y = 0; y < h; y += 1) {
        const y0 = Math.max(0, y - radius)
        const y1 = Math.min(h - 1, y + radius)
        for (let x = 0; x < w; x += 1) {
            const x0 = Math.max(0, x - radius)
            const x1 = Math.min(w - 1, x + radius)
            const sum = sat[(y1 + 1) * W + x1 + 1] - sat[(y1 + 1) * W + x0] - sat[y0 * W + x1 + 1] + sat[y0 * W + x0]
            out[y * w + x] = sum / ((y1 - y0 + 1) * (x1 - x0 + 1))
        }
    }
    return out
}

const refRefine = (rgba, w, h, gray, { band = 6, radius = 8, eps = 1e-3 } = {}) => {
    const size = w * h
    if (!gray || gray.length !== size) return { bandPixels: 0 }
    const p = new Float32Array(size)
    for (let i = 0; i < size; i += 1) p[i] = rgba[i * 4] >= 128 ? 1 : 0
    const dil = refMorph(p, w, h, band, true)
    const ero = refMorph(p, w, h, band, false)
    let bandPixels = 0
    for (let i = 0; i < size; i += 1) if (dil[i] > 0.5 && ero[i] < 0.5) bandPixels += 1
    if (!bandPixels) return { bandPixels: 0 }

    const meanI = refBoxMean(refIntegral(gray, w, h), w, h, radius, new Float32Array(size))
    const meanP = refBoxMean(refIntegral(p, w, h), w, h, radius, new Float32Array(size))
    const Ip = new Float32Array(size)
    const II = new Float32Array(size)
    for (let i = 0; i < size; i += 1) { Ip[i] = gray[i] * p[i]; II[i] = gray[i] * gray[i] }
    const meanIp = refBoxMean(refIntegral(Ip, w, h), w, h, radius, Ip)
    const meanII = refBoxMean(refIntegral(II, w, h), w, h, radius, II)
    const a = new Float32Array(size)
    const b = new Float32Array(size)
    for (let i = 0; i < size; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        a[i] = covIp / (varI + eps)
        b[i] = meanP[i] - a[i] * meanI[i]
    }
    const meanA = refBoxMean(refIntegral(a, w, h), w, h, radius, a)
    const meanB = refBoxMean(refIntegral(b, w, h), w, h, radius, b)
    for (let i = 0; i < size; i += 1) {
        if (!(dil[i] > 0.5 && ero[i] < 0.5)) continue
        let q = meanA[i] * gray[i] + meanB[i]
        if (q < 0) q = 0
        else if (q > 1) q = 1
        let v = Math.round(q * 255)
        if (v < 10) v = 0
        else if (v > 245) v = 255
        const j = i * 4
        rgba[j] = v; rgba[j + 1] = v; rgba[j + 2] = v
    }
    return { bandPixels }
}

const refLabel = (bin, w, h) => {
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

const refNearSeed = (labels, w, h, x, y, radius = 8) => {
    const cx = Math.round(x)
    const cy = Math.round(y)
    for (let r = 0; r <= radius; r += 1) {
        for (let dy = -r; dy <= r; dy += 1) {
            for (let dx = -r; dx <= r; dx += 1) {
                if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
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

const refClean = (rgba, w, h, seeds = []) => {
    const size = w * h
    const bin = new Uint8Array(size)
    let fgArea = 0
    for (let i = 0; i < size; i += 1) if (rgba[i * 4] >= 128) { bin[i] = 1; fgArea += 1 }
    if (!fgArea) return { kept: 0, dropped: 0, holesFilled: 0 }
    const { labels, areas } = refLabel(bin, w, h)
    const keep = new Set()
    for (const [sx, sy] of seeds) {
        const l = refNearSeed(labels, w, h, sx, sy)
        if (l) keep.add(l)
    }
    if (keep.size === 0) {
        let best = 1
        for (let k = 1; k < areas.length; k += 1) if (areas[k] > areas[best - 1]) best = k + 1
        keep.add(best)
    }
    let largestKept = 0
    for (const l of keep) largestKept = Math.max(largestKept, areas[l - 1])
    const minUnseeded = Math.max(48, largestKept * 0.01)
    for (let k = 0; k < areas.length; k += 1) if (!keep.has(k + 1) && areas[k] >= minUnseeded) keep.add(k + 1)
    let dropped = 0
    for (let i = 0; i < size; i += 1) {
        if (bin[i] && !keep.has(labels[i])) {
            bin[i] = 0
            dropped += 1
            const j = i * 4
            rgba[j] = 0; rgba[j + 1] = 0; rgba[j + 2] = 0
        }
    }
    const inv = new Uint8Array(size)
    for (let i = 0; i < size; i += 1) inv[i] = bin[i] ? 0 : 1
    const bg = refLabel(inv, w, h)
    const touches = new Set()
    for (let x = 0; x < w; x += 1) {
        if (bg.labels[x]) touches.add(bg.labels[x])
        if (bg.labels[(h - 1) * w + x]) touches.add(bg.labels[(h - 1) * w + x])
    }
    for (let y = 0; y < h; y += 1) {
        if (bg.labels[y * w]) touches.add(bg.labels[y * w])
        if (bg.labels[y * w + w - 1]) touches.add(bg.labels[y * w + w - 1])
    }
    const maxHole = Math.max(64, (fgArea - dropped) * 0.01)
    const fillLabel = new Set()
    for (let k = 0; k < bg.areas.length; k += 1) {
        if (!touches.has(k + 1) && bg.areas[k] <= maxHole) fillLabel.add(k + 1)
    }
    let holesFilled = 0
    if (fillLabel.size) {
        for (let i = 0; i < size; i += 1) {
            if (fillLabel.has(bg.labels[i])) {
                holesFilled += 1
                const j = i * 4
                rgba[j] = 255; rgba[j + 1] = 255; rgba[j + 2] = 255
            }
        }
    }
    return { kept: keep.size, dropped, holesFilled }
}

/* ─── Fixtures ───────────────────────────────────────────────────────────── */

let seed = 0x2f6e2b1
const rnd = () => {
    seed ^= seed << 13; seed >>>= 0
    seed ^= seed >> 17
    seed ^= seed << 5; seed >>>= 0
    return seed / 0x100000000
}

const makeGray = (w, h, kind) => {
    const g = new Float32Array(w * h)
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            const i = y * w + x
            if (kind === 'flat') g[i] = 0.5
            else if (kind === 'noise') g[i] = rnd()
            else g[i] = Math.min(1, Math.max(0, 0.4 + 0.35 * Math.sin(x / 31) * Math.cos(y / 27) + 0.05 * rnd()))
        }
    }
    return g
}

/** Shapes chosen to hit the edge cases: borders, corners, holes, crumbs. */
const makeMask = (w, h, kind) => {
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i += 1) rgba[i * 4 + 3] = 255
    const set = (x, y, v) => {
        if (x < 0 || y < 0 || x >= w || y >= h) return
        const j = (Math.round(y) * w + Math.round(x)) * 4
        rgba[j] = v; rgba[j + 1] = v; rgba[j + 2] = v
    }
    if (kind === 'empty') return rgba
    if (kind === 'full') { for (let i = 0; i < w * h; i += 1) { rgba[i * 4] = 255; rgba[i * 4 + 1] = 255; rgba[i * 4 + 2] = 255 } return rgba }
    if (kind === 'single') { set(w >> 1, h >> 1, 255); return rgba }
    if (kind === 'border') { for (let y = 0; y < h; y += 1) for (let x = 0; x < w / 3; x += 1) set(x, y, 255); return rgba }
    if (kind === 'corner') { for (let y = 0; y < h / 3; y += 1) for (let x = 0; x < w / 3; x += 1) set(x, y, 255); return rgba }
    if (kind === 'spanning') { for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) if (y > h / 3 && y < 2 * h / 3) set(x, y, 255); return rgba }
    if (kind === 'blobs') {
        for (let k = 0; k < 5; k += 1) {
            const cx = rnd() * w
            const cy = rnd() * h
            const r = 4 + rnd() * (Math.min(w, h) / 5)
            for (let y = Math.floor(cy - r); y <= cy + r; y += 1) {
                for (let x = Math.floor(cx - r); x <= cx + r; x += 1) {
                    if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) set(x, y, 255)
                }
            }
        }
        for (let k = 0; k < 60; k += 1) set(Math.floor(rnd() * w), Math.floor(rnd() * h), 255)
        return rgba
    }
    const cx = w / 2
    const cy = h / 2
    const R = Math.min(w, h) * 0.4
    for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= R * R) set(x, y, 255)
    }
    for (const [ox, oy, hr] of [[-R / 3, 0, 2], [R / 3, 0, R / 6], [0, R / 2, 1]]) {
        for (let y = Math.floor(cy + oy - hr); y <= cy + oy + hr; y += 1) {
            for (let x = Math.floor(cx + ox - hr); x <= cx + ox + hr; x += 1) {
                if ((x - cx - ox) ** 2 + (y - cy - oy) ** 2 <= hr * hr) set(x, y, 0)
            }
        }
    }
    return rgba
}

/* ─── Compare ────────────────────────────────────────────────────────────── */

const diff = (a, b) => {
    if (a.length !== b.length) return `length ${a.length} vs ${b.length}`
    let n = 0
    let worst = 0
    let at = -1
    for (let i = 0; i < a.length; i += 1) {
        const d = Math.abs(a[i] - b[i])
        if (d !== 0) { n += 1; if (d > worst) { worst = d; at = i } }
    }
    return n ? `${n} px differ, max Δ${worst} at idx ${at} (px ${Math.floor(at / 4)})` : null
}

let fails = 0
let checks = 0
const shapes = ['blobs', 'donut', 'border', 'corner', 'spanning', 'single', 'empty', 'full']
const grays = ['photo', 'flat', 'noise']
const dims = [[64, 64], [128, 96], [257, 131], [512, 512]]

console.log('\npost pipeline — fast path vs naive reference (must be EXACT)\n')

for (const [w, h] of dims) {
    for (const shape of shapes) {
        for (const gk of grays) {
            const gray = makeGray(w, h, gk)
            const seeds = [[w / 2, h / 2], [w / 3, h / 3]]

            // cleanupMaskRGBA — with and without a pooled scratch.
            const cRef = makeMask(w, h, shape)
            const cFast = cRef.slice()
            const cFast2 = cRef.slice()
            const rRef = refClean(cRef, w, h, seeds)
            const rFast = cleanupMaskRGBA(cFast, w, h, seeds)
            const rFast2 = cleanupMaskRGBA(cFast2, w, h, seeds, makeMaskScratch(w, h))
            checks += 2
            for (const [tag, got, res] of [['plain', cFast, rFast], ['scratch', cFast2, rFast2]]) {
                const bad = diff(cRef, got)
                if (bad) { console.log(`✗ clean/${tag} ${w}×${h} ${shape}/${gk}: ${bad}`); fails += 1 }
                else if (res.dropped !== rRef.dropped || res.holesFilled !== rRef.holesFilled || res.kept !== rRef.kept) {
                    console.log(`✗ clean/${tag} stats ${w}×${h} ${shape}/${gk}: ${JSON.stringify(res)} vs ${JSON.stringify(rRef)}`)
                    fails += 1
                }
            }

            // refineMaskEdges — raw-gray path and prebuilt-guide path.
            const fRef = makeMask(w, h, shape)
            const fA = fRef.slice()
            const fB = fRef.slice()
            const bRef = refRefine(fRef, w, h, gray)
            const bA = refineMaskEdges(fA, w, h, gray)
            const bB = refineMaskEdges(fB, w, h, buildGuide(gray, w, h, 8), { scratch: makeScratch() })
            checks += 2
            for (const [tag, got, band] of [['gray', fA, bA], ['guide', fB, bB]]) {
                const bad = diff(fRef, got)
                if (bad) { console.log(`✗ refine/${tag} ${w}×${h} ${shape}/${gk}: ${bad}`); fails += 1 }
                else if (band.bandPixels !== bRef.bandPixels) {
                    console.log(`✗ refine/${tag} bandPixels ${w}×${h} ${shape}/${gk}: ${band.bandPixels} vs ${bRef.bandPixels}`)
                    fails += 1
                }
            }

            checks += 1
            if (countMaskComponents(fRef, w, h) !== refLabel(
                Uint8Array.from({ length: w * h }, (_, i) => (fRef[i * 4] >= 128 ? 1 : 0)), w, h,
            ).areas.length) {
                console.log(`✗ countMaskComponents ${w}×${h} ${shape}/${gk}`)
                fails += 1
            }
        }
    }
}

// Pooled scratch reused across DIFFERENT shapes must not leak state.
{
    const w = 256
    const h = 256
    const gray = makeGray(w, h, 'photo')
    const guide = buildGuide(gray, w, h, 8)
    const sc = makeScratch()
    const ms = makeMaskScratch(w, h)
    for (const shape of ['blobs', 'donut', 'corner', 'single', 'spanning', 'blobs']) {
        const a = makeMask(w, h, shape)
        const b = a.slice()
        refClean(a, w, h, [[w / 2, h / 2]])
        refRefine(a, w, h, gray)
        const hy = cleanupMaskRGBA(b, w, h, [[w / 2, h / 2]], ms)
        refineMaskEdges(b, w, h, guide, { scratch: sc, bbox: hy.bbox })
        checks += 1
        const bad = diff(a, b)
        if (bad) { console.log(`✗ scratch-reuse ${shape}: ${bad}`); fails += 1 }
    }
}

// The bbox returned by hygiene must be the true extent of what survived.
{
    const w = 128
    const h = 128
    const m = makeMask(w, h, 'blobs')
    const hy = cleanupMaskRGBA(m, w, h, [[w / 2, h / 2]], null)
    let minX = w
    let minY = h
    let maxX = -1
    let maxY = -1
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            if (m[(y * w + x) * 4] >= 128) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }
    checks += 1
    const want = maxX >= 0 ? [minX, minY, maxX, maxY] : null
    if (JSON.stringify(hy.bbox) !== JSON.stringify(want)) {
        console.log(`✗ hygiene bbox: ${JSON.stringify(hy.bbox)} vs ${JSON.stringify(want)}`)
        fails += 1
    }
}

console.log(`\n${fails ? '✗ FAILED' : '✓ PASS'} — ${checks - fails}/${checks} checks exact\n`)
process.exit(fails ? 1 : 0)

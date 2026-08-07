/**
 * edge-refine — guided-filter boundary snapping (pure, no DOM)
 * --------------------------------------------------------------
 * SAM-family masks decode at 256×256 and get bilinearly upsampled, so the
 * boundary is a ~4px staircase that ignores the actual image edges. This
 * pass re-derives the edge FROM THE IMAGE: a guided filter (He et al.) with
 * the grayscale photo as guide, applied only inside a ±band around the mask
 * boundary. Inside the band the mask becomes soft (0..255) and hugs image
 * gradients; outside it stays exactly the decoder's binary decision.
 *
 * Three structural properties keep this cheap enough to run on EVERY mask,
 * which matters because text prompts produce N masks per query, not one:
 *
 *   1. GUIDE INVARIANTS ARE PER-IMAGE, NOT PER-MASK. mean(I) and var(I)
 *      depend only on the photo, so `buildGuide` computes them once per
 *      image and every mask on that image reuses them (see sam-engine's
 *      embedding cache). Only mean(p) and mean(I·p) are per-mask.
 *   2. WORK IS CONFINED TO THE MASK'S BBOX. The band lives within
 *      bbox ⊕ band, so everything runs on that rect dilated by band + 2·radius
 *      (the guided filter's reach) instead of the full frame. A 200px object
 *      in a 1024² frame costs ~5% of a full-frame pass.
 *   3. SCRATCH BUFFERS ARE POOLED. A `makeScratch()` pool is reused across
 *      instances, so N masks allocate once rather than N times — the
 *      allocation churn, not the arithmetic, was the memory spike.
 *
 * All passes are O(N) over typed arrays: integral-image box filters, and a
 * separable box COUNT for the boundary band (for a strictly 0/1 map,
 * Chebyshev dilate ≡ count>0 and erode ≡ count==windowArea, so one box sum
 * replaces both morphological passes and drops the old O(N·radius) sliding
 * min/max entirely).
 */

/* ─── O(N) primitives ────────────────────────────────────────────────────── */

/**
 * Separable box COUNT of a 0/1 Float32 map over the clamped (2r+1)² window,
 * plus the window area at each pixel. `count === area` ⇒ erode, `count > 0`
 * ⇒ dilate, so the boundary band is `0 < count < area`.
 */
const boxCount = (src, w, h, r, sum, area) => {
    const row = new Float32Array(w * h)
    // Horizontal running sum.
    for (let y = 0; y < h; y += 1) {
        const off = y * w
        let acc = 0
        for (let x = 0; x <= Math.min(w - 1, r); x += 1) acc += src[off + x]
        for (let x = 0; x < w; x += 1) {
            row[off + x] = acc
            const add = x + r + 1
            const drop = x - r
            if (add < w) acc += src[off + add]
            if (drop >= 0) acc -= src[off + drop]
        }
    }
    // Vertical running sum.
    for (let x = 0; x < w; x += 1) {
        let acc = 0
        for (let y = 0; y <= Math.min(h - 1, r); y += 1) acc += row[y * w + x]
        for (let y = 0; y < h; y += 1) {
            sum[y * w + x] = acc
            const add = y + r + 1
            const drop = y - r
            if (add < h) acc += row[add * w + x]
            if (drop >= 0) acc -= row[drop * w + x]
        }
    }
    // Clamped window areas (separable).
    for (let y = 0; y < h; y += 1) {
        const hy = Math.min(h - 1, y + r) - Math.max(0, y - r) + 1
        const off = y * w
        for (let x = 0; x < w; x += 1) {
            area[off + x] = hy * (Math.min(w - 1, x + r) - Math.max(0, x - r) + 1)
        }
    }
}

/** Integral image (summed-area table) of a Float32 map, into a (w+1)×(h+1). */
const integralInto = (src, w, h, sat) => {
    sat.fill(0, 0, (w + 1) * (h + 1))
    for (let y = 0; y < h; y += 1) {
        let rowSum = 0
        const srcRow = y * w
        const satRow = (y + 1) * (w + 1)
        const satPrev = y * (w + 1)
        for (let x = 0; x < w; x += 1) {
            rowSum += src[srcRow + x]
            sat[satRow + x + 1] = rowSum + sat[satPrev + x + 1]
        }
    }
    return sat
}

/** Box-mean via SAT, clamped windows at the borders. Writes into `out`. */
const boxMeanFrom = (sat, w, h, radius, out) => {
    const W = w + 1
    for (let y = 0; y < h; y += 1) {
        const y0 = Math.max(0, y - radius)
        const y1 = Math.min(h - 1, y + radius)
        const rowTop = y0 * W
        const rowBot = (y1 + 1) * W
        const invH = y1 - y0 + 1
        for (let x = 0; x < w; x += 1) {
            const x0 = Math.max(0, x - radius)
            const x1 = Math.min(w - 1, x + radius)
            const sum = sat[rowBot + x1 + 1] - sat[rowBot + x0] - sat[rowTop + x1 + 1] + sat[rowTop + x0]
            out[y * w + x] = sum / (invH * (x1 - x0 + 1))
        }
    }
    return out
}

/* ─── Scratch pool ───────────────────────────────────────────────────────── */

/**
 * Reusable working buffers. Pass the same scratch object across every mask of
 * one query and the pipeline allocates once instead of N times. Buffers grow
 * to the largest rect seen and are then reused as-is.
 */
export const makeScratch = () => ({ f32: new Map(), f64: null, f64Len: 0 })

const takeF32 = (scratch, name, len) => {
    if (!scratch) return new Float32Array(len)
    const held = scratch.f32.get(name)
    if (held && held.length >= len) return held
    const fresh = new Float32Array(len)
    scratch.f32.set(name, fresh)
    return fresh
}

const takeSat = (scratch, len) => {
    if (!scratch) return new Float64Array(len)
    if (scratch.f64 && scratch.f64.length >= len) return scratch.f64
    scratch.f64 = new Float64Array(len)
    scratch.f64Len = len
    return scratch.f64
}

/* ─── Per-image guide ────────────────────────────────────────────────────── */

/**
 * Precompute the guided filter's image-only terms. Depends solely on the
 * photo, so this runs ONCE per image and is shared by every mask decoded on
 * it (cache it next to the embeddings). ~3 floats per pixel: 12 MB at 1024².
 *
 * @param {Float32Array} gray  0..1 grayscale, w*h
 * @returns {{ gray, w, h, radius, meanI, varI }}
 */
export const buildGuide = (gray, w, h, radius = 8) => {
    const size = w * h
    if (!gray || gray.length !== size) return null
    const sat = new Float64Array((w + 1) * (h + 1))
    const meanI = boxMeanFrom(integralInto(gray, w, h, sat), w, h, radius, new Float32Array(size))
    const II = new Float32Array(size)
    for (let i = 0; i < size; i += 1) II[i] = gray[i] * gray[i]
    const meanII = boxMeanFrom(integralInto(II, w, h, sat), w, h, radius, II)
    const varI = meanII // reuse: meanII is dead after this
    for (let i = 0; i < size; i += 1) {
        const m = meanI[i]
        varI[i] = meanII[i] - m * m
    }
    return { gray, w, h, radius, meanI, varI }
}

const isGuide = (g) => !!g && typeof g === 'object' && !ArrayBuffer.isView(g) && !!g.meanI

/** Tight bbox of the mask's foreground, or null when empty. */
const maskBBox = (rgba, w, h) => {
    let minX = w
    let minY = h
    let maxX = -1
    let maxY = -1
    for (let y = 0; y < h; y += 1) {
        const off = y * w
        for (let x = 0; x < w; x += 1) {
            if (rgba[(off + x) * 4] >= 128) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }
    return maxX >= 0 ? [minX, minY, maxX, maxY] : null
}

/* ─── Refinement ─────────────────────────────────────────────────────────── */

/**
 * Refine a mask's boundary band in place. `rgba` is the white-on-black mask
 * (modified: band pixels become soft 0..255).
 *
 * `guideOrGray` accepts either a `buildGuide(...)` result (preferred — the
 * per-image terms are then already paid for) or a raw Float32 grayscale, in
 * which case the guide is built on the spot for backwards compatibility.
 *
 * `bbox` skips the foreground scan when the caller already knows it (the
 * detector hands one over per instance). `scratch` is a `makeScratch()` pool
 * shared across a multi-instance query.
 *
 * @returns {{ bandPixels: number }} how many pixels were refined
 */
export const refineMaskEdges = (rgba, w, h, guideOrGray, { band = 6, radius = 8, eps = 1e-3, bbox = null, scratch = null } = {}) => {
    const guide = isGuide(guideOrGray) ? guideOrGray : buildGuide(guideOrGray, w, h, radius)
    if (!guide || guide.w !== w || guide.h !== h) return { bandPixels: 0 }
    const r = guide.radius

    const box = bbox || maskBBox(rgba, w, h)
    if (!box) return { bandPixels: 0 }

    // The band lives in bbox ⊕ band; the guided filter reaches 2·radius
    // further (mean of a/b, each itself a mean). Inside this rect every value
    // we actually read back is bit-identical to a full-frame pass.
    const margin = band + 2 * r
    const rx0 = Math.max(0, box[0] - margin)
    const ry0 = Math.max(0, box[1] - margin)
    const rx1 = Math.min(w - 1, box[2] + margin)
    const ry1 = Math.min(h - 1, box[3] + margin)
    const rw = rx1 - rx0 + 1
    const rh = ry1 - ry0 + 1
    const n = rw * rh

    // Mask as 0/1 over the rect.
    const p = takeF32(scratch, 'p', n)
    for (let y = 0; y < rh; y += 1) {
        const src = (ry0 + y) * w + rx0
        const dst = y * rw
        for (let x = 0; x < rw; x += 1) p[dst + x] = rgba[(src + x) * 4] >= 128 ? 1 : 0
    }

    // Boundary band: 0 < boxCount < windowArea.
    const cnt = takeF32(scratch, 'cnt', n)
    const area = takeF32(scratch, 'area', n)
    boxCount(p, rw, rh, band, cnt, area)
    let bandPixels = 0
    for (let i = 0; i < n; i += 1) {
        if (cnt[i] > 0 && cnt[i] < area[i]) bandPixels += 1
    }
    if (!bandPixels) return { bandPixels: 0 }

    // Per-mask terms only — mean(I) and var(I) came from the guide.
    const sat = takeSat(scratch, (rw + 1) * (rh + 1))
    const meanP = boxMeanFrom(integralInto(p, rw, rh, sat), rw, rh, r, takeF32(scratch, 'meanP', n))
    const Ip = takeF32(scratch, 'Ip', n)
    for (let y = 0; y < rh; y += 1) {
        const src = (ry0 + y) * w + rx0
        const dst = y * rw
        for (let x = 0; x < rw; x += 1) Ip[dst + x] = guide.gray[src + x] * p[dst + x]
    }
    const meanIp = boxMeanFrom(integralInto(Ip, rw, rh, sat), rw, rh, r, Ip)

    const a = takeF32(scratch, 'a', n)
    const b = takeF32(scratch, 'b', n)
    for (let y = 0; y < rh; y += 1) {
        const src = (ry0 + y) * w + rx0
        const dst = y * rw
        for (let x = 0; x < rw; x += 1) {
            const i = dst + x
            const g = src + x
            const mI = guide.meanI[g]
            const av = (meanIp[i] - mI * meanP[i]) / (guide.varI[g] + eps)
            a[i] = av
            b[i] = meanP[i] - av * mI
        }
    }
    const meanA = boxMeanFrom(integralInto(a, rw, rh, sat), rw, rh, r, a)
    const meanB = boxMeanFrom(integralInto(b, rw, rh, sat), rw, rh, r, b)

    for (let y = 0; y < rh; y += 1) {
        const src = (ry0 + y) * w + rx0
        const dst = y * rw
        for (let x = 0; x < rw; x += 1) {
            const i = dst + x
            if (!(cnt[i] > 0 && cnt[i] < area[i])) continue
            let q = meanA[i] * guide.gray[src + x] + meanB[i]
            if (q < 0) q = 0
            else if (q > 1) q = 1
            // Snap near-extremes so the band doesn't carry a faint fog.
            let v = Math.round(q * 255)
            if (v < 10) v = 0
            else if (v > 245) v = 255
            const j = (src + x) * 4
            rgba[j] = v
            rgba[j + 1] = v
            rgba[j + 2] = v
        }
    }
    return { bandPixels }
}

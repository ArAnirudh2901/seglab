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
 * This is the CPU reference and fallback. When WebGPU is available the same
 * math runs as compute shaders (gpu-post.js) — one thread per pixel instead
 * of one thread per frame. Both paths must stay numerically equivalent.
 *
 * Every pass here is genuinely O(N) in the pixel count: integral-image box
 * filters, and sliding-window-count morphology (the mask is binary, so a
 * dilate is "window contains a 1" and an erode is "window is all 1s" — both
 * maintainable incrementally, no per-pixel radius loop).
 */

/**
 * Separable Chebyshev dilate (max) or erode (min) of a 0/1 Float32 map.
 * O(N) regardless of radius: each row/column keeps a running count of set
 * pixels in its clamped window and slides it one step at a time.
 */
const morph = (src, w, h, radius, isMax) => {
    const tmp = new Float32Array(src.length)
    const out = new Float32Array(src.length)

    // Horizontal pass.
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        let count = 0
        const seed = Math.min(w - 1, radius)
        for (let i = 0; i <= seed; i += 1) if (src[row + i] > 0.5) count += 1
        for (let x = 0; x < w; x += 1) {
            const lo = x > radius ? x - radius : 0
            const hi = x + radius < w - 1 ? x + radius : w - 1
            tmp[row + x] = isMax ? (count > 0 ? 1 : 0) : (count === hi - lo + 1 ? 1 : 0)
            const drop = x - radius
            const add = x + radius + 1
            if (drop >= 0 && src[row + drop] > 0.5) count -= 1
            if (add < w && src[row + add] > 0.5) count += 1
        }
    }

    // Vertical pass.
    for (let x = 0; x < w; x += 1) {
        let count = 0
        const seed = Math.min(h - 1, radius)
        for (let i = 0; i <= seed; i += 1) if (tmp[i * w + x] > 0.5) count += 1
        for (let y = 0; y < h; y += 1) {
            const lo = y > radius ? y - radius : 0
            const hi = y + radius < h - 1 ? y + radius : h - 1
            out[y * w + x] = isMax ? (count > 0 ? 1 : 0) : (count === hi - lo + 1 ? 1 : 0)
            const drop = y - radius
            const add = y + radius + 1
            if (drop >= 0 && tmp[drop * w + x] > 0.5) count -= 1
            if (add < h && tmp[add * w + x] > 0.5) count += 1
        }
    }
    return out
}

/** Integral image (summed-area table) of a Float32 map, (w+1)×(h+1). */
const integral = (src, w, h) => {
    const sat = new Float64Array((w + 1) * (h + 1))
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
const boxMean = (sat, w, h, radius, out) => {
    const W = w + 1
    for (let y = 0; y < h; y += 1) {
        const y0 = Math.max(0, y - radius)
        const y1 = Math.min(h - 1, y + radius)
        const rowTop = y0 * W
        const rowBot = (y1 + 1) * W
        for (let x = 0; x < w; x += 1) {
            const x0 = Math.max(0, x - radius)
            const x1 = Math.min(w - 1, x + radius)
            const sum = sat[rowBot + x1 + 1] - sat[rowBot + x0] - sat[rowTop + x1 + 1] + sat[rowTop + x0]
            out[y * w + x] = sum / ((y1 - y0 + 1) * (x1 - x0 + 1))
        }
    }
    return out
}

/**
 * Refine a mask's boundary band in place. `rgba` is the white-on-black mask
 * (modified: band pixels become soft 0..255); `gray` is the source photo as
 * a 0..1 grayscale Float32Array of the same dimensions.
 *
 * `guide` is an optional mutable cache object owned by the caller and tied to
 * one photo. mean(I) and mean(I²) depend only on the guide image, so they are
 * computed once per photo instead of once per click — that is 2 of the 6 box
 * filters and 2 of the 5 integral images gone from every repeat decode.
 *
 * @returns {{ bandPixels: number, backend: 'cpu' }} how many pixels were refined
 */
export const refineMaskEdges = (rgba, w, h, gray, { band = 6, radius = 8, eps = 1e-3, guide = null } = {}) => {
    const size = w * h
    if (!gray || gray.length !== size) return { bandPixels: 0, backend: 'cpu' }

    const p = new Float32Array(size)
    for (let i = 0; i < size; i += 1) p[i] = rgba[i * 4] >= 128 ? 1 : 0

    // Boundary band = dilate(mask) − erode(mask).
    const dil = morph(p, w, h, band, true)
    const ero = morph(p, w, h, band, false)
    let bandPixels = 0
    for (let i = 0; i < size; i += 1) {
        if (dil[i] > 0.5 && ero[i] < 0.5) bandPixels += 1
    }
    if (!bandPixels) return { bandPixels: 0, backend: 'cpu' }

    // Guide statistics — photo-only, so cacheable across clicks.
    const guideKey = `${w}x${h}:${radius}`
    let meanI
    let meanII
    if (guide && guide.key === guideKey) {
        meanI = guide.meanI
        meanII = guide.meanII
    } else {
        meanI = boxMean(integral(gray, w, h), w, h, radius, new Float32Array(size))
        meanII = new Float32Array(size)
        for (let i = 0; i < size; i += 1) meanII[i] = gray[i] * gray[i]
        meanII = boxMean(integral(meanII, w, h), w, h, radius, meanII)
        if (guide) {
            guide.key = guideKey
            guide.meanI = meanI
            guide.meanII = meanII
        }
    }

    // Guided filter q = mean_a · I + mean_b over box windows of `radius`.
    const meanP = boxMean(integral(p, w, h), w, h, radius, new Float32Array(size))
    const Ip = new Float32Array(size)
    for (let i = 0; i < size; i += 1) Ip[i] = gray[i] * p[i]
    // `integral` is fully materialised before boxMean writes, so reusing Ip
    // as its own destination is safe.
    const meanIp = boxMean(integral(Ip, w, h), w, h, radius, Ip)

    const a = new Float32Array(size)
    const b = new Float32Array(size)
    for (let i = 0; i < size; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        a[i] = covIp / (varI + eps)
        b[i] = meanP[i] - a[i] * meanI[i]
    }
    const meanA = boxMean(integral(a, w, h), w, h, radius, a)
    const meanB = boxMean(integral(b, w, h), w, h, radius, b)

    for (let i = 0; i < size; i += 1) {
        if (!(dil[i] > 0.5 && ero[i] < 0.5)) continue
        let q = meanA[i] * gray[i] + meanB[i]
        if (q < 0) q = 0
        else if (q > 1) q = 1
        // Snap near-extremes so the band doesn't carry a faint fog.
        let v = Math.round(q * 255)
        if (v < 10) v = 0
        else if (v > 245) v = 255
        const j = i * 4
        rgba[j] = v
        rgba[j + 1] = v
        rgba[j + 2] = v
    }
    return { bandPixels, backend: 'cpu' }
}

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
 * All O(N) passes over typed arrays (integral-image box filters + separable
 * Chebyshev dilate/erode), ~tens of ms at 1024² — cheap enough to run on
 * every decode, model-agnostic, so it upgrades every engine lane forever.
 */

/** Separable Chebyshev dilate (max) or erode (min) of a 0/1 Float32 map. */
const morph = (src, w, h, radius, isMax) => {
    const tmp = new Float32Array(src.length)
    const out = new Float32Array(src.length)
    // Horizontal pass.
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            let v = isMax ? 0 : 1
            const lo = Math.max(0, x - radius)
            const hi = Math.min(w - 1, x + radius)
            for (let i = lo; i <= hi; i += 1) {
                const s = src[row + i]
                v = isMax ? (s > v ? s : v) : (s < v ? s : v)
            }
            tmp[row + x] = v
        }
    }
    // Vertical pass.
    for (let x = 0; x < w; x += 1) {
        for (let y = 0; y < h; y += 1) {
            let v = isMax ? 0 : 1
            const lo = Math.max(0, y - radius)
            const hi = Math.min(h - 1, y + radius)
            for (let i = lo; i <= hi; i += 1) {
                const s = tmp[i * w + x]
                v = isMax ? (s > v ? s : v) : (s < v ? s : v)
            }
            out[y * w + x] = v
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
 * Refine a mask's boundary band in place. `alpha` is the one-channel mask
 * (modified: band pixels become soft 0..255); `gray` is the source photo as
 * a 0..1 grayscale Float32Array of the same dimensions.
 *
 * @returns {{ bandPixels: number }} how many pixels were refined
 */
export const refineMaskEdges = (alpha, w, h, gray, { band = 6, radius = 8, eps = 1e-3 } = {}) => {
    const size = w * h
    if (!gray || gray.length !== size) return { bandPixels: 0 }

    const p = new Float32Array(size)
    for (let i = 0; i < size; i += 1) p[i] = alpha[i] >= 128 ? 1 : 0

    // Boundary band = dilate(mask) − erode(mask).
    const dil = morph(p, w, h, band, true)
    const ero = morph(p, w, h, band, false)
    let bandPixels = 0
    for (let i = 0; i < size; i += 1) {
        if (dil[i] > 0.5 && ero[i] < 0.5) bandPixels += 1
    }
    if (!bandPixels) return { bandPixels: 0 }

    // Guided filter q = mean_a · I + mean_b over box windows of `radius`.
    const meanI = boxMean(integral(gray, w, h), w, h, radius, new Float32Array(size))
    const meanP = boxMean(integral(p, w, h), w, h, radius, new Float32Array(size))
    const Ip = new Float32Array(size)
    const II = new Float32Array(size)
    for (let i = 0; i < size; i += 1) {
        Ip[i] = gray[i] * p[i]
        II[i] = gray[i] * gray[i]
    }
    const meanIp = boxMean(integral(Ip, w, h), w, h, radius, Ip) // reuse buffers
    const meanII = boxMean(integral(II, w, h), w, h, radius, II)

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
        alpha[i] = v
    }
    return { bandPixels }
}

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

/**
 * Separable Chebyshev dilate (max) or erode (min) of a 0/1 Float32 map.
 *
 * This is the same rectangular morphology that OpenCV's `morphologyEx`
 * would apply, but it uses a monotonic deque rather than scanning every
 * pixel in every radius-sized window. It is O(pixels), not
 * O(pixels × radius), and the caller supplies the two raster buffers so a
 * 1024px interaction frame does not create short-lived heap spikes.
 */
const morph = (src, w, h, radius, isMax, tmp, out, deque) => {
    const dominates = isMax
        ? (a, b) => a >= b
        : (a, b) => a <= b

    // Horizontal pass. The deque holds source X coordinates whose values can
    // still win the current [x - radius, x + radius] window.
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        let head = 0
        let tail = 0
        let next = 0
        for (let x = 0; x < w; x += 1) {
            const hi = Math.min(w - 1, x + radius)
            while (next <= hi) {
                const v = src[row + next]
                while (tail > head && dominates(v, src[row + deque[tail - 1]])) tail -= 1
                deque[tail++] = next
                next += 1
            }
            const lo = Math.max(0, x - radius)
            while (head < tail && deque[head] < lo) head += 1
            tmp[row + x] = src[row + deque[head]]
        }
    }

    // Vertical pass. Reuse the same deque and write directly into `out`.
    for (let x = 0; x < w; x += 1) {
        let head = 0
        let tail = 0
        let next = 0
        for (let y = 0; y < h; y += 1) {
            const hi = Math.min(h - 1, y + radius)
            while (next <= hi) {
                const v = tmp[next * w + x]
                while (tail > head && dominates(v, tmp[deque[tail - 1] * w + x])) tail -= 1
                deque[tail++] = next
                next += 1
            }
            const lo = Math.max(0, y - radius)
            while (head < tail && deque[head] < lo) head += 1
            out[y * w + x] = tmp[deque[head] * w + x]
        }
    }
    return out
}

/** Integral image (summed-area table) of a Float32 map, (w+1)×(h+1). */
const integralInto = (src, w, h, sat) => {
    const W = w + 1
    // Reusing the buffer avoids allocating six >8 MB summed-area tables for
    // every 1024px mask. The first row and every leading column must be zero
    // because they form the virtual border of the integral image.
    sat.fill(0, 0, W)
    for (let y = 0; y < h; y += 1) {
        let rowSum = 0
        const srcRow = y * w
        const satRow = (y + 1) * W
        const satPrev = y * W
        sat[satRow] = 0
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
 * @returns {{ bandPixels: number }} how many pixels were refined
 */
export const refineMaskEdges = (rgba, w, h, gray, { band = 6, radius = 8, eps = 1e-3 } = {}) => {
    const size = w * h
    if (!gray || gray.length !== size) return { bandPixels: 0 }

    // Working set at 1024²: eight Float32 rasters (32 MB) plus one reusable
    // Float64 summed-area table (8 MB). Previously each filter pass allocated
    // another table, which could briefly push Safari/Chromium into a much
    // larger GC peak while the segmentation model was still resident.
    const p = new Float32Array(size)
    const morphTmp = new Float32Array(size)
    const dil = new Float32Array(size)
    const ero = new Float32Array(size)
    const deque = new Int32Array(Math.max(w, h))
    for (let i = 0; i < size; i += 1) p[i] = rgba[i * 4] >= 128 ? 1 : 0

    // Boundary band = dilate(mask) − erode(mask).
    morph(p, w, h, band, true, morphTmp, dil, deque)
    morph(p, w, h, band, false, morphTmp, ero, deque)
    let bandPixels = 0
    for (let i = 0; i < size; i += 1) {
        if (dil[i] > 0.5 && ero[i] < 0.5) bandPixels += 1
    }
    if (!bandPixels) return { bandPixels: 0 }

    // Guided filter q = mean_a · I + mean_b over box windows of `radius`.
    const sat = new Float64Array((w + 1) * (h + 1))
    const meanI = boxMean(integralInto(gray, w, h, sat), w, h, radius, new Float32Array(size))
    const meanP = boxMean(integralInto(p, w, h, sat), w, h, radius, new Float32Array(size))
    // `cross` and `square` become meanIp/meanII, then safely become b/meanB
    // and remain reusable after their previous value has been consumed.
    const cross = new Float32Array(size)
    const square = new Float32Array(size)
    for (let i = 0; i < size; i += 1) {
        cross[i] = gray[i] * p[i]
        square[i] = gray[i] * gray[i]
    }
    const meanIp = boxMean(integralInto(cross, w, h, sat), w, h, radius, cross)
    const meanII = boxMean(integralInto(square, w, h, sat), w, h, radius, square)

    // `p` is no longer needed after morphology, so it becomes `a`; `cross`
    // becomes `b` after its mean was consumed. This eliminates two more
    // large temporary rasters without changing the guided-filter math.
    const a = p
    const b = cross
    for (let i = 0; i < size; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        a[i] = covIp / (varI + eps)
        b[i] = meanP[i] - a[i] * meanI[i]
    }
    const meanA = boxMean(integralInto(a, w, h, sat), w, h, radius, a)
    const meanB = boxMean(integralInto(b, w, h, sat), w, h, radius, b)

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
    return { bandPixels }
}

/**
 * Tiled variant for original-resolution crops (HD export): a 12 MP frame
 * through the full-frame filter would churn ~580 MB (the SATs are Float64),
 * so process `tileSide`² tiles with an `overlap` halo instead. The guided
 * filter is strictly window-local — every output pixel depends on at most
 * band + 2·radius neighbours — so with overlap ≥ that influence radius the
 * tile interiors are bit-identical to a full-frame pass: seamless by
 * construction, peak memory ~tens of MB per tile.
 *
 * bandPixels is telemetry-only and may count overlap bands twice.
 */
export const refineMaskEdgesTiled = (rgba, w, h, gray, {
    band = 6, radius = 8, eps = 1e-3, tileSide = 1024, overlap = 64,
} = {}) => {
    if (w <= tileSide && h <= tileSide) return refineMaskEdges(rgba, w, h, gray, { band, radius, eps })

    let bandPixels = 0
    for (let ty = 0; ty < h; ty += tileSide) {
        for (let tx = 0; tx < w; tx += tileSide) {
            // Interior this tile owns, plus the halo it reads.
            const ix1 = Math.min(w, tx + tileSide)
            const iy1 = Math.min(h, ty + tileSide)
            const ex0 = Math.max(0, tx - overlap)
            const ey0 = Math.max(0, ty - overlap)
            const ex1 = Math.min(w, ix1 + overlap)
            const ey1 = Math.min(h, iy1 + overlap)
            const ew = ex1 - ex0
            const eh = ey1 - ey0

            const tileRgba = new Uint8ClampedArray(ew * eh * 4)
            const tileGray = new Float32Array(ew * eh)
            for (let y = 0; y < eh; y += 1) {
                const src = ((ey0 + y) * w + ex0)
                tileRgba.set(rgba.subarray(src * 4, (src + ew) * 4), y * ew * 4)
                tileGray.set(gray.subarray(src, src + ew), y * ew)
            }

            const r = refineMaskEdges(tileRgba, ew, eh, tileGray, { band, radius, eps })
            bandPixels += r.bandPixels

            // Write back the interior rows only — halo pixels belong to
            // whichever tile owns them as interior.
            for (let y = ty; y < iy1; y += 1) {
                const srcRow = ((y - ey0) * ew + (tx - ex0)) * 4
                rgba.set(tileRgba.subarray(srcRow, srcRow + (ix1 - tx) * 4), (y * w + tx) * 4)
            }
        }
    }
    return { bandPixels }
}

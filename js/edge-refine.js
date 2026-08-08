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
 * Two properties keep it cheap:
 *
 *  1. Every pass is genuinely O(N) in the pixels it touches: integral-image
 *     box filters, and sliding-window-count morphology (the mask is binary,
 *     so a dilate is "window contains a 1" and an erode is "window is all
 *     1s" — both maintainable incrementally, no per-pixel radius loop).
 *
 *  2. It only touches a shell around the mask boundary. The band is a few
 *     percent of a frame, and everything outside it is left exactly as the
 *     decoder left it, so filtering the whole frame is almost entirely
 *     wasted work. See CROP MARGIN below for why the shell is safe.
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
 * CROP MARGIN — how far around the mask box the filter must still be exact.
 *
 * Work backwards from the only pixels whose value can change, the band:
 *   band ⊆ maskBox ⊕ band                       (dilate/erode of radius `band`)
 *   meanA/meanB at a band pixel read a,b        within ±radius
 *   a,b are pointwise in meanP/meanIp, which read p,gray  within ±radius
 * so a band pixel's result depends on nothing further than band + 2·radius.
 * Clamping the filter windows to the crop instead of the frame therefore
 * changes nothing at band pixels. `3·radius` leaves 50% headroom over the
 * `2·radius` the derivation needs, at negligible extra area.
 *
 * The same margin makes the band itself exact: every pixel within `band` of
 * the crop edge has an all-zero neighbourhood (the mask is ≥ margin away), so
 * dilate = 0 there and it can never be misclassified as band.
 */
const cropMargin = (band, radius) => band + 3 * radius

/**
 * Refine a mask's boundary band in place. `rgba` is the white-on-black mask
 * (modified: band pixels become soft 0..255); `gray` is the source photo as
 * a 0..1 grayscale Float32Array of the same dimensions.
 *
 * `guide` is an optional mutable cache object owned by the caller and tied to
 * one photo. mean(I) and mean(I²) depend only on the guide image, not on the
 * mask, so they are computed once per photo and reused by every later click.
 * They are cached at FULL frame size, which keeps them valid no matter where
 * the next selection's crop lands.
 *
 * `bbox` is the mask's `[x0,y0,x1,y1]` extent (from cleanupMaskRGBA). Given
 * it, the filter runs only on that box plus `cropMargin`; without it, on the
 * whole frame. Both produce identical output.
 *
 * @returns {{ bandPixels:number, backend:'cpu', cropFraction:number }}
 */
export const refineMaskEdges = (
    rgba, w, h, gray,
    { band = 6, radius = 8, eps = 1e-3, guide = null, bbox = null } = {},
) => {
    const size = w * h
    if (!gray || gray.length !== size) return { bandPixels: 0, backend: 'cpu', cropFraction: 0 }

    const margin = cropMargin(band, radius)
    const x0 = bbox ? Math.max(0, bbox[0] - margin) : 0
    const y0 = bbox ? Math.max(0, bbox[1] - margin) : 0
    const x1 = bbox ? Math.min(w - 1, bbox[2] + margin) : w - 1
    const y1 = bbox ? Math.min(h - 1, bbox[3] + margin) : h - 1
    if (x1 < x0 || y1 < y0) return { bandPixels: 0, backend: 'cpu', cropFraction: 0 }
    const cw = x1 - x0 + 1
    const ch = y1 - y0 + 1
    const csize = cw * ch

    /** Copy a full-frame map's crop window into a compact cw×ch buffer. */
    const cropOf = (src, out = new Float32Array(csize)) => {
        for (let cy = 0; cy < ch; cy += 1) {
            const s = (y0 + cy) * w + x0
            const d = cy * cw
            for (let cx = 0; cx < cw; cx += 1) out[d + cx] = src[s + cx]
        }
        return out
    }

    const p = new Float32Array(csize)
    for (let cy = 0; cy < ch; cy += 1) {
        const s = (y0 + cy) * w + x0
        const d = cy * cw
        for (let cx = 0; cx < cw; cx += 1) p[d + cx] = rgba[(s + cx) * 4] >= 128 ? 1 : 0
    }

    // Boundary band = dilate(mask) − erode(mask).
    const dil = morph(p, cw, ch, band, true)
    const ero = morph(p, cw, ch, band, false)
    let bandPixels = 0
    for (let i = 0; i < csize; i += 1) {
        if (dil[i] > 0.5 && ero[i] < 0.5) bandPixels += 1
    }
    if (!bandPixels) return { bandPixels: 0, backend: 'cpu', cropFraction: csize / size }

    // Guide statistics — photo-only, so computed once per photo at full size
    // and windowed here. Full-frame values are exact everywhere, which is
    // what keeps the cropped result identical to the uncropped one.
    const guideKey = `${w}x${h}:${radius}`
    let fullMeanI
    let fullMeanII
    if (guide && guide.key === guideKey) {
        fullMeanI = guide.meanI
        fullMeanII = guide.meanII
    } else {
        fullMeanI = boxMean(integral(gray, w, h), w, h, radius, new Float32Array(size))
        const sq = new Float32Array(size)
        for (let i = 0; i < size; i += 1) sq[i] = gray[i] * gray[i]
        fullMeanII = boxMean(integral(sq, w, h), w, h, radius, sq)
        if (guide) {
            guide.key = guideKey
            guide.meanI = fullMeanI
            guide.meanII = fullMeanII
        }
    }
    const cgray = cropOf(gray)
    const meanI = cropOf(fullMeanI)
    const meanII = cropOf(fullMeanII)

    // Guided filter q = mean_a · I + mean_b over box windows of `radius`.
    const meanP = boxMean(integral(p, cw, ch), cw, ch, radius, new Float32Array(csize))
    const Ip = new Float32Array(csize)
    for (let i = 0; i < csize; i += 1) Ip[i] = cgray[i] * p[i]
    // `integral` is fully materialised before boxMean writes, so reusing Ip
    // as its own destination is safe.
    const meanIp = boxMean(integral(Ip, cw, ch), cw, ch, radius, Ip)

    const a = new Float32Array(csize)
    const b = new Float32Array(csize)
    for (let i = 0; i < csize; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        a[i] = covIp / (varI + eps)
        b[i] = meanP[i] - a[i] * meanI[i]
    }
    const meanA = boxMean(integral(a, cw, ch), cw, ch, radius, a)
    const meanB = boxMean(integral(b, cw, ch), cw, ch, radius, b)

    for (let cy = 0; cy < ch; cy += 1) {
        const d = cy * cw
        const s = (y0 + cy) * w + x0
        for (let cx = 0; cx < cw; cx += 1) {
            const i = d + cx
            if (!(dil[i] > 0.5 && ero[i] < 0.5)) continue
            let q = meanA[i] * cgray[i] + meanB[i]
            if (q < 0) q = 0
            else if (q > 1) q = 1
            // Snap near-extremes so the band doesn't carry a faint fog.
            let v = Math.round(q * 255)
            if (v < 10) v = 0
            else if (v > 245) v = 255
            const j = (s + cx) * 4
            rgba[j] = v
            rgba[j + 1] = v
            rgba[j + 2] = v
        }
    }
    return { bandPixels, backend: 'cpu', cropFraction: csize / size }
}

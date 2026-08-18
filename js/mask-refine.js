/**
 * mask-refine — closed-form mask edge refinement (DESIGN-MASK-LANE §10).
 *
 * SAM emits 256². A Z 8 export is 8256×5504 — a ~32× enlargement. How that gap
 * is bridged decides perceived quality far more than tiny-vs-small does, so all
 * of it is done on the CONTINUOUS score field and thresholded only at the end:
 *
 *   1. bicubic upsample of the logits            (sub-pixel zero crossing)
 *   2. guided filter against the photo's COLOUR  (snaps the edge to the object)
 *   3. threshold at output resolution
 *
 * Step 2 guides on Y/Cb/Cr, not luma. Luma alone cannot see an isoluminant
 * boundary — pink petals on green foliage, orange paint on grey road — and
 * those are common, not exotic: measured over seven real crops, moving to a
 * colour guide lifts mean boundary IoU 0.735 → 0.864.
 *
 * No library, no convolution loops: box filters run off a summed-area table, so
 * every stage is O(n) in pixels and independent of radius. The guided filter
 * uses He et al.'s FAST variant — coefficients solved on a subsampled grid and
 * upsampled — which is ~s² cheaper and visually indistinguishable, because a
 * and b are far smoother than the image they are applied to.
 */

/**
 * Chroma ε as a multiple of the luma ε — how much better than luma a chroma
 * edge must be before the filter follows it. Swept over seven real crops
 * (`scratchpad/bench.mjs`), not guessed: 3 keeps ~95 % of the colour gain on
 * chromatic subjects while removing the achromatic regression that raw RGB
 * causes on thin wires.
 *
 * Two adaptive rules were built and measured, and BOTH lose to this constant:
 *   - ε from the median local chroma variance (a noise floor): no effect on the
 *     wires at all, because their damaging chroma is a strong CA fringe, not
 *     noise, and it costs ~0.5 pt on every chromatic crop;
 *   - ε scaled by local luma variance ("use chroma only where luma failed"):
 *     the isoluminant boundary still carries some luma signal, so it suppresses
 *     chroma exactly where it is needed — rose boundary IoU +4.8 pt vs +9.1.
 * Do not replace this constant with a clever rule without re-running the bench.
 */
const CHROMA_EPS = 3

/**
 * Box mean of radius r — two separable running-sum passes, O(1) per pixel.
 *
 * Was a summed-area table. A SAT needs f64 (a 1024² plane of ~10-magnitude
 * logits overflows f32 well before the corner) and a whole extra (w+1)(h+1)
 * plane, and the filter calls this 17 times per click. The running window never
 * accumulates more than 2r+1 terms, so f32 is exact enough and the table is
 * gone. At the shipped settings (radius 8, scale 4) r is 2 — a 5-tap window.
 *
 * `tmp` is the horizontal intermediate; pass one in to reuse it across calls.
 * Edge windows divide by their own clipped width/height, so no border darkening
 * — same rule the SAT used.
 */
const boxMean = (src, w, h, r, tmp = new Float32Array(w * h)) => {
    const out = new Float32Array(w * h)
    // Clipped-window width depends only on x, so the division leaves the hot
    // loop entirely — it was the most expensive instruction in it.
    const invW = new Float32Array(w)
    for (let x = 0; x < w; x += 1) {
        invW[x] = 1 / ((x + r >= w ? w - 1 : x + r) - (x - r < 0 ? 0 : x - r) + 1)
    }
    for (let y = 0; y < h; y += 1) {
        const row = y * w
        let sum = 0
        for (let x = 0, e = r < w ? r : w - 1; x <= e; x += 1) sum += src[row + x]
        for (let x = 0; x < w; x += 1) {
            tmp[row + x] = sum * invW[x]
            const add = x + r + 1
            if (add < w) sum += src[row + add]
            if (x >= r) sum -= src[row + x - r]
        }
    }
    // Column sums advance a row at a time, so the vertical pass still reads
    // sequentially — a per-column loop would stride the whole plane w times.
    const col = new Float32Array(w)
    for (let y = 0, e = r < h ? r : h - 1; y <= e; y += 1) {
        const row = y * w
        for (let x = 0; x < w; x += 1) col[x] += tmp[row + x]
    }
    for (let y = 0; y < h; y += 1) {
        const inv = 1 / ((y + r >= h ? h - 1 : y + r) - (y - r < 0 ? 0 : y - r) + 1)
        const dst = y * w
        // Emit, advance and retire in one traversal; three separate loops over
        // the row read `col` three times for no reason.
        const addRow = y + r + 1 < h ? (y + r + 1) * w : -1
        const subRow = y >= r ? (y - r) * w : -1
        if (addRow >= 0 && subRow >= 0) {
            for (let x = 0; x < w; x += 1) {
                out[dst + x] = col[x] * inv
                col[x] += tmp[addRow + x] - tmp[subRow + x]
            }
        } else if (addRow >= 0) {
            for (let x = 0; x < w; x += 1) { out[dst + x] = col[x] * inv; col[x] += tmp[addRow + x] }
        } else if (subRow >= 0) {
            for (let x = 0; x < w; x += 1) { out[dst + x] = col[x] * inv; col[x] -= tmp[subRow + x] }
        } else {
            for (let x = 0; x < w; x += 1) out[dst + x] = col[x] * inv
        }
    }
    return out
}

/** Nearest-box downsample by integer factor s (means, not point samples, so
 *  aliasing does not leak into the coefficients). */
const downsample = (src, w, h, s, dw, dh) => {
    const out = new Float32Array(dw * dh)
    for (let y = 0; y < dh; y += 1) {
        const y0 = y * s
        const y1 = Math.min(h, y0 + s)
        for (let x = 0; x < dw; x += 1) {
            const x0 = x * s
            const x1 = Math.min(w, x0 + s)
            let acc = 0
            for (let yy = y0; yy < y1; yy += 1) {
                const row = yy * w
                for (let xx = x0; xx < x1; xx += 1) acc += src[row + xx]
            }
            out[y * dw + x] = acc / ((y1 - y0) * (x1 - x0))
        }
    }
    return out
}

/** The same downsample over four planes in ONE traversal. The colour filter
 *  reduces three guide channels and the field together, and four separate calls
 *  walk the same full-res block four times. */
const downsample4 = (a, b, c, d, w, h, s, dw, dh) => {
    const oa = new Float32Array(dw * dh)
    const ob = new Float32Array(dw * dh)
    const oc = new Float32Array(dw * dh)
    const od = new Float32Array(dw * dh)
    for (let y = 0; y < dh; y += 1) {
        const y0 = y * s
        const y1 = Math.min(h, y0 + s)
        const drow = y * dw
        for (let x = 0; x < dw; x += 1) {
            const x0 = x * s
            const x1 = Math.min(w, x0 + s)
            let sa = 0, sb = 0, sc = 0, sd = 0
            for (let yy = y0; yy < y1; yy += 1) {
                const row = yy * w
                for (let xx = x0; xx < x1; xx += 1) {
                    const i = row + xx
                    sa += a[i]; sb += b[i]; sc += c[i]; sd += d[i]
                }
            }
            const inv = 1 / ((y1 - y0) * (x1 - x0))
            oa[drow + x] = sa * inv
            ob[drow + x] = sb * inv
            oc[drow + x] = sc * inv
            od[drow + x] = sd * inv
        }
    }
    return [oa, ob, oc, od]
}

/** Bilinear upsample of a coefficient plane back to w×h. */
const upsample = (src, dw, dh, w, h) => {
    const out = new Float32Array(w * h)
    const sx = dw / w
    const sy = dh / h
    for (let y = 0; y < h; y += 1) {
        const fy = Math.min(dh - 1, Math.max(0, (y + 0.5) * sy - 0.5))
        const y0 = Math.floor(fy)
        const y1 = Math.min(dh - 1, y0 + 1)
        const wy = fy - y0
        const r0 = y0 * dw
        const r1 = y1 * dw
        const dst = y * w
        for (let x = 0; x < w; x += 1) {
            const fx = Math.min(dw - 1, Math.max(0, (x + 0.5) * sx - 0.5))
            const x0 = Math.floor(fx)
            const x1 = Math.min(dw - 1, x0 + 1)
            const wx = fx - x0
            out[dst + x] = (src[r0 + x0] * (1 - wx) + src[r0 + x1] * wx) * (1 - wy)
                + (src[r1 + x0] * (1 - wx) + src[r1 + x1] * wx) * wy
        }
    }
    return out
}

/**
 * Upsample the four coefficient planes and composite them against the guide in
 * ONE full-res pass: out = ar·R + ag·G + ab·B + b.
 *
 * Materialising each plane at w×h first meant four Float32Array(w·h) allocations
 * and five passes over the largest buffers in the filter — and at scale 4 the
 * full-res half is 16x the area of the coefficient half, so it dominated.
 * Bilinear weights are shared across all four reads.
 */
const compositeUp = (ar, ag, ab, b, dw, dh, R, G, B, w, h) => {
    const out = new Float32Array(w * h)
    const sx = dw / w
    const sy = dh / h
    // Column geometry repeats on every row — hoisting it takes three Math calls
    // and a floor out of the largest inner loop in the filter.
    const cx0s = new Int32Array(w)
    const cx1s = new Int32Array(w)
    const wxs = new Float32Array(w)
    for (let x = 0; x < w; x += 1) {
        const fx = Math.min(dw - 1, Math.max(0, (x + 0.5) * sx - 0.5))
        const c0 = Math.floor(fx)
        cx0s[x] = c0
        cx1s[x] = Math.min(dw - 1, c0 + 1)
        wxs[x] = fx - c0
    }
    for (let y = 0; y < h; y += 1) {
        const fy = Math.min(dh - 1, Math.max(0, (y + 0.5) * sy - 0.5))
        const cy0 = Math.floor(fy)
        const cy1 = Math.min(dh - 1, cy0 + 1)
        const wy = fy - cy0
        const r0 = cy0 * dw
        const r1 = cy1 * dw
        const dst = y * w
        for (let x = 0; x < w; x += 1) {
            const cx0 = cx0s[x]
            const cx1 = cx1s[x]
            const wx = wxs[x]
            const w00 = (1 - wx) * (1 - wy)
            const w10 = wx * (1 - wy)
            const w01 = (1 - wx) * wy
            const w11 = wx * wy
            const i00 = r0 + cx0, i10 = r0 + cx1, i01 = r1 + cx0, i11 = r1 + cx1
            const j = dst + x
            out[j] = (ar[i00] * w00 + ar[i10] * w10 + ar[i01] * w01 + ar[i11] * w11) * R[j]
                + (ag[i00] * w00 + ag[i10] * w10 + ag[i01] * w01 + ag[i11] * w11) * G[j]
                + (ab[i00] * w00 + ab[i10] * w10 + ab[i01] * w01 + ab[i11] * w11) * B[j]
                + (b[i00] * w00 + b[i10] * w10 + b[i01] * w01 + b[i11] * w11)
        }
    }
    return out
}

/**
 * Guided filter: refine score field `p` using photo luma `guide` (both w×h,
 * guide in [0,1]). Returns a new Float32Array — `p` is untouched.
 *
 * `eps` sets what counts as an edge worth following: variance below it is
 * treated as flat and smoothed across. 1e-4 ≈ luma steps under ~1 %.
 */
export const guidedFilter = (p, guide, w, h, { radius = 8, eps = 1e-4, scale = 4 } = {}) => {
    const s = Math.max(1, Math.round(scale))
    const dw = Math.max(1, Math.ceil(w / s))
    const dh = Math.max(1, Math.ceil(h / s))
    const r = Math.max(1, Math.round(radius / s))

    const I = s > 1 ? downsample(guide, w, h, s, dw, dh) : guide
    const P = s > 1 ? downsample(p, w, h, s, dw, dh) : p
    const n = dw * dh

    const Ip = new Float32Array(n)
    const II = new Float32Array(n)
    for (let i = 0; i < n; i += 1) { Ip[i] = I[i] * P[i]; II[i] = I[i] * I[i] }

    const meanI = boxMean(I, dw, dh, r)
    const meanP = boxMean(P, dw, dh, r)
    const meanIp = boxMean(Ip, dw, dh, r)
    const meanII = boxMean(II, dw, dh, r)

    // a = cov(I,p) / (var(I) + eps);  b = mean(p) - a·mean(I)
    const a = new Float32Array(n)
    const b = new Float32Array(n)
    for (let i = 0; i < n; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        const ai = covIp / (varI + eps)
        a[i] = ai
        b[i] = meanP[i] - ai * meanI[i]
    }

    const ma = boxMean(a, dw, dh, r)
    const mb = boxMean(b, dw, dh, r)
    const A = s > 1 ? upsample(ma, dw, dh, w, h) : ma
    const B = s > 1 ? upsample(mb, dw, dh, w, h) : mb

    const out = new Float32Array(w * h)
    for (let i = 0; i < w * h; i += 1) out[i] = A[i] * guide[i] + B[i]
    return out
}

/**
 * Colour guided filter — He et al. §4, the 3-channel form.
 *
 * The luma form above is blind to isoluminant boundaries, and those are not a
 * corner case: measured on a rose-against-foliage crop, 33 % of the true
 * boundary carries under 2 % luma contrast, and the mean boundary step (0.066)
 * is only 2.9x the mean step of the petal texture INSIDE the object. A guide
 * that cannot separate those two pulls the edge onto shading.
 *
 * Per window: a = (Σ + E)⁻¹ cov(I,p), b = mean(p) − aᵀ mean(I), with Σ the 3×3
 * guide covariance. Costs 13 box means and one symmetric 3×3 solve per
 * subsampled pixel instead of 4 means — and the solve runs on w·h/s² cells.
 *
 * E is DIAGONAL, not He's scalar εU, and that is what makes this safe on grey
 * subjects. Fed Y/Cb/Cr with a larger ε on the two chroma axes, an achromatic
 * window (where chroma is only demosaic noise) drives a_Cb, a_Cr → 0 and the
 * filter degenerates exactly to the luma form; a chromatic window still gets
 * the full solve. With a scalar ε and raw RGB the channels are collinear on
 * grey, Σ is near-singular, and the fit chases that noise — measured as a
 * −3.0 pt boundary-IoU regression on 1–2 px wires against sky.
 */
export const guidedFilterColor = (p, R, G, B, w, h, {
    radius = 8, eps = 1e-4, eps2 = null, eps3 = null, scale = 4,
} = {}) => {
    const e1 = eps
    const e2 = eps2 ?? eps
    const e3 = eps3 ?? eps
    const s = Math.max(1, Math.round(scale))
    const dw = Math.max(1, Math.ceil(w / s))
    const dh = Math.max(1, Math.ceil(h / s))
    const r = Math.max(1, Math.round(radius / s))
    const n = dw * dh

    const [dr, dg, db, P] = s > 1
        ? downsample4(R, G, B, p, w, h, s, dw, dh)
        : [R, G, B, p]

    const t = new Float32Array(n)
    const scratch = new Float32Array(n)   // boxMean's horizontal pass, 17 reuses
    const mean = (src) => boxMean(src, dw, dh, r, scratch)
    const prod = (a, b) => { for (let i = 0; i < n; i += 1) t[i] = a[i] * b[i]; return mean(t) }

    const mr = mean(dr), mg = mean(dg), mb = mean(db), mp = mean(P)
    const mrr = prod(dr, dr), mrg = prod(dr, dg), mrb = prod(dr, db)
    const mgg = prod(dg, dg), mgb = prod(dg, db), mbb = prod(db, db)
    const mrp = prod(dr, P), mgp = prod(dg, P), mbp = prod(db, P)

    const ar = new Float32Array(n)
    const ag = new Float32Array(n)
    const ab = new Float32Array(n)
    const bb = new Float32Array(n)
    for (let i = 0; i < n; i += 1) {
        const a11 = mrr[i] - mr[i] * mr[i] + e1
        const a22 = mgg[i] - mg[i] * mg[i] + e2
        const a33 = mbb[i] - mb[i] * mb[i] + e3
        const a12 = mrg[i] - mr[i] * mg[i]
        const a13 = mrb[i] - mr[i] * mb[i]
        const a23 = mgb[i] - mg[i] * mb[i]

        // Symmetric 3×3 inverse by cofactors.
        const c11 = a22 * a33 - a23 * a23
        const c12 = a13 * a23 - a12 * a33
        const c13 = a12 * a23 - a13 * a22
        let det = a11 * c11 + a12 * c12 + a13 * c13
        if (det > -1e-12 && det < 1e-12) det = det < 0 ? -1e-12 : 1e-12
        const c22 = a11 * a33 - a13 * a13
        const c23 = a13 * a12 - a11 * a23
        const c33 = a11 * a22 - a12 * a12

        const cr = mrp[i] - mr[i] * mp[i]
        const cg = mgp[i] - mg[i] * mp[i]
        const cb = mbp[i] - mb[i] * mp[i]

        const xr = (c11 * cr + c12 * cg + c13 * cb) / det
        const xg = (c12 * cr + c22 * cg + c23 * cb) / det
        const xb = (c13 * cr + c23 * cg + c33 * cb) / det
        ar[i] = xr; ag[i] = xg; ab[i] = xb
        bb[i] = mp[i] - xr * mr[i] - xg * mg[i] - xb * mb[i]
    }

    return compositeUp(mean(ar), mean(ag), mean(ab), mean(bb), dw, dh, R, G, B, w, h)
}

/**
 * Guided filter applied only where it can change anything.
 *
 * The filter rewrites the field near its zero crossing; far inside and far
 * outside the object it returns what it was given. Running it over the whole
 * frame is therefore mostly wasted work — a 0.1 %-coverage mask on a 1024×683
 * proxy touches ~700 px out of 700 000. This finds the band's bounding box,
 * pads it by the filter footprint so the box mean sees real context, and
 * refines that sub-rect in place.
 *
 * Measured on the 45 MP NEF proxy: 46.6 ms whole-frame → ~3 ms typical, which
 * is what keeps click → paint inside the 50 ms budget without deferring the
 * refinement to a second repaint.
 */
export const refineField = (field, rgba, w, h, bbox, {
    radius = 8, eps = 1e-4, scale = 4, color = true, chromaEps = CHROMA_EPS,
} = {}) => {
    if (!bbox || bbox[2] < bbox[0]) return null // no boundary in frame

    const pad = radius * 2 + scale * 2
    const x0 = Math.max(0, bbox[0] - pad)
    const y0 = Math.max(0, bbox[1] - pad)
    const x1 = Math.min(w, bbox[2] + pad + 1)
    const y1 = Math.min(h, bbox[3] + pad + 1)
    const rw = x1 - x0
    const rh = y1 - y0
    if (rw <= 2 || rh <= 2) return null

    // Field and guide for the region only — the other 99 % of a typical frame is
    // never read, so it is never computed.
    const sub = new Float32Array(rw * rh)
    const gy = new Float32Array(rw * rh)
    const cb = color ? new Float32Array(rw * rh) : null
    const cr = color ? new Float32Array(rw * rh) : null
    for (let y = 0; y < rh; y += 1) {
        const srow = (y0 + y) * w + x0
        const drow = y * rw
        sub.set(field.subarray(srow, srow + rw), drow)
        for (let x = 0; x < rw; x += 1) {
            const j = (srow + x) * 4
            const r = rgba[j] / 255
            const g = rgba[j + 1] / 255
            const b = rgba[j + 2] / 255
            const Y = 0.299 * r + 0.587 * g + 0.114 * b
            gy[drow + x] = Y
            if (color) {
                cb[drow + x] = 0.564 * (b - Y)   // Rec.601, ±0.5
                cr[drow + x] = 0.713 * (r - Y)
            }
        }
    }

    const ref = color
        ? guidedFilterColor(sub, gy, cb, cr, rw, rh,
            { radius, eps, eps2: eps * chromaEps, eps3: eps * chromaEps, scale })
        : guidedFilter(sub, gy, rw, rh, { radius, eps, scale })
    for (let y = 0; y < rh; y += 1) {  // written back over `field`, in place
        field.set(ref.subarray(y * rw, y * rw + rw), (y0 + y) * w + x0)
    }
    // The rect actually rewritten — everything outside is byte-identical, so a
    // caller re-thresholding only needs to revisit this.
    return [x0, y0, x1, y1]
}

/**
 * Narrow-band alpha (§10 step 3). Inside and outside stay hard; only a strip
 * around the zero crossing gets a soft ramp, so hair and foliage keep a real
 * gradient at negligible cost. `band` is in logit units, not pixels — the field
 * is already edge-aligned by the guided filter, so a value cut is the right
 * cut and costs one pass.
 */
export const bandAlpha = (field, w, h, band = 1.5, out = null) => {
    const dst = out || new Uint8ClampedArray(w * h * 4)
    const inv = 255 / (2 * band)
    for (let i = 0, j = 0; i < field.length; i += 1, j += 4) {
        const v = field[i]
        // The ramp goes in EVERY channel: the app reads coverage from R (>=128)
        // and composites from A, so both have to carry the same soft edge.
        const a = v <= -band ? 0 : (v >= band ? 255 : Math.round((v + band) * inv))
        dst[j] = dst[j + 1] = dst[j + 2] = dst[j + 3] = a
    }
    return dst
}

/** The same ramp restricted to `rect` — for re-deriving a mask after
 *  refineField rewrote only that band. Outside is left as the caller had it. */
export const bandAlphaRect = (field, w, rect, band, dst) => {
    const [x0, y0, x1, y1] = rect
    const inv = 255 / (2 * band)
    for (let y = y0; y < y1; y += 1) {
        const row = y * w
        for (let x = x0; x < x1; x += 1) {
            const v = field[row + x]
            const a = v <= -band ? 0 : (v >= band ? 255 : Math.round((v + band) * inv))
            const j = (row + x) * 4
            dst[j] = dst[j + 1] = dst[j + 2] = dst[j + 3] = a
        }
    }
    return dst
}

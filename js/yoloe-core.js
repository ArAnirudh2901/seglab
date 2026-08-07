/**
 * yoloe-core (pure — no DOM, no ONNX runtime)
 * ----------------------------------------------
 * Everything needed to turn a YOLOE-26 segmentation head's raw tensors into
 * instance masks in source-image coordinates. Dependency-free so the whole
 * decode path is unit-testable headless — which matters, because every
 * selection mode in the app now rides on these numbers.
 *
 * SHAPE OF A YOLOE-26-seg EXPORT
 *   output0  [1, 4 + nc + nm, N]   or   [1, N, 4 + nc + nm]
 *            per candidate: cx, cy, w, h (letterboxed pixels), nc class
 *            scores, nm mask coefficients. Both layouts appear in the wild
 *            depending on export flags, so `decodeDetections` detects it.
 *   output1  [1, nm, PH, PW]       mask prototypes (typically 32 × 160 × 160)
 *
 * A detection's mask is sigmoid(coeffs · protos), cropped to its own box.
 * The crop is not an optimization — prototype masks are global basis
 * functions and routinely fire on similar-looking objects elsewhere in the
 * frame, so an uncropped mask bleeds onto other instances.
 *
 * NMS-FREE. YOLO26 is end-to-end and emits one candidate per object, so no
 * NMS is applied by default. Some exports (and every pre-26 checkpoint) still
 * need it, so `dedupe` is available and `decodeDetections` will apply it when
 * asked — off by default, because running NMS on NMS-free output silently
 * deletes legitimately overlapping instances.
 */

/* ─── Letterbox geometry ─────────────────────────────────────────────────── */

/**
 * Ultralytics letterbox: scale the long side to `size`, centre, pad the rest.
 * Returns the mapping needed to undo it.
 */
export const letterbox = (srcW, srcH, size = 640) => {
    const scale = Math.min(size / srcW, size / srcH)
    const w = Math.round(srcW * scale)
    const h = Math.round(srcH * scale)
    return { scale, w, h, padX: (size - w) / 2, padY: (size - h) / 2, size }
}

/** Letterboxed [x0,y0,x1,y1] → source pixels, clamped to the frame. */
export const unletterboxBox = (box, lb, srcW, srcH) => {
    const x0 = (box[0] - lb.padX) / lb.scale
    const y0 = (box[1] - lb.padY) / lb.scale
    const x1 = (box[2] - lb.padX) / lb.scale
    const y1 = (box[3] - lb.padY) / lb.scale
    return [
        Math.max(0, Math.min(srcW - 1, Math.min(x0, x1))),
        Math.max(0, Math.min(srcH - 1, Math.min(y0, y1))),
        Math.max(0, Math.min(srcW - 1, Math.max(x0, x1))),
        Math.max(0, Math.min(srcH - 1, Math.max(y0, y1))),
    ]
}

/* ─── Detection decode ───────────────────────────────────────────────────── */

/** IoU of two [x0,y0,x1,y1] boxes. */
export const boxIoU = (a, b) => {
    const ix0 = Math.max(a[0], b[0])
    const iy0 = Math.max(a[1], b[1])
    const ix1 = Math.min(a[2], b[2])
    const iy1 = Math.min(a[3], b[3])
    const iw = ix1 - ix0
    const ih = iy1 - iy0
    if (iw <= 0 || ih <= 0) return 0
    const inter = iw * ih
    const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1])
    const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1])
    const union = areaA + areaB - inter
    return union > 0 ? inter / union : 0
}

/** Greedy NMS. Only for exports that are not end-to-end — see the header. */
export const dedupe = (dets, iouThreshold = 0.7) => {
    const order = [...dets].sort((a, b) => b.score - a.score)
    const kept = []
    for (const d of order) {
        if (kept.every((k) => boxIoU(k.box, d.box) < iouThreshold)) kept.push(d)
    }
    return kept
}

/**
 * Decode output0 into scored detections in SOURCE coordinates.
 *
 * @param {Float32Array} data  output0 contents
 * @param {number[]} dims      [1, A, B] — the decoder works out which axis is
 *                             the candidate axis from `numClasses + numMasks`
 * @param {object} opts
 * @param {number} opts.numClasses
 * @param {number} opts.numMasks      mask coefficients per candidate (32)
 * @param {object} opts.lb            letterbox() result
 * @param {number} opts.srcW
 * @param {number} opts.srcH
 * @param {number} [opts.confThreshold=0.25]
 * @param {number} [opts.maxDetections=100]
 * @param {number|null} [opts.nmsIoU=null]  set to run NMS (non-end-to-end exports)
 * @returns {Array<{box:number[], score:number, classIdx:number, coeffs:Float32Array}>}
 */
export const decodeDetections = (data, dims, {
    numClasses, numMasks = 32, lb, srcW, srcH,
    confThreshold = 0.25, maxDetections = 100, nmsIoU = null,
} = {}) => {
    const stride = 4 + numClasses + numMasks
    const [, a, b] = dims.length === 3 ? dims : [1, dims[0], dims[1]]

    // Which axis holds the attributes? Whichever matches 4+nc+nm.
    let numCandidates
    let attrMajor // true ⇒ [1, stride, N] (attribute-major, the usual export)
    if (a === stride) { numCandidates = b; attrMajor = true } else if (b === stride) { numCandidates = a; attrMajor = false } else {
        throw new Error(`output0 dims [${dims}] match neither 4+${numClasses}+${numMasks}=${stride} axis`)
    }
    const at = attrMajor
        ? (attr, i) => data[attr * numCandidates + i]
        : (attr, i) => data[i * stride + attr]

    const dets = []
    for (let i = 0; i < numCandidates; i += 1) {
        // Best class for this candidate.
        let best = 0
        let bestScore = -Infinity
        for (let c = 0; c < numClasses; c += 1) {
            const s = at(4 + c, i)
            if (s > bestScore) { bestScore = s; best = c }
        }
        if (!(bestScore >= confThreshold)) continue

        const cx = at(0, i)
        const cy = at(1, i)
        const w = at(2, i)
        const h = at(3, i)
        const box = unletterboxBox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], lb, srcW, srcH)
        if (box[2] - box[0] < 1 || box[3] - box[1] < 1) continue

        const coeffs = new Float32Array(numMasks)
        for (let m = 0; m < numMasks; m += 1) coeffs[m] = at(4 + numClasses + m, i)

        dets.push({ box, score: bestScore, classIdx: best, coeffs })
    }

    dets.sort((x, y) => y.score - x.score)
    const out = nmsIoU === null ? dets : dedupe(dets, nmsIoU)
    return out.slice(0, maxDetections)
}

/* ─── Mask assembly ──────────────────────────────────────────────────────── */

const sigmoid = (v) => 1 / (1 + Math.exp(-v))

/**
 * Build one instance's mask from its coefficients and the shared prototypes.
 *
 * Work happens at PROTOTYPE resolution (typically 160×160) and only inside
 * the detection's own box, then upsamples straight into a bbox-cropped plane
 * in source coordinates. Nothing full-frame is ever allocated — a 6000×4000
 * full-frame mask is ~24 MB as a plain Uint8 plane and ~96 MB as RGBA, per
 * instance, which is what makes "segment everything" fall over.
 *
 * @param {Float32Array} coeffs      numMasks coefficients
 * @param {Float32Array} protos      [numMasks, PH, PW] flattened
 * @param {{numMasks:number, ph:number, pw:number}} protoDims
 * @param {number[]} box             source-space [x0,y0,x1,y1]
 * @param {object} lb                letterbox() result
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} [threshold=0.5]
 * @returns {{plane: Uint8Array, x0, y0, w, h, area} | null}
 */
export const assembleMask = (coeffs, protos, protoDims, box, lb, srcW, srcH, threshold = 0.5) => {
    const { numMasks, ph, pw } = protoDims
    const x0 = Math.max(0, Math.floor(box[0]))
    const y0 = Math.max(0, Math.floor(box[1]))
    const x1 = Math.min(srcW - 1, Math.ceil(box[2]))
    const y1 = Math.min(srcH - 1, Math.ceil(box[3]))
    const w = x1 - x0 + 1
    const h = y1 - y0 + 1
    if (w <= 0 || h <= 0) return null

    // Source pixel → prototype cell. Prototypes live in letterboxed space,
    // so the padding has to go back on before scaling down.
    const protoScaleX = pw / lb.size
    const protoScaleY = ph / lb.size
    const planeArea = ph * pw

    const plane = new Uint8Array(w * h)
    let area = 0
    for (let y = 0; y < h; y += 1) {
        const py = Math.min(ph - 1, Math.max(0, Math.round(((y0 + y) * lb.scale + lb.padY) * protoScaleY)))
        const rowOff = py * pw
        const dst = y * w
        for (let x = 0; x < w; x += 1) {
            const px = Math.min(pw - 1, Math.max(0, Math.round(((x0 + x) * lb.scale + lb.padX) * protoScaleX)))
            const cell = rowOff + px
            let acc = 0
            for (let m = 0; m < numMasks; m += 1) acc += coeffs[m] * protos[m * planeArea + cell]
            if (sigmoid(acc) >= threshold) { plane[dst + x] = 1; area += 1 }
        }
    }
    return area > 0 ? { plane, x0, y0, w, h, area } : null
}

/** Paint a bbox-cropped plane into a white-on-black RGBA frame buffer. */
export const planeToRGBA = (inst, frameW, frameH, rgba = null) => {
    const out = rgba || new Uint8ClampedArray(frameW * frameH * 4)
    if (!rgba) for (let i = 0; i < frameW * frameH; i += 1) out[i * 4 + 3] = 255
    for (let y = 0; y < inst.h; y += 1) {
        const src = y * inst.w
        const dst = (inst.y0 + y) * frameW + inst.x0
        for (let x = 0; x < inst.w; x += 1) {
            if (!inst.plane[src + x]) continue
            const j = (dst + x) * 4
            out[j] = 255; out[j + 1] = 255; out[j + 2] = 255
        }
    }
    return out
}

/** Letterbox an image's pixels into a normalized NCHW Float32 tensor. */
export const preprocess = (pixels, srcW, srcH, size = 640) => {
    const lb = letterbox(srcW, srcH, size)
    const tensor = new Float32Array(3 * size * size)
    const plane = size * size
    for (let y = 0; y < size; y += 1) {
        // Outside the letterboxed content: Ultralytics pads with 114/255.
        const sy = (y - lb.padY) / lb.scale
        const inRowY = sy >= 0 && sy < srcH
        const row = Math.min(srcH - 1, Math.max(0, Math.round(sy)))
        for (let x = 0; x < size; x += 1) {
            const sx = (x - lb.padX) / lb.scale
            const i = y * size + x
            if (!inRowY || sx < 0 || sx >= srcW) {
                tensor[i] = 114 / 255
                tensor[plane + i] = 114 / 255
                tensor[2 * plane + i] = 114 / 255
                continue
            }
            const j = (row * srcW + Math.min(srcW - 1, Math.max(0, Math.round(sx)))) * 4
            tensor[i] = pixels[j] / 255
            tensor[plane + i] = pixels[j + 1] / 255
            tensor[2 * plane + i] = pixels[j + 2] / 255
        }
    }
    return { tensor, lb, dims: [1, 3, size, size] }
}

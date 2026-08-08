/**
 * gpu-post — the post pipeline as WebGPU compute kernels (WGSL)
 * ----------------------------------------------------------------
 * The guided-filter edge refinement is the heaviest CPU work in a decode:
 * ~20 full-frame passes over Float32/Float64 arrays plus a Chebyshev
 * morphology, all of it embarrassingly parallel and all of it running on one
 * JS thread. This module runs the identical math ON THE GPU.
 *
 * Why WGSL and not C/C++: a browser page cannot execute C or C++ on a GPU.
 * C/C++ through Emscripten becomes WASM, which is still a CPU target — it
 * would make this code maybe 2-3x faster while leaving it single-threaded and
 * scalar. The only language that runs *directly on the GPU* from a web page is
 * a shader, and for WebGPU that is WGSL. The kernels below are written in the
 * C-family style a CUDA/HIP author would recognise (one thread per pixel,
 * explicit indexing, no allocation); a C++/Dawn build of this same pipeline
 * would hand the GPU these exact shaders.
 *
 * Contract: `gpuRefineMaskEdges` is a drop-in async twin of
 * `refineMaskEdges` (edge-refine.js) — same inputs, same in-place mutation of
 * the mask bytes, same result shape. It returns `null` whenever the GPU path
 * is unavailable or fails, and the caller falls back to the CPU
 * implementation. A GPU failure must never cost the user a click.
 *
 * Two things keep the work small, both mirrored in the CPU path:
 *   - mean(I) and mean(I²) depend only on the guide image, so they are
 *     computed ONCE per photo, at full frame size, and reused by every click.
 *   - everything else runs only on the mask's bounding box plus a margin.
 *     The band is a few percent of a frame and nothing outside it changes;
 *     filtering the whole frame is almost entirely wasted. See CROP MARGIN in
 *     edge-refine.js for why the shell is provably safe.
 */

/* ─── Kernels ────────────────────────────────────────────────────────────────
 * One module, one shared binding namespace; `layout: 'auto'` gives each entry
 * point a layout containing only the bindings it statically uses, so the
 * 8-storage-buffers-per-stage floor is never approached (max here is 6).
 *
 * All buffers stay full-frame sized and full-frame indexed. A dispatch covers
 * only the crop rectangle; `idx()` turns the flat invocation id into a frame
 * index, and every filter window clamps to the crop, not to the frame.
 */
const WGSL = /* wgsl */ `
struct Params {
  w      : u32,
  h      : u32,
  r      : u32,
  mode   : u32,   // morphology: 1 = dilate (max), 0 = erode (min)
  stride : u32,   // invocations per dispatch row (2D grid, see gridFor)
  cx0    : u32,
  cy0    : u32,
  cw     : u32,
  ch     : u32,
  eps    : f32,
  pad0   : f32,
  pad1   : f32,
}

@group(0) @binding(0)  var<uniform>                   P   : Params;
@group(0) @binding(1)  var<storage, read>             A   : array<f32>;
@group(0) @binding(2)  var<storage, read>             B   : array<f32>;
@group(0) @binding(3)  var<storage, read>             C   : array<f32>;
@group(0) @binding(4)  var<storage, read>             D   : array<f32>;
@group(0) @binding(5)  var<storage, read_write>       X   : array<f32>;
@group(0) @binding(6)  var<storage, read_write>       Y   : array<f32>;
@group(0) @binding(7)  var<storage, read>             SU  : array<u32>;
@group(0) @binding(8)  var<storage, read_write>       DU  : array<u32>;
@group(0) @binding(9)  var<storage, read_write>       CNT : atomic<u32>;
@group(0) @binding(10) var<storage, read>             E   : array<f32>;

const NONE : u32 = 0xffffffffu;

// Flat invocation id -> full-frame pixel index, or NONE when past the crop.
fn idx(g : vec3<u32>) -> u32 {
  let ci = g.x + g.y * P.stride;
  if (ci >= P.cw * P.ch) { return NONE; }
  return (P.cy0 + ci / P.cw) * P.w + (P.cx0 + ci % P.cw);
}

fn loX(x : i32) -> i32 { return max(x - i32(P.r), i32(P.cx0)); }
fn hiX(x : i32) -> i32 { return min(x + i32(P.r), i32(P.cx0 + P.cw) - 1); }
fn loY(y : i32) -> i32 { return max(y - i32(P.r), i32(P.cy0)); }
fn hiY(y : i32) -> i32 { return min(y + i32(P.r), i32(P.cy0 + P.ch) - 1); }

// RGBA bytes -> 0/1 indicator. WebGPU buffers are little-endian, so the red
// channel is the low byte of each packed pixel.
@compute @workgroup_size(64)
fn binarize(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  X[i] = select(0.0, 1.0, (SU[i] & 0xffu) >= 128u);
}

// Separable Chebyshev morphology, windows clamped to the crop.
@compute @workgroup_size(64)
fn morph_h(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let row = i32(i) - i32(i % P.w);
  let x = i32(i % P.w);
  let isMax = P.mode == 1u;
  var v = select(1.0, 0.0, isMax);
  let hi = hiX(x);
  for (var k = loX(x); k <= hi; k = k + 1) {
    let s = A[u32(row + k)];
    v = select(min(v, s), max(v, s), isMax);
  }
  X[i] = v;
}

@compute @workgroup_size(64)
fn morph_v(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let y = i32(i / P.w);
  let isMax = P.mode == 1u;
  var v = select(1.0, 0.0, isMax);
  let hi = hiY(y);
  for (var k = loY(y); k <= hi; k = k + 1) {
    let s = A[u32(k * w + x)];
    v = select(min(v, s), max(v, s), isMax);
  }
  X[i] = v;
}

// Separable box mean. The CPU path needs a summed-area table to reach O(N);
// on the GPU every pixel owns a thread, so the direct window sum is both
// simpler and faster (and avoids the Float64 SAT entirely).
@compute @workgroup_size(64)
fn box_h(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let row = i32(i) - i32(i % P.w);
  let x = i32(i % P.w);
  let lo = loX(x);
  let hi = hiX(x);
  var sum = 0.0;
  for (var k = lo; k <= hi; k = k + 1) { sum = sum + A[u32(row + k)]; }
  X[i] = sum / f32(hi - lo + 1);
}

@compute @workgroup_size(64)
fn box_v(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let y = i32(i / P.w);
  let lo = loY(y);
  let hi = hiY(y);
  var sum = 0.0;
  for (var k = lo; k <= hi; k = k + 1) { sum = sum + A[u32(k * w + x)]; }
  X[i] = sum / f32(hi - lo + 1);
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  X[i] = A[i] * B[i];
}

// Boundary band = dilate - erode, plus the band population count.
@compute @workgroup_size(64)
fn bandmask(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let inBand = A[i] > 0.5 && B[i] < 0.5;
  X[i] = select(0.0, 1.0, inBand);
  if (inBand) { atomicAdd(&CNT, 1u); }
}

// Guided filter coefficients (He et al.): a = cov(I,p)/(var(I)+eps), b = p̄ - a·Ī
@compute @workgroup_size(64)
fn ab(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  let mI  = A[i];
  let mP  = B[i];
  let mIp = C[i];
  let mII = D[i];
  let av = (mIp - mI * mP) / ((mII - mI * mI) + P.eps);
  X[i] = av;
  Y[i] = mP - av * mI;
}

// q = ā·I + b̄ inside the band; the decoder's binary decision outside it.
// Pixels outside the crop are never visited — DU was pre-seeded with the
// input mask, so they pass through byte-for-byte.
@compute @workgroup_size(64)
fn compose(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = idx(g);
  if (i == NONE) { return; }
  var v : u32;
  if (D[i] > 0.5) {
    let q = clamp(A[i] * C[i] + B[i], 0.0, 1.0);
    var t = u32(round(q * 255.0));
    // Snap near-extremes so the band doesn't carry a faint fog.
    if (t < 10u) { t = 0u; } else if (t > 245u) { t = 255u; }
    v = t;
  } else {
    v = select(0u, 255u, E[i] > 0.5);
  }
  DU[i] = v | (v << 8u) | (v << 16u) | 0xff000000u;
}
`

/* ─── Device + pipelines ─────────────────────────────────────────────────── */

const ENTRIES = ['binarize', 'morph_h', 'morph_v', 'box_h', 'box_v', 'mul', 'bandmask', 'ab', 'compose']

// Must match cropMargin() in edge-refine.js — both paths crop identically.
const cropMargin = (band, radius) => band + 3 * radius

let broken = false      // sticky: one failure and the session stays on CPU
let gpuPromise = null
let res = null          // per-shape resources (buffers, bind-group cache)

/** Own adapter/device, independent of whatever transformers.js holds — a
 *  GPUAdapter cannot be shared, and our kernels must not die with an ORT
 *  session. */
const getGpu = () => {
    if (broken) return Promise.resolve(null)
    gpuPromise ??= (async () => {
        if (typeof navigator === 'undefined' || !navigator.gpu) return null
        const adapter = await navigator.gpu.requestAdapter()
        if (!adapter) return null
        // timestamp-query is the only way to measure what the GPU actually
        // executed, as opposed to how long the round trip took. It is gated
        // behind a flag on some platforms, so it stays strictly optional and
        // is only written to when profiling is switched on.
        const hasTimestamp = adapter.features?.has?.('timestamp-query') ?? false
        const device = await adapter.requestDevice(hasTimestamp ? { requiredFeatures: ['timestamp-query'] } : {})
        device.lost.then(() => {
            broken = true
            res = null
            console.warn('[seglab] WebGPU device lost — post pipeline falls back to CPU')
        })
        // A validation/OOM error anywhere in the pipeline is a hard demote:
        // silently wrong masks are worse than a slower correct path.
        device.onuncapturederror = (e) => {
            broken = true
            console.warn('[seglab] WebGPU error in post pipeline; falling back to CPU:', e?.error?.message)
        }
        const module = device.createShaderModule({ code: WGSL, label: 'seglab-post' })
        const pipelines = {}
        for (const name of ENTRIES) {
            pipelines[name] = device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: name },
                label: name,
            })
        }
        return { device, adapter, pipelines, hasTimestamp }
    })().catch((err) => {
        broken = true
        console.warn('[seglab] WebGPU post pipeline unavailable:', err?.message)
        return null
    })
    return gpuPromise
}

const F32 = (device, n, label) => device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
})

/** 2D dispatch grid. A dimension is capped (65535 by spec floor) and 1024²
 *  already needs 16384 groups, so a 2048² crop would blow a 1D layout. */
const gridFor = (device, count) => {
    const total = Math.max(1, Math.ceil(count / 64))
    const gx = Math.min(total, device.limits.maxComputeWorkgroupsPerDimension)
    return { gx, gy: Math.ceil(total / gx), stride: gx * 64 }
}

/** values = [w, h, r, mode, stride, cx0, cy0, cw, ch, eps] */
const writeUni = (device, buf, values) => {
    const words = new ArrayBuffer(48)
    new Uint32Array(words, 0, 9).set(values.slice(0, 9))
    new Float32Array(words, 36, 1)[0] = values[9]
    device.queue.writeBuffer(buf, 0, words)
}

const UNI_NAMES = ['uMorphMax', 'uMorphMin', 'uBox', 'uPlain', 'uBoxFull', 'uPlainFull']

/** (Re)allocate everything for one frame shape. Shapes change once per photo
 *  at most, so this is not on the hot path. */
const ensureResources = (gpu, w, h) => {
    const shape = `${w}x${h}`
    if (res && res.shape === shape) return res
    if (res) {
        for (const b of Object.values(res.buf)) b.destroy?.()
        for (const b of [res.stageMask, res.stageCnt, res.tsResolve, res.stageTs]) b?.destroy?.()
        res.querySet?.destroy?.()
    }
    const { device } = gpu
    const n = w * h
    const buf = {}
    for (const name of ['p', 'tmp', 'dil', 'ero', 'band', 'meanP', 'Ip', 'a', 'b', 'gray', 'meanI', 'meanII']) {
        buf[name] = F32(device, n, name)
    }
    buf.maskIn = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    buf.maskOut = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    buf.cnt = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    for (const name of UNI_NAMES) {
        buf[name] = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    }
    res = {
        shape,
        buf,
        binds: new Map(),
        guideKey: null,
        stageMask: device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        stageCnt: device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        full: gridFor(device, n),
        n,
        w,
        h,
    }
    if (gpu.hasTimestamp) {
        res.querySet = device.createQuerySet({ type: 'timestamp', count: 2 })
        res.tsResolve = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
        res.stageTs = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    }
    return res
}

/* ─── Dispatch helpers ───────────────────────────────────────────────────── */

const bindGroup = (gpu, r, name, entries) => {
    const key = `${name}|${entries.map(([b, n]) => `${b}:${n}`).join(',')}`
    let bg = r.binds.get(key)
    if (!bg) {
        bg = gpu.device.createBindGroup({
            layout: gpu.pipelines[name].getBindGroupLayout(0),
            entries: entries.map(([binding, buf]) => ({ binding, resource: { buffer: r.buf[buf] } })),
        })
        r.binds.set(key, bg)
    }
    return bg
}

const run = (pass, gpu, r, grid, name, entries) => {
    pass.setPipeline(gpu.pipelines[name])
    pass.setBindGroup(0, bindGroup(gpu, r, name, entries))
    pass.dispatchWorkgroups(grid.gx, grid.gy)
}

/** Separable box mean src → dst via `tmp`. `dst === src` is safe: the
 *  horizontal pass has already consumed src by the time the vertical pass
 *  writes. Dispatches inside one compute pass are ordered with implicit
 *  memory barriers, which is what makes the chaining below correct. */
const boxMean = (pass, gpu, r, grid, uni, src, dst) => {
    run(pass, gpu, r, grid, 'box_h', [[0, uni], [1, src], [5, 'tmp']])
    run(pass, gpu, r, grid, 'box_v', [[0, uni], [1, 'tmp'], [5, dst]])
}

/* ─── Public entry point ─────────────────────────────────────────────────── */

export const gpuPostAvailable = () => !broken

// Profiling writes GPU timestamps around the compute pass and reports true
// on-device nanoseconds. Off by default: it adds a second readback.
let profiling = false
export const setGpuProfiling = (on) => { profiling = !!on }

/** What the post pipeline is actually running on — for diagnostics and for
 *  proving the GPU path is live rather than silently falling back. */
export const getGpuInfo = async () => {
    const gpu = await getGpu()
    if (!gpu) return { available: false, broken }
    const info = gpu.adapter.info || {}
    return {
        available: true,
        broken,
        vendor: info.vendor ?? null,
        architecture: info.architecture ?? null,
        device: info.device ?? null,
        description: info.description ?? null,
        hasTimestamp: gpu.hasTimestamp,
        maxComputeInvocationsPerWorkgroup: gpu.device.limits.maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize,
    }
}

/**
 * GPU twin of `refineMaskEdges`. Mutates `rgba` in place on success.
 *
 * @param {Uint8ClampedArray} rgba   white-on-black mask (binary going in)
 * @param {number} w
 * @param {number} h
 * @param {Float32Array} gray        0..1 guide image, w*h
 * @param {{band?:number, radius?:number, eps?:number, guideKey?:string,
 *          bbox?:number[]|null}} opts
 * @returns {Promise<{bandPixels:number, backend:'gpu', cropFraction:number}|null>}
 *   null ⇒ caller must use the CPU path
 */
export const gpuRefineMaskEdges = async (rgba, w, h, gray, opts = {}) => {
    const { band = 6, radius = 8, eps = 1e-3, guideKey = null, bbox = null } = opts
    const n = w * h
    if (!n || !gray || gray.length !== n || rgba.length !== n * 4) return null

    const gpu = await getGpu()
    if (!gpu || broken) return null
    // A frame too large for this device's storage bindings is not a failure of
    // the GPU path — it is one frame the CPU should take. Bail WITHOUT the
    // sticky demote, or a single oversized image costs every later click too.
    if (n * 4 > gpu.device.limits.maxStorageBufferBindingSize) return null

    try {
        const r = ensureResources(gpu, w, h)
        const { device } = gpu
        const q = device.queue

        // Crop rectangle — identical derivation to the CPU path.
        const margin = cropMargin(band, radius)
        const cx0 = bbox ? Math.max(0, bbox[0] - margin) : 0
        const cy0 = bbox ? Math.max(0, bbox[1] - margin) : 0
        const cx1 = bbox ? Math.min(w - 1, bbox[2] + margin) : w - 1
        const cy1 = bbox ? Math.min(h - 1, bbox[3] + margin) : h - 1
        if (cx1 < cx0 || cy1 < cy0) return { bandPixels: 0, backend: 'gpu', cropFraction: 0 }
        const cw = cx1 - cx0 + 1
        const ch = cy1 - cy0 + 1
        const crop = gridFor(device, cw * ch)

        writeUni(device, r.buf.uMorphMax, [w, h, band, 1, crop.stride, cx0, cy0, cw, ch, eps])
        writeUni(device, r.buf.uMorphMin, [w, h, band, 0, crop.stride, cx0, cy0, cw, ch, eps])
        writeUni(device, r.buf.uBox, [w, h, radius, 0, crop.stride, cx0, cy0, cw, ch, eps])
        writeUni(device, r.buf.uPlain, [w, h, 0, 0, crop.stride, cx0, cy0, cw, ch, eps])
        writeUni(device, r.buf.uBoxFull, [w, h, radius, 0, r.full.stride, 0, 0, w, h, eps])
        writeUni(device, r.buf.uPlainFull, [w, h, 0, 0, r.full.stride, 0, 0, w, h, eps])

        const t0 = performance.now()
        const freshGuide = guideKey === null || r.guideKey !== guideKey
        if (freshGuide) q.writeBuffer(r.buf.gray, 0, gray.buffer, gray.byteOffset, gray.byteLength)
        q.writeBuffer(r.buf.maskIn, 0, rgba.buffer, rgba.byteOffset, rgba.byteLength)
        q.writeBuffer(r.buf.cnt, 0, new Uint32Array([0]))
        const tUploaded = performance.now()

        const useTs = profiling && gpu.hasTimestamp && !!r.querySet
        const enc = device.createCommandEncoder()
        // Seed the output with the input so untouched pixels pass through.
        enc.copyBufferToBuffer(r.buf.maskIn, 0, r.buf.maskOut, 0, n * 4)
        const pass = enc.beginComputePass(useTs
            ? { timestampWrites: { querySet: r.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
            : undefined)

        // Guide statistics: full frame, once per photo.
        if (freshGuide) {
            boxMean(pass, gpu, r, r.full, 'uBoxFull', 'gray', 'meanI')
            run(pass, gpu, r, r.full, 'mul', [[0, 'uPlainFull'], [1, 'gray'], [2, 'gray'], [5, 'Ip']])
            boxMean(pass, gpu, r, r.full, 'uBoxFull', 'Ip', 'meanII')
        }

        // Everything below runs on the crop only.
        run(pass, gpu, r, crop, 'binarize', [[0, 'uPlain'], [7, 'maskIn'], [5, 'p']])

        run(pass, gpu, r, crop, 'morph_h', [[0, 'uMorphMax'], [1, 'p'], [5, 'tmp']])
        run(pass, gpu, r, crop, 'morph_v', [[0, 'uMorphMax'], [1, 'tmp'], [5, 'dil']])
        run(pass, gpu, r, crop, 'morph_h', [[0, 'uMorphMin'], [1, 'p'], [5, 'tmp']])
        run(pass, gpu, r, crop, 'morph_v', [[0, 'uMorphMin'], [1, 'tmp'], [5, 'ero']])
        run(pass, gpu, r, crop, 'bandmask', [[0, 'uPlain'], [1, 'dil'], [2, 'ero'], [5, 'band'], [9, 'cnt']])

        boxMean(pass, gpu, r, crop, 'uBox', 'p', 'meanP')
        run(pass, gpu, r, crop, 'mul', [[0, 'uPlain'], [1, 'gray'], [2, 'p'], [5, 'Ip']])
        boxMean(pass, gpu, r, crop, 'uBox', 'Ip', 'Ip')          // Ip → mean(I·p)
        run(pass, gpu, r, crop, 'ab', [
            [0, 'uPlain'], [1, 'meanI'], [2, 'meanP'], [3, 'Ip'], [4, 'meanII'], [5, 'a'], [6, 'b'],
        ])
        boxMean(pass, gpu, r, crop, 'uBox', 'a', 'a')            // a → ā
        boxMean(pass, gpu, r, crop, 'uBox', 'b', 'b')            // b → b̄
        run(pass, gpu, r, crop, 'compose', [
            [0, 'uPlain'], [1, 'a'], [2, 'b'], [3, 'gray'], [4, 'band'], [10, 'p'], [8, 'maskOut'],
        ])

        pass.end()
        // Only the crop's rows can have changed, and rows are contiguous —
        // read back that byte range instead of the whole frame. getMappedRange
        // wants an 8-byte-aligned offset, so round the start down.
        let rbStart = cy0 * w * 4
        rbStart -= rbStart % 8
        const rbSize = (cy1 + 1) * w * 4 - rbStart
        enc.copyBufferToBuffer(r.buf.maskOut, rbStart, r.stageMask, rbStart, rbSize)
        enc.copyBufferToBuffer(r.buf.cnt, 0, r.stageCnt, 0, 4)
        if (useTs) {
            enc.resolveQuerySet(r.querySet, 0, 2, r.tsResolve, 0)
            enc.copyBufferToBuffer(r.tsResolve, 0, r.stageTs, 0, 16)
        }
        q.submit([enc.finish()])
        const tSubmitted = performance.now()

        const maps = [
            r.stageMask.mapAsync(GPUMapMode.READ, rbStart, rbSize),
            r.stageCnt.mapAsync(GPUMapMode.READ),
        ]
        if (useTs) maps.push(r.stageTs.mapAsync(GPUMapMode.READ))
        await Promise.all(maps)
        if (broken) { // onuncapturederror fired mid-flight
            r.stageMask.unmap()
            r.stageCnt.unmap()
            if (useTs) r.stageTs.unmap()
            return null
        }
        const bandPixels = new Uint32Array(r.stageCnt.getMappedRange().slice(0))[0]
        rgba.set(new Uint8ClampedArray(r.stageMask.getMappedRange(rbStart, rbSize)), rbStart)
        r.stageMask.unmap()
        r.stageCnt.unmap()
        let gpuNs = null
        if (useTs) {
            const ts = new BigUint64Array(r.stageTs.getMappedRange().slice(0))
            gpuNs = Number(ts[1] - ts[0])
            r.stageTs.unmap()
        }

        r.guideKey = guideKey
        return {
            bandPixels,
            backend: 'gpu',
            freshGuide,
            cropFraction: (cw * ch) / n,
            timings: {
                uploadMs: tUploaded - t0,
                encodeMs: tSubmitted - tUploaded,
                readbackMs: performance.now() - tSubmitted,
                totalMs: performance.now() - t0,
                gpuMs: gpuNs === null ? null : gpuNs / 1e6,
            },
        }
    } catch (err) {
        broken = true
        console.warn('[seglab] GPU post pipeline failed; falling back to CPU:', err?.message)
        return null
    }
}

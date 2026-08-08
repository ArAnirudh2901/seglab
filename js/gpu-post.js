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
 * the mask bytes, same `{ bandPixels }` result. It returns `null` whenever the
 * GPU path is unavailable or fails, and the caller falls back to the CPU
 * implementation. A GPU failure must never cost the user a click.
 *
 * Algorithmic note: `meanI` and `meanII` depend only on the guide image, not
 * on the mask, so they are computed ONCE per photo and reused by every click.
 * That removes 2 of the 6 box filters from the per-decode path.
 */

/* ─── Kernels ────────────────────────────────────────────────────────────────
 * One module, one shared binding namespace; `layout: 'auto'` gives each entry
 * point a layout containing only the bindings it statically uses, so the
 * 8-storage-buffers-per-stage floor is never approached (max here is 6).
 */
const WGSL = /* wgsl */ `
struct Params {
  w    : u32,
  h    : u32,
  r    : u32,
  mode : u32,   // morphology: 1 = dilate (max), 0 = erode (min)
  eps  : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
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

fn total() -> u32 { return P.w * P.h; }

// RGBA bytes -> 0/1 indicator. WebGPU buffers are little-endian, so the red
// channel is the low byte of each packed pixel.
@compute @workgroup_size(64)
fn binarize(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  X[i] = select(0.0, 1.0, (SU[i] & 0xffu) >= 128u);
}

// Separable Chebyshev morphology, clamped windows at the borders.
@compute @workgroup_size(64)
fn morph_h(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let row = i32(i) - x;
  let r = i32(P.r);
  let lo = max(x - r, 0);
  let hi = min(x + r, w - 1);
  let isMax = P.mode == 1u;
  var v = select(1.0, 0.0, isMax);
  for (var k = lo; k <= hi; k = k + 1) {
    let s = A[u32(row + k)];
    v = select(min(v, s), max(v, s), isMax);
  }
  X[i] = v;
}

@compute @workgroup_size(64)
fn morph_v(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let y = i32(i / P.w);
  let r = i32(P.r);
  let lo = max(y - r, 0);
  let hi = min(y + r, i32(P.h) - 1);
  let isMax = P.mode == 1u;
  var v = select(1.0, 0.0, isMax);
  for (var k = lo; k <= hi; k = k + 1) {
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
  let i = g.x;
  if (i >= total()) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let row = i32(i) - x;
  let r = i32(P.r);
  let lo = max(x - r, 0);
  let hi = min(x + r, w - 1);
  var sum = 0.0;
  for (var k = lo; k <= hi; k = k + 1) { sum = sum + A[u32(row + k)]; }
  X[i] = sum / f32(hi - lo + 1);
}

@compute @workgroup_size(64)
fn box_v(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  let w = i32(P.w);
  let x = i32(i % P.w);
  let y = i32(i / P.w);
  let r = i32(P.r);
  let lo = max(y - r, 0);
  let hi = min(y + r, i32(P.h) - 1);
  var sum = 0.0;
  for (var k = lo; k <= hi; k = k + 1) { sum = sum + A[u32(k * w + x)]; }
  X[i] = sum / f32(hi - lo + 1);
}

@compute @workgroup_size(64)
fn mul(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  X[i] = A[i] * B[i];
}

// Boundary band = dilate - erode, plus the band population count.
@compute @workgroup_size(64)
fn bandmask(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  let inBand = A[i] > 0.5 && B[i] < 0.5;
  X[i] = select(0.0, 1.0, inBand);
  if (inBand) { atomicAdd(&CNT, 1u); }
}

// Guided filter coefficients (He et al.): a = cov(I,p)/(var(I)+eps), b = p̄ - a·Ī
@compute @workgroup_size(64)
fn ab(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
  let mI  = A[i];
  let mP  = B[i];
  let mIp = C[i];
  let mII = D[i];
  let av = (mIp - mI * mP) / ((mII - mI * mI) + P.eps);
  X[i] = av;
  Y[i] = mP - av * mI;
}

// q = ā·I + b̄ inside the band; the decoder's binary decision outside it.
@compute @workgroup_size(64)
fn compose(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x;
  if (i >= total()) { return; }
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
        const device = await adapter.requestDevice()
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
        return { device, pipelines }
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

const UNI = (device, values) => {
    const buf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    const words = new ArrayBuffer(32)
    new Uint32Array(words, 0, 4).set(values.slice(0, 4))
    new Float32Array(words, 16, 4)[0] = values[4]
    device.queue.writeBuffer(buf, 0, words)
    return buf
}

/** (Re)allocate everything for one frame shape. Shapes change once per photo
 *  at most, so this is not on the hot path. */
const ensureResources = (gpu, w, h, band, radius, eps) => {
    const shape = `${w}x${h}:${band}:${radius}:${eps}`
    if (res && res.shape === shape) return res
    if (res) for (const b of Object.values(res.buf)) b.destroy?.()
    const { device } = gpu
    const n = w * h
    const buf = {}
    for (const name of ['p', 'tmp', 'dil', 'ero', 'band', 'meanP', 'Ip', 'a', 'b', 'gray', 'meanI', 'meanII']) {
        buf[name] = F32(device, n, name)
    }
    buf.maskIn = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    buf.maskOut = device.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    buf.cnt = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    })
    buf.uMorphMax = UNI(device, [w, h, band, 1, eps])
    buf.uMorphMin = UNI(device, [w, h, band, 0, eps])
    buf.uBox = UNI(device, [w, h, radius, 0, eps])
    buf.uPlain = UNI(device, [w, h, 0, 0, eps])
    res = {
        shape,
        buf,
        binds: new Map(),
        guideKey: null,
        stageMask: device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        stageCnt: device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
        groups: Math.ceil(n / 64),
        n,
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

const run = (pass, gpu, r, name, entries) => {
    pass.setPipeline(gpu.pipelines[name])
    pass.setBindGroup(0, bindGroup(gpu, r, name, entries))
    pass.dispatchWorkgroups(r.groups)
}

/** Separable box mean src → dst via `tmp`. `dst === src` is safe: the
 *  horizontal pass has already consumed src by the time the vertical pass
 *  writes. Dispatches inside one compute pass are ordered with implicit
 *  memory barriers, which is what makes the chaining below correct. */
const boxMean = (pass, gpu, r, src, dst) => {
    run(pass, gpu, r, 'box_h', [[0, 'uBox'], [1, src], [5, 'tmp']])
    run(pass, gpu, r, 'box_v', [[0, 'uBox'], [1, 'tmp'], [5, dst]])
}

/* ─── Public entry point ─────────────────────────────────────────────────── */

export const gpuPostAvailable = () => !broken

/**
 * GPU twin of `refineMaskEdges`. Mutates `rgba` in place on success.
 *
 * @param {Uint8ClampedArray} rgba   white-on-black mask (binary going in)
 * @param {number} w
 * @param {number} h
 * @param {Float32Array} gray        0..1 guide image, w*h
 * @param {{band?:number, radius?:number, eps?:number, guideKey?:string}} opts
 * @returns {Promise<{bandPixels:number, backend:'gpu'}|null>} null ⇒ use CPU
 */
export const gpuRefineMaskEdges = async (rgba, w, h, gray, opts = {}) => {
    const { band = 6, radius = 8, eps = 1e-3, guideKey = null } = opts
    const n = w * h
    if (!n || !gray || gray.length !== n || rgba.length !== n * 4) return null

    const gpu = await getGpu()
    if (!gpu || broken) return null

    try {
        const r = ensureResources(gpu, w, h, band, radius, eps)
        const { device, pipelines: _p } = gpu
        const q = device.queue

        // The guide statistics depend only on the photo: compute once per
        // image, then every subsequent click reuses them.
        const freshGuide = guideKey === null || r.guideKey !== guideKey
        if (freshGuide) q.writeBuffer(r.buf.gray, 0, gray.buffer, gray.byteOffset, gray.byteLength)
        q.writeBuffer(r.buf.maskIn, 0, rgba.buffer, rgba.byteOffset, rgba.byteLength)
        q.writeBuffer(r.buf.cnt, 0, new Uint32Array([0]))

        const enc = device.createCommandEncoder()
        const pass = enc.beginComputePass()

        if (freshGuide) {
            boxMean(pass, gpu, r, 'gray', 'meanI')
            run(pass, gpu, r, 'mul', [[0, 'uPlain'], [1, 'gray'], [2, 'gray'], [5, 'Ip']])
            boxMean(pass, gpu, r, 'Ip', 'meanII')
        }

        run(pass, gpu, r, 'binarize', [[0, 'uPlain'], [7, 'maskIn'], [5, 'p']])

        // Boundary band.
        run(pass, gpu, r, 'morph_h', [[0, 'uMorphMax'], [1, 'p'], [5, 'tmp']])
        run(pass, gpu, r, 'morph_v', [[0, 'uMorphMax'], [1, 'tmp'], [5, 'dil']])
        run(pass, gpu, r, 'morph_h', [[0, 'uMorphMin'], [1, 'p'], [5, 'tmp']])
        run(pass, gpu, r, 'morph_v', [[0, 'uMorphMin'], [1, 'tmp'], [5, 'ero']])
        run(pass, gpu, r, 'bandmask', [[0, 'uPlain'], [1, 'dil'], [2, 'ero'], [5, 'band'], [9, 'cnt']])

        // Guided filter against the photo.
        boxMean(pass, gpu, r, 'p', 'meanP')
        run(pass, gpu, r, 'mul', [[0, 'uPlain'], [1, 'gray'], [2, 'p'], [5, 'Ip']])
        boxMean(pass, gpu, r, 'Ip', 'Ip')                       // Ip → mean(I·p)
        run(pass, gpu, r, 'ab', [
            [0, 'uPlain'], [1, 'meanI'], [2, 'meanP'], [3, 'Ip'], [4, 'meanII'], [5, 'a'], [6, 'b'],
        ])
        boxMean(pass, gpu, r, 'a', 'a')                         // a → ā
        boxMean(pass, gpu, r, 'b', 'b')                         // b → b̄
        run(pass, gpu, r, 'compose', [
            [0, 'uPlain'], [1, 'a'], [2, 'b'], [3, 'gray'], [4, 'band'], [10, 'p'], [8, 'maskOut'],
        ])

        pass.end()
        enc.copyBufferToBuffer(r.buf.maskOut, 0, r.stageMask, 0, n * 4)
        enc.copyBufferToBuffer(r.buf.cnt, 0, r.stageCnt, 0, 4)
        q.submit([enc.finish()])

        await Promise.all([
            r.stageMask.mapAsync(GPUMapMode.READ),
            r.stageCnt.mapAsync(GPUMapMode.READ),
        ])
        if (broken) { // onuncapturederror fired mid-flight
            r.stageMask.unmap()
            r.stageCnt.unmap()
            return null
        }
        const bandPixels = new Uint32Array(r.stageCnt.getMappedRange().slice(0))[0]
        rgba.set(new Uint8ClampedArray(r.stageMask.getMappedRange()))
        r.stageMask.unmap()
        r.stageCnt.unmap()

        r.guideKey = guideKey
        return { bandPixels, backend: 'gpu' }
    } catch (err) {
        broken = true
        console.warn('[seglab] GPU post pipeline failed; falling back to CPU:', err?.message)
        return null
    }
}

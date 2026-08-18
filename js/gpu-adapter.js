/**
 * gpu-adapter — one adapter request policy + the vendor facts callers tune on.
 *
 * capability.js probed with 'high-performance' while sam21-lane asked bare. On a
 * dual-GPU Mac that is two different GPUs (Chrome: discrete on AC, integrated on
 * battery, when no preference is given), so the shader-f16 gate could clear on
 * one and the encoder run on the other. Same request everywhere, or the probe
 * proves nothing.
 *
 * The preference is honoured on macOS/Linux, ignored on Windows (Chrome cannot
 * composite across adapters), moot on mobile. So it never guarantees the fast
 * GPU — profileAdapter reports what actually came back.
 */

const MIB = 1024 * 1024

export const ADAPTER_OPTIONS = { powerPreference: 'high-performance' }

/** The browser may decline the preference; a bare request is still valid. */
export const requestAdapter = async () => {
    const gpu = typeof navigator === 'undefined' ? null : navigator.gpu
    if (!gpu) return null
    try {
        const adapter = await gpu.requestAdapter(ADAPTER_OPTIONS) || await gpu.requestAdapter()
        // Sync `adapter.info` replaced async `requestAdapterInfo()`. Without this
        // the older browsers see an empty profile and a software rasteriser
        // clears the gate on isFallbackAdapter alone — which SwiftShader fails to
        // set. profileAdapter stays sync, so stash it where it now lives.
        if (adapter && !adapter.info && adapter.requestAdapterInfo) {
            try { adapter.info = await adapter.requestAdapterInfo() } catch { /* leave empty */ }
        }
        return adapter
    } catch { return null }
}

// Chrome reaches SwiftShader by blocklisting the real driver, which leaves
// isFallbackAdapter FALSE while every capability answer stays yes. Match the
// known software names too, or a CPU rasteriser passes the f16 gate.
const SOFTWARE = /swiftshader|llvmpipe|softpipe|\bwarp\b|basic render|microsoft basic|lavapipe/i

// Apple is deliberately absent: unified memory is not weakness.
const INTEGRATED_VENDORS = new Set(['intel', 'arm', 'qualcomm', 'imagination', 'img', 'broadcom', 'mesa'])
// Intel ships discrete parts, so vendor alone would demote an Arc card.
const INTEL_IGPU_ARCH = /^(gen-|xe-lp|xe2-lp)/i

// The four backends that carry this app: d3d12 (Windows), metal (macOS/iOS),
// vulkan (Linux/Android/ChromeOS), and d3d11 — which Chrome reaches only on
// hardware too old for d3d12. There is no CUDA path: an NVIDIA card is driven
// through d3d12 or vulkan like any other.
//
// d3d11/opengl/opengles are Dawn's compatibility backends. They run the lane but
// with a narrower compute path, so they demote rather than refuse — and only
// when Chrome actually names the backend, which it does not by default.
const LEGACY_BACKENDS = /^(d3d11|opengl|opengles)$/

/**
 * Every field may be '' — browsers minimise these against fingerprinting — so
 * callers must read unknown as "no opinion", never "weak". `type`/`backend`
 * exist only behind Chrome's WebGPU Developer Features flag.
 */
export const profileAdapter = (adapter) => {
    if (!adapter) return null
    const info = adapter.info || {}
    const vendor = String(info.vendor || '').toLowerCase()
    const architecture = String(info.architecture || '').toLowerCase()
    const description = String(info.description || info.device || '').toLowerCase()
    const type = String(info.type || '').toLowerCase()
    const backend = String(info.backend || '').toLowerCase()
    const software = !!adapter.isFallbackAdapter || type === 'cpu'
        || SOFTWARE.test(`${vendor} ${architecture} ${description}`)
    const integrated = type === 'integrated gpu'
        || (!type && (vendor === 'intel'
            ? (!architecture || INTEL_IGPU_ARCH.test(architecture))
            : INTEGRATED_VENDORS.has(vendor)))
    return {
        vendor,
        architecture,
        description,
        type,
        backend,
        software,
        integrated: !software && integrated,
        legacyBackend: LEGACY_BACKENDS.test(backend),
        f16: adapter.features?.has?.('shader-f16') || false,
        subgroups: adapter.features?.has?.('subgroups') || false,
        textureLimit: adapter.limits?.maxTextureDimension2D || 0,
        storageBufferLimit: adapter.limits?.maxStorageBufferBindingSize || 0,
        name: [vendor, architecture].filter(Boolean).join(' ') || description || '',
    }
}

/**
 * Limits for a device we build ourselves (mask-lane loss recovery).
 *
 * Copying every adapter limit at max is the documented anti-pattern: the spec
 * has implementations warn about it and wgpu notes the driver then supports more
 * than the app uses — D3D12 drops a resource-heap tier, Metal widens the
 * argument buffer. This lane is fp16 compute over storage buffers, so raise only
 * what ORT allocates against and leave every render limit at its default.
 */
const COMPUTE_LIMITS = [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxUniformBufferBindingSize',
    'maxStorageBuffersPerShaderStage',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
    'maxComputeWorkgroupsPerDimension',
    'maxBindGroups',
    'maxBindingsPerBindGroup',
    'maxDynamicStorageBuffersPerPipelineLayout',
]

export const computeLimits = (adapter) => {
    const out = {}
    for (const k of COMPUTE_LIMITS) {
        const v = adapter?.limits?.[k]
        if (typeof v === 'number' && Number.isFinite(v)) out[k] = v
    }
    return out
}

/** Encoder workspace floor, so a refusal names the short limit. */
export const MIN_STORAGE_BUFFER = 128 * MIB

/**
 * Backend the adapter is really running on.
 *
 * `adapter.info.backend` is empty unless Chrome's WebGPU Developer Features flag
 * is on, so on a normal browser it never answers. The OS decides the backend
 * outright, so infer it — reporting only. It must NOT feed `legacyBackend`: d3d11
 * vs d3d12 is exactly the distinction this cannot make.
 */
export const inferBackend = (profile, ua = typeof navigator === 'undefined' ? null : navigator) => {
    if (profile?.backend) return profile.backend
    if (profile?.software) return 'cpu'
    const platform = String(ua?.userAgentData?.platform || '').toLowerCase()
    const s = `${platform} ${ua?.userAgent || ''}`
    if (/mac|iphone|ipad|ios/i.test(s)) return 'metal'
    if (/windows|win32|win64/i.test(s)) return 'd3d12'
    if (/android|linux|cros|chrome ?os/i.test(s)) return 'vulkan'
    return ''
}

/**
 * Workgroup width for the preproc kernel, from the device that will run it.
 *
 * Measured, Apple metal-3, 1024² frame, 15 runs, median ms — identical output
 * checksums, so every size computes the same buffer:
 *   32 → 0.75 · 64 → 0.47 · 128 → 0.47 · 256 → 0.42 · 512 → 0.115 · 1024 → 0.115
 * 512 is the knee: 4.1x the 64 this shipped with, and past it nothing. The win is
 * dispatch overhead, not bandwidth — the kernel is two texel loads and a store.
 *
 * WebGPU only GUARANTEES 256 invocations per workgroup, so this reads the device
 * rather than hard-coding the Metal number: ORT's own device takes the defaults
 * and lands on 256, the recovery device raises limits and reaches 512. Both beat
 * 64 and neither can fail pipeline creation.
 */
export const preprocWorkgroup = (device) => {
    const lim = device?.limits || {}
    const cap = Math.min(
        Number(lim.maxComputeInvocationsPerWorkgroup) || 256,
        Number(lim.maxComputeWorkgroupSizeX) || 256,
        512,
    )
    return cap >= 64 ? cap : 64
}

let gpuUsable = null
/**
 * Should a worker lane offer ORT the WebGPU EP at all?
 *
 * A software adapter builds the session happily and then runs the detector at
 * CPU speed through a shader compiler — strictly worse than the WASM EP, which
 * at least threads. Memoised: one adapter request per worker.
 */
export const webgpuWorthTrying = () => {
    gpuUsable ??= requestAdapter().then((a) => {
        const gpu = profileAdapter(a)
        return !!gpu && !gpu.software
    }).catch(() => false)
    return gpuUsable
}

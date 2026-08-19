/**
 * capability — device and resource-budget probe (main thread).
 *
 * A browser can request a high-performance WebGPU adapter, but it deliberately
 * cannot expose reliable VRAM, and browser memory reports are privacy-rounded
 * and unverifiable. Consequently a plain browser build never sizes budgets
 * from a browser memory report — only from a resource budget it can trust.
 * Phosmith may supply a trusted usable-memory budget before this module runs
 * through `window.__PHOSMITH_DEVICE_RESOURCES__`; see README for the contract.
 *
 * Memory sizing and GPU acceleration are intentionally separate: RAM/VRAM
 * decides canvas/cache/export limits, while WebGPU decides how inference runs.
 * This means a powerful RTX is used when the browser permits it, but cannot
 * silently turn a low-memory device into an unsafe large-canvas configuration.
 */

import { inferBackend, profileAdapter, requestAdapter } from './gpu-adapter.js'

const MIB = 1024 * 1024
const MODES = new Set(['conservative', 'balanced', 'performance'])

const safeGB = (value, max) => {
    const n = Number(value)
    return Number.isFinite(n) && n > 0 && n <= max ? n : 0
}

/**
 * Validate an optional Phosmith host hint. `memoryBudgetGB` is deliberately a
 * *usable editor budget*, not necessarily installed RAM: the native host can
 * account for other open apps and report less than physical memory. `ramGB`
 * and `memoryGB` are accepted as convenient host aliases.
 */
export const normalizePhosmithResources = (input = null) => {
    if (!input || typeof input !== 'object') return null
    const memoryGB = safeGB(
        input.memoryBudgetGB ?? input.availableMemoryGB ?? input.ramGB ?? input.memoryGB,
        1024,
    )
    const vramGB = safeGB(input.vramGB, 256)
    const mode = MODES.has(input.mode) ? input.mode : 'balanced'
    const allowFlagship = input.allowFlagship === true
    const gpuName = typeof input.gpuName === 'string' && input.gpuName.length <= 120
        ? input.gpuName.trim()
        : ''
    // A GPU-only hint is still useful telemetry, but it never raises the
    // memory profile without a trusted memory budget.
    if (!memoryGB && !vramGB && !allowFlagship && !gpuName && mode === 'balanced') return null
    return { memoryGB, vramGB, mode, allowFlagship, gpuName }
}

/** Read the host hint at boot. It is intentionally a plain injected object so
 * the same bundle works in a web page, an iframe, and a Phosmith WebView. */
export const readPhosmithResources = () => normalizePhosmithResources(
    typeof globalThis === 'undefined' ? null : globalThis.__PHOSMITH_DEVICE_RESOURCES__,
)


// Trusted-host tier ladder (usable editor budget the host vouches for, not
// installed RAM). Unverified budgets never enter it.


/**
 * No f16 is now 'none', not 'basic'. checkDevice refuses such an adapter and
 * there is no lane to demote to, so 'basic' claimed acceleration the app cannot
 * deliver — and policy read this field to set `samWebGPU`. 'basic' now means
 * what it says: a real f16 GPU whose limits sit under what the lane sizes for.
 * `software` is separate from `fallback`: a blocklisted driver puts Chrome on
 * SwiftShader with isFallbackAdapter false.
 */
const gpuTierFor = ({ webgpu, fallback, software, f16, legacyBackend, textureLimit, storageBufferLimit }) => {
    if (!webgpu || fallback || software || !f16) return 'none'
    const textureReady = !textureLimit || textureLimit >= 8192
    const storageReady = !storageBufferLimit || storageBufferLimit >= 128 * MIB
    // d3d11/GL is Chrome's compatibility path — hardware too old for d3d12.
    return textureReady && storageReady && !legacyBackend ? 'accelerated' : 'basic'
}

/**
 * The highest tier an UNVERIFIED browser may auto-run, chosen from signals that
 * cannot be spoofed *upward*:
 *   - `logicalProcessors` (hardwareConcurrency): not capped, a device-class proxy.
 *   - `gpuTier`: a usable, non-fallback WebGPU adapter — REQUIRED. It used to be
 *     required because the WASM alternative held a ~3 GB ORT heap (measured);
 *     now it is required because there is no alternative at all — the mask lane
 *     refuses an adapter without shader-f16. 'accelerated' is the strongest signal.
 *   - `deviceMemory`: used only DOWNWARD — a genuine sub-8 reading demotes; a
 *     reading of 8 (the privacy cap) never raises.
 *   - `mobile`: phones/tablets stay lite (small RAM, thermal throttling).
 * Capped at `standard8` — pro/ultra are Phosmith-verified-only or manual override.
 * A trusted host returns null (classifyCapability already has a real figure).
 */

const proxyFor = (profile, gpuTier, textureLimit) => {
    // The encoder resizes every input to a 1024 long edge, so 1024 is the baseline
    // that feeds the model its exact native frame (and a crisp preview) at no
    // extra model cost. Bigger values improve only interaction/preview
    // precision, so GPU strength earns a bounded increase above that.
    let size = ({ lite: 1024, standard: 1024, pro: 1280, ultra: 1536 }[profile] || 1024)
    if (gpuTier === 'none') size = Math.min(size, profile === 'ultra' ? 1280 : 1024)
    if (gpuTier === 'basic') size = Math.min(size, profile === 'ultra' ? 1280 : 1152)
    if (textureLimit && textureLimit < 4096) size = Math.min(size, 768)
    return size
}

/**
 * Pure capability classifier. Exported so the policy can be tested without a
 * browser and so a live Phosmith resource update does not need another GPU
 * adapter request. `hostResources` is treated as trusted only after the
 * normalizer has accepted it.
 */
export const classifyCapability = (input = {}) => {
    const host = normalizePhosmithResources(input.hostResources)
    // Telemetry only — never trusted for tier selection, so no special cap.
    const browserMemoryGB = safeGB(input.browserMemoryGB ?? input.deviceMemoryGB, 1024)
    const memoryGB = host?.memoryGB || browserMemoryGB
    const memorySource = host?.memoryGB ? 'phosmith' : (browserMemoryGB ? 'browser' : 'unknown')
    const resourceMode = host?.mode || 'balanced'
    const gpuTier = gpuTierFor(input)
    const vramGB = host?.vramGB || 0
    const cores = Number(input.logicalProcessors) || 0
    const mobile = !!input.mobile
    // One configuration (§11): there is no tier to estimate. `profile` is a
    // stable label for telemetry and for proxyFor's size table, not a choice.
    const profile = 'standard8'

    return {
        webgpu: !!input.webgpu,
        fallback: !!input.fallback,
        software: !!input.software,
        f16: !!input.f16,
        subgroups: !!input.subgroups,
        // Adapter identity. May be '' (browsers minimise it) — read unknown as
        // "no opinion", never "weak".
        gpuVendor: input.gpuVendor || '',
        gpuArchitecture: input.gpuArchitecture || '',
        // 'd3d12' | 'metal' | 'vulkan' | 'd3d11' | … — '' unless Chrome's WebGPU
        // Developer Features flag is on, so never a precondition for anything.
        gpuBackend: input.gpuBackend || '',
        legacyBackend: !!input.legacyBackend,
        integratedGPU: !!input.integratedGPU,
        // Keep this public alias for existing integrations/tests.
        deviceMemoryGB: browserMemoryGB,
        browserMemoryGB,
        memoryGB,
        memorySource,
        vramGB,
        resourceMode,
        hostManaged: memorySource === 'phosmith',
        allowFlagship: !!host?.allowFlagship,
        gpuName: host?.gpuName || input.gpuName || '',
        gpuTier,
        logicalProcessors: cores,
        mobile,
        textureLimit: Number(input.textureLimit) || 0,
        storageBufferLimit: Number(input.storageBufferLimit) || 0,
        gpuPreference: 'high-performance',
        profile,
        proxyMax: proxyFor(profile, gpuTier, Number(input.textureLimit) || 0),
        // Kept as a stable diagnostic field for existing host integrations; it
        // is deliberately never eligible.
        flagshipEligible: false,
    }
}

/** Reclassify a probed adapter using a new host budget (for example after
 * Phosmith receives an OS memory-pressure notification). */
export const withPhosmithResources = (capability = {}, resources = null) => classifyCapability({
    ...capability,
    browserMemoryGB: capability.browserMemoryGB ?? capability.deviceMemoryGB,
    hostResources: resources,
})

export const probeCapability = async ({ hostResources = readPhosmithResources() } = {}) => {
    const nav = typeof navigator === 'undefined' ? {} : navigator
    // Phones/tablets stay lite regardless of cores/GPU (small RAM, thermal).
    // userAgentData.mobile is the reliable signal; fall back to a UA sniff.
    const mobile = typeof nav.userAgentData?.mobile === 'boolean'
        ? nav.userAgentData.mobile
        : /Mobi|Android|iPhone|iPad|iPod/i.test(String(nav.userAgent || ''))
    const raw = {
        webgpu: false,
        fallback: false,
        software: false,
        f16: false,
        subgroups: false,
        browserMemoryGB: nav.deviceMemory || 0,
        logicalProcessors: nav.hardwareConcurrency || 0,
        mobile,
        textureLimit: 0,
        storageBufferLimit: 0,
        hostResources,
    }
    try {
        // Same request the mask lane makes — see gpu-adapter.
        const adapter = await requestAdapter()
        const gpu = profileAdapter(adapter)
        if (gpu) {
            raw.webgpu = true
            raw.fallback = !!adapter.isFallbackAdapter
            raw.software = gpu.software
            raw.f16 = gpu.f16
            raw.subgroups = gpu.subgroups
            raw.textureLimit = gpu.textureLimit
            raw.storageBufferLimit = gpu.storageBufferLimit
            raw.gpuVendor = gpu.vendor
            raw.gpuArchitecture = gpu.architecture
            raw.gpuBackend = inferBackend(gpu)
            raw.legacyBackend = gpu.legacyBackend
            raw.integratedGPU = gpu.integrated
            raw.gpuName = gpu.name
        }
    } catch { /* WebGPU is optional here; the lane gates on it separately. */ }
    return classifyCapability(raw)
}

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

const MIB = 1024 * 1024
const PROFILES = ['lite', 'standard', 'pro', 'ultra']
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

const lowerProfile = (profile) => ({ ultra: 'pro', pro: 'standard', standard: 'standard', lite: 'lite' }[profile] || 'standard')

// Trusted-host tier ladder (usable editor budget the host vouches for, not
// installed RAM). Unverified budgets never enter it.
const TIER_MIN_GB = { standard: 8, pro: 12, ultra: 24 }

const profileForMemory = (memoryGB, trusted, mode) => {
    // Browser memory reports are unverifiable, so they never pick a tier —
    // an untrusted budget always resolves to the conservative baseline.
    if (!trusted) return 'lite'
    let profile = 'standard'
    if (memoryGB > 0 && memoryGB < TIER_MIN_GB.standard) profile = 'lite'
    else if (memoryGB >= TIER_MIN_GB.ultra) profile = 'ultra'
    else if (memoryGB >= TIER_MIN_GB.pro) profile = 'pro'
    if (mode === 'conservative') profile = lowerProfile(profile)
    return profile
}

const gpuTierFor = ({ webgpu, fallback, f16, textureLimit, storageBufferLimit }) => {
    if (!webgpu || fallback) return 'none'
    const textureReady = !textureLimit || textureLimit >= 8192
    const storageReady = !storageBufferLimit || storageBufferLimit >= 128 * MIB
    return f16 && textureReady && storageReady ? 'accelerated' : 'basic'
}

/**
 * The highest tier an UNVERIFIED browser may auto-run, chosen from signals that
 * cannot be spoofed *upward*:
 *   - `logicalProcessors` (hardwareConcurrency): not capped, a device-class proxy.
 *   - `gpuTier`: a usable, non-fallback WebGPU adapter — REQUIRED, because a
 *     GPU-less device runs SlimSAM on WASM, whose ORT heap holds ~3 GB (measured),
 *     so it must stay bounded at lite. 'accelerated' (f16 + healthy limits) is
 *     the strongest signal.
 *   - `deviceMemory`: used only DOWNWARD — a genuine sub-8 reading demotes; a
 *     reading of 8 (the privacy cap) never raises.
 *   - `mobile`: phones/tablets stay lite (small RAM, thermal throttling).
 * Capped at `standard8` — pro/ultra are Phosmith-verified-only or manual override.
 * A trusted host returns null (classifyCapability already has a real figure).
 */
const autoTierFor = ({ trusted, cores, memoryGB, gpuTier, mobile }) => {
    if (trusted) return null
    if (mobile) return 'lite'
    if (gpuTier === 'none') return 'lite'            // WASM-only → ~3 GB risk, stay bounded
    if (memoryGB > 0 && memoryGB < 8) return 'lite'  // a real sub-8 reading is trusted down
    if (!cores || cores < 6) return 'lite'           // no / low multi-core signal
    // Usable GPU (SlimSAM runs at ~0.5 GB on WebGPU, not ~3 GB on WASM) + real
    // multi-core. Accelerated adapter or a strong core count earns standard8.
    return (gpuTier === 'accelerated' || cores >= 8) ? 'standard8' : 'lite'
}

const proxyFor = (profile, gpuTier, textureLimit) => {
    // SlimSAM resizes every input to a 1024 long edge, so 1024 is the baseline
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
    const profile = profileForMemory(memoryGB, memorySource === 'phosmith', resourceMode)
    const gpuTier = gpuTierFor(input)
    const vramGB = host?.vramGB || 0
    const cores = Number(input.logicalProcessors) || 0
    const mobile = !!input.mobile
    // The tier an unverified device may auto-run (capped at standard8, GPU- and
    // core-gated). Applied by resolveBudget as the locked-budget default.
    const autoTier = autoTierFor({ trusted: memorySource === 'phosmith', cores, memoryGB, gpuTier, mobile })

    return {
        webgpu: !!input.webgpu,
        fallback: !!input.fallback,
        f16: !!input.f16,
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
        // The tier an unverified device auto-runs (null once Phosmith supplies a
        // real figure). Applied by resolveBudget; the manual toggle can exceed it.
        autoTier,
        estimatedProfile: autoTier, // back-compat alias for the profile-toggle UI
        proxyMax: proxyFor(profile, gpuTier, Number(input.textureLimit) || 0),
        // Segmentation is SlimSAM-only. Kept as a stable diagnostic field for
        // existing host integrations; it is deliberately never eligible.
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
        f16: false,
        browserMemoryGB: nav.deviceMemory || 0,
        logicalProcessors: nav.hardwareConcurrency || 0,
        mobile,
        textureLimit: 0,
        storageBufferLimit: 0,
        hostResources,
    }
    try {
        if (nav.gpu) {
            // The browser may decline the preference (for example on battery),
            // so a regular request remains a valid fallback.
            const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' })
                || await nav.gpu.requestAdapter()
            if (adapter) {
                raw.webgpu = true
                raw.fallback = !!adapter.isFallbackAdapter
                raw.f16 = adapter.features?.has?.('shader-f16') || false
                raw.textureLimit = adapter.limits?.maxTextureDimension2D || 0
                raw.storageBufferLimit = adapter.limits?.maxStorageBufferBindingSize || 0
            }
        }
    } catch { /* WebGPU is optional; the engine has a WASM lane. */ }
    return classifyCapability(raw)
}

export const RESOURCE_PROFILES = PROFILES

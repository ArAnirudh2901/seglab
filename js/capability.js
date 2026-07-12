/**
 * capability — one boot-time device probe → profile (main thread).
 * WebGPU adapter (real vs software) + deviceMemory decide Lite/Standard/Pro;
 * policy.js turns the profile into a budget. Runs once at boot; ?profile=
 * overrides it. A GPU that is present but slow is caught reactively by the
 * pressure ladder (sam-engine.relievePressure), not a boot micro-benchmark.
 *
 * deviceMemory caps at 8 GB and only appears in secure contexts (localhost/
 * https) and Chromium — absent (Safari/Firefox) ⇒ treated as unknown, which
 * lands on Standard (never assume 8 GB we can't confirm).
 */

export const probeCapability = async () => {
    const cap = { webgpu: false, fallback: false, f16: false, deviceMemoryGB: navigator.deviceMemory || 0, profile: 'lite' }
    try {
        if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter()
            if (adapter) {
                cap.webgpu = true
                cap.fallback = !!adapter.isFallbackAdapter
                cap.f16 = adapter.features?.has?.('shader-f16') || false
            }
        }
    } catch { /* no WebGPU */ }

    const usableGpu = cap.webgpu && !cap.fallback
    const mem = cap.deviceMemoryGB
    if (!usableGpu) cap.profile = 'lite'                 // wasm-only device
    else if (mem >= 8) cap.profile = 'pro'              // real GPU + full memory
    else if (mem >= 4 || mem === 0) cap.profile = 'standard' // GPU, constrained/unknown memory
    else cap.profile = 'lite'                            // GPU but <4 GB → constrain
    return cap
}

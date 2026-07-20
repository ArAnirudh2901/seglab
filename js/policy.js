/**
 * policy — device profiles and runtime budgets (main thread).
 * One budget object gates proxy size, residency, cache caps, escalation,
 * HD export. Profiles scale HOW MUCH hardware a feature gets, never WHETHER.
 *
 * Memory-trust lock: a plain browser cannot verify its real memory headroom
 * (`navigator.deviceMemory` is privacy-rounded and capped), so any budget not
 * backed by a trusted Phosmith host hint is FORCED to the `lite` profile and
 * URL parameters cannot raise its limits. Precedence:
 *   hard device safety limit > memory pressure > URL parameter > feature request.
 *
 * Residency is blob-only: a file upload is never held as full-res RGBA (asset-
 * store decodes straight to the proxy and re-decodes bounded regions on demand).
 */

const PRESETS = {
    lite: {
        profile: 'lite',
        proxyMax: 1024,          // = SlimSAM's own internal encode edge; below it
                                 // the model upscales (softer input) for no memory
                                 // saving. The pressure floor drops it.
        proxyMode: 'auto',
        // On-screen preview frame, decoupled from the model proxy: a GPU-resident
        // display bitmap (not RGBA/tensor) so the photo looks crisp while the
        // model keeps its bounded ≤proxyMax buffer. Sized bounded-safe per host.
        displayMax: 2048,
        displayMode: 'auto',     // 'auto' safe | 'native' opts into a full-res decode
        directMaxMP: 2,
        directMaxSide: 2048,
        cropMaxSide: 1280,
        exportMaxSide: 4096,
        exportMaxMP: 8,
        escalateMaxMP: 8,
        draftCacheMax: 1,        // exactly one resident embedding
        flagshipCacheMax: 0,
        maxResidentHeavy: 1,
        flagship: false,
        // Only enables the 151 MB Grounding DINO q4f16 attempt after the
        // accelerator/f16 gate in sam-engine. OWLv2 still stays on WASM.
        detectorWebGPU: true,
        // Detector session stays warm across searches, evicted before the next
        // SAM encode: dispose:'now''s memory ceiling without its per-search
        // WebGPU rebuild (~13 s here).
        detectorEvictOnEncode: true,
        // Reclaim the warm session's (unified) GPU memory after this much
        // quiet time — an idle 151 MB model must not pin an 8 GB host into
        // swap. A later search pays one rebuild.
        detectorIdleMs: 45_000,
        // Baseline default: WASM. resolveBudget() promotes this to WebGPU
        // whenever a usable non-fallback adapter is probed (gpuTier != 'none'),
        // since GPU accel is independent of the memory tier. This false is the
        // no-GPU / no-probe floor and what memory pressure ratchets back down to.
        samWebGPU: false,
        autoEscalate: false,
        hdExportDecode: false,
        detectorDispose: 'idle',
        eagerEncode: true,       // bounded: one embedding, calm-gated, wasm lane
        cvRefine: true,          // wasm mask cleanup (skipped at pressure ≥ 2)
        rawDevelop: true,        // LibRaw develop for preview-less RAW; lazy,
                                 // disposed after use, off at pressure ≥ 2
        rawDevelopMaxMP: 40,     // refuse larger sensors here (bounds the peak)
        embedPersist: false,     // OPFS embedding persistence off (packing peaks)
        workingMaxSide: 1280,    // bounded re-decode copy cap (Safari-shaped hosts)
        pressureLevel: 0,
    },
    standard: {
        profile: 'standard',
        proxyMax: 1024,
        proxyMode: 'auto',
        displayMax: 2560,
        displayMode: 'auto',
        directMaxMP: 4,
        directMaxSide: 4096,
        cropMaxSide: 2048,
        exportMaxSide: 8192,
        exportMaxMP: 24,
        escalateMaxMP: 24,
        draftCacheMax: 3,
        flagshipCacheMax: 2,
        maxResidentHeavy: 1,
        flagship: false,
        detectorWebGPU: true,
        samWebGPU: true,
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'idle',
        // 'standard' is now reachable by an unverified-memory device (the
        // profile estimate's ceiling), not only a trusted Phosmith host — so
        // it gets the same encode-time detector eviction lite has, not just
        // pro/ultra's assumption of ample headroom.
        detectorEvictOnEncode: true,
        detectorIdleMs: 120_000,
        eagerEncode: true,
        cvRefine: true,
        rawDevelop: true,
        rawDevelopMaxMP: 60,
        embedPersist: true,
        workingMaxSide: 4096,
        pressureLevel: 0,
    },
    pro: {
        profile: 'pro',
        proxyMax: 1280,
        proxyMode: 'auto',
        displayMax: 3200,
        displayMode: 'auto',
        directMaxMP: 6,
        directMaxSide: 6144,
        cropMaxSide: 3072,
        exportMaxSide: 10000,
        exportMaxMP: 36,
        escalateMaxMP: 36,
        draftCacheMax: 4,
        flagshipCacheMax: 3,
        maxResidentHeavy: 2,
        flagship: false,         // retired: SlimSAM is the only segmentation lane
        detectorWebGPU: true,
        samWebGPU: true,
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'idle',
        detectorIdleMs: 180_000,
        eagerEncode: true,
        cvRefine: true,
        rawDevelop: true,
        rawDevelopMaxMP: 80,
        embedPersist: true,
        workingMaxSide: 4096,
        pressureLevel: 0,
    },
    ultra: {
        profile: 'ultra',
        proxyMax: 1536,
        proxyMode: 'auto',
        displayMax: 4096,
        displayMode: 'auto',
        directMaxMP: 8,
        directMaxSide: 8192,
        cropMaxSide: 4096,
        exportMaxSide: 12000,
        exportMaxMP: 64,
        escalateMaxMP: 64,
        draftCacheMax: 6,
        flagshipCacheMax: 4,
        maxResidentHeavy: 2,
        flagship: false,
        detectorWebGPU: true,
        samWebGPU: true,
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'idle',
        detectorIdleMs: 180_000,
        eagerEncode: true,
        cvRefine: true,
        rawDevelop: true,
        rawDevelopMaxMP: 100,
        embedPersist: true,
        workingMaxSide: 4096,
        pressureLevel: 0,
    },
}

/** True when nothing above `lite` can be proven: no capability yet, or the
 *  memory evidence is browser-reported/unknown (both unverifiable). */
export const isMemoryLocked = (probed = null) => {
    if (typeof probed === 'string') return false // explicit caller/test profile
    if (!probed || typeof probed !== 'object') return true
    return probed.memorySource !== 'phosmith'
}

/**
 * Session budget: preset + capability probe + URL overrides. On a memory-
 * locked device (unverified memory, or no probe yet) the profile is ALWAYS
 * `lite` by default — `cap.estimatedProfile` (capability.js) is a real-signal
 * guess surfaced for the UI to suggest, but it is deliberately never applied
 * on its own: the live test suite's A-phase guarantees (bounded export, one
 * resident embedding, no OPFS persistence, unsafe-flag lockout) assume every
 * plain browser gets the same bounded floor regardless of its actual core
 * count. Only the user's own profile-toggle choice — `override`, a persisted,
 * deliberate decision, not a URL param a page could set for itself — raises
 * it, and it may go beyond the estimate's own `standard` ceiling since the
 * user is vouching for their own device. URL parameters may only LOWER
 * limits on a locked budget — `?flagship=1`, `?proxy=max`, `?proxy=off`,
 * `?profile=ultra`, `?working=1` are all refused there.
 * `probed` is the boot capability object or a bare profile string (tests).
 */
export const resolveBudget = (search = typeof location !== 'undefined' ? location.search : '', probed = null, override = null) => {
    const params = new URLSearchParams(search)
    const cap = (probed && typeof probed === 'object') ? probed : null
    const locked = isMemoryLocked(probed)
    const requestedProfile = locked ? null : params.get('profile')
    const manualOverride = locked && override && PRESETS[override] ? override : null
    const name = locked
        ? (manualOverride || 'lite')
        : (requestedProfile || (cap ? cap.profile : probed))
    const budget = { ...(PRESETS[name] || PRESETS.lite) }
    // Adaptive proxy: the probe sizes it to the device. An explicit profile is
    // a developer override and gets that profile's normal proxy instead.
    if (cap && cap.proxyMax && !requestedProfile) {
        budget.proxyMax = locked ? Math.min(cap.proxyMax, budget.proxyMax) : cap.proxyMax
    }
    if (cap) {
        budget.memoryGB = cap.memoryGB || 0
        budget.memorySource = cap.memorySource || 'unknown'
        budget.vramGB = cap.vramGB || 0
        budget.gpuTier = cap.gpuTier || 'none'
        budget.resourceMode = cap.resourceMode || 'balanced'
        budget.hostManaged = !!cap.hostManaged
        budget.flagshipEligible = !!cap.flagshipEligible
        budget.textureLimit = cap.textureLimit || 0
        // GPU acceleration is independent of the memory tier (see capability.js):
        // any usable, non-fallback WebGPU adapter runs SlimSAM on the GPU, even
        // on the memory-locked lite baseline. SlimSAM is ~14 MB and the proxy is
        // bounded, so the upload burst is small; segment() still falls back to
        // WASM on any runtime failure, and ?force=wasm / memory pressure override.
        budget.samWebGPU = cap.gpuTier !== 'none'
    }
    budget.memoryLocked = locked
    // How `name` was picked — the profile toggle reads this to label its
    // "Auto" option and to know whether the user has already overridden it.
    budget.profileSource = locked
        ? (manualOverride ? 'manual' : 'default')
        : (cap ? 'trusted' : 'preset')
    if (cap?.memorySource === 'unknown') budget.memoryUncertain = true

    const adaptiveProxyMax = budget.proxyMax
    const pq = params.get('proxy')
    if (locked) {
        // Only a LOWER manual proxy is honored; off/max/large are unsafe here.
        if (pq && Number(pq) >= 256 && Number(pq) < adaptiveProxyMax) {
            budget.proxyMode = 'manual'
            budget.proxyMax = Math.round(Number(pq))
        }
    } else if (pq === 'off') {
        budget.proxyMode = 'disabled'
        budget.proxyMax = 0
        budget.safeProxyMax = adaptiveProxyMax
    } else if (pq === 'max') {
        budget.proxyMode = 'manual'
        budget.proxyMax = 4096
    } else if (pq && Number(pq) >= 256) {
        budget.proxyMode = 'manual'
        budget.proxyMax = Math.min(4096, Math.round(Number(pq)))
    }

    // SAM3/flagship is retired from the editor's interactive architecture.
    // Keep this explicit value for integrations and diagnostics, but never
    // accept a query parameter or host hint that would allocate a second
    // segmentation model alongside SlimSAM.
    budget.flagship = false

    if (params.get('force') === 'wasm') { budget.forceWasm = true; budget.flagship = false }
    if (params.get('escalate') === '0') budget.autoEscalate = false
    // Bounded "working" re-decode copy for hosts whose image decode is
    // unbounded (Safari). auto = feature-detect; ?working=1 forces it only on
    // trusted budgets (verify uses it); ?working=0 disables anywhere.
    if (params.get('working') === '0') budget.workingMode = 'off'
    else if (!locked && params.get('working') === '1') budget.workingMode = 'force'

    // On-screen preview quality (display only — never touches the segmentation
    // memory contract, so it is honored even on a locked budget). `native` opts
    // into a one-time full-res decode for a crisp preview on unbounded-decode
    // hosts (Safari); `off` pins the preview to the model proxy; a number caps
    // the display long edge lower.
    const dq = params.get('display')
    if (dq === 'native') budget.displayMode = 'native'
    else if (dq === 'off') budget.displayMode = 'off'
    else if (dq && Number(dq) >= 256) budget.displayMax = Math.min(4096, Math.round(Number(dq)))
    return budget
}

/**
 * Runtime safety valve — a one-way ratchet. It never re-enables a feature or
 * raises a cap; it reduces future allocation sizes and turns automation off.
 *   1: drop detector + prewarm, one embedding max
 *   2: +no wasm refine, no escalation/HD decode, crops ≤ 1280
 *   3: +proxy ≤ 768, exports capped (lite: 4 MP)
 */
export const applyMemoryPressure = (budget, level = 1) => {
    const nextLevel = Math.max(Number(budget?.pressureLevel) || 0, Math.min(3, Math.max(0, Math.floor(level))))
    if (!nextLevel) return budget
    const next = { ...budget, pressureLevel: nextLevel, autoEscalate: false }
    if (nextLevel >= 1) {
        next.maxResidentHeavy = 1
        next.draftCacheMax = Math.min(next.draftCacheMax || 1, 1)
        next.flagshipCacheMax = 0
        next.detectorWebGPU = false
        next.samWebGPU = false
        next.eagerEncode = false
    }
    if (nextLevel >= 2) {
        next.cropMaxSide = Math.min(next.cropMaxSide || 1280, 1280)
        next.escalateMaxMP = Math.min(next.escalateMaxMP || 8, 8)
        next.displayMax = Math.min(next.displayMax || 1600, 1600)
        next.hdExportDecode = false
        next.cvRefine = false
        next.rawDevelop = false
    }
    if (nextLevel >= 3) {
        const exportCap = next.profile === 'lite' ? 4 : (next.profile === 'standard' ? 16 : 24)
        next.exportMaxMP = Math.min(next.exportMaxMP || exportCap, exportCap)
        next.exportMaxSide = Math.min(next.exportMaxSide || 4096, next.profile === 'lite' ? 4096 : 6144)
        next.proxyMax = Math.min(next.proxyMax || 768, 768)
        next.displayMax = Math.min(next.displayMax || 1280, 1280)
        if (next.safeProxyMax) next.safeProxyMax = Math.min(next.safeProxyMax, 768)
    }
    return next
}

export const PROFILE_PRESETS = PRESETS

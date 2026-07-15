/**
 * policy — device profiles and runtime budgets (main thread).
 * One budget object gates proxy size, residency, cache caps, escalation,
 * HD export. Profiles scale HOW MUCH hardware a feature gets, never WHETHER.
 * M0: static presets + URL overrides; M4 fills these from a capability probe.
 *
 * Flagship (SAM3, ~300 MB co-resident) is opt-in. A Phosmith host can
 * explicitly authorize it after reporting a trusted large-memory WebGPU
 * budget; a standalone browser never assumes that `deviceMemory === 8` means
 * a 16/24/64 GB machine. The draft lane (SlimSAM) remains fully functional
 * everywhere, including a 4 GB browser or a WASM-only device.
 *
 * Residency is blob-only: a file upload is never held as full-res RGBA (asset-
 * store decodes straight to the proxy and re-decodes bounded regions on demand).
 * cropMaxSide/exportMaxSide cap the pixels any native op may materialize, so an
 * 8K photo stays interactive on 8 GB.
 */

const PRESETS = {
    lite: {
        profile: 'lite',
        proxyMax: 768,
        proxyMode: 'auto',
        directMaxMP: 2,         // only an explicitly disabled proxy may use this
        directMaxSide: 2048,
        cropMaxSide: 1280,       // caps native escalation/HD-export crop memory
        exportMaxSide: 4096,     // caps the exported cutout's long edge
        exportMaxMP: 8,          // 4 GB-safe: export can materialize several RGBA copies
        detectorWebGPU: false,   // fp16 on unified memory hangs the host — wasm/q8 only
        flagship: false,         // opt-in only (?flagship=1) — see header
        maxResidentHeavy: 1,     // flagship/detector swap, never co-resident
        draftCacheMax: 2,
        flagshipCacheMax: 0,
        autoEscalate: false,     // crop escalation manual-only
        escalateMaxMP: 12,       // skip interactive escalation above this MP (export still full-res)
        hdExportDecode: false,   // filter-only HD export (no crop re-decode)
        detectorDispose: 'now',  // free OWLv2 right after boxes
    },
    standard: {
        profile: 'standard',
        proxyMax: 1024,
        proxyMode: 'auto',
        directMaxMP: 4,
        directMaxSide: 4096,
        cropMaxSide: 2048,
        exportMaxSide: 8192,     // full 8K exports; caps only beyond
        // A 24 MP DSLR frame is ~96 MB RGBA. Blob-only input, bounded crops,
        // and pressure shedding make it practical on a confirmed 8 GB budget.
        exportMaxMP: 24,
        detectorWebGPU: false,   // 8 GB unified memory: see lite
        flagship: false,         // opt-in only (?flagship=1) — see header
        // Unknown GPU headroom is still common on 8 GB machines; never pin
        // the detector and optional flagship session at the same time.
        maxResidentHeavy: 1,
        draftCacheMax: 3,
        flagshipCacheMax: 2,
        autoEscalate: true,
        escalateMaxMP: 24,   // skip interactive native-crop escalation above this (export still full-res)
        hdExportDecode: true,
        detectorDispose: 'idle', // free OWLv2 ~10 s after boxes
    },
    pro: {
        profile: 'pro',
        proxyMax: 1280,
        proxyMode: 'auto',
        directMaxMP: 6,
        directMaxSide: 6144,
        cropMaxSide: 3072,
        exportMaxSide: 10000,
        exportMaxMP: 36,         // trusted 12–23 GB Phosmith budget
        detectorWebGPU: true,    // 12 GB+ discrete/large budget absorbs the fp16 session
        flagship: false,         // opt-in only (?flagship=1) — see header
        maxResidentHeavy: 2,
        draftCacheMax: 4,
        flagshipCacheMax: 3,
        autoEscalate: true,
        escalateMaxMP: 36,
        hdExportDecode: true,
        detectorDispose: 'idle',
    },
    ultra: {
        profile: 'ultra',
        proxyMax: 1536,
        proxyMode: 'auto',
        directMaxMP: 8,
        directMaxSide: 8192,
        cropMaxSide: 4096,
        exportMaxSide: 12000,
        // 24 GB+ host budget. 64 MP allows 45–61 MP camera photos to stay at
        // native resolution while still bounding browser canvas copies.
        exportMaxMP: 64,
        detectorWebGPU: true,
        flagship: false,
        maxResidentHeavy: 2,
        draftCacheMax: 6,
        flagshipCacheMax: 4,
        autoEscalate: true,
        escalateMaxMP: 64,
        hdExportDecode: true,
        detectorDispose: 'idle',
    },
}

/** Session budget: preset + capability probe + URL overrides. `probed` is the
 *  boot capability object ({ profile, proxyMax, memoryGB, … }) or a bare profile string.
 *  Overrides: ?profile=lite|standard|pro|ultra, ?flagship=0|1, ?force=wasm, ?escalate=0,
 *  ?proxy=<px>|off|auto. `off` is a request for a native interaction frame,
 *  not permission to allocate an unsafe DSLR-sized canvas: asset-store only
 *  honors it below `directMaxMP`/`directMaxSide`, otherwise it restores this
 *  device's safe adaptive cap. Flagship is off in every preset (opt-in). */
export const resolveBudget = (search = typeof location !== 'undefined' ? location.search : '', probed = null) => {
    const params = new URLSearchParams(search)
    const cap = (probed && typeof probed === 'object') ? probed : null
    const requestedProfile = params.get('profile')
    const name = requestedProfile || (cap ? cap.profile : probed)
    const budget = { ...(PRESETS[name] || PRESETS.standard) }
    // Adaptive proxy: the probe sizes it to the device. An explicit profile is
    // a developer/user override and gets that profile's normal proxy instead.
    if (cap && cap.proxyMax && !requestedProfile) budget.proxyMax = cap.proxyMax
    if (cap) {
        budget.memoryGB = cap.memoryGB || 0
        budget.memorySource = cap.memorySource || 'unknown'
        budget.vramGB = cap.vramGB || 0
        budget.gpuTier = cap.gpuTier || 'none'
        budget.resourceMode = cap.resourceMode || 'balanced'
        budget.hostManaged = !!cap.hostManaged
        budget.flagshipEligible = !!cap.flagshipEligible
    }
    // Safari/Firefox commonly expose no device-memory class. Keep the normal
    // feature set and WebGPU acceleration, but do not allocate as if the
    // machine were confirmed 8 GB. A Phosmith hint removes this uncertainty.
    if (cap?.memorySource === 'unknown' && !requestedProfile) {
        budget.memoryUncertain = true
        budget.cropMaxSide = Math.min(budget.cropMaxSide, 1536)
        budget.exportMaxSide = Math.min(budget.exportMaxSide, 6144)
        budget.exportMaxMP = Math.min(budget.exportMaxMP, 16)
        budget.detectorWebGPU = false // unproven memory: SAM keeps WebGPU, the detector doesn't
        budget.draftCacheMax = Math.min(budget.draftCacheMax, 2)
        budget.flagshipCacheMax = 0
        budget.maxResidentHeavy = 1
        budget.escalateMaxMP = Math.min(budget.escalateMaxMP, 16)
    }
    const adaptiveProxyMax = budget.proxyMax
    const pq = params.get('proxy')
    if (pq === 'off') {
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
    if (params.get('flagship') === '1') budget.flagship = true    // explicit user opt-in to SAM3
    else if (params.get('flagship') === '0') budget.flagship = false
    // The native Phosmith host may authorize a flagship preload only after it
    // has supplied a trusted, accelerated pro/ultra resource budget.
    else if (cap?.allowFlagship && cap.flagshipEligible) budget.flagship = true
    if (params.get('force') === 'wasm') { budget.forceWasm = true; budget.flagship = false }
    if (params.get('escalate') === '0') budget.autoEscalate = false
    // Bounded "working" re-decode copy for hosts whose image decode is
    // unbounded (Safari: no ImageDecoder, no scaled createImageBitmap).
    // auto = feature-detect; 1 forces it (the verify gate), 0 disables.
    if (params.get('working') === '1') budget.workingMode = 'force'
    else if (params.get('working') === '0') budget.workingMode = 'off'
    return budget
}

/** Runtime safety valve used by the heap watchdog and a Phosmith memory
 * notification. It never removes a feature; it reduces future allocation
 * sizes, turns off expensive automatic escalation, and asks the engine to
 * dispose reloadable residents separately. */
export const applyMemoryPressure = (budget, level = 1) => {
    const nextLevel = Math.max(Number(budget?.pressureLevel) || 0, Math.min(3, Math.max(0, Math.floor(level))))
    if (!nextLevel) return budget
    const next = { ...budget, pressureLevel: nextLevel, autoEscalate: false }
    if (nextLevel >= 1) {
        next.maxResidentHeavy = 1
        next.draftCacheMax = Math.min(next.draftCacheMax || 1, 1)
        next.flagshipCacheMax = 0
        next.detectorWebGPU = false // level 1 frees the detector; don't rebuild it heavy
    }
    if (nextLevel >= 2) {
        next.cropMaxSide = Math.min(next.cropMaxSide || 1536, 1536)
        next.escalateMaxMP = Math.min(next.escalateMaxMP || 12, 12)
        next.hdExportDecode = false
    }
    if (nextLevel >= 3) {
        const exportCap = next.profile === 'lite' ? 8 : (next.profile === 'standard' ? 16 : 24)
        next.exportMaxMP = Math.min(next.exportMaxMP || exportCap, exportCap)
        next.exportMaxSide = Math.min(next.exportMaxSide || 6144, next.profile === 'lite' ? 4096 : 6144)
        next.proxyMax = Math.min(next.proxyMax || 768, 768)
        if (next.safeProxyMax) next.safeProxyMax = Math.min(next.safeProxyMax, 768)
    }
    return next
}

export const PROFILE_PRESETS = PRESETS

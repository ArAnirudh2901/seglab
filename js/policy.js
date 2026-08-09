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

// ONE configuration (DESIGN-MASK-LANE §11). Five presets plus a capability
// ladder plus a manual override plus four pressure levels were untestable in
// combination, and a preset guaranteed nothing anyway: it picked numbers, and
// nothing enforced them at allocation time. Memory is bounded instead by the
// single shared SAM 2.1 instance (js/sam21-host.js) plus the governor.
// The numbers are standard8's — built for "the worst device that reaches it",
// which is now simply the device contract.
const CONFIG = {
        profile: 'standard8',
        samIdleMs: 300_000,
        memBudgetMB: 1900,
        proxyMax: 1024,          // floor on the LONG edge; see proxyShortMax
        // The encoder consumes a 1024x1024 SQUARE (sam21-lane drawImage), so the
        // detail it can use is capped PER AXIS. Sizing only by the long edge
        // starves the short one — a 3:2 frame at 1024x683 hands the encoder 683
        // real rows stretched to 1024, wasting a third of its vertical capacity.
        // Measured on the canonical NEF: reaching 1024 on the SHORT edge is
        // worth +8.2 pt boundary IoU, ~10x what the whole refinement stage buys.
        proxyShortMax: 1024,
        proxyLongMax: 2048,      // hard stop for panoramas
        proxyPixelMax: 2_100_000, // ~2:1 fully saturated; bounds proxy RGBA at 8.4 MB
        proxyMode: 'auto',
        displayMax: 2560,        // crisper preview (decoupled from the model proxy)
        displayMode: 'auto',
        directMaxMP: 3,
        directMaxSide: 2560,
        cropMaxSide: 1536,
        exportMaxSide: 5120,
        exportMaxMP: 12,         // the visible win: 8 → 12 MP cutouts (bounded peak)
        exportFullRes: false,    // deliberately bounded (memory-close to lite)
        escalateMaxMP: 12,
        escalateMinIoU: 0.5,     // crop re-decode must agree with the proxy mask, or it is a different object
        draftCacheMax: 1,        // still exactly one resident embedding
        flagshipCacheMax: 0,
        maxResidentHeavy: 1,
        flagship: false,
        detectorWebGPU: true,
        detectorEvictOnEncode: true,
        detectorIdleMs: 120_000,
        samWebGPU: true,
        autoEscalate: false,     // interaction-time native re-decode → manual tiers only
        hdExportDecode: true,    // sharp native-region export (bounded, export-time only)
        detectorDispose: 'idle',
        eagerEncode: true,
        cvRefine: true,
        rawDevelop: true,
        rawDevelopMaxMP: 50,
        embedPersist: true,      // SAM 2.1 embeddings are 8 MB; OPFS turns a
                                 //   revisit into a decode-only interaction
        // The working copy is the re-decode SOURCE for escalation and export.
        // 2560 capped it below its own consumers (exportMaxSide 5120,
        // escalateMaxMP 12), so a bounded host got a worse export than an
        // unbounded one for no reason. 4096² RGBA is ~67 MB — comfortably
        // inside the measured ~1.8 GB total.
        workingMaxSide: 4096,
        pressureLevel: 0,
}
const PRESETS = { standard8: CONFIG }

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
    // One config — no tier to pick, no override to honour, no ?profile= to read.
    const budget = { ...CONFIG }
    // The capability probe may still LOWER the proxy for a weak GPU; it can
    // never raise it, because 1024 is the model's native encode edge anyway.
    if (cap && cap.proxyMax) budget.proxyMax = Math.min(cap.proxyMax, budget.proxyMax)
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
        // any usable, non-fallback WebGPU adapter runs the mask lane on the GPU,
        // even on the memory-locked lite baseline. The proxy is
        // bounded, so the upload burst is small; segment() still falls back to
        // WASM on any runtime failure, and ?force=wasm / memory pressure override.
        budget.samWebGPU = cap.gpuTier !== 'none'
    }
    budget.memoryLocked = locked
    budget.profileSource = 'single' // kept for telemetry; there is nothing to pick
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
    // second segmentation model alongside the mask lane.
    budget.flagship = false

    if (params.get('force') === 'wasm') { budget.forceWasm = true; budget.flagship = false }
    // ?escalate=0 is a user opt-OUT, not merely "don't do it automatically".
    // It must also refuse the explicit action, or the flag silently means
    // nothing now that automatic escalation is off by default anyway.
    if (params.get('escalate') === '0') { budget.autoEscalate = false; budget.escalateDisabled = true }
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

    // Export resolution: full opts into a native-res tight cutout (an
    // export-time transient, not a resident tier), bounded forces the tier
    // cropMaxSide cap. Raising respects a locked trusted-host contract; lowering
    // is always allowed.
    const xq = params.get('export')
    if (xq === 'bounded') budget.exportFullRes = false
    else if (!locked && xq === 'full') budget.exportFullRes = true

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
        // NB: pressure does NOT move the mask lane off the GPU, and can't —
        // there is no wasm EP left to move it to. The rule outlived the code
        // that could break it: measured, the old WASM lane pinned ~3 GB against
        // the GPU's ~0.5 GB, so demoting under memory pressure made swap WORSE.
        // The lane is now WebGPU or nothing (sam21-lane checkDevice).
        next.eagerEncode = false
    }
    if (nextLevel >= 2) {
        next.cropMaxSide = Math.min(next.cropMaxSide || 1280, 1280)
        next.escalateMaxMP = Math.min(next.escalateMaxMP || 8, 8)
        next.displayMax = Math.min(next.displayMax || 1600, 1600)
        next.hdExportDecode = false
        next.exportFullRes = false // export a bounded (cropMaxSide) cutout under real pressure
        next.cvRefine = false
        next.rawDevelop = false
    }
    if (nextLevel >= 3) {
        // Flat, not profile-keyed (§11). The old ladder branched on the tier
        // name, so collapsing to standard8 silently landed in the permissive
        // branch and LOOSENED the export cap to 24 MP under maximum pressure —
        // the opposite of what this rung is for. Pressure only ever tightens.
        next.exportMaxMP = Math.min(next.exportMaxMP || 4, 4)
        next.exportMaxSide = Math.min(next.exportMaxSide || 4096, 4096)
        next.proxyMax = Math.min(next.proxyMax || 768, 768)
        next.proxyShortMax = 0 // give up the per-axis boost before anything visible
        next.displayMax = Math.min(next.displayMax || 1280, 1280)
        if (next.safeProxyMax) next.safeProxyMax = Math.min(next.safeProxyMax, 768)
    }
    return next
}

export const PROFILE_PRESETS = PRESETS

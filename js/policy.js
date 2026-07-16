/**
 * policy — device-safety profiles and effective limits
 * ------------------------------------------------------
 * One frozen policy object rules every memory-relevant limit in the app:
 * proxy size, export caps, model residency, feature gates. The safety
 * precedence is strict and cannot be re-ordered by callers:
 *
 *   hard device safety limit  >  memory-pressure reduction
 *     >  explicit URL parameter  >  normal feature request
 *
 * On a device that reports ≤ 8 GB — or reports nothing at all — the `lite`
 * profile is FORCED and URL flags cannot raise any limit (browsers cap
 * navigator.deviceMemory at 8 and Safari omits it entirely, so in practice
 * lite is universal; that is the intended conservative outcome).
 * Downgrade-only flags (?flagship=0, ?wasm=1, ?debug=1) are always honored:
 * lowering a limit is never unsafe.
 *
 * SAM3 note: `flagship:false` means the heavyweight lane NEVER loads
 * automatically. `allowFlagshipOptIn` leaves one door open — an explicit,
 * confirmed user gesture in the UI (never a URL flag, never auto).
 */

export const PROFILES = {
    lite: Object.freeze({
        profile: 'lite',

        // Interaction
        proxyMax: 768,
        proxyMode: 'auto',
        directMaxMP: 2,
        directMaxSide: 2048,

        // Original/crop/export operations
        cropMaxSide: 1280,
        exportMaxSide: 4096,
        exportMaxMP: 8,
        escalateMaxMP: 8,

        // Model residency
        draftCacheMax: 1,
        flagshipCacheMax: 0,
        maxResidentHeavy: 1,

        // Heavy optional features
        flagship: false,          // no automatic SAM3, ever
        allowFlagshipOptIn: true, // explicit confirmed user gesture only
        detectorWebGPU: false,
        autoEscalate: false,
        hdExportDecode: false,
        detectorDispose: 'now',
        speculative: false,       // no prewarm/pre-encode jobs
        wasmRefine: true,         // CV wasm allowed until pressure ≥ 2

        // Runtime limits
        pressureLevel: 0,
    }),

    // Data-only tiers for hypothetical >8 GB detection. Unreachable today:
    // Chrome clamps deviceMemory reporting to 8 and Safari has no signal,
    // so resolvePolicy can never select these from a real browser. They
    // exist so the precedence logic has something to clamp DOWN from.
    standard: Object.freeze({
        profile: 'standard',
        proxyMax: 1024,
        proxyMode: 'auto',
        directMaxMP: 4,
        directMaxSide: 2560,
        cropMaxSide: 2048,
        exportMaxSide: 6144,
        exportMaxMP: 16,
        escalateMaxMP: 16,
        draftCacheMax: 2,
        flagshipCacheMax: 1,
        maxResidentHeavy: 1,
        flagship: false,
        allowFlagshipOptIn: true,
        detectorWebGPU: false,
        autoEscalate: false,
        hdExportDecode: true,
        detectorDispose: 'now',
        speculative: false,
        wasmRefine: true,
        pressureLevel: 0,
    }),
}

/** Fields a URL parameter may NEVER raise past the device profile. */
const CEILING_FIELDS = [
    'proxyMax', 'directMaxMP', 'directMaxSide', 'cropMaxSide',
    'exportMaxSide', 'exportMaxMP', 'escalateMaxMP',
    'draftCacheMax', 'flagshipCacheMax', 'maxResidentHeavy',
]

/**
 * Resolve the boot policy from capability + URL search string.
 * `capability.deviceMemoryGB` of null/undefined (unknown) forces lite.
 */
export const resolvePolicy = ({ capability = {}, search = '' } = {}) => {
    const memGB = Number.isFinite(capability.deviceMemoryGB) ? capability.deviceMemoryGB : null
    const liteForced = memGB === null || memGB <= 8

    let params
    try {
        params = new URLSearchParams(search)
    } catch {
        params = new URLSearchParams('')
    }

    // Profile request honored only above the hard device limit.
    const requested = params.get('profile')
    const base = (!liteForced && requested && PROFILES[requested]) ? PROFILES[requested] : PROFILES.lite
    const policy = { ...base }

    if (!liteForced) {
        // Size upgrades allowed above the hard limit. `?flagship=1` is inert
        // on every tier: the only path to SAM3 is the confirmed UI opt-in.
        if (params.get('proxy') === 'max') policy.proxyMax = Math.max(policy.proxyMax, 1024)
    }

    // Downgrade-only flags: always safe, always honored.
    if (params.get('flagship') === '0') { policy.flagship = false; policy.allowFlagshipOptIn = false }
    const proxyParam = params.get('proxy')
    if (proxyParam && /^\d+$/.test(proxyParam)) {
        policy.proxyMax = Math.min(policy.proxyMax, Math.max(64, Number(proxyParam)))
    }
    policy.forceWasm = params.get('wasm') === '1'
    policy.debug = params.get('debug') === '1'

    // HARD DEVICE SAFETY LIMIT — nothing above may survive this clamp.
    if (liteForced) {
        const lite = PROFILES.lite
        for (const field of CEILING_FIELDS) {
            policy[field] = Math.min(policy[field], lite[field])
        }
        policy.profile = 'lite'
        policy.flagship = false
        policy.detectorWebGPU = false
        policy.autoEscalate = false
        policy.hdExportDecode = false
        policy.speculative = false
        policy.detectorDispose = 'now'
    }

    policy.liteForced = liteForced
    policy.deviceMemoryGB = memGB
    return Object.freeze(policy)
}

/**
 * Apply memory-pressure reductions on top of the boot policy.
 * Pure — callers re-derive whenever the pressure level changes.
 *   L1: stop speculative work, drop any non-current embedding.
 *   L2: + no wasm refinement, crop ops capped at 768, no HD re-decode.
 *   L3: + clear embedding after each run, exports ≤ 4 MP, safe-mode banner.
 */
export const effectivePolicy = (policy, pressureLevel = 0) => {
    const level = Math.max(policy.pressureLevel || 0, pressureLevel | 0)
    if (level <= 0) return policy
    const eff = { ...policy, pressureLevel: level }
    if (level >= 1) {
        eff.speculative = false
        eff.clearUnusedEmbeddings = true
        eff.detectorWebGPU = false
        eff.allowFlagshipOptIn = false
    }
    if (level >= 2) {
        eff.wasmRefine = false
        eff.cropMaxSide = Math.min(eff.cropMaxSide, 768)
        eff.hdExportDecode = false
        eff.autoEscalate = false
    }
    if (level >= 3) {
        eff.clearEmbeddingAfterRun = true
        eff.exportMaxMP = Math.min(eff.exportMaxMP, 4)
        eff.proxyMax = Math.min(eff.proxyMax, 768)
        eff.safeModeBanner = 'Memory pressure detected — running in safe mode.'
    }
    return Object.freeze(eff)
}

/**
 * policy — device profiles and runtime budgets (main thread).
 * One budget object gates proxy size, residency, cache caps, escalation,
 * HD export. Profiles scale HOW MUCH hardware a feature gets, never WHETHER.
 * M0: static presets + URL overrides; M4 fills these from a capability probe.
 */

const PRESETS = {
    lite: {
        profile: 'lite',
        proxyMax: 768,
        detectorCanvas: 640,     // OWLv2 input is fixed 960²; this only caps canvas memory
        flagship: false,
        maxResidentHeavy: 1,     // flagship/detector swap, never co-resident
        draftCacheMax: 3,
        flagshipCacheMax: 0,
        bitmapMaxMP: 0,          // original as Blob, re-decoded on demand
        autoEscalate: false,     // crop escalation manual-only
        hdExportDecode: false,   // filter-only HD export (no crop re-decode)
        detectorDispose: 'now',  // free OWLv2 right after boxes
    },
    standard: {
        profile: 'standard',
        proxyMax: 1024,
        detectorCanvas: 960,
        flagship: true,
        maxResidentHeavy: 2,
        draftCacheMax: 4,
        flagshipCacheMax: 2,
        bitmapMaxMP: 24,         // resident bitmap up to 24 MP, else Blob
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'idle', // free OWLv2 ~10 s after boxes
    },
    pro: {
        profile: 'pro',
        proxyMax: 1024,
        detectorCanvas: 960,
        flagship: true,
        maxResidentHeavy: 2,
        draftCacheMax: 4,
        flagshipCacheMax: 2,
        bitmapMaxMP: 24,
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'resident',
    },
}

/** Session budget: preset (Standard until M4) + URL overrides —
 *  ?profile=lite|standard|pro, ?flagship=0, ?force=wasm, ?escalate=0. */
export const resolveBudget = (search = typeof location !== 'undefined' ? location.search : '') => {
    const params = new URLSearchParams(search)
    const name = params.get('profile')
    const budget = { ...(PRESETS[name] || PRESETS.standard) }
    if (params.get('flagship') === '0') budget.flagship = false
    if (params.get('force') === 'wasm') { budget.forceWasm = true; budget.flagship = false }
    if (params.get('escalate') === '0') budget.autoEscalate = false
    return budget
}

export const PROFILE_PRESETS = PRESETS

/**
 * policy — device profiles and runtime budgets (main thread)
 * ------------------------------------------------------------
 * One budget object drives everything resource-shaped in the engine: proxy
 * sizes, model residency, embedding cache caps, escalation and HD-export
 * behaviour. Profiles change HOW MUCH hardware a feature gets — never
 * WHETHER a device gets the feature (the Offline-Pro promise: equal
 * features everywhere, seconds scale with silicon).
 *
 * M0 ships static presets + URL overrides; the M4 capability probe
 * (WebGPU adapter, deviceMemory, warm-up encode timing) will pick the
 * preset automatically and persist it. Computed on the main thread
 * (workers have no localStorage) and passed into warm({ budget }).
 */

const PRESETS = {
    lite: {
        profile: 'lite',
        proxyMax: 768,           // interaction proxy longest side
        detectorCanvas: 640,     // detector input canvas (OWLv2 model input is FIXED 960²
        //                          — this knob saves canvas memory, not FLOPs)
        flagship: false,         // SAM3 lane off — SlimSAM + post pipeline carries quality
        maxResidentHeavy: 1,     // one heavyweight model at a time (flagship/detector swap)
        draftCacheMax: 3,        // embedding LRU caps per lane
        flagshipCacheMax: 0,
        bitmapMaxMP: 0,          // original kept as Blob, re-decoded on demand
        autoEscalate: false,     // crop escalation is manual-only on Lite
        hdExportDecode: false,   // HD export refines with the guided filter only (no re-decode)
        detectorDispose: 'now',  // release OWLv2 right after boxes
    },
    standard: {
        profile: 'standard',
        proxyMax: 1024,
        detectorCanvas: 960,
        flagship: true,
        maxResidentHeavy: 2,
        draftCacheMax: 4,
        flagshipCacheMax: 2,
        bitmapMaxMP: 24,         // original resident as ImageBitmap up to 24 MP, else Blob
        autoEscalate: true,
        hdExportDecode: true,
        detectorDispose: 'idle', // dispose OWLv2 ~10 s after boxes (re-phrase stays warm)
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

/**
 * Resolve the session budget: preset (static Standard until the M4 probe
 * lands) + URL overrides. Overrides are for tests, data-saving, and forcing
 * weak-device behaviour on strong hardware:
 *   ?profile=lite|standard|pro   pick a preset explicitly
 *   ?flagship=0                  keep the session on the draft lane
 *   ?force=wasm                  pretend WebGPU does not exist
 *   ?escalate=0                  disable auto crop escalation (M3 control runs)
 */
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

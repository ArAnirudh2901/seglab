/**
 * memory-governor — keep the whole app under a hard RAM ceiling
 * ----------------------------------------------------------------
 * SEGLAB now has three heavy things that all want to be resident at once:
 * the SlimSAM draft encoder, the SAM3 flagship encoder (~302 MB of weights
 * plus 33 MB of embeddings per image), and the OWLv2 text encoder (~155 MB
 * quantized, and an ONNX Runtime arena several times that at 960²). Add a
 * decoded 24–45 MP photo — 96 MB for a 6000×4000 RGBA buffer alone — and a
 * naive "load everything" policy blows past 2.2 GB on the normal laptops
 * this is built for.
 *
 * The policy here is deliberately blunt because blunt is what holds:
 *   ONE heavy encoder is resident at a time. Loading another disposes the
 *   idle one's ONNX session first.
 * Everything else (cache sizes, whether the flagship lane is allowed to load
 * at all) is derived from a live budget rather than hardcoded, so a 4 GB
 * Chromebook and a 64 GB workstation each get a policy that fits.
 *
 * This module owns policy only — it never imports transformers.js. Callers
 * register their own dispose hooks, which keeps it unit-testable headless.
 */

const MB = 1024 * 1024

/** Declared cost of each heavy resident, in MB. Weights + typical arena. */
export const RESIDENT_COST = {
    draft: 80,        // SlimSAM 14 MB + activations
    flagship: 900,    // SAM3-tracker q4f16 302 MB + arena
    text: 550,        // OWLv2 quantized 155 MB + arena at 960²
}

const state = {
    ceilingMB: 2200,
    reservedMB: 300,        // browser/tab baseline we never get to spend
    // ONNX Runtime's arena fragments and overshoots its nominal footprint,
    // and a decoded photo spikes during draw/readback. Filling the ceiling
    // exactly is how you get an OOM tab, so a load must leave this much
    // slack or it does not count as affordable.
    marginMB: 250,
    imageMB: 0,             // decoded source currently held
    residents: new Map(),   // key → { costMB, dispose }
}

/**
 * @param {{ ceilingMB?: number, reservedMB?: number, marginMB?: number }} opts
 */
export const configure = ({ ceilingMB, reservedMB, marginMB } = {}) => {
    if (Number.isFinite(ceilingMB)) state.ceilingMB = ceilingMB
    if (Number.isFinite(reservedMB)) state.reservedMB = reservedMB
    if (Number.isFinite(marginMB)) state.marginMB = marginMB
    return budget()
}

/**
 * Device-derived ceiling. `navigator.deviceMemory` is coarse (2/4/8) and
 * absent on Firefox and Safari, so it is treated as a hint: when it says the
 * machine is small we shrink the ceiling, and when it says nothing we keep
 * the configured default rather than guessing upward.
 */
export const adoptDeviceCeiling = (nav = typeof navigator !== 'undefined' ? navigator : null) => {
    const gb = nav?.deviceMemory
    if (!Number.isFinite(gb)) return budget()
    // Never claim more than ~35% of system RAM for one tab.
    const cap = Math.round(gb * 1024 * 0.35)
    state.ceilingMB = Math.min(state.ceilingMB, Math.max(600, cap))
    return budget()
}

/** Record the decoded source image's cost (w × h × 4 bytes, plus a copy). */
export const setImageFootprint = (w, h) => {
    state.imageMB = Math.round(((w * h * 4) * 2) / MB)
    return budget()
}

const residentMB = () => {
    let sum = 0
    for (const r of state.residents.values()) sum += r.costMB
    return sum
}

/** Current budget picture, in MB. */
export const budget = () => {
    const used = state.reservedMB + state.imageMB + residentMB()
    return {
        ceilingMB: state.ceilingMB,
        reservedMB: state.reservedMB,
        marginMB: state.marginMB,
        imageMB: state.imageMB,
        residentMB: residentMB(),
        usedMB: used,
        freeMB: state.ceilingMB - used,
        // What is actually spendable once the safety margin is held back.
        spendableMB: state.ceilingMB - used - state.marginMB,
        residents: [...state.residents.keys()],
    }
}

/** Would loading `key` fit right now — with the safety margin intact? */
export const canAfford = (key, costMB = RESIDENT_COST[key] || 0) =>
    budget().spendableMB >= costMB

/**
 * Register a loaded heavy resident and its dispose hook. Call this AFTER the
 * model is live so the accounting matches reality.
 */
export const register = (key, dispose, costMB = RESIDENT_COST[key] || 0) => {
    state.residents.set(key, { costMB, dispose })
    return budget()
}

export const isResident = (key) => state.residents.has(key)

/** Drop one resident, running its dispose hook. Safe to call when absent. */
export const evict = async (key) => {
    const r = state.residents.get(key)
    if (!r) return budget()
    state.residents.delete(key)
    try { await r.dispose?.() } catch { /* already gone */ }
    return budget()
}

/**
 * Make room for `key`, evicting other heavy encoders (least important first)
 * until it fits. Returns whether it can now be afforded — a false here means
 * the caller must not load, even after evicting everything it may evict.
 *
 * `protect` names residents the caller still needs (typically the decoder
 * that will turn this lane's boxes into masks).
 */
export const makeRoomFor = async (key, { protect = [], costMB = RESIDENT_COST[key] || 0, order = ['flagship', 'text', 'draft'] } = {}) => {
    if (isResident(key)) return true
    for (const victim of order) {
        if (canAfford(key, costMB)) break
        if (victim === key || protect.includes(victim)) continue
        if (state.residents.has(victim)) await evict(victim)
    }
    return canAfford(key, costMB)
}

/**
 * How many image embeddings a lane may cache, given what is left over.
 * Replaces the hardcoded per-lane `cacheMax` when the budget is tight.
 */
export const embeddingCacheMax = (perImageMB, hardMax) => {
    const spare = Math.max(0, budget().spendableMB)
    return Math.max(1, Math.min(hardMax, Math.floor(spare / Math.max(1, perImageMB))))
}

/** Test/reset hook. */
export const reset = () => {
    state.residents.clear()
    state.imageMB = 0
    state.ceilingMB = 2200
    state.reservedMB = 300
    state.marginMB = 250
}

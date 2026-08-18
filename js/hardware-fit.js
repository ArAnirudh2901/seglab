/**
 * hardware-fit — judge what this machine can post-process, and size the work to it.
 *
 * Technique adapted from llmfit (github.com/AlexsJones/llmfit): estimate from a
 * class table, express fit as a utilisation band instead of a hard limit, walk
 * quality rungs downward until one fits, and let a real benchmark replace the
 * estimate. Two deliberate divergences, both because a browser is not a shell:
 *
 *  - llmfit predicts speed from GPU bandwidth. A page cannot read bandwidth,
 *    VRAM, or even the backend name (`adapter.info.backend` is empty without
 *    Chrome's developer flag — measured). So the table is keyed on the coarse
 *    class signals a page DOES get, and it is worth less than llmfit's.
 *  - llmfit's bench is a command the user runs. Here every click already
 *    reports `postMs`, so the estimate governs the first click and nothing else.
 *
 * What is being judged is POST-PROCESSING, not inference: the encoder squashes
 * every frame to 1024² and the decoder emits 256² logits, so encode and decode
 * are flat in proxy size. Only the CPU post pipeline scales with it — and it is
 * 60-70% of a warm click (measured 2026-08-14).
 *
 * Same judgement, second consumer: the text lane's tile grid (§ detector below).
 */

/**
 * The cost model, measured (2026-08-14, Apple metal-3 / headless Chromium, a
 * 1536x896 proxy, 12 click points spanning a 140x range of refined area):
 *
 *     postMs  =  proxyMP × rate × (FIXED_SHARE + (1 − FIXED_SHARE) × fraction)
 *
 * where `fraction` is the refined band's rect over the proxy's own area.
 *
 * TWO terms because the measurement has two: the bicubic upsample, the
 * full-frame matte and the raw-mask copy are paid per PROXY pixel whatever is
 * selected (11.9 ms at 1.376 MP ⇒ 8.6 ms/MP), while the guided filter and the
 * band re-matte are paid per REFINED pixel (~30 ms per rect-MP). A one-term
 * ms/MP figure therefore measures the scene, not the machine: on this device
 * the same proxy costs 12.2 ms for a 9.7 kpx band and 52.6 ms for a
 * frame-spanning one. Normalised through the shape below, the SAME twelve
 * clicks return 37-39 ms/MP — that is the device constant, and it is what gets
 * persisted.
 *
 * `REFERENCE_MS_PER_MP` is deliberately NOT that 38: it stays at the earlier
 * headed-Chrome figure, which is the slower of the two conditions measured on
 * this machine. A seed that is too fast hands a first click to a device that
 * cannot pay for it; a seed that is too slow costs one click's resolution and
 * is then replaced. Only the first click of a device's life rides it.
 */
export const REFERENCE_MS_PER_MP = 105
export const FIXED_SHARE = 0.22

/** Cost multiplier for a band covering `fraction` of the proxy. */
export const bandShape = (fraction) => {
    const f = fraction > 1 ? 1 : (fraction > 0 ? fraction : 0)
    return FIXED_SHARE + (1 - FIXED_SHARE) * f
}

/**
 * Estimates are pessimistic on purpose — llmfit's efficiency factor with the
 * sign that matters here. Overshooting costs latency on a machine that cannot
 * afford it; undershooting costs resolution that the next click corrects.
 * Applied to estimates only, never to a measurement.
 */
export const ESTIMATE_HEADROOM = 1.3

/**
 * Class penalties, worst applicable wins. NOT multiplied together: these are
 * overlapping views of the same device class, not independent factors, and
 * compounding them turned a 4-core laptop iGPU into a 3.2x penalty it does not
 * deserve.
 */
const CLASS_PENALTY = [
    ['mobile', 3.0, (c) => c.mobile],
    ['weak-gpu-tier', 2.2, (c) => c.gpuTier === 'none' || c.gpuTier === 'basic'],
    ['legacy-backend', 2.0, (c) => c.legacyBackend],
    ['few-cores', 2.0, (c) => c.logicalProcessors > 0 && c.logicalProcessors <= 4],
    ['integrated-gpu', 1.6, (c) => c.integratedGPU],
    ['modest-cores', 1.25, (c) => c.logicalProcessors > 0 && c.logicalProcessors <= 8],
]

/** Worst applicable class penalty, with the inputs that produced it — llmfit's
 *  "every estimate ships its inputs". */
const classPenalty = (cap = {}) => {
    let penalty = 1
    const reasons = []
    for (const [name, factor, test] of CLASS_PENALTY) {
        if (!test(cap)) continue
        reasons.push(name)
        if (factor > penalty) penalty = factor
    }
    return {
        penalty,
        reasons,
        inputs: {
            mobile: !!cap.mobile,
            gpuTier: cap.gpuTier || '',
            gpuBackend: cap.gpuBackend || '',
            integratedGPU: !!cap.integratedGPU,
            legacyBackend: !!cap.legacyBackend,
            logicalProcessors: cap.logicalProcessors || 0,
        },
    }
}

/** Estimate full-band ms per megapixel of post-processing. */
export const estimatePostMsPerMP = (cap = {}) => {
    const { penalty, reasons, inputs } = classPenalty(cap)
    return {
        msPerMP: Math.round(REFERENCE_MS_PER_MP * penalty * ESTIMATE_HEADROOM),
        source: 'estimated',
        penalty,
        reasons,
        inputs,
    }
}

/**
 * Fit bands, llmfit's Perfect/Good/Marginal/Too Tight. Its sweet spot is
 * two-sided (50-80% of memory) because under-using a machine wastes it. Ours is
 * one-sided: quality saturates at the encoder's 1024 edge, so there is nothing
 * to spend a surplus on and only the upper bound means anything.
 */
export const fitLevel = (utilisation) => {
    if (!(utilisation > 0)) return 'perfect'
    if (utilisation <= 0.80) return 'perfect'
    if (utilisation <= 1.00) return 'good'
    if (utilisation <= 1.25) return 'marginal'   // over budget by latency only
    return 'tight'
}

/** Short-edge rungs, best first — llmfit's quantisation walk. 1024 is the
 *  encoder's own edge, so the top rung is the best that exists, not a policy. */
export const SHORT_EDGE_LADDER = [1024, 896, 768, 640, 512]

/**
 * Planning rate: the device constant folded through the band fraction this
 * device is actually expected to refine. No fraction on record means the worst
 * case (1 = the band spans the frame), so a device is never handed a proxy on
 * the strength of clicks it has not made yet.
 */
export const planMsPerMP = (budget = {}) => {
    const rate = Number(budget.postMsPerMP) || 0
    if (rate <= 0) return 0
    const f = Number(budget.postBandFraction)
    return rate * bandShape(Number.isFinite(f) && f > 0 ? f : 1)
}

/**
 * Highest short edge whose proxy still post-processes inside the click budget.
 *
 * No judgement (no measurement, no seed) returns the ceiling untouched: the
 * ladder only ever tightens, and only for a stated reason.
 */
export const affordableShortEdge = (budget = {}, aspect = 1) => {
    const ceiling = Number(budget.proxyShortMax) || 0
    const msPerMP = planMsPerMP(budget)
    const budgetMs = Number(budget.postBudgetMs) || 0
    if (!ceiling || msPerMP <= 0 || budgetMs <= 0) return ceiling
    const affordableMP = budgetMs / msPerMP
    const a = Math.max(1, aspect)
    let last = 0
    for (const rung of SHORT_EDGE_LADDER) {
        if (rung > ceiling) continue
        last = rung
        if (fitLevel((rung * rung * a) / 1e6 / affordableMP) !== 'tight') return rung
    }
    // Nothing fits: take the smallest rung rather than shrinking without bound.
    // A panorama lands here and is then bounded again by proxyLongMax/PixelMax.
    return last || ceiling
}

/** Report the whole judgement for one source — the `llmfit info` equivalent. */
export const explainFit = (w, h, budget = {}) => {
    const aspect = Math.max(w, h) / Math.max(1, Math.min(w, h))
    const shortEdge = affordableShortEdge(budget, aspect)
    const msPerMP = planMsPerMP(budget)
    const budgetMs = Number(budget.postBudgetMs) || 0
    const mp = (shortEdge * shortEdge * aspect) / 1e6
    const affordableMP = msPerMP > 0 && budgetMs > 0 ? budgetMs / msPerMP : 0
    return {
        shortEdge,
        aspect: +aspect.toFixed(3),
        proxyMP: +mp.toFixed(3),
        affordableMP: +affordableMP.toFixed(3),
        utilisation: affordableMP ? +(mp / affordableMP).toFixed(3) : 0,
        fit: affordableMP ? fitLevel(mp / affordableMP) : 'unjudged',
        msPerMP: Math.round(msPerMP),
        fullBandMsPerMP: Math.round(Number(budget.postMsPerMP) || 0),
        bandFraction: +(Number(budget.postBandFraction) || 1).toFixed(3),
        postBudgetMs: budgetMs,
        source: budget.postMsPerMPSource || 'none',
        estimatedPostMs: msPerMP ? Math.round(mp * msPerMP) : 0,
    }
}

/**
 * Fold one real click into the device constant — llmfit's bench, run for free.
 *
 * `bandFraction` is what makes this a device measurement instead of a scene
 * measurement, so a click that did not report one is DROPPED rather than
 * normalised by a guess: crediting a cheap click with a full band would report
 * this machine as ~4.5x slower than it is and shrink every later proxy.
 *
 * Symmetric, unlike the figure this replaces. The old asymmetry (expensive
 * evidence at once, cheap evidence only with repetition) existed to stop
 * small-object clicks re-authorising a proxy the frame-spanning ones could not
 * pay for. That bias now lives in `observeBandFraction`, where the scene term
 * belongs, so the rate itself can average honestly.
 */
export const observePost = (prev, postMs, proxyMP, bandFraction) => {
    if (!(postMs > 0) || !(proxyMP > 0) || !(bandFraction > 0)) return prev || 0
    const sample = postMs / (proxyMP * bandShape(bandFraction))
    if (!(prev > 0)) return Math.round(sample)
    return Math.round(prev * 0.7 + sample * 0.3)
}

/**
 * Fold one click's band into the planning fraction — the scene term.
 *
 * Biased toward the expensive: a frame-spanning band is believed at once, a
 * small one has to repeat before it buys resolution back. Clicks are sampled
 * from whatever the user happens to select, so the cheap ones are the majority
 * and a symmetric average would keep authorising a proxy that the next
 * full-frame selection cannot pay for inside the budget.
 */
export const observeBandFraction = (prev, fraction) => {
    if (!(fraction > 0)) return prev || 0
    const f = fraction > 1 ? 1 : fraction
    if (!(prev > 0)) return +f.toFixed(3)
    return +(f > prev ? prev * 0.5 + f * 0.5 : prev * 0.9 + f * 0.1).toFixed(3)
}

/* ─── Persistence ─────────────────────────────────────────────────────────
 * One key holds the whole judgement (rate, fraction, when), so a device is
 * measured once rather than once per session.
 *
 * It EXPIRES. The rate is a property of the machine as it was running that
 * day: on battery, thermally throttled, or sharing the CPU with a build. A
 * figure taken then only ever ratchets DOWN (the asymmetric fraction and the
 * 0.7 rate memory both resist recovery), so without a stop it would follow a
 * laptop back onto AC power for weeks. Re-measuring costs one click, so the
 * cheap answer is to let it lapse.
 */
const KEY = 'seglab.postfit'
const LEGACY_KEY = 'seglab.postMsPerMP'
export const FIT_TTL_MS = 14 * 24 * 60 * 60_000

const sane = (v) => Number.isFinite(v) && v >= 10 && v <= 5000

/** { msPerMP, bandFraction, ageMs } or null. Junk, absurd and stale all read
 *  as null: a poisoned figure would shrink every proxy with no visible cause. */
export const loadPostFit = (now = Date.now()) => {
    try {
        const raw = globalThis.localStorage?.getItem(KEY)
        if (!raw) return null
        const v = JSON.parse(raw)
        const rate = Number(v?.r)
        if (!sane(rate)) return null
        const t = Number(v?.t) || 0
        const ageMs = now - t
        if (!t || ageMs < 0 || ageMs > FIT_TTL_MS) return null
        const f = Number(v?.f)
        return { msPerMP: rate, bandFraction: f > 0 && f <= 1 ? f : 1, ageMs }
    } catch { return null }
}

export const savePostFit = ({ msPerMP, bandFraction }, now = Date.now()) => {
    try {
        if (!sane(msPerMP)) return
        globalThis.localStorage?.setItem(KEY, JSON.stringify({
            r: Math.round(msPerMP),
            f: +(bandFraction > 0 && bandFraction <= 1 ? bandFraction : 1).toFixed(3),
            t: now,
        }))
    } catch { /* private mode / storage disabled — the estimate still works */ }
}

/** Forget this device's measurement (?post=reset, and the one-time migration
 *  away from the un-normalised key this replaced). */
export const clearPostFit = () => {
    try {
        globalThis.localStorage?.removeItem(KEY)
        globalThis.localStorage?.removeItem(LEGACY_KEY)
    } catch { /* storage disabled */ }
}

/* ─── Detector lane ───────────────────────────────────────────────────────
 * The same judgement, applied to the one other thing on this device that is
 * paid per unit of work: the text lane's tile grid.
 *
 * A cell is one 640² pass through the YOLOE graph, and a search runs 1 (full
 * frame) or 1+g² of them. Cells cost MEMORY as well as time, and memory is
 * still ruled by class signals in policy.js — an arena that only grows cannot
 * be benched by a stopwatch. This bounds the other axis: how long a search may
 * spend on inference, which IS measurable and is measured on every search.
 */

/** Reference cost of one cell, measured on this machine (WebGPU, nc swept
 *  1→128: 128·132·134·134·137·138·146 ms — flat in slot count). */
export const REFERENCE_MS_PER_CELL = 140

export const estimateDetectMsPerCell = (cap = {}) => {
    const { penalty, reasons, inputs } = classPenalty(cap)
    return {
        msPerCell: Math.round(REFERENCE_MS_PER_CELL * penalty * ESTIMATE_HEADROOM),
        source: 'estimated',
        penalty,
        reasons,
        inputs,
    }
}

/**
 * Cells this device can pay for inside `detectorBudgetMs`.
 *
 * Returns Infinity when unjudged, so the caller's own (memory) ceiling stands
 * alone — the same rule the proxy ladder follows: no judgement changes nothing.
 */
export const affordableCells = (budget = {}) => {
    const msPerCell = Number(budget.detectorMsPerCell) || 0
    const budgetMs = Number(budget.detectorBudgetMs) || 0
    if (msPerCell <= 0 || budgetMs <= 0) return Infinity
    return Math.max(1, Math.floor(budgetMs / msPerCell))
}

/** Fold one real search in. Cheap evidence repeats, for the same reason the
 *  band fraction does: a cached-session search is not a cold one. */
export const observeDetect = (prev, detectMs, cells) => {
    if (!(detectMs > 0) || !(cells > 0)) return prev || 0
    const sample = detectMs / cells
    if (!(prev > 0)) return Math.round(sample)
    return Math.round(sample > prev ? prev * 0.5 + sample * 0.5 : prev * 0.9 + sample * 0.1)
}

const CELL_KEY = 'seglab.detectMsPerCell'

export const loadDetectMsPerCell = (now = Date.now()) => {
    try {
        const raw = globalThis.localStorage?.getItem(CELL_KEY)
        if (!raw) return 0
        const v = JSON.parse(raw)
        const ms = Number(v?.r)
        const t = Number(v?.t) || 0
        if (!Number.isFinite(ms) || ms < 10 || ms > 20_000) return 0
        return t && now - t >= 0 && now - t <= FIT_TTL_MS ? ms : 0
    } catch { return 0 }
}

export const saveDetectMsPerCell = (ms, now = Date.now()) => {
    try {
        if (!Number.isFinite(ms) || ms < 10 || ms > 20_000) return
        globalThis.localStorage?.setItem(CELL_KEY, JSON.stringify({ r: Math.round(ms), t: now }))
    } catch { /* storage disabled */ }
}

/**
 * memory-governor — the runtime safety net + tier-climb signal.
 *
 * The old watchdog watched `performance.memory.usedJSHeapSize`, which is blind
 * to the WASM/GPU memory that actually OOMs the app: it stayed flat at ~41 MB
 * while a WASM SlimSAM heap ballooned to ~3 GB (measured). This governor watches
 * the signals that actually move:
 *
 *  - `measureUserAgentSpecificMemory()` (crossOriginIsolated only): real bytes
 *    for the whole same-origin agent cluster — main thread + workers + WASM.
 *    This SEES a runaway WASM heap. (GPU-process memory is separate and not
 *    counted, which is why timer-drift is the paired signal.)
 *  - timer drift: a 1 s tick landing seconds late means the OS is swapping /
 *    starving this device — the one device-relative signal, works everywhere.
 *  - `performance.memory` JS heap: coarse tertiary (only when no measured bytes).
 *
 * Bidirectional but asymmetric: it sheds (down) the instant any signal shows
 * pressure; it emits a climb signal (up) only when measured bytes stay well
 * under the tier budget AND there is no drift for a sustained cooldown. Down
 * always wins. A device without the measured signal never climbs (no positive
 * proof) — it simply stays safe at its detected tier.
 */

const MB = 1024 * 1024

/**
 * Pure decision from a single sample → { level: 0..3, headroom: bool }.
 * `bytesMB` is the measured agent-cluster figure (0 = unavailable this cycle).
 * Exported so verify can exercise the ladder without a browser.
 */
export const decidePressure = ({ bytesMB = 0, budgetMB = 1800, driftMs = 0, heapMB = 0 } = {}) => {
    let level = 0
    // Measured footprint vs the tier's soft ceiling — catches the app itself
    // over-allocating (runaway WASM heap, oversized export, stacked images).
    if (bytesMB > 0) {
        if (bytesMB > budgetMB * 1.2) level = 3
        else if (bytesMB > budgetMB) level = 2
        else if (bytesMB > budgetMB * 0.85) level = 1
    } else if (heapMB > 0) {
        // No measured bytes (non-COI / rate-limited): coarse JS-heap floor.
        if (heapMB > 650) level = 3
        else if (heapMB > 450) level = 2
        else if (heapMB > 300) level = 1
    }
    // Timer drift is the device-relative swap signal — and on a unified-memory
    // host it is the ONLY early warning we get: the GPU/unified bytes that
    // actually push the machine into swap are invisible to the measure API
    // above, so a green byte reading routinely coincides with a swapping OS.
    // Tuned to fire at swap ONSET (a foreground 1 s tick landing a few hundred
    // ms late) instead of after the multi-second freeze the old 2.5/5 s
    // thresholds waited through — by then the machine was already frozen.
    // L3 (which now releases the ORT session arena — an expensive rebuild) is
    // reserved for an unambiguous multi-second OS freeze; a lone 2 s spike can
    // be GC/compile jank and only defers work at L2.
    if (driftMs > 4000) level = Math.max(level, 3)
    else if (driftMs > 900) level = Math.max(level, 2)
    else if (driftMs > 450) level = Math.max(level, 1)
    // Headroom is only ever PROVEN by a real byte reading well under budget with
    // no drift — never inferred from the absence of a signal.
    const headroom = level === 0 && bytesMB > 0 && bytesMB < budgetMB * 0.5 && driftMs < 500
    return { level, headroom }
}

/**
 * Driver around `decidePressure`. Callbacks:
 *  - `getBudget()`  → current budget object (reads `memBudgetMB`, `pressureLevel`).
 *  - `onPressure(level)` → shed to at least `level` (one-way ratchet on the app side).
 *  - `onHeadroom()` → sustained proven headroom; caller may climb ONE tier.
 *  - `isActive()`   → only monitor while a document is loaded.
 *  - `onSample(sample)` → optional telemetry hook (`?debug=1`).
 *
 * `measureUserAgentSpecificMemory()` is intentionally slow (batched with GC,
 * randomly delayed to defeat timing attacks) — often many seconds — so it CANNOT
 * be awaited in the decision loop. It runs in the background on its own cadence
 * and updates a cached value; the fast, synchronous decision loop reads that
 * cache alongside timer-drift (the responsive swap signal) and JS heap.
 */
export const createMemoryGovernor = ({
    getBudget, onPressure, onHeadroom, isActive, onSample,
    intervalMs = 2500, headroomCycles = 4, measureCooldownMs = 15000, staleAfterMs = 60000,
} = {}) => {
    const canMeasure = typeof performance !== 'undefined'
        && typeof performance.measureUserAgentSpecificMemory === 'function'
    let decisionTimer = null
    let driftTimer = null
    let lastTick = 0
    let drift = 0
    let cleanStreak = 0
    let lastFiredLevel = 0
    // Background measurement cache.
    let measuredMB = 0
    let measuredAt = 0
    let measuring = false
    let lastKick = 0

    // Shed to at least `level`, but don't re-fire the same-or-lower level (the
    // app-side ratchet makes a repeat a no-op — this only trims the noise).
    // Exception: L3 may repeat — its arena release is real work each time (the
    // session regrows on the next selection), and a machine that freezes again
    // after a rebuild needs the arena back again.
    const firePressure = (level) => {
        if (level <= 0) return
        if (level < 3 && level <= lastFiredLevel) return
        lastFiredLevel = level
        onPressure?.(level)
    }

    const driftLoop = () => {
        const now = performance.now()
        // A backgrounded/idle span throttles these very timers, which looks
        // exactly like swap drift — reset across it so it never drives a shed.
        if (isActive && !isActive()) { drift = 0; lastTick = now; lastFiredLevel = 0; return }
        if (lastTick) drift = Math.max(0, now - lastTick - 1000)
        lastTick = now
        // Act on drift HERE, on the fast 1 s loop. The slow decision cycle can
        // itself be starved by the swap we're trying to catch, so the swap
        // signal must not wait for it; the byte path stays on that cycle.
        if (drift > 450) firePressure(decidePressure({ driftMs: drift }).level)
    }

    // Fire-and-forget: never blocks the decision loop.
    const kickMeasure = () => {
        if (!canMeasure || measuring) return
        const now = Date.now()
        if (now - lastKick < measureCooldownMs) return
        lastKick = now
        measuring = true
        performance.measureUserAgentSpecificMemory()
            .then((r) => { measuredMB = Math.round((r?.bytes || 0) / MB); measuredAt = Date.now() })
            .catch(() => { /* rate-limited / context-specific */ })
            .finally(() => { measuring = false })
    }

    const cycle = () => {
        if (isActive && !isActive()) { cleanStreak = 0; return }
        kickMeasure()
        const budget = (getBudget && getBudget()) || {}
        const budgetMB = budget.memBudgetMB || 1800
        // Use the cached measurement only while it's fresh; a stale reading must
        // not drive decisions after conditions have changed.
        const fresh = measuredMB > 0 && (Date.now() - measuredAt) < staleAfterMs
        const bytesMB = fresh ? measuredMB : 0
        const heapMB = performance.memory ? Math.round(performance.memory.usedJSHeapSize / MB) : 0
        const { level, headroom } = decidePressure({ bytesMB, budgetMB, driftMs: drift, heapMB })
        onSample?.({ bytesMB, heapMB, driftMs: Math.round(drift), budgetMB, level, headroom, measuring, pressureLevel: budget.pressureLevel || 0 })
        if (level > 0) {
            cleanStreak = 0
            firePressure(level)
        } else if (headroom && (budget.pressureLevel || 0) === 0) {
            // Climb only from a calm, un-shed state, and only on a real byte reading.
            cleanStreak += 1
            if (cleanStreak >= headroomCycles) { cleanStreak = 0; onHeadroom?.() }
        } else {
            cleanStreak = 0
        }
    }

    return {
        start() {
            if (driftTimer) return
            lastFiredLevel = 0
            lastTick = (typeof performance !== 'undefined' ? performance.now() : 0)
            driftTimer = setInterval(driftLoop, 1000)
            decisionTimer = setInterval(cycle, intervalMs)
        },
        stop() {
            if (driftTimer) clearInterval(driftTimer)
            if (decisionTimer) clearInterval(decisionTimer)
            driftTimer = decisionTimer = null
        },
        // Test/debug hooks.
        cycleNow: cycle,
        feedMeasurement: (mb) => { measuredMB = mb; measuredAt = Date.now() }, // inject bytes (tests/climb)
        canMeasure,
    }
}

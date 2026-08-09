/**
 * memory-governor — the runtime safety net + tier-climb signal.
 *
 * The old watchdog watched `performance.memory.usedJSHeapSize`, which is blind
 * to the WASM/GPU memory that actually OOMs the app: it stayed flat at ~41 MB
 * while a WASM SlimSAM heap ballooned to ~3 GB (measured). This governor watches
 * the signals that actually move:
 *
 * It judges THIS APP's footprint against the app's ceiling — never the machine's
 * RAM. Work runs on WebGPU; a host that is swapping because of everything else
 * the user has open is not a reason to degrade seglab.
 *
 *  - the caller's ledger: what the app knows it allocated (worker arena, device
 *    pool, encoder session, its own canvases). Available everywhere.
 *  - `measureUserAgentSpecificMemory()` (crossOriginIsolated only): real bytes,
 *    but ONLY for this agent cluster. The SAM lane's SharedWorker is a different
 *    one, so the ORT arena — the multi-GB resident — is not in it: measured
 *    75 MB while the app held ~2 GB. It is a floor, never a ceiling, and a small
 *    reading must not be allowed to silence the ledger.
 *  - `performance.memory` JS heap: coarse last resort, no ledger and no bytes.
 *  - timer drift: device-relative, so it can no longer shed. It only vetoes a
 *    CLIMB — don't get greedier on a machine that is already struggling.
 *
 * Asymmetric: sheds the instant the footprint crosses a band, climbs only after
 * a sustained run well under the ceiling.
 */

const MB = 1024 * 1024

/**
 * Pure decision from a single sample → { level: 0..3, headroom: bool }.
 * `bytesMB` is the measured agent-cluster figure (0 = unavailable this cycle).
 * Exported so verify can exercise the ladder without a browser.
 */
export const decidePressure = ({ bytesMB = 0, budgetMB = 2200, driftMs = 0, heapMB = 0, estimateMB = 0 } = {}) => {
    // Whichever signal saw MORE. `bytesMB` misses the SharedWorker cluster
    // entirely, so it can only ever raise the floor, never lower it.
    const footprintMB = Math.max(bytesMB, estimateMB)
    let level = 0
    if (footprintMB > 0) {
        // Bands sit ABOVE normal operation. With an encoder live the ledger rests
        // at ~1980 MB, so the old 0.85 warn band (1615 of an 1800 budget) put a
        // healthy app in permanent pressure — which is exactly what it did on
        // WebKit, where the ledger is the only input.
        if (footprintMB > budgetMB * 1.15) level = 3
        else if (footprintMB > budgetMB) level = 2
        else if (footprintMB > budgetMB * 0.95) level = 1
    } else if (heapMB > 0) {
        // No ledger and no bytes: coarse JS-heap floor.
        if (heapMB > 650) level = 3
        else if (heapMB > 450) level = 2
        else if (heapMB > 300) level = 1
    }
    // Drift measures the MACHINE, not this app — another process swapping the
    // host is not a reason to pause our WebGPU work. It only blocks a climb.
    const headroom = level === 0 && footprintMB > 0 && footprintMB < budgetMB * 0.5 && driftMs < 500
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
    getBudget, onPressure, onHeadroom, isActive, onSample, getEstimateMB,
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
        // ALWAYS: a byte reading that cannot see the lane's worker must not
        // suppress the one signal that can.
        const estimateMB = Math.round(getEstimateMB?.() || 0)
        const { level, headroom } = decidePressure({ bytesMB, budgetMB, driftMs: drift, heapMB, estimateMB })
        onSample?.({ bytesMB, estimateMB, heapMB, driftMs: Math.round(drift), budgetMB, level, headroom, measuring, pressureLevel: budget.pressureLevel || 0 })
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

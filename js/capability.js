/**
 * capability — device detection + memory-pressure signal
 * --------------------------------------------------------
 * Detection is deliberately pessimistic: browsers cap
 * navigator.deviceMemory at 8 GB (Chrome) or omit it (Safari/Firefox), so
 * `deviceMemoryGB` is a floor-of-truth, never a promise of headroom.
 *
 * The pressure signal is a monotonic latch (0→3) for the session:
 *  - performance.memory is a Chrome-only HINT (JS heap only — it cannot see
 *    GPU, bitmap, or browser-native allocations) and is polled gently;
 *  - real failures (allocation throws, WebGPU device loss, decode errors)
 *    escalate immediately via reportPressure;
 *  - levels never auto-decrease; the debug release-memory action may reset.
 */

export const detectCapability = () => {
    const nav = typeof navigator !== 'undefined' ? navigator : {}
    const ua = String(nav.userAgent || '')
    return {
        deviceMemoryGB: Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
        hardwareConcurrency: Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : null,
        isSafari: /Safari\//.test(ua) && !/Chrom(e|ium)|Edg\//.test(ua),
        hasImageDecoder: typeof ImageDecoder !== 'undefined',
        hasOffscreenCanvas: typeof OffscreenCanvas !== 'undefined',
        hasPerformanceMemory: typeof performance !== 'undefined' && !!performance.memory,
    }
}

/* ─── Pressure latch ─────────────────────────────────────────────────────── */

let pressureLevel = 0
let lastReason = null
const listeners = new Set()

export const getPressureLevel = () => pressureLevel
export const getPressureReason = () => lastReason

/** Subscribe to level changes; returns an unsubscribe function. */
export const onPressureChange = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }

/** Raise (never lower) the session pressure level. */
export const reportPressure = (level, reason) => {
    const next = Math.max(0, Math.min(3, level | 0))
    if (next <= pressureLevel) return pressureLevel
    pressureLevel = next
    lastReason = reason || null
    console.warn(`[seglab] memory pressure → level ${pressureLevel}${reason ? ` (${reason})` : ''}`)
    for (const cb of listeners) { try { cb(pressureLevel, lastReason) } catch { /* listener bug */ } }
    return pressureLevel
}

/** Debug-only escape hatch (release-memory action). */
export const resetPressure = () => {
    pressureLevel = 0
    lastReason = null
    for (const cb of listeners) { try { cb(0, null) } catch { /* listener bug */ } }
}

/** Classify a caught error: does it smell like memory exhaustion? */
export const isMemoryError = (err) => {
    const msg = String(err?.message || err || '')
    return /out of memory|allocation failed|OOM|Array buffer allocation|RangeError/i.test(msg)
        || err instanceof RangeError
}

let monitorTimer = null

/**
 * Start the Chrome-only heap-hint poll. Thresholds are conservative and the
 * result is only ever an escalation hint — the static lite caps are the real
 * defense on every browser.
 */
export const startPressureMonitor = ({ intervalMs = 5000 } = {}) => {
    if (monitorTimer || typeof performance === 'undefined' || !performance.memory) return
    monitorTimer = setInterval(() => {
        try {
            const { usedJSHeapSize, jsHeapSizeLimit } = performance.memory
            if (!usedJSHeapSize || !jsHeapSizeLimit) return
            const ratio = usedJSHeapSize / jsHeapSizeLimit
            if (ratio > 0.92) reportPressure(2, `JS heap ${(ratio * 100) | 0}% of limit`)
            else if (ratio > 0.8) reportPressure(1, `JS heap ${(ratio * 100) | 0}% of limit`)
        } catch { /* hint only */ }
    }, intervalMs)
}

export const stopPressureMonitor = () => {
    clearInterval(monitorTimer)
    monitorTimer = null
}

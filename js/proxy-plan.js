/**
 * proxy-plan (pure) — the one authoritative proxy-sizing function plus the
 * budget-aware interaction plan. Every import path (JPEG/PNG/WebP/AVIF/TIFF/
 * GIF/BMP/RAW preview/demo/paste/drop) sizes through here.
 */

/** Bounded proxy dimensions for a source. Throws on invalid dimensions. */
export function getBoundedProxySize(sourceWidth, sourceHeight, maxLongSide = 768) {
    if (
        !Number.isFinite(sourceWidth)
        || !Number.isFinite(sourceHeight)
        || sourceWidth <= 0
        || sourceHeight <= 0
    ) {
        throw new Error('Invalid image dimensions')
    }
    const longSide = Math.max(sourceWidth, sourceHeight)
    const scale = Math.min(1, maxLongSide / longSide)
    return {
        width: Math.max(1, Math.round(sourceWidth * scale)),
        height: Math.max(1, Math.round(sourceHeight * scale)),
        scale,
        proxyActive: scale < 1,
    }
}

/**
 * Decide whether this image needs a proxy under `budget`. In auto mode a
 * source at or below the device cap is native-sized. `proxy=disabled` (only
 * reachable on trusted budgets) is honored only below the direct-image
 * budget; larger sources return to the device's safe cap.
 */
export const interactionPlan = (w, h, budget = {}) => {
    const longSide = Math.max(w, h)
    const directSafe = (w * h) <= (budget.directMaxMP || 2) * 1e6
        && longSide <= (budget.directMaxSide || 2048)
    const disabled = budget.proxyMode === 'disabled'
    const cap = disabled && directSafe
        ? longSide
        : (disabled ? (budget.safeProxyMax || 768) : (budget.proxyMax || 768))
    const { scale, proxyActive } = getBoundedProxySize(w, h, cap)
    return {
        scale,
        proxyActive,
        proxyReason: disabled && !directSafe ? 'safety' : (proxyActive ? 'device' : 'native'),
    }
}

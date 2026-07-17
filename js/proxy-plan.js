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

/**
 * Ceiling (in megapixels) on the one-shot full-raster decode an unbounded
 * host (no ImageDecoder — Safari) may pay for the display frame. Reuses the
 * budget's already-trusted one-shot crop size; the pressure ratchet lowers
 * it. Pixel budgets only — never a RAM figure.
 */
export const decodeBudgetMP = (budget = {}) => {
    const base = budget.escalateMaxMP || 8
    const level = budget.pressureLevel || 0
    if (level >= 3) return 2
    if (level >= 2) return 4
    if (level >= 1) return Math.min(base, 6)
    return base
}

/**
 * Display long edge for ANY source (JPEG/PNG/HEIC, RAW embedded preview,
 * LibRaw-developed). The screen is the anchor: the preview never needs more
 * pixels than the viewport can show (plus zoom slack), never exceeds the
 * profile ceiling or the GPU texture limit, and never upscales the source.
 * `allowFullDecode` gates whether an unbounded host may decode the full
 * raster once to build it.
 */
export const displayPlan = ({ srcW, srcH, budget = {}, viewport = {}, textureLimit = 0 }) => {
    if (budget.displayMode === 'off') return { side: 0, allowFullDecode: false }
    const srcLong = Math.max(srcW || 0, srcH || 0)
    const texCap = textureLimit > 0 ? textureLimit : Infinity
    if (budget.displayMode === 'native') {
        return { side: Math.min(srcLong, texCap), allowFullDecode: true } // explicit opt-in
    }
    const dpr = Math.min(viewport.dpr || 1, 3) // clamp odd hosts
    const ZOOM_SLACK = 1.5                     // pinch-zoom headroom
    const viewportLong = Math.max(viewport.w || 0, viewport.h || 0) || 1280
    const need = Math.round(viewportLong * dpr * ZOOM_SLACK)
    const side = Math.min(need, budget.displayMax || 2048, texCap, srcLong)
    const allowFullDecode = ((srcW * srcH) / 1e6) <= decodeBudgetMP(budget)
    return { side, allowFullDecode }
}

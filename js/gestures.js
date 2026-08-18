/* gestures — zoom, pan, and the input vocabulary each device actually has.
 *
 * One selection surface, three very different hands on it:
 *
 *   mouse     wheel zooms, middle-drag (or space-drag) pans, right-click excludes
 *   trackpad  pinch zooms, two-finger scroll pans, ⌥-click excludes
 *   touch     pinch zooms and pans together, long-press excludes
 *
 * The device is not asked for once at boot and trusted forever. `pointer:
 * coarse` separates a finger from a pointer, but nothing in CSS separates a
 * trackpad from a mouse — both are `fine` + `hover`. That only shows up in the
 * shape of the wheel events: a trackpad emits fractional deltas, a horizontal
 * axis, and ctrl+wheel for pinch; a wheel emits coarse multiples on one axis.
 * So the class starts as `mouse` and is upgraded the first time a wheel event
 * could only have come from a trackpad. Nothing is lost if that never happens —
 * the mouse bindings are the safe superset.
 *
 * Only #frame is transformed. The scope control and the zoom readout are its
 * siblings, so they keep their real size at every zoom level, and every
 * pointer-to-canvas conversion in the host still works untouched: it goes
 * through getBoundingClientRect, which already reports the transformed box.
 */

const MIN_ZOOM = 1
const MAX_ZOOM = 8
/** Trackpad-shaped events needed before the bindings switch. Two, because one
 *  stray fractional delta must not take wheel-zoom away from a mouse. */
const TRACKPAD_CONFIDENCE = 2

const isTypingTarget = (t) =>
    !!t && (t.isContentEditable || /^(input|textarea|select)$/i.test(t.tagName || ''))

const KEY_HINT = '<kbd>Z</kbd> undo <span class="sep">·</span> <kbd>R</kbd> reset <span class="sep">·</span> <kbd>E</kbd> raw'

const HINTS = {
    mouse: `wheel zoom <span class="sep">·</span> middle-drag pan <span class="sep">·</span> right-click excludes <span class="sep">·</span> ${KEY_HINT}`,
    trackpad: `pinch zoom <span class="sep">·</span> two-finger scroll pans <span class="sep">·</span> ⌥-click excludes <span class="sep">·</span> ${KEY_HINT}`,
    touch: 'pinch to zoom <span class="sep">·</span> two fingers pan <span class="sep">·</span> long-press excludes <span class="sep">·</span> ± switches sign',
}

/**
 * @param {object} hooks
 * @param {HTMLElement} hooks.stage    the clipping frame (positioning context)
 * @param {HTMLElement} hooks.frame    the transformed element holding the canvases
 * @param {HTMLElement} hooks.surface  the top canvas the host listens on
 * @param {HTMLElement} [hooks.hint]   footer element that names the gestures
 * @param {HTMLElement} [hooks.readout] the zoom pill (also the way back to 1×)
 * @param {() => boolean} hooks.active  is there an image to zoom?
 * @param {() => void} hooks.onGestureStart  a two-finger gesture took over — drop any draft
 * @param {(zoom: number) => void} [hooks.onZoom] zoom or pan changed
 */
export const createGestures = ({ stage, frame, surface, hint, readout, active, onGestureStart, onZoom }) => {
    let zoom = 1
    let tx = 0
    let ty = 0
    let input = matchMedia('(pointer: coarse)').matches ? 'touch' : 'mouse'
    let trackpadScore = 0
    let spaceHeld = false
    let pan = null              // { id, x, y } — middle-drag or space-drag
    const touches = new Map()   // pointerId → [x, y] in stage space
    let pinch = null            // { dist, cx, cy }
    let suppress = false        // a multi-touch gesture owns the surface

    const setInput = (next) => {
        if (input === next) return
        input = next
        document.body.dataset.input = next
        if (hint) hint.innerHTML = HINTS[next]
    }
    document.body.dataset.input = input
    if (hint) hint.innerHTML = HINTS[input]

    const rel = (e) => {
        const r = stage.getBoundingClientRect()
        return [e.clientX - r.left, e.clientY - r.top]
    }

    /** Keep the photo covering the frame: pan is only ever slack, never a way
     *  to lose the image off the edge of the stage. */
    const clamp = () => {
        const w = frame.offsetWidth * zoom
        const h = frame.offsetHeight * zoom
        const sw = stage.clientWidth
        const sh = stage.clientHeight
        tx = w <= sw ? (sw - w) / 2 : Math.min(0, Math.max(sw - w, tx))
        ty = h <= sh ? (sh - h) / 2 : Math.min(0, Math.max(sh - h, ty))
    }

    const apply = () => {
        clamp()
        frame.style.transform = zoom === 1 && !tx && !ty
            ? '' : `translate3d(${tx}px, ${ty}px, 0) scale(${zoom})`
        if (readout) {
            readout.hidden = zoom === 1
            readout.textContent = `${zoom < 10 ? zoom.toFixed(1) : Math.round(zoom)}×  reset`
        }
        stage.dataset.pan = zoom > 1 ? (pan ? 'active' : 'ready') : ''
        onZoom?.(zoom)
    }

    /** Zoom about a stage-space point, so whatever is under the cursor or
     *  between the fingers stays there. */
    const zoomAt = (x, y, factor) => {
        const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor))
        if (next === zoom) return
        const ix = (x - tx) / zoom
        const iy = (y - ty) / zoom
        zoom = next
        tx = x - ix * zoom
        ty = y - iy * zoom
        apply()
    }

    const panBy = (dx, dy) => {
        if (zoom === 1) return
        tx += dx
        ty += dy
        apply()
    }

    const reset = () => {
        zoom = 1
        tx = 0
        ty = 0
        apply()
    }

    /* ── Wheel: the same physical gesture means different things by device ──
       The scope control listens on the surface and calls preventDefault when
       the wheel lands on the live selection, so "how much to select" always
       wins over zoom — a wheel over the subject changes the reading, a wheel
       anywhere else changes the view. */
    const onWheel = (e) => {
        if (e.defaultPrevented || !active()) return
        // ctrl+wheel is the pinch gesture on every trackpad, and a browser
        // page-zoom request otherwise — either way it means scale, here.
        if (e.ctrlKey) {
            trackpadScore = TRACKPAD_CONFIDENCE
            setInput(input === 'touch' ? 'touch' : 'trackpad')
        } else if (input !== 'touch' && e.deltaMode === 0
            && (e.deltaX !== 0 || !Number.isInteger(e.deltaY))) {
            // A mouse wheel is integer notches on one axis. A fraction or a
            // horizontal component can only be a surface with two fingers on it.
            if (++trackpadScore >= TRACKPAD_CONFIDENCE) setInput('trackpad')
        }
        e.preventDefault()
        const [x, y] = rel(e)
        // One flick of a free-spinning wheel can carry a delta of several
        // hundred; unclamped that is a jump from 1× to 7× in a single event.
        const dy = Math.max(-240, Math.min(240, e.deltaY))
        if (e.ctrlKey) { zoomAt(x, y, Math.exp(-dy * 0.01)); return }
        // A trackpad's two-finger scroll is a pan; a wheel's only axis is zoom.
        // At 1× there is nothing to pan, so scroll zooms whatever the device —
        // which also makes a misread device harmless instead of inert.
        if (input === 'trackpad' && zoom > 1) panBy(-e.deltaX, -e.deltaY)
        else zoomAt(x, y, Math.exp(-dy * 0.0016))
    }

    /* ── Pointers ─────────────────────────────────────────────────────────
       Capture phase on the stage, one level above the surface the host listens
       on, so a second finger can cancel the host's in-progress stroke before
       it ever sees the event. Nothing is intercepted while a single pointer is
       drawing — that is the host's gesture and it keeps it. */
    const onDown = (e) => {
        if (!active()) return
        if (e.pointerType === 'touch') {
            // `pointer: coarse` is a boot-time guess and a hybrid laptop answers
            // no: a finger on the glass is the proof, and it is what decides the
            // 44 px targets and the sign toggle.
            setInput('touch')
            touches.set(e.pointerId, rel(e))
            if (touches.size === 2) {
                suppress = true
                onGestureStart?.()
                const [a, b] = [...touches.values()]
                pinch = { dist: Math.hypot(a[0] - b[0], a[1] - b[1]), cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2 }
            }
            if (suppress) { e.stopPropagation(); e.preventDefault() }
            return
        }
        // Middle-drag, or space-drag: the two pans every pointer app has.
        if (e.button === 1 || (spaceHeld && e.button === 0)) {
            const [x, y] = rel(e)
            pan = { id: e.pointerId, x, y }
            stage.setPointerCapture?.(e.pointerId)
            apply()
            e.stopPropagation()
            e.preventDefault()
        }
    }

    const onMove = (e) => {
        if (pan && e.pointerId === pan.id) {
            const [x, y] = rel(e)
            panBy(x - pan.x, y - pan.y)
            pan.x = x
            pan.y = y
            e.stopPropagation()
            return
        }
        if (e.pointerType !== 'touch' || !touches.has(e.pointerId)) return
        touches.set(e.pointerId, rel(e))
        if (suppress) e.stopPropagation()
        if (!pinch || touches.size < 2) return
        const [a, b] = [...touches.values()]
        const dist = Math.hypot(a[0] - b[0], a[1] - b[1])
        const cx = (a[0] + b[0]) / 2
        const cy = (a[1] + b[1]) / 2
        // Pan by the midpoint first, then scale about it — the photo tracks the
        // fingers instead of sliding out from under them.
        panBy(cx - pinch.cx, cy - pinch.cy)
        if (pinch.dist > 0 && dist > 0) zoomAt(cx, cy, dist / pinch.dist)
        pinch = { dist, cx, cy }
    }

    const onUp = (e) => {
        if (pan && e.pointerId === pan.id) {
            pan = null
            apply()
            e.stopPropagation()
            return
        }
        if (e.pointerType !== 'touch') return
        touches.delete(e.pointerId)
        if (touches.size < 2) pinch = null
        // Keep swallowing until every finger is off: the one still down after a
        // pinch is a leftover, not a new stroke.
        if (suppress) {
            e.stopPropagation()
            if (touches.size === 0) suppress = false
        }
    }

    const onKey = (e) => {
        if (isTypingTarget(e.target)) return
        if (e.key === ' ' && !e.repeat && !spaceHeld && input !== 'touch') {
            spaceHeld = true
            if (zoom > 1) stage.dataset.pan = 'ready'
            e.preventDefault()
            return
        }
        if (e.metaKey || e.ctrlKey || e.altKey || !active()) return
        const cx = stage.clientWidth / 2
        const cy = stage.clientHeight / 2
        if (e.key === '+' || e.key === '=') { zoomAt(cx, cy, 1.35); e.preventDefault() }
        else if (e.key === '-' || e.key === '_') { zoomAt(cx, cy, 1 / 1.35); e.preventDefault() }
        else if (e.key === '0') { reset(); e.preventDefault() }
    }

    const onKeyUp = (e) => {
        if (e.key !== ' ') return
        spaceHeld = false
        if (!pan && zoom > 1) stage.dataset.pan = 'ready'
    }

    stage.addEventListener('wheel', onWheel, { passive: false })
    stage.addEventListener('pointerdown', onDown, { capture: true })
    stage.addEventListener('pointermove', onMove, { capture: true })
    stage.addEventListener('pointerup', onUp, { capture: true })
    stage.addEventListener('pointercancel', onUp, { capture: true })
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKeyUp)
    addEventListener('resize', apply)
    readout?.addEventListener('click', reset)
    // A trackpad's two-finger tap arrives as a contextmenu on the surface; the
    // host already reads button 2 as exclude, so it only has to not open a menu.
    surface?.addEventListener('contextmenu', (e) => e.preventDefault())

    return {
        reset,
        zoom: () => zoom,
        input: () => input,
        /** True while a multi-finger gesture owns the surface. */
        busy: () => suppress || !!pan,
    }
}

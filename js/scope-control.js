/* Scope control — how much of the thing the click selected.
 *
 * SAM returns three readings of one click and no ranking rule resolves a
 * genuinely ambiguous one (DESIGN-MASK-LANE §10a). The alternatives have to be
 * OFFERED, and the offer has to work for someone who has never read a word
 * about the model. Three things had to go before it did:
 *
 *   - `1.9% · 6.3% · 13.2%` names a measurement, not a choice.
 *   - `Part · Object · Whole` names a MEANING the model does not give. What
 *     comes back is three sizes; when the biggest reading is a pole plus a
 *     wire plus a patch of sky, calling it "Whole" makes it worse.
 *   - `C` is a key nobody discovers.
 *   - `− Less ○ ○ ● More +` names a direction but never a destination: the
 *     dots count the readings without showing one of them, so the only way to
 *     find out what "more" means is to spend a repaint and look.
 *
 * What is left is the choice itself. One swatch per reading, painted from that
 * candidate's own mask: the lamp head, the lamp with its arm, the pole with the
 * wire. That is the same bargain Photoshop's object finder makes — show the
 * shape, take it on click — with the ranking left to the eye, since no ranking
 * rule works here (§10a).
 *
 * A shared crop was tried first and measured: readings of 0.2 / 0.8 / 17 % of
 * the frame drew two invisible specks beside one filled subject, so the swatch
 * that mattered most — the one you are about to reject as too small — was the
 * one you could not see. Each shape gets its own crop instead, and SIZE carries
 * only the ORDER: the areas are spread across the swatch on a log scale, so
 * every reading is legible and smaller always looks smaller. Exact percentages
 * live in the tooltip and the status line, where a number belongs.
 *
 * The picking is direct, but every OTHER route in still steps by exactly one
 * and stops at the ends — scroll, drag, arrows, and the host's shortcut alike.
 * One rule, no wrap: a selection that jumps from the whole subject back to a
 * speck reads as a bug.
 *
 * No app state, no imports, no framework: a mount element, a surface to listen
 * on, and callbacks. Mask Studio and Phosmith import it as-is.
 */

/** Vertical travel that commits to a scrub instead of a tap. Above the tap
 *  slop of both pointer types, so a shaky hand never steps by accident. */
const SCRUB_ENTER_PX = 14

/** CSS px of scrub travel per step, and wheel delta per step. */
const SCRUB_STEP_PX = 34
const WHEEL_STEP = 40

/** Drawn size of a swatch cell, square. Fixed in CSS px on purpose: the tap
 *  target grows on touch (--tap), the drawing does not, so the shapes read the
 *  same anywhere. */
const SWATCH_H = 22

/** How much of the cell the smallest reading fills. Below about a third of a
 *  22 px cell a silhouette stops being a shape and becomes a dot. */
const MIN_FILL = 0.42

/** One step, clamped. The only stepping rule in the control. */
export const stepScope = (index, delta, count) => {
    if (count < 2) return index
    return Math.min(count - 1, Math.max(0, index + delta))
}

/** Scrub direction: up (negative dy) selects more, matching the wheel and the
 *  host's shortcut. One mental model everywhere — up is more. */
export const scrubIndex = (startIndex, dy, count) =>
    stepScope(startIndex, -Math.trunc(dy / SCRUB_STEP_PX), count)

const el = (tag, cls) => {
    const node = document.createElement(tag)
    if (cls) node.className = cls
    return node
}

const pctText = (coverage) => {
    const pct = coverage * 100
    return `${pct < 1 ? pct.toFixed(1) : Math.round(pct)}% of the frame`
}

/** Tight box of the set pixels and how many there are, or null for an empty
 *  shape. The count is the area the size ladder is built from — a bbox would
 *  call a diagonal wire as big as the pole it hangs from. */
const cropOf = (s) => {
    let x0 = s.side, y0 = s.side, x1 = -1, y1 = -1, area = 0
    for (let y = 0; y < s.side; y += 1) {
        const row = y * s.side
        for (let x = 0; x < s.side; x += 1) {
            if (!s.alpha[row + x]) continue
            area += 1
            if (x < x0) x0 = x
            if (x > x1) x1 = x
            if (y < y0) y0 = y
            if (y > y1) y1 = y
        }
    }
    return x1 < 0 ? null : { x0, y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area }
}

/** The cropped mask as a white silhouette, ready to be tinted and scaled. */
const silhouette = (s, crop) => {
    const img = new ImageData(crop.w, crop.h)
    for (let y = 0; y < crop.h; y += 1) {
        const src = (crop.y0 + y) * s.side + crop.x0
        for (let x = 0; x < crop.w; x += 1) {
            const o = (y * crop.w + x) * 4
            img.data[o] = 255
            img.data[o + 1] = 255
            img.data[o + 2] = 255
            img.data[o + 3] = s.alpha[src + x]
        }
    }
    const c = el('canvas')
    c.width = crop.w
    c.height = crop.h
    c.getContext('2d').putImageData(img, 0, 0)
    return c
}

/** How much of its cell each reading fills, on a log ladder between the
 *  smallest and largest of THIS click. Linear area would clamp a 20× spread
 *  into two identical dots; the ladder guarantees the gaps are visible while
 *  keeping the order true. */
const fills = (crops) => {
    const areas = crops.map((c) => Math.log(Math.max(1, c.area)))
    const lo = Math.min(...areas)
    const hi = Math.max(...areas)
    if (hi - lo < 1e-6) return crops.map(() => 1)
    return areas.map((a) => MIN_FILL + (1 - MIN_FILL) * ((a - lo) / (hi - lo)))
}

/**
 * Paint one swatch: this reading alone, fitted to `fill` of the cell and
 * centred.
 *
 * `aspect` un-squashes the mask, and it is the FRAME's, not the container's.
 * The decoder's field is a square resize of the frame, so a tram measuring
 * 0.74:1 in mask space is 1.26:1 on the canvas — the swatch has to show what
 * the user sees, or it is showing another shape.
 *
 * One `source-in` pass takes the colour from the button's computed `color`,
 * which keeps the palette in the host CSS instead of in here.
 */
const paint = (node, src, fill, aspect) => {
    const dpr = Math.min(3, devicePixelRatio || 1)
    const box = Math.round(SWATCH_H * dpr)
    node.width = box
    node.height = box
    node.style.width = `${SWATCH_H}px`
    node.style.height = `${SWATCH_H}px`
    const ctx = node.getContext('2d')
    ctx.imageSmoothingEnabled = true
    const sw = src.width * aspect
    const k = (box * fill) / Math.max(sw, src.height)
    const w = Math.max(2, sw * k)
    const h = Math.max(2, src.height * k)
    ctx.drawImage(src, (box - w) / 2, (box - h) / 2, w, h)
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = getComputedStyle(node).color
    ctx.fillRect(0, 0, box, box)
}

/**
 * @param {object} hooks
 * @param {HTMLElement} hooks.mount     container for the control (positioned by this module)
 * @param {HTMLElement} hooks.surface   element the gestures listen on (the overlay canvas)
 * @param {() => boolean} hooks.enabled gestures allowed right now (right tool, not busy)
 * @param {(x: number, y: number) => boolean} hooks.inside  is this CSS-relative point on the live selection?
 * @param {() => {anchor: [number, number], bounds: [number, number] | null, width: number, height: number, aspect: number}} hooks.geometry
 *        placement box (the container) plus `aspect`, the displayed FRAME's
 *        width/height — the swatches are drawn in frame space, and the two
 *        boxes are only equal while the container hugs the photo.
 * @param {(index: number) => void} hooks.onPick     adopt a reading
 * @param {(index: number | null) => void} hooks.onPreview  ghost a reading without taking it
 * @param {(index: number) => {alpha: Uint8ClampedArray, side: number} | null} [hooks.shape]
 *        that reading's coarse mask, for the swatch. Without it the control
 *        falls back to plain blocks.
 */
export const createScopeControl = ({ mount, surface, enabled, inside, geometry, onPick, onPreview, shape }) => {
    let view = null      // { count, index, items }
    let scrub = null     // { id, x, y, index, stepped, shown }
    let wheelAcc = 0

    const live = () => view && view.count >= 2 && enabled()

    const place = () => {
        if (!view || mount.hidden) return
        const g = geometry()
        if (!g || !g.width) return
        const w = mount.offsetWidth || 180
        const h = mount.offsetHeight || 28
        // Clear of the selection, not on it: the control exists to compare
        // shapes, and a bar parked over the subject hides the evidence.
        const [x, y] = g.anchor
        const b = g.bounds || [y, y]
        const top = (b[1] + 14 + h < g.height) ? b[1] + 14
            : (b[0] - 14 - h > 6 ? b[0] - 14 - h : Math.max(6, Math.min(g.height - h - 6, y + 18)))
        mount.style.left = `${Math.min(g.width - w / 2 - 6, Math.max(w / 2 + 6, x))}px`
        mount.style.top = `${top}px`
    }

    const hide = () => {
        view = null
        mount.hidden = true
        mount.replaceChildren()
    }

    /** Hovering a control shows what it would do — the answer to "what is this
     *  button?" is the shape itself, not a word. */
    const previewOn = (node, index) => {
        const show = () => { if (index !== view.index) onPreview(index) }
        const clear = () => onPreview(null)
        node.addEventListener('pointerenter', show)
        node.addEventListener('pointerleave', clear)
        node.addEventListener('focus', show)
        node.addEventListener('blur', clear)
    }

    const render = () => {
        if (!view || view.count < 2) { hide(); return }
        // A repaint rebuilds the row, so a keyboard user who just stepped would
        // lose the focus ring unless it is put back on the new active swatch.
        const refocus = mount.contains(document.activeElement)
        const shapes = shape ? view.items.map((_, i) => shape(i)) : []
        const crops = shapes.length === view.count ? shapes.map((s) => s && cropOf(s)) : []
        const drawable = crops.length === view.count && crops.every(Boolean)
        const fill = drawable ? fills(crops) : []
        const g = geometry()
        const aspect = g && g.aspect > 0 ? g.aspect : 1
        const frag = document.createDocumentFragment()
        const painters = []
        view.items.forEach((item, i) => {
            const b = el('button', 'scope-shape')
            b.type = 'button'
            b.setAttribute('aria-pressed', String(i === view.index))
            // The eye gets the shape; a screen reader gets the size in words.
            b.setAttribute('aria-label', `${pctText(item.coverage)}, ${i + 1} of ${view.count}`)
            b.title = `${pctText(item.coverage)} · confidence ${item.score.toFixed(2)}`
            b.addEventListener('click', () => { if (i !== view.index) onPick(i) })
            previewOn(b, i)
            if (drawable) {
                const c = el('canvas')
                b.append(c)
                const src = silhouette(shapes[i], crops[i])
                // Painting reads the button's computed colour, so it can only
                // happen once the button is in the document.
                painters.push(() => paint(c, src, fill[i], aspect))
            } else {
                // No shape available (a host without one, or a dropped
                // candidate): a plain block still says how many there are.
                b.classList.add('bare')
            }
            frag.append(b)
        })
        mount.replaceChildren(frag)
        mount.hidden = false
        painters.forEach((p) => p())
        if (refocus) mount.children[view.index]?.focus()
        place()
    }

    /** @param {{count: number, index: number, items: Array<{coverage: number, score: number}>} | null} next */
    const update = (next) => {
        if (!next || next.count < 2) { hide(); return }
        view = next
        render()
    }

    /* ── Gestures ─────────────────────────────────────────────────────────
       Both listeners sit alongside the host's own pointer handling and never
       cancel it: a scrub only starts after the host has already written the
       drag off as moved, so nothing is taken over mid-stroke. */

    const rel = (e) => {
        const r = surface.getBoundingClientRect()
        return [e.clientX - r.left, e.clientY - r.top]
    }

    const onDown = (e) => {
        scrub = null
        if (!live() || !e.isPrimary) return
        const [x, y] = rel(e)
        if (!inside(x, y)) return
        scrub = { id: e.pointerId, x, y, index: view.index, stepped: false, shown: null }
    }

    const onMove = (e) => {
        if (!scrub || e.pointerId !== scrub.id || !live()) return
        const [x, y] = rel(e)
        const dy = y - scrub.y
        // Vertical intent only — a horizontal drag is somebody else's gesture.
        if (!scrub.stepped && (Math.abs(dy) < SCRUB_ENTER_PX || Math.abs(dy) <= Math.abs(x - scrub.x))) return
        scrub.stepped = true
        const want = scrubIndex(scrub.index, dy, view.count)
        if (want !== scrub.shown) { scrub.shown = want; onPreview(want === view.index ? null : want) }
    }

    const onUp = (e) => {
        const s = scrub
        scrub = null
        if (!s || e.pointerId !== s.id || !s.stepped) return
        onPreview(null)
        if (live() && s.shown != null && s.shown !== view.index) onPick(s.shown)
    }

    const onWheel = (e) => {
        if (!live()) { wheelAcc = 0; return }
        const [x, y] = rel(e)
        if (!inside(x, y)) { wheelAcc = 0; return }
        e.preventDefault()
        wheelAcc += e.deltaY
        if (Math.abs(wheelAcc) < WHEEL_STEP) return
        const dir = wheelAcc > 0 ? -1 : 1 // wheel up selects more, like the drag
        wheelAcc = 0
        const want = stepScope(view.index, dir, view.count)
        if (want !== view.index) onPick(want)
    }

    /** Arrows move along the row, one step, clamped like everything else. */
    const onKey = (e) => {
        if (!live()) return
        const delta = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1
            : (e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0)
        if (!delta) return
        e.preventDefault()
        const want = stepScope(view.index, delta, view.count)
        if (want !== view.index) onPick(want)
    }

    mount.addEventListener('keydown', onKey)
    surface.addEventListener('pointerdown', onDown)
    surface.addEventListener('pointermove', onMove)
    surface.addEventListener('pointerup', onUp)
    surface.addEventListener('pointercancel', onUp)
    surface.addEventListener('wheel', onWheel, { passive: false })

    return {
        update,
        hide,
        place,
        /** True while a scrub owns the pointer, so the host can skip its tap. */
        scrubbing: () => !!(scrub && scrub.stepped),
        destroy() {
            mount.removeEventListener('keydown', onKey)
            surface.removeEventListener('pointerdown', onDown)
            surface.removeEventListener('pointermove', onMove)
            surface.removeEventListener('pointerup', onUp)
            surface.removeEventListener('pointercancel', onUp)
            surface.removeEventListener('wheel', onWheel)
            hide()
        },
    }
}

/**
 * app — SEGLAB interface
 * ------------------------
 * Import a photo → select anything by clicking, dragging a box, drawing a
 * lasso, or describing it in words. All inference on-device (engine-client →
 * worker → YOLOE-26); this module owns only UI state, interaction, overlay
 * rendering, and export.
 *
 * ONE ANALYSIS, MANY SELECTIONS. The model runs once per photo and finds every
 * instance in it. Each mode is then a query over that set, so switching
 * between clicking and describing is instant and they always agree with each
 * other — a click and the phrase for the same object return the same pixels,
 * because they return the same instance.
 *
 * One reference frame: the photo is downscaled once into a ≤1024 canonical
 * canvas (#view). Interactions, masks, overlay, and the cutout all live in
 * that frame — display scaling is pure CSS, undone at the pointer.
 */

import { countMaskComponents } from './mask-core.js'
import { clientState, select, subscribe, warmUp } from './engine-client.js'

const CANON_MAX = 1024
const ACCENT = '#35e0c2'
const POS_COLOR = '#35e08a'
const NEG_COLOR = '#ff5d6c'

const $ = (id) => document.getElementById(id)
const els = {
    main: $('main'), dropzone: $('dropzone'), stage: $('stage'),
    view: $('view'), overlay: $('overlay'), file: $('file'),
    pick: $('pick'), demo: $('demo'), newimg: $('newimg'),
    status: $('status'), loadbar: $('loadbar'),
    chipMode: $('chip-mode'), chipDevice: $('chip-device'), chipTiming: $('chip-timing'),
    undo: $('undo'), reset: $('reset'), cutout: $('cutout'),
    signtoggle: $('signtoggle'),
    textbar: $('textbar'), textq: $('textq'), textgo: $('textgo'),
    instances: $('instances'),
    modes: {
        click: $('mode-click'), box: $('mode-box'),
        lasso: $('mode-lasso'), text: $('mode-text'),
    },
}

/* ─── State ──────────────────────────────────────────────────────────────── */

const state = {
    hasImage: false,
    mode: 'click',            // 'click' | 'box' | 'lasso' | 'text'
    sign: 1,                  // primary-tap label for touch devices
    selected: [],             // instance indices currently selected
    marks: [],                // [[x, y, label]] click markers, for the overlay
    lastRegion: null,         // { kind: 'box'|'lasso', box?, poly? } for the overlay
    query: '',                // active text phrase
    instances: [],            // descriptors of the selected instances
    detected: 0,              // how many the model found in this photo
    mask: null,               // refined ImageData (white-on-black)
    maskRaw: null,            // pre-refinement mask (E toggle)
    showRaw: false,
    maskSummary: null,
    drag: null,
    runSeq: 0,
    running: false,
    runQueued: false,
}

const viewCtx = els.view.getContext('2d')
const overlayCtx = els.overlay.getContext('2d')

/* ─── Status / chips ─────────────────────────────────────────────────────── */

const setStatus = (msg) => { els.status.textContent = msg }

const refreshChips = () => {
    els.chipMode.textContent = `engine: ${clientState.mode || '—'} · yoloe26`
    els.chipDevice.textContent = `device: ${clientState.device || '—'}`
    els.chipDevice.classList.toggle('on', clientState.device === 'webgpu')
    const run = clientState.lastRun
    els.chipTiming.textContent = run
        ? (run.analyzed
            ? `analyze ${run.analyzeMs}ms · ${run.detected} found · post ${run.postMs}ms`
            : `select ${run.selectMs}ms · post ${run.postMs}ms (cached)`)
        : '— ms'
}

subscribe((event) => {
    if (event.type === 'progress') {
        const d = event.detail || {}
        if (d.status === 'progress' && d.total) {
            els.loadbar.style.width = `${Math.round((d.loaded / d.total) * 100)}%`
            setStatus(`Loading the model — ${Math.round((d.loaded / d.total) * 100)}% (one-time)`)
        } else if (d.status === 'done') {
            els.loadbar.style.width = '0%'
        }
        return
    }
    refreshChips()
    if (clientState.ready && !state.running && !state.hasImage) {
        setStatus('Model ready — import a photo to begin')
    }
})

/* ─── Image import ───────────────────────────────────────────────────────── */

const showImage = (bitmapOrCanvas) => {
    const w0 = bitmapOrCanvas.width
    const h0 = bitmapOrCanvas.height
    const scale = Math.min(1, CANON_MAX / Math.max(w0, h0))
    els.view.width = Math.max(1, Math.round(w0 * scale))
    els.view.height = Math.max(1, Math.round(h0 * scale))
    els.overlay.width = els.view.width
    els.overlay.height = els.view.height
    viewCtx.drawImage(bitmapOrCanvas, 0, 0, els.view.width, els.view.height)

    state.hasImage = true
    clearSelection()
    els.dropzone.style.display = 'none'
    els.stage.classList.add('visible')
    setStatus(clientState.ready
        ? 'Ready — click any object, or describe it'
        : 'Preparing the model in the background — you can aim already')
    warmUp().catch((err) => setStatus(`Model load failed: ${err?.message}`))
}

const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    try {
        // from-image: honour EXIF orientation (phone photos).
        const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' })
        showImage(bmp)
        bmp.close()
    } catch (err) {
        setStatus(`Could not read that image: ${err?.message}`)
    }
}

/** Synthetic scene with known answers — instant demo and headless fixture. */
const buildDemoScene = () => {
    const w = 900
    const h = 620
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, '#3c4250')
    grad.addColorStop(1, '#20242d')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = '#d8433b'
    ctx.beginPath()
    ctx.arc(230, 340, 105, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#3b6fd8'
    ctx.beginPath()
    ctx.roundRect(520, 150, 190, 190, 24)
    ctx.fill()
    ctx.fillStyle = '#e8c33b'
    ctx.beginPath()
    ctx.arc(700, 480, 9, 0, Math.PI * 2)
    ctx.fill()
    return c
}

els.pick.addEventListener('click', () => els.file.click())
els.file.addEventListener('change', () => loadFile(els.file.files?.[0]))
els.demo.addEventListener('click', () => showImage(buildDemoScene()))
els.newimg.addEventListener('click', () => {
    state.hasImage = false
    clearSelection()
    els.stage.classList.remove('visible')
    els.dropzone.style.display = ''
    setStatus('Idle — import a photo to begin')
})

window.addEventListener('dragover', (e) => { e.preventDefault(); els.dropzone.classList.add('drag') })
window.addEventListener('dragleave', () => els.dropzone.classList.remove('drag'))
window.addEventListener('drop', (e) => {
    e.preventDefault()
    els.dropzone.classList.remove('drag')
    loadFile(e.dataTransfer?.files?.[0])
})
window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
    if (item) loadFile(item.getAsFile())
})

/* ─── Modes ──────────────────────────────────────────────────────────────── */

const setMode = (mode) => {
    state.mode = mode
    for (const [name, btn] of Object.entries(els.modes)) {
        btn.classList.toggle('active', name === mode)
    }
    els.signtoggle.style.display = mode === 'click' ? '' : 'none'
    els.textbar.classList.toggle('visible', mode === 'text')
    els.overlay.style.cursor = mode === 'text' ? 'default' : 'crosshair'
    if (mode === 'text') els.textq.focus()
}
for (const name of ['click', 'box', 'lasso', 'text']) {
    els.modes[name].addEventListener('click', () => setMode(name))
}

els.signtoggle.addEventListener('click', () => {
    state.sign = state.sign ? 0 : 1
    els.signtoggle.textContent = state.sign ? '＋ include' : '－ exclude'
    els.signtoggle.classList.toggle('pos', !!state.sign)
    els.signtoggle.classList.toggle('neg', !state.sign)
})

/* ─── Selection state ────────────────────────────────────────────────────── */

const refreshButtons = () => {
    const any = state.selected.length > 0 || state.marks.length > 0
    els.undo.disabled = !any
    els.reset.disabled = !any
    els.cutout.disabled = !state.mask
}

function clearSelection() {
    state.selected = []
    state.marks = []
    state.lastRegion = null
    state.query = ''
    state.instances = []
    state.mask = null
    state.maskRaw = null
    state.maskSummary = null
    state.drag = null
    state.runSeq += 1 // orphan any in-flight result
    renderInstances()
    renderOverlay()
    refreshButtons()
}

const undoLast = () => {
    if (state.marks.length > 0) state.marks.pop()
    if (state.selected.length > 0) state.selected.pop()
    if (state.selected.length === 0) {
        clearMask()
        state.lastRegion = null
        state.runSeq += 1
        renderInstances()
        renderOverlay()
        refreshButtons()
        setStatus('Cleared')
        return
    }
    runSelection({ mode: 'indices', indices: state.selected })
}

els.undo.addEventListener('click', undoLast)
els.reset.addEventListener('click', () => { clearSelection(); setStatus('Cleared') })
window.addEventListener('keydown', (e) => {
    if (e.target === els.textq) return
    if (e.key === 'z' || e.key === 'Z') undoLast()
    if (e.key === 'r' || e.key === 'R') { clearSelection(); setStatus('Cleared') }
    if (e.key === 'a' || e.key === 'A') runSelection({ mode: 'all' })
    if (e.key === 'e' || e.key === 'E') {
        state.showRaw = !state.showRaw
        renderOverlay()
        setStatus(state.showRaw ? 'Showing RAW model mask (E to toggle back)' : 'Showing refined mask')
    }
})

/* ─── Pointer handling ───────────────────────────────────────────────────── */

const toCanvas = (e) => {
    const rect = els.overlay.getBoundingClientRect()
    return [
        Math.min(els.overlay.width, Math.max(0, (e.clientX - rect.left) * (els.overlay.width / rect.width))),
        Math.min(els.overlay.height, Math.max(0, (e.clientY - rect.top) * (els.overlay.height / rect.height))),
    ]
}

els.overlay.addEventListener('contextmenu', (e) => e.preventDefault())

els.overlay.addEventListener('pointerdown', (e) => {
    if (!state.hasImage || state.mode === 'text') return
    e.preventDefault()
    els.overlay.setPointerCapture(e.pointerId)
    const [x, y] = toCanvas(e)
    if (state.mode === 'click') {
        state.drag = { kind: 'tap', start: [x, y], moved: false, negative: e.button === 2 || e.altKey }
    } else if (state.mode === 'box') {
        state.drag = { kind: 'box', start: [x, y], now: [x, y] }
    } else if (state.mode === 'lasso') {
        state.drag = { kind: 'lasso', points: [[x, y]] }
    }
    renderOverlay()
})

els.overlay.addEventListener('pointermove', (e) => {
    if (!state.drag) return
    const [x, y] = toCanvas(e)
    if (state.drag.kind === 'tap') {
        if (Math.hypot(x - state.drag.start[0], y - state.drag.start[1]) > 4) state.drag.moved = true
        return
    }
    if (state.drag.kind === 'box') state.drag.now = [x, y]
    else if (state.drag.kind === 'lasso') {
        const pts = state.drag.points
        const [lx, ly] = pts[pts.length - 1]
        if (Math.hypot(x - lx, y - ly) > 3) pts.push([x, y])
    }
    renderOverlay()
})

els.overlay.addEventListener('pointerup', (e) => {
    const drag = state.drag
    state.drag = null
    if (!drag || !state.hasImage) return
    const [x, y] = toCanvas(e)

    if (drag.kind === 'tap' && !drag.moved) {
        const include = !(drag.negative || state.sign === 0)
        state.marks.push([x, y, include ? 1 : 0])
        runSelection({ mode: 'click', point: [x, y], include })
    } else if (drag.kind === 'box') {
        const [sx, sy] = drag.start
        // Discard degenerate boxes (a stray click instead of a drag).
        if (Math.abs(x - sx) > 8 && Math.abs(y - sy) > 8) {
            const box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]
            state.lastRegion = { kind: 'box', box }
            runSelection({ mode: 'box', box })
        }
    } else if (drag.kind === 'lasso' && drag.points.length >= 3) {
        state.lastRegion = { kind: 'lasso', poly: drag.points }
        runSelection({ mode: 'lasso', poly: drag.points })
    }
    renderOverlay()
    refreshButtons()
})

els.textbar.addEventListener('submit', (e) => {
    e.preventDefault()
    const q = els.textq.value.trim()
    if (q) runSelection({ mode: 'text', query: q })
})

/* ─── Selection pipeline ─────────────────────────────────────────────────── */

const sameSet = (a, b) => a.length === b.length && a.every((v, i) => v === b[i])

/**
 * One path for every mode.
 *
 * The interaction resolves to a set of instance indices; that set is the
 * selection. When folding it into what was already selected changes nothing
 * (a first click, a box, a phrase), the result just rendered IS the answer
 * and there is no second call. Only corrective interactions — a negative
 * click, an undo, dropping a chip — need a replay, and that replay is an
 * `indices` query, which never re-runs the model.
 */
async function runSelection(request) {
    if (!state.hasImage) return
    if (state.running) { state.runQueued = request; return }

    const seq = ++state.runSeq
    state.running = true
    setStatus(clientState.ready ? 'Selecting…' : 'Selecting… (first run loads the model)')
    try {
        let res = await select(els.view, request)
        if (seq !== state.runSeq) return
        state.detected = res.detected

        // Fold this interaction into the running selection.
        const before = state.selected
        let next
        if (request.mode === 'click') {
            const hit = res.indices[0]
            if (hit === undefined) {
                setStatus(`Nothing detected there — ${res.detected} objects found in this photo`)
                return
            }
            next = request.include
                ? (before.includes(hit) ? before : [...before, hit])
                : before.filter((i) => i !== hit)
        } else if (request.mode === 'indices') {
            next = res.indices
        } else {
            next = res.indices
        }
        state.selected = next
        state.query = request.mode === 'text' ? request.query : ''

        // Only replay when the folded set differs from what we just rendered.
        if (!sameSet(next, res.indices)) {
            if (next.length === 0) {
                clearMask()
                setStatus('Cleared')
                return
            }
            res = await select(els.view, { mode: 'indices', indices: next })
            if (seq !== state.runSeq) return
        }

        state.mask = res.imageData
        state.maskRaw = res.rawImageData
        state.maskSummary = res.summary
        state.instances = res.instances

        if (!res.instances.length) {
            clearMask()
            setStatus(state.query
                ? `Nothing matched “${state.query}” — ${res.reason || 'not in this photo'}`
                : `Nothing selected — ${res.reason || 'try again'}`)
            return
        }
        const n = res.instances.length
        const what = state.query ? `“${state.query}”` : `${n} ${n === 1 ? 'object' : 'objects'}`
        setStatus(`Selected ${what} — ${((res.summary?.coverage || 0) * 100).toFixed(1)}% of frame · ${res.detected} found in photo${res.analyzed ? '' : ' · cached'}`)
    } catch (err) {
        if (seq !== state.runSeq) return
        console.error('[seglab] selection failed:', err)
        setStatus(`Selection failed: ${err?.message}`)
    } finally {
        state.running = false
        renderInstances()
        renderOverlay()
        refreshButtons()
        const queued = state.runQueued
        state.runQueued = null
        if (queued) runSelection(queued)
    }
}

const clearMask = () => {
    state.mask = null
    state.maskRaw = null
    state.maskSummary = null
    state.instances = []
}

/* ─── Instance chips ─────────────────────────────────────────────────────── */

function renderInstances() {
    els.instances.innerHTML = ''
    els.instances.classList.toggle('visible', state.instances.length > 1)
    if (state.instances.length <= 1) return
    state.instances.forEach((inst, idx) => {
        const chip = document.createElement('span')
        chip.className = 'inst on'
        chip.textContent = `${inst.label} ${(inst.score * 100).toFixed(0)}%`
        chip.title = 'Click to drop this object from the selection'
        chip.addEventListener('click', () => {
            const next = state.selected.filter((_, i) => i !== idx)
            runSelection({ mode: 'indices', indices: next })
        })
        els.instances.appendChild(chip)
    })
}

/* ─── Overlay rendering ──────────────────────────────────────────────────── */

/** Colorize the white-on-black mask; returns {fill, ring} canvases. */
const buildMaskLayers = (mask) => {
    const { width, height } = mask
    const raw = new OffscreenCanvas(width, height)
    raw.getContext('2d').putImageData(mask, 0, 0)

    const alpha = new OffscreenCanvas(width, height)
    const alphaCtx = alpha.getContext('2d', { willReadFrequently: true })
    alphaCtx.drawImage(raw, 0, 0)
    // Convert luma → alpha and tint in one pixel pass.
    const img = alphaCtx.getImageData(0, 0, width, height)
    const d = img.data
    for (let i = 0; i < d.length; i += 4) {
        d[i + 3] = d[i]
        d[i] = 53; d[i + 1] = 224; d[i + 2] = 194
    }
    alphaCtx.putImageData(img, 0, 0)

    // Ring = 8-direction dilate of the alpha mask minus the mask itself.
    const ring = new OffscreenCanvas(width, height)
    const ringCtx = ring.getContext('2d')
    const r = Math.max(1.25, Math.min(width, height) / 480)
    for (const [dx, dy] of [[-r, 0], [r, 0], [0, -r], [0, r], [-r, -r], [r, -r], [-r, r], [r, r]]) {
        ringCtx.drawImage(alpha, dx, dy)
    }
    ringCtx.globalCompositeOperation = 'destination-out'
    ringCtx.drawImage(alpha, 0, 0)
    return { fill: alpha, ring }
}

function renderOverlay() {
    const ctx = overlayCtx
    const { width, height } = els.overlay
    ctx.clearRect(0, 0, width, height)

    const shownMask = state.showRaw ? state.maskRaw : state.mask
    if (shownMask) {
        const { fill, ring } = buildMaskLayers(shownMask)
        ctx.globalAlpha = 0.32
        ctx.drawImage(fill, 0, 0)
        ctx.globalAlpha = 0.95
        ctx.drawImage(ring, 0, 0)
        ctx.globalAlpha = 1
    }

    const markerR = Math.max(4, Math.min(width, height) * 0.009)

    if (state.lastRegion?.kind === 'box') {
        const b = state.lastRegion.box
        ctx.setLineDash([7, 5])
        ctx.strokeStyle = ACCENT
        ctx.lineWidth = 1.75
        ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1])
        ctx.setLineDash([])
    }
    if (state.lastRegion?.kind === 'lasso') {
        const poly = state.lastRegion.poly
        ctx.beginPath()
        ctx.moveTo(poly[0][0], poly[0][1])
        for (const [px, py] of poly.slice(1)) ctx.lineTo(px, py)
        ctx.closePath()
        ctx.strokeStyle = state.mask ? 'rgba(53,224,194,0.25)' : 'rgba(90,160,255,0.9)'
        ctx.lineWidth = 2.5
        ctx.stroke()
    }

    if (state.drag?.kind === 'box') {
        const [sx, sy] = state.drag.start
        const [nx, ny] = state.drag.now
        ctx.setLineDash([7, 5])
        ctx.strokeStyle = ACCENT
        ctx.lineWidth = 1.75
        ctx.strokeRect(Math.min(sx, nx), Math.min(sy, ny), Math.abs(nx - sx), Math.abs(ny - sy))
        ctx.setLineDash([])
    }
    if (state.drag?.kind === 'lasso' && state.drag.points.length > 1) {
        ctx.beginPath()
        ctx.moveTo(state.drag.points[0][0], state.drag.points[0][1])
        for (const [px, py] of state.drag.points.slice(1)) ctx.lineTo(px, py)
        ctx.strokeStyle = 'rgba(90,160,255,0.95)'
        ctx.lineWidth = 3
        ctx.lineJoin = 'round'
        ctx.lineCap = 'round'
        ctx.stroke()
        ctx.fillStyle = 'rgba(90,160,255,0.12)'
        ctx.fill()
    }

    for (const [x, y, label] of state.marks) {
        ctx.beginPath()
        ctx.arc(x, y, markerR, 0, Math.PI * 2)
        ctx.fillStyle = label ? POS_COLOR : NEG_COLOR
        ctx.fill()
        ctx.lineWidth = 2
        ctx.strokeStyle = 'rgba(255,255,255,0.9)'
        ctx.stroke()
    }
}

/* ─── Cutout export ──────────────────────────────────────────────────────── */

els.cutout.addEventListener('click', async () => {
    if (!state.mask) return
    const c = document.createElement('canvas')
    c.width = els.view.width
    c.height = els.view.height
    const ctx = c.getContext('2d')
    ctx.drawImage(els.view, 0, 0)
    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = c.width
    maskCanvas.height = c.height
    // Mask luma → alpha for destination-in.
    const md = new ImageData(new Uint8ClampedArray(state.mask.data), c.width, c.height)
    for (let i = 0; i < md.data.length; i += 4) md.data[i + 3] = md.data[i]
    maskCanvas.getContext('2d').putImageData(md, 0, 0)
    ctx.globalCompositeOperation = 'destination-in'
    ctx.drawImage(maskCanvas, 0, 0)
    const blob = await new Promise((res) => c.toBlob(res, 'image/png'))
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'seglab-cutout.png'
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 5000)
})

/* ─── Headless test hooks (verify.mjs, bench.mjs) ────────────────────────── */

const waitForRun = async () => {
    await new Promise((res) => setTimeout(res, 60))
    while (state.running || state.runQueued) {
        await new Promise((res) => setTimeout(res, 40))
    }
}

window.__seglab = {
    loadDemo: () => { showImage(buildDemoScene()) },
    reset: () => clearSelection(),
    warm: () => warmUp(),
    state: () => ({
        ready: clientState.ready,
        device: clientState.device,
        mode: clientState.mode,
        lastRun: clientState.lastRun,
        maskSummary: state.maskSummary,
        detected: state.detected,
        selected: state.selected.length,
        query: state.query,
        instances: state.instances,
    }),
    maskStats: () => {
        if (!state.mask) return null
        const { data, width, height } = state.mask
        let soft = 0
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 16 && data[i] < 240) soft += 1
        }
        return { components: countMaskComponents(data, width, height), softPixels: soft }
    },
    /** Load an image from a data/blob URL — how bench.mjs feeds in DSLR files. */
    loadURL: async (url) => {
        const blob = await (await fetch(url)).blob()
        const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' })
        showImage(bmp)
        const dims = { width: bmp.width, height: bmp.height, canonW: els.view.width, canonH: els.view.height }
        bmp.close()
        return dims
    },
    /** Synthetic scene at an arbitrary (DSLR) resolution. */
    loadSynthetic: ({ w = 6000, h = 4000, objects = [] } = {}) => {
        const c = document.createElement('canvas')
        c.width = w
        c.height = h
        const ctx = c.getContext('2d')
        const grad = ctx.createLinearGradient(0, 0, 0, h)
        grad.addColorStop(0, '#8fa4bd')
        grad.addColorStop(1, '#3b4654')
        ctx.fillStyle = grad
        ctx.fillRect(0, 0, w, h)
        for (const o of objects) {
            ctx.fillStyle = o.color
            ctx.beginPath()
            if (o.kind === 'circle') ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2)
            else ctx.rect(o.x - o.r, o.y - o.r, o.r * 2, o.r * 2)
            ctx.fill()
        }
        showImage(c)
        return { width: w, height: h, canonW: els.view.width, canonH: els.view.height }
    },
    clickAt: async (x, y, negative = false) => {
        setMode('click')
        state.marks.push([x, y, negative ? 0 : 1])
        runSelection({ mode: 'click', point: [x, y], include: !negative })
        await waitForRun()
        return window.__seglab.state()
    },
    boxAt: async (x0, y0, x1, y1) => {
        setMode('box')
        const box = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]
        state.lastRegion = { kind: 'box', box }
        runSelection({ mode: 'box', box })
        await waitForRun()
        return window.__seglab.state()
    },
    lassoCircle: async (cx, cy, r, n = 28) => {
        setMode('lasso')
        const poly = []
        for (let i = 0; i < n; i += 1) {
            const a = (i / n) * Math.PI * 2
            poly.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r])
        }
        state.lastRegion = { kind: 'lasso', poly }
        runSelection({ mode: 'lasso', poly })
        await waitForRun()
        return window.__seglab.state()
    },
    textSearch: async (query) => {
        setMode('text')
        runSelection({ mode: 'text', query })
        await waitForRun()
        return window.__seglab.state()
    },
    selectAll: async () => {
        runSelection({ mode: 'all' })
        await waitForRun()
        return window.__seglab.state()
    },
}
window.__seglabReady = true

/* ─── Boot ───────────────────────────────────────────────────────────────── */

setMode('click')
refreshChips()
console.log('[seglab] ready')

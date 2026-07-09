/**
 * app — SEGLAB interface
 * ------------------------
 * Import a photo → select anything with clicks (+/−), a box, or a rough
 * lasso that snaps to the object. All inference on-device (sam-client →
 * worker → SlimSAM); this module owns only UI state, prompt collection,
 * overlay rendering, and the lasso clamp.
 *
 * One reference frame: the photo is downscaled once into a ≤1024 canonical
 * canvas (#view). Prompts, masks, overlay, and the cutout all live in that
 * frame — display scaling is pure CSS, undone at the pointer.
 */

import { countMaskComponents, lassoToPrompts } from './sam-core.js'
import { clientState, segment, subscribe, warmUp } from './sam-client.js'

// ?flagship=0 keeps the session on the 14 MB draft lane (tests, data saver).
const FLAGSHIP_ENABLED = new URLSearchParams(location.search).get('flagship') !== '0'

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
    modes: {
        click: $('mode-click'), box: $('mode-box'),
        lasso: $('mode-lasso'), text: $('mode-text'),
    },
}

/* ─── State ──────────────────────────────────────────────────────────────── */

const state = {
    hasImage: false,
    mode: 'click',            // 'click' | 'box' | 'lasso'
    sign: 1,                  // primary-tap label for touch devices
    clicks: [],               // [[x, y, label], ...] canonical coords
    box: null,                // [x0, y0, x1, y1] canonical
    lasso: null,              // { poly, box, point, margin } from lassoToPrompts
    mask: null,               // refined ImageData (white-on-black), canonical size
    maskRaw: null,            // decoder output before hygiene/refinement (E toggle)
    showRaw: false,
    maskSummary: null,
    score: 0,
    drag: null,               // in-progress interaction {kind, points|start}
    runSeq: 0,                // stale-result guard
    running: false,
    runQueued: false,
}

const viewCtx = els.view.getContext('2d')
const overlayCtx = els.overlay.getContext('2d')

/* ─── Status / chips ─────────────────────────────────────────────────────── */

const setStatus = (msg) => { els.status.textContent = msg }

const refreshChips = () => {
    const lane = clientState.lane ? ` · ${clientState.lane}` : ''
    els.chipMode.textContent = `engine: ${clientState.mode || '—'}${lane}`
    els.chipDevice.textContent = `device: ${clientState.device || '—'}`
    els.chipDevice.classList.toggle('on', clientState.device === 'webgpu')
    const run = clientState.lastRun
    els.chipTiming.textContent = run
        ? (run.encoded
            ? `encode ${run.encodeMs}ms · decode ${run.decodeMs}ms · post ${run.postMs}ms`
            : `decode ${run.decodeMs}ms · post ${run.postMs}ms (cached)`)
        : '— ms'
}

subscribe((event) => {
    if (event.type === 'progress') {
        const d = event.detail || {}
        if (d.status === 'progress' && d.total) {
            const pct = Math.round((d.loaded / d.total) * 100)
            els.loadbar.style.width = `${pct}%`
            // Flagship downloads in the background — say so without hiding
            // that the tool is already usable on the draft lane.
            if (d.lane === 'flagship') {
                if (!state.running) setStatus(`Ready on SlimSAM — upgrading to SAM3 in the background, ${pct}% of ~300 MB (one-time)`)
            } else {
                setStatus(`Downloading model — ${String(d.file || '').split('/').pop()} ${pct}% (one-time, ~14 MB)`)
            }
        } else if (d.status === 'done') {
            els.loadbar.style.width = '0%'
        }
        return
    }
    if (event.type === 'lane') {
        refreshChips()
        if (event.label === 'sam3') {
            setStatus('Upgraded to SAM3 — masks are sharper from here on')
            // Prompt replay: silently re-run the current selection at
            // flagship quality (first replay re-encodes the image).
            if (state.hasImage && (state.clicks.length || state.box || state.lasso)) scheduleRun()
        } else {
            setStatus('SAM3 unavailable — continuing on SlimSAM')
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
    clearPrompts()
    els.dropzone.style.display = 'none'
    els.stage.classList.add('visible')
    setStatus(clientState.ready
        ? 'Ready — click any object'
        : 'Preparing the model in the background — you can aim already')
    // Start the one-time model download NOW, while the user is aiming.
    warmUp({ flagship: FLAGSHIP_ENABLED }).catch((err) => setStatus(`Model load failed: ${err?.message}`))
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

/** Synthetic scene with known answers — instant demo, and what the headless
 *  verify drives: big disc, rounded square, and a MINUTE dot (r=9). */
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
    clearPrompts()
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
}
els.modes.click.addEventListener('click', () => setMode('click'))
els.modes.box.addEventListener('click', () => setMode('box'))
els.modes.lasso.addEventListener('click', () => setMode('lasso'))

els.signtoggle.addEventListener('click', () => {
    state.sign = state.sign ? 0 : 1
    els.signtoggle.textContent = state.sign ? '＋ include' : '－ exclude'
    els.signtoggle.classList.toggle('pos', !!state.sign)
    els.signtoggle.classList.toggle('neg', !state.sign)
})

/* ─── Prompt state ───────────────────────────────────────────────────────── */

const refreshButtons = () => {
    const any = state.clicks.length > 0 || state.box || state.lasso
    els.undo.disabled = !any
    els.reset.disabled = !any
    els.cutout.disabled = !state.mask
}

function clearPrompts() {
    state.clicks = []
    state.box = null
    state.lasso = null
    state.mask = null
    state.maskRaw = null
    state.maskSummary = null
    state.score = 0
    state.drag = null
    state.runSeq += 1 // orphan any in-flight result
    renderOverlay()
    refreshButtons()
}

const undoPrompt = () => {
    if (state.clicks.length > 0) state.clicks.pop()
    else if (state.box) state.box = null
    else if (state.lasso) state.lasso = null
    if (state.clicks.length === 0 && !state.box && !state.lasso) {
        state.mask = null
        state.maskRaw = null
        state.maskSummary = null
        state.runSeq += 1
        renderOverlay()
        refreshButtons()
        setStatus('Cleared')
        return
    }
    scheduleRun()
}

els.undo.addEventListener('click', undoPrompt)
els.reset.addEventListener('click', () => { clearPrompts(); setStatus('Cleared') })
window.addEventListener('keydown', (e) => {
    if (e.key === 'z' || e.key === 'Z') undoPrompt()
    if (e.key === 'r' || e.key === 'R') { clearPrompts(); setStatus('Cleared') }
    if (e.key === 'e' || e.key === 'E') {
        state.showRaw = !state.showRaw
        renderOverlay()
        setStatus(state.showRaw ? 'Showing RAW decoder mask (E to toggle back)' : 'Showing refined mask')
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
    if (!state.hasImage) return
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
    if (state.drag.kind === 'box') {
        state.drag.now = [x, y]
    } else if (state.drag.kind === 'lasso') {
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
        const label = drag.negative || state.sign === 0 ? 0 : 1
        state.clicks.push([x, y, label])
        scheduleRun()
    } else if (drag.kind === 'box') {
        const [sx, sy] = drag.start
        // Discard degenerate boxes (a stray click instead of a drag).
        if (Math.abs(x - sx) > 8 && Math.abs(y - sy) > 8) {
            state.box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]
            state.lasso = null // a box replaces a lasso region
            scheduleRun()
        }
    } else if (drag.kind === 'lasso') {
        const prompts = lassoToPrompts(drag.points)
        if (prompts) {
            // A fresh lasso is a fresh selection: it replaces earlier
            // prompts (the reference interaction), and later clicks refine
            // INSIDE it (the clamp keeps everything within lasso ∪ margin).
            state.lasso = { poly: drag.points, ...prompts }
            state.clicks = []
            state.box = null
            scheduleRun()
        }
    }
    renderOverlay()
    refreshButtons()
})

/* ─── Segmentation pipeline ──────────────────────────────────────────────── */

let debounceTimer = null
const scheduleRun = () => {
    clearTimeout(debounceTimer)
    debounceTimer = setTimeout(runNow, 80)
}

async function runNow() {
    if (!state.hasImage) return
    if (state.running) { state.runQueued = true; return }
    const clicks = state.lasso ? [state.lasso.point, ...state.clicks] : state.clicks
    const box = state.lasso ? state.lasso.box : state.box
    if (clicks.length === 0 && !box) return

    const seq = ++state.runSeq
    state.running = true
    setStatus(clientState.ready ? 'Selecting…' : 'Selecting… (first run loads the model)')
    try {
        // The engine runs the whole post pipeline (lasso clamp → hygiene →
        // edge refinement) off-thread and returns both masks.
        const res = await segment(els.view, {
            clicks,
            box,
            clampPoly: state.lasso?.poly || null,
            clampMargin: state.lasso?.margin || 0,
        })
        if (seq !== state.runSeq) return // superseded — a newer prompt owns the state

        if (!res.usable) {
            state.mask = null
            state.maskRaw = null
            state.maskSummary = null
            setStatus(`No selection — ${res.reason}. Try another click.`)
        } else {
            state.mask = res.imageData
            state.maskRaw = res.rawImageData
            state.maskSummary = res.summary
            state.score = res.score
            setStatus(`Selected — ${res.lane} · confidence ${res.score.toFixed(2)} · ${(res.summary.coverage * 100).toFixed(1)}% of frame${res.encoded ? '' : ' · cached'}`)
        }
        renderOverlay()
        refreshButtons()
    } catch (err) {
        if (seq !== state.runSeq) return
        console.error('[seglab] selection failed:', err)
        setStatus(`Selection failed: ${err?.message}`)
    } finally {
        state.running = false
        if (state.runQueued) { state.runQueued = false; scheduleRun() }
    }
}

/* ─── Overlay rendering ──────────────────────────────────────────────────── */

/** Colorize the white-on-black mask; returns {fill, ring} canvases. */
const buildMaskLayers = (mask) => {
    const { width, height } = mask
    const raw = new OffscreenCanvas(width, height)
    const rawCtx = raw.getContext('2d')
    rawCtx.putImageData(mask, 0, 0)

    // Only the white pixels, as accent color (black pixels are opaque in the
    // ImageData, so tint via source-in on a luma→alpha copy).
    const alpha = new OffscreenCanvas(width, height)
    const alphaCtx = alpha.getContext('2d', { willReadFrequently: true })
    alphaCtx.drawImage(raw, 0, 0)
    // Convert luma → alpha and tint in one pixel pass.
    const img = alphaCtx.getImageData(0, 0, width, height)
    const d = img.data
    for (let i = 0; i < d.length; i += 4) {
        d[i + 3] = d[i] // alpha = luma
        d[i] = 53; d[i + 1] = 224; d[i + 2] = 194 // accent RGB
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

    // Persisted box (dashed).
    if (state.box) {
        ctx.setLineDash([7, 5])
        ctx.strokeStyle = ACCENT
        ctx.lineWidth = 1.75
        ctx.strokeRect(state.box[0], state.box[1], state.box[2] - state.box[0], state.box[3] - state.box[1])
        ctx.setLineDash([])
    }

    // Lasso region (kept faint once the mask lands, so the clamp is visible).
    if (state.lasso) {
        ctx.beginPath()
        ctx.moveTo(state.lasso.poly[0][0], state.lasso.poly[0][1])
        for (const [px, py] of state.lasso.poly.slice(1)) ctx.lineTo(px, py)
        ctx.closePath()
        ctx.strokeStyle = state.mask ? 'rgba(53,224,194,0.25)' : 'rgba(90,160,255,0.9)'
        ctx.lineWidth = 2.5
        ctx.stroke()
    }

    // In-progress drags.
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

    // Click markers.
    for (const [x, y, label] of state.clicks) {
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

/* ─── Headless test hooks (verify.mjs) ───────────────────────────────────── */

const waitForRun = async () => {
    // Wait out the debounce + the run; poll because runs can chain.
    await new Promise((res) => setTimeout(res, 120))
    while (state.running || state.runQueued) {
        await new Promise((res) => setTimeout(res, 40))
    }
}

window.__seglab = {
    loadDemo: () => { showImage(buildDemoScene()) },
    reset: () => clearPrompts(),
    state: () => ({
        ready: clientState.ready,
        device: clientState.device,
        mode: clientState.mode,
        lane: clientState.lane,
        lastRun: clientState.lastRun,
        maskSummary: state.maskSummary,
        score: state.score,
        clicks: state.clicks.length,
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
    clickAt: async (x, y, negative = false) => {
        state.clicks.push([x, y, negative ? 0 : 1])
        scheduleRun()
        await waitForRun()
        return window.__seglab.state()
    },
    lassoCircle: async (cx, cy, r, n = 28) => {
        const poly = []
        for (let i = 0; i < n; i += 1) {
            const a = (i / n) * Math.PI * 2
            poly.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r])
        }
        const prompts = lassoToPrompts(poly)
        if (!prompts) throw new Error('degenerate test lasso')
        state.lasso = { poly, ...prompts }
        state.clicks = []
        state.box = null
        scheduleRun()
        await waitForRun()
        return window.__seglab.state()
    },
}
window.__seglabReady = true

/* ─── Boot ───────────────────────────────────────────────────────────────── */

setMode('click')
refreshChips()
console.log('[seglab] ready')

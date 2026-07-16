/**
 * app — SEGLAB interface
 * ------------------------
 * Import a photo → select anything with clicks (+/−), a box, or a rough
 * lasso that snaps to the object. All inference on-device (sam-client →
 * worker → SlimSAM); this module owns only UI state, prompt collection,
 * overlay rendering, and the lasso clamp.
 *
 * One reference frame: the photo is downscaled once into a bounded canonical
 * canvas (#view, ≤ policy.proxyMax — 768 px on the lite profile). Prompts,
 * masks, overlay, and the cutout all live in that frame — display scaling is
 * pure CSS, undone at the pointer.
 *
 * Memory safety: the device policy (js/policy.js) is resolved once at boot
 * and rules every limit. On ≤8 GB or unknown-memory devices the lite profile
 * is forced and URL flags cannot raise any cap. SlimSAM is the only
 * automatic model; SAM3 exists solely behind an explicit opt-in.
 */

import { alphaToImageData, countMaskComponents, lassoToPrompts } from './sam-core.js'
import { clientState, getEngineSnapshot, resetImage, segment, startFlagship, subscribe, warmUp } from './sam-client.js'
import { exportCutout } from './export-hd.js'
import { resolvePolicy, effectivePolicy } from './policy.js'
import { detectCapability, getPressureLevel, isMemoryError, onPressureChange, reportPressure, startPressureMonitor } from './capability.js'
import { PRIORITY, cancelHeavyBefore, clearHeavyQueue, enqueueHeavy, getHeavyQueueState } from './heavy-job-queue.js'
import { getBoundedProxySize } from './image-io.js'
import { decodeProxy } from './decode-client.js'
import { disposeCv, ensureGuide, isCvUsable, refine as cvRefine, resetCv } from './cv-refine-client.js'

const capability = detectCapability()
const policy = resolvePolicy({ capability, search: location.search })
/** Current limits with memory-pressure reductions applied. */
const eff = () => effectivePolicy(policy, getPressureLevel())

/** Structured memory telemetry — dev mode (?debug=1) only. */
const memlog = (event, extra = {}) => {
    if (!policy.debug) return
    const entry = {
        event,
        pressureLevel: getPressureLevel(),
        queueActive: getHeavyQueueState().activeLabel,
        modelLane: clientState.lane || 'slimsam',
        ...extra,
    }
    ;(window.__seglabMemLog ??= []).push(entry)
    console.log('[seglab][memory]', JSON.stringify(entry))
}
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
    mask: null,               // refined { alpha: Uint8Array, width, height }
    maskRaw: null,            // decoder output before hygiene/refinement (E toggle)
    showRaw: false,
    maskSummary: null,
    lastRefiner: null,        // 'wasm' | 'js' (cv worker) | 'engine-js'
    score: 0,
    drag: null,               // in-progress interaction {kind, points|start}
    runSeq: 0,                // stale-result guard
    running: false,
    runQueued: false,
}

const viewCtx = els.view.getContext('2d')
const overlayCtx = els.overlay.getContext('2d')

/* ─── Document custody ───────────────────────────────────────────────────── */
// The original image exists ONLY as this compressed Blob. Every import bumps
// `revision`; stale heavy jobs and worker replies are dropped against it.
const doc = { revision: 0, blob: null, meta: null }

const beginDocument = (blob) => {
    doc.revision += 1
    doc.blob = blob
    doc.meta = null
    cancelHeavyBefore(doc.revision)
    resetImage(doc.revision) // engine frees its embedding slot
    resetCv(doc.revision) // refine worker drops its stale guide
    return doc.revision
}

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
            // Flagship progress only ever appears after the explicit opt-in.
            if (d.lane === 'flagship') {
                if (!state.running) setStatus(`Downloading SAM3 (your opt-in) — ${pct}% of ~300 MB (one-time)`)
            } else {
                setStatus(`Downloading model — ${String(d.file || '').split('/').pop()} ${pct}% (one-time)`)
            }
        } else if (d.status === 'done') {
            els.loadbar.style.width = '0%'
        }
        return
    }
    if (event.type === 'lane') {
        refreshChips()
        if (event.label === 'sam3') {
            setStatus('SAM3 enabled — masks are sharper from here on')
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

const revealEditor = () => {
    state.hasImage = true
    clearPrompts()
    els.dropzone.style.display = 'none'
    els.stage.classList.add('visible')
    setStatus(clientState.ready
        ? 'Ready — click any object'
        : 'Preparing the model in the background — you can aim already')
    // Warm strictly AFTER the proxy is visible — decode and model work
    // serialize through the heavy queue, never overlapping.
    scheduleWarm()
}

/** In-memory canvas sources (demo scene) — no decode, same size bound. */
const showImage = (canvas) => {
    beginDocument(null)
    const { width, height } = getBoundedProxySize(canvas.width, canvas.height, eff().proxyMax)
    els.view.width = width
    els.view.height = height
    els.overlay.width = width
    els.overlay.height = height
    viewCtx.drawImage(canvas, 0, 0, width, height)
    doc.meta = {
        original: { width: canvas.width, height: canvas.height, orientation: 1, format: 'canvas' },
        proxy: { width, height, scale: width / canvas.width },
    }
    revealEditor()
}

/** Bounded bitmap from the decode worker — drawn once, closed immediately. */
const showProxy = (bitmap) => {
    try {
        els.view.width = bitmap.width
        els.view.height = bitmap.height
        els.overlay.width = bitmap.width
        els.overlay.height = bitmap.height
        viewCtx.drawImage(bitmap, 0, 0)
    } finally {
        try { bitmap.close() } catch { /* already closed */ }
    }
    revealEditor()
}

let warmQueued = false
/** Warm the draft (SlimSAM) lane only — never the flagship. Queued so it
 *  can never overlap another heavy job (e.g. an import decode). */
const scheduleWarm = () => {
    if (warmQueued) return
    warmQueued = true
    enqueueHeavy('model-warm', () => warmUp(), { priority: PRIORITY.model })
        .then(() => memlog('warm-done'))
        .catch((err) => {
            warmQueued = false
            if (!err?.stale) setStatus(`Model load failed: ${err?.message}`)
        })
}

// RAW containers carry a decodable embedded JPEG preview (best-effort lane).
const RAW_EXTENSIONS = /\.(cr2|cr3|nef|arw|dng|raf|orf|rw2|srw|pef)$/i

const loadFile = async (file) => {
    if (!file) return
    if (!file.type?.startsWith('image/') && !RAW_EXTENSIONS.test(file.name || '')) return
    const revision = beginDocument(file)
    setStatus('Reading image…')
    try {
        const res = await enqueueHeavy(
            'proxy-decode',
            () => decodeProxy(file, eff().proxyMax, revision),
            { priority: PRIORITY.import, revision, isCurrent: () => revision === doc.revision },
        )
        if (revision !== doc.revision) {
            try { res?.bitmap?.close?.() } catch { /* already closed */ }
            return
        }
        doc.meta = { original: res.original, proxy: res.proxy }
        memlog('proxy-decoded', {
            original: `${res.original.width}x${res.original.height}`,
            proxy: `${res.proxy.width}x${res.proxy.height}`,
            format: res.original.format,
        })
        showProxy(res.bitmap)
    } catch (err) {
        if (err?.stale || revision !== doc.revision) return
        if (isMemoryError(err)) reportPressure(1, 'import decode allocation failure')
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
    beginDocument(null)
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
    const docRev = doc.revision
    state.running = true
    setStatus(clientState.ready ? 'Selecting…' : 'Selecting… (first run loads the model)')
    try {
        // One heavy job covers decode + refinement: SlimSAM decodes (clamp
        // only), then the cv worker (wasm, JS fallback) runs hygiene + edge
        // refinement. If the cv lane is gone, the engine's own JS pipeline
        // serves — a lost refiner never costs a selection.
        const prompts = {
            clicks,
            box,
            clampPoly: state.lasso?.poly || null,
            clampMargin: state.lasso?.margin || 0,
        }
        const res = await enqueueHeavy('prompt-decode', async () => {
            const useCv = eff().wasmRefine && isCvUsable()
            const seg = await segment(els.view, prompts, { revision: docRev, post: useCv ? 'clamp-only' : 'js' })
            if (!useCv) return { ...seg, refiner: 'engine-js' }
            try {
                await ensureGuide(docRev, els.view)
                const seeds = clicks.filter((c) => c[2]).map((c) => [c[0], c[1]])
                const refined = await cvRefine({
                    revision: docRev,
                    width: seg.width,
                    height: seg.height,
                    alpha: seg.alpha,
                    alphaRaw: seg.alphaRaw,
                    seeds,
                })
                return { ...seg, ...refined }
            } catch (err) {
                if (err?.stale) throw err
                console.warn('[seglab] cv refinement unavailable; engine pipeline serves:', err?.message)
                const retry = await segment(els.view, prompts, { revision: docRev, post: 'js' })
                return { ...retry, refiner: 'engine-js' }
            }
        }, {
            priority: PRIORITY.interaction,
            revision: docRev,
            isCurrent: () => seq === state.runSeq && docRev === doc.revision,
        })
        if (seq !== state.runSeq || docRev !== doc.revision) return // superseded

        if (!res.usable) {
            state.mask = null
            state.maskRaw = null
            state.maskSummary = null
            setStatus(`No selection — ${res.reason}. Try another click.`)
        } else {
            state.mask = { alpha: res.alpha, width: res.width, height: res.height }
            state.maskRaw = { alpha: res.alphaRaw, width: res.width, height: res.height }
            state.maskSummary = res.summary
            state.lastRefiner = res.refiner || null
            state.score = res.score
            setStatus(`Selected — ${res.lane} · confidence ${res.score.toFixed(2)} · ${(res.summary.coverage * 100).toFixed(1)}% of frame${res.encoded ? '' : ' · cached'}`)
            memlog('selection-done', {
                encoded: res.encoded,
                encodeMs: res.encodeMs,
                decodeMs: res.decodeMs,
                refiner: res.refiner || null,
                ms: res.ms,
            })
        }
        renderOverlay()
        refreshButtons()
    } catch (err) {
        if (seq !== state.runSeq || err?.stale) return
        console.error('[seglab] selection failed:', err)
        if (isMemoryError(err)) reportPressure(2, 'selection allocation failure')
        setStatus(`Selection failed: ${err?.message}`)
    } finally {
        state.running = false
        // Pressure L3: hold no embedding between interactions.
        if (eff().clearEmbeddingAfterRun) resetImage(doc.revision)
        if (state.runQueued) { state.runQueued = false; scheduleRun() }
    }
}

/* ─── Overlay rendering ──────────────────────────────────────────────────── */

/** Colorize the one-channel mask; returns {fill, ring} canvases. */
const buildMaskLayers = (mask) => {
    const { width, height } = mask
    const fill = new OffscreenCanvas(width, height)
    fill.getContext('2d').putImageData(alphaToImageData(mask.alpha, width, height, [53, 224, 194]), 0, 0)

    // Ring = 8-direction dilate of the fill minus the fill itself.
    const ring = new OffscreenCanvas(width, height)
    const ringCtx = ring.getContext('2d')
    const r = Math.max(1.25, Math.min(width, height) / 480)
    for (const [dx, dy] of [[-r, 0], [r, 0], [0, -r], [0, r], [-r, -r], [r, -r], [-r, r], [r, r]]) {
        ringCtx.drawImage(fill, dx, dy)
    }
    ringCtx.globalCompositeOperation = 'destination-out'
    ringCtx.drawImage(fill, 0, 0)
    return { fill, ring }
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

/** Explicit export: bounded original re-decode (≤4096px/8MP), mask upscaled. */
const runExport = async () => {
    if (!state.mask) return null
    const t0 = Date.now()
    const res = await exportCutout({
        doc,
        mask: { alpha: state.mask.alpha.slice(), width: state.mask.width, height: state.mask.height },
        policy: eff(),
        sourceCanvas: els.view,
    })
    memlog('export-done', { out: `${res.width}x${res.height}`, reduced: res.reduced, durMs: Date.now() - t0 })
    return res
}

els.cutout.addEventListener('click', async () => {
    if (!state.mask) return
    els.cutout.disabled = true
    setStatus('Exporting cutout…')
    try {
        const res = await runExport()
        if (!res) return
        const a = document.createElement('a')
        a.href = URL.createObjectURL(res.blob)
        a.download = 'seglab-cutout.png'
        a.click()
        setTimeout(() => URL.revokeObjectURL(a.href), 5000)
        setStatus(res.reduced
            ? `Exported ${res.width}×${res.height} — reduced to fit this device's safe memory profile`
            : `Exported ${res.width}×${res.height}`)
    } catch (err) {
        if (!err?.stale) setStatus(`Export failed: ${err?.message}`)
    } finally {
        refreshButtons()
    }
})

/* ─── SAM3 opt-in (explicit, confirmed, never automatic) ─────────────────── */

const els_sam3 = $('sam3optin')
if (policy.allowFlagshipOptIn && typeof navigator !== 'undefined' && navigator.gpu) {
    els_sam3.hidden = false
    els_sam3.addEventListener('click', async () => {
        if (!eff().allowFlagshipOptIn) { setStatus('SAM3 opt-in unavailable under memory pressure'); return }
        const ok = window.confirm(
            'Enable SAM3 (optional flagship model)?\n\n'
            + '• ~300 MB one-time download + WebGPU compile\n'
            + '• Needs more memory than the 8 GB safe profile guarantees\n'
            + '• SlimSAM keeps working either way',
        )
        if (!ok) return
        els_sam3.disabled = true
        setStatus('Loading SAM3 — your explicit opt-in…')
        try {
            const engine = await enqueueHeavy('flagship-warm', () => startFlagship(), { priority: PRIORITY.model })
            if (engine?.flagship === 'ready') {
                els_sam3.textContent = 'SAM3 ✓'
            } else {
                setStatus('SAM3 unavailable on this device — continuing on SlimSAM')
                els_sam3.disabled = false
            }
        } catch (err) {
            setStatus(`SAM3 load failed: ${err?.message}`)
            els_sam3.disabled = false
        }
    })
}

/* ─── Headless test hooks (verify.mjs) ───────────────────────────────────── */

const waitForRun = async () => {
    // Wait out the debounce + the run; poll because runs can chain.
    await new Promise((res) => setTimeout(res, 120))
    while (state.running || state.runQueued) {
        await new Promise((res) => setTimeout(res, 40))
    }
}

/** Synthetic DSLR-scale photo (red disc at 0.3w/0.5h on a gradient) — lets
 *  the harness exercise a real multi-megapixel JPEG decode in-page. */
const makeSyntheticPhoto = async (w, h) => {
    const c = new OffscreenCanvas(w, h)
    const ctx = c.getContext('2d')
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, '#3c4250')
    grad.addColorStop(1, '#20242d')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = '#d8433b'
    ctx.beginPath()
    ctx.arc(w * 0.3, h * 0.5, Math.min(w, h) * 0.15, 0, Math.PI * 2)
    ctx.fill()
    return c.convertToBlob({ type: 'image/jpeg', quality: 0.9 })
}

window.__seglab = {
    loadDemo: () => { showImage(buildDemoScene()) },
    reset: () => clearPrompts(),
    policy: () => ({ ...eff() }),
    capability: () => ({ ...capability }),
    viewSize: () => ({ width: els.view.width, height: els.view.height }),
    queueLog: () => getHeavyQueueState(),
    docMeta: () => ({
        revision: doc.revision,
        hasBlob: !!doc.blob,
        blobBytes: doc.blob?.size ?? null,
        original: doc.meta?.original || null,
        proxy: doc.meta?.proxy || null,
    }),
    importSynthetic: async (w, h) => {
        await loadFile(await makeSyntheticPhoto(w, h))
        return window.__seglab.docMeta()
    },
    // Two overlapping imports: the second must win, the first must drop.
    importRace: async () => {
        const [b1, b2] = await Promise.all([makeSyntheticPhoto(6000, 4000), makeSyntheticPhoto(2400, 3600)])
        await Promise.allSettled([loadFile(b1), loadFile(b2)])
        return { ...window.__seglab.docMeta(), view: window.__seglab.viewSize() }
    },
    // Test-only: a rejecting job must not stall the lane.
    enqueueFailing: () => enqueueHeavy('test-fail', () => Promise.reject(new Error('synthetic failure')))
        .catch((e) => String(e?.message)),
    // Test-only: proves priority order — blocker occupies the lane while a
    // speculative and an interaction job queue behind it.
    queueProbe: async () => {
        const order = []
        const mk = (name, pri, ms) => enqueueHeavy(name, () => new Promise((res) => {
            setTimeout(() => { order.push(name); res() }, ms)
        }), { priority: pri })
        const jobs = [
            mk('probe-blocker', PRIORITY.model, 120),
            mk('probe-spec', PRIORITY.speculative, 10),
            mk('probe-interaction', PRIORITY.interaction, 10),
        ]
        await Promise.all(jobs)
        return order
    },
    state: () => ({
        ready: clientState.ready,
        device: clientState.device,
        mode: clientState.mode,
        lane: clientState.lane,
        lastRun: clientState.lastRun,
        maskSummary: state.maskSummary,
        lastRefiner: state.lastRefiner,
        score: state.score,
        clicks: state.clicks.length,
    }),
    maskStats: () => {
        if (!state.mask) return null
        const { alpha, width, height } = state.mask
        let soft = 0
        for (let i = 0; i < alpha.length; i += 1) {
            if (alpha[i] > 16 && alpha[i] < 240) soft += 1
        }
        return { components: countMaskComponents(alpha, width, height), softPixels: soft }
    },
    engineState: () => getEngineSnapshot(),
    exportProbe: async () => {
        const res = await runExport()
        return res ? { width: res.width, height: res.height, reduced: res.reduced, bytes: res.blob.size } : null
    },
    enableFlagship: () => enqueueHeavy('flagship-warm', () => startFlagship(), { priority: PRIORITY.model }),
    setPressure: (level) => reportPressure(level, 'test hook'),
    releaseMemory: () => {
        clearHeavyQueue()
        resetImage(doc.revision)
        resetCv(doc.revision)
        disposeCv()
        state.mask = null
        state.maskRaw = null
        state.maskSummary = null
        renderOverlay()
        refreshButtons()
        memlog('release-memory')
        return true
    },
    // Test-only: drives the cv worker directly with crafted masks.
    cvTest: async () => {
        const W = 100
        const results = {}
        const mk = (n = W * W) => new Uint8Array(n)
        const sq = (a, x0, y0, x1, y1, v = 255) => {
            for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) a[y * W + x] = v
        }
        const run = (alpha, seeds, options = {}) => cvRefine({
            revision: doc.revision, width: W, height: W, alpha, alphaRaw: mk(), seeds, options,
        })
        results.dimReject = await cvRefine({
            revision: doc.revision, width: 800, height: 800,
            alpha: new Uint8Array(800 * 800), alphaRaw: new Uint8Array(800 * 800), seeds: [],
        }).then(() => false, () => true)
        {
            const a = mk()
            sq(a, 30, 30, 69, 69) // 40×40 object
            sq(a, 48, 48, 53, 53, 0) // interior hole
            sq(a, 5, 5, 7, 7) // 9 px crumb < minArea 48
            const r = await run(a, [[35, 35]])
            results.holeFilled = r.alpha[50 * W + 50] === 255
            results.crumbRemoved = r.alpha[6 * W + 6] === 0
            results.objectKept = r.alpha[40 * W + 40] === 255
            results.refiner = r.refiner
        }
        {
            const a = mk()
            sq(a, 10, 10, 29, 29) // seeded 400 px
            sq(a, 60, 60, 79, 79) // unseeded 400 px < minArea 500
            const r = await run(a, [[15, 15]], { minArea: 500 })
            results.seededKept = r.alpha[15 * W + 15] === 255
            results.unseededRemoved = r.alpha[70 * W + 70] === 0
        }
        {
            const a = mk()
            sq(a, 0, 0, W - 1, W - 1) // full frame — morphology must respect bounds
            const r = await run(a, [[50, 50]], { closeRadius: 3, openRadius: 2 })
            let on = 0
            for (let i = 0; i < r.alpha.length; i += 1) if (r.alpha[i] === 255) on += 1
            results.morphBounds = on === W * W
        }
        results.invalidRejected = await cvRefine({
            revision: doc.revision, width: W, height: W, alpha: mk(10), alphaRaw: mk(10), seeds: [],
        }).then(() => false, () => true)
        {
            const a = mk()
            sq(a, 30, 30, 69, 69)
            const input = a
            const r = await run(a, [[35, 35]])
            results.recoveredAfterInvalid = r.alpha[40 * W + 40] === 255
            results.transferred = input.buffer.byteLength === 0
        }
        return results
    },
    // Test-only: prompt then instant re-import — the stale result must never
    // commit to the replaced document.
    staleProbe: async () => {
        state.clicks.push([230, 256, 1])
        scheduleRun()
        await window.__seglab.importSynthetic(1600, 1200)
        await waitForRun()
        return { maskSummary: state.maskSummary, clicks: state.clicks.length }
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
try {
    navigator.serviceWorker?.register('./sw.js')
} catch { /* http/file contexts — offline reuse just stays off */ }
startPressureMonitor()
onPressureChange((level) => {
    memlog('pressure-change', { level })
    if (level >= 2) disposeCv() // wasm refinement off at L2; free its heap
    if (level >= 3) setStatus('Memory pressure detected — running in safe mode.')
})
console.log(`[seglab] ready — profile ${policy.profile}, proxy ≤${policy.proxyMax}px`)

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

import { countMaskComponents, lassoToPrompts, summarizeMaskRGBA } from './sam-core.js'
import { cancelBefore, clientState, encodeImage, segment, subscribe, warmUp, relievePressure } from './sam-client.js'
import { resolveBudget } from './policy.js'
import { probeCapability } from './capability.js'
import { importOriginal, hasOriginal, getTransform, releaseAsset } from './asset-store.js'
import { buildCutout, exportCutoutBlob, escalateCrop, getHdPatch, clearHdPatch } from './export-hd.js'
import { detectCandidates } from './text-ui.js'

// Session budget: profile preset + URL overrides (?flagship=0, ?profile=,
// ?force=wasm, ?escalate=0). The capability probe (M4) picks the preset at
// boot unless ?profile= forces it; provisional Standard covers the ~50 ms gap.
let BUDGET = resolveBudget()
let capability = null
const bootProbe = probeCapability().then((cap) => {
    capability = cap
    BUDGET = resolveBudget(location.search, cap.profile)
    return cap
}).catch((err) => {
    capability = { profile: BUDGET.profile, error: String(err?.message || err) }
    return capability
})

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
    signtoggle: $('signtoggle'), textinput: $('textinput'), selectall: $('selectall'),
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
    clicks: [],               // [[x, y, label], ...] canonical coords
    box: null,                // [x0, y0, x1, y1] canonical
    lasso: null,              // { poly, box, point, margin } from lassoToPrompts
    textCandidates: [],       // [{ box, score, label }] proxy coords, from the detector
    textMulti: false,         // phrase implied "all/every"
    mask: null,               // refined ImageData (white-on-black), canonical size
    maskRaw: null,            // decoder output before hygiene/refinement (E toggle)
    showRaw: false,
    maskSummary: null,
    score: 0,
    drag: null,               // in-progress interaction {kind, points|start}
    revision: 0,              // document revision — bumped on every prompt/image change
    running: false,
    runQueued: false,
}

// Commit bookkeeping for the headless gate: every finished run records
// whether it committed or was stale/superseded (capped, newest last).
const commitLog = []
const logCommit = (revision, outcome) => {
    commitLog.push({ revision, outcome })
    if (commitLog.length > 20) commitLog.shift()
}

/** Any prompt or document change: newer truth exists, so obsolete every
 *  older in-flight job (they skip post-processing and can never commit). */
const bumpRevision = () => {
    state.revision += 1
    clearHdPatch() // a new selection retires the escalation patch
    cancelBefore(state.revision)
}

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

// `source` is a decoded ImageBitmap or a canvas; asset-store takes custody
// of it as the ORIGINAL (the export truth) and builds the ≤proxyMax
// interaction proxy into #view. `blob` (when present) lets weak-device /
// huge-photo custody fall back to re-decoding instead of holding pixels.
const showImage = async (source, blob = null) => {
    await bootProbe // BUDGET reflects the capability probe before we size the proxy
    importOriginal(source, { blob, budget: BUDGET, proxyCanvas: els.view })
    els.overlay.width = els.view.width
    els.overlay.height = els.view.height

    state.hasImage = true
    clearPrompts()
    els.dropzone.style.display = 'none'
    els.stage.classList.add('visible')
    setStatus(clientState.ready
        ? 'Ready — click any object'
        : 'Preparing the model in the background — you can aim already')
    // Start the one-time model download NOW, while the user is aiming.
    warmUp({ budget: BUDGET }).catch((err) => setStatus(`Model load failed: ${err?.message}`))
    // M5 encode-at-import: queue the encode behind the warm so the first
    // click only pays a decode. encoded:false ⇒ came from memory/OPFS.
    state.eagerEncode = encodeImage(els.view).catch(() => null)
}

const loadFile = async (file) => {
    if (!file || !file.type?.startsWith('image/')) return
    try {
        // from-image: honour EXIF orientation (phone photos). asset-store
        // owns the bitmap from here (do NOT close it); it keeps the File as
        // the blob for re-decode custody on constrained profiles.
        const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' })
        showImage(bmp, file)
    } catch (err) {
        setStatus(`Could not read that image: ${err?.message}`)
    }
}

// Canonical demo geometry in a 900×620 logical space (the known answers the
// headless verify asserts against): big disc, rounded square, MINUTE dot.
const DEMO = {
    baseW: 900,
    baseH: 620,
    disc: { x: 230, y: 340, r: 105 },
    square: { x: 520, y: 150, w: 190, h: 190 },
    dot: { x: 700, y: 480, r: 9 },
}

/** Render the demo scene at `longSide` px on the long edge (default 900).
 *  A larger longSide makes the ORIGINAL exceed the ≤1024 proxy, so HD export
 *  genuinely upscales — that's what the M1 gate needs. */
const buildDemoScene = (longSide = DEMO.baseW) => {
    const mult = longSide / DEMO.baseW
    const c = document.createElement('canvas')
    c.width = Math.round(DEMO.baseW * mult)
    c.height = Math.round(DEMO.baseH * mult)
    const ctx = c.getContext('2d')
    ctx.scale(mult, mult) // draw in 900×620 space, rasterize at native size
    const grad = ctx.createLinearGradient(0, 0, 0, DEMO.baseH)
    grad.addColorStop(0, '#3c4250')
    grad.addColorStop(1, '#20242d')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, DEMO.baseW, DEMO.baseH)
    ctx.fillStyle = '#d8433b'
    ctx.beginPath()
    ctx.arc(DEMO.disc.x, DEMO.disc.y, DEMO.disc.r, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#3b6fd8'
    ctx.beginPath()
    ctx.roundRect(DEMO.square.x, DEMO.square.y, DEMO.square.w, DEMO.square.h, 24)
    ctx.fill()
    ctx.fillStyle = '#e8c33b'
    ctx.beginPath()
    ctx.arc(DEMO.dot.x, DEMO.dot.y, DEMO.dot.r, 0, Math.PI * 2)
    ctx.fill()
    return c
}

els.pick.addEventListener('click', () => els.file.click())
els.file.addEventListener('change', () => loadFile(els.file.files?.[0]))
els.demo.addEventListener('click', () => showImage(buildDemoScene()))
els.newimg.addEventListener('click', () => {
    state.hasImage = false
    clearPrompts()
    releaseAsset() // drop the held original (bitmap custody) before a new import
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

// Backgrounded tab → drop the detector (cheap to reload); flagship stays.
document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.hasImage) relievePressure(1).catch(() => {})
})

/* ─── Modes ──────────────────────────────────────────────────────────────── */

const setMode = (mode) => {
    state.mode = mode
    for (const [name, btn] of Object.entries(els.modes)) {
        btn.classList.toggle('active', name === mode)
    }
    els.signtoggle.style.display = mode === 'click' ? '' : 'none'
    els.textinput.hidden = mode !== 'text'
    els.selectall.hidden = mode !== 'text' || state.textCandidates.length === 0
    if (mode === 'text') els.textinput.focus()
}
els.modes.click.addEventListener('click', () => setMode('click'))
els.modes.box.addEventListener('click', () => setMode('box'))
els.modes.lasso.addEventListener('click', () => setMode('lasso'))
els.modes.text.addEventListener('click', () => setMode('text'))

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
    state.textCandidates = []
    state.mask = null
    state.maskRaw = null
    state.maskSummary = null
    state.score = 0
    state.drag = null
    els.selectall.hidden = true
    bumpRevision() // orphan + cancel any in-flight result
    renderOverlay()
    refreshButtons()
}

const undoPrompt = () => {
    if (state.clicks.length > 0) state.clicks.pop()
    else if (state.box) state.box = null
    else if (state.lasso) state.lasso = null
    bumpRevision()
    if (state.clicks.length === 0 && !state.box && !state.lasso) {
        state.mask = null
        state.maskRaw = null
        state.maskSummary = null
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
    } else if (state.mode === 'text') {
        const i = candidateAt(x, y) // tap a detected box to select it
        if (i >= 0) selectCandidate(i)
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
        bumpRevision()
        scheduleRun()
    } else if (drag.kind === 'box') {
        const [sx, sy] = drag.start
        // Discard degenerate boxes (a stray click instead of a drag).
        if (Math.abs(x - sx) > 8 && Math.abs(y - sy) > 8) {
            state.box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]
            state.lasso = null // a box replaces a lasso region
            bumpRevision()
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
            bumpRevision()
            scheduleRun()
        }
    }
    renderOverlay()
    refreshButtons()
})

/* ─── Segmentation pipeline ──────────────────────────────────────────────── */

let debounceTimer = null
let debounceArmed = false // a run is scheduled but not yet started (waitForRun must see this)
const scheduleRun = () => {
    clearTimeout(debounceTimer)
    debounceArmed = true
    debounceTimer = setTimeout(() => { debounceArmed = false; runNow() }, 80)
}

async function runNow() {
    if (!state.hasImage) return
    if (state.running) { state.runQueued = true; return }
    const clicks = state.lasso ? [state.lasso.point, ...state.clicks] : state.clicks
    const box = state.lasso ? state.lasso.box : state.box
    if (clicks.length === 0 && !box) return

    // Snapshot the document revision this run speaks for; prompts bump it,
    // so any change while we are in flight makes this run stale.
    const revision = state.revision
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
            revision,
        })
        if (res.stale) { logCommit(revision, 'stale'); return } // cancelled mid-flight
        if (revision !== state.revision) { logCommit(revision, 'superseded'); return } // a newer prompt owns the state

        logCommit(revision, res.usable ? 'committed' : 'unusable')
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
        // Show the coarse mask first, then sharpen a tiny object at native res.
        if (res.usable) await maybeEscalate(revision)
    } catch (err) {
        logCommit(revision, 'error')
        if (revision !== state.revision) return
        console.error('[seglab] selection failed:', err)
        setStatus(`Selection failed: ${err?.message}`)
    } finally {
        state.running = false
        if (state.runQueued) { state.runQueued = false; scheduleRun() }
    }
}

/* ─── Crop escalation (M3) ───────────────────────────────────────────────── */

// A small object loses detail in the ≤1024 proxy. When the committed mask is
// tiny AND the original out-resolves the proxy, re-decode ONE native crop,
// merge it back, and cache it for HD export. Auto on Std/Pro (autoEscalate);
// ?escalate=0 disables. No native headroom (proxy == original) ⇒ nothing to gain.
const shouldEscalate = (summary) => {
    if (!BUDGET.autoEscalate || !hasOriginal() || !summary?.bbox) return false
    const tf = getTransform()
    if (!tf || tf.originalW / tf.proxyW < 1.2) return false
    const [minX, minY, maxX, maxY] = summary.bbox
    const diag = Math.hypot(maxX - minX, maxY - minY)
    return diag < Math.hypot(els.view.width, els.view.height) * 0.15
}

async function maybeEscalate(revision) {
    if (!shouldEscalate(state.maskSummary)) return
    try {
        const crop = await escalateCrop(state.mask, currentPrompts(), { budget: BUDGET, revision })
        if (!crop || revision !== state.revision) return // rejected or superseded
        mergeCropIntoProxy(crop)
        renderOverlay()
    } catch (err) {
        console.warn('[seglab] escalation skipped:', err?.message)
    }
}

// Downscale the native crop mask into its proxy subrect and REPLACE that
// region (same object, sharper — the padded crop fully contains it).
function mergeCropIntoProxy({ alpha, width, height, proxySubrect }) {
    const { sx, sy, sw, sh } = proxySubrect
    const W = els.view.width
    const H = els.view.height
    const cropCanvas = new OffscreenCanvas(width, height)
    cropCanvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(alpha), width, height), 0, 0)
    const layer = new OffscreenCanvas(W, H)
    const lctx = layer.getContext('2d', { willReadFrequently: true })
    lctx.imageSmoothingEnabled = true
    lctx.imageSmoothingQuality = 'high'
    lctx.drawImage(cropCanvas, 0, 0, width, height, sx, sy, sw, sh)
    const down = lctx.getImageData(0, 0, W, H).data

    const merged = new Uint8ClampedArray(state.mask.data)
    const x0 = Math.max(0, Math.floor(sx))
    const y0 = Math.max(0, Math.floor(sy))
    const x1 = Math.min(W, Math.ceil(sx + sw))
    const y1 = Math.min(H, Math.ceil(sy + sh))
    for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
            const i = (y * W + x) * 4
            const v = down[i]
            merged[i] = v; merged[i + 1] = v; merged[i + 2] = v; merged[i + 3] = 255
        }
    }
    state.mask = new ImageData(merged, W, H)
    state.maskSummary = summarizeMaskRGBA(merged, W, H)
}

/* ─── Text select ────────────────────────────────────────────────────────── */

let detectTimer = null

async function runDetect(phrase) {
    if (!state.hasImage || !phrase.trim()) {
        state.textCandidates = []
        els.selectall.hidden = true
        renderOverlay()
        return
    }
    bumpRevision()
    const revision = state.revision
    setStatus(`Looking for “${phrase.trim()}”…`)
    try {
        const res = await detectCandidates(phrase, BUDGET)
        if (revision !== state.revision) return // superseded by newer input
        if (!res || res.candidates.length === 0) {
            state.textCandidates = []
            els.selectall.hidden = true
            setStatus(`No matches for “${phrase.trim()}”. Try different words.`)
            renderOverlay()
            return
        }
        state.textCandidates = res.candidates
        state.textMulti = res.multi
        els.selectall.hidden = res.candidates.length < 2
        const n = res.candidates.length
        setStatus(`${n} match${n > 1 ? `es — tap one or Select all` : ' — tap it'}`)
        renderOverlay()
    } catch (err) {
        if (revision !== state.revision) return
        console.error('[seglab] detect failed:', err)
        setStatus(`Text detection failed: ${err?.message}`)
    }
}

/** A detector box → a box prompt through the normal pipeline (mask + HD export). */
const selectCandidate = (i) => {
    const c = state.textCandidates[i]
    if (!c) return
    state.box = c.box.slice()
    state.clicks = []
    state.lasso = null
    state.textCandidates = []
    els.selectall.hidden = true
    bumpRevision()
    scheduleRun()
}

/** Union every candidate into one mask (multi-instance: "all bottles"). */
async function selectAll() {
    const boxes = state.textCandidates.map((c) => c.box.slice())
    if (boxes.length === 0 || state.running) return
    bumpRevision()
    const revision = state.revision
    state.running = true
    state.textCandidates = []
    els.selectall.hidden = true
    setStatus(`Selecting ${boxes.length} objects…`)
    try {
        let union = null
        for (const box of boxes) {
            const res = await segment(els.view, { box, revision })
            if (res.stale || revision !== state.revision) return
            if (!res.usable) continue
            if (!union) {
                union = new ImageData(new Uint8ClampedArray(res.imageData.data), res.width, res.height)
            } else {
                const u = union.data
                const m = res.imageData.data
                for (let k = 0; k < u.length; k += 4) if (m[k] > u[k]) { u[k] = m[k]; u[k + 1] = m[k + 1]; u[k + 2] = m[k + 2] }
            }
        }
        if (!union) { setStatus('No objects selected'); return }
        state.mask = union
        state.maskRaw = new ImageData(new Uint8ClampedArray(union.data), union.width, union.height)
        state.maskSummary = summarizeMaskRGBA(union.data, union.width, union.height)
        state.box = null
        setStatus(`Selected ${boxes.length} objects`)
        renderOverlay()
        refreshButtons()
    } finally {
        state.running = false
    }
}

/** Candidate index under a proxy-space point, or -1. */
const candidateAt = (x, y) => {
    for (let i = 0; i < state.textCandidates.length; i += 1) {
        const [x0, y0, x1, y1] = state.textCandidates[i].box
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return i
    }
    return -1
}

els.textinput.addEventListener('input', () => {
    clearTimeout(detectTimer)
    detectTimer = setTimeout(() => runDetect(els.textinput.value), 400)
})
els.textinput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { clearTimeout(detectTimer); runDetect(els.textinput.value) }
})
els.selectall.addEventListener('click', selectAll)

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

    // Text candidates: numbered boxes to tap.
    for (let i = 0; i < state.textCandidates.length; i += 1) {
        const [x0, y0, x1, y1] = state.textCandidates[i].box
        ctx.setLineDash([6, 4])
        ctx.strokeStyle = 'rgba(90,160,255,0.95)'
        ctx.lineWidth = 2
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0)
        ctx.setLineDash([])
        const tag = String(i + 1)
        ctx.font = '600 13px -apple-system, sans-serif'
        const tw = ctx.measureText(tag).width + 10
        ctx.fillStyle = 'rgba(90,160,255,0.95)'
        ctx.fillRect(x0, Math.max(0, y0 - 18), tw, 18)
        ctx.fillStyle = '#fff'
        ctx.fillText(tag, x0 + 5, Math.max(13, y0 - 5))
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

/** Proxy-resolution cutout — the fallback when no original is held (or HD
 *  export fails): composite the ≤1024 #view against the proxy mask. */
const proxyCutoutBlob = async () => {
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
    return { blob: await new Promise((res) => c.toBlob(res, 'image/png')), width: c.width, height: c.height }
}

/** Prompts in proxy coords for the current selection — the exact set the
 *  proxy decode used, so an HD re-decode reproduces the same object. */
const currentPrompts = () => ({
    clicks: state.lasso ? [state.lasso.point, ...state.clicks] : state.clicks,
    box: state.lasso ? state.lasso.box : state.box,
    clampPoly: state.lasso?.poly || null,
    clampMargin: state.lasso?.margin || 0,
})

els.cutout.addEventListener('click', async () => {
    if (!state.mask) return
    els.cutout.disabled = true
    const wasNative = hasOriginal()
    setStatus(wasNative ? 'Exporting at full resolution…' : 'Exporting…')
    try {
        let out = null
        if (wasNative) {
            out = await exportCutoutBlob(state.mask, currentPrompts(), { budget: BUDGET, revision: state.revision })
        }
        if (!out) out = await proxyCutoutBlob() // fallback: proxy-res
        const a = document.createElement('a')
        a.href = URL.createObjectURL(out.blob)
        a.download = 'seglab-cutout.png'
        a.click()
        setTimeout(() => URL.revokeObjectURL(a.href), 5000)
        setStatus(`Exported ${out.width}×${out.height} cutout${out.decoded ? ' · HD re-decode' : ''}`)
    } catch (err) {
        console.error('[seglab] export failed:', err)
        setStatus(`Export failed: ${err?.message}`)
    } finally {
        els.cutout.disabled = !state.mask
    }
})

/* ─── Headless test hooks (verify.mjs) ───────────────────────────────────── */

const waitForRun = async () => {
    // Wait out the debounce + the run; poll because runs can chain. A
    // just-finished run may have re-armed the debounce for a queued one
    // (running=false, runQueued=false, timer pending) — debounceArmed
    // covers that window.
    await new Promise((res) => setTimeout(res, 120))
    while (debounceArmed || state.running || state.runQueued) {
        await new Promise((res) => setTimeout(res, 40))
    }
}

window.__seglab = {
    loadDemo: async (longSide) => { await showImage(buildDemoScene(longSide)) },
    reset: () => clearPrompts(),
    revision: () => state.revision,
    commitLog,
    // Boot capability probe result { webgpu, fallback, f16, deviceMemoryGB,
    // profile } — awaits the probe so it is never null.
    capability: async () => { await bootProbe; return capability },
    // Drive the memory-pressure ladder; returns what was freed.
    relievePressure: (level) => relievePressure(level),
    // Resolves when the import-time eager encode (and its OPFS save) lands.
    // { encoded, lane } — encoded:false on a revisit ⇒ served from OPFS.
    eagerEncode: () => Promise.resolve(state.eagerEncode),
    // Demo ground truth in ORIGINAL pixels + the proxy transform, so the
    // headless gate derives click points (proxy space) and export checks
    // (original space) from one source instead of hardcoding scaled numbers.
    demoGeometry: () => {
        const tf = getTransform()
        if (!tf) return null
        const mult = tf.originalW / DEMO.baseW
        const scaleShape = (s) => Object.fromEntries(
            Object.entries(s).map(([k, v]) => [k, v * mult]),
        )
        return {
            originalW: tf.originalW,
            originalH: tf.originalH,
            proxyScale: tf.scale, // original px → proxy px
            disc: scaleShape(DEMO.disc),
            square: scaleShape(DEMO.square),
            dot: scaleShape(DEMO.dot),
        }
    },
    state: () => ({
        ready: clientState.ready,
        device: clientState.device,
        mode: clientState.mode,
        lane: clientState.lane,
        lastRun: clientState.lastRun,
        maskSummary: state.maskSummary,
        score: state.score,
        clicks: state.clicks.length,
        revision: state.revision,
        candidates: state.textCandidates.length,
        escalated: !!getHdPatch(state.revision),
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
    // Run the real HD export and measure the composited full-res cutout.
    // `probe` = { cx, cy, r } in ORIGINAL px (a synthetic disc): reports the
    // max radial error of the alpha boundary vs the analytic circle — the
    // quantitative "no loss at native resolution" check.
    exportCutout: async (probe = null) => {
        if (!state.mask) return null
        const res = await buildCutout(state.mask, currentPrompts(), { budget: BUDGET, revision: state.revision })
        if (!res) return null
        const { canvas, width, height, decoded } = res
        const d = canvas.getContext('2d').getImageData(0, 0, width, height).data
        const alphaAt = (x, y) => {
            const xi = Math.round(x)
            const yi = Math.round(y)
            if (xi < 0 || yi < 0 || xi >= width || yi >= height) return 0
            return d[(yi * width + xi) * 4 + 3]
        }
        let opaque = 0
        let soft = 0
        for (let i = 3; i < d.length; i += 4) {
            if (d[i] >= 128) opaque += 1
            if (d[i] > 16 && d[i] < 240) soft += 1
        }
        const out = { w: width, h: height, decoded, coverage: opaque / (width * height), softPixels: soft }
        if (probe) {
            let maxErr = 0
            const N = 64
            for (let k = 0; k < N; k += 1) {
                const ang = (k / N) * Math.PI * 2
                const cos = Math.cos(ang)
                const sin = Math.sin(ang)
                let cross = probe.r * 1.5 // no crossing found ⇒ big error (too-large mask)
                for (let rr = probe.r * 0.5; rr <= probe.r * 1.5; rr += 0.5) {
                    if (alphaAt(probe.cx + cos * rr, probe.cy + sin * rr) < 128) { cross = rr; break }
                }
                maxErr = Math.max(maxErr, Math.abs(cross - probe.r))
            }
            out.radialErr = maxErr
            out.centerOpaque = alphaAt(probe.cx, probe.cy) >= 128
            out.outsideTransparent = alphaAt(probe.cx + probe.r * 1.3, probe.cy) < 16
        }
        return out
    },
    // Crop-escalation probe. Measures the current small-object boundary in
    // ORIGINAL px against a synthetic disc `probe` {cx,cy,r}: the native patch
    // when escalation fired, else the proxy mask upscaled (the control). Lets
    // the gate compare escalate=1 vs ?escalate=0 on one metric.
    escalation: (probe = null) => {
        const patch = getHdPatch(state.revision)
        const out = { fired: !!patch, decoded: !!(patch && patch.decoded) }
        if (!probe) return out
        let d
        let W
        let H
        let s
        let ox
        let oy
        if (patch) {
            d = patch.alpha; W = patch.width; H = patch.height
            s = patch.width / patch.rect.w; ox = patch.rect.x; oy = patch.rect.y
        } else if (state.mask) {
            const tf = getTransform()
            d = state.mask.data; W = state.mask.width; H = state.mask.height
            s = tf ? tf.scale : 1; ox = 0; oy = 0
        } else {
            return { ...out, radialErr: null }
        }
        const cx = (probe.cx - ox) * s
        const cy = (probe.cy - oy) * s
        const r = probe.r * s
        const alphaAt = (x, y) => {
            const xi = Math.round(x)
            const yi = Math.round(y)
            if (xi < 0 || yi < 0 || xi >= W || yi >= H) return 0
            return d[(yi * W + xi) * 4]
        }
        let maxErr = 0
        const N = 64
        for (let k = 0; k < N; k += 1) {
            const ang = (k / N) * Math.PI * 2
            const cos = Math.cos(ang)
            const sin = Math.sin(ang)
            let cross = r * 1.6 // no crossing ⇒ big error
            for (let rr = r * 0.4; rr <= r * 1.6; rr += 0.5) {
                if (alphaAt(cx + cos * rr, cy + sin * rr) < 128) { cross = rr; break }
            }
            maxErr = Math.max(maxErr, Math.abs(cross - r))
        }
        return { ...out, radialErr: maxErr / s, centerOpaque: alphaAt(cx, cy) >= 128 }
    },
    clickAt: async (x, y, negative = false) => {
        state.clicks.push([x, y, negative ? 0 : 1])
        bumpRevision() // mirror the real pointerup path
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
        bumpRevision()
        scheduleRun()
        await waitForRun()
        return window.__seglab.state()
    },
    // Run the OWLv2 detector; returns candidate boxes (for the WARN-level
    // detector gate — headless model quality is not asserted hard).
    detect: async (phrase) => {
        await runDetect(phrase)
        return { candidates: state.textCandidates.length, boxes: state.textCandidates.map((c) => c.box), multi: state.textMulti }
    },
    // Deterministic text-select PLUMBING (no detector): drive given proxy
    // boxes through the box→mask→union path exactly as a real pick would.
    selectBoxes: async (boxes) => {
        state.textCandidates = boxes.map((box) => ({ box, score: 1, label: 'test' }))
        if (boxes.length === 1) { selectCandidate(0); await waitForRun() } else { await selectAll() }
        const s = window.__seglab.state()
        return { ...s, ...(window.__seglab.maskStats() || {}) }
    },
}
window.__seglabReady = true

/* ─── Boot ───────────────────────────────────────────────────────────────── */

setMode('click')
refreshChips()
console.log('[seglab] ready')

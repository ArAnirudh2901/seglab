/**
 * app — SEGLAB interface
 * ------------------------
 * Import a photo → select anything with clicks (+/−), a box, or a rough
 * lasso that snaps to the object. All inference on-device (sam-client →
 * worker → SlimSAM); this module owns only UI state, prompt collection,
 * overlay rendering, and the lasso clamp.
 *
 * One reference frame: the photo is downscaled once into a ≤768 canonical
 * canvas (#view). Prompts, masks, overlay, and the cutout all live in that
 * frame — display scaling is pure CSS, undone at the pointer.
 */

import { countMaskComponents, lassoToPrompts, summarizeMaskRGBA, maskToChannel, composeChannels, pointInMask, dilateChannel } from './sam-core.js'
import {
    cancelBefore, clientState, encodeImage, engineState, releaseDocument, segment, subscribe, warmUp, relievePressure,
} from './sam-client.js'
import { applyMemoryPressure, resolveBudget } from './policy.js'
import { probeCapability, readPhosmithResources, withPhosmithResources } from './capability.js'
import { importOriginal, hasOriginal, getTransform, releaseAsset, getOriginalBlob, getAssetKey } from './asset-store.js'
import { saveSession, loadSession, clearSession } from './session-store.js'
import { isRawFile, extractRawPreview } from './image-raw.js'
import { developRaw } from './raw-develop-client.js'
import { buildCutout, exportCutoutBlob, escalateCrop, getHdPatch, clearHdPatch } from './export-hd.js'
import { detectCandidates } from './text-ui.js'
import { acceleratedDetectorAvailable, detectorCached, detectorDownloadMB } from './detect-engine.js'
import { clearHeavyQueue, getHeavyQueueState, getHeavyQueueLog } from './heavy-job-queue.js'
import { refineAlpha, disposeCvRefine, cvRefineAvailable } from './cv-refine-client.js'

// Session budget: profile preset + URL overrides. The provisional budget is
// LITE — nothing above it can be proven before the capability probe, and a
// plain browser stays lite forever (only a trusted Phosmith hint raises it).
let BUDGET = resolveBudget()
let capability = null
let pendingPhosmithResources = null
const bootProbe = probeCapability().then((cap) => {
    capability = pendingPhosmithResources
        ? withPhosmithResources(cap, pendingPhosmithResources)
        : cap
    BUDGET = applyMemoryPressure(resolveBudget(location.search, capability), BUDGET.pressureLevel || 0)
    refreshChips()
    return cap
}).catch((err) => {
    capability = { profile: BUDGET.profile, error: String(err?.message || err) }
    return capability
})

// No model loads before the user supplies an image. The warm request is
// enqueued AFTER the import decode releases its bitmaps (heavy-job queue
// serializes them regardless).
let warmStarted = false
const ensureWarm = () => {
    warmStarted = true
    return warmUp({ budget: BUDGET })
}

const ACCENT = '#35e0c2'
const POS_COLOR = '#35e08a'
const NEG_COLOR = '#ff5d6c'

const $ = (id) => document.getElementById(id)
const els = {
    main: $('main'), dropzone: $('dropzone'), stage: $('stage'),
    photo: $('photo'), view: $('view'), overlay: $('overlay'), file: $('file'),
    pick: $('pick'), demo: $('demo'), newimg: $('newimg'),
    status: $('status'), loadbar: $('loadbar'),
    prep: $('prep'), prepText: $('prep-text'),
    chipMode: $('chip-mode'), chipDevice: $('chip-device'), chipTiming: $('chip-timing'),
    undo: $('undo'), reset: $('reset'), cutout: $('cutout'),
    signtoggle: $('signtoggle'), textinput: $('textinput'), selectall: $('selectall'),
    toleranceWrap: $('tolerance-wrap'), tolerance: $('tolerance'), toleranceValue: $('tolerance-value'),
    modes: {
        click: $('mode-click'), box: $('mode-box'),
        lasso: $('mode-lasso'), region: $('mode-region'), rect: $('mode-rect'),
        ellipse: $('mode-ellipse'), polygon: $('mode-polygon'), magic: $('mode-magic'), color: $('mode-color'),
        brush: $('mode-brush'), text: $('mode-text'),
    },
}

/* ─── State ──────────────────────────────────────────────────────────────── */

const state = {
    hasImage: false,
    mode: 'click',
    sign: 1,                  // primary-tap label for touch devices
    clicks: [],               // [[x, y, label], ...] canonical coords
    box: null,                // [x0, y0, x1, y1] canonical
    lasso: null,              // { poly, box, point, margin } from lassoToPrompts
    manual: null,             // { kind, poly?|box?|seed? } direct Photoshop-style selection
    polygonDraft: [],
    // Per-channel RGB distance used only by Magic and Color. 36 is a useful
    // middle ground for photos; 72 was broad enough to merge unrelated hues.
    wandTolerance: 36,
    textCandidates: [],       // [{ box, score, label }] proxy coords, from the detector
    textMulti: false,         // phrase implied "all/every"
    mask: null,               // DERIVED: baseMask ∪ liveMask — the composed selection
    maskRaw: null,            // baseMask ∪ live raw decoder mask (E toggle)
    showRaw: false,
    maskSummary: null,        // summary of the composed mask
    // Committed-selection stack: every finished object/manual op lands here so
    // a new click can never unselect earlier regions. Z peels one op.
    baseMask: null,           // ImageData replay of baseFloor+baseOps
    baseOps: [],              // [{op:'add'|'sub', chan:Uint8Array, kind}]
    baseFloor: null,          // flattened overflow of the op stack (Uint8Array)
    liveMask: null,           // the object the current clicks/box/lasso describe
    liveRaw: null,
    liveSummary: null,
    score: 0,
    drag: null,               // in-progress interaction {kind, points|start}
    brush: null,              // transient canvas-backed stroke; committed on pointerup
    revision: 0,              // document revision — bumped on every prompt/image change
    running: false,
    runQueued: false,
    eagerEncode: null,        // idle-time encode promise; resolves null when input supersedes it
    encodePending: false,     // idle window was not calm enough; next selection will encode normally
    imageEpoch: 0,            // newest requested import; stale queued files never decode
    preprocessEpoch: 0,       // invalidates a queued idle encode on any user input
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
    // A click belongs ahead of speculative preprocessing. An encode already
    // inside an ONNX kernel cannot be interrupted safely, but this prevents a
    // not-yet-started idle encode from jumping ahead of the user's selection.
    state.preprocessEpoch += 1
    clearHdPatch() // a new selection retires the escalation patch
    cancelBefore(state.revision)
}

const overlayCtx = els.overlay.getContext('2d')

/* ─── Status / chips ─────────────────────────────────────────────────────── */

const setStatus = (msg) => { els.status.textContent = msg }

const refreshChips = () => {
    const lane = clientState.lane ? ` · ${clientState.lane}` : ''
    els.chipMode.textContent = `engine: ${clientState.mode || '—'}${lane}`
    const profile = BUDGET.profile || 'standard'
    const gpu = capability?.gpuTier || 'probing'
    // Only a trusted host budget is shown as a memory figure; a browser's
    // deviceMemory report is unverifiable, so it never appears as a number.
    const memory = (capability?.memorySource === 'phosmith' && capability.memoryGB)
        ? ` · ${capability.memoryGB} GB host`
        : ''
    els.chipDevice.textContent = `device: ${clientState.device || '—'} · ${profile} · ${gpu}${memory}`
    els.chipDevice.classList.toggle('on', clientState.device === 'webgpu')
    const run = clientState.lastRun
    els.chipTiming.textContent = run
        ? (run.encoded
            ? `encode ${run.encodeMs}ms · decode ${run.decodeMs}ms · post ${run.postMs}ms`
            : `decode ${run.decodeMs}ms · post ${run.postMs}ms (cached)`)
        : '— ms'
}

/** A Phosmith WebView can tighten or expand its usable-memory budget after the
 * editor has loaded. Existing image/model allocations are never enlarged in
 * place; the revised budget governs future imports, re-decodes and exports. */
const applyPhosmithResources = (resources = readPhosmithResources()) => {
    if (!capability) {
        pendingPhosmithResources = resources
        return null
    }
    const previous = capability
    capability = withPhosmithResources(capability, resources)
    // Pressure is a one-way ratchet: a live budget update never resets it.
    BUDGET = applyMemoryPressure(resolveBudget(location.search, capability), BUDGET.pressureLevel || 0)
    const profileRank = { lite: 0, standard: 1, pro: 2, ultra: 3 }
    const memoryReduced = previous.memoryGB > 0 && capability.memoryGB > 0
        && capability.memoryGB < previous.memoryGB
    const needsHeavyRelease = memoryReduced
        || (profileRank[capability.profile] || 0) < (profileRank[previous.profile] || 0)
    // A host downgrade is a real resource event, not only a UI-label change:
    // immediately release reloadable residents before the next allocation.
    if (needsHeavyRelease) relievePressure(3).catch(() => {})
    refreshChips()
    if (state.hasImage) {
        const suffix = needsHeavyRelease ? '; heavy GPU residents released' : ''
        setStatus(`Resource budget updated — ${BUDGET.profile} profile applies to the next import/export${suffix}`)
    }
    return capability
}

window.addEventListener('phosmithresourceschange', (event) => {
    applyPhosmithResources(event.detail ?? readPhosmithResources())
})

subscribe((event) => {
    if (event.type === 'progress') {
        const d = event.detail || {}
        if (d.status === 'progress' && d.total) {
            const pct = Math.round((d.loaded / d.total) * 100)
            els.loadbar.style.width = `${pct}%`
            setStatus(`Downloading SlimSAM — ${String(d.file || '').split('/').pop()} ${pct}% (one-time, ~14 MB)`)
        } else if (d.status === 'done') {
            els.loadbar.style.width = '0%'
        }
        return
    }
    if (event.type === 'lane') { refreshChips(); return }
    if (event.type === 'stage') {
        const d = event.detail || {}
        if (state.running && d.revision === state.revision) {
            if (d.stage === 'encode') setStatus('Selecting… analyzing the photo (one-time per image)')
            else if (d.stage === 'decode') setStatus(d.encoded ? 'Selecting… tracing the object' : 'Selecting… tracing the object (photo already prepared)')
        }
        return
    }
    refreshChips()
    if (clientState.ready && !state.running && !state.hasImage) {
        setStatus('Model ready — import a photo to begin')
    }
})

/* ─── Image import ───────────────────────────────────────────────────────── */

// Pressure sentinel: rAF gaps expose swap/GPU starvation. true after
// `frames` calm frames; false if the machine never settles in maxWaitMs.
const waitForCalm = ({ frames = 8, budgetMs = 90, maxWaitMs = 8000 } = {}) => new Promise((resolve) => {
    const t0 = performance.now()
    let calm = 0
    let prev = t0
    const tick = (now) => {
        const gap = now - prev
        prev = now
        if (gap > 2500) shedMemory(1, { announce: false }) // visibly choking — free reloadables
        calm = gap <= budgetMs ? calm + 1 : 0
        if (calm >= frames) { resolve(true); return }
        if (now - t0 > maxWaitMs) { resolve(false); return }
        requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
})

// Veil only for the small period where an image has no usable interaction
// frame yet (or for a native-res export). Model work and eager embedding stay
// visible in the status line, never behind a modal editor lock.
// Epochs stop a finishing path from hiding a veil a newer one just raised.
let prepEpoch = 0
const showPrep = (text) => {
    els.prepText.textContent = text
    els.prep.hidden = false
    return ++prepEpoch
}
const hidePrep = (epoch = prepEpoch) => {
    if (epoch === prepEpoch) els.prep.hidden = true
}

// A queued upload is only compressed bytes until it reaches importOriginal.
// Serializing that transition prevents two decoders from briefly owning full
// source frames at once when the user changes their mind mid-import.
let importChain = Promise.resolve()
const beginImageRequest = () => {
    state.imageEpoch += 1
    state.preprocessEpoch += 1
    state.eagerEncode = null
    state.encodePending = false
    return state.imageEpoch
}
const imageRequestIsCurrent = (epoch) => epoch === state.imageEpoch
const queueImage = (source, options, imageEpoch = beginImageRequest()) => {
    const task = importChain.catch(() => {}).then(() => showImage(source, options, imageEpoch))
    importChain = task.catch(() => {})
    return task
}

// Give pointer/keyboard input a chance to arrive before speculative work. The
// 750 ms deadline still gets the first-click embedding ready promptly when the
// editor is actually idle; Safari lacks requestIdleCallback, so it uses the
// same short timer without blocking the renderer.
const waitForIdle = (timeout = 750) => new Promise((resolve) => {
    if (typeof requestIdleCallback === 'function') {
        requestIdleCallback(() => resolve(), { timeout })
    } else {
        setTimeout(resolve, Math.min(timeout, 350))
    }
})

const scheduleEagerEncode = (imageEpoch, revision, readyStatus) => {
    const preprocessEpoch = ++state.preprocessEpoch
    const current = () => imageRequestIsCurrent(imageEpoch)
        && preprocessEpoch === state.preprocessEpoch
        && revision === state.revision
    state.eagerEncode = (async () => {
        await waitForIdle()
        if (!current() || document.hidden) return null
        // Do not stack the proxy encoder immediately behind the model's own
        // compilation/upload burst. If frames do not settle, defer safely to
        // the first selection instead of forcing an unsafe memory peak. The
        // baseline profile waits longer for the same reason.
        const lite = BUDGET.profile === 'lite'
        const calm = await waitForCalm(lite ? { frames: 16, maxWaitMs: 10000 } : { frames: 12, maxWaitMs: 6000 })
        if (!current()) return null
        if (!calm) {
            state.encodePending = true
            console.warn('[seglab] idle encode deferred — device did not settle after model warm')
            return null
        }
        // prime: one throwaway decode initializes the decoder session too.
        const result = await encodeImage(els.view, { revision, prime: true })
        if (!current() || result?.stale) return null
        state.encodePending = false
        if (!state.running) setStatus(`${readyStatus} · prepared locally`)
        return result
    })().catch((err) => {
        if (current()) {
            state.encodePending = true
            console.warn('[seglab] idle encode failed; first selection will retry:', err?.message)
        }
        return null
    })
}

// `source` is a Blob/File (upload) or a canvas/bitmap (demo). asset-store
// builds the ≤proxyMax interaction proxy into #view and takes custody of the
// ORIGINAL — blob-only for uploads (decode straight to the proxy, re-decode
// bounded regions on demand), so a huge photo never lands in RAM as full-res.
const showImage = async (source, {
    proxyBlob = null, orientation = null, raw = false, sourceBytes = source?.size || 0,
} = {}, imageEpoch) => {
    if (!imageRequestIsCurrent(imageEpoch)) return null
    const epoch = showPrep('Building a lightweight interaction preview…')
    await bootProbe // BUDGET reflects the capability probe before we size the proxy
    if (!imageRequestIsCurrent(imageEpoch)) { hidePrep(epoch); return null }

    // A new document obsoletes every queued job and the old embeddings before
    // any new decode starts. Model weights stay; the queue serializes the
    // decode against any still-running kernel.
    clearHeavyQueue()
    if (warmStarted) releaseDocument()

    // A RAW container's embedded JPEG preview (not the 33 MB sensor payload)
    // becomes the only image decoded during interaction.
    if (raw) {
        els.prepText.textContent = 'Extracting the camera preview…'
        setStatus('Reading the camera preview — original sensor data stays untouched')
        const preview = await extractRawPreview(source, {
            // A 1024px interaction proxy never benefits from decoding a larger
            // embedded thumbnail. Higher trusted profiles can request more.
            proxyMinEdge: Math.max(768, BUDGET.proxyMax || 1024),
        })
        if (!imageRequestIsCurrent(imageEpoch)) { hidePrep(epoch); return null }
        if (preview) {
            console.log('[seglab][ui] raw-import-ready', {
                full: `${preview.width}x${preview.height}`,
                hasProxy: Boolean(preview.proxyBlob),
                orientation: preview.orientation,
            })
            source = preview.blob
            proxyBlob = preview.proxyBlob
            orientation = preview.orientation
        } else {
            // No embedded preview — develop the sensor on-device with LibRaw
            // (rare fallback; worker disposed after use). Still no cloud, no
            // Python: only a compact developed JPEG crosses back.
            els.prepText.textContent = 'Developing the RAW on-device…'
            setStatus('No embedded preview — developing the sensor on-device (this stays local)')
            const developed = await developRaw(source, { budget: BUDGET })
            if (!imageRequestIsCurrent(imageEpoch)) { hidePrep(epoch); return null }
            if (!developed) throw new Error('No readable JPEG preview is embedded in this RAW, and on-device develop was unavailable. Export a JPEG/TIFF and try again.')
            console.log('[seglab][ui] raw-develop-ready', { full: `${developed.width}x${developed.height}` })
            source = developed.blob
            proxyBlob = null
            orientation = 1 // LibRaw already applies the camera's flip
        }
    }

    els.prepText.textContent = 'Building a lightweight interaction preview…'
    const transform = await importOriginal(source, {
        budget: BUDGET,
        proxyCanvas: els.view,
        displayCanvas: els.photo,
        proxyBlob,
        orientation,
        sourceWasRaw: raw,
        sourceBytes,
        isCurrent: () => imageRequestIsCurrent(imageEpoch),
    })
    if (!transform || !imageRequestIsCurrent(imageEpoch)) {
        // importChain guarantees no later import has started yet, so this is
        // safe and releases an abandoned proxy before the next request runs.
        releaseAsset()
        hidePrep(epoch)
        return null
    }
    els.overlay.width = els.view.width
    els.overlay.height = els.view.height

    state.hasImage = true
    clearPrompts()
    els.dropzone.style.display = 'none'
    els.stage.classList.add('visible')
    const frameMode = transform.proxyActive
        ? `adaptive ${transform.proxyW}×${transform.proxyH} proxy`
        : 'native interaction frame (proxy disabled)'
    // The display frame is decoupled from the model proxy — show its real size.
    const displayNote = (els.photo.width > transform.proxyW)
        ? ` · ${els.photo.width}×${els.photo.height} preview`
        : ''
    const readyStatus = `Ready — click any object · ${frameMode}${displayNote}`
    setStatus(clientState.ready ? readyStatus : `${readyStatus} · preparing the model in the background`)
    // Persist the document now: until the first commit the whole import would
    // otherwise be lost to a power cut.
    persistSession({ immediate: true })
    // The proxy is on screen; ONLY NOW does SlimSAM warm (queued — it can
    // never overlap the decode that just released its bitmaps).
    hidePrep(epoch)
    ensureWarm().catch((err) => {
        console.warn('[seglab] model warm failed (first selection will retry):', err?.message)
    })
    // Speculative encode-at-import: bounded (one embedding, calm-gated) so
    // even the baseline profile pays the encode before the first click. The
    // pressure ratchet turns it back off.
    if (BUDGET.eagerEncode) {
        scheduleEagerEncode(imageEpoch, state.revision, readyStatus)
    }
    return transform
}

const loadFile = async (file) => {
    if (!file) return
    const imageEpoch = beginImageRequest()
    try {
        // The RAW stays a Blob; showImage extracts only its embedded
        // developed JPEG preview, and the decode rides the heavy-job queue.
        if (isRawFile(file)) {
            console.log('[seglab][ui] raw-import-start', { name: file.name, bytes: file.size })
            return await queueImage(file, { raw: true, sourceBytes: file.size }, imageEpoch)
        }
        if (!file.type?.startsWith('image/')) return
        // Hand the compressed File straight to asset-store — it decodes only a
        // ≤proxyMax proxy (never the full-res frame) and keeps the Blob for
        // bounded re-decodes. EXIF orientation is honoured inside importOriginal.
        return await queueImage(file, { sourceBytes: file.size }, imageEpoch)
    } catch (err) {
        if (imageRequestIsCurrent(imageEpoch)) {
            hidePrep() // any epoch — an import error must never leave the veil up
            setStatus(`Could not read that image: ${err?.message}`)
        }
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
els.file.addEventListener('change', () => {
    const file = els.file.files?.[0]
    // Selecting the same file twice is a new import request too.
    els.file.value = ''
    loadFile(file)
})
els.demo.addEventListener('click', () => queueImage(buildDemoScene()))
els.newimg.addEventListener('click', () => {
    beginImageRequest() // invalidate a queued file/idle encode before cleanup
    clearHeavyQueue()
    if (warmStarted) releaseDocument()
    state.hasImage = false
    hidePrep()
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

const shedMemory = (level, { announce = true } = {}) => {
    const previous = BUDGET.pressureLevel || 0
    BUDGET = applyMemoryPressure(BUDGET, level)
    if ((BUDGET.pressureLevel || 0) >= 2) disposeCvRefine() // idle wasm worker goes first
    if (announce && BUDGET.pressureLevel > previous) {
        if (BUDGET.pressureLevel >= 3) {
            setStatus('Memory pressure detected — running in safe mode.')
        } else {
            const detail = BUDGET.pressureLevel === 2
                ? 'wasm refinement, escalation and heavy caches paused'
                : 'detector and extra model caches released'
            setStatus(`Memory pressure — ${detail}`)
        }
        refreshChips()
    }
    return relievePressure(level).catch(() => [])
}

// Backgrounded tab → drop the detector (cheap to reload) and stop optional
// detail escalation. The full image remains a compressed Blob, not RGBA RAM.
document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.hasImage) shedMemory(1, { announce: false })
})

// Heap watchdog (Chrome-only; sees only this thread's JS heap, so thresholds
// are deliberately conservative): stop escalation, then shed detector and any
// reloadable model state long before the host starts heavy swapping.
if (performance.memory) {
    setInterval(() => {
        if (!state.hasImage) return
        const used = performance.memory.usedJSHeapSize
        if (used > 650e6) {
            shedMemory(3)
        } else if (used > 450e6) {
            shedMemory(2)
        } else if (used > 300e6) {
            shedMemory(1)
        }
    }, 5000)
} else {
    // No heap signal (Safari): timer drift is the live pressure sentinel.
    // A visible tab whose 1 s tick lands seconds late is swapping/starved —
    // generalizes the one-shot rAF-gap check inside waitForCalm.
    let lastTick = performance.now()
    let stalls = 0
    setInterval(() => {
        const now = performance.now()
        const drift = now - lastTick - 1000
        lastTick = now
        if (!state.hasImage || document.hidden) { stalls = 0; return }
        if (drift > 2500) {
            stalls += 1
            shedMemory(stalls >= 2 ? 2 : 1, { announce: stalls >= 2 })
        } else if (drift < 250) {
            stalls = 0
        }
    }, 1000)
}

/* ─── Modes ──────────────────────────────────────────────────────────────── */

// Modes where include/exclude applies — via the sign toggle or right/Alt-click.
const SIGN_MODES = new Set(['click', 'magic', 'color', 'region', 'rect', 'ellipse', 'polygon'])

const setMode = (mode) => {
    if (mode !== 'polygon') state.polygonDraft = []
    state.mode = mode
    for (const [name, btn] of Object.entries(els.modes)) {
        btn.classList.toggle('active', name === mode)
    }
    els.signtoggle.style.display = SIGN_MODES.has(mode) ? '' : 'none'
    els.toleranceWrap.hidden = mode !== 'magic' && mode !== 'color'
    els.textinput.hidden = mode !== 'text'
    els.selectall.hidden = mode !== 'text' || state.textCandidates.length === 0
    if (mode === 'text') {
        els.textinput.focus()
        void hintTextSearch()
    }
}
els.modes.click.addEventListener('click', () => setMode('click'))
els.modes.box.addEventListener('click', () => setMode('box'))
els.modes.lasso.addEventListener('click', () => setMode('lasso'))
els.modes.region.addEventListener('click', () => setMode('region'))
els.modes.rect.addEventListener('click', () => setMode('rect'))
els.modes.ellipse.addEventListener('click', () => setMode('ellipse'))
els.modes.polygon.addEventListener('click', () => setMode('polygon'))
els.modes.magic.addEventListener('click', () => setMode('magic'))
els.modes.color.addEventListener('click', () => setMode('color'))
els.modes.brush.addEventListener('click', () => setMode('brush'))
els.modes.text.addEventListener('click', () => setMode('text'))

els.tolerance.addEventListener('input', () => {
    state.wandTolerance = Number(els.tolerance.value)
    els.toleranceValue.value = String(state.wandTolerance)
})

els.signtoggle.addEventListener('click', () => {
    state.sign = state.sign ? 0 : 1
    els.signtoggle.textContent = state.sign ? '＋ include' : '－ exclude'
    els.signtoggle.classList.toggle('pos', !!state.sign)
    els.signtoggle.classList.toggle('neg', !state.sign)
})

/* ─── Prompt state ───────────────────────────────────────────────────────── */

const refreshButtons = () => {
    const any = state.clicks.length > 0 || state.box || state.lasso || state.manual || state.baseOps.length > 0
    els.undo.disabled = !any
    els.reset.disabled = !any
    els.cutout.disabled = !state.mask
}

function clearPrompts() {
    state.clicks = []
    state.box = null
    state.lasso = null
    state.manual = null
    state.polygonDraft = []
    state.textCandidates = []
    state.mask = null
    state.maskRaw = null
    state.maskSummary = null
    state.baseMask = null
    state.baseOps = []
    state.baseFloor = null
    clearLive()
    state.score = 0
    state.drag = null
    state.brush = null
    els.selectall.hidden = true
    layerCache = { mask: null, fill: null } // release the committed overlay cache
    bumpRevision() // orphan + cancel any in-flight result
    renderOverlay()
    refreshButtons()
    persistSession() // a cleared selection is work too — never resurrect it
}

const undoPrompt = () => {
    if (state.manual) {
        // A manual gesture is the top op on the stack — peel it with its marker.
        state.manual = null
        if (state.baseOps.length) { state.baseOps.pop(); recomposeBase() }
    } else if (state.clicks.length > 1) {
        state.clicks.pop()
    } else if (state.clicks.length === 1) {
        state.clicks = []
        clearLive() // discard the live object; committed regions survive
    } else if (state.box) {
        state.box = null
        clearLive()
    } else if (state.lasso) {
        state.lasso = null
        clearLive()
    } else if (state.baseOps.length) {
        state.baseOps.pop()
        recomposeBase()
    }
    bumpRevision()
    persistSession()
    if (state.clicks.length === 0 && !state.box && !state.lasso && !state.manual) {
        recomposeMask()
        renderOverlay()
        refreshButtons()
        setStatus(state.mask ? 'Removed last step' : 'Cleared')
        return
    }
    scheduleRun()
}

els.undo.addEventListener('click', undoPrompt)
els.reset.addEventListener('click', () => { clearPrompts(); setStatus('Cleared') })
window.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && state.mode === 'polygon' && state.polygonDraft.length >= 3) {
        e.preventDefault()
        finishPolygon()
        return
    }
    if (e.key === 'Escape' && state.polygonDraft.length) {
        state.polygonDraft = []
        renderOverlay()
        return
    }
    if (e.key === 'z' || e.key === 'Z') undoPrompt()
    if (e.key === 'r' || e.key === 'R') { clearPrompts(); setStatus('Cleared') }
    if (e.key === 'e' || e.key === 'E') {
        state.showRaw = !state.showRaw
        renderOverlay()
        setStatus(state.showRaw ? 'Showing RAW decoder mask (E to toggle back)' : 'Showing refined mask')
    }
})

/* ─── Pointer handling ───────────────────────────────────────────────────── */

const manualMask = (draw) => {
    const c = new OffscreenCanvas(els.view.width, els.view.height)
    const ctx = c.getContext('2d', { willReadFrequently: true })
    draw(ctx)
    return ctx.getImageData(0, 0, c.width, c.height)
}

const polygonMask = (poly) => manualMask((ctx) => {
    ctx.beginPath()
    ctx.moveTo(poly[0][0], poly[0][1])
    for (let i = 1; i < poly.length; i += 1) ctx.lineTo(poly[i][0], poly[i][1])
    ctx.closePath()
    ctx.fillStyle = '#fff'
    ctx.fill()
})

const rectMask = ([x0, y0, x1, y1]) => manualMask((ctx) => {
    ctx.fillStyle = '#fff'
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0)
})

const ellipseMask = ([x0, y0, x1, y1]) => manualMask((ctx) => {
    ctx.beginPath()
    ctx.ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0, 0, Math.PI * 2)
    ctx.fillStyle = '#fff'
    ctx.fill()
})

// Union two white-on-black masks per-pixel max, so the refined soft boundary
// band survives composition (a hard 0/255 write would harden every edge).
const softUnion = (base, patch) => {
    const w = patch.width
    const h = patch.height
    const out = base && base.width === w && base.height === h
        ? new ImageData(new Uint8ClampedArray(base.data), w, h)
        : new ImageData(w, h)
    const o = out.data
    const p = patch.data
    for (let i = 0; i < p.length; i += 4) {
        if (p[i] > o[i]) { o[i] = p[i]; o[i + 1] = p[i + 1]; o[i + 2] = p[i + 2] }
        o[i + 3] = 255
    }
    return out
}

// ~0.8 MB per op at the 1024 proxy; beyond the cap the oldest op flattens
// into the floor so undo depth stays bounded.
const MAX_BASE_OPS = 12

const recomposeBase = () => {
    const W = els.view.width
    const H = els.view.height
    if (!state.baseOps.length && !state.baseFloor) { state.baseMask = null; return }
    const res = composeChannels(state.baseOps, W, H, state.baseFloor)
    state.baseMask = res ? new ImageData(res.rgba, W, H) : null
}

const pushBaseOp = (op, imageData, kind, { dilate = 0 } = {}) => {
    // A decode-based subtract grows a little so removing an object never
    // leaves a residue ring where two decodes disagree by a pixel. Manual
    // geometry (rect/region/…) stays exact.
    const chan = dilate
        ? dilateChannel(maskToChannel(imageData), imageData.width, imageData.height, dilate)
        : maskToChannel(imageData)
    state.baseOps.push({ op, chan, kind })
    if (state.baseOps.length > MAX_BASE_OPS) {
        const oldest = state.baseOps.shift()
        const floor = state.baseFloor || new Uint8Array(els.view.width * els.view.height)
        if (oldest.op === 'sub') { for (let i = 0; i < floor.length; i += 1) if (oldest.chan[i] >= 128) floor[i] = 0 }
        else { for (let i = 0; i < floor.length; i += 1) if (oldest.chan[i] > floor[i]) floor[i] = oldest.chan[i] }
        state.baseFloor = floor
    }
    recomposeBase()
}

/* ─── Crash-safe session persistence ─────────────────────────────────────── */
// A power cut fires no pagehide/beforeunload, so saving on unload saves
// nothing. Every commit writes durably (debounced, atomic) instead.
let restoring = false
let sessionName = 'restored'

const persistSession = ({ immediate = false } = {}) => {
    // A restore's own import must not write its null mask over the saved work.
    if (restoring) return
    saveSession(() => {
        if (!state.hasImage) return null
        const blob = getOriginalBlob()
        if (!blob) return null // resident drawable (demo) — nothing to re-import from
        return {
            blob,
            assetKey: getAssetKey(),
            w: els.view.width,
            h: els.view.height,
            mime: blob.type,
            name: sessionName,
            // Composed truth (base ∪ live) — a lone click never reaches baseOps.
            mask: state.mask ? maskToChannel(state.mask) : null,
        }
    }, { delay: immediate ? 0 : 600 })
}

/** Reopen the last session. Returns true if a document was restored. */
const restoreSession = async () => {
    const saved = await loadSession()
    if (!saved?.blob) return false
    restoring = true
    try {
        sessionName = saved.name
        // Claim an epoch: a user drop (or a test's own import) landing mid-restore
        // supersedes us, and we must not write the old ops over their document.
        const epoch = beginImageRequest()
        await queueImage(new File([saved.blob], saved.name, { type: saved.blob.type }), { sourceBytes: saved.blob.size }, epoch)
        if (!imageRequestIsCurrent(epoch) || !state.hasImage) return false
        // An op stack only composes at the proxy size it was captured at; a
        // different budget on this run means the geometry no longer lines up.
        const sameFrame = els.view.width === saved.w && els.view.height === saved.h
        if (sameFrame && saved.mask) {
            state.baseOps = []
            state.baseFloor = saved.mask
            recomposeBase()
            recomposeMask()
            renderOverlay()
            refreshButtons()
            const pct = ((state.maskSummary?.coverage || 0) * 100).toFixed(1)
            setStatus(`Restored your last session — selection ${pct}% of frame`)
        } else {
            setStatus(sameFrame ? 'Restored your last image' : 'Restored your last image — selection dropped (frame size changed)')
        }
        console.log('[seglab][ui] session-restored', { sameFrame, mask: !!saved.mask, savedAt: saved.savedAt })
        return true
    } catch (err) {
        console.warn('[seglab] session restore failed:', err?.message)
        return false
    } finally {
        restoring = false
    }
}

/** Rebuild the composed truth (state.mask/maskSummary) from base ∪ live. */
const recomposeMask = () => {
    if (state.liveMask) {
        state.mask = softUnion(state.baseMask, state.liveMask)
    } else if (state.baseMask) {
        state.mask = new ImageData(new Uint8ClampedArray(state.baseMask.data), state.baseMask.width, state.baseMask.height)
    } else {
        state.mask = null
    }
    state.maskRaw = state.liveRaw ? softUnion(state.baseMask, state.liveRaw) : null
    state.maskSummary = state.mask ? summarizeMaskRGBA(state.mask.data, state.mask.width, state.mask.height) : null
    persistSession() // debounced + snapshot-at-write: a drag costs one save
}

const clearLive = () => {
    state.liveMask = null
    state.liveRaw = null
    state.liveSummary = null
}

const commitManualMask = (kind, imageData, geometry = {}, negative = false) => {
    // A region that landed on nothing (miss / empty geometry) is a no-op — it
    // must never wipe an existing selection.
    if (!summarizeMaskRGBA(imageData.data, imageData.width, imageData.height).bbox) return
    const hadMask = !!(state.maskSummary && state.maskSummary.bbox)
    // A live SAM object is part of the visible selection — keep it.
    if (state.liveMask) pushBaseOp('add', state.liveMask, 'click')
    pushBaseOp(negative ? 'sub' : 'add', imageData, kind)
    state.clicks = []
    state.box = null
    state.lasso = null
    state.textCandidates = []
    clearLive()
    state.manual = state.baseMask ? { kind, ...geometry } : null
    recomposeMask()
    state.score = 0
    state.showRaw = false
    bumpRevision()
    logCommit(state.revision, 'manual')
    const summary = state.maskSummary || { coverage: 0, bbox: null }
    const op = !hadMask ? 'set' : negative ? 'subtract' : 'add'
    console.log('[seglab][ui] manual-mask', { kind, op, revision: state.revision, coverage: summary.coverage })
    setStatus(summary.bbox
        ? `${kind} ${op === 'set' ? 'mask' : op} · ${(summary.coverage * 100).toFixed(1)}% of frame`
        : 'Selection cleared')
    renderOverlay()
    refreshButtons()
}

const brushRadius = () => Math.max(10, Math.round(Math.min(els.view.width, els.view.height) * 0.025))

// Strokes stay in a native canvas while the pointer is down. The former
// per-pixel JS loops and full-size mask/ring rebuilds on every pointermove
// starved the main thread on large interaction frames. Canvas compositing is
// incremental, and we read back exactly once when the stroke is committed.
const createBrush = () => {
    const canvas = document.createElement('canvas')
    canvas.width = els.view.width
    canvas.height = els.view.height
    const ctx = canvas.getContext('2d')
    if (state.mask) {
        // Segmentation masks may be opaque black outside the selection. A
        // brush preview needs transparency there, so copy luma → alpha once
        // when a stroke begins rather than doing that work on every move.
        const base = new ImageData(canvas.width, canvas.height)
        for (let i = 0; i < base.data.length; i += 4) {
            base.data[i] = 255; base.data[i + 1] = 255; base.data[i + 2] = 255
            base.data[i + 3] = state.mask.data[i]
        }
        ctx.putImageData(base, 0, 0)
    }
    return { canvas, ctx }
}

const paintBrush = (brush, from, to, erase) => {
    if (!brush) return
    const { ctx } = brush
    const radius = brushRadius()
    ctx.save()
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
    ctx.fillStyle = '#fff'
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = radius * 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    if (Math.hypot(to[0] - from[0], to[1] - from[1]) < 0.25) {
        ctx.beginPath()
        ctx.arc(from[0], from[1], radius, 0, Math.PI * 2)
        ctx.fill()
    } else {
        ctx.beginPath()
        ctx.moveTo(from[0], from[1])
        ctx.lineTo(to[0], to[1])
        ctx.stroke()
    }
    ctx.restore()
}

const startBrush = (point, erase) => {
    state.brush = createBrush()
    state.clicks = []
    state.box = null
    state.lasso = null
    state.textCandidates = []
    state.maskRaw = null
    state.showRaw = false
    state.manual = { kind: 'brush' }
    paintBrush(state.brush, point, point, erase)
}

const commitBrushMask = () => {
    const brush = state.brush
    if (!brush) return
    const { width, height } = brush.canvas
    // This is the only readback for a whole stroke. It deliberately avoids
    // applyMaskCombine: the brush canvas was initialized from the current mask
    // and already represents the final add/erase result.
    const imageData = brush.ctx.getImageData(0, 0, width, height)
    const summary = summarizeMaskRGBA(imageData.data, width, height)
    state.brush = null
    state.clicks = []
    state.box = null
    state.lasso = null
    state.textCandidates = []
    // The stroke canvas was seeded from the composed mask, so it already IS
    // the final whole-selection result: flatten the stack to this one op.
    state.baseOps = []
    state.baseFloor = null
    state.baseMask = null
    clearLive()
    if (summary.bbox) pushBaseOp('add', imageData, 'brush')
    state.manual = summary.bbox ? { kind: 'brush' } : null
    recomposeMask()
    state.score = 0
    state.showRaw = false
    bumpRevision()
    logCommit(state.revision, 'manual')
    setStatus(summary.bbox
        ? `brush mask · ${(summary.coverage * 100).toFixed(1)}% of frame`
        : 'Selection cleared')
    renderOverlay()
    refreshButtons()
}

const magicMask = (x, y, tolerance) => {
    const { width: w, height: h } = els.view
    const pixels = els.view.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    const seed = (Math.floor(y) * w + Math.floor(x)) * 4
    const sr = pixels[seed]; const sg = pixels[seed + 1]; const sb = pixels[seed + 2]
    const out = new Uint8ClampedArray(w * h * 4)
    const seen = new Uint8Array(w * h)
    const queue = new Int32Array(w * h)
    let head = 0
    let tail = 1
    queue[0] = Math.floor(y) * w + Math.floor(x)
    seen[queue[0]] = 1
    const limit = tolerance * tolerance * 3
    while (head < tail) {
        const p = queue[head]
        head += 1
        const i = p * 4
        const dr = pixels[i] - sr; const dg = pixels[i + 1] - sg; const db = pixels[i + 2] - sb
        if (dr * dr + dg * dg + db * db > limit) continue
        out[i] = 255; out[i + 1] = 255; out[i + 2] = 255; out[i + 3] = 255
        const px = p % w
        if (px > 0 && !seen[p - 1]) { seen[p - 1] = 1; queue[tail++] = p - 1 }
        if (px < w - 1 && !seen[p + 1]) { seen[p + 1] = 1; queue[tail++] = p + 1 }
        if (p >= w && !seen[p - w]) { seen[p - w] = 1; queue[tail++] = p - w }
        if (p < w * (h - 1) && !seen[p + w]) { seen[p + w] = 1; queue[tail++] = p + w }
    }
    return new ImageData(out, w, h)
}

const colorRangeMask = (x, y, tolerance) => {
    const { width: w, height: h } = els.view
    const pixels = els.view.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    const seed = (Math.floor(y) * w + Math.floor(x)) * 4
    const sr = pixels[seed]; const sg = pixels[seed + 1]; const sb = pixels[seed + 2]
    const out = new Uint8ClampedArray(pixels.length)
    const limit = tolerance * tolerance * 3
    for (let i = 0; i < pixels.length; i += 4) {
        const dr = pixels[i] - sr; const dg = pixels[i + 1] - sg; const db = pixels[i + 2] - sb
        if (dr * dr + dg * dg + db * db <= limit) {
            out[i] = 255; out[i + 1] = 255; out[i + 2] = 255; out[i + 3] = 255
        }
    }
    return new ImageData(out, w, h)
}

const finishPolygon = () => {
    const poly = state.polygonDraft
    state.polygonDraft = []
    if (poly.length >= 3) commitManualMask('polygon', polygonMask(poly), { poly }, state.sign === 0)
    renderOverlay()
}

const toCanvas = (e) => {
    const rect = els.overlay.getBoundingClientRect()
    return [
        Math.min(els.overlay.width, Math.max(0, (e.clientX - rect.left) * (els.overlay.width / rect.width))),
        Math.min(els.overlay.height, Math.max(0, (e.clientY - rect.top) * (els.overlay.height / rect.height))),
    ]
}

// Exclude when the pointer says so (right/Alt-click) or the sign toggle is set
// to exclude — the same gesture that carves out a click prompt.
const eventNegative = (e) => e.button === 2 || e.altKey || state.sign === 0

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
    } else if (state.mode === 'region') {
        state.drag = { kind: 'region', points: [[x, y]], negative: e.button === 2 || e.altKey }
    } else if (state.mode === 'rect' || state.mode === 'ellipse') {
        state.drag = { kind: state.mode, start: [x, y], now: [x, y], negative: e.button === 2 || e.altKey }
    } else if (state.mode === 'polygon') {
        const last = state.polygonDraft[state.polygonDraft.length - 1]
        if (!last || Math.hypot(x - last[0], y - last[1]) > 3) state.polygonDraft.push([x, y])
    } else if (state.mode === 'magic') {
        commitManualMask('magic', magicMask(x, y, state.wandTolerance), { seed: [x, y] }, eventNegative(e))
    } else if (state.mode === 'color') {
        commitManualMask('color', colorRangeMask(x, y, state.wandTolerance), { seed: [x, y] }, eventNegative(e))
    } else if (state.mode === 'brush') {
        const erase = e.button === 2 || e.altKey
        startBrush([x, y], erase)
        state.drag = { kind: 'brush', last: [x, y], erase }
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
    if (state.drag.kind === 'box' || state.drag.kind === 'rect' || state.drag.kind === 'ellipse') {
        state.drag.now = [x, y]
    } else if (state.drag.kind === 'lasso' || state.drag.kind === 'region') {
        const pts = state.drag.points
        const [lx, ly] = pts[pts.length - 1]
        if (Math.hypot(x - lx, y - ly) > 3) pts.push([x, y])
    } else if (state.drag.kind === 'brush') {
        // Coalesced pointer samples retain a smooth stroke without scheduling
        // a render or allocating a mask per hardware sample.
        const samples = e.getCoalescedEvents?.() || [e]
        for (const sample of samples) {
            const point = toCanvas(sample)
            paintBrush(state.brush, state.drag.last, point, state.drag.erase)
            state.drag.last = point
        }
    }
    renderOverlay()
})

els.overlay.addEventListener('pointerup', (e) => {
    const drag = state.drag
    state.drag = null
    if (!drag || !state.hasImage) return
    const [x, y] = toCanvas(e)

    if (drag.kind === 'tap' && !drag.moved) {
        applyClickPrompt(x, y, drag.negative || state.sign === 0 ? 0 : 1)
    } else if (drag.kind === 'box') {
        const [sx, sy] = drag.start
        // Discard degenerate boxes (a stray click instead of a drag).
        if (Math.abs(x - sx) > 8 && Math.abs(y - sy) > 8) {
            state.manual = null
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
            state.manual = null
            state.clicks = []
            state.box = null
            bumpRevision()
            scheduleRun()
        }
    } else if (drag.kind === 'region' && drag.points.length >= 3) {
        commitManualMask('region', polygonMask(drag.points), { poly: drag.points }, drag.negative || state.sign === 0)
    } else if (drag.kind === 'rect' || drag.kind === 'ellipse') {
        const [sx, sy] = drag.start
        const box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]
        if (box[2] - box[0] > 8 && box[3] - box[1] > 8) {
            const makeMask = drag.kind === 'rect' ? rectMask : ellipseMask
            commitManualMask(drag.kind, makeMask(box), { box }, drag.negative || state.sign === 0)
        }
    } else if (drag.kind === 'brush') {
        commitBrushMask()
    }
    renderOverlay()
    refreshButtons()
})

els.overlay.addEventListener('dblclick', (e) => {
    if (state.mode !== 'polygon' || state.polygonDraft.length < 3) return
    e.preventDefault()
    finishPolygon()
})

/* ─── Segmentation pipeline ──────────────────────────────────────────────── */

/** One Click-mode tap (pointerup and the headless clickAt hook). */
function applyClickPrompt(x, y, label) {
    state.manual = null
    if (label === 1) {
        if (state.liveMask && !pointInMask(state.liveMask, x, y)) {
            // A click OUTSIDE the live object selects a NEW object: commit the
            // finished one so it can never be unselected by later clicks.
            pushBaseOp('add', state.liveMask, 'click')
            clearLive()
            state.clicks = [[x, y, 1]]
        } else {
            state.clicks.push([x, y, 1])
        }
        bumpRevision()
        scheduleRun()
        return
    }
    // Exclude: a committed region (not the live object) carves out whole;
    // anything near the live object is a negative refinement point.
    const insideLive = state.liveMask && pointInMask(state.liveMask, x, y)
    if (!insideLive && state.baseMask && pointInMask(state.baseMask, x, y)) {
        void runObjectSubtract(x, y)
        return
    }
    if (state.clicks.length || state.box || state.lasso) {
        state.clicks.push([x, y, 0])
        bumpRevision()
        scheduleRun()
        return
    }
    setStatus('Nothing to exclude here — click an object first')
}

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
    console.log('[seglab][ui] selection-start', { revision, clicks: clicks.length, box: Boolean(box), mode: state.mode })
    setStatus(state.encodePending
        ? 'Preparing this photo for selection…'
        : (clientState.ready ? 'Selecting…' : 'Selecting… (first run loads the model)'))
    try {
        // The engine runs the whole post pipeline (lasso clamp → hygiene →
        // edge refinement) off-thread and returns both masks.
        const res = await segment(els.view, {
            clicks,
            box,
            clampPoly: state.lasso?.poly || null,
            clampMargin: state.lasso?.margin || 0,
            revision,
            budget: BUDGET,
        })
        if (res.stale) {
            console.log('[seglab][ui] selection-stale', { revision })
            logCommit(revision, 'stale')
            return
        }
        if (revision !== state.revision) {
            console.log('[seglab][ui] selection-superseded', { revision, current: state.revision })
            logCommit(revision, 'superseded')
            return
        }

        logCommit(revision, res.usable ? 'committed' : 'unusable')
        console.log('[seglab][ui] selection-result', { revision, usable: res.usable, lane: res.lane, score: res.score, encoded: res.encoded })
        if (res.encoded) state.encodePending = false // paid; the cache serves the rest
        if (!res.usable) {
            // Only the live object misses; committed regions survive.
            clearLive()
            recomposeMask()
            setStatus(`No selection — ${res.reason}. Try another click.`)
        } else {
            state.liveMask = res.imageData
            state.liveRaw = res.rawImageData
            state.liveSummary = res.summary
            recomposeMask()
            state.score = res.score
            const coverage = state.maskSummary ? state.maskSummary.coverage : res.summary.coverage
            setStatus(`Selected — ${res.lane} · confidence ${res.score.toFixed(2)} · ${(coverage * 100).toFixed(1)}% of frame${res.encoded ? '' : ' · cached'}`)
        }
        renderOverlay()
        refreshButtons()
        // Optional wasm cleanup on the committed one-channel mask (lazy-loads
        // on first use; skipped under pressure ≥ 2; failure keeps this mask).
        if (res.usable) await maybeCvRefine(revision, clicks)
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

/** Exclude-click on a committed region: one ephemeral single-point decode
 *  identifies that object, then it subtracts from the base stack. Never
 *  becomes the live object; refinement/escalation don't apply. */
async function runObjectSubtract(x, y) {
    if (state.running) { setStatus('Busy — try the exclude click again in a moment'); return }
    bumpRevision()
    const revision = state.revision
    state.running = true
    setStatus('Removing object from selection…')
    try {
        const res = await segment(els.view, { clicks: [[x, y, 1]], revision, budget: BUDGET })
        if (res.stale || revision !== state.revision) { logCommit(revision, 'stale'); return }
        if (!res.usable) {
            logCommit(revision, 'unusable')
            setStatus(`Nothing removed — ${res.reason}`)
            return
        }
        pushBaseOp('sub', res.imageData, 'click-exclude', { dilate: 2 })
        recomposeMask()
        logCommit(revision, 'committed')
        setStatus(state.maskSummary?.bbox
            ? `Removed object · ${(state.maskSummary.coverage * 100).toFixed(1)}% of frame`
            : 'Selection cleared')
        renderOverlay()
        refreshButtons()
    } catch (err) {
        logCommit(revision, 'error')
        if (revision === state.revision) {
            console.error('[seglab] exclude failed:', err)
            setStatus(`Exclude failed: ${err?.message}`)
        }
    } finally {
        state.running = false
        if (state.runQueued) { state.runQueued = false; scheduleRun() }
    }
}

/* ─── Wasm mask cleanup (cv-refine) ──────────────────────────────────────── */

// One-channel contract: the mask leaves as W*H bytes (transferred), returns
// as W*H bytes, and only the final committed ImageData expands to RGBA.
async function maybeCvRefine(revision, clicks) {
    if (!state.liveMask || !cvRefineAvailable()) return
    if (BUDGET.cvRefine === false || (BUDGET.pressureLevel || 0) >= 2) return
    const { width, height } = state.liveMask
    if (Math.max(width, height) > 1024) return
    const src = state.liveMask.data
    const alpha = new Uint8Array(width * height)
    for (let p = 0; p < alpha.length; p += 1) alpha[p] = src[p * 4]
    const seeds = (clicks || []).filter((c) => c[2]).map((c) => [Math.round(c[0]), Math.round(c[1])])
    const refined = await refineAlpha({
        alpha, // transferred — detached after this call
        width,
        height,
        seeds,
        options: { minArea: 24, openRadius: 0, closeRadius: 0 },
        revision,
        budget: BUDGET,
    })
    if (!refined || revision !== state.revision || !state.liveMask) return
    const out = new ImageData(width, height)
    const d = out.data
    for (let p = 0, i = 0; p < refined.length; p += 1, i += 4) {
        const v = refined[p]
        d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255
    }
    state.liveMask = out
    state.liveSummary = summarizeMaskRGBA(d, width, height)
    recomposeMask()
    renderOverlay()
}

/* ─── Crop escalation (M3) ───────────────────────────────────────────────── */

// A small object loses detail in the ≤1024 proxy. When the committed mask is
// tiny AND the original out-resolves the proxy, re-decode ONE native crop,
// merge it back, and cache it for HD export. Auto on Std/Pro (autoEscalate);
// ?escalate=0 disables. No native headroom (proxy == original) ⇒ nothing to gain.
const shouldEscalate = (summary) => {
    if (!BUDGET.autoEscalate || !hasOriginal() || !summary?.bbox) return false
    const tf = getTransform()
    if (!tf) return false
    // Escalation crops decode from the working copy when one exists (bounded
    // hosts), else from the original — headroom and cost gate on that source.
    const wActive = !!tf.workingActive
    const srcW = wActive ? tf.workingW : tf.originalW
    const srcH = wActive ? tf.workingH : tf.originalH
    if (Math.max(srcW, srcH) / Math.max(tf.proxyW, tf.proxyH) < 1.2) return false
    // Without a working copy, Safari cannot region-decode a JPEG: a RAW
    // preview or a very large compressed upload would turn this optional
    // convenience into another full-frame decode. Keep selection on the
    // bounded proxy there; export still relinks at native detail.
    if (!wActive && (tf.sourceWasRaw || (tf.sourceBytes || 0) >= 24 * 1024 * 1024)) return false
    if ((srcW * srcH) / 1e6 > (BUDGET.escalateMaxMP || 24)) return false
    const [minX, minY, maxX, maxY] = summary.bbox
    const diag = Math.hypot(maxX - minX, maxY - minY)
    return diag < Math.hypot(els.view.width, els.view.height) * 0.15
}

async function maybeEscalate(revision) {
    // The tiny-bbox test must see the live object, not the composed union.
    if (!state.liveMask || !shouldEscalate(state.liveSummary)) return
    // One escalation per SETTLED selection: skip while more input is pending,
    // and let a click-burst supersede us before the heavy crop pipeline starts.
    if (state.runQueued || debounceArmed) return
    await new Promise((r) => setTimeout(r, 250))
    if (revision !== state.revision || state.runQueued || debounceArmed) return
    // Deliberately NOT the modal veil: the mask is already on screen and usable,
    // and this only sharpens it. Blocking here would claim the app is unusable
    // when it isn't. The status line reports the work instead.
    const previous = els.status.textContent
    setStatus('Sharpening detail from the full-resolution photo…')
    try {
        const crop = await escalateCrop(state.liveMask, currentPrompts(), { budget: BUDGET, revision })
        if (!crop || revision !== state.revision) { setStatus(previous); return } // rejected or superseded
        mergeCropIntoProxy(crop)
        renderOverlay()
        setStatus(`${previous} · native detail`)
    } catch (err) {
        setStatus(previous)
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

    const merged = new Uint8ClampedArray(state.liveMask.data)
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
    state.liveMask = new ImageData(merged, W, H)
    state.liveSummary = summarizeMaskRGBA(merged, W, H)
    recomposeMask()
}

/* ─── Text select ────────────────────────────────────────────────────────── */

let detectTimer = null

/** Set expectations before the user types: the detector is a one-time download
 *  and the first search is the slow one. Local state only — no network. */
async function hintTextSearch() {
    if (!state.hasImage) return
    const accelerated = acceleratedDetectorAvailable && BUDGET.detectorWebGPU === true && BUDGET.gpuTier === 'accelerated' && BUDGET.forceWasm !== true
    const cached = await detectorCached({ accelerated })
    if (state.mode !== 'text') return // left Text while checking
    setStatus(cached
        ? 'Text search ready'
        : `Text search ready — the first search downloads the detector (~${detectorDownloadMB({ accelerated })} MB), then it's cached`)
}

async function runDetect(phrase) {
    if (!state.hasImage || !phrase.trim()) {
        state.textCandidates = []
        els.selectall.hidden = true
        renderOverlay()
        return
    }
    bumpRevision()
    const revision = state.revision
    const accelerated = acceleratedDetectorAvailable && BUDGET.detectorWebGPU === true && BUDGET.gpuTier === 'accelerated' && BUDGET.forceWasm !== true
    const detectorMB = detectorDownloadMB({ accelerated })
    setStatus(await detectorCached({ accelerated })
        ? `Looking for “${phrase.trim()}”…`
        : `Downloading the detector (~${detectorMB} MB), then looking for “${phrase.trim()}”…`)
    if (revision !== state.revision) return // superseded while checking the cache
    try {
        const res = await detectCandidates(phrase)
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
        setStatus(BUDGET.profile === 'lite' && /load|memory|alloc|abort/i.test(String(err?.message))
            ? 'Text selection is unavailable on this device’s safe memory profile.'
            : `Text detection failed: ${err?.message}`)
    }
}

/** A detector box → a box prompt through the normal pipeline (mask + HD export). */
const selectCandidate = (i) => {
    const c = state.textCandidates[i]
    if (!c) return
    state.box = c.box.slice()
    state.clicks = []
    state.lasso = null
    state.manual = null
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
    state.manual = null
    els.selectall.hidden = true
    setStatus(`Selecting ${boxes.length} objects…`)
    try {
        let union = null
        for (const box of boxes) {
            const res = await segment(els.view, { box, revision, budget: BUDGET })
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
        // The multi-instance union commits as one op; Z removes it whole.
        pushBaseOp('add', union, 'text')
        clearLive()
        recomposeMask()
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

// Layers rebuild only when the committed mask object changes. A brush stroke
// draws its own lightweight canvas preview, so pointermove never allocates a
// full-size overlay or runs an 8-pass outline dilation on the UI thread.
let layerCache = { mask: null, fill: null }

/** Colorize the white-on-black mask; cached per committed mask object. */
const getMaskLayers = (mask) => {
    if (layerCache.mask === mask) return layerCache
    const { width, height, data: src } = mask
    // luma → alpha + accent tint in one pass — no intermediate canvas/readback.
    const img = new ImageData(width, height)
    const d = img.data
    for (let i = 0; i < d.length; i += 4) {
        d[i] = 53; d[i + 1] = 224; d[i + 2] = 194
        d[i + 3] = src[i]
    }
    const fill = new OffscreenCanvas(width, height)
    fill.getContext('2d').putImageData(img, 0, 0)

    layerCache = { mask, fill }
    return layerCache
}

// Coalesce paint bursts (pointermove) into one paint per frame.
let overlayScheduled = false
function renderOverlay() {
    if (overlayScheduled) return
    overlayScheduled = true
    requestAnimationFrame(() => { overlayScheduled = false; paintOverlay() })
}

function paintOverlay() {
    const ctx = overlayCtx
    const { width, height } = els.overlay
    ctx.clearRect(0, 0, width, height)

    const brushPreview = state.drag?.kind === 'brush' ? state.brush?.canvas : null
    const shownMask = brushPreview ? null : (state.showRaw ? state.maskRaw : state.mask)
    if (brushPreview) {
        // The brush canvas has transparent unselected pixels. Tint it in-place
        // on the overlay; no ImageData conversion is needed while drawing.
        ctx.save()
        ctx.globalAlpha = 0.32
        ctx.drawImage(brushPreview, 0, 0)
        ctx.globalAlpha = 1
        ctx.globalCompositeOperation = 'source-in'
        ctx.fillStyle = ACCENT
        ctx.fillRect(0, 0, width, height)
        ctx.restore()
    } else if (shownMask) {
        const { fill } = getMaskLayers(shownMask)
        ctx.globalAlpha = 0.32
        ctx.drawImage(fill, 0, 0)
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

    if (state.manual?.poly?.length >= 3) {
        const { poly, kind } = state.manual
        ctx.beginPath()
        ctx.moveTo(poly[0][0], poly[0][1])
        for (const [px, py] of poly.slice(1)) ctx.lineTo(px, py)
        ctx.closePath()
        ctx.strokeStyle = kind === 'region' ? 'rgba(255,194,94,0.9)' : 'rgba(173,130,255,0.9)'
        ctx.lineWidth = 2
        ctx.stroke()
    }
    if (state.manual?.box) {
        const [x0, y0, x1, y1] = state.manual.box
        ctx.strokeStyle = 'rgba(255,194,94,0.9)'
        ctx.lineWidth = 2
        if (state.manual.kind === 'ellipse') {
            ctx.beginPath()
            ctx.ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0, 0, Math.PI * 2)
            ctx.stroke()
        } else ctx.strokeRect(x0, y0, x1 - x0, y1 - y0)
    }
    if (state.manual?.seed) {
        ctx.beginPath()
        ctx.arc(state.manual.seed[0], state.manual.seed[1], markerR * 1.7, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(255,194,94,0.95)'
        ctx.lineWidth = 2
        ctx.stroke()
    }
    if (state.polygonDraft.length) {
        const [first, ...rest] = state.polygonDraft
        ctx.beginPath()
        ctx.moveTo(first[0], first[1])
        for (const [px, py] of rest) ctx.lineTo(px, py)
        ctx.setLineDash([5, 4])
        ctx.strokeStyle = 'rgba(173,130,255,0.95)'
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.setLineDash([])
    }

    // In-progress drags.
    if (state.drag?.kind === 'box' || state.drag?.kind === 'rect' || state.drag?.kind === 'ellipse') {
        const [sx, sy] = state.drag.start
        const [nx, ny] = state.drag.now
        ctx.setLineDash([7, 5])
        ctx.strokeStyle = state.drag.kind === 'box' ? ACCENT : 'rgba(255,194,94,0.95)'
        ctx.lineWidth = 1.75
        const x0 = Math.min(sx, nx); const y0 = Math.min(sy, ny)
        const w = Math.abs(nx - sx); const h = Math.abs(ny - sy)
        if (state.drag.kind === 'ellipse') {
            ctx.beginPath()
            ctx.ellipse(x0 + w / 2, y0 + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2)
            ctx.stroke()
        } else ctx.strokeRect(x0, y0, w, h)
        ctx.setLineDash([])
    }
    if ((state.drag?.kind === 'lasso' || state.drag?.kind === 'region') && state.drag.points.length > 1) {
        ctx.beginPath()
        ctx.moveTo(state.drag.points[0][0], state.drag.points[0][1])
        for (const [px, py] of state.drag.points.slice(1)) ctx.lineTo(px, py)
        ctx.strokeStyle = state.drag.kind === 'lasso' ? 'rgba(90,160,255,0.95)' : 'rgba(255,194,94,0.95)'
        ctx.lineWidth = 3
        ctx.lineJoin = 'round'
        ctx.lineCap = 'round'
        ctx.stroke()
        ctx.fillStyle = state.drag.kind === 'lasso' ? 'rgba(90,160,255,0.12)' : 'rgba(255,194,94,0.12)'
        ctx.fill()
    }
    if (state.drag?.kind === 'brush') {
        ctx.beginPath()
        ctx.arc(state.drag.last[0], state.drag.last[1], brushRadius(), 0, Math.PI * 2)
        ctx.strokeStyle = state.drag.erase ? NEG_COLOR : '#ffc25e'
        ctx.lineWidth = 2
        ctx.stroke()
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
    // Re-decodes the original at native resolution — the heaviest pixel work
    // in the app, and the user is waiting on a file, not the canvas.
    const epoch = showPrep(wasNative ? 'Rebuilding the cutout at full resolution…' : 'Building the cutout…')
    try {
        // Detector and wasm-refine workers are unrelated to a cutout export
        // and hold memory. Release both before the bounded export canvases.
        await relievePressure(1)
        disposeCvRefine()
        let out = null
        if (wasNative) {
            out = await exportCutoutBlob(state.mask, currentPrompts(), {
                budget: BUDGET,
                revision: state.revision,
                preserveShape: Boolean(state.manual),
            })
        }
        if (!out) out = await proxyCutoutBlob() // fallback: proxy-res
        const a = document.createElement('a')
        a.href = URL.createObjectURL(out.blob)
        a.download = 'seglab-cutout.png'
        a.click()
        setTimeout(() => URL.revokeObjectURL(a.href), 5000)
        const tf = getTransform()
        const reduced = wasNative && tf && (out.width < tf.originalW || out.height < tf.originalH)
        setStatus(`Exported ${out.width}×${out.height} cutout${out.decoded ? ' · HD re-decode' : ''}${reduced ? ` · reduced from ${tf.originalW}×${tf.originalH} for this device’s safe memory profile` : ''}`)
    } catch (err) {
        console.error('[seglab] export failed:', err)
        setStatus(`Export failed: ${err?.message}`)
    } finally {
        hidePrep(epoch)
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
    loadDemo: async (longSide) => { await queueImage(buildDemoScene(longSide)) },
    // Demo scene through the real UPLOAD path (Blob custody, proxy decode,
    // working copy) instead of the resident-drawable demo path.
    loadDemoBlob: async (longSide) => {
        const scene = buildDemoScene(longSide)
        const blob = await new Promise((r) => scene.toBlob(r, 'image/jpeg', 0.9))
        await queueImage(new File([blob], 'demo.jpg', { type: 'image/jpeg' }), { sourceBytes: blob.size })
    },
    reset: () => clearPrompts(),
    revision: () => state.revision,
    commitLog,
    // Drive the LibRaw develop fallback directly (verify.mjs). Fetches a RAW at
    // `url`, develops the sensor on-device (no MP cap for the test), and returns
    // the developed JPEG's dimensions + a decode check — proving the whole
    // File → worker → wasm → libjpeg → Blob chain end to end.
    developRawUrl: async (url) => {
        const bytes = await (await fetch(url)).arrayBuffer()
        const file = new File([bytes], 'fixture.raw', { type: 'application/octet-stream' })
        const out = await developRaw(file, { budget: { ...BUDGET, rawDevelop: true, rawDevelopMaxMP: 0 } })
        if (!out) return { ok: false }
        const bmp = await createImageBitmap(out.blob)
        const res = { ok: true, w: out.width, h: out.height, jpegBytes: out.blob.size, decodedW: bmp.width, decodedH: bmp.height }
        bmp.close()
        return res
    },
    // Boot capability probe result { webgpu, fallback, f16, deviceMemoryGB,
    // profile } — awaits the probe so it is never null.
    capability: async () => { await bootProbe; return capability },
    // Current resolved budget (including a dynamic Phosmith resource hint).
    resourceBudget: async () => { await bootProbe; return { ...BUDGET } },
    // Native host test/integration hook. Production hosts normally inject the
    // global hint then dispatch `phosmithresourceschange` instead.
    setPhosmithResources: (resources) => applyPhosmithResources(resources),
    // Current image mapping, including whether a real proxy is active. Useful
    // to inspect the native-frame path without relying on UI status strings.
    imageTransform: () => getTransform(),
    // Drive the memory-pressure ladder; returns what was freed.
    relievePressure: (level) => shedMemory(level),
    // Engine residency snapshot (worker): { cachedImages, lane, profile, … }.
    engineState: () => engineState(),
    // Heavy-job queue state + recent telemetry (overlap/ordering gates).
    queueState: () => getHeavyQueueState(),
    queueLog: () => getHeavyQueueLog(),
    // Direct cv-refine lane access for the wasm gates. `alpha` arrives as a
    // plain array; returns { alpha: number[]|null } (null = unavailable/failed).
    cvRefine: async ({ alpha, width, height, seeds = [], options = {} }) => {
        const out = await refineAlpha({
            alpha: Uint8Array.from(alpha), width, height, seeds, options,
            revision: null, budget: BUDGET,
        })
        return { alpha: out ? Array.from(out) : null, available: cvRefineAvailable() }
    },
    // Developer "Release memory" action (debug only, not in normal UI).
    releaseMemory: async () => {
        clearHeavyQueue()
        cancelBefore(state.revision + 1)
        disposeCvRefine()
        layerCache = { mask: null, fill: null }
        if (warmStarted) await releaseDocument()
        return shedMemory(3, { announce: false })
    },
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
        hasImage: state.hasImage,
        maskSummary: state.maskSummary,
        score: state.score,
        clicks: state.clicks.length,
        baseOps: state.baseOps.length,
        revision: state.revision,
        candidates: state.textCandidates.length,
        manual: state.manual?.kind || null,
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
        const res = await buildCutout(state.mask, currentPrompts(), {
            budget: BUDGET,
            revision: state.revision,
            preserveShape: Boolean(state.manual),
        })
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
        applyClickPrompt(x, y, negative ? 0 : 1) // the real pointerup path
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
    manualRegionCircle: (cx, cy, r, n = 28, negative = false) => {
        const poly = []
        for (let i = 0; i < n; i += 1) {
            const a = (i / n) * Math.PI * 2
            poly.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r])
        }
        commitManualMask('region', polygonMask(poly), { poly }, negative)
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    manualRect: (x0, y0, x1, y1, negative = false) => {
        const box = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]
        commitManualMask('rect', rectMask(box), { box }, negative)
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    manualEllipse: (x0, y0, x1, y1, negative = false) => {
        const box = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]
        commitManualMask('ellipse', ellipseMask(box), { box }, negative)
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    magicAt: (x, y, tolerance = state.wandTolerance, negative = false) => {
        commitManualMask('magic', magicMask(x, y, tolerance), { seed: [x, y] }, negative)
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    colorAt: (x, y, tolerance = state.wandTolerance, negative = false) => {
        commitManualMask('color', colorRangeMask(x, y, tolerance), { seed: [x, y] }, negative)
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    // Drives the same native-canvas brush path as a pointer stroke. This keeps
    // the responsiveness regression covered without relying on synthetic DOM
    // pointer coalescing in headless Chromium.
    brushStroke: async (points, erase = false) => {
        if (!state.hasImage || !Array.isArray(points) || points.length === 0) return null
        const [first, ...rest] = points
        startBrush(first, erase)
        state.drag = { kind: 'brush', last: first, erase }
        for (const point of rest) {
            paintBrush(state.brush, state.drag.last, point, erase)
            state.drag.last = point
        }
        state.drag = null
        commitBrushMask()
        await new Promise((resolve) => requestAnimationFrame(resolve))
        return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
    },
    // Run the resource-gated text detector; returns candidate boxes (for the
    // WARN-level detector gate — headless model quality is not asserted hard).
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
    // Crash/power-cut persistence (verify.mjs): a real cut fires no unload, so
    // the gate reloads without one and asserts the work came back.
    restoreSession,
    clearSession,
    sessionSaved: async () => {
        const s = await loadSession()
        return s ? { mask: !!s.mask, w: s.w, h: s.h } : null
    },
}
window.__seglabReady = true

// Reopen the last session on load. A power cut leaves the record behind; this
// is what brings the work back. Never blocks readiness — a failed restore just
// leaves the dropzone up. __seglabNoRestore lets verify's synthetic phases opt
// out, so a session from one phase cannot import itself into the next.
if (!window.__seglabNoRestore) restoreSession().catch(() => {})

/* ─── Service Worker registration (model cache) ─────────────────────────── */

// Register the SW as early as possible so it's controlling the page before any
// model fetches start. On second+ visits the SW intercepts from the cache —
// eliminating repeated SlimSAM downloads entirely.
// Only runs in secure contexts (HTTPS or localhost); file:// is silently skipped.
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js', { scope: './' })
        .then((reg) => {
            console.log('[seglab] service worker registered (model cache scope:', reg.scope, ')')
        })
        .catch((err) => {
            // Non-fatal: the app works without the SW, just re-downloads on every visit.
            console.warn('[seglab] service worker registration failed (non-fatal):', err?.message)
        })
}

/* ─── Boot ───────────────────────────────────────────────────────────────── */

setMode('click')
refreshChips()

// Deliberately NO model warm here: nothing downloads or compiles until the
// user has supplied an image and its bounded proxy is on screen.
bootProbe.catch(() => {})

console.log('[seglab] ready')

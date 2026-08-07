/**
 * engine — one contract for every selection mode
 * -------------------------------------------------
 * Sits on top of yoloe-engine (the single model) and select-core (the pure
 * query layer), and adds the shared post pipeline that makes the masks good.
 *
 * The whole flow, once per image:
 *   analyze  → YOLOE-26 forward → every instance in the frame (cached)
 * then, per interaction, with NO model in the loop:
 *   select   → click / box / lasso / text picks instances out of that set
 *   polish   → mask hygiene + guided-filter edge refinement, per instance
 *
 * Polish is where the quality comes from. YOLOE's prototype masks are fast
 * and coarse; re-deriving their boundary from the photo is what turns them
 * into something you would cut out and keep. It is model-agnostic, so it
 * would upgrade any future detector too.
 *
 * Refined instances are memoized on the analysis entry, so re-selecting an
 * object the user already touched costs nothing at all.
 */

import { cleanupMaskRGBA, makeMaskScratch, summarizeMaskRGBA } from './mask-core.js'
import { buildGuide, makeScratch, refineMaskEdges } from './edge-refine.js'
import { planeToRGBA } from './yoloe-core.js'
import {
    pickByPoint, pickByRegion, selectByQuery,
} from './select-core.js'
import * as yoloe from './yoloe-engine.js'
import * as governor from './memory-governor.js'

let eventSink = null
export const setEventSink = (fn) => { eventSink = fn }
const PROGRESS_THROTTLE_MS = 120
let lastProgressAt = 0
const emitEvent = (event) => {
    if (!eventSink) return
    if (event.type === 'progress' && event.detail?.status === 'progress') {
        const now = Date.now()
        if (now - lastProgressAt < PROGRESS_THROTTLE_MS) return
        lastProgressAt = now
    }
    try { eventSink(event) } catch { /* sink gone */ }
}

export const getEngineState = () => yoloe.getEngineState()

export const warm = async () => {
    const s = await yoloe.warm(emitEvent)
    emitEvent({ type: 'state' })
    return s
}

/* ─── Post pipeline ──────────────────────────────────────────────────────── */

const scratch = { edge: makeScratch(), mask: null, w: 0, h: 0 }
const maskScratchFor = (w, h) => {
    if (!scratch.mask || scratch.w !== w || scratch.h !== h) {
        scratch.mask = makeMaskScratch(w, h)
        scratch.w = w
        scratch.h = h
    }
    return scratch.mask
}

/**
 * Turn one raw instance into a polished, bbox-cropped soft mask.
 *
 * Work is confined to the instance's own box plus the refinement band, so a
 * dozen instances cost about as much as the pixels they actually cover — not
 * a dozen full-frame passes. The result is cached on the instance.
 */
const polish = (inst, entry) => {
    if (inst.refined) return inst.refined
    const pad = 16 // room for the refinement band to work outside the mask
    const x0 = Math.max(0, inst.x0 - pad)
    const y0 = Math.max(0, inst.y0 - pad)
    const x1 = Math.min(entry.w - 1, inst.x0 + inst.w - 1 + pad)
    const y1 = Math.min(entry.h - 1, inst.y0 + inst.h - 1 + pad)
    const w = x1 - x0 + 1
    const h = y1 - y0 + 1

    // A local RGBA tile, not a frame-sized buffer.
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i += 1) rgba[i * 4 + 3] = 255
    for (let y = 0; y < inst.h; y += 1) {
        const src = y * inst.w
        const dst = (inst.y0 - y0 + y) * w + (inst.x0 - x0)
        for (let x = 0; x < inst.w; x += 1) {
            if (!inst.plane[src + x]) continue
            const j = (dst + x) * 4
            rgba[j] = 255; rgba[j + 1] = 255; rgba[j + 2] = 255
        }
    }

    // Guide terms are per-IMAGE; slice the tile's view out of the full guide.
    entry.guide ??= buildGuide(entry.gray, entry.w, entry.h)
    const tileGray = new Float32Array(w * h)
    const tileMeanI = new Float32Array(w * h)
    const tileVarI = new Float32Array(w * h)
    for (let y = 0; y < h; y += 1) {
        const src = (y0 + y) * entry.w + x0
        const dst = y * w
        for (let x = 0; x < w; x += 1) {
            tileGray[dst + x] = entry.guide.gray[src + x]
            tileMeanI[dst + x] = entry.guide.meanI[src + x]
            tileVarI[dst + x] = entry.guide.varI[src + x]
        }
    }
    const tileGuide = { gray: tileGray, w, h, radius: entry.guide.radius, meanI: tileMeanI, varI: tileVarI }

    const hy = cleanupMaskRGBA(rgba, w, h, [[inst.x0 - x0 + inst.w / 2, inst.y0 - y0 + inst.h / 2]], maskScratchFor(w, h))
    refineMaskEdges(rgba, w, h, tileGuide, { scratch: scratch.edge, bbox: hy.bbox })

    inst.refined = { rgba, x0, y0, w, h }
    return inst.refined
}

/** Composite polished instances into one frame-sized white-on-black mask. */
const composite = (instances, entry) => {
    const { w, h } = entry
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < w * h; i += 1) rgba[i * 4 + 3] = 255
    for (const inst of instances) {
        const r = polish(inst, entry)
        for (let y = 0; y < r.h; y += 1) {
            const src = y * r.w
            const dst = (r.y0 + y) * w + r.x0
            for (let x = 0; x < r.w; x += 1) {
                const v = r.rgba[(src + x) * 4]
                if (!v) continue
                const j = (dst + x) * 4
                // Soft band pixels take the max so overlapping instances
                // never darken each other's edges.
                if (v > rgba[j]) { rgba[j] = v; rgba[j + 1] = v; rgba[j + 2] = v }
            }
        }
    }
    return rgba
}

/** The same composite without refinement — the UI's raw-vs-refined toggle. */
const compositeRaw = (instances, entry) => {
    const rgba = new Uint8ClampedArray(entry.w * entry.h * 4)
    for (let i = 0; i < entry.w * entry.h; i += 1) rgba[i * 4 + 3] = 255
    for (const inst of instances) planeToRGBA(inst, entry.w, entry.h, rgba)
    return rgba
}

/* ─── The one public call ────────────────────────────────────────────────── */

const describe = (inst) => ({
    label: inst.label,
    score: inst.score,
    box: inst.box,
    area: inst.area,
    bbox: [inst.x0, inst.y0, inst.x0 + inst.w - 1, inst.y0 + inst.h - 1],
})

/**
 * Select instances by any mode. One entry point, one result shape.
 *
 * @param {object} req
 * @param {string} req.imageKey
 * @param {ImageBitmap|OffscreenCanvas|HTMLCanvasElement} req.source
 * @param {'click'|'box'|'lasso'|'text'|'all'} req.mode
 * @param {[number,number]} [req.point]     click
 * @param {number[]} [req.box]              [x0,y0,x1,y1]
 * @param {Array<[number,number]>} [req.poly]  lasso
 * @param {string} [req.query]              text
 * @param {number[]} [req.exclude]          instance indices to subtract
 */
export const select = async (req) => {
    const t0 = Date.now()
    await governor.makeRoomFor('yoloe')
    const { entry, analyzed } = await yoloe.analyze({
        imageKey: req.imageKey,
        source: req.source,
        confThreshold: req.confThreshold ?? 0.25,
        emit: emitEvent,
    })
    const tAnalyzed = Date.now()

    let chosen = []
    let reason = null
    let concepts = []
    switch (req.mode) {
        case 'click': {
            const hit = pickByPoint(entry.instances, req.point[0], req.point[1])
            if (hit) chosen = [hit]
            else reason = 'nothing detected under that point'
            break
        }
        case 'box':
            chosen = pickByRegion(entry.instances, req.box, { wantsAll: true })
            if (!chosen.length) reason = 'nothing detected inside that box'
            break
        case 'lasso':
            chosen = pickByRegion(entry.instances, req.poly, { wantsAll: true })
            if (!chosen.length) reason = 'nothing detected inside that lasso'
            break
        case 'text': {
            const r = selectByQuery(entry.instances, req.query)
            chosen = r.instances
            reason = r.reason
            concepts = r.concepts
            if (!MODEL_HAS_LABELS()) {
                reason = 'no vocabulary shipped with this model — text search needs models/vocabulary.json'
            }
            break
        }
        case 'all':
            chosen = entry.instances
            if (!chosen.length) reason = 'nothing detected in this photo'
            break
        case 'indices':
            // An explicit instance list — how the UI replays a selection it
            // has already built up (add a click, drop a chip, undo) without
            // re-deriving it from the original interaction.
            chosen = (req.indices || [])
                .map((i) => entry.instances[i])
                .filter(Boolean)
            if (!chosen.length) reason = 'nothing selected'
            break
        default:
            throw new Error(`Unknown selection mode: ${req.mode}`)
    }

    if (Array.isArray(req.exclude) && req.exclude.length) {
        const drop = new Set(req.exclude)
        chosen = chosen.filter((inst) => !drop.has(entry.instances.indexOf(inst)))
    }

    const tSelected = Date.now()
    const rgba = chosen.length ? composite(chosen, entry) : null
    const rawRgba = chosen.length ? compositeRaw(chosen, entry) : null
    const tPolished = Date.now()

    return {
        rgba,
        rawRgba,
        width: entry.w,
        height: entry.h,
        mode: req.mode,
        query: req.query || null,
        concepts,
        instances: chosen.map(describe),
        indices: chosen.map((inst) => entry.instances.indexOf(inst)),
        detected: entry.instances.length,
        reason,
        device: yoloe.getEngineState().device,
        analyzed,
        // Timings that tell the truth about where the time went.
        analyzeMs: analyzed ? entry.analyzeMs : 0,
        maskMs: analyzed ? entry.maskMs : 0,
        selectMs: tSelected - tAnalyzed,
        postMs: tPolished - tSelected,
        totalMs: Date.now() - t0,
    }
}

const MODEL_HAS_LABELS = () => yoloe.MODEL.labels.length > 0

/** Everything the model found, for the UI's instance browser. */
export const listInstances = async (req) => {
    await governor.makeRoomFor('yoloe')
    const { entry } = await yoloe.analyze({ imageKey: req.imageKey, source: req.source, emit: emitEvent })
    return {
        width: entry.w,
        height: entry.h,
        instances: entry.instances.map(describe),
        labels: [...new Set(entry.instances.map((i) => i.label))].sort(),
    }
}

export { summarizeMaskRGBA }

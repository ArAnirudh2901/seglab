/**
 * model-prefetch — pull a lane's weights into the model cache before the lane
 * is asked to run, without ever holding the machine.
 *
 * The mask lane prefetches inside the SAM host (it owns that queue). This is
 * the TAB-side equivalent, for the text lane's detector: 126 MB of YOLOE plus
 * the 4-bit CLIP text encoder, otherwise paid in full on the first search.
 *
 * Rules, same as the host's: rides the heavy-job queue so nothing overlaps,
 * lowest rank, one file per idle window, and aborts the moment real work is
 * enqueued. Completed files are cached per URL, so an aborted attempt only
 * repeats the file it was on.
 *
 * NOT started at boot. A session that never opens text search must not pay for
 * it; entering text mode is the intent signal (app: setMode).
 */

import { enqueueHeavy, onHeavyActivity, STALE } from './heavy-job-queue.js'

// Same specifiers the workers resolve, so the cache key matches byte for byte.
const FILES = [
    new URL('../models/yoloe/yoloe-26l-text.fp16.onnx', import.meta.url).href,
    new URL('../models/clip-text/mclip2-text.q4.onnx', import.meta.url).href,
]

// Let the machine settle after the job that woke us, and give up rather than
// retry a URL forever (offline, 404 after a model bump).
const SETTLE_MS = 2000
const MAX_FAILS = 3

const cached = new Set()
let wanted = false
let inflight = false
let fails = 0
let timer = null

/** Drain `url` to nothing; the caches keep the bytes, this must not. */
const pull = async (url, signal) => {
    const r = await fetch(url, { signal })
    if (!r.ok) throw new Error(`prefetch ${r.status} ${url.split('/').pop()}`)
    const reader = r.body?.getReader()
    if (!reader) { await r.arrayBuffer(); return 0 }
    let bytes = 0
    for (;;) {
        const { done, value } = await reader.read()
        if (done) return bytes
        bytes += value.byteLength
    }
}

const next = () => FILES.find((u) => !cached.has(u)) || null

const arm = () => {
    if (!wanted || inflight || fails >= MAX_FAILS || !next()) return
    clearTimeout(timer)
    timer = setTimeout(async () => {
        const url = next()
        if (!wanted || inflight || !url) return
        inflight = true
        const ctl = new AbortController()
        try {
            const r = await enqueueHeavy('model-prefetch', async () => {
                const bytes = await pull(url, ctl.signal)
                console.log('[seglab] prefetched', url.split('/').pop(), `${(bytes / 1e6).toFixed(1)} MB`)
                return true
            }, { priority: 'idle', signal: ctl.signal, onPreempt: () => ctl.abort(), timeoutMs: 10 * 60 * 1000 })
            if (r !== STALE) cached.add(url)
        } catch (err) {
            if (!ctl.signal.aborted) {
                fails += 1
                console.warn('[seglab] text-lane prefetch failed:', err?.message)
            }
        } finally {
            inflight = false
        }
        arm()   // next file, or a retry of this one after the work that preempted it
    }, SETTLE_MS)
}

// The busy→idle edge is the only thing that starts a pull.
onHeavyActivity((busy) => { if (!busy) arm() })

/** Intent signal: the user opened text search. Idempotent. */
export const prefetchTextLane = () => { wanted = true; arm() }

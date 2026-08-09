/**
 * sam21-host — the ONE SAM 2.1 instance for the whole origin.
 *
 * Runs as a SharedWorker: every tab connects to the same instance, so there is
 * exactly one ORT session, one arena, one GPUDevice and one embedding no matter
 * how many tabs are open. Measured on an M2/8GB, WebGPU + shader-f16 are both
 * available in a SharedWorker (spikes/sam21/probe.html), which is what makes
 * this possible.
 *
 * This replaces the cross-tab Web Lock in DESIGN-MASK-LANE §9. A lock serialises
 * N sessions that each still hold an arena — it bounds the SPIKE but not the
 * steady state. One shared instance bounds both, and serialisation falls out for
 * free because a single job queue is the only way in. Duplicating the session
 * for a "second concurrent user" would buy nothing: there is one GPU, so two
 * encodes time-share rather than overlap — same wall clock, double the peak.
 *
 * Same file also runs as a DedicatedWorker (no `onconnect`) for browsers where
 * a SharedWorker cannot reach WebGPU; the client picks. In that mode the
 * single-instance property is per-tab, which is the O(N) fallback.
 */

import {
    adoptEmbedding, buildDecoder, buildEncoder, checkDevice, decode, destroyDevice,
    embedStats, encode, exportEmbedding, recycleEncoderDevice, setDeviceLostHandler,
    laneConfig, laneState, releaseAll, releaseDecoder, releaseEncoder, setEmbedCap, touchEmbedding, warmDecoder,
} from './sam21-lane.js'

const PRIORITY = { interactive: 0, import: 1, normal: 2, idle: 3 }

// Two dormancy rungs, because releasing sessions does NOT return the memory.
// Measured on an M2/8GB: releaseEncoder() left the GPU process unchanged
// (1347→1309 MB) — ORT-Web pools freed WebGPU buffers per instance rather than
// handing them back. Only tearing the whole worker down returns it.
const DORMANT_MS = 20_000       // all tabs hidden → drop sessions, keep embeddings
const DEEP_IDLE_MS = 120_000    // still hidden → exit; clients respawn on demand

// Third rung, driven by ACTIVITY rather than visibility. Once an encoder
// session has existed, this instance holds ~1.2 GB that `release()` cannot give
// back (§1.4) — only exiting can, measured 1879 → 446 MB. Visibility is the
// wrong trigger for that: someone studying their photo for two minutes has a
// visible tab and a completely idle instance. Being wrong costs ONE slow click
// (~1.1 s: worker restart + decoder build) and no re-encode, because the
// embedding comes back from OPFS — measured encoded=false on that click.
// Shorter when this file runs as a DedicatedWorker. There the single-instance
// guarantee is gone and memory is O(tabs): measured 3 tabs on one photo, shared
// 2135 MB / 1 encode vs dedicated 5021 MB / 3 encodes. Every tab holds its own
// ~1.2 GB session, so idle ones have to give it back sooner — and the resume is
// no more expensive than in shared mode.
const SHARED = typeof SharedWorkerGlobalScope !== 'undefined' || typeof onconnect !== 'undefined'
const IDLE_EXIT_MS = SHARED ? 120_000 : 30_000

// Escape hatch for the post-encode device recycle (see the encode op). A memory
// optimisation that swaps ORT's GPUDevice mid-session has to be switchable
// without a rebuild. Carried on `hello` rather than on the worker URL: a query
// string there changes the SharedWorker's identity and the module cache key, so
// a debug flag would silently fork the single-instance guarantee.
let recycleOff = false

// Tunable: the governor can shed sooner under pressure, and the rungs are
// otherwise only observable by waiting two real minutes.
let dormantMs = DORMANT_MS
let deepMs = DEEP_IDLE_MS
let idleExitMs = IDLE_EXIT_MS
let idleTimer = null
// Only worth exiting for once a real encode has happened. Before that the
// instance is a ~10 MB decoder and a restart would cost more than it frees.
let builtEncoder = false

const clients = new Map() // port → { id, visible, imageKey, label }
const queue = []
let active = null
let seq = 0
let dormantTimer = null
let deepTimer = null

const send = (port, msg, transfer = []) => {
    try { port.postMessage(msg, transfer); return true } catch { return false }
}

// A closed tab leaves a port that nothing tells us about — SharedWorker has no
// disconnect event. A failed post IS that signal, so reap on it. Left alone,
// dead clients keep `anyVisible()` true and dormancy never runs, which means
// the memory never sheds: a silent leak that only shows up hours in.
const broadcast = (msg) => {
    let reaped = false
    for (const port of [...clients.keys()]) {
        if (!send(port, msg)) { clients.delete(port); reaped = true }
    }
    if (reaped) {
        setEmbedCap(clients.size)
        if (clients.size === 0) releaseAll()
        else reviewDormancy()
    }
}

// Identifies THIS worker instance. A restart is otherwise invisible in the
// status stream — the lane just looks empty, which reads as "something released
// my embedding" when in fact the whole host was replaced.
const HOST_ID = `h${Math.random().toString(36).slice(2, 8)}`

const status = () => ({
    type: 'status',
    hostId: HOST_ID,
    // Mirrors the retired engine's shape: callers (chips, verify) read these.
    device: 'webgpu',   // §4 — WebGPU is a hard requirement, there is no other lane
    cachedImages: laneState().embedKeys.length,  // retired engine's field name
    tabs: clients.size,
    visible: [...clients.values()].filter((c) => c.visible).length,
    activeLabel: active?.label || null,
    activeClient: active?.client?.id ?? null,
    queued: queue.length,
    builtEncoder,
    lane: laneState(),
})
const pushStatus = () => broadcast(status())

// A lost GPU device invalidates every session and embedding at once. Tell the
// tabs so the next click re-encodes silently instead of surfacing a failure.
setDeviceLostHandler((info) => {
    broadcast({ type: 'device-lost', reason: info?.reason || 'unknown' })
    pushStatus()
})

// --- dormancy: with one shared instance, "hidden" only counts when EVERY tab
//     is hidden. The embedding survives; only the sessions go.
const anyVisible = () => [...clients.values()].some((c) => c.visible)

/** Exit the worker. Only way to give ORT's WebGPU pool back; clients reconnect
 *  transparently and rebuild on next use. */
const shutdown = () => {
    releaseAll()
    // Report which generation is dying. Clients retire exactly that one, so
    // several tabs seeing this converge on a single fresh name rather than each
    // bumping past the others into private instances.
    broadcast({ type: 'closing', gen: GEN })
    setTimeout(() => self.close?.(), 0) // let the notice flush first
}

/** Re-arm the idle exit. Called on every job completion, so any activity — from
 *  any tab — pushes it back out. */
const reviewIdleExit = () => {
    clearTimeout(idleTimer)
    idleTimer = null
    if (!builtEncoder || active || queue.length || clients.size === 0) return
    idleTimer = setTimeout(() => {
        idleTimer = null
        if (!active && !queue.length) shutdown()
    }, idleExitMs)
}

const reviewDormancy = () => {
    clearTimeout(dormantTimer)
    clearTimeout(deepTimer)
    dormantTimer = deepTimer = null
    if (anyVisible() || clients.size === 0) return
    dormantTimer = setTimeout(() => {
        dormantTimer = null
        if (anyVisible()) return
        // Encoder only. Dropping the LAST ORT session destroys the WebGPU
        // device — measured: release encoder then decoder and `deviceLost` goes
        // true ~400 ms later — and a device loss clears every embedding, which
        // is exactly what this rung promises not to do. The decoder is 9.9 MB
        // and holds the device alive; the encoder is the session with the
        // multi-GB arena. Rung 2 takes the rest.
        releaseEncoder()
        pushStatus()
    }, dormantMs)
    deepTimer = setTimeout(() => {
        deepTimer = null
        if (!anyVisible() && !active) shutdown()
    }, deepMs)
}

// --- single job queue. One in flight, ever. That IS invariant I3.
const pump = () => {
    if (active || queue.length === 0) return
    active = queue.shift()
    if (!clients.has(active.port)) { active = null; pump(); return } // tab closed
    pushStatus()
    const started = performance.now()
    Promise.resolve()
        .then(() => active.run())
        .then(
            (result) => send(active.port, {
                id: active.id, ok: true, result,
                waitMs: +(started - active.queuedAt).toFixed(1),
            }, active.transfer?.(result) || []),
            (err) => send(active.port, { id: active.id, ok: false, error: String(err?.message || err) }),
        )
        .finally(() => { active = null; pushStatus(); pump(); reviewIdleExit() })
}

const submit = (port, id, label, run, { priority = 'normal', transfer = null } = {}) => {
    const job = {
        id, port, label, run, transfer, client: clients.get(port),
        rank: PRIORITY[priority] ?? PRIORITY.normal, seq: (seq += 1),
        queuedAt: performance.now(),
    }
    let i = queue.length
    while (i > 0 && queue[i - 1].rank > job.rank) i -= 1
    queue.splice(i, 0, job)
    // A tab that has to wait should say so, not look frozen.
    if (active || i < queue.length - 1) {
        send(port, { type: 'waiting', id, ahead: queue.length - 1 + (active ? 1 : 0), on: active?.label || null })
    }
    pump()
}

// The name a SharedWorker was constructed with is its generation tag; a plain
// Worker has none. Never queued — a ping must answer while an encode runs, or
// it proves nothing about liveness.
const GEN = Number(String(self.name || '').split('.').pop()) || 0

const OPS = {
    ping: () => ({ gen: GEN }),
    hello: async (port, p) => {
        const c = clients.get(port)
        c.label = p?.label || c.label
        c.visible = p?.visible !== false
        if (p?.recycle === false) recycleOff = true   // ?recycle=0 — never re-enabled by a later tab
        reviewDormancy()
        return { clientId: c.id, device: await checkDevice(), ...status() }
    },
    visibility: (port, p) => {
        clients.get(port).visible = !!p?.visible
        reviewDormancy()
        return status()
    },
    state: () => status(),
    // Free everything for this origin. Governor top rung / explicit teardown.
    releaseAll: () => { releaseAll(); return status() },
    // Read-only, and now the only shape this has: the lane's identity is a set
    // of module constants, so there is no setter to call by accident. The old
    // `configure` op released every session before applying its patch, which
    // meant a client reading a value destroyed both sessions and the embedding
    // — the first click's encode was thrown away and the next one repaid it.
    config: () => laneConfig(),
    destroyDevice: () => { destroyDevice(); return status() },  // terminal — see lane
    shutdown: () => { shutdown(); return { closing: true } },
    embedStats: (port, p) => embedStats(p.key, p.which),
    // A second tab on the same photo rides the resident embedding: skip its
    // 8 MB OPFS read and re-upload. Claims rather than merely asks — it also
    // touches the LRU (a tab that never encodes never refreshes it, so its
    // image could be evicted out from under it) and records the tab's image, so
    // an unkeyed decode still lands on the right one.
    claimEmbedding: (port, p) => {
        const has = touchEmbedding(p?.key)
        if (has) clients.get(port).imageKey = p.key
        return { has }
    },
    setDormancy: (port, p) => {
        dormantMs = Math.max(50, Number(p?.dormantMs) || DORMANT_MS)
        deepMs = Math.max(dormantMs, Number(p?.deepIdleMs) || DEEP_IDLE_MS)
        idleExitMs = Math.max(50, Number(p?.idleExitMs) || IDLE_EXIT_MS)
        reviewDormancy()
        reviewIdleExit()
        return { dormantMs, deepMs, idleExitMs }
    },
    // OPFS seam — the tab owns the cache (SharedWorkers cannot open OPFS).
    adoptEmbedding: async (port, p) => {
        const ok = await adoptEmbedding(p.key, p.buffers)
        if (ok) clients.get(port).imageKey = p.key
        return { adopted: ok }
    },
    exportEmbedding: (port, p) => exportEmbedding(p.key),
    buildEncoder: async () => { await buildEncoder(); return laneState() },
    releaseEncoder: () => { releaseEncoder(); return laneState() },
    releaseDecoder: () => { releaseDecoder(); return laneState() },
}

const handle = async (port, msg) => {
    const { id, op, payload } = msg || {}
    if (!op) return
    if (op === 'encode') {
        submit(port, id, 'encode', async () => {
            clients.get(port).imageKey = payload.key
            // Another encode already waiting? Hold the session through it —
            // a batch import must not recompile shaders per image.
            const more = queue.some((j) => j.label === 'encode')
            const r = await encode({ ...payload, keepEncoder: more })
            if (r && !r.cached) builtEncoder = true   // a session now exists; see IDLE_EXIT_MS
            try { payload.bitmap?.close?.() } catch { /* already closed */ }
            // Give the encoder's ~976 MB pool back NOW rather than at the lane's
            // idle timer, which every click re-arms (lane: recycleEncoderDevice).
            //
            // DEDICATED mode only. There the instance is per-tab and — this is
            // the part that matters — it runs inside the PAGE's process, so the
            // encoder's workspace counts against the same per-process footprint
            // the browser reaps the tab on. Shared mode amortises one pool across
            // every tab, in its own process, and a second tab may be about to
            // encode into it; that arrangement is measured good and is left
            // alone. Skipped mid-batch: `more` means another import is queued and
            // would only rebuild what this just tore down.
            if (r && !r.cached && !SHARED && !more && !recycleOff) {
                const rec = await recycleEncoderDevice(payload.key)
                if (rec) r.recycled = true
            }
            if (payload.warm !== false) await warmDecoder(payload.key)
            return r
        }, { priority: 'import' })
        return
    }
    if (op === 'decode') {
        // Default to this tab's own image, so a decode can never land on
        // whichever embedding another tab happened to write last.
        // `sid` is this connection's own id, so the lane's continuity anchor
        // ("the mask this caller was last shown") stays per-tab. One shared
        // instance serves every tab; keying continuity on the image alone would
        // let one tab's refinement pin another tab's click.
        submit(port, id, 'decode', () => decode({
            ...payload,
            key: payload.key ?? clients.get(port)?.imageKey,
            sid: clients.get(port)?.id ?? null,
        }), {
            priority: 'interactive',
            // Logit planes are the only thing crossing back: 256 KB each,
            // zero-copy. The alternates ride along so a "wrong part" correction
            // costs a repaint instead of a decode.
            transfer: (r) => [r?.logits?.buffer, ...(r?.alternates || []).map((a) => a.buffer)].filter(Boolean),
        })
        return
    }
    if (op === 'warm') {
        submit(port, id, 'warm', async () => { await buildDecoder(); return laneState() })
        return
    }
    const fn = OPS[op]
    if (!fn) { send(port, { id, ok: false, error: `unknown op: ${op}` }); return }
    try {
        const result = await fn(port, payload)
        // The embedding readback is 8 MB — transfer it rather than letting
        // structured clone copy it.
        const transfer = op === 'exportEmbedding' && Array.isArray(result) ? result : []
        send(port, { id, ok: true, result }, transfer)
    } catch (err) {
        send(port, { id, ok: false, error: String(err?.message || err) })
    }
}

const attach = (port) => {
    clients.set(port, { id: `t${clients.size + 1}-${Date.now() % 100000}`, visible: true, imageKey: null, label: null })
    port.onmessage = (e) => {
        if (e.data?.op === 'bye') { detach(port); return }
        handle(port, e.data)
    }
    port.onmessageerror = () => detach(port)
    port.start?.()
    setEmbedCap(clients.size)
    reviewIdleExit()
    pushStatus()
}

const detach = (port) => {
    clients.delete(port)
    setEmbedCap(clients.size)
    for (let i = queue.length - 1; i >= 0; i -= 1) if (queue[i].port === port) queue.splice(i, 1)
    // Last tab gone: nothing to serve, so drop everything including the embedding.
    if (clients.size === 0) releaseAll()
    else { reviewDormancy(); pushStatus() }
}

if (typeof onconnect !== 'undefined' || typeof SharedWorkerGlobalScope !== 'undefined') {
    self.onconnect = (e) => attach(e.ports[0])
} else {
    attach(self) // DedicatedWorker fallback: one instance per tab
}

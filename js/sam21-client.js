/**
 * sam21-client — tab-side handle on the shared SAM 2.1 instance.
 *
 * Prefers a SharedWorker (one instance for the whole origin, memory O(1) in tab
 * count). Falls back to a DedicatedWorker where a SharedWorker cannot reach
 * WebGPU — same host file, same protocol, but then memory is O(tabs).
 *
 * Visibility is reported up so the host can shed sessions once EVERY tab is
 * hidden; a single tab going background must not disturb another tab's work.
 */

import { loadEmbedding, saveEmbedding } from './sam21-store.js'
import { PRECISION } from './sam21-lane.js'

const HOST = new URL('./sam21-host.js', import.meta.url)

const state = {
    port: null,
    mode: null,          // 'shared' | 'dedicated'
    seq: 0,
    pending: new Map(),  // id → { resolve, reject, timer }
    status: null,
    listeners: new Set(),
    gen: 0,              // generation of the host we are attached to
}

// A SharedWorker's name IS its sharing key, so every tab must agree on one.
// That makes a deliberate shutdown hazardous: Chrome keeps the registration
// resolvable for a moment while the worker tears down, so a reconnect under the
// same name attaches to a corpse whose event loop is already gone — `connect`
// never fires and every op rides its full timeout. Measured: releaseAll right
// after a level-3 teardown hung 20 s on `state`. Retiring the generation moves
// all tabs to a fresh name together.
const GEN_KEY = 'seglab.sam21.gen'
// Without localStorage the generation cannot be SHARED, and a per-tab counter
// would give tabs different names — splitting them into private instances,
// which costs far more than the hang it avoids. Pin the name there and let the
// liveness ping converge instead: a corpse stops resolving within a moment.
let storageOk = true
const readGen = () => {
    if (!storageOk) return 0
    try { return Number(localStorage.getItem(GEN_KEY)) || 0 } catch { storageOk = false; return 0 }
}
/** Retire `gen` — idempotent, so several tabs seeing the same `closing`
 *  converge on ONE new name instead of racing each other forward. */
const retireGen = (gen) => {
    if (!storageOk) return
    const g = Number.isFinite(gen) ? gen : readGen()
    if (readGen() > g) return
    try { localStorage.setItem(GEN_KEY, String(g + 1)) } catch { storageOk = false }
}
const workerName = () => `seglab.sam21.${readGen()}`

// Per-op timeouts. A control call must never inherit the encode budget: a
// 10-minute wait on `state` is indistinguishable from a hang, and that is
// exactly how it surfaced (verify died on "sam21: state timed out").
const TIMEOUT_MS = {
    encode: 10 * 60 * 1000,     // model download on a cold cache can be minutes
    warm: 10 * 60 * 1000,
    adoptEmbedding: 60 * 1000,
    exportEmbedding: 60 * 1000,
    decode: 120 * 1000,
    default: 20 * 1000,         // hello / state / visibility / releaseAll / …
}

const notify = () => { for (const fn of state.listeners) { try { fn(state.status) } catch { /* ignore */ } } }

/** Subscribe to host status (tab count, queue depth, what is running where). */
export const subscribe = (fn) => { state.listeners.add(fn); return () => state.listeners.delete(fn) }
export const hostStatus = () => state.status
export const hostMode = () => state.mode

/** Host exited (deep idle). Drop the dead port so the next call respawns it. */
const onClosing = () => {
    state.port = null
    state.status = null
    for (const [, p] of state.pending) { clearTimeout(p.timer); p.reject(new Error('sam21: host restarting')) }
    state.pending.clear()
    notify()
}

/** The port we hold is dead or unresponsive. Retire its generation so the next
 *  connect builds a fresh host rather than re-attaching to the same corpse. */
const recycle = (why) => {
    console.warn('[seglab][sam21] recycling host generation:', why)
    noteHostDeath()
    retireGen(state.gen)
    onClosing()
}

const onMessage = (msg) => {
    noteAlive()
    if (msg?.type === 'closing') { retireGen(msg.gen); onClosing(); return }
    if (msg?.type === 'device-lost') {
        // Embeddings died with the device; drop our view so the next encode
        // rebuilds rather than decoding against a freed buffer.
        console.warn('[seglab][sam21] GPU device lost:', msg.reason, '— next click re-encodes')
        for (const fn of state.listeners) { try { fn({ type: 'device-lost', reason: msg.reason }) } catch { /* ignore */ } }
        return
    }
    if (msg?.type === 'status') { state.status = msg; notify(); return }
    if (msg?.type === 'waiting') {
        // Another tab holds the instance; surface it rather than looking frozen.
        state.pending.get(msg.id)?.onWait?.(msg)
        return
    }
    const p = state.pending.get(msg?.id)
    if (!p) return
    state.pending.delete(msg.id)
    clearTimeout(p.timer)
    if (msg.ok) p.resolve(msg.result)
    else p.reject(new Error(msg.error || 'sam21 host error'))
}

// `state.force` is set in exactly one place — hello(), when a shared host turns
// out to be unable to reach WebGPU (Firefox/Safari). It used to also have an
// exported `forceMode()` setter for bench/A-B use, with no callers: a transport
// override nobody turns is a way to make one tab silently disagree with the
// rest about how many ORT sessions the origin is holding.

const connect = () => {
    if (state.port) return state.port
    state.gen = readGen()
    let shared = null
    if (typeof SharedWorker !== 'undefined' && state.force !== 'dedicated' && sharedIsUsable()) {
        try {
            shared = new SharedWorker(HOST, { type: 'module', name: workerName() })
            state.port = shared.port
            state.mode = 'shared'
        } catch { /* fall through */ }
    }
    if (!state.port) {
        const w = new Worker(HOST, { type: 'module' })
        state.port = w
        state.mode = 'dedicated'
    }
    state.port.onmessage = (e) => onMessage(e.data)
    // A SharedWorker that dies (OOM, crash) never answers. Without this every
    // pending call would sit on the 10-minute timeout and the UI would look
    // hung; instead fail them now and let the next call respawn the host.
    const onDead = (label) => (e) => {
        console.warn(`[seglab][sam21] host ${label}:`, e?.message || e?.type || '')
        onClosing()
    }
    if (state.mode === 'shared') {
        state.port.onmessageerror = onDead('message error')
        // onerror belongs to the SharedWorker OBJECT, not the port. Constructing
        // a second one to hang it on opened a second connection: every tab
        // registered TWICE, so `tabs` double-counted, setEmbedCap allowed double
        // the resident embeddings, and — worst — the phantom client defaulted to
        // visible and never sent visibility, pinning anyVisible() true so
        // dormancy could never fire and memory never shed.
        if (shared) shared.onerror = onDead('error')
    } else {
        state.port.onerror = onDead('error')
        state.port.onmessageerror = onDead('message error')
    }
    state.port.start?.()
    probeLiveness()
    // Re-announce: after a deep-idle restart the new host has never seen us.
    if (state.label !== undefined) hello(state.label).catch(() => null)

    if (typeof document !== 'undefined' && !state.wired) {
        state.wired = true // listeners outlive reconnects; register once
        document.addEventListener('visibilitychange', () =>
            setVisible(document.visibilityState === 'visible').catch(() => null))
        // Closing without a bye leaves the host holding a dead port until GC.
        addEventListener('pagehide', () => { try { state.port?.postMessage({ op: 'bye' }) } catch { /* gone */ } })
    }
    return state.port
}

const call = (op, payload = {}, transfer = [], onWait = null) => new Promise((resolve, reject) => {
    const port = connect()
    const id = (state.seq += 1)
    const timer = setTimeout(() => {
        state.pending.delete(id)
        // A timed-out control op means the host is wedged or gone; drop the
        // port so the NEXT call respawns it instead of piling up on a corpse.
        if ((TIMEOUT_MS[op] ?? TIMEOUT_MS.default) <= TIMEOUT_MS.default) state.port = null
        reject(new Error(`sam21: ${op} timed out`))
    }, TIMEOUT_MS[op] ?? TIMEOUT_MS.default)
    state.pending.set(id, { resolve, reject, timer, onWait })
    port.postMessage({ id, op, payload }, transfer)
})

// Prove a freshly-connected host is actually running. A corpse answers nothing,
// and without this every op on it waits out its own timeout — up to 10 minutes
// for an encode.
//
// The budget was 2 s, which is a CHROME number: it assumes spawning the host and
// loading its module graph beats the timer. On a cold Safari start it does not,
// so the probe declared a perfectly healthy host dead, retired its generation
// and rejected the very first `hello` as "host restarting" — then did it again
// on the respawn. That is the whole "doesn't work on Safari" symptom, and it is
// a self-inflicted spawn loop, not a platform limit.
//
// Two changes make it browser-agnostic. Any INBOUND MESSAGE now counts as proof
// of life (`noteAlive`): the host pushes a status frame from `attach()`, so a
// live host clears the probe on its own schedule rather than ours, however slow
// the start was. And the fallback budget is generous, because it now only has to
// catch a genuine corpse — nothing waits on it in the healthy case.
const PROBE_MS = 8000
let probeStrikes = 0
let probeTimer = null
const clearProbe = () => { if (probeTimer) { clearTimeout(probeTimer); probeTimer = null } }
const noteAlive = () => { probeStrikes = 0; clearProbe() }
const probeLiveness = () => {
    const port = state.port
    // Back off on repeats. A host that is merely SLOW to start (cold module
    // fetch, loaded machine) would otherwise be recycled every time and never
    // get far enough to answer — trading a hang for a spawn loop.
    const wait = Math.min(PROBE_MS * 2 ** probeStrikes, 30_000)
    clearProbe()
    probeTimer = setTimeout(() => {
        probeTimer = null
        if (state.port !== port) return
        probeStrikes += 1
        recycle(`no answer in ${wait}ms`)
    }, wait)
    call('ping').then(
        noteAlive,
        (err) => {
            // Any real answer proves the event loop is alive — even "unknown op"
            // from an older host that predates ping. Only our own recycle
            // rejection proves nothing.
            if (!/host restarting/.test(String(err?.message))) noteAlive()
            else clearProbe()
        },
    )
}

// Cached embeddings are precision-specific: adopting across a precision change
// is at best a wasted read and at worst silently wrong, so the key carries it.
// PRECISION is a module constant imported from the lane, not a value fetched
// from the host and memoised — the fetched version was read once per page and
// never revisited, so it went stale the moment anything changed the host.
const storeKey = (key) => `${PRECISION}:${key}`

/**
 * A SharedWorker can EXIST and still not reach WebGPU: `navigator.gpu` is not
 * exposed in SharedWorker scope outside Chrome (approved by the GPU-for-the-Web
 * CG, unimplemented in Firefox/WebKit as of writing). The constructor succeeds
 * there, so feature-detecting `SharedWorker` cannot see it — the host's own
 * device verdict is the only signal, and without acting on it the app is simply
 * dead on those browsers even though a DedicatedWorker would have worked.
 * Exported so the decision is testable without a browser that has the defect.
 */
export const shouldFallbackToDedicated = (mode, device, alreadyTried) =>
    mode === 'shared' && !alreadyTried && device?.ok === false

// …but `device.ok === false` is NOT the only way a shared host is unusable, and
// on Safari it is not the one that fires. Measured on Safari 26.6 (macOS 26.6,
// _safari-ortw.html): a SharedWorker there reports navigator.gpu, an adapter,
// shader-f16 and a device — checkDevice returns ok — and ORT still cannot run in
// it. The session CREATES, then the worker is killed and respawned in an endless
// loop; the run never completes. The identical script in a DedicatedWorker on
// the same browser builds the session in 1.2 s and runs in 0.96 s.
//
// So the device verdict cannot detect this: the host dies before it can report
// anything, and to the tab it is indistinguishable from a slow encode. The
// SharedWorker is only a memory OPTIMISATION (one ORT session per origin instead
// of per tab) — correctness everywhere outranks it, so it is opt-IN to engines
// where it is measured to work rather than opt-out where it visibly breaks.
//
// `navigator.userAgentData` is Chromium-only and needs no UA string parsing;
// Chromium is exactly where WebGPU-in-SharedWorker is implemented and measured
// good. Everything else gets a DedicatedWorker: correct on every engine, and it
// only costs extra memory when several tabs are open at once.
const DEDICATED_KEY = 'seglab.sam21.dedicated'
const latchDedicated = () => { try { localStorage.setItem(DEDICATED_KEY, '1') } catch { /* private mode */ } }
const sharedIsUsable = () => {
    try { if (localStorage.getItem(DEDICATED_KEY) === '1') return false } catch { /* no storage */ }
    return typeof navigator !== 'undefined' && !!navigator.userAgentData
}

// Safety net for an engine that starts crash-looping without announcing it (a
// future Safari that adds the API before it works, a driver regression). Two
// unexplained host deaths in one minute and this tab — and every later one —
// stops paying for the shared instance.
const RESTART_WINDOW_MS = 60_000
let restarts = []
const noteHostDeath = () => {
    if (state.mode !== 'shared') return
    const now = Date.now()
    restarts = restarts.filter((t) => now - t < RESTART_WINDOW_MS)
    restarts.push(now)
    if (restarts.length < 3) return
    console.warn('[seglab][sam21] shared host keeps dying — latching to a dedicated worker for this origin')
    latchDedicated()
    state.force = 'dedicated'
    restarts = []
}

/** Announce this tab and get the device verdict + current host status. */
export const hello = async (label = null) => {
    state.label = label ?? state.label ?? null
    const ask = () => call('hello', {
        label: state.label,
        visible: typeof document === 'undefined' || document.visibilityState === 'visible',
        // Post-encode device recycle (host encode op). Off with ?recycle=0.
        recycle: !/[?&]recycle=0\b/.test(typeof location === 'undefined' ? '' : location.search),
    })
    // A host recycled mid-handshake rejects everything in flight as "host
    // restarting". The next connect builds a fresh one, so retrying once turns
    // a dead first click into a slightly slow one; every caller treats a hello
    // failure as "no lane", which is why this must not surface.
    let r
    try {
        r = await ask()
    } catch (err) {
        if (!/host restarting/.test(String(err?.message))) throw err
        console.warn('[seglab][sam21] host recycled during hello; retrying once')
        r = await ask()
    }
    if (shouldFallbackToDedicated(state.mode, r?.device, state.triedDedicated)) {
        state.triedDedicated = true
        console.warn('[seglab][sam21] shared host cannot reach WebGPU:', r?.device?.reason,
            '— retrying with a dedicated worker (memory becomes O(tabs))')
        state.force = 'dedicated'
        onClosing()                       // drop the useless shared port
        return hello(label)
    }
    state.status = { ...r, type: 'status' }
    notify()
    return r
}

/** Build the decoder ahead of the first click. Cheap (9.9 MB). */
export const warm = () => call('warm')

/**
 * Encode `bitmap` (transferred — the caller must not touch it afterwards) into
 * the shared embedding, keyed by `key`. A second tab opening the same image
 * gets it for free: the key already matches and no encode runs.
 */
export const encodeImage = async (bitmap, key, { warmDecoder = true, onWait = null, persist = true } = {}) => {
    // OPFS cache lives HERE, in the tab: navigator.storage.getDirectory() throws
    // SecurityError inside a SharedWorker in Chrome (measured, same-origin
    // script — not just blob:). An 8 MB read beats a ~1.4 s encode that also has
    // to open a >1 GB session, so try it before paying for one.
    // Another tab may already have this photo resident in the shared host, in
    // which case the OPFS read below would load 8 MB from disk only to hand the
    // host bytes it already has. Ask first — one small message.
    if (await call('claimEmbedding', { key }).then((r) => r?.has).catch(() => false)) {
        try { bitmap.close?.() } catch { /* already gone */ }
        if (warmDecoder) await call('warm').catch(() => null)
        return { cached: 'resident', key }
    }
    if (persist) {
        try {
            const raw = await loadEmbedding(storeKey(key))
            if (raw) {
                const r = await call('adoptEmbedding', { key, buffers: raw }, raw)
                if (r?.adopted) {
                    try { bitmap.close?.() } catch { /* already gone */ }
                    if (warmDecoder) await call('warm').catch(() => null)
                    return { cached: 'opfs', key }
                }
            }
        } catch { /* miss / corrupt / quota — just encode */ }
    }
    const res = await call('encode', { bitmap, key, warm: warmDecoder }, [bitmap], onWait)
    // AWAIT the persist rather than detaching it. A detached write is lost if
    // the tab closes right after an import — which is exactly what a user who
    // imports and immediately navigates away does. The cost is one ~8 MB
    // GPU→CPU readback on an operation that just spent ~1 s encoding, and it
    // only runs on a real encode (a cache hit returns above).
    if (persist && res && !res.cached) {
        try {
            const raw = await call('exportEmbedding', { key })
            if (raw?.length) await saveEmbedding(storeKey(key), raw)
        } catch (err) {
            // Never fatal — a failed write just costs one re-encode later — but
            // never silent either: a cache that always fails is indistinguishable
            // from "the encode is always slow".
            console.warn('[seglab][sam21] embedding persist failed:', err?.message)
        }
    }
    return res
}

/** Clicks in 1024² space: [{ x, y, label }] — 1 include, 0 exclude.
 *  `key` defaults to this tab's last encode. */
export const decodeMask = (clicks, { key = null, onWait = null } = {}) =>
    call('decode', { clicks, key }, [], onWait)

/** Report this tab's visibility. Wired to visibilitychange automatically;
 *  exported so tests can drive dormancy without backgrounding a real tab. */
export const setVisible = (visible) => call('visibility', { visible })
export const releaseAll = () => call('releaseAll')
/** Swap encoder/decoder file or session options. Drops resident state first. */
/** Escape hatch for bench/governor ops (buildEncoder, destroyDevice, …).
 *  `shutdown` retires the generation HERE, before the call: the host's own
 *  `closing` notice can lose the race with a page that closes right after asking
 *  (verify does exactly that), and then the next page in the same origin adopts
 *  the corpse by name. localStorage is written synchronously, so it cannot. */
export const op = (name, payload = {}) => {
    if (name !== 'shutdown') return call(name, payload)
    // Order matters: call() connects and posts synchronously, so the message
    // reaches the CURRENT generation; retiring first would spawn a fresh host
    // and shut THAT one down, leaving the live one running.
    const done = call(name, payload)
    retireGen(state.gen)
    return done.finally(() => { state.port = null })
}
export const refreshStatus = () => call('state').then((r) => { state.status = r; notify(); return r })

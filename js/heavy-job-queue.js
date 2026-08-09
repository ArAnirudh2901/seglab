/**
 * heavy-job-queue — ONE memory-heavy operation at a time (pure, main thread).
 * Image decode, model warm/encode/decode, detector runs, wasm CV refinement
 * and export re-decodes all pass through here, so their peak allocations can
 * never stack. Priorities (lower runs first): import 0 · interactive 1 ·
 * normal 2 · idle 3; FIFO within a priority. A queued job re-checks currency
 * before it starts; a stale job resolves to the STALE sentinel and never runs.
 * Rejections release ownership, so a failed job cannot deadlock.
 *
 * A job that never SETTLES used to deadlock it anyway. Ownership was released
 * in a `.finally` on the task's own promise, so an await that never resolves —
 * a decode worker the OS killed without firing `onerror`, a wedged port — held
 * `active` forever and every later job queued behind it. Observed as a stuck
 * "Selecting…" followed by an import spinner that never cleared: the import was
 * not slow, it was waiting on a selection that could no longer finish. Every job
 * now carries a watchdog; on expiry it rejects, releases ownership and the queue
 * moves on. A late settle is ignored (see `settle`) so it can never clear a
 * SUCCESSOR's ownership.
 */

const PRIORITY = { import: 0, high: 1, interactive: 1, normal: 2, low: 3, idle: 3 }

// Per-label ceilings — generous enough that a cold model download or a big RAW
// decode finishes normally, finite so nothing can wedge the app. These bound
// the WHOLE job including its queue-internal awaits; the transports below have
// their own tighter ones.
const TIMEOUT_MS = {
    'decode-proxy': 60_000,
    'cv-refine': 60_000,
    'model-warm': 240_000,     // first run compiles shaders after an 81 MB fetch
    'encode-prewarm': 240_000,
    segment: 240_000,
    'export-refine': 300_000,
    detect: 420_000,           // backstop behind sam-client's own 6-minute limit
    default: 120_000,
}

/** Resolved instead of running when a queued job is no longer current. */
export const STALE = Object.freeze({ stale: true })

const state = {
    queue: [],          // waiting jobs, kept sorted (priority, seq)
    active: null,       // running job or null
    seq: 0,
    log: [],            // dev telemetry: { label, outcome, waitMs, runMs }
}

const DEV = typeof location !== 'undefined' && /^(localhost|127\.)/.test(location.hostname || '')
const logJob = (entry) => {
    state.log.push(entry)
    if (state.log.length > 200) state.log.shift()
    if (DEV) console.log('[seglab][queue]', entry)
}

const isStale = (job) => {
    if (job.cancelled) return true
    if (typeof job.isCurrent === 'function') {
        try { if (!job.isCurrent()) return true } catch { return true }
    }
    if (job.signal?.aborted) return true
    return false
}

/** Settle `job` exactly once and hand ownership on. A task that resolves after
 *  its watchdog already fired lands here too, and must be dropped: by then the
 *  slot belongs to a different job. */
const settle = (job, outcome, deliver) => {
    if (job.settled) return
    job.settled = true
    clearTimeout(job.timer)
    logJob({ label: job.label, outcome, waitMs: job.startedAt - job.queuedAt, runMs: Date.now() - job.startedAt })
    deliver()
    if (state.active === job) {
        state.active = null
        pump() // ownership always released — after a rejection or a timeout too
    }
}

const pump = () => {
    if (state.active || state.queue.length === 0) return
    const job = state.queue.shift()
    if (isStale(job)) {
        logJob({ label: job.label, outcome: 'stale', waitMs: Date.now() - job.queuedAt, runMs: 0 })
        job.resolve(STALE)
        pump()
        return
    }
    state.active = job
    job.startedAt = Date.now()
    job.timer = setTimeout(
        () => settle(job, 'timeout', () => job.reject(
            new Error(`${job.label} timed out after ${Math.round(job.timeoutMs / 1000)}s`),
        )),
        job.timeoutMs,
    )
    Promise.resolve()
        .then(() => job.task())
        .then(
            (value) => settle(job, 'done', () => job.resolve(value)),
            (err) => settle(job, 'error', () => job.reject(err)),
        )
}

/**
 * Enqueue `task` (async fn). Resolves with the task's value, or the STALE
 * sentinel when the job was invalidated before it started. `revision` scopes
 * the job for cancelHeavyBefore; `isCurrent` is re-checked at dequeue.
 */
export const enqueueHeavy = (label, task, {
    priority = 'normal', signal = null, revision = null, isCurrent = null, timeoutMs = null,
} = {}) => new Promise((resolve, reject) => {
    const job = {
        label,
        task,
        rank: PRIORITY[priority] ?? PRIORITY.normal,
        seq: ++state.seq,
        signal,
        revision,
        isCurrent,
        timeoutMs: timeoutMs ?? TIMEOUT_MS[label] ?? TIMEOUT_MS.default,
        settled: false,
        timer: null,
        cancelled: false,
        queuedAt: Date.now(),
        resolve,
        reject,
    }
    let i = state.queue.length
    while (i > 0 && (state.queue[i - 1].rank > job.rank)) i -= 1
    state.queue.splice(i, 0, job)
    pump()
})

/** Cancel every queued job whose revision is older than `revision`.
 *  (An in-flight kernel cannot be interrupted; its consumer must reject the
 *  result on the revision check instead.) */
export const cancelHeavyBefore = (revision) => {
    for (const job of state.queue) {
        if (job.revision !== null && job.revision < revision) job.cancelled = true
    }
}

/** New-document reset: drop every queued DOCUMENT-SCOPED job (one that
 *  carries a revision or an isCurrent check). Document-agnostic jobs like a
 *  model warm survive — the new document needs them too.
 *
 *  A job already RUNNING cannot be interrupted, but it can be disowned. One that
 *  has outlived `STUCK_MS` belongs to a document the user has just replaced and
 *  is, by then, far past any healthy duration — waiting out its full watchdog
 *  would make the new import look hung for minutes. Release the slot so the
 *  import starts now; the abandoned task runs to completion unobserved and its
 *  late settle is dropped. The grace period is what keeps the memory invariant:
 *  a merely slow job still gets the queue to itself. */
const STUCK_MS = 10_000
export const clearHeavyQueue = () => {
    for (const job of state.queue) {
        if (job.revision !== null || typeof job.isCurrent === 'function') job.cancelled = true
    }
    const job = state.active
    if (!job || (job.revision === null && typeof job.isCurrent !== 'function')) return
    if (Date.now() - job.startedAt < STUCK_MS) return
    settle(job, 'abandoned', () => job.resolve(STALE))
}

export const getHeavyQueueState = () => ({
    activeLabel: state.active?.label || null,
    activeRevision: state.active?.revision ?? null,
    queuedCount: state.queue.length,
})

/** Dev/verify telemetry: recent job outcomes. */
export const getHeavyQueueLog = () => state.log.slice()

/**
 * heavy-job-queue — ONE memory-heavy operation at a time (pure, main thread).
 * Image decode, model warm/encode/decode, detector runs, wasm CV refinement
 * and export re-decodes all pass through here, so their peak allocations can
 * never stack. Priorities (lower runs first): import 0 · interactive 1 ·
 * normal 2 · idle 3; FIFO within a priority. A queued job re-checks currency
 * before it starts; a stale job resolves to the STALE sentinel and never runs.
 * Rejections release ownership (finally), so a failed job cannot deadlock.
 */

const PRIORITY = { import: 0, high: 1, interactive: 1, normal: 2, low: 3, idle: 3 }

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
    Promise.resolve()
        .then(() => job.task())
        .then(
            (value) => {
                logJob({ label: job.label, outcome: 'done', waitMs: job.startedAt - job.queuedAt, runMs: Date.now() - job.startedAt })
                job.resolve(value)
            },
            (err) => {
                logJob({ label: job.label, outcome: 'error', waitMs: job.startedAt - job.queuedAt, runMs: Date.now() - job.startedAt })
                job.reject(err)
            },
        )
        .finally(() => {
            state.active = null
            pump() // ownership always released, even after a rejection
        })
}

/**
 * Enqueue `task` (async fn). Resolves with the task's value, or the STALE
 * sentinel when the job was invalidated before it started. `revision` scopes
 * the job for cancelHeavyBefore; `isCurrent` is re-checked at dequeue.
 */
export const enqueueHeavy = (label, task, {
    priority = 'normal', signal = null, revision = null, isCurrent = null,
} = {}) => new Promise((resolve, reject) => {
    const job = {
        label,
        task,
        rank: PRIORITY[priority] ?? PRIORITY.normal,
        seq: ++state.seq,
        signal,
        revision,
        isCurrent,
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
 *  model warm survive — the new document needs them too. */
export const clearHeavyQueue = () => {
    for (const job of state.queue) {
        if (job.revision !== null || typeof job.isCurrent === 'function') job.cancelled = true
    }
}

export const getHeavyQueueState = () => ({
    activeLabel: state.active?.label || null,
    activeRevision: state.active?.revision ?? null,
    queuedCount: state.queue.length,
})

/** Dev/verify telemetry: recent job outcomes. */
export const getHeavyQueueLog = () => state.log.slice()

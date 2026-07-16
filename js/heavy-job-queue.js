/**
 * heavy-job-queue — the single lane for memory-heavy work
 * ---------------------------------------------------------
 * Every operation with a large transient footprint (proxy decode, model
 * warm, image encode, prompt decode, CV refinement, export) runs here,
 * strictly one at a time — serialization is the memory guarantee: peaks
 * never stack. Cancellation is cooperative: queued jobs are rejected while
 * waiting; in-flight jobs run to completion and their consumers drop stale
 * results by revision. revision 0 marks image-agnostic jobs (model warm)
 * that survive image switches.
 */

export const PRIORITY = { interaction: 3, import: 2, model: 1, speculative: 0 }

export class StaleJobError extends Error {
    constructor(label) {
        super(`stale heavy job: ${label}`)
        this.name = 'StaleJobError'
        this.stale = true
    }
}

const queue = []
let active = null // { label, revision } while a job runs
let seq = 0
let peakDepth = 0
const eventLog = [] // ring buffer for tests/telemetry
const LOG_MAX = 300

const logEvent = (ev, job) => {
    eventLog.push({
        t: Date.now(),
        ev,
        label: job.label,
        revision: job.revision,
        depth: queue.length,
    })
    if (eventLog.length > LOG_MAX) eventLog.splice(0, eventLog.length - LOG_MAX)
}

const pump = () => {
    while (!active && queue.length > 0) {
        queue.sort((a, b) => b.priority - a.priority || a.seq - b.seq)
        const job = queue.shift()
        // Currency re-check at start: a job enqueued for a replaced image
        // must never begin.
        if (job.signal?.aborted || (job.isCurrent && !job.isCurrent())) {
            logEvent('drop-stale', job)
            job.reject(new StaleJobError(job.label))
            continue
        }
        active = { label: job.label, revision: job.revision }
        logEvent('start', job)
        Promise.resolve()
            .then(() => job.task({ signal: job.signal }))
            .then(job.resolve, job.reject)
            .finally(() => {
                logEvent('end', job)
                active = null
                pump() // a failed job must not stall the lane
            })
        return
    }
}

export const enqueueHeavy = (label, task, {
    priority = PRIORITY.model,
    signal = null,
    revision = 0,
    isCurrent = null,
} = {}) => new Promise((resolve, reject) => {
    const job = { label, task, priority, signal, revision, isCurrent, resolve, reject, seq: ++seq }
    queue.push(job)
    peakDepth = Math.max(peakDepth, queue.length + (active ? 1 : 0))
    logEvent('enqueue', job)
    pump()
})

/** Reject queued jobs belonging to revisions older than `revision`. */
export const cancelHeavyBefore = (revision) => {
    for (let i = queue.length - 1; i >= 0; i -= 1) {
        const job = queue[i]
        if (job.revision > 0 && job.revision < revision) {
            queue.splice(i, 1)
            logEvent('cancel', job)
            job.reject(new StaleJobError(job.label))
        }
    }
}

export const clearHeavyQueue = () => {
    for (const job of queue.splice(0)) {
        logEvent('clear', job)
        job.reject(new StaleJobError(job.label))
    }
}

export const getHeavyQueueState = () => ({
    activeLabel: active?.label || null,
    activeRevision: active?.revision ?? null,
    queuedCount: queue.length,
    peakDepth,
    log: eventLog.slice(),
})

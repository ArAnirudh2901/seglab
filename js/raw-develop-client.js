/**
 * raw-develop-client — main-thread API over raw-develop-worker, queued through
 * the shared heavy scheduler. The FALLBACK for a camera RAW with no usable
 * embedded JPEG preview: LibRaw demosaics the sensor in the worker and hands
 * back a compact image/jpeg Blob that rejoins the normal decode-to-proxy path.
 *
 * Lazy: the worker/wasm (~2 MB) loads only when a preview-less RAW is actually
 * opened — never at boot, never on a common RAW. The worker is DISPOSED after
 * every develop, because a full-sensor demosaic grows the wasm heap and wasm
 * memory never shrinks; disposing returns it to the OS. Failure is silent — the
 * caller falls back to the existing "no readable preview" error. SIMD-only
 * build (matches the runtime): unsupported hosts skip develop entirely.
 */

import { enqueueHeavy, STALE } from './heavy-job-queue.js'
import { simdSupported } from './cv-refine-client.js'

const MAX_BYTES = 512 * 1024 * 1024
const DEVELOP_TIMEOUT_MS = 45_000

let broken = false
let seq = 0

export const rawDevelopAvailable = () => !broken && simdSupported()

/**
 * Develop RAW `file` (a Blob/File) into an image/jpeg Blob, or null when
 * develop is unavailable/failed/stale/disabled. `budget` gates it: off when
 * `rawDevelop === false` or memory pressure ≥ 2; `rawDevelopMaxMP` caps the
 * sensor size (bounds the develop peak, profile-scaled by policy).
 */
export const developRaw = async (file, { budget = {}, half = true, revision = null } = {}) => {
    if (!rawDevelopAvailable() || !file) return null
    if (budget.rawDevelop === false || (budget.pressureLevel || 0) >= 2) return null
    if (file.size <= 0 || file.size > MAX_BYTES) return null

    const bytes = await file.arrayBuffer()
    const maxMP = Math.max(0, Math.round(Number(budget.rawDevelopMaxMP) || 0))

    const result = await enqueueHeavy('raw-develop', () => {
        let worker = null
        try {
            worker = new Worker(new URL('./raw-develop-worker.js', import.meta.url), { type: 'module' })
        } catch (err) {
            console.warn('[seglab][raw] develop worker unavailable:', err?.message)
            broken = true
            return null
        }
        seq += 1
        const requestId = `rd${seq}`
        // The worker handles exactly one develop, then we terminate it to
        // reclaim the grown heap — so there is no persistent worker to leak.
        return new Promise((resolve) => {
            const done = (value) => {
                clearTimeout(timer)
                try { worker.terminate() } catch { /* dead */ }
                resolve(value)
            }
            const timer = setTimeout(() => {
                console.warn('[seglab][raw] develop timed out; keeping the no-preview error')
                done(null)
            }, DEVELOP_TIMEOUT_MS)
            worker.onmessage = (event) => {
                const data = event.data || {}
                if (data.requestId !== requestId) return
                if (data.type === 'result') done(data)
                else {
                    console.warn('[seglab][raw] develop failed; no embedded preview to fall back to:', data.error)
                    done(null)
                }
            }
            worker.onerror = (event) => {
                console.warn('[seglab][raw] develop worker crashed:', event?.message)
                broken = true // one crash is enough; no retry loop
                done(null)
            }
            try {
                worker.postMessage({ type: 'develop', requestId, bytes, options: { half, maxMP } }, [bytes])
            } catch (err) {
                console.warn('[seglab][raw] develop post failed:', err?.message)
                done(null)
            }
        })
    }, { priority: 'import', revision })

    if (result === STALE || !result || !result.jpeg) return null
    return {
        blob: new Blob([result.jpeg], { type: 'image/jpeg' }),
        width: result.width,
        height: result.height,
    }
}

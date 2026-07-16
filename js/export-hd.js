/**
 * export-hd — explicit, bounded cutout export
 * ---------------------------------------------
 * Runs ONLY from a user's export action, never from selection flow. The
 * original Blob is re-decoded bounded to policy.exportMaxSide /
 * exportMaxMP (no best-effort full resolution); demo/canvas documents
 * export at proxy size. The refine worker is released first so the export
 * decode is the only heavy resident.
 */

import { PRIORITY, enqueueHeavy } from './heavy-job-queue.js'
import { exportComposite } from './decode-client.js'
import { disposeCv } from './cv-refine-client.js'

/**
 * @param {{ doc: { revision, blob }, mask: { alpha, width, height },
 *           policy: { exportMaxSide, exportMaxMP }, sourceCanvas }} req
 *   `mask.alpha` must be a private copy — its buffer transfers away.
 * @returns {Promise<{ blob: Blob, width, height, reduced: boolean }>}
 */
export const exportCutout = ({ doc, mask, policy, sourceCanvas }) => {
    const revision = doc.revision
    return enqueueHeavy('export', async () => {
        disposeCv()
        const caps = { maxSide: policy.exportMaxSide, maxMP: policy.exportMaxMP }
        if (doc.blob) return exportComposite({ blob: doc.blob, mask, caps }, revision)
        const bitmap = await createImageBitmap(sourceCanvas)
        return exportComposite({ bitmap, mask, caps }, revision)
    }, { priority: PRIORITY.interaction, revision, isCurrent: () => revision === doc.revision })
}

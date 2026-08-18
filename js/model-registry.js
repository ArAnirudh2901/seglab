/**
 * model-registry — the lightweight "notepad" of models this browser already
 * holds. localStorage, read synchronously at boot: the UI can say "sam21 ✓"
 * without opening Cache Storage, OPFS, or fetching a manifest. Written when a
 * download completes / a lane becomes ready; absence of an entry means "will
 * download on first use", never an error. Registry is a HINT for display —
 * the Cache Storage / OPFS copy is the truth.
 */

const KEY = 'seglab.modelRegistry.v1'

let cache = null

const read = () => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {} } catch { return {} }
}

/** { sam21: {device, notedAt, …}, yoloe: {…}, clip: {…} } */
export const modelRegistry = () => (cache ??= read())

/** Record (or update) a model the browser now holds. Merges meta. */
export const noteModel = (id, meta = {}) => {
    if (!id) return null
    const reg = modelRegistry()
    const prev = reg[id] || {}
    reg[id] = { ...prev, ...meta, notedAt: prev.notedAt || Date.now() }
    try { localStorage.setItem(KEY, JSON.stringify(reg)) } catch { /* quota / private mode — hint only */ }
    return reg[id]
}

export const isModelNoted = (id) => Boolean(modelRegistry()[id])

/** Lane id from a download-progress event's model name. */
export const laneOfModel = (name = '') => {
    if (/clip/i.test(name)) return 'clip' // text encoder + its BPE/table assets
    if (/yoloe/i.test(name)) return 'yoloe'
    if (/sam/i.test(name)) return 'sam21' // one mask lane, so any SAM is that one
    return null
}

/**
 * model-registry — the lightweight "notepad" of models this browser already
 * holds. localStorage, read synchronously at boot: the UI can say "slimsam ✓"
 * without opening Cache Storage, OPFS, or fetching a manifest. Written when a
 * download completes / a lane becomes ready; absence of an entry means "will
 * download on first use", never an error. Registry is a HINT for display —
 * loading still goes through model-assets/transformers cache as the truth.
 */

const KEY = 'seglab.modelRegistry.v1'

let cache = null

const read = () => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {} } catch { return {} }
}

/** { slimsam: {device, notedAt, …}, gdino: {…}, owlv2: {…}, yoloe: {…} } */
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

/** Lane id from a transformers/detect progress event's model name. */
export const laneOfModel = (name = '') => {
    if (/yolo-?world/i.test(name)) return 'yoloworld'
    if (/clip/i.test(name)) return 'clip'
    if (/yoloe/i.test(name)) return 'yoloe'
    if (/slimsam|sam/i.test(name)) return 'slimsam'
    return null
}

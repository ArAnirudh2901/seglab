/**
 * search-taxonomy (pure) — main-class → sub-class hierarchy over the YOLOE
 * baked vocabulary, plus the sub-class facet + autocomplete logic. Testable
 * headless like text-core (its only import is text-core's whole-word matcher).
 *
 * The model's vocabulary is FLAT (~4585 labels) and open-vocabulary detection
 * has no notion of "flower" containing "rose". Three jobs here:
 *   • recall — a search for a main class ("flower") matches every one of its
 *     vocab-child kinds ("rose", "tulip", "sunflower"), which are separate
 *     classes the flat matcher would otherwise miss. This is the quality win.
 *   • facets — after ONE detection pass, group the results into sub-classes on
 *     three axes (colour · kind · size/position) computed client-side, so
 *     refining costs no extra inference.
 *   • autocomplete — suggest main classes, their kinds, and colour combos as
 *     the user types, all from a static index (no model).
 *
 * Every `kind` below is a REAL vocab label (verified against the .vocab.json);
 * suggesting a kind YOLOE cannot detect would be a dead end. Categories the
 * taxonomy doesn't cover still work as before via freeform phrase matching.
 */
import { phraseMatchesLabel } from './text-core.js'

/** main class → { kinds: real vocab sub-labels, colors: common colours (hints
 *  for autocomplete only — colour is never a detected class) }. */
export const TAXONOMY = {
    flower: {
        kinds: ['rose', 'tulip', 'sunflower', 'daisy', 'orchid', 'lily', 'lotus', 'poppy', 'hydrangea', 'marigold', 'peony', 'iris', 'dandelion', 'carnation', 'hibiscus flower', 'water lily', 'cherry blossom', 'wildflower', 'blossom', 'bloom', 'chinese rose'],
        colors: ['red', 'pink', 'white', 'yellow', 'purple'],
    },
    tree: {
        kinds: ['oak tree', 'palm tree', 'birch tree', 'cherry tree', 'apple tree', 'christmas tree', 'fir tree', 'coconut tree', 'olive tree', 'banana tree', 'banyan tree', 'cypress tree', 'eucalyptus tree', 'ginkgo tree', 'beech tree', 'autumn tree', 'fruit tree', 'orange tree', 'bamboo', 'cactus', 'fern', 'bush', 'shrub', 'houseplant'],
        colors: ['green', 'brown'],
    },
    animal: {
        kinds: ['dog', 'cat', 'horse', 'cow', 'sheep', 'goat', 'lion', 'tiger', 'bear', 'elephant', 'zebra', 'giraffe', 'deer', 'rabbit', 'fox', 'wolf', 'monkey', 'squirrel', 'panda', 'kangaroo', 'koala', 'camel', 'donkey', 'pig', 'hippo', 'rhinoceros', 'leopard', 'cheetah', 'buffalo', 'bison', 'moose', 'antelope', 'raccoon', 'otter', 'beaver', 'hedgehog', 'bat', 'mouse', 'rat', 'hamster', 'ferret', 'alpaca', 'reindeer', 'pony'],
        colors: ['brown', 'white', 'black'],
    },
    bird: {
        kinds: ['eagle', 'owl', 'parrot', 'duck', 'goose', 'swan', 'penguin', 'flamingo', 'pigeon', 'sparrow', 'crow', 'peacock', 'chicken', 'hummingbird', 'seabird', 'hen', 'turkey', 'ostrich', 'pelican', 'woodpecker', 'robin', 'cardinal', 'blackbird', 'bluebird', 'magpie', 'canary', 'toucan', 'heron', 'stork', 'crane', 'falcon', 'vulture', 'raven'],
        colors: ['white', 'black', 'brown'],
    },
    vehicle: {
        kinds: ['car', 'truck', 'bus', 'van', 'minivan', 'motorcycle', 'bicycle', 'scooter', 'train', 'boat', 'plane', 'jeep', 'ambulance', 'taxi', 'tractor', 'forklift', 'helicopter', 'yacht', 'canoe', 'kayak', 'ferry', 'submarine', 'tank', 'trolley', 'sedan', 'suv', 'sports car', 'race car', 'fire truck', 'tow truck', 'police car', 'golf cart'],
        colors: ['red', 'white', 'black', 'blue'],
    },
    car: {
        kinds: ['sports car', 'race car', 'sedan', 'suv', 'taxi', 'police car', 'convertible', 'muscle car', 'toy car'],
        colors: ['red', 'white', 'black', 'blue'],
    },
    dog: {
        kinds: ['bulldog', 'french bulldog', 'poodle', 'beagle', 'sheepdog', 'husky', 'labrador', 'chihuahua', 'pug', 'dachshund', 'rottweiler', 'dalmatian', 'corgi', 'greyhound', 'boxer', 'shepherd', 'retriever'],
        colors: ['brown', 'white', 'black'],
    },
    fruit: {
        kinds: ['apple', 'banana', 'orange', 'grape', 'mango', 'strawberry', 'cherry', 'peach', 'pear', 'watermelon', 'pineapple', 'lemon', 'lime', 'blueberry', 'raspberry', 'blackberry', 'cranberry', 'kiwi', 'plum', 'apricot', 'fig', 'pomegranate', 'papaya', 'coconut', 'avocado', 'melon', 'grapefruit', 'tangerine', 'mandarin orange', 'passion fruit', 'starfruit', 'durian', 'persimmon', 'mulberry'],
        colors: ['red', 'green', 'yellow', 'orange'],
    },
    furniture: {
        kinds: ['chair', 'table', 'couch', 'bed', 'cabinet', 'shelf', 'stool', 'bench', 'lamp', 'armchair', 'bookshelf', 'dresser', 'nightstand', 'bunk bed', 'side table', 'rocking chair', 'office chair', 'folding chair', 'bar stool', 'table lamp'],
        colors: ['brown', 'white', 'black'],
    },
}

export const MAIN_CLASSES = Object.keys(TAXONOMY)

// kind → main (first main that claims it; a kind under several mains keeps the
// earliest, which is fine — it only seeds facet grouping and autocomplete).
const KIND_TO_MAIN = new Map()
for (const main of MAIN_CLASSES) {
    for (const kind of TAXONOMY[main].kinds) if (!KIND_TO_MAIN.has(kind)) KIND_TO_MAIN.set(kind, main)
}

/** Basic colours the facet/autocomplete layer knows, plus common synonyms that
 *  map onto them (mirrors text-core's COLOR_WORDS so a typed "violet"/"grey"
 *  resolves the same way). */
export const FACET_COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'white', 'gray', 'black']
const COLOR_SYNONYMS = new Map([...FACET_COLORS.map((c) => [c, c]), ['violet', 'purple'], ['grey', 'gray']])

/**
 * Expand a normalized object phrase into the label set a search should accept.
 *   • a main class → the class label + all its kinds (the recall win)
 *   • a specific kind → just that kind (a rose search shouldn't select tulips)
 *   • anything else → null (caller keeps its flat matcher)
 */
export const expandQuery = (objectCore) => {
    const core = String(objectCore || '').trim().toLowerCase()
    if (!core) return null
    if (TAXONOMY[core]) return { main: core, labels: [core, ...TAXONOMY[core].kinds] }
    if (KIND_TO_MAIN.has(core)) return { main: KIND_TO_MAIN.get(core), labels: [core] }
    return null
}

/** True when a detected `label` satisfies the phrase — taxonomy-aware: a main
 *  class matches any of its kinds. Falls back to text-core's flat whole-word
 *  matcher for phrases the taxonomy doesn't cover. */
export const labelMatchesQuery = (objectCore, label) => {
    const expanded = expandQuery(objectCore)
    if (!expanded) return phraseMatchesLabel(objectCore, label)
    return expanded.labels.some((l) => phraseMatchesLabel(l, label))
}

/** size + position of a proxy-coord box within the w×h image, for the
 *  size/position sub-class axes. `foreground` = a large box low in the frame. */
export const regionOf = (box, w, h) => {
    if (!(w > 0) || !(h > 0)) return { size: 'medium', where: 'center', foreground: false }
    const area = (Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1])) / (w * h)
    const cx = ((box[0] + box[2]) / 2) / w
    const cy = ((box[1] + box[3]) / 2) / h
    const size = area >= 0.12 ? 'large' : area <= 0.025 ? 'small' : 'medium'
    const where = cx < 0.34 ? 'left' : cx > 0.66 ? 'right' : 'center'
    return { size, where, foreground: size === 'large' && cy >= 0.55 }
}

const pushFacet = (map, value, i) => {
    if (!value) return
    if (!map.has(value)) map.set(value, [])
    map.get(value).push(i)
}

// An axis is only worth showing when it actually partitions the results:
// ≥2 candidates and ≥2 distinct values (one shared value refines nothing).
const facetAxis = (map) => (map.size < 2 ? [] : [...map.entries()]
    .map(([value, idx]) => ({ value, label: value, count: idx.length, idx }))
    .sort((a, b) => b.count - a.count))

/**
 * Group ranked candidates into sub-class facets on three axes. Each candidate
 * carries `.label` (its detected class → kind axis) and an optional `.color`
 * (dominant-colour bucket the caller tagged from pixels → colour axis); the
 * region axis is derived here from `.box` (proxy coords) and image `dims`.
 * Returns { colour, kind, size, position }, each [{ value, label, count, idx }]
 * or empty when that axis doesn't split the results.
 */
export const buildFacets = (candidates, dims = {}) => {
    const cands = candidates || []
    const colour = new Map()
    const kind = new Map()
    const size = new Map()
    const position = new Map()
    const w = dims.width || 0
    const h = dims.height || 0
    cands.forEach((c, i) => {
        pushFacet(colour, c.color, i)
        pushFacet(kind, String(c.label || '').toLowerCase().trim() || null, i)
        const r = regionOf(c.box, w, h)
        pushFacet(size, r.foreground ? 'foreground' : r.size, i)
        pushFacet(position, r.where, i)
    })
    if (cands.length < 2) return { colour: [], kind: [], size: [], position: [] }
    return { colour: facetAxis(colour), kind: facetAxis(kind), size: facetAxis(size), position: facetAxis(position) }
}

/** Leading/trailing colour token in a typed prefix → { color, rest }. Lets
 *  "red flo" and "flo" both drive main-class suggestions. */
const splitColor = (words) => {
    let color = null
    const rest = []
    for (const w of words) {
        const c = COLOR_SYNONYMS.get(w)
        if (c && !color) color = c
        else rest.push(w)
    }
    return { color, rest: rest.join(' ') }
}

const rankPrefix = (candidate, needle) => (candidate === needle ? 0 : candidate.startsWith(needle) ? 1 : candidate.includes(needle) ? 2 : -1)

/**
 * Autocomplete the taxonomy for a typed prefix. Returns up to `limit` rows
 * { text (the phrase to search), group: 'category'|'kind'|'colour', detail },
 * ranked prefix-first. Pure string work — never touches the model.
 */
export const suggest = (prefix, { limit = 8 } = {}) => {
    const words = String(prefix || '').trim().toLowerCase().replace(/\s+/g, ' ').split(' ').filter(Boolean)
    if (words.length === 0) return []
    const { color, rest } = splitColor(words)
    const needle = rest || (color ? '' : words.join(' '))
    if (!needle && !color) return []
    const withColor = (t) => (color ? `${color} ${t}` : t)
    const seen = new Set()
    const rows = []
    const add = (text, group, detail, rank) => {
        const key = text.toLowerCase()
        if (seen.has(key)) return
        seen.add(key)
        rows.push({ text, group, detail, rank })
    }

    for (const main of MAIN_CLASSES) {
        const r = needle ? rankPrefix(main, needle) : 0
        if (r < 0) continue
        const kinds = TAXONOMY[main].kinds
        add(withColor(main), 'category', `${kinds.length} kinds`, r)
        // Colour combos only when the user hasn't already typed one.
        if (!color) for (const c of TAXONOMY[main].colors.slice(0, 2)) add(`${c} ${main}`, 'colour', main, r + 0.5)
    }
    for (const [kind, main] of KIND_TO_MAIN) {
        const r = needle ? rankPrefix(kind, needle) : -1
        if (r < 0) continue
        add(withColor(kind), 'kind', main, r + 0.25)
    }
    return rows.sort((a, b) => a.rank - b.rank).slice(0, limit).map(({ rank, ...row }) => row)
}

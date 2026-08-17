/**
 * clip-tokenizer — OpenAI CLIP byte-pair encoding, pure JS.
 *
 * MobileCLIP2 uses CLIP's tokenizer unchanged, so an arbitrary phrase becomes
 * the exact 77-token sequence the text tower was trained on. Any deviation here
 * is silent: the model still returns a confident-looking vector, just for the
 * wrong string.
 *
 * Only `merges.txt` ships (~0.5 MB). The 49408-entry vocabulary is DERIVED from
 * it the way the reference implementation does — 256 byte symbols, the same 256
 * with a `</w>` word-final marker, one entry per merge, then the two specials —
 * so the redundant 0.9 MB vocab.json stays out of the bundle. The build step
 * asserts the derivation matches upstream (scripts/export-clip-text.py).
 */

export const CONTEXT = 77       // CLIP's fixed sequence length
export const SOT = 49406        // <|startoftext|>
export const EOT = 49407        // <|endoftext|>
export const VOCAB = 49408

const bpeURL = new URL('../models/clip-text/merges.txt', import.meta.url).href

/** GPT-2/CLIP byte→printable-codepoint map: keeps every byte representable as a
 *  single character so BPE can operate on strings.
 *
 *  Returns the symbols in `bs` order — printable ranges first, then the bytes
 *  displaced to U+0100+ — because that is the order the vocabulary's first 512
 *  ids follow. Building them in plain byte order 0..255 yields the same SET but
 *  shifts every one of those ids, which silently mistokenizes punctuation and
 *  non-ASCII text. Asserted against upstream vocab.json at build time. */
const bytesToUnicode = () => {
    const bs = []
    for (let b = 0x21; b <= 0x7e; b += 1) bs.push(b)
    for (let b = 0xa1; b <= 0xac; b += 1) bs.push(b)
    for (let b = 0xae; b <= 0xff; b += 1) bs.push(b)
    const cs = bs.slice()
    const present = new Set(bs)
    let n = 0
    for (let b = 0; b < 256; b += 1) {
        if (!present.has(b)) { bs.push(b); cs.push(256 + n); n += 1 }
    }
    const map = new Array(256)   // byte → symbol, for encoding text
    const order = []             // vocab order, for deriving the id table
    for (let i = 0; i < bs.length; i += 1) {
        const sym = String.fromCodePoint(cs[i])
        map[bs[i]] = sym
        order.push(sym)
    }
    return { map, order }
}

// 'd/'s/... must be split off before the letter run, hence the alternation order.
const PAT = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+/gu

let byteEncoder = null
let ranks = null        // Map<"a b", rank>
let encoder = null      // Map<symbol, id>
let loadPromise = null
const bpeCache = new Map()

const build = (mergesText) => {
    const { map, order } = bytesToUnicode()
    byteEncoder = map
    // Line 0 is a version banner; the reference slices [1 : 49152-256-2+1].
    const merges = mergesText.split('\n').slice(1, 49152 - 256 - 2 + 1).filter(Boolean)
    ranks = new Map()
    merges.forEach((m, i) => ranks.set(m, i))

    const vocab = order.slice()
    for (const s of order) vocab.push(`${s}</w>`)
    for (const m of merges) vocab.push(m.split(' ').join(''))
    vocab.push('<|startoftext|>', '<|endoftext|>')
    encoder = new Map()
    vocab.forEach((t, i) => { if (!encoder.has(t)) encoder.set(t, i) })
    if (encoder.size !== VOCAB) {
        throw new Error(`clip-tokenizer: derived vocab ${encoder.size}, expected ${VOCAB}`)
    }
}

export const loadTokenizer = (progress) => {
    loadPromise ??= (async () => {
        progress?.({ status: 'progress', name: 'clip-bpe', progress: 0 })
        const text = await (await fetch(bpeURL)).text()
        build(text)
        progress?.({ status: 'done', name: 'clip-bpe' })
        return true
    })()
    return loadPromise
}

/** Greedy BPE over one pre-tokenized word; `</w>` marks the final symbol. */
const bpe = (token) => {
    const hit = bpeCache.get(token)
    if (hit) return hit
    let word = [...token.slice(0, -1), `${token[token.length - 1]}</w>`]
    if (word.length === 1) {
        bpeCache.set(token, word)
        return word
    }
    for (;;) {
        let best = null
        let bestRank = Infinity
        for (let i = 0; i < word.length - 1; i += 1) {
            const rank = ranks.get(`${word[i]} ${word[i + 1]}`)
            if (rank !== undefined && rank < bestRank) { bestRank = rank; best = i }
        }
        if (best === null) break
        // In place: the spread form built three arrays per merge, and a merge
        // runs once per symbol pair in the word.
        word.splice(best, 2, word[best] + word[best + 1])
        if (word.length === 1) break
    }
    bpeCache.set(token, word)
    return word
}

const clean = (text) => String(text || '').replace(/\s+/g, ' ').trim().toLowerCase()

const utf8 = new TextEncoder()

/** One phrase → token ids WITHOUT the specials. */
const encodeOne = (text) => {
    const out = []
    for (const m of clean(text).matchAll(PAT)) {
        // UTF-8 bytes → byte symbols, so any script survives the round trip.
        let sym = ''
        for (const b of utf8.encode(m[0])) sym += byteEncoder[b]
        for (const piece of bpe(sym)) {
            const id = encoder.get(piece)
            if (id !== undefined) out.push(id)
        }
    }
    return out
}

/**
 * `texts` → Int32Array [texts.length * CONTEXT], SOT/EOT wrapped and zero
 * padded. Over-long phrases are truncated with EOT forced into the last slot,
 * matching CLIP's `truncate=True` — the text tower reads the EOT position, so
 * losing it would take the embedding from the wrong token.
 */
export const tokenize = (texts) => {
    if (!encoder) throw new Error('clip-tokenizer: loadTokenizer() first')
    const n = texts.length
    const out = new Int32Array(n * CONTEXT)
    for (let i = 0; i < n; i += 1) {
        const ids = [SOT, ...encodeOne(texts[i]), EOT]
        if (ids.length > CONTEXT) {
            ids.length = CONTEXT
            ids[CONTEXT - 1] = EOT
        }
        out.set(ids, i * CONTEXT)
    }
    return out
}

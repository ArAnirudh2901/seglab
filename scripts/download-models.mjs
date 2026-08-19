#!/usr/bin/env node
/**
 * Vendors every runtime asset SEGLAB needs, so the app serves fully offline:
 *   lib/ort-web/…              — onnxruntime-web (WebGPU ESM bundle + wasm loader)
 *   models/manifest.json       — presence signal read at runtime
 *
 * Everything else is BUILT, not fetched — no hub publishes the exact artifacts
 * this app runs — so this script verifies those exist and names the script that
 * produces each one.
 *
 * lib/ and models/ are gitignored (~25 MB fetched, ~92 MB built core).
 * Idempotent: complete files are skipped.
 *
 * Usage: node scripts/download-models.mjs
 *
 *   (core)       click/box/lasso selection + export run with no network.
 */

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// The ONE runtime. There is no second ORT build and no transformers.js: the
// SlimSAM lane that needed them is gone, and with it ~35 MB of wasm and a
// version-pin check that existed only to keep those two in step.
//
// MUST be >= 1.24.3. The 1.22 WebGPU fp16 kernels compute SAM 2.1's encoder
// WRONG — silently, with a confident-looking mask covering 99.6% of the frame
// (spikes/sam21/FINDINGS.md §1.2). fp16 is what keeps resident memory under
// 2 GB, so downgrading this re-breaks the memory contract as well as quality.
// spikes/sam21/ortver.html is the gate: re-run it on any change here.
const ORT_WEB_VERSION = '1.27.0'

const ORT_WEB_FILES = [
    'ort.webgpu.bundle.min.mjs',
    // 1.23+ renamed the WebGPU-capable wasm from .jsep to .asyncify; the bundle
    // dynamically imports it, so the old names produce "no available backend".
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
]

// The mask lane. Core, not optional — without these the app cannot segment at
// all, so a missing one is a hard failure rather than a note.
const CORE_ASSETS = [
    ['models/sam21/encoder.fp16.onnx', 'python3 scripts/export-sam21.py'],
    ['models/sam21/decoder.fp16.onnx', 'python3 scripts/export-sam21.py'],
    ['models/sam21/model.json', 'python3 scripts/export-sam21.py'],
]

const jobs = ORT_WEB_FILES.map((f) => ({
    url: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_WEB_VERSION}/dist/${f}`,
    dest: `lib/ort-web/${f}`,
}))

const log = (msg) => console.log(`[download-models] ${msg}`)

// ORT-Web forwards only a fixed key list from a WebGPU EP options object to the
// native EP, and the buffer cache modes are not on it — so the one setting that
// keeps this app under its RAM ceiling is unreachable through public API.
// (js/ort-loader.js webgpuEP: Bucket 1128 MB of GPU process vs 290 MB with
// lazyRelease, identical logits.) Forward an `epConfig` bag instead.
//
// A one-line, anchored patch on a DERIVED artifact. It throws if the anchor is
// gone, so an ORT bump fails here rather than silently costing ~840 MB;
// verify.mjs gates the vendored copy carrying it.
const ORT_EP_ANCHOR = 'S.validationMode&&ot(l,"validationMode",S.validationMode,s)'
const ORT_EP_PATCH = ',S.epConfig&&Object.entries(S.epConfig).forEach(([Ck,Cv])=>ot(l,Ck,String(Cv),s))'
const patchOrtBundle = (buf) => {
    const src = buf.toString('utf8')
    if (src.includes('S.epConfig')) return buf
    if (!src.includes(ORT_EP_ANCHOR)) {
        throw new Error(`ORT ${ORT_WEB_VERSION}: epConfig anchor missing in the freshly downloaded bundle`
            + ' — upstream minified shape changed; re-derive ORT_EP_ANCHOR in download-models.mjs.'
            + ' (If this file was NOT just downloaded, delete lib/ort-web/ and re-run.)')
    }
    return Buffer.from(src.replace(ORT_EP_ANCHOR, ORT_EP_ANCHOR + ORT_EP_PATCH), 'utf8')
}
const mb = (n) => `${(n / 1024 / 1024).toFixed(2)} MB`

// Idempotency is manifest-based: CDN content-length reports the compressed
// size when content-encoding is active, so it cannot be compared to disk.
const priorManifest = await readFile(path.join(ROOT, 'models', 'manifest.json'), 'utf8')
    .then((s) => JSON.parse(s))
    .catch(() => null)
const priorBytes = new Map((priorManifest?.files || []).map((f) => [f.path, f.bytes]))
// Size is not identity. The runtime is PINNED, so a bump has to invalidate every
// vendored ORT file even when the stale one happens to match its recorded size —
// otherwise the skip path hands patchOrtBundle a bundle from the previous
// version and the run fails claiming the patch anchor is gone.
const ortStale = priorManifest?.onnxruntimeWeb !== ORT_WEB_VERSION
if (ortStale && priorManifest) {
    log(`ORT pin changed (${priorManifest.onnxruntimeWeb || 'unrecorded'} -> ${ORT_WEB_VERSION}) — refetching lib/ort-web/`)
}

const download = async ({ url, dest, optional }) => {
    const target = path.join(ROOT, dest)
    const local = await stat(target).catch(() => null)
    const pinnedRuntime = dest.startsWith('lib/ort-web/')
    if (local && priorBytes.get(dest) === local.size && !(pinnedRuntime && ortStale)) {
        log(`have ${dest} (${mb(local.size)})`)
        return { path: dest, bytes: local.size }
    }
    const res = await fetch(url)
    if (!res.ok) {
        if (optional) { log(`skip (optional, HTTP ${res.status}): ${dest}`); return null }
        throw new Error(`GET failed for ${url} (HTTP ${res.status})`)
    }
    let buf = Buffer.from(await res.arrayBuffer())
    const remoteBytes = Number(res.headers.get('content-length')) || null
    const encoded = !!res.headers.get('content-encoding')
    if (!encoded && remoteBytes && buf.length !== remoteBytes) {
        throw new Error(`size mismatch for ${dest}: got ${buf.length}, expected ${remoteBytes}`)
    }
    if (buf.length === 0) throw new Error(`empty download for ${dest}`)
    if (dest.endsWith('ort.webgpu.bundle.min.mjs')) buf = patchOrtBundle(buf)
    await mkdir(path.dirname(target), { recursive: true })
    await writeFile(target, buf)
    log(`got  ${dest} (${mb(buf.length)})`)
    return { path: dest, bytes: buf.length }
}

/** Verify built assets rather than fetching them, and say exactly how to
 *  produce a missing one instead of failing with a bare path. */
const verifyBuilt = async (assets, files) => {
    const missing = []
    for (const [rel, how] of assets) {
        const local = await stat(path.join(ROOT, rel)).catch(() => null)
        if (local?.size) { log(`have ${rel} (${mb(local.size)})`); files.push({ path: rel, bytes: local.size }) }
        else missing.push([rel, how])
    }
    for (const [rel, how] of missing) log(`MISSING ${rel} — build it with: ${how}`)
    return missing
}

const files = []
for (const job of jobs) {
    const entry = await download(job)
    if (entry) files.push(entry)
}

// Also on the cache-hit path: a checked-out tree already has the bundle at the
// manifest's size, so nothing would re-download it and the patch would be lost.
{
    const entry = files.find((f) => f.path.endsWith('ort.webgpu.bundle.min.mjs'))
    const target = path.join(ROOT, entry.path)
    const patched = patchOrtBundle(await readFile(target))
    if (patched.length !== entry.bytes) {
        await writeFile(target, patched)
        entry.bytes = patched.length
        log(`patched ${entry.path} (epConfig passthrough)`)
    }
}

const missingCore = await verifyBuilt(CORE_ASSETS, files)

const manifest = {
    onnxruntimeWeb: ORT_WEB_VERSION,
    lane: 'sam2.1-small',
    precision: 'fp16',
    generatedAt: new Date().toISOString(),
    files,
}
await mkdir(path.join(ROOT, 'models'), { recursive: true })
await writeFile(path.join(ROOT, 'models', 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
const total = files.reduce((sum, f) => sum + f.bytes, 0)
log(`done — ${files.length} files, ${mb(total)} total; wrote models/manifest.json`)
// Loud, and last, so it is the thing left on screen: the app cannot segment
// without these, and a silent manifest would let that surface as a runtime bug.
if (missingCore.length) {
    throw new Error(`mask lane incomplete — ${missingCore.length} core asset(s) missing; see MISSING lines above`)
}

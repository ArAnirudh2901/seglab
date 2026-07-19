#!/usr/bin/env node
/**
 * Vendors every runtime asset SEGLAB needs, so the app serves fully offline:
 *   lib/transformers/transformers.min.js     — pinned transformers.js bundle
 *   lib/ort/ort-wasm-simd-threaded*.{mjs,wasm} — ORT wasm (asyncify = Chromium/FF
 *                                              pair, plain = Safari pair)
 *   models/Xenova/slimsam-77-uniform/…       — SlimSAM fp32 (WebGPU) + q8 (WASM)
 *   models/onnx-community/grounding-dino-…/…  — Grounding DINO q4f16, --detector
 *   models/onnx-community/owlv2-…/…          — OWLv2 q8, only with --detector
 *   models/manifest.json                     — presence signal read at runtime
 *
 * lib/ and models/ are gitignored (~90 MB core, +314 MB with --detector).
 * Idempotent: complete files are skipped. Fails loudly if the
 * transformers→ORT version pin ever drifts.
 *
 * Usage: node scripts/download-models.mjs [--detector]
 *
 *   (core)       click/box/lasso selection + export run with no network.
 *   --detector   also vendors both text-select lanes so "phrase → boxes" is
 *                offline on any device: Grounding DINO q4f16 (accelerated) and
 *                OWLv2 q8 (universal fallback). Skipped by default — 314 MB,
 *                and the feature is optional/disposable at runtime.
 */

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const TRANSFORMERS_VERSION = '4.2.0'
// Must equal dependencies["onnxruntime-web"] of the pinned transformers build.
const ORT_VERSION = '1.26.0-dev.20260416-b7804b056c'
const SLIMSAM = 'Xenova/slimsam-77-uniform'
// Must equal DETECTORS in js/detect-engine.js (model + weightFile per lane).
const OWLV2 = 'onnx-community/owlv2-base-patch16-ensemble-ONNX'
const GDINO = 'onnx-community/grounding-dino-tiny-ONNX'

const withDetector = process.argv.includes('--detector')

const ORT_FILES = [
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.wasm',
]
// fp32 (unsuffixed) serves device:webgpu, q8 (_quantized) serves device:wasm —
// exactly the dtypes pinned in js/sam-engine.js LANES.draft.
const SLIMSAM_FILES = [
    'config.json',
    'preprocessor_config.json',
    { file: 'quantize_config.json', optional: true },
    'onnx/vision_encoder.onnx',
    'onnx/vision_encoder_quantized.onnx',
    'onnx/prompt_encoder_mask_decoder.onnx',
    'onnx/prompt_encoder_mask_decoder_quantized.onnx',
]

// q8 only — the exact dtype js/detect-engine.js pins for the OWLv2 WASM lane.
const OWLV2_FILES = [
    'config.json',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'onnx/model_quantized.onnx',
]

// q4f16 only — the exact dtype js/detect-engine.js pins for the accelerated
// Grounding DINO WebGPU lane.
const GDINO_FILES = [
    'config.json',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'onnx/model_q4f16.onnx',
]

const hubJobs = (repo, files) => files.map((entry) => {
    const file = typeof entry === 'string' ? entry : entry.file
    return {
        url: `https://huggingface.co/${repo}/resolve/main/${file}`,
        dest: `models/${repo}/${file}`,
        optional: typeof entry === 'object' && entry.optional,
    }
})

const jobs = [
    {
        url: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/dist/transformers.min.js`,
        dest: 'lib/transformers/transformers.min.js',
    },
    ...ORT_FILES.map((f) => ({
        url: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/${f}`,
        dest: `lib/ort/${f}`,
    })),
    ...hubJobs(SLIMSAM, SLIMSAM_FILES),
    ...(withDetector ? hubJobs(GDINO, GDINO_FILES) : []),
    ...(withDetector ? hubJobs(OWLV2, OWLV2_FILES) : []),
]

const log = (msg) => console.log(`[download-models] ${msg}`)
const mb = (n) => `${(n / 1024 / 1024).toFixed(2)} MB`

const assertOrtPin = async () => {
    const res = await fetch(`https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/package.json`)
    if (!res.ok) throw new Error(`cannot read transformers package.json (HTTP ${res.status})`)
    const pinned = (await res.json()).dependencies?.['onnxruntime-web']
    if (pinned !== ORT_VERSION) {
        throw new Error(`ORT pin drift: transformers@${TRANSFORMERS_VERSION} wants onnxruntime-web@${pinned}, `
            + `this script vendors ${ORT_VERSION} — update ORT_VERSION and re-run`)
    }
    log(`pin ok: transformers@${TRANSFORMERS_VERSION} ↔ onnxruntime-web@${ORT_VERSION}`)
}

// Idempotency is manifest-based: CDN content-length reports the compressed
// size when content-encoding is active, so it cannot be compared to disk.
const priorBytes = await readFile(path.join(ROOT, 'models', 'manifest.json'), 'utf8')
    .then((s) => new Map(JSON.parse(s).files.map((f) => [f.path, f.bytes])))
    .catch(() => new Map())

const download = async ({ url, dest, optional }) => {
    const target = path.join(ROOT, dest)
    const local = await stat(target).catch(() => null)
    if (local && priorBytes.get(dest) === local.size) {
        log(`have ${dest} (${mb(local.size)})`)
        return { path: dest, bytes: local.size }
    }
    const res = await fetch(url)
    if (!res.ok) {
        if (optional) { log(`skip (optional, HTTP ${res.status}): ${dest}`); return null }
        throw new Error(`GET failed for ${url} (HTTP ${res.status})`)
    }
    const buf = Buffer.from(await res.arrayBuffer())
    const remoteBytes = Number(res.headers.get('content-length')) || null
    const encoded = !!res.headers.get('content-encoding')
    if (!encoded && remoteBytes && buf.length !== remoteBytes) {
        throw new Error(`size mismatch for ${dest}: got ${buf.length}, expected ${remoteBytes}`)
    }
    if (buf.length === 0) throw new Error(`empty download for ${dest}`)
    await mkdir(path.dirname(target), { recursive: true })
    await writeFile(target, buf)
    log(`got  ${dest} (${mb(buf.length)})`)
    return { path: dest, bytes: buf.length }
}

await assertOrtPin()
const files = []
for (const job of jobs) {
    const entry = await download(job)
    if (entry) files.push(entry)
}
const manifest = {
    transformers: TRANSFORMERS_VERSION,
    onnxruntimeWeb: ORT_VERSION,
    model: SLIMSAM,
    models: withDetector ? [SLIMSAM, GDINO, OWLV2] : [SLIMSAM],
    detector: withDetector ? OWLV2 : null,
    detectors: withDetector ? [GDINO, OWLV2] : [],
    generatedAt: new Date().toISOString(),
    files,
}
await mkdir(path.join(ROOT, 'models'), { recursive: true })
await writeFile(path.join(ROOT, 'models', 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
const total = files.reduce((sum, f) => sum + f.bytes, 0)
log(`done — ${files.length} files, ${mb(total)} total; wrote models/manifest.json`)
if (!withDetector) log('text-select (Grounding DINO + OWLv2) not vendored — re-run with --detector to make it offline too')

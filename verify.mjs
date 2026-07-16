#!/usr/bin/env node
/**
 * Headless-browser verification of SEGLAB end-to-end.
 *
 * Serves this directory, drives the REAL app (index.html + workers + models)
 * with Playwright Chromium through the window.__seglab test hooks, and
 * asserts on the demo scene's known answers. Runs under node ≥ 20 or bun.
 *
 * Phase 0 (static, no browser) — memory-contract source assertions:
 *   no automatic flagship starter, no toDataURL, no Python/OpenCV runtimes.
 *
 * Phase A (default URL — SlimSAM draft lane):
 *   proxy is exactly ≤768px · disc click quality · hygiene · edge softness ·
 *   embedding cache hit · minute-dot selection · lasso clamp ·
 *   ZERO SAM3 network requests end-to-end.
 *
 * Phase P (URL-flag bypass attempt — ?flagship=1&profile=ultra&proxy=max):
 *   the lite policy must clamp every value; selection still works on
 *   SlimSAM; still zero SAM3 network requests.
 *
 * Playwright is resolved from the local install if present, else from the
 * Pixxel repo's node_modules (already downloaded there). First run also
 * downloads model files into a persistent Chromium profile (.cache/profile),
 * so later runs are fast.
 *
 * Usage: node verify.mjs   (or: bun verify.mjs)
 */

import { createServer } from 'node:http'
import { readFile, readdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.dirname(fileURLToPath(import.meta.url))
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)

const log = (msg) => console.log(`[verify] ${msg}`)
const results = []
const check = (label, ok, detail) => {
  results.push({ label, ok })
  console.log(`[verify] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
}

/* ─── Phase 0: static memory-contract assertions ────────────────────────── */

const sources = {}
for (const f of await readdir(path.join(ROOT, 'js'))) {
  if (f.endsWith('.js')) sources[`js/${f}`] = await readFile(path.join(ROOT, 'js', f), 'utf8')
}
sources['index.html'] = await readFile(path.join(ROOT, 'index.html'), 'utf8')
const allSources = Object.entries(sources)

check(
  'static: no automatic flagship starter remains',
  allSources.every(([, s]) => !s.includes('maybeStartFlagship')) && !sources['js/app.js'].includes('FLAGSHIP_ENABLED'),
  'maybeStartFlagship/FLAGSHIP_ENABLED absent',
)
{
  const warmBody = /export const warm = async \(\) => \{([\s\S]*?)\n\}/.exec(sources['js/sam-engine.js'])?.[1] || 'MISSING'
  check(
    'static: warm() never touches the flagship lane',
    warmBody !== 'MISSING' && !/flagship/i.test(warmBody),
    warmBody === 'MISSING' ? 'warm() not found' : 'warm body is draft-only',
  )
}
check(
  'static: no toDataURL anywhere',
  allSources.every(([, s]) => !s.includes('toDataURL')),
  'exports use toBlob only',
)
check(
  'static: no Python runtime / no full OpenCV.js',
  allSources.every(([, s]) => !/pyodide|pyscript|micropython|brython|opencv/i.test(s)),
  'pyodide/pyscript/opencv absent from sources',
)
{
  // Full-frame pixel readbacks are allowed only where they are proxy-bounded
  // by construction: the engine's encoder input and the cv worker's guide.
  const readbackFiles = allSources
    .filter(([f, s]) => f.endsWith('.js') && /getImageData\(/.test(s))
    .map(([f]) => f)
  const allowed = new Set(['js/sam-engine.js', 'js/cv-refine-worker.js'])
  check(
    'static: getImageData only in proxy-bounded worker code',
    readbackFiles.every((f) => allowed.has(f)),
    `getImageData in: ${readbackFiles.join(', ') || 'none'}`,
  )
  check(
    'static: no unbounded working-copy path',
    allSources.every(([, s]) => !s.includes('WORKING_MAX_SIDE')),
    'WORKING_MAX_SIDE absent',
  )
}
{
  const sw = await readFile(path.join(ROOT, 'sw.js'), 'utf8')
  check(
    'static: sw.js is cache-on-fetch only (no precache, GET only)',
    !/caches\.addAll|precacheAndRoute/.test(sw) && sw.includes("request.method !== 'GET'") && sw.includes('seglab-static-v'),
    'no addAll/precache calls; GET-gated; versioned cache key',
  )
}

/* ─── Phase N: image-io unit tests (pure module, node-side) ─────────────── */

const io = await import(new URL('./js/image-io.js', import.meta.url).href)
{
  const r1 = io.getBoundedProxySize(6000, 4000, 768)
  const r2 = io.getBoundedProxySize(4000, 6000, 768)
  const r3 = io.getBoundedProxySize(768, 512, 768)
  const r4 = io.getBoundedProxySize(500, 300, 768)
  check('sizing: 6000×4000 → 768×512', r1.width === 768 && r1.height === 512 && r1.proxyActive, `${r1.width}×${r1.height}`)
  check('sizing: 4000×6000 portrait → 512×768', r2.width === 512 && r2.height === 768, `${r2.width}×${r2.height}`)
  check('sizing: 768×512 stays native, 500×300 never upscales',
    r3.scale === 1 && !r3.proxyActive && r4.width === 500 && r4.height === 300 && r4.scale === 1,
    `native ${r3.width}×${r3.height}, small ${r4.width}×${r4.height}`)
  const throws = (w, h) => { try { io.getBoundedProxySize(w, h, 768); return false } catch { return true } }
  check('sizing: invalid dimensions reject cleanly',
    throws(0, 100) && throws(NaN, 50) && throws(-3, 5) && throws(100, Infinity),
    '0/NaN/negative/Infinity all throw')
  const o1 = io.orientedSize(6000, 4000, 6)
  const o2 = io.orientedSize(6000, 4000, 8)
  const o3 = io.orientedSize(6000, 4000, 3)
  check('sizing: EXIF orientations 5–8 transpose the frame (1–4 do not)',
    o1.width === 4000 && o1.height === 6000 && o2.width === 4000 && o3.width === 6000,
    `or6→${o1.width}×${o1.height}, or8→${o2.width}×${o2.height}, or3→${o3.width}×${o3.height}`)
}
{
  // Minimal JPEG: APP1/EXIF (orientation 6) + SOF0 (4000 high, 6000 wide).
  const jpeg = new Uint8Array([
    0xff, 0xd8,
    0xff, 0xe1, 0x00, 0x22,
    0x45, 0x78, 0x69, 0x66, 0x00, 0x00,
    0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x01, 0x00,
    0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    0xff, 0xc0, 0x00, 0x11, 0x08, 0x0f, 0xa0, 0x17, 0x70, 0x03,
    0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
  ])
  const h = io.parseImageHeader(jpeg)
  check('header: JPEG SOF dims + EXIF orientation parsed from bounded bytes',
    h?.format === 'jpeg' && h.width === 6000 && h.height === 4000 && h.orientation === 6,
    `${h?.format} ${h?.width}×${h?.height} or=${h?.orientation}`)

  const png = new Uint8Array([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x17, 0x70, 0x00, 0x00, 0x0f, 0xa0,
    0x08, 0x06, 0x00, 0x00, 0x00,
  ])
  const hp = io.parseImageHeader(png)
  check('header: PNG IHDR dims parsed', hp?.format === 'png' && hp.width === 6000 && hp.height === 4000, `${hp?.width}×${hp?.height}`)

  // Minimal TIFF/RAW container: dims + embedded JPEG preview span.
  const tiff = new Uint8Array(8 + 2 + 4 * 12 + 4)
  {
    const v = new DataView(tiff.buffer)
    tiff[0] = 0x49; tiff[1] = 0x49
    v.setUint16(2, 42, true)
    v.setUint32(4, 8, true)
    v.setUint16(8, 4, true)
    const entry = (i, tag, type, val) => {
      const e = 10 + i * 12
      v.setUint16(e, tag, true)
      v.setUint16(e + 2, type, true)
      v.setUint32(e + 4, 1, true)
      if (type === 3) v.setUint16(e + 8, val, true)
      else v.setUint32(e + 8, val, true)
    }
    entry(0, 0x0100, 4, 8192)
    entry(1, 0x0101, 4, 5464)
    entry(2, 0x0201, 4, 4096)
    entry(3, 0x0202, 4, 500000)
  }
  const ht = io.parseImageHeader(tiff)
  check('header: TIFF/RAW dims + embedded-preview span parsed',
    ht?.format === 'tiff' && ht.width === 8192 && ht.height === 5464
      && ht.rawPreview?.offset === 4096 && ht.rawPreview?.length === 500000,
    `${ht?.width}×${ht?.height}, preview @${ht?.rawPreview?.offset}+${ht?.rawPreview?.length}`)
}

/* ─── Playwright (local, else Pixxel's install) ─────────────────────────── */
// Prefer the Pixxel repo's install: its browser build is known-downloaded
// (a bare 'playwright' can resolve to a different version whose browser
// binary was never fetched).
let chromium
try {
  ({ chromium } = await import('/Users/andhetharuntej/Pixxel/node_modules/playwright/index.mjs'))
} catch {
  try {
    ({ chromium } = await import('playwright'))
  } catch {
    log('skip — playwright not found in ~/Pixxel or locally')
    process.exit(0)
  }
}

/* ─── Static server ─────────────────────────────────────────────────────── */
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
}
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost')
    const rel = url.pathname === '/' ? '/index.html' : url.pathname
    const file = path.join(ROOT, path.normalize(rel))
    if (!file.startsWith(ROOT) || !existsSync(file)) {
      res.writeHead(404).end('not found')
      return
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
    res.end(await readFile(file))
  } catch (e) {
    res.writeHead(500).end(String(e?.message || e))
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const port = server.address().port
log(`serving SEGLAB on http://127.0.0.1:${port}`)

/* ─── Drive the app ─────────────────────────────────────────────────────── */
// Demo-scene ground truth: buildDemoScene (js/app.js) draws at 900×620,
// which the lite policy proxies down to 768×529. All assertions live in
// proxy-canvas coordinates, so scale the source geometry once here.
const SRC_W = 900
const SRC_H = 620
const PROXY_MAX = 768
const S = PROXY_MAX / SRC_W                       // 0.8533…
const VIEW_W = Math.round(SRC_W * S)              // 768
const VIEW_H = Math.round(SRC_H * S)              // 529
const DISC = { x: 230 * S, y: 340 * S, r: 105 * S }
const SQUARE = { x: 615 * S, y: 245 * S, half: 95 * S }
const DOT = { x: 700 * S, y: 480 * S, r: 9 * S }
const FRAME = VIEW_W * VIEW_H

// deviceMemory: number to simulate that report, null to simulate "unknown"
// (Safari-like), undefined to leave the real value. Deterministic policy
// tests must not depend on the dev machine's RAM.
const newAppPage = async (context, query, { deviceMemory, demo = true } = {}) => {
  const page = await context.newPage()
  page.sam3Requests = []
  page.on('request', (req) => {
    if (/sam3/i.test(req.url())) page.sam3Requests.push(req.url())
  })
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[seglab]') || msg.type() === 'error') log(`browser: ${text.slice(0, 180)}`)
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
  page.setDefaultTimeout(TIMEOUT_MS)
  if (deviceMemory !== undefined) {
    await page.addInitScript((mem) => {
      Object.defineProperty(navigator, 'deviceMemory', {
        get: () => (mem === null ? undefined : mem),
        configurable: true,
      })
    }, deviceMemory)
  }
  await page.goto(`http://127.0.0.1:${port}/${query}`)
  await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  if (demo) await page.evaluate(() => window.__seglab.loadDemo())
  return page
}

/** The disc-quality bar, shared by phases. */
const checkDisc = (tag, s) => {
  const discArea = (Math.PI * DISC.r * DISC.r) / FRAME
  const b = s.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    `${tag}: click selects the disc`,
    b[0] <= DISC.x && DISC.x <= b[2] && b[1] <= DISC.y && DISC.y <= b[3],
    `bbox [${b.map((v) => Math.round(v))}] vs centre (${Math.round(DISC.x)},${Math.round(DISC.y)})`,
  )
  check(
    `${tag}: disc mask is object-sized, not a flood`,
    s.maskSummary && s.maskSummary.coverage > discArea * 0.4 && s.maskSummary.coverage < discArea * 3,
    `coverage ${((s.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discArea * 100).toFixed(1)}%`,
  )
}

let context = null
let failed = false
try {
  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
  })

  /* ─── Phase A: default URL, SlimSAM draft lane ──────────────────────── */
  const page = await newAppPage(context, '')
  log('phase A (draft lane) — first click downloads the model on a cold profile…')

  // 0. The interaction proxy is bounded at 768 px.
  const vs = await page.evaluate(() => window.__seglab.viewSize())
  check(
    'proxy: demo scene lands on the ≤768px canvas',
    vs.width === VIEW_W && vs.height === VIEW_H,
    `view ${vs.width}×${vs.height} (expected ${VIEW_W}×${VIEW_H})`,
  )

  // 1+2+3. Disc click: selection quality + hygiene + edge refinement.
  const s1 = await page.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: Math.round(DISC.x), y: Math.round(DISC.y) },
  )
  checkDisc('draft', s1)
  const stats1 = await page.evaluate(() => window.__seglab.maskStats())
  check(
    'hygiene: single component, no crumbs',
    stats1 && stats1.components === 1,
    `components=${stats1?.components}`,
  )
  check(
    'edge refinement: soft boundary band present',
    stats1 && stats1.softPixels > 100,
    `softPixels=${stats1?.softPixels}`,
  )

  // 4. Negative click far away — must be a cached decoder-only pass.
  const s2 = await page.evaluate(() => window.__seglab.clickAt(51, 51, true))
  check(
    'repeat click skips the encoder (cache hit)',
    s2.lastRun && s2.lastRun.encoded === false,
    `encoded=${s2.lastRun?.encoded}, decode ${s2.lastRun?.decodeMs}ms, post ${s2.lastRun?.postMs}ms`,
  )

  // 5. Minute object: the dot must survive the hygiene pass (it is seeded
  // by the click, so island filtering must NOT eat it).
  await page.evaluate(() => window.__seglab.reset())
  const s3 = await page.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: Math.round(DOT.x), y: Math.round(DOT.y) },
  )
  const dotArea = (Math.PI * DOT.r * DOT.r) / FRAME
  const b3 = s3.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    `minute object (r=${DOT.r.toFixed(1)}px) is selectable`,
    s3.maskSummary && b3[0] <= DOT.x && DOT.x <= b3[2] && s3.maskSummary.coverage < dotArea * 40,
    `coverage ${((s3.maskSummary?.coverage || 0) * 100).toFixed(2)}% (dot ${(dotArea * 100).toFixed(3)}%), bbox [${b3.map((v) => Math.round(v))}]`,
  )

  // 6. Lasso around the square: mask must stay inside lasso ∪ margin.
  await page.evaluate(() => window.__seglab.reset())
  const lassoR = SQUARE.half * 1.5
  const s4 = await page.evaluate(
    ({ x, y, r }) => window.__seglab.lassoCircle(x, y, r),
    { x: SQUARE.x, y: SQUARE.y, r: lassoR },
  )
  const b4 = s4.maskSummary?.bbox || [0, 0, -1, -1]
  const clampR = lassoR + 30 // margin ≈ 4% of lasso diag + slack
  const inClamp = b4[0] >= SQUARE.x - clampR && b4[2] <= SQUARE.x + clampR
    && b4[1] >= SQUARE.y - clampR && b4[3] <= SQUARE.y + clampR
  check(
    'lasso snaps to the square and stays clamped',
    s4.maskSummary && inClamp && b4[0] <= SQUARE.x && SQUARE.x <= b4[2],
    `bbox [${b4.map((v) => Math.round(v))}] within ±${Math.round(clampR)} of (${Math.round(SQUARE.x)},${Math.round(SQUARE.y)}); coverage ${((s4.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )

  // 7. The memory contract's headline: NOTHING pulled the SAM3 lane.
  check(
    'no SAM3 download on the default path',
    page.sam3Requests.length === 0,
    page.sam3Requests.length ? `saw ${page.sam3Requests[0]}` : 'zero sam3-matching requests',
  )

  const stA = await page.evaluate(() => window.__seglab.state())
  check(
    'draft lane is the serving lane',
    stA.lane === 'slimsam',
    `lane=${stA.lane}, engine=${stA.mode}, device=${stA.device}`,
  )

  // 8. Synthetic DSLR import: bounded decode, Blob custody, then selection.
  log('phase A: importing a synthetic 6000×4000 JPEG (24 MP)…')
  const meta = await page.evaluate(() => window.__seglab.importSynthetic(6000, 4000))
  check(
    'DSLR: 24 MP import lands on a 768×512 proxy',
    meta.proxy?.width === 768 && meta.proxy?.height === 512
      && meta.original?.width === 6000 && meta.original?.height === 4000,
    `original ${meta.original?.width}×${meta.original?.height} → proxy ${meta.proxy?.width}×${meta.proxy?.height}`,
  )
  check(
    'DSLR: original retained as compressed Blob only',
    meta.hasBlob === true && meta.blobBytes > 50_000,
    `blob ${((meta.blobBytes || 0) / 1024 / 1024).toFixed(2)} MB compressed`,
  )
  const sD = await page.evaluate(() => window.__seglab.clickAt(230, 256))
  const discD = (Math.PI * (512 * 0.15) ** 2) / (768 * 512)
  check(
    'DSLR: selection works on the imported photo',
    sD.maskSummary && sD.maskSummary.coverage > discD * 0.4 && sD.maskSummary.coverage < discD * 3,
    `coverage ${((sD.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discD * 100).toFixed(1)}%`,
  )

  // 9. Overlapping imports: the second must win, the first must drop.
  const race = await page.evaluate(() => window.__seglab.importRace())
  check(
    'import race: second import wins, first dropped',
    race.proxy?.width === 512 && race.proxy?.height === 768
      && race.view?.width === 512 && race.view?.height === 768,
    `final proxy ${race.proxy?.width}×${race.proxy?.height}, view ${race.view?.width}×${race.view?.height}`,
  )

  // 10. Embedding residency: one slot, encode only on first selection.
  const engBefore = await page.evaluate(() => window.__seglab.engineState())
  await page.evaluate(() => window.__seglab.importSynthetic(1600, 1200))
  const engImported = await page.evaluate(() => window.__seglab.engineState())
  check(
    'embedding: import alone never encodes (no speculative encode)',
    engImported.encodeCount === engBefore.encodeCount && engImported.residentEmbeddings === 0,
    `encodeCount ${engBefore.encodeCount}→${engImported.encodeCount}, resident=${engImported.residentEmbeddings}`,
  )
  await page.evaluate(() => window.__seglab.clickAt(230, 288)) // disc on the 768×576 proxy
  const engClicked = await page.evaluate(() => window.__seglab.engineState())
  check(
    'embedding: first selection encodes exactly once into one slot',
    engClicked.encodeCount === engBefore.encodeCount + 1 && engClicked.residentEmbeddings === 1,
    `encodeCount=${engClicked.encodeCount}, resident=${engClicked.residentEmbeddings}`,
  )
  await page.evaluate(() => window.__seglab.clickAt(60, 60, true))
  const engRepeat = await page.evaluate(() => window.__seglab.engineState())
  check(
    'embedding: repeat prompts reuse the slot (no re-encode, never >1 resident)',
    engRepeat.encodeCount === engBefore.encodeCount + 1 && engRepeat.residentEmbeddings === 1,
    `encodeCount=${engRepeat.encodeCount}, resident=${engRepeat.residentEmbeddings}`,
  )

  // 11. Stale results must never commit to a replaced document.
  const stale = await page.evaluate(() => window.__seglab.staleProbe())
  check(
    'stale worker responses never commit mask state',
    stale.maskSummary === null && stale.clicks === 0,
    `maskSummary=${JSON.stringify(stale.maskSummary)}, clicks=${stale.clicks}`,
  )

  // 12. C++/Wasm refinement suite (drives the cv worker directly).
  const cv = await page.evaluate(() => window.__seglab.cvTest())
  check(
    'wasm: rejects dimensions above 768',
    cv.dimReject === true,
    '800×800 refine request rejected',
  )
  check(
    'wasm: hole fill, small-object removal, seeded keep',
    cv.holeFilled === true && cv.crumbRemoved === true && cv.objectKept === true,
    `holeFilled=${cv.holeFilled} crumbRemoved=${cv.crumbRemoved} objectKept=${cv.objectKept}`,
  )
  check(
    'wasm: refiner really is the wasm module (not JS fallback)',
    cv.refiner === 'wasm',
    `refiner=${cv.refiner}`,
  )
  check(
    'wasm: min-area floor removes unseeded components',
    cv.seededKept === true && cv.unseededRemoved === true,
    `seededKept=${cv.seededKept} unseededRemoved=${cv.unseededRemoved}`,
  )
  check(
    'wasm: morphology respects image bounds',
    cv.morphBounds === true,
    'full-frame close(3)+open(2) left the mask intact',
  )
  check(
    'wasm: invalid buffers rejected without leaking worker state',
    cv.invalidRejected === true && cv.recoveredAfterInvalid === true,
    `invalidRejected=${cv.invalidRejected}, next call ok=${cv.recoveredAfterInvalid}`,
  )
  check(
    'wasm: mask buffers transfer, not copy',
    cv.transferred === true,
    'input buffer detached after refine',
  )
  const stCv = await page.evaluate(() => window.__seglab.state())
  check(
    'wasm: interactive selections run through the wasm refiner',
    stCv.lastRefiner === 'wasm',
    `lastRefiner=${stCv.lastRefiner}`,
  )

  // 13. Export: explicit, bounded to 4096px/8MP, reduced flag honest.
  await page.evaluate(() => window.__seglab.clickAt(230, 288)) // re-select on the 1600×1200 doc
  const expSmall = await page.evaluate(() => window.__seglab.exportProbe())
  check(
    'export: sources under the caps export at native size',
    expSmall?.width === 1600 && expSmall?.height === 1200 && expSmall?.reduced === false && expSmall?.bytes > 1000,
    `out ${expSmall?.width}×${expSmall?.height}, reduced=${expSmall?.reduced}, ${((expSmall?.bytes || 0) / 1024) | 0} KB`,
  )
  await page.evaluate(() => window.__seglab.importSynthetic(6000, 4000))
  await page.evaluate(() => window.__seglab.clickAt(230, 256))
  const expBig = await page.evaluate(() => window.__seglab.exportProbe())
  // 24 MP → MP cap binds: floor(6000·√(8/24)) = 3464 long edge (7.998 MP).
  check(
    'export: 24 MP source is capped to ≤4096px and ≤8MP with reduced flag',
    expBig?.width === 3464 && expBig?.height === 2309 && expBig?.reduced === true
      && expBig.width <= 4096 && (expBig.width * expBig.height) <= 8e6,
    `out ${expBig?.width}×${expBig?.height} (${((expBig?.width * expBig?.height) / 1e6).toFixed(2)} MP), reduced=${expBig?.reduced}`,
  )

  // 14. Stability: repeated prompts on one image — one embedding, no drift.
  const engS0 = await page.evaluate(() => window.__seglab.engineState())
  for (let i = 0; i < 5; i += 1) {
    await page.evaluate(({ i: k }) => window.__seglab.clickAt(200 + k * 9, 250, k % 2 === 1), { i })
  }
  const engS1 = await page.evaluate(() => window.__seglab.engineState())
  check(
    'stability: repeated clicks never re-encode or grow residency',
    engS1.encodeCount === engS0.encodeCount && engS1.residentEmbeddings === 1,
    `encodeCount ${engS0.encodeCount}→${engS1.encodeCount}, resident=${engS1.residentEmbeddings}`,
  )

  // 15. Release-memory debug action leaves the app alive.
  const released = await page.evaluate(() => window.__seglab.releaseMemory())
  const engR = await page.evaluate(() => window.__seglab.engineState())
  const sR = await page.evaluate(() => window.__seglab.clickAt(230, 256))
  check(
    'release-memory: clears residency, next selection still works',
    released === true && engR.residentEmbeddings === 0 && !!sR.maskSummary,
    `resident after release=${engR.residentEmbeddings}, next click coverage ${((sR.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )

  // 16. Heavy-job queue: strict serialization + priority + failure recovery.
  const qlog = await page.evaluate(() => window.__seglab.queueLog())
  let concurrent = 0
  let maxConcurrent = 0
  for (const e of qlog.log) {
    if (e.ev === 'start') { concurrent += 1; maxConcurrent = Math.max(maxConcurrent, concurrent) }
    if (e.ev === 'end') concurrent -= 1
  }
  check(
    'queue: at most one heavy job active at any time',
    maxConcurrent === 1,
    `maxConcurrent=${maxConcurrent} across ${qlog.log.length} events (peak depth ${qlog.peakDepth})`,
  )
  check(
    'queue: decode, warm, and prompt jobs all ran through the lane',
    qlog.log.some((e) => e.label === 'model-warm' && e.ev === 'end')
      && qlog.log.some((e) => e.label === 'prompt-decode' && e.ev === 'end')
      && qlog.log.some((e) => e.label === 'proxy-decode' && e.ev === 'end'),
    `labels: ${[...new Set(qlog.log.map((e) => e.label))].join(', ')}`,
  )
  const failMsg = await page.evaluate(() => window.__seglab.enqueueFailing())
  const order = await page.evaluate(() => window.__seglab.queueProbe())
  check(
    'queue: rejected job does not deadlock the lane',
    failMsg === 'synthetic failure' && order.length === 3,
    `reject="${failMsg}", probe completed ${order.length}/3 jobs after it`,
  )
  check(
    'queue: interaction preempts queued speculative work',
    order[0] === 'probe-blocker' && order[1] === 'probe-interaction' && order[2] === 'probe-spec',
    `order=${order.join(' → ')}`,
  )

  log(`phase A done: engine=${stA.mode} device=${stA.device} lane=${stA.lane}`)
  await page.close()

  /* ─── Phase P: URL flags must NOT bypass the lite policy on the target
         device — simulate an 8 GB report so the test is machine-independent. */
  const pageP = await newAppPage(context, '?flagship=1&profile=ultra&proxy=max&working=1', { deviceMemory: 8 })
  const pol = await pageP.evaluate(() => window.__seglab.policy())
  check(
    'policy: lite profile forced on an 8GB device',
    pol.profile === 'lite' && pol.liteForced === true,
    `profile=${pol.profile}, deviceMemoryGB=${pol.deviceMemoryGB}`,
  )
  check(
    'policy: ?proxy=max cannot raise the 768px cap',
    pol.proxyMax === 768,
    `proxyMax=${pol.proxyMax}`,
  )
  check(
    'policy: ?flagship=1 cannot enable the flagship lane',
    pol.flagship === false,
    `flagship=${pol.flagship}`,
  )
  check(
    'policy: lite residency + feature caps hold',
    pol.draftCacheMax === 1 && pol.flagshipCacheMax === 0 && pol.maxResidentHeavy === 1
      && pol.autoEscalate === false && pol.hdExportDecode === false && pol.detectorWebGPU === false,
    `draftCacheMax=${pol.draftCacheMax} flagshipCacheMax=${pol.flagshipCacheMax} autoEscalate=${pol.autoEscalate} hdExportDecode=${pol.hdExportDecode}`,
  )
  check(
    'policy: export caps at 4096px / 8MP',
    pol.exportMaxSide === 4096 && pol.exportMaxMP === 8,
    `exportMaxSide=${pol.exportMaxSide} exportMaxMP=${pol.exportMaxMP}`,
  )
  const sP = await pageP.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: Math.round(DISC.x), y: Math.round(DISC.y) },
  )
  check(
    'policy: selection still works under bypass flags (on SlimSAM)',
    !!sP.maskSummary && sP.lastRun?.lane === 'slimsam',
    `lane=${sP.lastRun?.lane}, coverage ${((sP.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )
  check(
    'policy: zero SAM3 requests even with ?flagship=1',
    pageP.sam3Requests.length === 0,
    pageP.sam3Requests.length ? `saw ${pageP.sam3Requests[0]}` : 'zero sam3-matching requests',
  )
  await pageP.close()

  /* ─── Phase U: unknown memory (Safari-like) also forces lite ────────── */
  const pageU = await newAppPage(context, '?flagship=1&proxy=max', { deviceMemory: null, demo: false })
  const polU = await pageU.evaluate(() => window.__seglab.policy())
  check(
    'policy: unknown device memory forces lite + clamps flags',
    polU.profile === 'lite' && polU.liteForced === true && polU.proxyMax === 768 && polU.flagship === false,
    `profile=${polU.profile}, deviceMemoryGB=${polU.deviceMemoryGB}, proxyMax=${polU.proxyMax}, flagship=${polU.flagship}`,
  )
  await pageU.close()

  /* ─── Phase T: ?debug=1 memory telemetry schema. */
  {
    const memEvents = []
    const pageT = await context.newPage()
    pageT.on('console', (msg) => {
      const t = msg.text()
      if (t.startsWith('[seglab][memory]')) {
        try { memEvents.push(JSON.parse(t.slice('[seglab][memory] '.length))) } catch { /* partial */ }
      }
    })
    pageT.setDefaultTimeout(TIMEOUT_MS)
    await pageT.goto(`http://127.0.0.1:${port}/?debug=1`)
    await pageT.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
    await pageT.evaluate(() => window.__seglab.importSynthetic(3000, 2000))
    await pageT.evaluate(() => window.__seglab.clickAt(230, 256))
    const events = await pageT.evaluate(() => window.__seglabMemLog || [])
    const decoded = events.find((e) => e.event === 'proxy-decoded')
    const selected = events.find((e) => e.event === 'selection-done')
    check(
      'telemetry: structured [seglab][memory] events with required fields',
      decoded?.original === '3000x2000' && decoded?.proxy === '768x512'
        && typeof decoded.pressureLevel === 'number' && 'queueActive' in decoded
        && selected && 'refiner' in selected && 'encodeMs' in selected
        && memEvents.length > 0, // the console-line form also emits
      `events=${events.map((e) => e.event).join(',')} consoleLines=${memEvents.length}`,
    )
    // Pressure L2 hook: wasm refinement drops to the JS/engine path.
    await pageT.evaluate(() => window.__seglab.setPressure(2))
    const polT = await pageT.evaluate(() => window.__seglab.policy())
    await pageT.evaluate(() => window.__seglab.clickAt(240, 260))
    const stT = await pageT.evaluate(() => window.__seglab.state())
    check(
      'pressure L2: wasm refine disabled, selection stays functional',
      polT.wasmRefine === false && polT.pressureLevel === 2
        && !!stT.maskSummary && stT.lastRefiner === 'engine-js',
      `wasmRefine=${polT.wasmRefine}, refiner=${stT.lastRefiner}`,
    )
    await pageT.close()
  }

  /* ─── Phase W: wasm module unreachable → JS fallback keeps selections.
         Fresh SW-blocked context — the service worker must not serve the
         wasm from cache and defeat the abort. */
  {
    const browserW = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu'] })
    const ctxW = await browserW.newContext({ serviceWorkers: 'block' })
    const pageW = await ctxW.newPage()
    pageW.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
    pageW.setDefaultTimeout(TIMEOUT_MS)
    await pageW.route('**/public/wasm/**', (r) => r.abort())
    await pageW.goto(`http://127.0.0.1:${port}/`)
    await pageW.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
    await pageW.evaluate(() => window.__seglab.loadDemo())
    const sW = await pageW.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: Math.round(DISC.x), y: Math.round(DISC.y) },
    )
    const statsW = await pageW.evaluate(() => window.__seglab.maskStats())
    check(
      'wasm failure: selection survives on the JS fallback',
      !!sW.maskSummary && sW.lastRefiner === 'js' && statsW?.components === 1,
      `refiner=${sW.lastRefiner}, components=${statsW?.components}, coverage ${((sW.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
    )
    await browserW.close().catch(() => {})
  }

  /* ─── Phase O: vendored assets serve fully offline. Fresh cache-less
         context so nothing can come from the persistent profile's caches. */
  if (existsSync(path.join(ROOT, 'models', 'manifest.json'))) {
    const browserO = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu'] })
    try {
      const ctxO = await browserO.newContext()
      const pageO = await ctxO.newPage()
      const blocked = []
      await pageO.route(/https:\/\/(huggingface\.co|cdn\.jsdelivr\.net)\//, (r) => {
        blocked.push(r.request().url())
        r.abort()
      })
      const localModelHits = []
      pageO.on('request', (req) => { if (req.url().includes('/models/')) localModelHits.push(req.url()) })
      pageO.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
      pageO.setDefaultTimeout(TIMEOUT_MS)
      await pageO.goto(`http://127.0.0.1:${port}/`)
      await pageO.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
      await pageO.evaluate(() => window.__seglab.loadDemo())
      const sO = await pageO.evaluate(
        ({ x, y }) => window.__seglab.clickAt(x, y),
        { x: Math.round(DISC.x), y: Math.round(DISC.y) },
      )
      check(
        'offline: selection works with both CDNs blocked (fresh cache)',
        !!sO.maskSummary && sO.lastRun?.lane === 'slimsam',
        `coverage ${((sO.maskSummary?.coverage || 0) * 100).toFixed(1)}%, blocked ${blocked.length} external requests`,
      )
      check(
        'offline: weights served from vendored models/',
        localModelHits.length > 0,
        `${localModelHits.length} local model fetches`,
      )
    } finally {
      await browserO.close().catch(() => {})
    }
  } else {
    log('skip offline phase — run `node scripts/download-models.mjs` to vendor assets')
  }
  /* ─── Phase F: SAM3 explicit opt-in (WARN-only — headless WebGPU ≠ real
         Chrome; the contract that matters is "never automatic", tested hard
         above). First run pulls ~300 MB into the persistent profile. */
  try {
    const pageF = await newAppPage(context, '')
    log('phase F (SAM3 opt-in) — warn-only…')
    await pageF.evaluate(() => window.__seglab.enableFlagship())
    await pageF.waitForFunction(() => window.__seglab.state().lane === 'sam3', null, { timeout: TIMEOUT_MS })
    const sF = await pageF.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: Math.round(DISC.x), y: Math.round(DISC.y) },
    )
    checkDisc('sam3 (opt-in)', sF)
    check('sam3 opt-in: lane confirmed on the result', sF.lastRun?.lane === 'sam3', `lane=${sF.lastRun?.lane}`)
    await pageF.close()
  } catch (err) {
    log(`⚠ phase F: SAM3 opt-in not confirmed in headless Chromium (${String(err?.message || err).slice(0, 120)})`)
    log('⚠ warn only — the hard guarantee (never automatic) is gated above; confirm opt-in in real Chrome')
  }
} catch (err) {
  console.error(`[verify] ✗ ${err?.message || err}`)
  failed = true
} finally {
  await context?.close().catch(() => {})
  server.close()
}

if (failed || results.some((r) => !r.ok)) {
  console.error('\n[verify] ✗ SEGLAB self-test FAILED')
  process.exit(1)
}
console.log('\n[verify] ✓ SEGLAB verified end-to-end in a real browser')

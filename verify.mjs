#!/usr/bin/env bun
/**
 * Headless-browser verification of SEGLAB end-to-end.
 *
 * Serves this directory, drives the REAL app (index.html + worker + models
 * via CDN) with Playwright Chromium through the window.__seglab test hooks,
 * and asserts on the demo scene's known answers.
 *
 * Phase A (?flagship=0 — deterministic, draft lane only):
 *   1. click the red disc      → mask is the disc (bbox + coverage)
 *   2. hygiene                 → exactly ONE component (no crumbs/islands)
 *   3. edge refinement         → soft (0<v<255) pixels exist in the band
 *   4. second click (negative) → decoder-only pass (embedding cache hit)
 *   5. click the MINUTE dot    → a genuinely small mask is returned
 *   6. lasso around the square → mask confined to lasso ∪ margin
 *
 * Phases A2–A6 gate the milestones: revision/cancel, HD export, crop
 * escalation, text-select plumbing, capability + weak-device.
 *
 * Phase A7/A8 (M5): the network is CUT (context.setOffline) after a warm
 * import and click/text/export must still pass — zero-cloud as a tested
 * property; then a fresh session must serve its import-time encode from
 * OPFS (encoded=false) and decode a sane mask from the restored tensors.
 *
 * Phase B (`VERIFY_FLAGSHIP=1` — optional flagship upgrade path):
 *   7. background SAM3 download → lane flips to 'sam3' (first run pulls
 *      ~300 MB into the persistent profile; cached after)
 *   8. click the disc on SAM3   → same quality bars pass on the flagship
 *   This phase is opt-in because that download/compile is inappropriate as a
 *   default validation workload on an 8 GB machine. Failure is WARN-only.
 *
 * Playwright is resolved from the local install if present, else from the
 * Pixxel repo's node_modules (already downloaded there). First run also
 * downloads ~14 MB of model files into a persistent Chromium profile
 * (.cache/profile), so later runs are fast.
 *
 * Usage: bun verify.mjs
 */

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import {
  DETECTOR_INPUT, letterboxPlan, normalizePhrase, nms, rankDetections, scaleBox, unletterboxBox,
} from './js/text-core.js'
import { classifyCapability } from './js/capability.js'
import { refineMaskEdges } from './js/edge-refine.js'
import { applyMemoryPressure, resolveBudget } from './js/policy.js'

const ROOT = path.resolve(import.meta.dir)
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)
const RAW_FIXTURE = process.env.RAW_FIXTURE || ''

const log = (msg) => console.log(`[verify] ${msg}`)
const results = []
const check = (label, ok, detail) => {
  results.push({ label, ok })
  console.log(`[verify] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
}

/* ─── Playwright (local dev dependency) ─────────────────────────────────── */
// A missing playwright is a HARD failure: a gate that silently exits green
// is worse than no gate. CI_SKIP_BROWSER=1 is the only sanctioned skip.
let chromium
try {
  ({ chromium } = await import('playwright'))
} catch (err) {
  if (process.env.CI_SKIP_BROWSER === '1') {
    log('skip — CI_SKIP_BROWSER=1')
    process.exit(0)
  }
  console.error('[verify] ✗ playwright not installed — run: bun add -d playwright && bunx playwright install chromium')
  console.error(`[verify]   (${err?.message})`)
  process.exit(1)
}

/* ─── Static server ─────────────────────────────────────────────────────── */
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
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
// Demo-scene ground truth (must match buildDemoScene in js/app.js).
const DISC = { x: 230, y: 340, r: 105 }
const SQUARE = { x: 615, y: 245, half: 95 }
const DOT = { x: 700, y: 480, r: 9 }
const FRAME = 900 * 620

const newAppPage = async (context, query, longSide) => {
  const page = await context.newPage()
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[seglab]') || msg.type() === 'error') log(`browser: ${text.slice(0, 180)}`)
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
  page.setDefaultTimeout(TIMEOUT_MS)
  await page.goto(`http://127.0.0.1:${port}/${query}`)
  await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  await page.evaluate((ls) => window.__seglab.loadDemo(ls), longSide)
  return page
}

/** The disc-quality bar, shared by both phases. */
const checkDisc = (tag, s) => {
  const discArea = (Math.PI * DISC.r * DISC.r) / FRAME
  const b = s.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    `${tag}: click selects the disc`,
    b[0] <= DISC.x && DISC.x <= b[2] && b[1] <= DISC.y && DISC.y <= b[3],
    `bbox [${b}] vs centre (${DISC.x},${DISC.y})`,
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
  /* ─── Phase T: text-core pure logic (no browser) ────────────────────── */
  const np = normalizePhrase('all red cars')
  check(
    'text-core: phrase → template + multi intent',
    np && np.multi === true && np.labels[0] === 'a photo of a red car',
    `${JSON.stringify(np)}`,
  )
  const deduped = nms([
    { box: [0, 0, 100, 100], score: 0.9 },
    { box: [5, 5, 105, 105], score: 0.8 },  // ~IoU 0.8 → dropped
    { box: [400, 400, 500, 500], score: 0.7 }, // disjoint → kept
  ], 0.5)
  check('text-core: NMS drops overlaps, keeps distinct', deduped.length === 2, `kept ${deduped.length}`)
  const ranked = rankDetections(
    [{ box: [0, 0, 10, 10], score: 0.05 }, { box: [20, 20, 30, 30], score: 0.4 }],
    { threshold: 0.15, topK: 8 },
  )
  const sb = scaleBox([10, 20, 30, 40], 2, 3)
  check(
    'text-core: rank filters by threshold; scaleBox maps coords',
    ranked.length === 1 && ranked[0].score === 0.4 && sb[0] === 20 && sb[3] === 120,
    `ranked=${ranked.length} scaled=[${sb}]`,
  )
  const grass = normalizePhrase('grass')
  const many = normalizePhrase('red flowers')
  check(
    'text-core: depluralize keeps -ss words whole, "flowers" implies multi',
    grass.core === 'grass' && grass.multi === false && many.core === 'red flower' && many.multi === true,
    `${grass.core}/${grass.multi} ${many.core}/${many.multi}`,
  )
  // 3:2 landscape letterboxed into the 960² the detector actually sees.
  const plan = letterboxPlan(1200, 800, DETECTOR_INPUT)
  const full = unletterboxBox([0, 0, 1, 640 / 960], plan)
  check(
    'text-core: letterbox plan preserves aspect; boxes map back to source px',
    plan.dw === 960 && plan.dh === 640 && full[2] === 1200 && Math.round(full[3]) === 800,
    `plan=${plan.dw}x${plan.dh} full=[${full}]`,
  )
  check(
    'text-core: a box on the gray padding is rejected, not clamped onto the photo',
    unletterboxBox([0, 0.75, 1, 0.95], plan) === null,
    'padding box survived',
  )
  const gated = rankDetections(
    [{ box: [0, 0, 100, 100], score: 0.6 }, { box: [0, 0, 1200, 800], score: 0.15 }],
    { threshold: 0.12, relative: 0.35 },
  )
  check(
    'text-core: relative gate drops the low-score whole-frame guess',
    gated.length === 1 && gated[0].score === 0.6,
    `kept ${gated.length}`,
  )
  // The edge refiner intentionally owns only its boundary band. This small,
  // deterministic check catches both a broken morphology pass and a filter
  // that spills soft alpha across the entire interaction image.
  const ew = 37
  const eh = 29
  const edgeMask = new Uint8ClampedArray(ew * eh * 4)
  const edgeGuide = new Float32Array(ew * eh)
  for (let y = 0; y < eh; y += 1) {
    for (let x = 0; x < ew; x += 1) {
      const i = y * ew + x
      const j = i * 4
      const inside = x >= 9 && x <= 27 && y >= 7 && y <= 22
      edgeMask[j] = edgeMask[j + 1] = edgeMask[j + 2] = inside ? 255 : 0
      edgeMask[j + 3] = 255
      edgeGuide[i] = (x + y) / (ew + eh - 2)
    }
  }
  const edgeBefore = edgeMask.slice()
  const edgeResult = refineMaskEdges(edgeMask, ew, eh, edgeGuide, { band: 4, radius: 5 })
  let soft = 0
  for (let i = 0; i < ew * eh; i += 1) {
    const v = edgeMask[i * 4]
    if (v > 0 && v < 255) soft += 1
  }
  check(
    'edge refiner: boundary stays local while producing soft alpha',
    edgeResult.bandPixels > 0 && soft > 0
      && edgeMask[(14 * ew + 18) * 4] === edgeBefore[(14 * ew + 18) * 4]
      && edgeMask[0] === edgeBefore[0],
    `band=${edgeResult.bandPixels}, soft=${soft}`,
  )
  const gpu = {
    webgpu: true,
    f16: true,
    textureLimit: 16384,
    storageBufferLimit: 256 * 1024 * 1024,
  }
  const fourGB = classifyCapability({ ...gpu, browserMemoryGB: 4 })
  const browserEightGB = classifyCapability({ ...gpu, browserMemoryGB: 8 })
  const unknownMemory = classifyCapability({ ...gpu, browserMemoryGB: 0 })
  const phosmith16GB = classifyCapability({
    ...gpu,
    browserMemoryGB: 8,
    hostResources: { memoryBudgetGB: 16, vramGB: 12, mode: 'performance' },
  })
  const phosmith24GB = classifyCapability({
    ...gpu,
    browserMemoryGB: 8,
    hostResources: { memoryBudgetGB: 24, vramGB: 24, allowFlagship: true, gpuName: 'RTX 4090' },
  })
  check(
    'capability: 4 GB gets the constrained profile',
    fourGB.profile === 'lite' && fourGB.proxyMax === 768 && fourGB.gpuTier === 'accelerated',
    JSON.stringify(fourGB),
  )
  check(
    'capability: browser 8 GB remains a safe standard profile',
    browserEightGB.profile === 'standard' && browserEightGB.proxyMax === 1024
      && browserEightGB.memorySource === 'browser',
    JSON.stringify(browserEightGB),
  )
  const unknownBudget = resolveBudget('', unknownMemory)
  check(
    'policy: unknown browser memory uses the standard-safe allocation cap',
    unknownMemory.profile === 'standard' && unknownMemory.memorySource === 'unknown'
      && unknownBudget.memoryUncertain === true && unknownBudget.exportMaxMP === 16
      && unknownBudget.maxResidentHeavy === 1,
    JSON.stringify({ unknownMemory, unknownBudget }),
  )
  check(
    'capability: trusted 16 GB Phosmith budget unlocks pro',
    phosmith16GB.profile === 'pro' && phosmith16GB.proxyMax === 1280
      && phosmith16GB.memorySource === 'phosmith' && phosmith16GB.vramGB === 12,
    JSON.stringify(phosmith16GB),
  )
  check(
    'capability: trusted 24 GB / RTX-class budget unlocks ultra',
    phosmith24GB.profile === 'ultra' && phosmith24GB.proxyMax === 1536
      && phosmith24GB.flagshipEligible && phosmith24GB.gpuName === 'RTX 4090',
    JSON.stringify(phosmith24GB),
  )
  const noProxy = resolveBudget('?proxy=off', browserEightGB)
  check(
    'policy: proxy-off is explicit but retains the device safety cap',
    noProxy.proxyMode === 'disabled' && noProxy.proxyMax === 0 && noProxy.safeProxyMax === 1024
      && noProxy.exportMaxMP === 24 && noProxy.maxResidentHeavy === 1,
    JSON.stringify(noProxy),
  )
  const ultraBudget = resolveBudget('', phosmith24GB)
  const pressuredUltra = applyMemoryPressure(ultraBudget, 3)
  check(
    'policy: trusted host can authorize ultra/SAM3; pressure degrades safely',
    ultraBudget.profile === 'ultra' && ultraBudget.exportMaxMP === 64 && ultraBudget.flagship === true
      && pressuredUltra.pressureLevel === 3 && pressuredUltra.exportMaxMP === 24
      && pressuredUltra.proxyMax === 768 && pressuredUltra.flagshipCacheMax === 0,
    JSON.stringify({ ultraBudget, pressuredUltra }),
  )

  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
  })

  // M5 phases assume a cold embed store; the persistent profile could carry
  // one over if a previous run landed on the same port. Clear once.
  const pageZ = await context.newPage()
  await pageZ.goto(`http://127.0.0.1:${port}/`)
  await pageZ.evaluate(() => navigator.storage.getDirectory()
    .then((r) => r.removeEntry('seglab-embeds', { recursive: true }))
    .catch(() => {}))
  await pageZ.close()

  // Phosmith injects its memory/VRAM budget before load, but it can also lower
  // it after an OS pressure event. Exercise the documented event contract
  // without loading a model or image.
  const pageHost = await context.newPage()
  await pageHost.goto(`http://127.0.0.1:${port}/`)
  await pageHost.waitForFunction(() => window.__seglabReady === true)
  const hostUpdate = await pageHost.evaluate(async () => {
    window.__PHOSMITH_DEVICE_RESOURCES__ = {
      memoryBudgetGB: 24, vramGB: 24, gpuName: 'RTX 4090', allowFlagship: true,
    }
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', {
      detail: window.__PHOSMITH_DEVICE_RESOURCES__,
    }))
    const high = await window.__seglab.resourceBudget()
    window.__PHOSMITH_DEVICE_RESOURCES__ = { memoryBudgetGB: 4, mode: 'conservative' }
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', {
      detail: window.__PHOSMITH_DEVICE_RESOURCES__,
    }))
    const low = await window.__seglab.resourceBudget()
    return { high, low }
  })
  check(
    'Phosmith event: resource budget upgrades and safely downgrades live',
    hostUpdate.high.profile === 'ultra' && hostUpdate.high.exportMaxMP === 64 && hostUpdate.high.flagship
      && hostUpdate.low.profile === 'lite' && hostUpdate.low.exportMaxMP === 8
      && hostUpdate.low.flagship === false,
    JSON.stringify(hostUpdate),
  )
  await pageHost.close()

  /* ─── Phase A: draft lane, deterministic ────────────────────────────── */
  const page = await newAppPage(context, '?flagship=0')
  log('phase A (draft lane) — first click downloads the model on a cold profile…')
  const nativeFrame = await page.evaluate(() => window.__seglab.imageTransform())
  check(
    'adaptive proxy: small source disables the proxy frame',
    nativeFrame && nativeFrame.proxyActive === false && nativeFrame.proxyReason === 'native',
    JSON.stringify(nativeFrame),
  )
  const idlePrewarmUi = await page.evaluate(() => ({
    prepHidden: document.getElementById('prep')?.hidden,
    stageVisible: document.getElementById('stage')?.classList.contains('visible'),
    pickEnabled: !document.getElementById('pick')?.disabled,
  }))
  check(
    'idle preprocessing leaves the editor usable',
    idlePrewarmUi.prepHidden && idlePrewarmUi.stageVisible && idlePrewarmUi.pickEnabled,
    JSON.stringify(idlePrewarmUi),
  )

  // 1+2+3. Disc click: selection quality + hygiene + edge refinement.
  const s1 = await page.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: DISC.x, y: DISC.y },
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
  const s2 = await page.evaluate(() => window.__seglab.clickAt(60, 60, true))
  check(
    'repeat click skips the encoder (cache hit)',
    s2.lastRun && s2.lastRun.encoded === false,
    `encoded=${s2.lastRun?.encoded}, decode ${s2.lastRun?.decodeMs}ms, post ${s2.lastRun?.postMs}ms`,
  )

  // 5. Minute object: the 9px dot must survive the hygiene pass (it is
  // seeded by the click, so island filtering must NOT eat it).
  await page.evaluate(() => window.__seglab.reset())
  const s3 = await page.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: DOT.x, y: DOT.y },
  )
  const dotArea = (Math.PI * DOT.r * DOT.r) / FRAME
  const b3 = s3.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    'minute object (r=9px) is selectable',
    s3.maskSummary && b3[0] <= DOT.x && DOT.x <= b3[2] && s3.maskSummary.coverage < dotArea * 40,
    `coverage ${((s3.maskSummary?.coverage || 0) * 100).toFixed(2)}% (dot ${(dotArea * 100).toFixed(3)}%), bbox [${b3}]`,
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
    `bbox [${b4}] within ±${Math.round(clampR)} of (${SQUARE.x},${SQUARE.y}); coverage ${((s4.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )

  // Region-fill modes now accumulate (include unions, exclude subtracts), so a
  // standalone-coverage check must start from a clean mask.
  await page.evaluate(() => window.__seglab.reset())
  const region = await page.evaluate(
    ({ x, y, r }) => window.__seglab.manualRegionCircle(x, y, r),
    { x: DISC.x, y: DISC.y, r: DISC.r * 0.7 },
  )
  const regionArea = Math.PI * (DISC.r * 0.7) ** 2 / FRAME
  check(
    'manual region: drawn area is selected without object snapping',
    region.manual === 'region' && region.components === 1
      && region.maskSummary.coverage > regionArea * 0.85 && region.maskSummary.coverage < regionArea * 1.15,
    `coverage ${((region.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs drawn ${(regionArea * 100).toFixed(1)}%`,
  )

  await page.evaluate(() => window.__seglab.reset())
  const colour = await page.evaluate(
    ({ x, y }) => window.__seglab.colorAt(x, y, 24),
    { x: DISC.x, y: DISC.y },
  )
  const discArea = Math.PI * DISC.r * DISC.r / FRAME
  check(
    'color range: matching pixels select without object inference',
    colour.manual === 'color' && colour.components === 1
      && colour.maskSummary.coverage > discArea * 0.8 && colour.maskSummary.coverage < discArea * 1.2,
    `coverage ${((colour.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discArea * 100).toFixed(1)}%`,
  )
  const manualExport = await page.evaluate(() => window.__seglab.exportCutout())
  check(
    'manual masks: export preserves the drawn boundary without AI refinement',
    manualExport && manualExport.decoded === false
      && Math.abs(manualExport.coverage - colour.maskSummary.coverage) < 0.01,
    `decoded=${manualExport?.decoded}, coverage ${(manualExport?.coverage * 100 || 0).toFixed(1)}%`,
  )

  // Include/exclude on the predictive/region modes: a second disjoint region
  // unions into the mask (include), and re-applying it as negative (exclude —
  // right/Alt-click or the sign toggle) carves it back out.
  await page.evaluate(() => window.__seglab.reset())
  const addA = await page.evaluate(() => window.__seglab.manualRect(50, 50, 150, 150))
  const addB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150))
  const areaA = addA.maskSummary?.coverage || 0
  check(
    'include: a second region unions into the mask (two components)',
    addB.components === 2 && Math.abs((addB.maskSummary?.coverage || 0) - areaA * 2) < areaA * 0.1,
    `A ${(areaA * 100).toFixed(2)}% → A∪B ${((addB.maskSummary?.coverage || 0) * 100).toFixed(2)}%, components=${addB.components}`,
  )
  const subB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150, true))
  check(
    'exclude: re-applying the region as negative carves it back out',
    subB.components === 1 && Math.abs((subB.maskSummary?.coverage || 0) - areaA) < areaA * 0.1,
    `A∪B∖B ${((subB.maskSummary?.coverage || 0) * 100).toFixed(2)}% vs A ${(areaA * 100).toFixed(2)}%`,
  )

  const stA = await page.evaluate(() => window.__seglab.state())
  log(`phase A done: engine=${stA.mode} device=${stA.device} lane=${stA.lane}`)
  await page.close()

  // A user can request `?proxy=off`, but an 8 GB-safe budget must reject that
  // request for a multi-megapixel source rather than allocate a giant canvas.
  const pageProxy = await newAppPage(context, '?flagship=0&proxy=off', 4000)
  const safeFrame = await pageProxy.evaluate(() => window.__seglab.imageTransform())
  check(
    'adaptive proxy: unsafe proxy-off request restores device cap',
    safeFrame && safeFrame.proxyActive === true && safeFrame.proxyReason === 'safety'
      && safeFrame.proxyW < safeFrame.originalW,
    JSON.stringify(safeFrame),
  )
  await pageProxy.close()

  if (RAW_FIXTURE) {
    if (!existsSync(RAW_FIXTURE)) throw new Error(`RAW_FIXTURE not found: ${RAW_FIXTURE}`)
    log(`RAW fixture — ${path.basename(RAW_FIXTURE)}`)
    const pageRaw = await newAppPage(context, '?flagship=0')
    await pageRaw.locator('#file').setInputFiles(RAW_FIXTURE)
    await pageRaw.waitForFunction(() => {
      const t = window.__seglab.imageTransform()
      return t?.proxyActive && t.originalW > t.proxyW && t.originalH > t.proxyH
    }, null, { timeout: 60_000 })
    const rawFrame = await pageRaw.evaluate(() => window.__seglab.imageTransform())
    check(
      'RAW import: embedded preview stays inside the interaction budget',
      rawFrame && rawFrame.proxyW <= 1024 && rawFrame.proxyH <= 1024
        && rawFrame.originalW > rawFrame.proxyW && rawFrame.originalH > rawFrame.proxyH
        && rawFrame.sourceWasRaw === true && rawFrame.sourceBytes > 0,
      JSON.stringify(rawFrame),
    )
    await pageRaw.close()
  }

  /* ─── Phase A2: M0 revision/cancel — overlapping prompts, one commit ──── */
  // The 920px size is unique to this phase so the import-time eager encode
  // (M5) is genuinely cold: click A queues behind a multi-second encode and
  // click B must obsolete it (stale or superseded) — only B commits.
  // (Same content as Phase A would hit OPFS and make the race flaky.)
  const pageC = await newAppPage(context, '?flagship=0', 920)
  log('phase A2 (revision/cancel) — second click lands while the first is in flight…')
  const m0 = await pageC.evaluate(async ({ disc, sq }) => {
    const a = window.__seglab.clickAt(disc.x, disc.y)
    await new Promise((r) => setTimeout(r, 350)) // run A started (80 ms debounce), mid-encode
    const b = window.__seglab.clickAt(sq.x, sq.y) // bumps revision → cancels A
    await Promise.all([a, b])
    return { log: window.__seglab.commitLog.slice(), revision: window.__seglab.revision() }
  }, { disc: DISC, sq: SQUARE })
  const committed = m0.log.filter((e) => e.outcome === 'committed')
  check(
    'revision/cancel: only the newest prompt commits',
    committed.length === 1 && committed[0].revision === m0.revision
      && m0.log.some((e) => e.revision < m0.revision && (e.outcome === 'stale' || e.outcome === 'superseded')),
    `log ${JSON.stringify(m0.log)} (current revision ${m0.revision})`,
  )
  await pageC.close()

  /* ─── Phase A3: M1 native-resolution HD export ──────────────────────── */
  // Original 2400 px wide > 1024 proxy, so export genuinely upscales. Click
  // the disc (proxy coords), export, and measure the composited full-res
  // cutout against the analytic disc (original coords).
  const pageD = await newAppPage(context, '?flagship=0', 2400)
  log('phase A3 (HD export) — original 2400px, proxy 1024px, native-res cutout…')
  const geo = await pageD.evaluate(() => window.__seglab.demoGeometry())
  await pageD.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geo.disc.x * geo.proxyScale, y: geo.disc.y * geo.proxyScale },
  )
  const ex = await pageD.evaluate(
    (probe) => window.__seglab.exportCutout(probe),
    { cx: geo.disc.x, cy: geo.disc.y, r: geo.disc.r },
  )
  check(
    'HD export: cutout matches the original dimensions',
    ex && ex.w === geo.originalW && ex.h === geo.originalH,
    `export ${ex?.w}×${ex?.h} vs original ${geo.originalW}×${geo.originalH}`,
  )
  check(
    'HD export: re-decoded the crop at native resolution',
    ex && ex.decoded === true,
    `decoded=${ex?.decoded}`,
  )
  check(
    'HD export: alpha is opaque on the disc, clear off it',
    ex && ex.centerOpaque && ex.outsideTransparent,
    `centerOpaque=${ex?.centerOpaque} outsideTransparent=${ex?.outsideTransparent}`,
  )
  check(
    'HD export: boundary within 3px of the analytic disc (no loss)',
    ex && ex.radialErr <= 3 && ex.softPixels > 100,
    `radialErr=${ex?.radialErr?.toFixed(1)}px (r=${geo.disc.r.toFixed(0)}), softPixels=${ex?.softPixels}`,
  )
  await pageD.close()

  /* ─── Phase A4: M3 crop escalation (small-object native recovery) ────── */
  // 4000 px scene, an ~80 px dot: in the ≤1024 proxy it is ~20 px and coarse.
  // Escalation re-decodes ONE native crop and merges it back; its boundary
  // must beat the ?escalate=0 control (proxy only) by a wide margin.
  const runDot = async (query) => {
    const p = await newAppPage(context, query, 4000)
    const geo = await p.evaluate(() => window.__seglab.demoGeometry())
    await p.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: geo.dot.x * geo.proxyScale, y: geo.dot.y * geo.proxyScale },
    )
    const esc = await p.evaluate(
      (probe) => window.__seglab.escalation(probe),
      { cx: geo.dot.x, cy: geo.dot.y, r: geo.dot.r },
    )
    await p.close()
    return { esc, r: geo.dot.r }
  }
  log('phase A4 (crop escalation) — 4000px scene, small dot, native re-decode vs ?escalate=0…')
  const on = await runDot('?flagship=0')
  const off = await runDot('?flagship=0&escalate=0')
  check(
    'escalation: fires on Std/Pro for a small object, off under ?escalate=0',
    on.esc.fired === true && on.esc.decoded === true && on.esc.centerOpaque === true && off.esc.fired === false,
    `on{fired:${on.esc.fired},decoded:${on.esc.decoded},center:${on.esc.centerOpaque}} off{fired:${off.esc.fired}}`,
  )
  check(
    'escalation: native re-decode recovers the boundary (materially better)',
    on.esc.radialErr != null && off.esc.radialErr != null
      && on.esc.radialErr <= 6 && on.esc.radialErr < off.esc.radialErr * 0.7,
    `escalate=1 ${on.esc.radialErr?.toFixed(1)}px vs control ${off.esc.radialErr?.toFixed(1)}px (r=${on.r.toFixed(0)})`,
  )

  /* ─── Phase A4b: bounded working copy (Safari-shaped unbounded decode) ── */
  // Hosts without ImageDecoder (Safari) full-raster-decode every blob
  // operation. ?working=1 forces that ladder here: an oversized UPLOAD keeps
  // a ≤4096 working copy, escalation decodes correctly from it (prompt
  // rescale), and export still composites the native original.
  log('phase A4b (working copy) — ?working=1, 5000px blob upload…')
  const pageWk = await newAppPage(context, '?flagship=0&working=1')
  await pageWk.evaluate((side) => window.__seglab.loadDemoBlob(side), 5000)
  const wkFrame = await pageWk.evaluate(() => window.__seglab.imageTransform())
  check(
    'working copy: oversized upload keeps a ≤4096 bounded re-decode source',
    wkFrame && wkFrame.proxyActive === true && wkFrame.workingActive === true
      && Math.max(wkFrame.workingW || 0, wkFrame.workingH || 0) === 4096,
    JSON.stringify({ working: `${wkFrame?.workingW}x${wkFrame?.workingH}`, proxy: `${wkFrame?.proxyW}x${wkFrame?.proxyH}` }),
  )
  const geoWk = await pageWk.evaluate(() => window.__seglab.demoGeometry())
  await pageWk.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geoWk.dot.x * geoWk.proxyScale, y: geoWk.dot.y * geoWk.proxyScale },
  )
  const escWk = await pageWk.evaluate(
    (probe) => window.__seglab.escalation(probe),
    { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r },
  )
  check(
    'working copy: escalation decodes a bounded crop (rescaled prompts land)',
    escWk.fired === true && escWk.decoded === true && escWk.centerOpaque === true && escWk.radialErr <= 6,
    `fired=${escWk.fired} decoded=${escWk.decoded} radialErr=${escWk.radialErr?.toFixed(1)}px`,
  )
  const exWk = await pageWk.evaluate(
    (probe) => window.__seglab.exportCutout(probe),
    { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r },
  )
  check(
    'working copy: export bypasses it — native dimensions, fresh decode',
    exWk && exWk.w === geoWk.originalW && exWk.h === geoWk.originalH
      && exWk.decoded === true && exWk.radialErr <= 3,
    `export ${exWk?.w}×${exWk?.h} vs original ${geoWk?.originalW}×${geoWk?.originalH}, decoded=${exWk?.decoded}, radialErr=${exWk?.radialErr?.toFixed(1)}px`,
  )
  await pageWk.close()

  /* ─── Phase A5: M2 text-select plumbing (box → mask → union) ─────────── */
  // The detector (OWLv2) can't build a session headless (ORT op gap), so the
  // live phrase→boxes step is confirmed in real Chrome (WARN below). The
  // machinery that turns a chosen box into a mask and unions instances is
  // SAM-based and gated deterministically here.
  const pageE = await newAppPage(context, '?flagship=0')
  log('phase A5 (text-select plumbing) — candidate box → mask → multi-union…')
  const discBox = [120, 230, 340, 450]
  const squareBox = [515, 145, 715, 345]
  const one = await pageE.evaluate((b) => window.__seglab.selectBoxes([b]), discBox)
  checkDisc('text', one)
  await pageE.evaluate(() => window.__seglab.reset())
  const both = await pageE.evaluate(([a, b]) => window.__seglab.selectBoxes([a, b]), [discBox, squareBox])
  check(
    'text select: "all" unions instances → 2 components',
    both && both.components === 2,
    `components=${both?.components}, coverage ${((both?.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )
  await pageE.evaluate(() => window.__seglab.reset())
  const brushAdd = await pageE.evaluate(
    (points) => window.__seglab.brushStroke(points),
    [[390, 300], [490, 300], [560, 345]],
  )
  const brushErase = await pageE.evaluate(
    (points) => window.__seglab.brushStroke(points, true),
    [[390, 300], [435, 300]],
  )
  check(
    'brush: canvas stroke commits a mask and erases incrementally',
    brushAdd?.manual === 'brush' && brushAdd.maskSummary?.coverage > 0
      && brushErase?.manual === 'brush' && brushErase.maskSummary?.coverage > 0
      && brushErase.maskSummary.coverage < brushAdd.maskSummary.coverage,
    `add=${((brushAdd?.maskSummary?.coverage || 0) * 100).toFixed(2)}% erase=${((brushErase?.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  // The live OWLv2 detector is NOT exercised here: it can't build a session
  // under headless ORT (the trimmed ladder now pulls one WebGPU + one q4 WASM
  // build, not four). Confirm phrase→boxes in real Chrome (chip shows candidates).
  log('⚠ phase A5: OWLv2 phrase→boxes not gated headless (ORT op gap) — confirm in real Chrome')
  await pageE.close()

  /* ─── Phase A6: capability probe + weak-device (wasm) full pipeline ─── */
  // (a) browser-only 8 GB reports remain Standard (not guessed Pro); (b)
  // forced WASM still completes click + text + export; (c) the pressure ladder
  // frees reloadable residents on demand.
  const pageP = await newAppPage(context, '?flagship=0', 900)
  const cap = await pageP.evaluate(() => window.__seglab.capability())
  check(
    'capability: 8 GB browser probe remains standard on a real WebGPU adapter',
    cap && cap.profile === 'standard' && cap.webgpu === true && cap.fallback === false
      && cap.memorySource === 'browser',
    JSON.stringify(cap),
  )
  await pageP.close()

  const pageW = await newAppPage(context, '?force=wasm', 900)
  log('phase A6 (weak-device) — full pipeline forced onto wasm (click + text + export)…')
  const wClick = await pageW.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: DISC.x, y: DISC.y })
  await pageW.evaluate(() => window.__seglab.reset())
  const wText = await pageW.evaluate((b) => window.__seglab.selectBoxes([b]), [120, 230, 340, 450])
  const wExport = await pageW.evaluate(() => window.__seglab.exportCutout())
  check(
    'weak-device (wasm): click + text + export all complete',
    wClick.device === 'wasm' && !!wClick.maskSummary && !!wText.maskSummary
      && wExport && wExport.w === 900 && wExport.coverage > 0,
    `device=${wClick.device} click=${!!wClick.maskSummary} text=${!!wText?.maskSummary} export=${wExport?.w}×${wExport?.h} cov=${((wExport?.coverage || 0) * 100).toFixed(1)}%`,
  )
  const freed = await pageW.evaluate(() => window.__seglab.relievePressure(3))
  check(
    'pressure ladder: level 3 frees detector + flagship embeddings + session',
    Array.isArray(freed) && ['detector', 'flagship-embeddings', 'flagship-session'].every((f) => freed.includes(f)),
    `freed=${JSON.stringify(freed)}`,
  )
  await pageW.close()

  /* ─── Phase A7: M5 zero-cloud proof — warmed profile, then offline ────── */
  // Import a 2400px doc online (models + OPFS warm from earlier phases),
  // then CUT the network at the context level and run a fresh click, the
  // text plumbing, and a native-crop export. The export encodes a crop from
  // scratch — real inference, provably needing no network. Route
  // interception is deliberately NOT used: it would bypass the HTTP cache
  // and false-fail the very caching this proves.
  const pageO = await newAppPage(context, '?flagship=0', 2400)
  await pageO.evaluate(() => window.__seglab.eagerEncode()) // embedding resident before the cut
  const geoO = await pageO.evaluate(() => window.__seglab.demoGeometry())
  let attempted = 0
  let succeeded = 0
  try {
    await context.setOffline(true)
    // Attached after the cut: only requests inside the offline window count.
    pageO.on('request', () => { attempted += 1 })
    pageO.on('requestfinished', () => { succeeded += 1 })
    log('phase A7 (offline) — network cut; click + text plumbing + HD export…')
    const oClick = await pageO.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: geoO.disc.x * geoO.proxyScale, y: geoO.disc.y * geoO.proxyScale },
    )
    await pageO.evaluate(() => window.__seglab.reset())
    const dbox = [
      (geoO.disc.x - geoO.disc.r * 1.2) * geoO.proxyScale,
      (geoO.disc.y - geoO.disc.r * 1.2) * geoO.proxyScale,
      (geoO.disc.x + geoO.disc.r * 1.2) * geoO.proxyScale,
      (geoO.disc.y + geoO.disc.r * 1.2) * geoO.proxyScale,
    ]
    const oText = await pageO.evaluate((b) => window.__seglab.selectBoxes([b]), dbox)
    const oEx = await pageO.evaluate(
      (probe) => window.__seglab.exportCutout(probe),
      { cx: geoO.disc.x, cy: geoO.disc.y, r: geoO.disc.r },
    )
    check(
      'offline: click + text + native export all pass with the network cut',
      !!oClick?.maskSummary && !!oText?.maskSummary
        && oEx && oEx.w === geoO.originalW && oEx.decoded === true && oEx.centerOpaque,
      `click=${!!oClick.maskSummary} text=${!!oText?.maskSummary} export=${oEx?.w}×${oEx?.h} decoded=${oEx?.decoded} radialErr=${oEx?.radialErr?.toFixed(1)}px`,
    )
    check(
      'offline: zero successful network fetches during inference',
      succeeded === 0,
      `${attempted} attempted, ${succeeded} succeeded`,
    )
  } finally {
    await context.setOffline(false) // Phase B needs the network back
  }
  await pageO.close()

  /* ─── Phase A8: M5 revisit — OPFS embedding survives a fresh session ──── */
  // The 1600px size is unique to this phase (store cleared at run start).
  // Visit 1: cold encode + OPFS save (eagerEncode resolves after the write).
  // Visit 2 is a NEW page — empty in-memory cache — so encoded:false can
  // only mean the embedding came back from OPFS; the click then proves the
  // restored tensors actually decode.
  log('phase A8 (revisit) — OPFS persistence across sessions…')
  const pageR1 = await newAppPage(context, '?flagship=0', 1600)
  const r1 = await pageR1.evaluate(() => window.__seglab.eagerEncode())
  await pageR1.close()
  check('revisit: first visit encodes fresh (cold store)', r1?.encoded === true, `encoded=${r1?.encoded} lane=${r1?.lane}`)

  const pageR2 = await newAppPage(context, '?flagship=0', 1600)
  const r2 = await pageR2.evaluate(() => window.__seglab.eagerEncode())
  const geoR = await pageR2.evaluate(() => window.__seglab.demoGeometry())
  const sR = await pageR2.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geoR.disc.x * geoR.proxyScale, y: geoR.disc.y * geoR.proxyScale },
  )
  const bR = sR.maskSummary?.bbox || [0, 0, -1, -1]
  const cxp = geoR.disc.x * geoR.proxyScale
  const cyp = geoR.disc.y * geoR.proxyScale
  const discFracR = (Math.PI * geoR.disc.r * geoR.disc.r) / (geoR.originalW * geoR.originalH)
  check(
    'revisit: fresh session serves the import encode from OPFS',
    r2?.encoded === false && sR.lastRun?.encoded === false,
    `eager encoded=${r2?.encoded}, first click encoded=${sR.lastRun?.encoded}`,
  )
  check(
    'revisit: restored embedding decodes a sane disc',
    !!sR?.maskSummary && bR[0] <= cxp && cxp <= bR[2] && bR[1] <= cyp && cyp <= bR[3]
      && sR.maskSummary.coverage > discFracR * 0.4 && sR.maskSummary.coverage < discFracR * 3,
    `bbox [${bR.map((v) => Math.round(v))}] vs (${cxp.toFixed(0)},${cyp.toFixed(0)}); coverage ${((sR.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discFracR * 100).toFixed(1)}%`,
  )
  await pageR2.close()

  /* ─── Phase B: explicit flagship upgrade (WARN-only) ────────────────── */
  if (process.env.VERIFY_FLAGSHIP === '1') {
    try {
    const pageB = await newAppPage(context, '?flagship=1') // flagship is opt-in now
    log('phase B (flagship) — waiting for the background SAM3 upgrade (first run downloads ~300 MB)…')
    await pageB.evaluate(
      () => window.__seglab.clickAt(230, 340), // draft-lane click while SAM3 downloads
    )
    await pageB.waitForFunction(
      () => window.__seglab.state().lane === 'sam3',
      null,
      { timeout: TIMEOUT_MS },
    )
    // The lane flip triggers prompt replay; wait for it to settle, then
    // re-click to measure a clean flagship pass.
    await pageB.evaluate(() => window.__seglab.reset())
    const sB = await pageB.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: DISC.x, y: DISC.y },
    )
    checkDisc('sam3', sB)
    check(
      'sam3: lane confirmed on the result',
      sB.lastRun?.lane === 'sam3',
      `lane=${sB.lastRun?.lane}, encode ${sB.lastRun?.encodeMs}ms, decode ${sB.lastRun?.decodeMs}ms`,
    )
    await pageB.close()
    } catch (err) {
      log(`⚠ phase B: flagship lane not confirmed in headless Chromium (${String(err?.message || err).slice(0, 120)})`)
      log('⚠ this is a WARN, not a failure — confirm SAM3 in real Chrome (chip should read "· sam3")')
    }
  } else {
    log('phase B skipped — set VERIFY_FLAGSHIP=1 to download and test the optional ~300 MB SAM3 lane')
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

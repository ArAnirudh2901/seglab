#!/usr/bin/env bun
/**
 * Headless-browser verification of SEGLAB end-to-end.
 *
 * Serves this directory, drives the REAL app (index.html + workers + models
 * via CDN) with Playwright Chromium through the window.__seglab test hooks.
 *
 * Suites:
 *   T  — pure logic (text-core, edge-refine, capability, policy, sizing)
 *   Q  — heavy-job queue contracts (node-side, no browser)
 *   S  — memory-contract static source scans
 *   A  — lite (memory-locked) browser phases: 1024 proxy, unsafe-flag lockout,
 *        one embedding, no OPFS persistence, revision/cancel, export caps,
 *        wasm cv-refine gates, decode/model serialization
 *   H  — trusted-host (Phosmith pro) phases: HD export, escalation, working
 *        copy, OPFS revisit — the paths a locked browser never reaches
 *   O  — offline zero-cloud proof
 *
 * Usage: bun verify.mjs
 */

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import {
  collapseToObject, colorEvidenceForBox, DETECTOR_INPUT, letterboxPlan, normalizePhrase, nms, rankDetections, scaleBox, unletterboxBox,
} from './js/text-core.js'
import { classifyCapability } from './js/capability.js'
import { refineMaskEdges } from './js/edge-refine.js'
import { applyMemoryPressure, resolveBudget, PROFILE_PRESETS } from './js/policy.js'
import { getBoundedProxySize, displayPlan, decodeBudgetMP } from './js/proxy-plan.js'
import {
  composeChannels, maskChannelCoverages, maskToChannel, pickBestMask, pointInMask, RUNAWAY_COVERAGE,
} from './js/sam-core.js'
import { enqueueHeavy, cancelHeavyBefore, STALE, getHeavyQueueState } from './js/heavy-job-queue.js'
import { extractRawPreview } from './js/image-raw.js'

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
    // Serve the optional RAW fixture to the browser suite (the develop worker
    // fetches it same-origin instead of shuttling megabytes over evaluate()).
    if (url.pathname === '/__raw_fixture' && RAW_FIXTURE && existsSync(RAW_FIXTURE)) {
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' })
      res.end(await readFile(RAW_FIXTURE))
      return
    }
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
// Demo-scene ground truth in the 900×620 logical space (buildDemoScene).
const DISC = { x: 230, y: 340, r: 105 }
const SQUARE = { x: 615, y: 245, half: 95 }
const DOT = { x: 700, y: 480, r: 9 }
const FRAME = 900 * 620

// Trusted Phosmith budgets used by the H phases (a plain browser is locked
// to lite; these are the only route to standard/pro behaviour).
const HOST_PRO = { memoryBudgetGB: 16, vramGB: 12 }
const HOST_ULTRA = { memoryBudgetGB: 24, vramGB: 24, gpuName: 'RTX 4090' }

let pageSeq = 0
const newAppPage = async (context, query, longSide, { host = null, restore = false } = {}) => {
  const page = await context.newPage()
  const tag = `p${++pageSeq}`
  // Synthetic phases must not inherit a session an earlier phase persisted.
  // Phase P opts back in — it is the gate that exercises restore.
  if (!restore) await page.addInitScript(() => { window.__seglabNoRestore = true })
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[seglab]') || msg.type() === 'error') log(`browser[${tag}]: ${text.slice(0, 180)}`)
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
  page.setDefaultTimeout(TIMEOUT_MS)
  if (host) await page.addInitScript((h) => { window.__PHOSMITH_DEVICE_RESOURCES__ = h }, host)
  await page.goto(`http://127.0.0.1:${port}/${query}`)
  await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  if (longSide !== null) await page.evaluate((ls) => window.__seglab.loadDemo(ls), longSide ?? undefined)
  return page
}

/** Disc-quality bar in whatever frame the page uses; coords pre-scaled. */
const checkDisc = (tag, s, cx, cy, discFrac) => {
  const b = s.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    `${tag}: click selects the disc`,
    b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3],
    `bbox [${b.map((v) => Math.round(v))}] vs centre (${cx.toFixed(0)},${cy.toFixed(0)})`,
  )
  check(
    `${tag}: disc mask is object-sized, not a flood`,
    s.maskSummary && s.maskSummary.coverage > discFrac * 0.4 && s.maskSummary.coverage < discFrac * 3,
    `coverage ${((s.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discFrac * 100).toFixed(1)}%`,
  )
}
const DISC_FRAC = (Math.PI * DISC.r * DISC.r) / FRAME

let context = null
let failed = false
try {
  /* ─── Phase T: pure logic (no browser) ──────────────────────────────── */
  const np = normalizePhrase('all red cars')
  check(
    'text-core: phrase → template + multi intent',
    np && np.multi === true && np.color === 'red' && np.labels[0] === 'a photo of a red car' && np.labels[1] === 'a photo of a car',
    `${JSON.stringify(np)}`,
  )
  const colorFrame = { data: new Uint8ClampedArray(12), width: 4, height: 1, contentWidth: 4, contentHeight: 1 }
  // Two red pixels at left; the broad candidate includes two blue pixels too.
  colorFrame.data.set([230, 35, 30, 240, 45, 35, 30, 70, 220, 20, 60, 210])
  const tightRed = colorEvidenceForBox(colorFrame, [0, 0, 0.5, 1], 'red')
  const broadScene = colorEvidenceForBox(colorFrame, [0, 0, 1, 1], 'red')
  check(
    'text-core: requested-colour evidence favours a tight matching box',
    tightRed === 1 && broadScene > 0 && broadScene < tightRed,
    `tight=${tightRed.toFixed(2)} broad=${broadScene.toFixed(2)}`,
  )
  const deduped = nms([
    { box: [0, 0, 100, 100], score: 0.9 },
    { box: [5, 5, 105, 105], score: 0.8 },
    { box: [400, 400, 500, 500], score: 0.7 },
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
  const plan = letterboxPlan(1200, 800, DETECTOR_INPUT)
  const full = unletterboxBox([0, 0, 1, 640 / 960], plan)
  check(
    'text-core: letterbox plan preserves aspect; boxes map back to source px',
    plan.dw === 960 && plan.dh === 640 && full[2] === 1200 && Math.round(full[3]) === 800,
    `plan=${plan.dw}x${plan.dh} full=[${full}]`,
  )

  // Singular phrase: a train split front/rear collapses to one box; a distinct
  // far object and a small sign stay out of it.
  const fragments = [
    { box: [560, 635, 815, 920], score: 0.42, label: 'train' }, // front
    { box: [1105, 690, 1420, 900], score: 0.31, label: 'train' }, // rear (gap)
    { box: [1780, 800, 1840, 880], score: 0.2, label: 'train' }, // far small object
  ]
  const [merged] = collapseToObject(fragments)
  const twoTrains = collapseToObject([
    { box: [0, 0, 300, 300], score: 0.5 }, { box: [1400, 0, 1700, 300], score: 0.4 },
  ])
  check(
    'text-core: singular phrase collapses split fragments, not distinct objects',
    merged.box[0] === 560 && merged.box[2] === 1420 && merged.box[3] === 920
      && twoTrains.length === 1 && twoTrains[0].box[2] === 300,
    `merged=[${merged.box}] distinct kept ${twoTrains.length}`,
  )

  // Edge refiner: boundary-local soft alpha (unchanged contract).
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

  /* ── Policy: the memory-trust lock ── */
  const gpu = { webgpu: true, f16: true, textureLimit: 16384, storageBufferLimit: 256 * 1024 * 1024 }
  const fourGB = classifyCapability({ ...gpu, browserMemoryGB: 4 })
  const browserEightGB = classifyCapability({ ...gpu, browserMemoryGB: 8 })
  const unknownMemory = classifyCapability({ ...gpu, browserMemoryGB: 0 })
  const phosmith16GB = classifyCapability({ ...gpu, browserMemoryGB: 8, hostResources: HOST_PRO })
  const phosmith24GB = classifyCapability({ ...gpu, browserMemoryGB: 8, hostResources: HOST_ULTRA })
  check(
    'capability: browser-reported and unknown memory NEVER raise the profile — all lite',
    browserEightGB.profile === 'lite' && fourGB.profile === 'lite' && unknownMemory.profile === 'lite'
      && browserEightGB.proxyMax === 1024 && unknownMemory.proxyMax === 1024,
    JSON.stringify({ eight: browserEightGB.profile, four: fourGB.profile, unknown: unknownMemory.profile }),
  )
  check(
    'capability: trusted Phosmith budgets earn pro/ultra',
    phosmith16GB.profile === 'pro' && phosmith16GB.proxyMax === 1280
      && phosmith24GB.profile === 'ultra' && phosmith24GB.flagshipEligible === false,
    JSON.stringify({ pro: phosmith16GB.profile, ultra: phosmith24GB.profile }),
  )

  const liteDefault = resolveBudget('', browserEightGB)
  const unknownBudget = resolveBudget('', unknownMemory)
  const provisional = resolveBudget('', null)
  check(
    'policy: browser-reported / unknown / no-probe all resolve to the locked lite budget',
    liteDefault.profile === 'lite' && unknownBudget.profile === 'lite' && provisional.profile === 'lite'
      && liteDefault.memoryLocked && unknownBudget.memoryLocked && provisional.memoryLocked
      && liteDefault.proxyMax === 1024,
    JSON.stringify({ lite: liteDefault.profile, unknown: unknownBudget.profile, provisional: provisional.profile }),
  )
  check(
    'policy: lite caps — one embedding, accelerator-gated detector, no escalation, bounded export',
    liteDefault.draftCacheMax === 1 && liteDefault.flagshipCacheMax === 0
      && liteDefault.maxResidentHeavy === 1 && liteDefault.flagship === false
      && liteDefault.detectorWebGPU === true && liteDefault.autoEscalate === false
      && liteDefault.hdExportDecode === false && liteDefault.eagerEncode === true
      && liteDefault.embedPersist === false && liteDefault.exportMaxSide === 4096
      && liteDefault.exportMaxMP === 8 && liteDefault.cropMaxSide === 1280
      && liteDefault.detectorDispose === 'idle' && liteDefault.detectorEvictOnEncode === true,
    JSON.stringify(liteDefault),
  )
  check(
    'policy: SAM stays on WASM for the unverified baseline (the OWLv2 lesson applied to SlimSAM)',
    liteDefault.samWebGPU === false && PROFILE_PRESETS.standard.samWebGPU === true
      && resolveBudget('', unknownMemory).samWebGPU === false,
    JSON.stringify({ lite: liteDefault.samWebGPU, standard: PROFILE_PRESETS.standard.samWebGPU }),
  )
  const flagged = resolveBudget('?flagship=1', browserEightGB)
  const ultraReq = resolveBudget('?profile=ultra', browserEightGB)
  const proxyMax = resolveBudget('?proxy=max', browserEightGB)
  const proxyOff = resolveBudget('?proxy=off', unknownMemory)
  const workingForce = resolveBudget('?working=1', browserEightGB)
  check(
    'safety: ?flagship=1 cannot enable SAM3 on an unverified budget',
    flagged.flagship === false && resolveBudget('?flagship=1', unknownMemory).flagship === false,
    `flagship=${flagged.flagship}`,
  )
  check(
    'safety: ?profile=ultra / ?proxy=max / ?proxy=off cannot raise the 1024 px cap',
    ultraReq.profile === 'lite' && ultraReq.proxyMax === 1024
      && proxyMax.proxyMax === 1024 && proxyMax.proxyMode === 'auto'
      && proxyOff.proxyMax === 1024 && proxyOff.proxyMode === 'auto',
    JSON.stringify({ ultra: ultraReq.proxyMax, max: proxyMax.proxyMax, off: proxyOff.proxyMax }),
  )
  check(
    'safety: ?working=1 refused on a locked budget; lowering params still work',
    workingForce.workingMode === undefined
      && resolveBudget('?proxy=512', browserEightGB).proxyMax === 512
      && resolveBudget('?escalate=0', phosmith16GB).autoEscalate === false,
    JSON.stringify({ working: workingForce.workingMode }),
  )
  check(
    'policy: lite working copy is capped at 1280 px',
    PROFILE_PRESETS.lite.workingMaxSide <= 1280 && liteDefault.workingMaxSide <= 1280,
    `workingMaxSide=${PROFILE_PRESETS.lite.workingMaxSide}`,
  )
  const dispNative = resolveBudget('?display=native', browserEightGB)
  const dispOff = resolveBudget('?display=off', browserEightGB)
  const dispCap = resolveBudget('?display=1600', browserEightGB)
  check(
    'display: ?display native/off/<px> honored on a locked budget (display-only, not the memory contract)',
    liteDefault.displayMax === 2048 && dispNative.displayMode === 'native'
      && dispOff.displayMode === 'off' && dispCap.displayMax === 1600,
    JSON.stringify({ base: liteDefault.displayMax, native: dispNative.displayMode, off: dispOff.displayMode, cap: dispCap.displayMax }),
  )
  const ultraBudget = resolveBudget('?flagship=1', phosmith24GB)
  const pressured = applyMemoryPressure(liteDefault, 2)
  const pressured3 = applyMemoryPressure(liteDefault, 3)
  check(
    'policy: SAM3 cannot be enabled on any profile; pressure only tightens',
    ultraBudget.flagship === false && ultraBudget.profile === 'ultra'
      && pressured.cvRefine === false && pressured.hdExportDecode === false
      && pressured.cropMaxSide <= 1280 && pressured.eagerEncode === false
      && pressured3.exportMaxMP === 4 && pressured3.proxyMax === 768
      && applyMemoryPressure(ultraBudget, 3).exportMaxMP <= 24,
    JSON.stringify({ ultraFlagship: ultraBudget.flagship, p2: pressured.cvRefine, p3: pressured3.exportMaxMP }),
  )
  check(
    'policy: pressure ratchet drops SAM WebGPU and lowers the display ceiling',
    applyMemoryPressure(ultraBudget, 1).samWebGPU === false
      && pressured.displayMax === 1600 && pressured3.displayMax === 1280,
    JSON.stringify({ p1sam: applyMemoryPressure(ultraBudget, 1).samWebGPU, p2disp: pressured.displayMax, p3disp: pressured3.displayMax }),
  )

  /* ── Sizing: the authoritative proxy function ── */
  const s1 = getBoundedProxySize(6000, 4000, 768)
  const s2 = getBoundedProxySize(4000, 6000, 768)
  const s3 = getBoundedProxySize(768, 512, 768)
  check(
    'sizing: 6000×4000 → 768×512, portrait → 512×768, small stays native',
    s1.width === 768 && s1.height === 512 && s1.proxyActive
      && s2.width === 512 && s2.height === 768 && s2.proxyActive
      && s3.width === 768 && s3.height === 512 && !s3.proxyActive && s3.scale === 1,
    JSON.stringify({ s1, s2, s3 }),
  )
  let sizingThrew = false
  try { getBoundedProxySize(0, 4000) } catch { sizingThrew = true }
  let sizingThrew2 = false
  try { getBoundedProxySize(NaN, 10) } catch { sizingThrew2 = true }
  check('sizing: invalid dimensions reject cleanly', sizingThrew && sizingThrew2, 'both threw')

  /* ── Display formula: viewport-anchored, decode-budget-gated ── */
  const liteB = { ...PROFILE_PRESETS.lite }
  const vp = { w: 1728, h: 1117, dpr: 2 }
  const dCeiling = displayPlan({ srcW: 8000, srcH: 6000, budget: liteB, viewport: vp, textureLimit: 8192 })
  const dSource = displayPlan({ srcW: 1600, srcH: 1000, budget: liteB, viewport: vp, textureLimit: 8192 })
  const dViewport = displayPlan({ srcW: 8000, srcH: 6000, budget: liteB, viewport: { w: 500, h: 400, dpr: 1 }, textureLimit: 8192 })
  const dTexture = displayPlan({ srcW: 8000, srcH: 6000, budget: { ...liteB, displayMode: 'native' }, viewport: vp, textureLimit: 4096 })
  const dOff = displayPlan({ srcW: 8000, srcH: 6000, budget: { ...liteB, displayMode: 'off' }, viewport: vp })
  check(
    'display formula: side = min(viewport·dpr·slack, profile ceiling, texture cap, source) for any source',
    dCeiling.side === 2048 && !dCeiling.allowFullDecode // 48 MP > decode budget
      && dSource.side === 1600 && dSource.allowFullDecode // 1.6 MP fits
      && dViewport.side === 750 // 500·1·1.5 viewport-bound
      && dTexture.side === 4096 && dTexture.allowFullDecode // native opt-in, texture-capped
      && dOff.side === 0 && !dOff.allowFullDecode,
    JSON.stringify({ ceiling: dCeiling, source: dSource, viewport: dViewport, texture: dTexture, off: dOff }),
  )
  check(
    'display formula: decode budget reuses escalateMaxMP and ratchets 6/4/2 under pressure',
    decodeBudgetMP(liteB) === 8
      && decodeBudgetMP({ ...liteB, pressureLevel: 1 }) === 6
      && decodeBudgetMP({ ...liteB, pressureLevel: 2 }) === 4
      && decodeBudgetMP({ ...liteB, pressureLevel: 3 }) === 2,
    JSON.stringify([0, 1, 2, 3].map((level) => decodeBudgetMP({ ...liteB, pressureLevel: level }))),
  )

  /* ── Click-union composition: add/sub replay + peel ── */
  const chanOf = (bits) => Uint8Array.from(bits.map((v) => (v ? 255 : 0)))
  const rAt = (res, i) => (res ? res.rgba[i * 4] : null)
  const unionOps = [
    { op: 'add', chan: chanOf([1, 1, 0, 0]) },
    { op: 'add', chan: chanOf([0, 0, 1, 0]) },
    { op: 'sub', chan: chanOf([1, 0, 0, 0]) },
  ]
  const composed = composeChannels(unionOps, 4, 1)
  const peeled = composeChannels(unionOps.slice(0, 2), 4, 1)
  const netZero = composeChannels([unionOps[0], { op: 'sub', chan: chanOf([1, 1, 0, 0]) }], 4, 1)
  const floorKeeps = composeChannels([unionOps[2]], 4, 1, chanOf([1, 1, 1, 1]))
  check(
    'click-union: op stack replays in order (add∪add∖sub), peels, nets to empty, respects the floor',
    rAt(composed, 0) === 0 && rAt(composed, 1) === 255 && rAt(composed, 2) === 255 && rAt(composed, 3) === 0
      && rAt(peeled, 0) === 255 && netZero === null
      && rAt(floorKeeps, 0) === 0 && rAt(floorKeeps, 3) === 255,
    JSON.stringify({ composed: composed && [...composed.rgba].filter((_, i) => i % 4 === 0) }),
  )
  /* ── Candidate pick: the whole-scene mask never wins an ambiguous click ── */
  // Three 10x10 candidates: a 9% object, a 25% part, a 96% whole-scene guess
  // scored the way SAM scores a click into a field of near-identical objects
  // — the runaway outscores the object actually pointed at.
  const candidates = (fracs) => {
    const size = 100
    const data = new Uint8Array(size * fracs.length)
    fracs.forEach((f, c) => data.fill(1, c * size, c * size + Math.round(f * size)))
    return data
  }
  const fieldOfFlowers = candidates([0.09, 0.25, 0.96])
  const fieldCov = maskChannelCoverages(fieldOfFlowers, 10, 10, 3)
  const fieldScores = Float32Array.from([0.87, 0.71, 0.94])
  // A frame-filling close-up: every candidate is large and the biggest is the
  // real subject, so the argmax winner must survive the runaway test.
  const closeUpCov = maskChannelCoverages(candidates([0.91, 0.95, 0.98]), 10, 10, 3)
  const closeUpScores = Float32Array.from([0.6, 0.7, 0.9])
  check(
    'candidate pick: an ambiguous click rejects the whole-scene runaway, box/refine and all-large keep argmax',
    pickBestMask(fieldScores, { coverages: fieldCov, ambiguous: true }) === 0
      && pickBestMask(fieldScores, { coverages: fieldCov, ambiguous: false }) === 2
      && pickBestMask(fieldScores) === 2
      && pickBestMask(closeUpScores, { coverages: closeUpCov, ambiguous: true }) === 2,
    `coverages ${fieldCov.map((c) => `${(c * 100).toFixed(0)}%`).join('/')} · runaway ≥ ${RUNAWAY_COVERAGE * 100}%`,
  )

  const maskLike = { data: Uint8ClampedArray.from([255, 255, 255, 255, 0, 0, 0, 0]), width: 2, height: 1 }
  check(
    'click-union: maskToChannel + pointInMask (tolerance) agree with the RGBA mask',
    maskToChannel(maskLike)[0] === 255 && maskToChannel(maskLike)[1] === 0
      && pointInMask(maskLike, 0, 0, 0) === true && pointInMask(maskLike, 1, 0, 0) === false
      && pointInMask(maskLike, 1, 0, 1) === true && pointInMask(null, 0, 0) === false,
    'channel + hit tests agree',
  )

  // The optional fixture exercises the bounded RAW-container parser directly.
  // It deliberately avoids Playwright's slow multi-megabyte file-upload bridge;
  // after extraction, the JPEG preview follows the already-covered Blob decode
  // path in the browser suite below.
  if (RAW_FIXTURE) {
    if (!existsSync(RAW_FIXTURE)) throw new Error(`RAW_FIXTURE not found: ${RAW_FIXTURE}`)
    const rawFile = Bun.file(RAW_FIXTURE)
    Object.defineProperty(rawFile, 'name', { value: path.basename(RAW_FIXTURE) })
    const preview = await extractRawPreview(rawFile, { proxyMinEdge: 768 })
    check(
      'RAW: bounded parser extracts an embedded JPEG and a bounded proxy preview',
      !!preview && preview.width > 768 && preview.height > 0 && preview.blob.size > 0
        && (!preview.proxyBlob || preview.proxyBlob.size <= preview.blob.size),
      JSON.stringify({
        full: preview ? `${preview.width}x${preview.height}` : null,
        jpegBytes: preview?.blob.size || 0,
        proxyBytes: preview?.proxyBlob?.size || 0,
      }),
    )
  }

  /* ─── Phase Q: heavy-job queue contracts ────────────────────────────── */
  {
    let active = 0
    let maxActive = 0
    const order = []
    const job = (name, ms) => async () => {
      active += 1
      maxActive = Math.max(maxActive, active)
      order.push(name)
      await new Promise((r) => setTimeout(r, ms))
      active -= 1
      return name
    }
    const p1 = enqueueHeavy('decode', job('decode', 30), { priority: 'import' })
    const pIdle = enqueueHeavy('prewarm', job('prewarm', 10), { priority: 'idle' })
    const pHigh = enqueueHeavy('segment', job('segment', 10), { priority: 'high' })
    await Promise.all([p1, pIdle, pHigh])
    check(
      'queue: one heavy job at a time; interaction preempts queued prewarm',
      maxActive === 1 && order[0] === 'decode' && order[1] === 'segment' && order[2] === 'prewarm',
      `maxActive=${maxActive} order=${order.join('→')}`,
    )
    const rejected = enqueueHeavy('boom', async () => { throw new Error('boom') })
    let threw = false
    await rejected.catch(() => { threw = true })
    const after = await enqueueHeavy('after', async () => 'ran')
    check('queue: a rejected job does not deadlock the scheduler', threw && after === 'ran', `threw=${threw} after=${after}`)
    const blocker = enqueueHeavy('blocker', () => new Promise((r) => setTimeout(r, 40)))
    const oldJob = enqueueHeavy('old', async () => 'old-ran', { revision: 1 })
    const staleJob = enqueueHeavy('stale', async () => 'stale-ran', { isCurrent: () => false })
    cancelHeavyBefore(2)
    const [oldRes, staleRes] = await Promise.all([oldJob, staleJob])
    await blocker
    check(
      'queue: cancelHeavyBefore + isCurrent stop queued jobs before they run',
      oldRes === STALE && staleRes === STALE && getHeavyQueueState().queuedCount === 0,
      `old=${String(oldRes?.stale)} stale=${String(staleRes?.stale)}`,
    )
  }

  /* ─── Phase S: memory-contract static scans ─────────────────────────── */
  {
    const jsFiles = ['app.js', 'asset-store.js', 'image-io.js', 'capability.js', 'policy.js',
      'sam-client.js', 'sam-worker.js', 'sam-engine.js', 'sam-core.js', 'edge-refine.js',
      'export-hd.js', 'detect-engine.js', 'detect-worker.js', 'embed-store.js', 'text-core.js',
      'text-ui.js', 'image-raw.js', 'heavy-job-queue.js', 'decode-worker.js', 'decode-client.js',
      'decode-core.js', 'proxy-plan.js', 'cv-refine-client.js', 'cv-refine-worker.js',
      'raw-develop-client.js', 'raw-develop-worker.js']
    const sources = Object.fromEntries(jsFiles.map((f) => [f, readFileSync(path.join(ROOT, 'js', f), 'utf8')]))
    const all = Object.values(sources).join('\n') + readFileSync(path.join(ROOT, 'sw.js'), 'utf8')
      + readFileSync(path.join(ROOT, 'index.html'), 'utf8')
    check('static: no toDataURL in source', !/\btoDataURL\s*\(/.test(all), 'clean')
    check(
      'static: no Python runtime (Pyodide/PyScript/…) and no full OpenCV.js',
      !/pyodide|pyscript|micropython|brython|skulpt|transcrypt/i.test(all)
        && !/opencv[._-]?js|\bcv\.imread\b/i.test(all),
      'clean',
    )
    check(
      'static: no startup model warm — warm starts only after the proxy is shown',
      !/startupWarm/.test(sources['app.js'])
        && /ensureWarm\(\)/.test(sources['app.js'])
        && !/^bootProbe\.then\(\(\) => warmUp/m.test(sources['app.js']),
      'app.js warms post-import only',
    )
    check(
      'static: SlimSAM is the only segmentation lane; SAM3 has no warm or upgrade path',
      !/sam3|flagship|maybeStartFlagship|Sam3TrackerModel/i.test(sources['sam-engine.js'])
        && /const activeLaneKey = \(\) => 'draft'/.test(sources['sam-engine.js']),
      'single SlimSAM lane',
    )
    check(
      'static: OPFS embedding persistence explicitly gated on embedPersist',
      /embedPersist === true/.test(sources['sam-engine.js']),
      'gated',
    )
    check(
      'static: SAM device pick is budget-gated — the baseline never probes WebGPU',
      /state\.budget\.samWebGPU !== true/.test(sources['sam-engine.js'])
        && /samWebGPU: false/.test(sources['policy.js'])
        && /next\.samWebGPU = false/.test(sources['policy.js']),
      'pickDevice honors samWebGPU; lite preset + pressure ratchet keep it off',
    )
    const assetGetImageData = (sources['asset-store.js'].match(/getImageData\(/g) || []).length
    check(
      'static: asset-store reads back only the 16×16 hash canvas, never a full frame',
      assetGetImageData === 1 && /getImageData\(0, 0, 16, 16\)/.test(sources['asset-store.js']),
      `getImageData sites=${assetGetImageData}`,
    )
    check(
      'static: wasm artifacts exist (cv-refine.js + cv-refine.wasm)',
      existsSync(path.join(ROOT, 'public/wasm/cv-refine.js')) && existsSync(path.join(ROOT, 'public/wasm/cv-refine.wasm')),
      'built',
    )
    check(
      'static: raw-develop wasm artifacts exist (raw-develop.js + raw-develop.wasm)',
      existsSync(path.join(ROOT, 'public/wasm/raw-develop.js')) && existsSync(path.join(ROOT, 'public/wasm/raw-develop.wasm')),
      'built',
    )
    check(
      'static: LibRaw develop is the preview-less fallback — lazy, disposed after use, off at pressure ≥ 2',
      /developRaw\(source, \{ budget: BUDGET \}\)/.test(sources['app.js'])
        && /worker\.terminate\(\)/.test(sources['raw-develop-client.js'])
        && /budget\.rawDevelop === false \|\| \(budget\.pressureLevel \|\| 0\) >= 2/.test(sources['raw-develop-client.js'])
        && /next\.rawDevelop = false/.test(sources['policy.js']),
      'wired only in the no-preview branch; terminates worker; pressure-gated',
    )
    check(
      'static: develop returns a JPEG Blob through the normal decode path (no full-res RGBA crosses threads)',
      /new Blob\(\[result\.jpeg\], \{ type: 'image\/jpeg' \}\)/.test(sources['raw-develop-client.js'])
        && !/getImageData|Uint8ClampedArray|RGBA/.test(sources['raw-develop-worker.js'])
        && /half_size/.test(readFileSync(path.join(ROOT, 'cpp/raw_develop.cpp'), 'utf8')),
      'jpeg blob only; half-size demosaic bounds the peak',
    )
    const sw = readFileSync(path.join(ROOT, 'sw.js'), 'utf8')
    check(
      'static: service worker caches only on demand and versions obsolete cache cleanup',
      /const CACHE_NAME = 'seglab-models-v3'/.test(sw)
        && /cache\.match\(request\)/.test(sw)
        && /cache\.put\(request, response\.clone\(\)\)/.test(sw)
        && !/cache\.addAll|event\.waitUntil\([^)]*fetch/i.test(sw),
      'cache-first after request; no install-time model/Wasm/detector prefetch',
    )
  }

  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
  })

  // Cold embed store + session for the persistence phases. A stale session
  // would otherwise auto-restore into every later page load.
  const pageZ = await context.newPage()
  await pageZ.goto(`http://127.0.0.1:${port}/`)
  await pageZ.evaluate(async () => {
    const root = await navigator.storage.getDirectory()
    await root.removeEntry('seglab-embeds', { recursive: true }).catch(() => {})
    await root.removeEntry('seglab-session', { recursive: true }).catch(() => {})
  })
  await pageZ.close()

  // Phosmith live-update contract (no model, no image).
  const pageHost = await context.newPage()
  await pageHost.goto(`http://127.0.0.1:${port}/`)
  await pageHost.waitForFunction(() => window.__seglabReady === true)
  const hostUpdate = await pageHost.evaluate(async (hostUltra) => {
    window.__PHOSMITH_DEVICE_RESOURCES__ = hostUltra
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', { detail: hostUltra }))
    const high = await window.__seglab.resourceBudget()
    window.__PHOSMITH_DEVICE_RESOURCES__ = { memoryBudgetGB: 4, mode: 'conservative' }
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', {
      detail: window.__PHOSMITH_DEVICE_RESOURCES__,
    }))
    const low = await window.__seglab.resourceBudget()
    return { high, low }
  }, HOST_ULTRA)
  check(
    'Phosmith event: resource budget upgrades and safely downgrades live without enabling SAM3',
    hostUpdate.high.profile === 'ultra' && hostUpdate.high.exportMaxMP === 64 && hostUpdate.high.flagship === false
      && hostUpdate.low.profile === 'lite' && hostUpdate.low.exportMaxMP === 8
      && hostUpdate.low.flagship === false,
    JSON.stringify({ high: hostUpdate.high.profile, low: hostUpdate.low.profile }),
  )
  await pageHost.close()

  /* ─── Phase A: lite (memory-locked) — the bounded-memory contract ───── */
  const page = await newAppPage(context, '?flagship=0', null)
  const bootState = await page.evaluate(() => window.__seglab.state())
  const bootBudget = await page.evaluate(() => window.__seglab.resourceBudget())
  check(
    'lite: no model loads before an image; locked lite budget resolved',
    bootState.ready === false && bootState.mode === null
      && bootBudget.profile === 'lite' && bootBudget.memoryLocked === true && bootBudget.proxyMax === 1024,
    JSON.stringify({ ready: bootState.ready, profile: bootBudget.profile, proxyMax: bootBudget.proxyMax }),
  )
  const swSmoke = await page.evaluate(async () => {
    const registration = await navigator.serviceWorker?.getRegistration()
    await navigator.serviceWorker?.ready
    return { registered: !!registration, active: !!registration?.active }
  })
  check(
    'service worker: the real app boots headless and registers the on-demand cache worker',
    swSmoke.registered && swSmoke.active,
    JSON.stringify(swSmoke),
  )
  await page.evaluate(() => window.__seglab.loadDemo(1400))
  const liteFrame = await page.evaluate(() => window.__seglab.imageTransform())
  check(
    'lite: 1400 px source becomes a ≤1024 px interaction proxy',
    liteFrame && liteFrame.proxyActive === true && Math.max(liteFrame.proxyW, liteFrame.proxyH) === 1024,
    JSON.stringify({ proxy: `${liteFrame?.proxyW}x${liteFrame?.proxyH}` }),
  )
  const disp = await page.evaluate(() => ({
    photo: document.getElementById('photo').width,
    view: document.getElementById('view').width,
    overlay: document.getElementById('overlay').width,
    bound: Math.round(Math.max(window.innerWidth, window.innerHeight) * Math.min(window.devicePixelRatio || 1, 3) * 1.5),
  }))
  check(
    'display: crisp preview out-resolves the proxy, stays within the viewport-anchored bound; overlay tracks the model buffer',
    disp.photo > disp.view && disp.overlay === disp.view
      && disp.photo <= Math.max(disp.bound, disp.view) && disp.photo <= 2048,
    JSON.stringify(disp),
  )
  const idleUi = await page.evaluate(() => ({
    prepHidden: document.getElementById('prep')?.hidden,
    stageVisible: document.getElementById('stage')?.classList.contains('visible'),
    pickEnabled: !document.getElementById('pick')?.disabled,
  }))
  check('lite: editor is usable while the model warms in the background',
    idleUi.prepHidden && idleUi.stageVisible && idleUi.pickEnabled, JSON.stringify(idleUi))
  const textModeUi = await page.evaluate(() => {
    document.getElementById('mode-text')?.click()
    const tolerance = document.getElementById('tolerance-wrap')
    const result = { hidden: tolerance?.hidden, display: getComputedStyle(tolerance).display }
    document.getElementById('mode-click')?.click()
    return result
  })
  check(
    'text mode: colour tolerance is hidden and cannot be mistaken for text confidence',
    textModeUi.hidden === true && textModeUi.display === 'none',
    JSON.stringify(textModeUi),
  )
  log('phase A (lite) — eager encode warms model + embedding at import…')
  const eager = await page.evaluate(() => window.__seglab.eagerEncode())
  const engEager = await page.evaluate(() => window.__seglab.engineState())
  const chipDevice = await page.evaluate(() => document.getElementById('chip-device')?.textContent || '')
  check(
    'lite: bounded eager encode lands before the first selection (one embedding, wasm lane)',
    eager && eager.stale !== true && engEager.cachedImages === 1 && engEager.device === 'wasm',
    JSON.stringify({ eager, cached: engEager?.cachedImages, device: engEager?.device }),
  )
  check(
    'lite: SAM runs on wasm — the baseline never probes WebGPU (status pill agrees)',
    engEager.device === 'wasm' && /device: wasm/.test(chipDevice),
    `chip="${chipDevice}"`,
  )

  const geo = await page.evaluate(() => window.__seglab.demoGeometry())
  const p = geo.proxyScale
  const sA = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.disc.x * p, y: geo.disc.y * p })
  checkDisc('lite draft', sA, geo.disc.x * p, geo.disc.y * p, DISC_FRAC)
  check(
    'first click is decode-only (the eager embedding served it)',
    sA.lastRun && sA.lastRun.encoded === false,
    `encoded=${sA.lastRun?.encoded}, decode ${sA.lastRun?.decodeMs}ms`,
  )
  const statsA = await page.evaluate(() => window.__seglab.maskStats())
  check('hygiene: single component, no crumbs', statsA && statsA.components === 1, `components=${statsA?.components}`)
  check('edge refinement: soft boundary band present', statsA && statsA.softPixels > 100, `softPixels=${statsA?.softPixels}`)

  const sA2 = await page.evaluate(() => window.__seglab.clickAt(50, 50, true))
  check(
    'repeat click skips the encoder (cache hit)',
    sA2.lastRun && sA2.lastRun.encoded === false,
    `encoded=${sA2.lastRun?.encoded}, decode ${sA2.lastRun?.decodeMs}ms`,
  )

  // Click-union: separate objects accumulate; exclude carves one out; Z peels.
  await page.evaluate(() => window.__seglab.reset())
  const sqc = { x: (geo.square.x + geo.square.w / 2) * p, y: (geo.square.y + geo.square.h / 2) * p }
  const u1 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.disc.x * p, y: geo.disc.y * p })
  const u2 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), sqc)
  const statsU2 = await page.evaluate(() => window.__seglab.maskStats())
  const covU1 = u1.maskSummary?.coverage || 0
  check(
    'click-union: clicking a second object ADDS it (commit + two components), never replaces',
    u2.baseOps === 1 && u2.clicks === 1 && statsU2?.components === 2
      && (u2.maskSummary?.coverage || 0) > covU1 * 1.3,
    `baseOps=${u2.baseOps} components=${statsU2?.components} coverage ${(covU1 * 100).toFixed(2)}%→${((u2.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const u3 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y, true), { x: geo.disc.x * p, y: geo.disc.y * p })
  const statsU3 = await page.evaluate(() => window.__seglab.maskStats())
  check(
    'click-union: exclude on a committed object carves it out; the other object survives',
    u3.baseOps === 2 && statsU3?.components === 1
      && (u3.maskSummary?.coverage || 0) < (u2.maskSummary?.coverage || 0)
      && (u3.maskSummary?.coverage || 0) > 0,
    `baseOps=${u3.baseOps} components=${statsU3?.components} coverage ${((u3.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const z1 = await page.evaluate(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z' }))
    return window.__seglab.state()
  })
  const z2 = await page.evaluate(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z' }))
    return window.__seglab.state()
  })
  check(
    'click-union: Z discards the live object first, then peels committed ops in order',
    // z1: live square gone; base nets ~empty (add(disc)∖sub(disc) may leave
    // a sub-pixel residue where the two decodes disagree). z2: disc returns.
    z1.clicks === 0 && (z1.maskSummary?.coverage || 0) < covU1 * 0.15 && z1.baseOps === 2
      && z2.baseOps === 1 && Math.abs((z2.maskSummary?.coverage || 0) - covU1) < covU1 * 0.2,
    JSON.stringify({ z1: { clicks: z1.clicks, baseOps: z1.baseOps, coverage: z1.maskSummary?.coverage || 0 }, z2: { baseOps: z2.baseOps, coverage: z2.maskSummary?.coverage } }),
  )

  // Minute object.
  await page.evaluate(() => window.__seglab.reset())
  const sDot = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.dot.x * p, y: geo.dot.y * p })
  const dotFrac = (Math.PI * DOT.r * DOT.r) / FRAME
  const bDot = sDot.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    'minute object (r=9px logical) is selectable',
    sDot.maskSummary && bDot[0] <= geo.dot.x * p && geo.dot.x * p <= bDot[2] && sDot.maskSummary.coverage < dotFrac * 40,
    `coverage ${((sDot.maskSummary?.coverage || 0) * 100).toFixed(2)}% (dot ${(dotFrac * 100).toFixed(3)}%)`,
  )

  // Lasso clamp.
  await page.evaluate(() => window.__seglab.reset())
  const lassoR = SQUARE.half * 1.5 * (geo.originalW / 900) * p
  const sqx = geo.square.x + geo.square.w / 2
  const sqy = geo.square.y + geo.square.h / 2
  const sLasso = await page.evaluate(
    ({ x, y, r }) => window.__seglab.lassoCircle(x, y, r),
    { x: sqx * p, y: sqy * p, r: lassoR },
  )
  const bL = sLasso.maskSummary?.bbox || [0, 0, -1, -1]
  const clampR = lassoR + 30
  const inClamp = bL[0] >= sqx * p - clampR && bL[2] <= sqx * p + clampR
    && bL[1] >= sqy * p - clampR && bL[3] <= sqy * p + clampR
  check(
    'lasso snaps to the square and stays clamped',
    sLasso.maskSummary && inClamp && bL[0] <= sqx * p && sqx * p <= bL[2],
    `bbox [${bL.map((v) => Math.round(v))}] within ±${Math.round(clampR)} of (${(sqx * p).toFixed(0)},${(sqy * p).toFixed(0)})`,
  )

  // Manual/predictive modes (proxy coords).
  await page.evaluate(() => window.__seglab.reset())
  const region = await page.evaluate(
    ({ x, y, r }) => window.__seglab.manualRegionCircle(x, y, r),
    { x: geo.disc.x * p, y: geo.disc.y * p, r: geo.disc.r * 0.7 * p },
  )
  const regionArea = Math.PI * (DISC.r * 0.7) ** 2 / FRAME
  check(
    'manual region: drawn area is selected without object snapping',
    region.manual === 'region' && region.components === 1
      && region.maskSummary.coverage > regionArea * 0.85 && region.maskSummary.coverage < regionArea * 1.15,
    `coverage ${((region.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs drawn ${(regionArea * 100).toFixed(1)}%`,
  )
  await page.evaluate(() => window.__seglab.reset())
  const colour = await page.evaluate(({ x, y }) => window.__seglab.colorAt(x, y, 24), { x: geo.disc.x * p, y: geo.disc.y * p })
  check(
    'color range: matching pixels select without object inference',
    colour.manual === 'color' && colour.components === 1
      && colour.maskSummary.coverage > DISC_FRAC * 0.8 && colour.maskSummary.coverage < DISC_FRAC * 1.2,
    `coverage ${((colour.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(DISC_FRAC * 100).toFixed(1)}%`,
  )
  const manualExport = await page.evaluate(() => window.__seglab.exportCutout())
  check(
    'manual masks: export preserves the drawn boundary without AI refinement',
    manualExport && manualExport.decoded === false
      && Math.abs(manualExport.coverage - colour.maskSummary.coverage) < 0.01,
    `decoded=${manualExport?.decoded}, coverage ${(manualExport?.coverage * 100 || 0).toFixed(1)}%`,
  )

  // Include/exclude accumulation.
  await page.evaluate(() => window.__seglab.reset())
  const addA = await page.evaluate(() => window.__seglab.manualRect(50, 50, 150, 150))
  const addB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150))
  const areaA = addA.maskSummary?.coverage || 0
  check(
    'include: a second region unions into the mask (two components)',
    addB.components === 2 && Math.abs((addB.maskSummary?.coverage || 0) - areaA * 2) < areaA * 0.1,
    `A ${(areaA * 100).toFixed(2)}% → A∪B ${((addB.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const subB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150, true))
  check(
    'exclude: re-applying the region as negative carves it back out',
    subB.components === 1 && Math.abs((subB.maskSummary?.coverage || 0) - areaA) < areaA * 0.1,
    `A∪B∖B ${((subB.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )

  // One-embedding contract across imports (same page, second document).
  await page.evaluate(() => window.__seglab.loadDemo(1200))
  const geoB = await page.evaluate(() => window.__seglab.demoGeometry())
  await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoB.disc.x * geoB.proxyScale, y: geoB.disc.y * geoB.proxyScale })
  const engAfter = await page.evaluate(() => window.__seglab.engineState())
  check(
    'embedding: exactly one resident embedding after a second import + click',
    engAfter && engAfter.cachedImages === 1 && engAfter.profile === 'lite',
    JSON.stringify(engAfter),
  )
  const opfsFiles = await page.evaluate(async () => {
    try {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('seglab-embeds')
      const names = []
      for await (const [name] of dir.entries()) names.push(name)
      return names
    } catch { return [] }
  })
  check('embedding: nothing persisted to OPFS in the lite policy', opfsFiles.length === 0, `files=${JSON.stringify(opfsFiles)}`)

  // Queue serialization + decode path over the real Blob upload route.
  await page.evaluate((side) => window.__seglab.loadDemoBlob(side), 3000)
  const geoQ = await page.evaluate(() => window.__seglab.demoGeometry())
  await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoQ.disc.x * geoQ.proxyScale, y: geoQ.disc.y * geoQ.proxyScale })
  const qlog = await page.evaluate(() => window.__seglab.queueLog())
  const qframe = await page.evaluate(() => window.__seglab.imageTransform())
  check(
    'queue: blob import runs decode-proxy as a heavy job; proxy stays ≤1024',
    qlog.some((e) => e.label === 'decode-proxy' && e.outcome === 'done')
      && qlog.some((e) => e.label === 'model-warm')
      && qframe && Math.max(qframe.proxyW, qframe.proxyH) <= 1024,
    `labels=${[...new Set(qlog.map((e) => e.label))].join(',')} proxy=${qframe?.proxyW}x${qframe?.proxyH}`,
  )
  // Rapid double import: the first must never commit.
  const raced = await page.evaluate(async () => {
    const a = window.__seglab.loadDemoBlob(2600)
    await new Promise((r) => setTimeout(r, 120)) // A has begun; B supersedes it
    const b = window.__seglab.loadDemoBlob(2000)
    await Promise.all([a, b])
    return window.__seglab.imageTransform()
  })
  check(
    'queue: a second import invalidates the prior queued import',
    raced && raced.originalW === 2000,
    `final original=${raced?.originalW}`,
  )
  await page.close()

  // Unsafe URL flags in the real app.
  const pageFlags = await newAppPage(context, '?flagship=1&profile=ultra&proxy=max&working=1', 4000)
  const flagBudget = await pageFlags.evaluate(() => window.__seglab.resourceBudget())
  const flagFrame = await pageFlags.evaluate(() => window.__seglab.imageTransform())
  check(
    'safety (live): unsafe flags are refused — lite, no flagship, ≤1024 proxy',
    flagBudget.profile === 'lite' && flagBudget.flagship === false && flagBudget.proxyMax === 1024
      && flagFrame && Math.max(flagFrame.proxyW, flagFrame.proxyH) <= 1024,
    JSON.stringify({ profile: flagBudget.profile, flagship: flagBudget.flagship, proxy: `${flagFrame?.proxyW}x${flagFrame?.proxyH}` }),
  )
  await pageFlags.close()

  /* ── Revision/cancel: overlapping prompts, one commit ── */
  const pageC = await newAppPage(context, '?flagship=0', 920)
  log('phase A2 (revision/cancel) — second click lands while the first is in flight…')
  const geoC = await pageC.evaluate(() => window.__seglab.demoGeometry())
  const m0 = await pageC.evaluate(async ({ disc, sq, scale }) => {
    const a = window.__seglab.clickAt(disc.x * scale, disc.y * scale)
    await new Promise((r) => setTimeout(r, 350))
    const b = window.__seglab.clickAt(sq.x * scale, sq.y * scale)
    await Promise.all([a, b])
    return { log: window.__seglab.commitLog.slice(), revision: window.__seglab.revision() }
  }, {
    disc: { x: geoC.disc.x, y: geoC.disc.y },
    sq: { x: geoC.square.x + geoC.square.w / 2, y: geoC.square.y + geoC.square.h / 2 },
    scale: geoC.proxyScale,
  })
  const committed = m0.log.filter((e) => e.outcome === 'committed')
  check(
    'revision/cancel: only the newest prompt commits',
    committed.length === 1 && committed[0].revision === m0.revision
      && m0.log.some((e) => e.revision < m0.revision && (e.outcome === 'stale' || e.outcome === 'superseded')),
    `log ${JSON.stringify(m0.log)} (current revision ${m0.revision})`,
  )
  await pageC.close()

  /* ── Lite export caps: >8 MP source exports reduced, never native ── */
  const pageX = await newAppPage(context, '?flagship=0', 4200)
  const geoX = await pageX.evaluate(() => window.__seglab.demoGeometry())
  await pageX.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoX.disc.x * geoX.proxyScale, y: geoX.disc.y * geoX.proxyScale })
  const exLite = await pageX.evaluate(() => window.__seglab.exportCutout())
  check(
    'lite export: bounded to ≤4096 px / ≤8 MP, no crop re-decode, coverage sane',
    exLite && exLite.w <= 4096 && exLite.h <= 4096 && (exLite.w * exLite.h) <= 8.05e6
      && exLite.w < geoX.originalW && exLite.decoded === false && exLite.coverage > 0.01,
    `export ${exLite?.w}×${exLite?.h} (${((exLite?.w * exLite?.h || 0) / 1e6).toFixed(1)} MP), decoded=${exLite?.decoded}`,
  )
  await pageX.close()

  /* ── Wasm cv-refine gates (through the real worker + transferables) ── */
  const pageV = await newAppPage(context, '?flagship=0', 900)
  log('phase A5 (wasm cv-refine) — hole fill, min-area, seeded cleanup, bounds…')
  const mk = (w, h, fn) => Array.from({ length: w * h }, (_, i) => (fn(i % w, (i / w) | 0) ? 255 : 0))
  const holeIn = mk(64, 64, (x, y) => x > 10 && x < 54 && y > 10 && y < 54 && !(x >= 30 && x < 35 && y >= 30 && y < 35))
  const holeOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: holeIn, width: 64, height: 64, seeds: [[20, 20]], options: { minArea: 0 },
  })
  check(
    'wasm: hole fill closes an interior pinhole, keeps the outside empty',
    holeOut.available && holeOut.alpha && holeOut.alpha[32 * 64 + 32] === 255 && holeOut.alpha[0] === 0,
    `available=${holeOut.available} center=${holeOut.alpha?.[32 * 64 + 32]}`,
  )
  const crumbs = mk(96, 96, (x, y) => (x > 10 && x < 60 && y > 10 && y < 60) || (x >= 80 && x < 83 && y >= 80 && y < 83))
  const crumbOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: crumbs, width: 96, height: 96, seeds: [[30, 30]], options: { minArea: 20 },
  })
  check(
    'wasm: components below min-area are removed, the seeded object stays',
    crumbOut.alpha && crumbOut.alpha[81 * 96 + 81] === 0 && crumbOut.alpha[30 * 96 + 30] === 255,
    `crumb=${crumbOut.alpha?.[81 * 96 + 81]} seed=${crumbOut.alpha?.[30 * 96 + 30]}`,
  )
  const twoComps = mk(96, 96, (x, y) => (x > 5 && x < 40 && y > 5 && y < 40) || (x > 55 && x < 90 && y > 55 && y < 90))
  const seededOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: twoComps, width: 96, height: 96, seeds: [[20, 20]], options: { minArea: 0 },
  })
  check(
    'wasm: seeded cleanup retains the seeded component, removes the other',
    seededOut.alpha && seededOut.alpha[20 * 96 + 20] === 255 && seededOut.alpha[70 * 96 + 70] === 0,
    `seed=${seededOut.alpha?.[20 * 96 + 20]} other=${seededOut.alpha?.[70 * 96 + 70]}`,
  )
  const tooBig = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: [255], width: 2048, height: 1, seeds: [], options: {},
  })
  const badBuf = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: [255, 255], width: 5, height: 5, seeds: [], options: {},
  })
  const stillWorks = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: mk(8, 8, (x, y) => x > 1 && y > 1), width: 8, height: 8, seeds: [], options: { minArea: 0 },
  })
  check(
    'wasm: >1024 px and mismatched buffers are rejected without leaking worker state',
    tooBig.alpha === null && badBuf.alpha === null && Array.isArray(stillWorks.alpha),
    `tooBig=${tooBig.alpha} badBuf=${badBuf.alpha} recovered=${Array.isArray(stillWorks.alpha)}`,
  )
  const detached = await pageV.evaluate(async () => {
    const { refineAlpha } = await import('./js/cv-refine-client.js')
    const alpha = new Uint8Array(16 * 16).fill(255)
    const out = await refineAlpha({ alpha, width: 16, height: 16, seeds: [], options: { minArea: 0 }, budget: { cvRefine: true } })
    return { detached: alpha.buffer.byteLength === 0, got: !!out }
  })
  check(
    'wasm: request buffers TRANSFER (input detached), result returns a new buffer',
    detached.detached === true && detached.got === true,
    JSON.stringify(detached),
  )
  await pageV.close()

  /* ── LibRaw develop fallback (fixture-gated): real sensor → JPEG on-device ── */
  if (RAW_FIXTURE) {
    const pageR = await newAppPage(context, '?flagship=0', null)
    log('phase A5b (raw develop) — LibRaw demosaic + libjpeg encode in a disposed worker…')
    const dev = await pageR.evaluate(() => window.__seglab.developRawUrl('/__raw_fixture'))
    check(
      'raw develop: preview-less fallback demosaics the sensor to a decodable JPEG on-device',
      dev.ok && dev.w > 0 && dev.h > 0 && dev.jpegBytes > 0
        && dev.decodedW === dev.w && dev.decodedH === dev.h,
      JSON.stringify(dev),
    )
    await pageR.close()
  }

  /* ── Text-select plumbing + brush (proxy coords) ── */
  const pageE = await newAppPage(context, '?flagship=0', 900)
  log('phase A6 (text plumbing + brush) — box → mask → union…')
  const geoE = await pageE.evaluate(() => window.__seglab.demoGeometry())
  const pE = geoE.proxyScale
  const discBox = [120 * pE, 230 * pE, 340 * pE, 450 * pE]
  const squareBox = [515 * pE, 145 * pE, 715 * pE, 345 * pE]
  const one = await pageE.evaluate((b) => window.__seglab.selectBoxes([b]), discBox)
  checkDisc('text', one, geoE.disc.x * pE, geoE.disc.y * pE, DISC_FRAC)
  await pageE.evaluate(() => window.__seglab.reset())
  const both = await pageE.evaluate(([a, b]) => window.__seglab.selectBoxes([a, b]), [discBox, squareBox])
  check('text select: "all" unions instances → 2 components', both && both.components === 2, `components=${both?.components}`)
  await pageE.evaluate(() => window.__seglab.reset())
  const brushAdd = await pageE.evaluate((points) => window.__seglab.brushStroke(points), [[333, 256], [418, 256], [478, 294]])
  const brushErase = await pageE.evaluate((points) => window.__seglab.brushStroke(points, true), [[333, 256], [371, 256]])
  check(
    'brush: canvas stroke commits a mask and erases incrementally',
    brushAdd?.manual === 'brush' && brushAdd.maskSummary?.coverage > 0
      && brushErase?.manual === 'brush' && brushErase.maskSummary?.coverage > 0
      && brushErase.maskSummary.coverage < brushAdd.maskSummary.coverage,
    `add=${((brushAdd?.maskSummary?.coverage || 0) * 100).toFixed(2)}% erase=${((brushErase?.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  log('⚠ text-detector phrase→boxes not gated headless (avoids a 151 MB model pull) — GDINO q4f16/WebGPU → OWLv2/WASM ladder confirmed in a real browser')
  await pageE.close()

  /* ── Weak device: everything still works on forced WASM ── */
  const pageW = await newAppPage(context, '?force=wasm', 760)
  log('phase A7 (weak-device) — full pipeline forced onto wasm…')
  const geoW = await pageW.evaluate(() => window.__seglab.demoGeometry())
  const pW = geoW.proxyScale
  const wClick = await pageW.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoW.disc.x * pW, y: geoW.disc.y * pW })
  await pageW.evaluate(() => window.__seglab.reset())
  const wText = await pageW.evaluate((b) => window.__seglab.selectBoxes([b]), [120 * pW, 230 * pW, 340 * pW, 450 * pW])
  const wExport = await pageW.evaluate(() => window.__seglab.exportCutout())
  check(
    'weak-device (wasm): click + text + export all complete',
    wClick.device === 'wasm' && !!wClick.maskSummary && !!wText.maskSummary
      && wExport && wExport.coverage > 0,
    `device=${wClick.device} export=${wExport?.w}×${wExport?.h}`,
  )
  const freed = await pageW.evaluate(() => window.__seglab.relievePressure(3))
  check(
    'pressure ladder: level 3 frees detector and the current embedding',
    Array.isArray(freed) && ['detector', 'embedding'].every((f) => freed.includes(f)),
    `freed=${JSON.stringify(freed)}`,
  )
  const releasedAll = await pageW.evaluate(() => window.__seglab.releaseMemory().then(() => window.__seglab.engineState()))
  check('debug release-memory action clears residents', releasedAll && releasedAll.cachedImages === 0, JSON.stringify(releasedAll))
  await pageW.close()

  /* ─── Phase H: trusted-host (Phosmith pro) — HD/escalation/working ────── */
  const pageD = await newAppPage(context, '?flagship=0', 2400, { host: HOST_PRO })
  log('phase H1 (HD export, trusted pro) — 2400px original, native-res cutout…')
  const geoD = await pageD.evaluate(() => window.__seglab.demoGeometry())
  await pageD.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geoD.disc.x * geoD.proxyScale, y: geoD.disc.y * geoD.proxyScale },
  )
  const ex = await pageD.evaluate(
    (probe) => window.__seglab.exportCutout(probe),
    { cx: geoD.disc.x, cy: geoD.disc.y, r: geoD.disc.r },
  )
  check(
    'HD export (pro): native dimensions, crop re-decoded, alpha correct',
    ex && ex.w === geoD.originalW && ex.h === geoD.originalH && ex.decoded === true
      && ex.centerOpaque && ex.outsideTransparent,
    `export ${ex?.w}×${ex?.h} decoded=${ex?.decoded}`,
  )
  check(
    'HD export (pro): boundary within 3px of the analytic disc',
    ex && ex.radialErr <= 3 && ex.softPixels > 100,
    `radialErr=${ex?.radialErr?.toFixed(1)}px, softPixels=${ex?.softPixels}`,
  )
  await pageD.close()

  const runDot = async (query) => {
    const pg = await newAppPage(context, query, 4000, { host: HOST_PRO })
    const g = await pg.evaluate(() => window.__seglab.demoGeometry())
    await pg.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: g.dot.x * g.proxyScale, y: g.dot.y * g.proxyScale })
    const esc = await pg.evaluate((probe) => window.__seglab.escalation(probe), { cx: g.dot.x, cy: g.dot.y, r: g.dot.r })
    await pg.close()
    return { esc, r: g.dot.r }
  }
  log('phase H2 (crop escalation, trusted pro) — native re-decode vs ?escalate=0…')
  const on = await runDot('?flagship=0')
  const off = await runDot('?flagship=0&escalate=0')
  check(
    'escalation (pro): fires for a small object, off under ?escalate=0',
    on.esc.fired === true && on.esc.decoded === true && on.esc.centerOpaque === true && off.esc.fired === false,
    `on{fired:${on.esc.fired},decoded:${on.esc.decoded}} off{fired:${off.esc.fired}}`,
  )
  check(
    'escalation (pro): native re-decode recovers the boundary',
    on.esc.radialErr != null && off.esc.radialErr != null
      && on.esc.radialErr <= 6 && on.esc.radialErr < off.esc.radialErr * 0.7,
    `escalate=1 ${on.esc.radialErr?.toFixed(1)}px vs control ${off.esc.radialErr?.toFixed(1)}px`,
  )

  log('phase H3 (working copy, trusted pro) — ?working=1, 5000px blob upload…')
  const pageWk = await newAppPage(context, '?flagship=0&working=1', null, { host: HOST_PRO })
  await pageWk.evaluate((side) => window.__seglab.loadDemoBlob(side), 5000)
  const wkFrame = await pageWk.evaluate(() => window.__seglab.imageTransform())
  check(
    'working copy: oversized upload keeps a ≤4096 bounded re-decode source',
    wkFrame && wkFrame.proxyActive === true && wkFrame.workingActive === true
      && Math.max(wkFrame.workingW || 0, wkFrame.workingH || 0) === 4096,
    JSON.stringify({ working: `${wkFrame?.workingW}x${wkFrame?.workingH}`, proxy: `${wkFrame?.proxyW}x${wkFrame?.proxyH}` }),
  )
  const geoWk = await pageWk.evaluate(() => window.__seglab.demoGeometry())
  await pageWk.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoWk.dot.x * geoWk.proxyScale, y: geoWk.dot.y * geoWk.proxyScale })
  const escWk = await pageWk.evaluate((probe) => window.__seglab.escalation(probe), { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r })
  check(
    'working copy: escalation decodes a bounded crop (rescaled prompts land)',
    escWk.fired === true && escWk.decoded === true && escWk.centerOpaque === true && escWk.radialErr <= 6,
    `fired=${escWk.fired} decoded=${escWk.decoded} radialErr=${escWk.radialErr?.toFixed(1)}px`,
  )
  const exWk = await pageWk.evaluate((probe) => window.__seglab.exportCutout(probe), { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r })
  check(
    'working copy: export bypasses it — native dimensions, fresh decode',
    exWk && exWk.w === geoWk.originalW && exWk.decoded === true && exWk.radialErr <= 3,
    `export ${exWk?.w}×${exWk?.h}, decoded=${exWk?.decoded}, radialErr=${exWk?.radialErr?.toFixed(1)}px`,
  )
  await pageWk.close()

  log('phase H4 (OPFS revisit, trusted pro) — persistence stays a trusted-only feature…')
  const pageR1 = await newAppPage(context, '?flagship=0', 1600, { host: HOST_PRO })
  const r1 = await pageR1.evaluate(() => window.__seglab.eagerEncode())
  await pageR1.close()
  check('revisit (pro): first visit encodes fresh (cold store)', r1?.encoded === true, `encoded=${r1?.encoded}`)
  const pageR2 = await newAppPage(context, '?flagship=0', 1600, { host: HOST_PRO })
  const r2 = await pageR2.evaluate(() => window.__seglab.eagerEncode())
  const geoR = await pageR2.evaluate(() => window.__seglab.demoGeometry())
  const sR = await pageR2.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoR.disc.x * geoR.proxyScale, y: geoR.disc.y * geoR.proxyScale })
  check(
    'revisit (pro): fresh session serves the import encode from OPFS and decodes',
    r2?.encoded === false && sR.lastRun?.encoded === false && !!sR.maskSummary,
    `eager encoded=${r2?.encoded}, first click encoded=${sR.lastRun?.encoded}`,
  )
  await pageR2.close()

  /* ─── Phase O: zero-cloud proof (lite) ──────────────────────────────── */
  const pageO = await newAppPage(context, '?flagship=0', 1900)
  const geoO = await pageO.evaluate(() => window.__seglab.demoGeometry())
  await pageO.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoO.disc.x * geoO.proxyScale, y: geoO.disc.y * geoO.proxyScale })
  let attempted = 0
  let succeeded = 0
  try {
    await context.setOffline(true)
    pageO.on('request', () => { attempted += 1 })
    pageO.on('requestfinished', () => { succeeded += 1 })
    log('phase O (offline) — network cut; fresh import + click + text + export…')
    // A NEW document offline forces a fresh encode with zero network.
    await pageO.evaluate(() => window.__seglab.loadDemo(1400))
    const gOff = await pageO.evaluate(() => window.__seglab.demoGeometry())
    const pOff = gOff.proxyScale
    const oClick = await pageO.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: gOff.disc.x * pOff, y: gOff.disc.y * pOff })
    await pageO.evaluate(() => window.__seglab.reset())
    const dbox = [
      (gOff.disc.x - gOff.disc.r * 1.2) * pOff,
      (gOff.disc.y - gOff.disc.r * 1.2) * pOff,
      (gOff.disc.x + gOff.disc.r * 1.2) * pOff,
      (gOff.disc.y + gOff.disc.r * 1.2) * pOff,
    ]
    const oText = await pageO.evaluate((b) => window.__seglab.selectBoxes([b]), dbox)
    const oEx = await pageO.evaluate(() => window.__seglab.exportCutout())
    check(
      'offline: fresh import + click + text + export all pass with the network cut',
      !!oClick?.maskSummary && oClick.lastRun?.encoded === true && !!oText?.maskSummary
        && oEx && oEx.coverage > 0,
      `click=${!!oClick.maskSummary} encoded=${oClick.lastRun?.encoded} export=${oEx?.w}×${oEx?.h}`,
    )
    check('offline: zero successful network fetches during inference', succeeded === 0, `${attempted} attempted, ${succeeded} succeeded`)
  } finally {
    await context.setOffline(false)
  }
  await pageO.close()

  /* ─── Phase V: vendored assets serve a cold start with no network.
         Phase O proves zero-cloud inference, but only once the profile has
         cached the weights. This is the stronger claim: fresh cache-less
         browser, both CDNs blocked, selection still works. Skipped when
         nothing is vendored. */
  if (existsSync(path.join(ROOT, 'models', 'manifest.json'))) {
    const browserV = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu'] })
    try {
      const ctxV = await browserV.newContext() // fresh: no profile, no Cache Storage
      const blocked = []
      const localHits = []
      await ctxV.route(/https:\/\/(huggingface\.co|cdn(-lfs[a-z0-9-]*)?\.(jsdelivr\.net|huggingface\.co))\//, (r) => {
        blocked.push(r.request().url())
        r.abort()
      })
      const pageV = await newAppPage(ctxV, '?flagship=0', 1400)
      pageV.on('request', (req) => {
        const u = req.url()
        if (u.includes('/models/') || u.includes('/lib/')) localHits.push(u)
      })
      const gV = await pageV.evaluate(() => window.__seglab.demoGeometry())
      const sV = await pageV.evaluate(
        ({ x, y }) => window.__seglab.clickAt(x, y),
        { x: gV.disc.x * gV.proxyScale, y: gV.disc.y * gV.proxyScale },
      )
      const exV = await pageV.evaluate(() => window.__seglab.exportCutout())
      check(
        'vendored: cold cache-less start selects + exports with both CDNs blocked',
        !!sV?.maskSummary && sV.lastRun?.lane === 'slimsam' && exV?.coverage > 0,
        `coverage ${((sV?.maskSummary?.coverage || 0) * 100).toFixed(1)}%, export ${exV?.w}×${exV?.h}, `
          + `${blocked.length} CDN requests aborted`,
      )
      check(
        'vendored: weights + runtime served from local models/ and lib/',
        localHits.some((u) => u.endsWith('.onnx')) && localHits.some((u) => u.includes('transformers.min.js')),
        `${localHits.length} local fetches incl. ${localHits.filter((u) => u.endsWith('.onnx')).length} onnx`,
      )
      await pageV.close()
    } finally {
      await browserV.close().catch(() => {})
    }
  } else {
    log('skip phase V — run `bun run models` to vendor assets for the offline gate')
  }

  /* ─── Phase P: work survives a power cut. The record must be durable while
         the page is still open — a real cut fires no unload, so anything saved
         only on pagehide would be lost. Reopen must bring the work back. */
  {
    // Drop any session an earlier phase persisted BEFORE pageP boots: it would
    // otherwise auto-restore, and saves are suppressed while a restore is in
    // flight (so the click below would never reach disk).
    const pageClr = await newAppPage(context, '?flagship=0', null)
    await pageClr.evaluate(() => window.__seglab.clearSession())
    await pageClr.close()

    const pageP = await newAppPage(context, '?flagship=0', null, { restore: true })
    await pageP.evaluate((side) => window.__seglab.loadDemoBlob(side), 1400)
    const gP = await pageP.evaluate(() => window.__seglab.demoGeometry())
    const before = await pageP.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: gP.disc.x * gP.proxyScale, y: gP.disc.y * gP.proxyScale },
    )
    // Poll for the MASK to land, not just the import-time record — the click's
    // save is debounced. Still no unload of any kind has fired.
    let durable = null
    for (let i = 0; i < 30 && !durable?.mask; i += 1) {
      durable = await pageP.evaluate(() => window.__seglab.sessionSaved())
      if (!durable?.mask) await pageP.waitForTimeout(500)
    }
    check(
      'persistence: session is durable while the page is still open (no unload fired)',
      durable?.mask === true,
      `saved mask=${durable?.mask} at ${durable?.w}×${durable?.h}`,
    )
    // Simulate the cut: abandon the page without letting it save anything more.
    await pageP.close()

    const pageQ = await newAppPage(context, '?flagship=0', null, { restore: true })
    await pageQ.waitForFunction(() => window.__seglab.state()?.hasImage === true, null, { timeout: 30_000 }).catch(() => {})
    const after = await pageQ.evaluate(() => ({ ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }))
    const cov = (n) => (n?.maskSummary?.coverage ?? n?.coverage ?? 0)
    check(
      'persistence: reopen restores the document + committed selection',
      after.hasImage === true && cov(after) > 0,
      `hasImage=${after.hasImage}, coverage ${(cov(after) * 100).toFixed(1)}% (was ${(cov(before) * 100).toFixed(1)}%)`,
    )
    await pageQ.evaluate(() => window.__seglab.clearSession())
    await pageQ.close()
  }

} catch (err) {
  console.error(`[verify] ✗ ${err?.message || err}`)
  console.error(err?.stack || '')
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

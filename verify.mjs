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
 * Phase B (default URL — flagship upgrade path):
 *   7. background SAM3 download → lane flips to 'sam3' (first run pulls
 *      ~300 MB into the persistent profile; cached after)
 *   8. click the disc on SAM3   → same quality bars pass on the flagship
 *   Phase B failure is a WARN, not a gate failure: headless Chromium's
 *   WebGPU (esp. f16) differs from real Chrome — flagship must be
 *   confirmed manually where headless can't.
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
import { normalizePhrase, nms, rankDetections, scaleBox } from './js/text-core.js'

const ROOT = path.resolve(import.meta.dir)
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)

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

  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
  })

  /* ─── Phase A: draft lane, deterministic ────────────────────────────── */
  const page = await newAppPage(context, '?flagship=0')
  log('phase A (draft lane) — first click downloads the model on a cold profile…')

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

  const stA = await page.evaluate(() => window.__seglab.state())
  log(`phase A done: engine=${stA.mode} device=${stA.device} lane=${stA.lane}`)
  await page.close()

  /* ─── Phase A2: M0 revision/cancel — overlapping prompts, one commit ──── */
  // Fresh page = cold encode, so the first click's run is in flight for
  // seconds; the second click must obsolete it (stale or superseded) and be
  // the only revision that commits.
  const pageC = await newAppPage(context, '?flagship=0')
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
  // The live OWLv2 detector is NOT exercised here: it can't build a session
  // under headless ORT, and its fallback ladder would download ~1 GB failing.
  // Confirm phrase→boxes manually in real Chrome (chip should show candidates).
  log('⚠ phase A5: OWLv2 phrase→boxes not gated headless (ORT op gap) — confirm in real Chrome')
  await pageE.close()

  /* ─── Phase B: flagship upgrade (WARN on failure, headless WebGPU ≠
         real Chrome) ──────────────────────────────────────────────────── */
  try {
    const pageB = await newAppPage(context, '')
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

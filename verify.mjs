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

const ROOT = path.resolve(import.meta.dir)
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)

const log = (msg) => console.log(`[verify] ${msg}`)
const results = []
const check = (label, ok, detail) => {
  results.push({ label, ok })
  console.log(`[verify] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
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

const newAppPage = async (context, query) => {
  const page = await context.newPage()
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[seglab]') || msg.type() === 'error') log(`browser: ${text.slice(0, 180)}`)
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
  page.setDefaultTimeout(TIMEOUT_MS)
  await page.goto(`http://127.0.0.1:${port}/${query}`)
  await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  await page.evaluate(() => window.__seglab.loadDemo())
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

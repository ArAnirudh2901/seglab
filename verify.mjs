#!/usr/bin/env bun
/**
 * Headless verification of SEGLAB end-to-end.
 *
 * Two tiers, deliberately separated so a green run always means something:
 *
 *   PURE GATES (always run, no browser, no network, no weights)
 *     The decode maths, the selection rules, and the memory policy — every
 *     decision that determines whether a mask lands in the right place, the
 *     right instance gets picked, and the 1 GB ceiling holds. These are hard
 *     failures.
 *
 *   BROWSER GATES (need Chromium AND an exported model)
 *     Drives the real app through the window.__seglab hooks: the model loads,
 *     an image analyzes, each of the four modes returns a coherent selection,
 *     and an absent phrase returns nothing.
 *
 * Why the browser tier asserts PLUMBING and not detection accuracy: the demo
 * scene is flat synthetic geometry, and YOLOE-26 is trained on photographs.
 * Asserting that it finds "a red circle" in a gradient with a disc on it
 * would be a test of the fixture, not the app. Detection accuracy is measured
 * where it means something — `bun bench.mjs`, on real photographs.
 *
 * Weights are NOT in the repo (YOLOE-26 is AGPL-3.0; see js/yoloe-engine.js
 * for the export command). Without models/yoloe26-s-seg.onnx the browser tier
 * reports SKIPPED — never passed.
 *
 * Usage: bun verify.mjs
 */

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { execSync } from 'node:child_process'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dir)
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')
const MODEL_PATH = path.join(ROOT, 'models', 'yoloe26-s-seg.onnx')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)

const log = (msg) => console.log(`[verify] ${msg}`)
const results = []
const check = (label, ok, detail) => {
  results.push({ label, ok })
  console.log(`[verify] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
}

/* ─── Tier 1: pure gates (always) ───────────────────────────────────────── */

log('pure gates — no browser, no weights')
for (const [file, what] of [
  ['test-yoloe-core.mjs', 'letterbox geometry, tensor layouts, mask assembly'],
  ['test-select-core.mjs', 'click/region/phrase rules, absent-phrase rejection'],
  ['test-post-pipeline.mjs', 'optimized hygiene + edge refinement vs naive reference'],
  ['test-governor.mjs', 'the 1 GB ceiling under 45 MP load'],
]) {
  try {
    execSync(`${process.execPath} ${path.join(ROOT, file)}`, { stdio: 'inherit' })
    check(`unit: ${file}`, true, what)
  } catch {
    check(`unit: ${file}`, false, 'see failures above')
  }
}

/* ─── Tier 2: browser gates ─────────────────────────────────────────────── */

let browserSkipped = null

if (!existsSync(MODEL_PATH)) {
  browserSkipped = 'models/yoloe26-s-seg.onnx not found'
  log(`skip — ${browserSkipped}`)
  log('skip — export it first: yolo export model=yoloe26-s-seg.pt format=onnx half=True imgsz=640 simplify=True opset=12')
}

let chromium = null
if (!browserSkipped) {
  try {
    ({ chromium } = await import('playwright'))
  } catch {
    browserSkipped = 'playwright not installed'
    log('skip — playwright not installed (`bun add -d playwright && npx playwright install chromium`)')
  }
}

let failed = false
let server = null
let context = null

if (!browserSkipped) {
  const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
    '.onnx': 'application/octet-stream',
  }
  server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost')
      const rel = url.pathname === '/' ? '/index.html' : url.pathname
      const file = path.join(ROOT, path.normalize(rel))
      if (!file.startsWith(ROOT) || !existsSync(file)) { res.writeHead(404).end('not found'); return }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
      res.end(await readFile(file))
    } catch (e) {
      res.writeHead(500).end(String(e?.message || e))
    }
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  log(`serving SEGLAB on http://127.0.0.1:${port}`)

  try {
    context = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    })
    const page = await context.newPage()
    page.setDefaultTimeout(TIMEOUT_MS)
    page.on('console', (m) => { if (m.text().startsWith('[seglab]')) log(`browser: ${m.text().slice(0, 160)}`) })
    page.on('pageerror', (e) => log(`pageerror: ${String(e).slice(0, 160)}`))
    await page.goto(`http://127.0.0.1:${port}/`)
    await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 60_000 })
    await page.evaluate(() => window.__seglab.loadDemo())

    // 1. The model loads and analyzes an image at all.
    const all = await page.evaluate(() => window.__seglab.selectAll())
    check(
      'model loads and analyzes an image',
      typeof all.detected === 'number',
      `detected=${all.detected} instances, device=${all.device}`,
    )
    if (all.detected === 0) {
      log('⚠ 0 instances on the synthetic demo scene — expected for a photo-trained model;')
      log('⚠ real detection accuracy is measured by `bun bench.mjs` on photographs')
    }

    // 2. Every mode returns a coherent result shape and never throws.
    const click = await page.evaluate(() => window.__seglab.clickAt(230, 340))
    check('click mode returns a coherent selection', Array.isArray(click.instances),
      `selected=${click.selected}, instances=${click.instances.length}`)

    const box = await page.evaluate(() => window.__seglab.boxAt(520, 150, 710, 340))
    check('box mode returns a coherent selection', Array.isArray(box.instances),
      `selected=${box.selected}`)

    const lasso = await page.evaluate(() => window.__seglab.lassoCircle(615, 245, 140))
    check('lasso mode returns a coherent selection', Array.isArray(lasso.instances),
      `selected=${lasso.selected}`)

    // 3. The control that matters everywhere: an absent concept selects
    //    NOTHING rather than the best-scoring piece of background.
    const absent = await page.evaluate(() => window.__seglab.textSearch('a giraffe'))
    check(
      'absent concept returns an empty selection',
      absent.instances.length === 0,
      `instances=${absent.instances.length} for "a giraffe"`,
    )

    // 4. Selection is cached — a second interaction must not re-analyze.
    await page.evaluate(() => window.__seglab.reset())
    await page.evaluate(() => window.__seglab.selectAll())
    const second = await page.evaluate(() => window.__seglab.selectAll())
    check(
      'second selection reuses the analysis (no re-run of the model)',
      second.lastRun && second.lastRun.analyzed === false,
      `analyzed=${second.lastRun?.analyzed}, select ${second.lastRun?.selectMs}ms`,
    )

    // 5. Mask hygiene and edge refinement actually ran.
    if (all.detected > 0) {
      await page.evaluate(() => window.__seglab.reset())
      await page.evaluate(() => window.__seglab.selectAll())
      const stats = await page.evaluate(() => window.__seglab.maskStats())
      check('edge refinement produced a soft boundary band',
        stats && stats.softPixels > 0, `softPixels=${stats?.softPixels}`)
    }

    await page.close()
  } catch (err) {
    const msg = String(err?.message || err)
    // An environment gap must never masquerade as a product failure.
    if (/Executable doesn't exist|browserType\.launch|Failed to launch|ENOENT/i.test(msg)) {
      browserSkipped = msg.split('\n')[0].slice(0, 160)
      log(`skip — browser unavailable: ${browserSkipped}`)
      log('skip — run `npx playwright install chromium` to enable the browser gates')
    } else {
      console.error(`[verify] ✗ ${msg}`)
      failed = true
    }
  } finally {
    await context?.close().catch(() => {})
    server?.close()
  }
}

if (failed || results.some((r) => !r.ok)) {
  console.error('\n[verify] ✗ SEGLAB self-test FAILED')
  process.exit(1)
}
if (browserSkipped) {
  console.log('\n[verify] ✓ pure gates passed — browser gates were SKIPPED, not verified')
  process.exit(0)
}
console.log('\n[verify] ✓ SEGLAB verified end-to-end in a real browser')

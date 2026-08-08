#!/usr/bin/env node
/**
 * Offline end-to-end verification of SEGLAB.
 *
 * `verify.mjs` is the real gate: it drives the app against the actual models
 * over the network. That makes it useless anywhere egress is blocked — a
 * sandbox, a locked-down CI runner — where it fails on the CDN import and
 * tells you nothing about the app.
 *
 * This runs the SAME application end to end (index.html -> app.js ->
 * sam-client -> dedicated worker -> sam-engine -> gpu-post), with only the
 * ONNX layer replaced. test/stub-transformers.js implements exactly the
 * transformers.js surface sam-engine uses, and its "decoder" is a real
 * flood-fill segmenter over the actual canvas pixels — so mask geometry,
 * coverage and component counts stay meaningful and are asserted against the
 * demo scene's known answers. It is swapped in by rewriting the one CDN
 * constant as sam-engine.js is served; nothing else is mocked.
 *
 * What this covers that unit tests cannot: worker transport, the embedding
 * cache, real pointer interaction, the post pipeline on a real WebGPU device,
 * the overlay, and the export path.
 * What it cannot cover: the accuracy of the actual SAM weights. That is
 * verify.mjs's job.
 *
 * Usage: node verify-offline.mjs
 *   SEGLAB_PLAYWRIGHT   module path if a bare 'playwright' import won't resolve
 *   SEGLAB_CHROMIUM     browser binary if it does not match the pinned build
 *   SEGLAB_SWIFTSHADER  software WebGPU, for boxes with no real GPU
 *   SEGLAB_NO_SANDBOX   pass --no-sandbox (containers)
 */
import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const APP = path.dirname(fileURLToPath(import.meta.url))
const STUB = path.join(APP, 'test')

let chromium
for (const spec of [process.env.SEGLAB_PLAYWRIGHT, 'playwright'].filter(Boolean)) {
  try { ({ chromium } = await import(spec)); break } catch { /* next */ }
}
if (!chromium) {
  console.log('[offline] skip — playwright not found (npm i -D playwright, or set SEGLAB_PLAYWRIGHT)')
  process.exit(0)
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript' }

const server = createServer(async (req, res) => {
  const p = req.url === '/' ? '/index.html' : req.url.split('?')[0]
  let body
  try {
    if (p === '/stub-transformers.js') body = await readFile(path.join(STUB, 'stub-transformers.js'))
    else {
      body = await readFile(path.join(APP, p))
      if (p === '/js/sam-engine.js') {
        body = Buffer.from(String(body).replace(
          /const TRANSFORMERS_CDN = '[^']*'/,
          "const TRANSFORMERS_CDN = '/stub-transformers.js'"))
      }
    }
  } catch { res.writeHead(404).end('nf'); return }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' })
  res.end(body)
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port

const results = []
const check = (label, ok, detail = '') => { results.push({ label, ok }); console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ' — ' + detail : ''}`) }

const browser = await chromium.launch({
  headless: true,
  ...(process.env.SEGLAB_CHROMIUM ? { executablePath: process.env.SEGLAB_CHROMIUM } : {}),
  args: [
    '--enable-unsafe-webgpu',
    ...(process.env.SEGLAB_SWIFTSHADER ? ['--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-features=Vulkan'] : []),
    ...(process.env.SEGLAB_NO_SANDBOX ? ['--no-sandbox'] : []),
  ],
})
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } })
const pageErrors = []
page.on('pageerror', (e) => pageErrors.push(String(e)))
const notFound = []
page.on('response', (r) => { if (r.status() === 404) notFound.push(r.url()) })
page.on('console', (m) => { const t = m.text(); if (m.type() === 'error' && !/Failed to load resource/.test(t)) pageErrors.push('console: ' + t) })
page.setDefaultTimeout(120000)
await page.goto(`http://127.0.0.1:${port}/`)
await page.waitForFunction(() => window.__seglabReady === true)

const DISC = { x: 230, y: 340, r: 105 }, SQUARE = { x: 615, y: 245, half: 95 }, DOT = { x: 700, y: 480, r: 9 }
const FRAME = 900 * 620

// --- boot state -----------------------------------------------------------
check('dropzone visible before import', await page.isVisible('#dropzone'))
check('stage hidden before import', !(await page.isVisible('#stage')))
await page.click('#demo')
await page.waitForFunction(() => document.getElementById('stage').classList.contains('visible'))
check('demo scene loads, stage shown', true)
check('canvas is the canonical frame', await page.evaluate(() => {
  const v = document.getElementById('view'); return v.width === 900 && v.height === 620 }))

// --- a REAL pointer click on the overlay, not a test hook -----------------
const box = await page.locator('#overlay').boundingBox()
const toView = (x, y) => ({ x: box.x + (x / 900) * box.width, y: box.y + (y / 620) * box.height })
const pt = toView(DISC.x, DISC.y)
await page.mouse.click(pt.x, pt.y)
await page.waitForFunction(() => !window.__seglab.state().lastRun === false, null, { timeout: 120000 })
await page.waitForFunction(() => { const s = window.__seglab.state(); return s.maskSummary !== null })
let s = await page.evaluate(() => window.__seglab.state())
const discArea = (Math.PI * DISC.r * DISC.r) / FRAME
let b = s.maskSummary?.bbox || [0,0,-1,-1]
check('real mouse click selects the disc',
  b[0] <= DISC.x && DISC.x <= b[2] && b[1] <= DISC.y && DISC.y <= b[3], `bbox [${b}]`)
check('disc mask is object-sized',
  s.maskSummary.coverage > discArea * 0.4 && s.maskSummary.coverage < discArea * 3,
  `coverage ${(s.maskSummary.coverage * 100).toFixed(2)}% vs disc ${(discArea * 100).toFixed(2)}%`)
check('engine ran in a dedicated worker', s.mode === 'worker', `mode=${s.mode}`)
check('coordinate round-trip through reshaped space is exact',
  Math.abs(b[0] - (DISC.x - DISC.r)) <= 2 && Math.abs(b[2] - (DISC.x + DISC.r)) <= 2, `bbox [${b}]`)

const stats = await page.evaluate(() => window.__seglab.maskStats())
check('hygiene: single component, no crumbs', stats.components === 1, `components=${stats.components}`)
check('edge refinement: soft boundary band present', stats.softPixels > 100, `softPixels=${stats.softPixels}`)
check('post pipeline ran on the GPU', s.lastRun.postBackend === 'gpu',
  `postBackend=${s.lastRun.postBackend}, device=${s.device}, post=${s.lastRun.postMs}ms`)
check('first click paid for the encoder', s.lastRun.encoded === true, `encoded=${s.lastRun.encoded}`)
check('UI buttons enabled after a selection',
  await page.evaluate(() => !document.getElementById('undo').disabled && !document.getElementById('cutout').disabled))

// --- embedding cache: second click must skip the encoder ------------------
s = await page.evaluate(() => window.__seglab.clickAt(60, 60, true))
check('repeat click hits the embedding cache', s.lastRun.encoded === false,
  `encoded=${s.lastRun.encoded}, decode ${s.lastRun.decodeMs}ms, post ${s.lastRun.postMs}ms ${s.lastRun.postBackend}`)
check('negative click did not destroy the disc', s.maskSummary.coverage > discArea * 0.4,
  `coverage ${(s.maskSummary.coverage * 100).toFixed(2)}%`)

// --- minute object --------------------------------------------------------
await page.evaluate(() => window.__seglab.reset())
s = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: DOT.x, y: DOT.y })
const dotArea = (Math.PI * DOT.r * DOT.r) / FRAME
b = s.maskSummary?.bbox || [0,0,-1,-1]
check('minute object (r=9px) is selectable',
  b[0] <= DOT.x && DOT.x <= b[2] && s.maskSummary.coverage < dotArea * 40,
  `coverage ${(s.maskSummary.coverage * 100).toFixed(3)}% (dot ${(dotArea * 100).toFixed(3)}%)`)

// --- lasso clamp ----------------------------------------------------------
await page.evaluate(() => window.__seglab.reset())
const lassoR = SQUARE.half * 1.5
s = await page.evaluate(({ x, y, r }) => window.__seglab.lassoCircle(x, y, r), { x: SQUARE.x, y: SQUARE.y, r: lassoR })
b = s.maskSummary?.bbox || [0,0,-1,-1]
const clampR = lassoR + 30
check('lasso snaps to the square and stays clamped',
  b[0] >= SQUARE.x - clampR && b[2] <= SQUARE.x + clampR && b[1] >= SQUARE.y - clampR && b[3] <= SQUARE.y + clampR
  && b[0] <= SQUARE.x && SQUARE.x <= b[2], `bbox [${b}]`)

// --- box mode via a real drag --------------------------------------------
await page.evaluate(() => window.__seglab.reset())
await page.click('#mode-box')
const p0 = toView(500, 130), p1 = toView(730, 360)
await page.mouse.move(p0.x, p0.y); await page.mouse.down()
await page.mouse.move((p0.x + p1.x) / 2, (p0.y + p1.y) / 2); await page.mouse.move(p1.x, p1.y)
await page.mouse.up()
await page.waitForFunction(() => window.__seglab.state().maskSummary !== null, null, { timeout: 120000 })
s = await page.evaluate(() => window.__seglab.state())
b = s.maskSummary.bbox
check('box drag selects the square', b[0] <= SQUARE.x && SQUARE.x <= b[2] && b[1] <= SQUARE.y && SQUARE.y <= b[3], `bbox [${b}]`)

// --- undo / reset / raw toggle -------------------------------------------
await page.click('#mode-click')
await page.evaluate(() => window.__seglab.reset())
await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: DISC.x, y: DISC.y })
await page.keyboard.press('e')
check('E toggles the raw decoder mask', (await page.textContent('#status')).includes('RAW'))
await page.keyboard.press('e')
await page.keyboard.press('r')
check('R clears the selection', await page.evaluate(() => window.__seglab.state().maskSummary === null || document.getElementById('undo').disabled))

// --- cutout export --------------------------------------------------------
await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: DISC.x, y: DISC.y })
const cut = await page.evaluate(async () => {
  const c = document.createElement('canvas')
  const view = document.getElementById('view')
  c.width = view.width; c.height = view.height
  const ctx = c.getContext('2d'); ctx.drawImage(view, 0, 0)
  return { w: c.width, h: c.height }
})
const dl = page.waitForEvent('download', { timeout: 30000 }).catch(() => null)
await page.click('#cutout')
const download = await dl
check('cutout PNG downloads', !!download, download ? `filename=${download.suggestedFilename()}` : 'no download event')

// --- chips reflect real state --------------------------------------------
const chips = await page.evaluate(() => ({
  mode: document.getElementById('chip-mode').textContent,
  device: document.getElementById('chip-device').textContent,
  timing: document.getElementById('chip-timing').textContent,
}))
check('chips report engine/device/timing', /worker/.test(chips.mode) && /ms/.test(chips.timing),
  `${chips.mode} | ${chips.device} | ${chips.timing}`)
check('no uncaught page errors', pageErrors.length === 0, pageErrors.slice(0, 2).join(' ; ') || 'none')
const realMisses = notFound.filter((u) => !/favicon/.test(u))
check('no unexpected 404s (favicon aside)', realMisses.length === 0,
  `404s: ${notFound.map((u) => u.replace(/^https?:\/\/[^/]+/, '')).join(', ') || 'none'}`)

await browser.close(); server.close()
const bad = results.filter((r) => !r.ok)
console.log(`\n[offline] ${results.length - bad.length}/${results.length} passed`)
if (bad.length) { console.error('[offline] ✗ SEGLAB offline self-test FAILED'); process.exit(1) }
console.log('[offline] ✓ app verified end-to-end (models stubbed — run verify.mjs for model accuracy)')

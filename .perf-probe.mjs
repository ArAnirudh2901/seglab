// Per-click cost, measured on the real page: what the lane spends (encode /
// decode / post) and what the app spends after it (compose, regularise, paint,
// persist). The app-side number is wall time minus the lane's own total, which
// is the half no in-lane timer can see.
//
//   bun .perf-probe.mjs [--cft] [--nef]
//
// Needs scripts/dev-server.mjs on :8788.
import { chromium } from 'playwright'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

const channel = resolveChannel()
const useNef = process.argv.includes('--nef')
console.log(`[perf] ${channelLabel(channel)}`)

const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    viewport: { width: 1512, height: 900 },
}, channel))
const page = ctx.pages()[0] ?? await ctx.newPage()
await page.addInitScript(() => { window.__seglabNoRestore = true })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))

await page.goto('http://127.0.0.1:8788/?norestore=1', { waitUntil: 'load' })
await page.waitForFunction(() => !!window.__seglab, { timeout: 60000 })
await page.evaluate(new Function(`return window.__seglab.importUrl(${JSON.stringify(useNef ? '/2680558334.nef' : '/.cache/testimg/cubes.webp')})`))
await page.waitForFunction(() => window.__seglab.state().ready && window.__seglab.state().hasImage, { timeout: 240000 })

const geo = await page.evaluate(() => window.__seglab.demoGeometry())
const p = geo?.proxyScale ?? 1
// The proxy canvas, which is the frame every per-pixel pass actually walks.
const [W, H] = await page.evaluate(() => [document.getElementById('view').width, document.getElementById('view').height])

// Six adjacent cubes for the demo; a grid otherwise. Each click is a separate
// object, so the op stack grows and composition cost grows with it.
const POINTS = useNef
    ? [[0.35, 0.45], [0.5, 0.5], [0.62, 0.4], [0.45, 0.62], [0.7, 0.6], [0.3, 0.3]].map(([fx, fy]) => [fx * W, fy * H])
    : [[440, 340], [420, 200], [555, 190], [610, 300], [660, 210], [300, 230]].map(([x, y]) => [x * p, y * p])

const click = async ([x, y]) => page.evaluate(async ([cx, cy]) => {
    const t0 = performance.now()
    await window.__seglab.clickAt(cx, cy)
    const wall = performance.now() - t0
    const st = window.__seglab.state()
    const r = st.lastRun || {}
    const q = st.lastPaint || {}
    return { wall, encode: r.encodeMs || 0, decode: r.decodeMs || 0, post: r.postMs || 0, lane: r.ms || 0,
        compose: q.composeMs || 0, paint: q.paintMs || 0, cv: q.cvMs || 0, esc: q.escalateMs || 0 }
}, [x, y])

await page.evaluate(() => window.__seglab.reset())
await click(POINTS[0])          // warm: model build + first encode
await page.evaluate(() => window.__seglab.reset())

const rows = []
for (const pt of POINTS) rows.push(await click(pt))

console.log(`\nframe ${W}x${H}${useNef ? '  (nef)' : ''}`)
const COLS = [['wall', (r) => r.wall], ['encode', (r) => r.encode], ['decode', (r) => r.decode],
    ['post', (r) => r.post], ['app', (r) => r.wall - r.lane], ['compose', (r) => r.compose],
    ['paint', (r) => r.paint], ['cv', (r) => r.cv], ['esc', (r) => r.esc]]
console.log(`#   ${COLS.map(([n]) => n.padStart(7)).join(' ')}`)
rows.forEach((r, i) => {
    console.log(`${String(i + 1).padEnd(3)} ${COLS.map(([, f]) => f(r).toFixed(1).padStart(7)).join(' ')}`)
})
const mean = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length
console.log(`mean${COLS.map(([, f]) => mean(f).toFixed(1).padStart(8)).join('')}`)

console.log(`\npage errors: ${errors.length ? errors.join(' | ') : 'none'}`)
await ctx.close()

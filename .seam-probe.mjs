// Seam + boundary probe: what happens where two selections meet, and how
// jagged the boundary is — measured on real decodes, with the shipped code.
//
//   bun .seam-probe.mjs [--cft] [--corpus] [--demo]
//
// The report: selecting several adjacent objects leaves a hairline of
// background between them, so the outline draws a border THROUGH the middle of
// what was selected as one thing; and the boundary itself wobbles by a pixel.
// This pulls the composed mask out of the page and sweeps the two radii with
// js/sam-core.js itself, so the numbers below are the shipping functions.
//
// Needs scripts/dev-server.mjs on :8788 and .cache/testimg/.
import { writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { bridgeGaps, smoothBoundary } from './js/sam-core.js'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

const SHOTS = process.env.SHOTS || '/Users/anirudharavalli/.claude/jobs/070e5fa6/tmp'
const channel = resolveChannel()
console.log(`[seam] ${channelLabel(channel)}`)

const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    viewport: { width: 1512, height: 900 },
}, channel))
const page = ctx.pages()[0] ?? await ctx.newPage()
await page.addInitScript(() => { window.__seglabNoRestore = true })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))

const boot = async (url) => {
    await page.goto('http://127.0.0.1:8788/?norestore=1', { waitUntil: 'load' })
    await page.waitForFunction(() => !!window.__seglab, { timeout: 60000 })
    await page.evaluate(new Function(`return window.__seglab.${url}`))
    await page.waitForFunction(() => window.__seglab.state().ready && window.__seglab.state().hasImage, { timeout: 240000 })
    return page.evaluate(() => window.__seglab.demoGeometry())
}
const shot = async (name) => writeFile(`${SHOTS}/${name}`, await page.locator('#view').screenshot())

/* ── the composed mask, as RGBA, so the real functions can run on it ──── */
const grab = async () => {
    const got = await page.evaluate(() => window.__seglab.maskPixels())
    if (!got) return null
    const chan = new Uint8Array(Buffer.from(got.b64, 'base64'))
    return { w: got.w, h: got.h, chan }
}
const toRGBA = ({ w, h, chan }) => {
    const rgba = new Uint8ClampedArray(w * h * 4)
    for (let i = 0; i < chan.length; i += 1) {
        const j = i * 4
        rgba[j] = chan[i]; rgba[j + 1] = chan[i]; rgba[j + 2] = chan[i]; rgba[j + 3] = 255
    }
    return rgba
}

/* ── metrics: components, enclosed holes, boundary length ─────────────── */
const label = (core, w, h) => {
    const n = w * h
    const lab = new Int32Array(n).fill(-1)
    const stack = new Int32Array(n)
    let count = 0
    for (let i = 0; i < n; i += 1) {
        if (!core[i] || lab[i] >= 0) continue
        count += 1
        let sp = 0
        stack[sp] = i; sp += 1
        lab[i] = count
        while (sp) {
            sp -= 1
            const p = stack[sp]
            const x = p % w
            const y = (p - x) / w
            for (let k = 0; k < 8; k += 1) {
                const xx = x + [0, 1, 1, 1, 0, -1, -1, -1][k]
                const yy = y + [-1, -1, 0, 1, 1, 1, 0, -1][k]
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                const q = yy * w + xx
                if (core[q] && lab[q] < 0) { lab[q] = count; stack[sp] = q; sp += 1 }
            }
        }
    }
    return count
}
const metrics = (rgba, w, h) => {
    const n = w * h
    const core = new Uint8Array(n)
    let area = 0
    for (let i = 0; i < n; i += 1) if (rgba[i * 4] >= 128) { core[i] = 1; area += 1 }
    // enclosed background = flood the outside, keep the rest
    const seen = new Uint8Array(n)
    const stack = new Int32Array(n)
    let sp = 0
    const push = (i) => { if (!core[i] && !seen[i]) { seen[i] = 1; stack[sp] = i; sp += 1 } }
    for (let x = 0; x < w; x += 1) { push(x); push((h - 1) * w + x) }
    for (let y = 0; y < h; y += 1) { push(y * w); push(y * w + w - 1) }
    while (sp) {
        sp -= 1
        const p = stack[sp]
        const x = p % w
        if (x > 0) push(p - 1)
        if (x < w - 1) push(p + 1)
        if (p >= w) push(p - w)
        if (p < n - w) push(p + w)
    }
    const inner = new Uint8Array(n)
    let holePx = 0
    for (let i = 0; i < n; i += 1) if (!core[i] && !seen[i]) { inner[i] = 1; holePx += 1 }
    // boundary length: core cells with a 4-neighbour outside. A staircase costs
    // more perimeter than the straight edge it approximates, so this is the
    // jaggedness number.
    let per = 0
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            const p = y * w + x
            if (!core[p]) continue
            if ((x === 0 || !core[p - 1]) || (x === w - 1 || !core[p + 1])
                || (y === 0 || !core[p - w]) || (y === h - 1 || !core[p + w])) per += 1
        }
    }
    return {
        area,
        components: label(core, w, h),
        holes: label(inner, w, h),
        holePx,
        perimeter: per,
        // 1.0 for a disc; higher means a longer boundary for the same area
        rough: +(per / (2 * Math.sqrt(Math.PI * area))).toFixed(3),
        core,
    }
}
const iou = (a, b) => {
    let i = 0
    let u = 0
    for (let k = 0; k < a.length; k += 1) {
        if (a[k] || b[k]) u += 1
        if (a[k] && b[k]) i += 1
    }
    return u ? i / u : 1
}

/* ── 1. several adjacent objects, the case from the report ────────────── */
if (!process.argv.includes('--skip-cubes')) {
const geo = await boot('importUrl("/.cache/testimg/cubes.webp")')
const p = geo.proxyScale
console.log(`cubes ${geo.originalW}x${geo.originalH}  proxyScale ${p.toFixed(3)}`)
const CUBES = [['front', 440, 340], ['big', 420, 200], ['tallright', 555, 190],
    ['midright', 665, 245], ['frontright', 645, 350], ['backleft', 320, 295]]
await page.evaluate(() => window.__seglab.reset())
for (const [, x, y] of CUBES) {
    await page.evaluate(async ([cx, cy]) => window.__seglab.clickAt(cx, cy), [x * p, y * p])
}
await shot('seam-cubes.png')

const union = await grab()
const base = toRGBA(union)
const m0 = metrics(base, union.w, union.h)
console.log(`\nunion of ${CUBES.length} cubes: area ${m0.area}  components ${m0.components}`
    + `  holes ${m0.holes} (${m0.holePx} px)  perimeter ${m0.perimeter}  rough ${m0.rough}`)

console.log('\nbridgeGaps sweep (holes left, px added, boundary):')
for (const r of [1, 2, 3, 4]) {
    const rgba = Uint8ClampedArray.from(base)
    const filled = bridgeGaps(rgba, union.w, union.h, { radius: r, rect: null })
    const m = metrics(rgba, union.w, union.h)
    console.log(`  r=${r}  filled ${String(filled).padStart(5)} (+${(100 * filled / m0.area).toFixed(2)}%)`
        + `  holes ${m.holes} (${m.holePx} px)  perimeter ${m.perimeter}  rough ${m.rough}`)
}

console.log('\nsmoothBoundary sweep (on top of bridge r=2):')
{
    const bridged = Uint8ClampedArray.from(base)
    bridgeGaps(bridged, union.w, union.h, { radius: 2 })
    const mb = metrics(bridged, union.w, union.h)
    for (const r of [1, 2, 3]) {
        const rgba = Uint8ClampedArray.from(bridged)
        const changed = smoothBoundary(rgba, union.w, union.h, { radius: r })
        const m = metrics(rgba, union.w, union.h)
        console.log(`  r=${r}  changed ${String(changed).padStart(6)}  area ${m.area} (${(100 * (m.area - mb.area) / mb.area).toFixed(2)}%)`
            + `  perimeter ${m.perimeter} (${(100 * (m.perimeter - mb.perimeter) / mb.perimeter).toFixed(1)}%)`
            + `  rough ${m.rough}  components ${m.components}  IoU ${iou(m.core, mb.core).toFixed(4)}`)
    }
}

}

/* ── 2. the shapes with exact geometry: what smoothing costs a corner ─── */
if (process.argv.includes('--demo')) {
    const g = await boot('loadDemo(1800)')
    const s = g.proxyScale
    await page.evaluate(() => window.__seglab.reset())
    await page.evaluate(async (a) => window.__seglab.clickAt(a[0], a[1]),
        [(g.square.x + g.square.w / 2) * s, (g.square.y + g.square.h / 2) * s])
    const sq = await grab()
    const rgba0 = toRGBA(sq)
    // analytic truth for the square, in proxy px
    const truth = new Uint8Array(sq.w * sq.h)
    for (let y = 0; y < sq.h; y += 1) {
        for (let x = 0; x < sq.w; x += 1) {
            const inX = x >= g.square.x * s && x < (g.square.x + g.square.w) * s
            const inY = y >= g.square.y * s && y < (g.square.y + g.square.h) * s
            truth[y * sq.w + x] = inX && inY ? 1 : 0
        }
    }
    const m0d = metrics(rgba0, sq.w, sq.h)
    console.log(`\ndemo square ${(g.square.w * s).toFixed(0)}px: IoU vs truth ${iou(m0d.core, truth).toFixed(4)}`
        + `  perimeter ${m0d.perimeter}  rough ${m0d.rough}`)
    for (const r of [1, 2, 3]) {
        const rgba = Uint8ClampedArray.from(rgba0)
        smoothBoundary(rgba, sq.w, sq.h, { radius: r })
        const m = metrics(rgba, sq.w, sq.h)
        console.log(`  smooth r=${r}  IoU vs truth ${iou(m.core, truth).toFixed(4)}`
            + `  area ${(100 * (m.area - m0d.area) / m0d.area).toFixed(2)}%  perimeter ${m.perimeter}  rough ${m.rough}`)
    }
}

/* ── 3. corpus: a thin subject is where smoothing can do damage ───────── */
if (process.argv.includes('--corpus')) {
    const CORPUS = [
        ['streetlight', '/streetlight.jpg', [[0.5, 0.5], [0.42, 0.3]]],
        ['toon', '/.cache/testimg/toon.png', [[0.4, 0.5], [0.6, 0.35]]],
        ['nef', '/.cache/testimg/2680558334.nef', [[0.5, 0.5], [0.35, 0.45]]],
    ]
    for (const [name, url, pts] of CORPUS) {
        let g
        try { g = await boot(`importUrl(${JSON.stringify(url)})`) } catch (e) { console.log(`${name}: ${e.message}`); continue }
        const PW = g.originalW * g.proxyScale
        const PH = g.originalH * g.proxyScale
        for (const [rx, ry] of pts) {
            await page.evaluate(() => window.__seglab.reset())
            await page.evaluate(async (a) => window.__seglab.clickAt(a[0], a[1]), [rx * PW, ry * PH])
            const got = await grab()
            if (!got) { console.log(`${name} ${rx},${ry}: no mask`); continue }
            const rgba0 = toRGBA(got)
            const m0c = metrics(rgba0, got.w, got.h)
            const line = [`${name.padEnd(11)} ${rx},${ry}  area ${String(m0c.area).padStart(6)}`
                + ` comp ${m0c.components} holes ${m0c.holes} rough ${m0c.rough}`]
            for (const r of [1, 2]) {
                const rgba = Uint8ClampedArray.from(rgba0)
                const filled = bridgeGaps(rgba, got.w, got.h, { radius: 2 })
                const changed = smoothBoundary(rgba, got.w, got.h, { radius: r })
                const m = metrics(rgba, got.w, got.h)
                line.push(`    b2+s${r}: area ${(100 * (m.area - m0c.area) / m0c.area).toFixed(2)}%`
                    + ` comp ${m.components} holes ${m.holes} rough ${m.rough}`
                    + ` IoU ${iou(m.core, m0c.core).toFixed(4)} (filled ${filled}, changed ${changed})`)
            }
            console.log(line.join('\n'))
        }
    }
}

if (errors.length) console.log(`\npage errors: ${errors.length}\n${errors.slice(0, 5).join('\n')}`)
await ctx.close()

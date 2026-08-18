// Coexistence + mask-hygiene probe.
//
//   bun .coexist-probe.mjs [--cft]
//
// Two things verify.mjs now covers but only at the end of a multi-hour suite:
//
//   1. Every tool that opens a new live object COMMITS the finished one first.
//      Four of the six paths used to drop it, so a box drawn after a click
//      silently unselected what the click selected. The failure is invisible in
//      any single-tool test — it needs the matrix.
//   2. The outliers on the cubes photo, measured rather than eyeballed: what
//      each hygiene rule removed, and what SAM's own field says about what is
//      left (`--dump` writes the 256² plane, `--census` counts removals over a
//      grid of clicks on four photographs).
//
// Needs scripts/dev-server.mjs on :8788, and .cache/testimg/cubes.webp.
import { writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

const VW = 1512
const VH = 900
const channel = resolveChannel()
console.log(`[coexist] ${channelLabel(channel)}`)

let failed = 0
const check = (label, ok, detail) => {
    if (!ok) failed += 1
    console.log(`${ok ? 'ok  ' : '✗   '}${label} — ${detail}`)
}

const errors = []
const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    viewport: { width: VW, height: VH },
}, channel))
const page = ctx.pages()[0] ?? await ctx.newPage()
await page.addInitScript(() => { window.__seglabNoRestore = true })
page.on('pageerror', (e) => errors.push(String(e)))

const boot = async (load) => {
    await page.goto('http://127.0.0.1:8788/?norestore=1', { waitUntil: 'load' })
    await page.waitForFunction(() => !!window.__seglab, { timeout: 60000 })
    await page.evaluate(load)
    await page.waitForFunction(() => window.__seglab.state().ready && window.__seglab.state().hasImage,
        { timeout: 240000 })
    return page.evaluate(() => window.__seglab.demoGeometry())
}
const reset = () => page.evaluate(() => window.__seglab.reset())
const covOf = (s) => s?.maskSummary?.coverage || 0

const cubesOnly = process.argv.includes('--cubes')

/* ── 1. the coexistence matrix, on the demo scene ─────────────────────── */
if (!cubesOnly) {
const geo = await boot(() => window.__seglab.loadDemo(1800))
const p = geo.proxyScale
const W0 = geo.originalW
const H0 = geo.originalH

const target = (shape) => (shape === 'disc'
    ? {
        pt: [geo.disc.x * p, geo.disc.y * p],
        r: geo.disc.r * p * 1.15,
        box: [(geo.disc.x - geo.disc.r) * p, (geo.disc.y - geo.disc.r) * p,
            (geo.disc.x + geo.disc.r) * p, (geo.disc.y + geo.disc.r) * p],
        stroke: [[(geo.disc.x - geo.disc.r * 0.5) * p, geo.disc.y * p],
            [(geo.disc.x + geo.disc.r * 0.5) * p, geo.disc.y * p]],
    }
    : shape === 'square'
        ? {
            pt: [(geo.square.x + geo.square.w / 2) * p, (geo.square.y + geo.square.h / 2) * p],
            r: geo.square.w * p * 0.8,
            box: [geo.square.x * p, geo.square.y * p,
                (geo.square.x + geo.square.w) * p, (geo.square.y + geo.square.h) * p],
            stroke: [[(geo.square.x + 20) * p, (geo.square.y + geo.square.h / 2) * p],
                [(geo.square.x + geo.square.w - 20) * p, (geo.square.y + geo.square.h / 2) * p]],
        }
        : { // empty background, so a manual shape there always adds area
            pt: [0.08 * W0 * p, 0.15 * H0 * p],
            r: 0.05 * W0 * p,
            box: [0.02 * W0 * p, 0.06 * H0 * p, 0.14 * W0 * p, 0.24 * H0 * p],
            stroke: [[geo.dot.x * p - 4, geo.dot.y * p], [geo.dot.x * p + 4, geo.dot.y * p]],
        })

const applyTool = (tool, arg) => page.evaluate(async ({ tool: t, arg: a }) => {
    const S = window.__seglab
    if (t === 'click') return S.clickAt(a.pt[0], a.pt[1])
    if (t === 'box') return S.boxAt(a.box[0], a.box[1], a.box[2], a.box[3])
    if (t === 'lasso') return S.lassoCircle(a.pt[0], a.pt[1], a.r)
    if (t === 'rect') return S.manualRect(a.box[0], a.box[1], a.box[2], a.box[3])
    if (t === 'ellipse') return S.manualEllipse(a.box[0], a.box[1], a.box[2], a.box[3])
    if (t === 'region') return S.manualRegionCircle(a.pt[0], a.pt[1], a.r)
    if (t === 'brush') return S.brushStroke(a.stroke)
    if (t === 'text') return S.selectBoxes([a.box])
    throw new Error(`unknown tool ${t}`)
}, { tool, arg })

const FOLLOWERS = ['box', 'lasso', 'rect', 'ellipse', 'region', 'brush', 'text']
const coexist = []
for (const tool of FOLLOWERS) {
    await reset()
    const first = await applyTool('click', target('disc'))
    const after = await applyTool(tool, target('square'))
    coexist.push({ tool, before: +covOf(first).toFixed(4), after: +covOf(after).toFixed(4), baseOps: after.baseOps })
}
const lost = coexist.filter((r) => !(r.before > 0 && r.after > r.before && r.baseOps >= 1))
check('a clicked object survives every tool used after it', lost.length === 0,
    lost.length ? JSON.stringify(lost) : coexist.map((r) => `${r.tool} ${r.before}→${r.after}`).join(', '))

const producers = []
for (const tool of ['box', 'lasso', 'text']) {
    await reset()
    const first = await applyTool(tool, target('disc'))
    const after = await applyTool('click', target('square'))
    producers.push({ tool, before: +covOf(first).toFixed(4), after: +covOf(after).toFixed(4), baseOps: after.baseOps })
}
const dropped = producers.filter((r) => !(r.before > 0 && r.after > r.before && r.baseOps >= 1))
check('a box / lasso / text object survives a later click', dropped.length === 0,
    dropped.length ? JSON.stringify(dropped) : producers.map((r) => `${r.tool} ${r.before}→${r.after}`).join(', '))

await reset()
const chain = []
chain.push(await applyTool('click', target('disc')))
chain.push(await applyTool('box', target('square')))
chain.push(await applyTool('brush', target('free')))
chain.push(await applyTool('rect', target('free')))
const chainCov = chain.map(covOf)
const chainStats = await page.evaluate(() => window.__seglab.maskStats())
check('a four-tool chain only ever grows the selection',
    chainCov.every((c, i) => c > 0 && (i === 0 || c > chainCov[i - 1])) && chainStats?.components >= 3,
    `coverage ${chainCov.map((c) => (c * 100).toFixed(2)).join('% → ')}%, components=${chainStats?.components}`)

// Every ORDERED pair of the eight tools, which is the combination the report
// asked for: whatever the first tool selected is still selected after the
// second one runs, whichever two they are.
const ALL = ['click', 'box', 'lasso', 'rect', 'ellipse', 'region', 'brush', 'text']
const pairs = []
for (const a of ALL) {
    for (const b of ALL) {
        await reset()
        const s1 = await applyTool(a, target('disc'))
        const s2 = await applyTool(b, target('square'))
        const c1 = covOf(s1)
        const c2 = covOf(s2)
        if (!(c1 > 0 && c2 > c1 * 1.05)) pairs.push(`${a}→${b} ${c1.toFixed(4)}→${c2.toFixed(4)}`)
    }
}
check(`all ${ALL.length * ALL.length} ordered tool pairs coexist`, pairs.length === 0,
    pairs.length ? pairs.join('; ') : 'every second tool grew the selection')
}

/* ── 2. the cubes photo the outlier report came from ─────────────────── */
const geoC = await boot(() => window.__seglab.importUrl('/.cache/testimg/cubes.webp'))
const pc = geoC.proxyScale
console.log(`\ncubes ${geoC.originalW}x${geoC.originalH}  proxyScale ${pc.toFixed(3)}`)

// One click per cube: the bulge is a property of which cube was asked for, and
// the report's screenshot does not say which one that was.
const SHOTS = '/private/tmp/claude-501/-Users-anirudharavalli-Web-Dev-NextJS-seglab/e9cc4c93-31c6-462f-b892-7337b265505a/scratchpad'
const shot = async (name) => {
    const buf = await page.locator('#view').screenshot()
    await writeFile(`${SHOTS}/${name}`, buf)
}
const CUBES = {
    front: [440, 340],        // tall front cube
    backleft: [320, 295],     // low wide box, far left
    big: [420, 200],          // big cube behind it
    tallright: [555, 190],    // tall cube, top face bright
    midright: [665, 245],     // small cube, right shoulder
    frontright: [645, 350],   // front-right cube
    middle: [530, 300],       // the slab between front and right
}
for (const [name, [x, y]] of Object.entries(CUBES)) {
    await reset()
    const s = await page.evaluate(async ([cx, cy]) => window.__seglab.clickAt(cx, cy), [x * pc, y * pc])
    const pick = s.lastRun?.pick
    console.log(`${name.padEnd(11)} cov ${(covOf(s) * 100).toFixed(2).padStart(6)}%  score ${(s.score || 0).toFixed(3)}`
        + `  grid ${JSON.stringify(pick?.regions)}  proxy ${JSON.stringify(pick?.cleaned)}`)
    await shot(`cubes-${name}.png`)
    if (process.argv.includes('--dump')) {
        const pl = await page.evaluate(() => Array.from(globalThis.__seglabPlane?.plane || []))
        await writeFile(`${SHOTS}/plane-${name}.bin`, Buffer.from(new Float32Array(pl).buffer))
    }
}
/* ── 2b. a diffuse subject, where a confidence rule is most dangerous ── */
if (process.argv.includes('--toon')) {
    const tag = process.argv[process.argv.indexOf('--toon') + 1] || 'on'
    const g = await boot(new Function('return window.__seglab.importUrl("/.cache/testimg/toon.png")'))
    const PW = g.originalW * g.proxyScale
    const PH = g.originalH * g.proxyScale
    for (const [rx, ry] of [[0.4, 0.5], [0.6, 0.35], [0.5, 0.7]]) {
        await reset()
        const s = await page.evaluate(async ([x, y]) => window.__seglab.clickAt(x, y), [rx * PW, ry * PH])
        const r = s.lastRun?.pick?.regions
        console.log(`toon ${tag} ${rx},${ry}  cov ${(covOf(s) * 100).toFixed(2)}%  ${JSON.stringify(r)}`)
        await shot(`toon-${tag}-${rx}-${ry}.png`)
    }
}

/* ── 3. what hygiene removes across real photographs ─────────────────── */
// Twelve clicks per photo, counting what each rule took. This is what a new
// hygiene rule has to be measured against before it ships: the confidence-based
// lobe cut looked right on synthetic cones and took 84% of a real mask here.
if (process.argv.includes('--census')) {
    const census = process.env.CENSUS ? process.env.CENSUS.split(',') : ['/streetlight.jpg', '/2680558334.nef', '/.cache/testimg/DSC_0139.NEF', '/.cache/testimg/toon.png']
    for (const url of census) {
        const g = await boot(new Function('return window.__seglab.importUrl(' + JSON.stringify(url) + ')'))
        const PW = g.originalW * g.proxyScale
        const PH = g.originalH * g.proxyScale
        const tally = { clicks: 0, islands: 0, holes: 0, empty: 0, cov: [] }
        for (let ry = 1; ry <= 3; ry += 1) {
            for (let rx = 1; rx <= 4; rx += 1) {
                await reset()
                const s = await page.evaluate(async ([x, y]) => window.__seglab.clickAt(x, y),
                    [(rx / 5) * PW, (ry / 4) * PH])
                const r = s.lastRun?.pick?.regions
                tally.clicks += 1
                tally.cov.push(covOf(s) * 100)
                tally.islands += r?.islands || 0
                tally.holes += r?.holes || 0
                if (!covOf(s)) tally.empty += 1
                if ((r?.islands || 0) > 200) {   // a big removal has to be looked at
                    const name = `${url.split('/').pop()}-${rx}${ry}`
                    console.log(`  big cut ${name} cov ${(covOf(s) * 100).toFixed(2)}% ${JSON.stringify(r)}`)
                    await shot(`cut-${name}.png`)
                }
            }
        }
        console.log(`${url.padEnd(32)} ${JSON.stringify(tally)}  cov ${tally.cov.map((c) => c.toFixed(2)).join(' ')}`)
    }
}

console.log(`\npage errors: ${errors.length ? errors.join(' | ') : 'none'}`)
await ctx.close()
console.log(failed ? `\n${failed} FAILED` : '\nall ok')
process.exit(failed ? 1 : 0)

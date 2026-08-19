// Gesture probe: zoom, pan, and the arbitration between the view and the tools.
//
//   bun .zoom-probe.mjs [--cft]
//
// gestures.js transforms #frame, and every click is measured against that
// transformed box — so a wrong sign or a stale rect does not throw, it selects
// the wrong pixel. Nothing else drives it: `clickAt` feeds canvas coordinates
// straight in, bypassing `toCanvas` entirely. Everything here is the real
// mouse, the real wheel, real keys and real touch points.
//
// The demo scene carries the geometry checks: a click one pixel off the pole in
// streetlight.jpg selects something else entirely, which would read as a
// mapping bug. Its disc is 100 px of flat red — a mask that moves means the
// coordinates moved. streetlight then carries the scope checks, because a
// synthetic disc gives SAM nothing to be ambiguous about.
//
// Needs scripts/dev-server.mjs on :8788.
import { chromium } from 'playwright'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

const VW = 1512
const VH = 900
const channel = resolveChannel()
console.log(`[zoom] ${channelLabel(channel)}`)

let failed = 0
const check = (label, ok, detail) => {
    if (!ok) failed += 1
    console.log(`${ok ? 'ok  ' : '✗   '}${label} — ${detail}`)
}

// Two passes: the device class is a property of the CONTEXT. Chromium reports
// `pointer: coarse` as soon as touch points exist, so a context that can pinch
// boots the app in its touch vocabulary and cannot stand in for a mouse.
const errors = []
const launch = async (touch) => {
    const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-gpu'],
        viewport: { width: VW, height: VH },
        hasTouch: touch,
    }, channel))
    const page = ctx.pages()[0] ?? await ctx.newPage()
    // restoreSession() runs at module eval and reads this flag: setting it after
    // goto is a race the last session's photo wins, and every coordinate here is
    // then measured against the wrong image. `?norestore=1` is not read at all.
    await page.addInitScript(() => { window.__seglabNoRestore = true })
    page.on('pageerror', (e) => errors.push(String(e)))
    return [ctx, page]
}
let [ctx, page] = await launch(false)

/* ── page-side helpers ────────────────────────────────────────────────── */
const geom = () => page.evaluate(() => {
    const c = document.getElementById('overlay')
    const o = c.getBoundingClientRect()
    const s = document.getElementById('stage').getBoundingClientRect()
    const f = document.getElementById('frame')
    const m = new DOMMatrixReadOnly(getComputedStyle(f).transform)
    return {
        o: { x: o.left, y: o.top, w: o.width, h: o.height },
        s: { x: s.left, y: s.top, w: s.width, h: s.height },
        zoom: m.a, tx: m.e, ty: m.f,
        inline: f.getAttribute('style') || '',
        cw: c.width, ch: c.height,
        vw: document.getElementById('view').width,
        vh: document.getElementById('view').height,
        input: document.body.dataset.input,
        readoutHidden: !!document.getElementById('zoomreset')?.hidden,
        readout: (document.getElementById('zoomreset')?.textContent || '').trim(),
    }
})

const st = () => page.evaluate(() => window.__seglab.state())
const stats = () => page.evaluate(() => window.__seglab.maskStats())

/** Viewport point currently showing canvas pixel (cx, cy). */
const screenOf = async (cx, cy) => {
    const g = await geom()
    return [g.o.x + (cx + 0.5) * (g.o.w / g.cw), g.o.y + (cy + 0.5) * (g.o.h / g.ch)]
}

/** Import swaps the display canvas in mid-flight; a rect read a frame early
 *  belongs to the old layout and every coordinate off it lands elsewhere. The
 *  box holds its old ASPECT for a beat while carrying the new pixel grid, and
 *  that state is stable enough to look settled — so the aspect has to agree
 *  with the canvas before the rect can be trusted. */
const settle = async (label = '') => {
    let last = null
    for (let i = 0; i < 120; i += 1) {
        const g = await geom()
        const key = `${g.o.x.toFixed(1)},${g.o.y.toFixed(1)},${g.o.w.toFixed(1)},${g.o.h.toFixed(1)},${g.cw}`
        const skew = Math.abs((g.o.w / g.o.h) / (g.cw / g.ch) - 1)
        if (key === last && skew < 0.01) return g
        last = key
        await page.waitForTimeout(100)
    }
    throw new Error(`layout never settled ${label}: ${last}`)
}

// The prompt count is the only honest signal that the CLICK is what produced
// the mask: revision also ticks for an encode, so waiting on it lets a mask
// that was already on screen pass for a fresh selection.
const clickCanvas = async (cx, cy, button = 'left') => {
    const before = (await st()).clicks
    const [sx, sy] = await screenOf(cx, cy)
    if (sx < 0 || sy < 0 || sx > VW || sy > VH) {
        throw new Error(`canvas ${cx},${cy} sits off-screen at ${sx.toFixed(0)},${sy.toFixed(0)}`)
    }
    await page.mouse.click(sx, sy, { button })
    await page.waitForFunction(
        (n) => window.__seglab.state().clicks > n && window.__seglab.state().maskSummary,
        before, { timeout: 30000 },
    ).catch(async () => {
        const s = await st()
        throw new Error(`click ${cx},${cy} → ${sx.toFixed(0)},${sy.toFixed(0)} never settled:`
            + ` clicks ${before}→${s.clicks}, mask ${!!s.maskSummary}, geom ${JSON.stringify(await geom())}`)
    })
    await page.waitForTimeout(350) // cv-refine lands after the result
    return { ...(await st()), ...(await stats()) }
}

const wheelAt = async (sx, sy, dy, n = 1) => {
    await page.mouse.move(sx, sy)
    for (let i = 0; i < n; i += 1) { await page.mouse.wheel(0, dy); await page.waitForTimeout(40) }
    await page.waitForTimeout(140)
}

const dragMiddle = async (from, dx, dy) => {
    await page.mouse.move(from[0], from[1])
    await page.mouse.down({ button: 'middle' })
    await page.mouse.move(from[0] + dx, from[1] + dy, { steps: 8 })
    await page.mouse.up({ button: 'middle' })
    await page.waitForTimeout(120)
}

const boot = async (load) => {
    await page.goto('http://127.0.0.1:8788/?norestore=1', { waitUntil: 'load' })
    await page.waitForFunction(() => !!window.__seglab, { timeout: 60000 })
    await page.evaluate(load)
    await page.waitForFunction(() => window.__seglab.state().ready && window.__seglab.state().hasImage,
        { timeout: 240000 })
    return settle('after load')
}

const resetZoom = async () => { await page.keyboard.press('0'); await page.waitForTimeout(80) }
const clearMask = () => page.evaluate(() => window.__seglab.reset())

/* ── 1. the demo scene: does a click land where it is aimed? ──────────── */
const base = await boot(() => window.__seglab.loadDemo(1800))
console.log(`stage ${base.s.w.toFixed(0)}x${base.s.h.toFixed(0)}  photo ${base.o.w.toFixed(0)}x${base.o.h.toFixed(0)}`
    + `  canvas ${base.cw}x${base.ch}  input ${base.input}\n`)
check('boot: a mouse is not mistaken for a finger', base.input === 'mouse', `input=${base.input}`)
// #view alone is in flow, so it is the only thing sizing the box #photo and
// #overlay are stretched across. Let its grid drift from the overlay's, or the
// box drift from its aspect, and the photo is drawn distorted while every
// click is still measured against the box — a wrong pixel, never an error.
check('layout: the overlay grid is the interaction frame',
    base.cw === base.vw && base.ch === base.vh, `overlay ${base.cw}x${base.ch}, view ${base.vw}x${base.vh}`)
check('layout: the photo box keeps the frame aspect',
    Math.abs((base.o.w / base.o.h) / (base.vw / base.vh) - 1) < 0.01,
    `box ${base.o.w.toFixed(0)}x${base.o.h.toFixed(0)} (${(base.o.w / base.o.h).toFixed(3)})`
    + ` vs frame ${(base.vw / base.vh).toFixed(3)}`)

// A window wider than the photo is the case where a box sized by its container
// instead of its content shows itself, so ask for one.
await page.setViewportSize({ width: 1900, height: 760 })
await page.waitForTimeout(400)
const wide = await geom()
check('layout: a window wider than the photo does not stretch it',
    Math.abs((wide.o.w / wide.o.h) / (wide.vw / wide.vh) - 1) < 0.01,
    `box ${wide.o.w.toFixed(0)}x${wide.o.h.toFixed(0)} (${(wide.o.w / wide.o.h).toFixed(3)})`
    + ` vs frame ${(wide.vw / wide.vh).toFixed(3)}`)
await page.setViewportSize({ width: VW, height: VH })
await page.waitForTimeout(400)
await settle('after resize')

const geo = await page.evaluate(() => window.__seglab.demoGeometry())
// demoGeometry reports the LOADED transform, so a scene that never arrived is
// silently answered with the previous photo's numbers.
check('scene: the geometry describes the photo on screen', geo.originalW === 1800,
    `original ${geo.originalW}x${geo.originalH}`)
const DX = Math.round(geo.disc.x * geo.proxyScale)
const DY = Math.round(geo.disc.y * geo.proxyScale)
const idle = await st()
console.log(`disc at canvas ${DX},${DY}  (r ${(geo.disc.r * geo.proxyScale).toFixed(0)} px)`
    + `  idle clicks ${idle.clicks} mask ${idle.maskSummary ? 'PRESENT' : 'none'}`)
const flat = await clickCanvas(DX, DY)
const flatPt = flat.clickPoints?.[0] || []
const flatCov = flat.maskSummary?.coverage ?? 0
await clearMask()

// Zoom about the disc itself: whatever is under the cursor must stay under it.
// Only a pan that has run out of slack may move it, so the anchor is chosen
// mid-frame where clamp cannot bind.
const anchor = await screenOf(DX, DY)
await wheelAt(anchor[0], anchor[1], -240, 3)
const zoomed = await geom()
const held = await screenOf(DX, DY)
const slip = Math.hypot(held[0] - anchor[0], held[1] - anchor[1])
check(`zoom ${zoomed.zoom.toFixed(2)}×: the pixel under the cursor stays under the cursor`,
    slip <= 2, `(${anchor.map((v) => v.toFixed(0))}) → (${held.map((v) => v.toFixed(0))}), slip ${slip.toFixed(2)} px`)

// Then move it somewhere else entirely: an offset the code forgot to re-read
// shows up as a click on the wrong pixel, never as an error.
await dragMiddle(anchor, -180, -120)
const deep = await clickCanvas(DX, DY)
const deepPt = deep.clickPoints?.[0] || []
const drift = Math.hypot((deepPt[0] ?? 1e6) - flatPt[0], (deepPt[1] ?? 1e6) - flatPt[1])
check(`zoom ${zoomed.zoom.toFixed(2)}× + pan: the click still lands on the aimed pixel`,
    drift <= 2, `[${flatPt}] → [${deepPt}], drift ${drift.toFixed(2)} px`)
const ratio = flatCov ? (deep.maskSummary?.coverage ?? 0) / flatCov : 0
check('zoom + pan: the same pixel selects the same object',
    ratio > 0.9 && ratio < 1.1, `coverage ${flatCov.toFixed(4)} → ${(deep.maskSummary?.coverage ?? 0).toFixed(4)}`)

/* ── 2. pan is slack, never a way to lose the photo ───────────────────── */
await clearMask()
const mid = [base.s.x + base.s.w / 2, base.s.y + base.s.h / 2]
await dragMiddle(mid, 2000, 2000)
const panned = await geom()
const covers = panned.o.x <= panned.s.x + 1 && panned.o.y <= panned.s.y + 1
    && panned.o.x + panned.o.w >= panned.s.x + panned.s.w - 1
    && panned.o.y + panned.o.h >= panned.s.y + panned.s.h - 1
check('pan: dragging 2000 px cannot open a gap at the edge', covers,
    `photo at (${(panned.o.x - panned.s.x).toFixed(0)}, ${(panned.o.y - panned.s.y).toFixed(0)})`
    + ` size ${panned.o.w.toFixed(0)}x${panned.o.h.toFixed(0)} in ${panned.s.w.toFixed(0)}x${panned.s.h.toFixed(0)}`)
check('pan: a middle-drag is not a selection', (await st()).clicks === 0, `clicks ${(await st()).clicks}`)

/* ── 3. the zoom range is closed at both ends ─────────────────────────── */
await wheelAt(mid[0], mid[1], -240, 10)
const top = await geom()
check('zoom: clamped at 8× however hard the wheel is spun', top.zoom <= 8.001, `${top.zoom.toFixed(3)}×`)
check('zoom: the readout says where the view is', !top.readoutHidden && /8(\.0)?×/.test(top.readout),
    `"${top.readout}"`)
await wheelAt(mid[0], mid[1], 240, 18)
const home = await geom()
check('zoom: winds back to 1× and drops the transform',
    home.zoom === 1 && !/transform/.test(home.inline) && home.readoutHidden,
    `${home.zoom}× inline="${home.inline}" readout ${home.readoutHidden ? 'hidden' : 'shown'}`)

/* ── 4. keys ──────────────────────────────────────────────────────────── */
await page.keyboard.press('=')
await page.waitForTimeout(80)
check('key =: steps in about the centre', (await geom()).zoom > 1.3, `${(await geom()).zoom.toFixed(2)}×`)
await resetZoom()
check('key 0: resets the view', (await geom()).zoom === 1, `${(await geom()).zoom}×`)

await page.keyboard.press('=')
await page.waitForTimeout(80)
const beforeSpace = await geom()
const clicksBefore = (await st()).clicks
await page.keyboard.down(' ')
await page.mouse.move(mid[0], mid[1])
await page.mouse.down()
await page.mouse.move(mid[0] - 120, mid[1] - 90, { steps: 6 })
await page.mouse.up()
await page.keyboard.up(' ')
await page.waitForTimeout(150)
check('space-drag: pans instead of drawing',
    ((await geom()).tx !== beforeSpace.tx || (await geom()).ty !== beforeSpace.ty)
    && (await st()).clicks === clicksBefore,
    `Δtx ${((await geom()).tx - beforeSpace.tx).toFixed(0)}, clicks ${clicksBefore} → ${(await st()).clicks}`)

await resetZoom()
await page.locator('#mode-click').click()
await page.waitForTimeout(100)

/* ── 5. a real photo: the wheel over a selection is a scope step ──────── */
await boot(() => window.__seglab.importUrl('/streetlight.jpg'))
const real = await geom()
const RX = Math.round(real.cw * 0.42)
const RY = Math.round(real.ch * 0.55)
const live = await clickCanvas(RX, RY)
const scope0 = await page.evaluate(() => window.__seglab.scope())
const onMask = await screenOf(RX, RY)
await wheelAt(onMask[0], onMask[1], -240, 1)
const scope1 = await page.evaluate(() => window.__seglab.scope())
const afterScope = await geom()
check('arbitration: a wheel over the selection changes the reading, not the view',
    afterScope.zoom === 1 && !!scope0 && !!scope1 && scope1.index !== scope0.index,
    `zoom ${afterScope.zoom}×, scope ${scope0?.index} → ${scope1?.index} of ${scope1?.count}`)
await wheelAt(real.s.x + 12, real.s.y + 12, -240, 1)
check('arbitration: a wheel away from the selection still zooms',
    (await geom()).zoom > 1.2, `${(await geom()).zoom.toFixed(2)}×`)
check('arbitration: neither wheel committed an object',
    (await st()).baseOps === live.baseOps, `baseOps ${live.baseOps} → ${(await st()).baseOps}`)

const pill = await page.evaluate(() => {
    const el = document.getElementById('scope')
    if (!el || el.hidden) return null
    const r = el.getBoundingClientRect()
    const s = document.getElementById('stage').getBoundingClientRect()
    return {
        in: r.left >= s.left - 1 && r.right <= s.right + 1 && r.top >= s.top - 1 && r.bottom <= s.bottom + 1,
        box: [Math.round(r.left - s.left), Math.round(r.top - s.top), Math.round(r.width), Math.round(r.height)],
    }
})
check('scope: the control stays inside the stage under zoom', !pill || pill.in,
    pill ? `at ${pill.box.join(',')}` : 'hidden (no live selection)')

/* ── 6. touch: the whole vocabulary changes hands ─────────────────────── */
await ctx.close()
;[ctx, page] = await launch(true)
const glass = await boot(() => window.__seglab.loadDemo(1800))
check('touch: a finger is not mistaken for a mouse', glass.input === 'touch', `input=${glass.input}`)

const tapped = await clickCanvas(DX, DY)
check('touch: a tap selects', !!tapped.maskSummary && tapped.clicks === 1,
    `clicks ${tapped.clicks}, coverage ${(tapped.maskSummary?.coverage ?? 0).toFixed(4)}`)
await clearMask()

const cdp = await ctx.newCDPSession(page)
const touch = (type, pts) => cdp.send('Input.dispatchTouchEvent', {
    type, touchPoints: pts.map(([x, y], i) => ({ x, y, id: i })),
})
const [gx, gy] = [glass.s.x + glass.s.w / 2, glass.s.y + glass.s.h / 2]
await touch('touchStart', [[gx - 40, gy], [gx + 40, gy]])
for (let k = 1; k <= 6; k += 1) {
    await touch('touchMove', [[gx - 40 - k * 25, gy], [gx + 40 + k * 25, gy]])
    await page.waitForTimeout(30)
}
await touch('touchEnd', [])
await page.waitForTimeout(250)
check('touch: a pinch zooms', (await geom()).zoom > 1.4, `${(await geom()).zoom.toFixed(2)}×`)
check('touch: a pinch is a view gesture, not a selection',
    (await st()).clicks === 0 && !(await st()).maskSummary,
    `clicks ${(await st()).clicks}, mask ${(await st()).maskSummary ? 'yes' : 'none'}`)

// A finger lifting out of a pinch must not leave the surface swallowing taps.
await resetZoom()
const after = await clickCanvas(DX, DY)
check('touch: the surface is handed back after a pinch', !!after.maskSummary,
    `coverage ${(after.maskSummary?.coverage ?? 0).toFixed(4)}`)

console.log(`\n${failed ? `✗ ${failed} failed` : '✓ all gesture checks passed'} — page errors ${errors.length}`,
    errors.slice(0, 3))
await ctx.close()
process.exit(failed ? 1 : 0)

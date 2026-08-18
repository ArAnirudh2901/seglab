// Real-mouse probe: swatch click, wheel, vertical scrub, tap-the-same-spot.
// Synthetic events would skip pointer capture and the tap/scrub arbitration,
// which is the part worth proving.
import { chromium } from 'playwright'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

// The installed Google Chrome, like verify.mjs; `--cft` for the bundled build.
const channel = resolveChannel()
console.log(`[scope] ${channelLabel(channel)}`)
const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    viewport: { width: 1512, height: 900 },
}, channel))
const page = ctx.pages()[0] ?? await ctx.newPage()
// Before the document: restoreSession() reads this at module eval, and a
// restored photo makes #stage visible before the import below has landed.
await page.addInitScript(() => { window.__seglabNoRestore = true })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
const scope = () => page.evaluate(() => window.__seglab.scope())
const clicks = () => page.evaluate(() => window.__seglab.state().clicks)

await page.goto('http://127.0.0.1:8788/', { waitUntil: 'load' })
await page.evaluate(async () => {
    const blob = await (await fetch('/streetlight.jpg')).blob()
    const dt = new DataTransfer()
    dt.items.add(new File([blob], 'streetlight.jpg', { type: 'image/jpeg' }))
    const el = document.getElementById('file')
    el.files = dt.files
    el.dispatchEvent(new Event('change', { bubbles: true }))
})
await page.waitForFunction(() => document.getElementById('stage')?.classList.contains('visible'), { timeout: 120000 })
await page.waitForTimeout(1500)

const box = await page.locator('#overlay').boundingBox()
const px = box.x + box.width * 0.42
const py = box.y + box.height * 0.55
await page.mouse.click(px, py)
await page.waitForFunction(() => (window.__seglab?.scope()?.count ?? 0) > 1, { timeout: 180000 })
await page.waitForTimeout(400)
console.log('first click  ', JSON.stringify(await scope()), 'clicks', await clicks())

// Swatch click adopts.
await page.locator('#scope .scope-shape').last().click()
await page.waitForTimeout(2200)
console.log('last swatch  ', JSON.stringify(await scope()))

// Wheel over the selection steps down one, clamped at the bottom.
await page.mouse.move(px, py)
await page.mouse.wheel(0, 60)
await page.waitForTimeout(2000)
console.log('wheel down   ', JSON.stringify(await scope()))
await page.mouse.wheel(0, 60)
await page.waitForTimeout(2000)
await page.mouse.wheel(0, 60)
await page.waitForTimeout(2000)
console.log('wheel x3     ', JSON.stringify(await scope()), '(clamped at 0)')

// Vertical scrub up over the selection: 80 px = 2 steps.
await page.mouse.move(px, py)
await page.mouse.down()
for (let i = 1; i <= 8; i += 1) await page.mouse.move(px, py - i * 10)
await page.mouse.up()
await page.waitForTimeout(2500)
console.log('drag up 80px ', JSON.stringify(await scope()))

// Tapping the same spot is still a prompt, never a step.
const before = await clicks()
await page.mouse.click(px, py)
await page.waitForTimeout(3000)
console.log('tap same spot', 'clicks', before, '→', await clicks(), JSON.stringify(await scope()))
console.log('errors', errors.slice(0, 3))
await ctx.close()

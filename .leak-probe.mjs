// Leakage probe: real DSLR frames, adversarial click points, one report row
// per click. Measures what region hygiene claims to police — components that
// hold no click, how far they sit from the one that was clicked, and how solid
// they are — so a leak is a number rather than a screenshot.
//
//   bun .leak-probe.mjs [scene ...] [--shot]
//
// Needs scripts/dev-server.mjs on :8788 (COI + model cache), and the RAWs
// staged into .cache/testimg (gitignored — they are the machine's own photos,
// ~100 MB): DSC_0139.NEF, DSC_0221.NEF, d750-lossless.nef, 2680558334.nef.
import { chromium } from 'playwright'
import { channelLabel, onChannel, profileDir, resolveChannel } from './scripts/harness/browser.mjs'

// Points are FRACTIONS of the frame, picked off the preview of each photo.
// `note` is what the click is meant to be a hard case for.
const SCENES = {
    // Repeated subject at its worst: a ball of identical rivets on cloth.
    rivets: {
        url: '/.cache/testimg/d750-lossless.nef',
        clicks: [
            { x: 0.50, y: 0.47, note: 'one rivet mid-ball (repeated subject)' },
            { x: 0.50, y: 0.50, note: 'ball centre (whole object)' },
            { x: 0.33, y: 0.33, note: 'rivet near the ball rim' },
            { x: 0.12, y: 0.20, note: 'cloth fold (soft, low-contrast)' },
            { x: 0.86, y: 0.14, note: 'dark background corner' },
        ],
    },
    // Two red tulips in a field of hundreds of blue hyacinths. The classic
    // "matches the setting" trap, and thin stems/leaves for the wire rule.
    flowers: {
        url: '/.cache/testimg/2680558334.nef',
        clicks: [
            { x: 0.41, y: 0.27, note: 'left tulip (twin exists at 0.55,0.21)' },
            { x: 0.55, y: 0.21, note: 'right tulip (twin exists at 0.41,0.27)' },
            { x: 0.20, y: 0.45, note: 'one hyacinth in a field of them' },
            { x: 0.44, y: 0.42, note: 'thin leaf blade' },
            { x: 0.05, y: 0.05, note: 'blown highlight at the frame edge' },
        ],
    },
    // Cluttered room: repeated laptops, two near-identical backpacks, cables,
    // a thin chair frame, people overlapping.
    lab: {
        url: '/.cache/testimg/DSC_0139.NEF',
        clicks: [
            { x: 0.50, y: 0.65, note: 'left backpack (twin at 0.82,0.62)' },
            { x: 0.42, y: 0.24, note: 'one laptop among four' },
            { x: 0.20, y: 0.55, note: 'seated person (occluded by chair)' },
            { x: 0.12, y: 0.72, note: 'white chair (thin frame, on white floor)' },
            { x: 0.47, y: 0.30, note: 'cable on the desk (thin, dark on dark)' },
        ],
    },
    // The regression the dominance gate exists for: a subject whose real parts
    // are thin, far apart and legitimately fragmented. Nothing here may be
    // removed — 42 components on this crop are the streetlight, not speckle.
    streetlight: {
        url: '/streetlight.jpg',
        clicks: [
            { x: 0.42, y: 0.55, note: 'pole (thin, wires attached)' },
            { x: 0.42, y: 0.20, note: 'lamp head' },
            { x: 0.42, y: 0.55, note: 'tram window, then EXCLUDE the sky above', extra: [[0.60, 0.10, 1]] },
            { x: 0.42, y: 0.55, note: 'tram window, +2 more points along the tram', extra: [[0.55, 0.55, 0], [0.65, 0.58, 0]] },
        ],
    },
    // Everything that makes a correction sequence different from one click:
    // negatives that carve, positives that grow across a boundary, and the
    // same object approached three times.
    sequences: {
        url: '/.cache/testimg/d750-lossless.nef',
        clicks: [
            { x: 0.50, y: 0.50, note: 'ball, then EXCLUDE a rivet inside it', extra: [[0.44, 0.44, 1]] },
            { x: 0.50, y: 0.50, note: 'ball, then grow onto the cloth', extra: [[0.15, 0.70, 0]] },
            { x: 0.50, y: 0.47, note: 'rivet ×3 (accumulating on one subject)', extra: [[0.51, 0.48, 0], [0.49, 0.46, 0]] },
            { x: 0.12, y: 0.20, note: 'cloth, then EXCLUDE the ball', extra: [[0.50, 0.50, 1]] },
        ],
    },
    // Crowd: five people touching, dark clothing on dark, a thin lanyard.
    crowd: {
        url: '/.cache/testimg/DSC_0221.NEF',
        clicks: [
            { x: 0.60, y: 0.55, note: 'centre person among touching people' },
            { x: 0.40, y: 0.45, note: 'black shirt (dark on dark neighbour)' },
            { x: 0.57, y: 0.66, note: 'lanyard badge (small, on a person)' },
            { x: 0.18, y: 0.55, note: 'edge person, cropped by the frame' },
            { x: 0.93, y: 0.10, note: 'blue wall panel (flat background)' },
        ],
    },
}

const args = process.argv.slice(2)
const wanted = args.filter((a) => !a.startsWith('-'))
const keys = wanted.length ? wanted : Object.keys(SCENES)
// --shot writes stage/<scene>-<n>.png per click, so a stray component can be
// looked at instead of guessed about.
const shotDir = args.includes('--shot') ? '.cache/leak-shots' : null
if (shotDir) (await import('node:fs/promises')).mkdir(shotDir, { recursive: true }).catch(() => {})

// Same browser verify.mjs proves the phases in: the installed Google Chrome,
// with Chrome-for-Testing only as the fallback (`--cft`). A leak measured on
// one WebGPU/ORT revision says nothing certain about another.
const channel = resolveChannel()
console.log(`[leak] ${channelLabel(channel)}`)
const ctx = await chromium.launchPersistentContext(profileDir('.cache/profile', channel), onChannel({
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    viewport: { width: 1512, height: 900 },
}, channel))
const page = ctx.pages()[0] ?? await ctx.newPage()
// restoreSession() runs at module eval, so the flag has to be in place before
// the document does — an evaluate() after goto is a race the last session's
// photo can win, and the clicks below then land on the wrong photo.
await page.addInitScript(() => { window.__seglabNoRestore = true })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))

const gap = (a, b) => Math.max(
    0, Math.max(a[0] - b[2], b[0] - a[2]),
    Math.max(a[1] - b[3], b[1] - a[3]),
)

const rows = []
for (const key of keys) {
    const scene = SCENES[key]
    if (!scene) { console.log(`skip ${key}: no such scene`); continue }
    await page.goto('http://127.0.0.1:8788/', { waitUntil: 'load' })
    const loaded = await page.evaluate(async (url) => {
        const blob = await (await fetch(url)).blob()
        const name = url.split('/').pop()
        const dt = new DataTransfer()
        const type = /\.jpe?g$/i.test(name) ? 'image/jpeg'
            : /\.png$/i.test(name) ? 'image/png' : 'image/x-raw'
        dt.items.add(new File([blob], name, { type }))
        const el = document.getElementById('file')
        el.files = dt.files
        el.dispatchEvent(new Event('change', { bubbles: true }))
        return blob.size
    }, scene.url)
    await page.waitForFunction(
        () => document.getElementById('stage')?.classList.contains('visible'),
        { timeout: 240000 },
    )
    await page.waitForFunction(() => window.__seglab?.state().ready, { timeout: 240000 })
    await page.waitForTimeout(1200)

    const dims = await page.evaluate(() => {
        const v = document.getElementById('view')
        return { w: v.width, h: v.height }
    })
    console.log(`\n=== ${key}  ${scene.url}  ${(loaded / 1e6).toFixed(1)} MB  proxy ${dims.w}x${dims.h} ===`)

    for (const c of scene.clicks) {
        const x = Math.round(c.x * dims.w)
        const y = Math.round(c.y * dims.h)
        await page.evaluate(() => window.__seglab.reset())
        await page.waitForTimeout(120)
        // `extra` is what a real correction looks like: the first click, then
        // more points on the same object. Hygiene runs again on every one of
        // them, so a rule that is safe on click 1 still has to be safe on 3.
        const extra = (c.extra || []).map(([fx, fy, neg]) => [
            Math.round(fx * dims.w), Math.round(fy * dims.h), neg ? 1 : 0,
        ])
        const out = await page.evaluate(async ([px, py, more]) => {
            await window.__seglab.clickAt(px, py)
            for (const [mx, my, neg] of more) await window.__seglab.clickAt(mx, my, !!neg)
            return { ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }
        }, [x, y, extra])

        // EVERY clicked component is an anchor. Taking only the first reads a
        // deliberate two-object selection as a 4.6%-anchor catastrophe, and
        // separation has to be measured against the nearest anchor anyway —
        // that is the rule's own test.
        const regions = out.regions || []
        const anchors = regions.filter((r) => r.clicked)
        if (!anchors.length && regions[0]) anchors.push(regions[0])
        const sepOf = (r) => Math.min(...anchors.map((a) => gap(r.box, a.box)
            / Math.max(1, Math.max(a.box[2] - a.box[0], a.box[3] - a.box[1]))))
        const strays = regions.filter((r) => !anchors.includes(r)).map((r) => ({
            area: r.area,
            share: +(r.share * 100).toFixed(2),
            sep: +sepOf(r).toFixed(3),
            solidity: +r.solidity.toFixed(3),
            box: r.box,
        }))
        const anchorArea = anchors.reduce((a, r) => a + r.area, 0)
        const total = regions.reduce((a, r) => a + r.area, 0)
        const row = {
            scene: key,
            note: c.note,
            at: [x, y],
            cover: out.maskSummary?.coverage ?? null,
            score: out.score ?? null,
            components: out.components,
            anchors: anchors.length,
            anchorShare: total ? +((anchorArea / total) * 100).toFixed(1) : null,
            anchorSolidity: anchors[0] ? +anchors[0].solidity.toFixed(3) : null,
            strays: strays.slice(0, 6),
            strayPixels: strays.reduce((a, s) => a + s.area, 0),
        }
        rows.push(row)
        if (shotDir && strays.length) {
            const n = scene.clicks.indexOf(c)
            await page.locator('#stage').screenshot({ path: `${shotDir}/${key}-${n}.png` })
        }
        const worst = strays.slice(0, 3)
            .map((s) => `${s.area}px sep${s.sep} sol${s.solidity} @${s.box.join(',')}`)
            .join(' | ') || '—'
        // A click outside the live mask COMMITS the finished object and starts a
        // new one, so the composite is several masks and only the last carries
        // clicks. Without this, a deliberate multi-object selection reads as a
        // 10%-anchor catastrophe — and hygiene never saw the composite at all.
        row.baseOps = out.baseOps
        console.log(
            `${row.baseOps ? `+${row.baseOps}obj ` : ''}`
            + `${String(row.components).padStart(3)} comp  anchor ${String(row.anchorShare).padStart(5)}%`
            + ` sol ${String(row.anchorSolidity).padStart(5)}  stray ${String(row.strayPixels).padStart(6)}px`
            + `  ${c.note}\n        ${worst}`,
        )
    }
}

console.log(`\nerrors ${errors.length}`, errors.slice(0, 3))
console.log('\nJSON', JSON.stringify(rows))
await ctx.close()

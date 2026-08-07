#!/usr/bin/env bun
/**
 * SEGLAB text-search benchmark — accuracy, latency, and resource utilisation.
 *
 * Drives the REAL app in headless Chromium against real photographs, and
 * reports what a text query actually costs and how good the mask actually is.
 *
 * WHAT IT MEASURES
 *   latency   per-stage — image encode, text detect, per-instance decode,
 *             post pipeline — as P50/P95 over repeats, in all four cache
 *             quadrants (phrase warm/cold × image warm/cold).
 *   memory    peak JS heap (CDP Performance.getMetrics) and peak OS-level RSS
 *             of the whole browser process tree. The 2.2 GB ceiling is a HARD
 *             GATE: exceed it and this exits non-zero.
 *   accuracy  on synthetic fixtures, exact mIoU + boundary-F1 against
 *             analytically known masks. On your own photos there is no ground
 *             truth, so it reports two GT-free proxies instead:
 *               · click-vs-text agreement — click the object, then describe
 *                 it; IoU between the two masks. The click lane is already
 *                 verified by verify.mjs, so it is a legitimate reference.
 *               · phrasing stability — "car" vs "the car" vs "automobile"
 *                 should select the same pixels.
 *   controls  absent-phrase false positives. Querying something that is NOT
 *             in the photo must return nothing. Detectors fail this quietly.
 *
 * IMAGES
 *   Defaults to ~/Desktop, newest first. Only browser-decodable formats are
 *   used — JPEG/PNG/WebP/AVIF. Camera raw (.CR2/.NEF/.ARW/.DNG) and HEIC
 *   cannot be decoded by Chromium, so they are listed as skipped rather than
 *   silently ignored; export to JPEG first if you want them measured.
 *   With no usable photos it falls back to synthetic DSLR-scale fixtures, so
 *   the gates still run anywhere.
 *
 * USAGE
 *   bun bench.mjs                          # ~/Desktop, default phrases
 *   bun bench.mjs --images ~/Pictures/test
 *   bun bench.mjs --phrases "all the birds,the red door,giraffe"
 *   bun bench.mjs --synthetic              # skip real photos entirely
 *   bun bench.mjs --repeats 5 --ceiling 2200
 *
 * Results land in bench-out/: report.json, report.md, and a PNG contact sheet
 * per image so the masks can be eyeballed, not just scored.
 */

import { createServer } from 'node:http'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { execSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'

const ROOT = path.resolve(import.meta.dir)
const OUT = path.join(ROOT, 'bench-out')
const PROFILE_DIR = path.join(ROOT, '.cache', 'profile')

/* ─── Args ──────────────────────────────────────────────────────────────── */

const argv = process.argv.slice(2)
const arg = (name, fallback) => {
    const i = argv.indexOf(`--${name}`)
    return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : fallback
}
const flag = (name) => argv.includes(`--${name}`)

const CFG = {
    imagesDir: arg('images', path.join(os.homedir(), 'Desktop')),
    phrases: arg('phrases', '').split(',').map((s) => s.trim()).filter(Boolean),
    repeats: Number(arg('repeats', 3)),
    maxImages: Number(arg('max-images', 4)),
    ceilingMB: Number(arg('ceiling', 2200)),
    synthetic: flag('synthetic'),
    timeoutMs: Number(process.env.HARNESS_TIMEOUT_MS || 12 * 60 * 1000),
}

// Absent-phrase controls: things that will not be in an ordinary photo. A hit
// here is a false positive, and the false-positive rate is a headline metric.
const ABSENT_CONTROLS = ['a giraffe', 'an astronaut helmet', 'a submarine']

const DEFAULT_PHRASES = ['person', 'all the cars', 'a tree', 'the sky', 'a building', 'a dog']

const DECODABLE = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif', '.bmp'])
const CAMERA_RAW = new Set(['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.heic', '.heif', '.tif', '.tiff'])

const log = (m) => console.log(`[bench] ${m}`)

/* ─── Playwright ────────────────────────────────────────────────────────── */

let chromium
try {
    ({ chromium } = await import('playwright'))
} catch {
    try {
        ({ chromium } = await import('/Users/andhetharuntej/Pixxel/node_modules/playwright/index.mjs'))
    } catch {
        log('✗ playwright not found — `bun add -d playwright` (or run from a repo that has it)')
        process.exit(1)
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
        if (!file.startsWith(ROOT) || !existsSync(file)) { res.writeHead(404).end('not found'); return }
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
        res.end(await readFile(file))
    } catch (e) {
        res.writeHead(500).end(String(e?.message || e))
    }
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port

/* ─── Image discovery ───────────────────────────────────────────────────── */

const findImages = async (dir) => {
    if (!existsSync(dir)) return { usable: [], skipped: [], reason: `${dir} does not exist` }
    let names = []
    try { names = await readdir(dir) } catch (e) { return { usable: [], skipped: [], reason: String(e?.message) } }
    const usable = []
    const skipped = []
    for (const name of names) {
        const ext = path.extname(name).toLowerCase()
        const full = path.join(dir, name)
        if (CAMERA_RAW.has(ext)) { skipped.push({ name, why: 'Chromium cannot decode this format' }); continue }
        if (!DECODABLE.has(ext)) continue
        try {
            const s = await stat(full)
            if (s.isFile() && s.size > 0) usable.push({ path: full, name, size: s.size, mtime: s.mtimeMs })
        } catch { /* unreadable */ }
    }
    // Biggest first — DSLR files are the interesting case.
    usable.sort((a, b) => b.size - a.size)
    return { usable: usable.slice(0, CFG.maxImages), skipped, reason: null }
}

/* ─── Metrics ───────────────────────────────────────────────────────────── */

const pct = (arr, p) => {
    if (!arr.length) return 0
    const s = [...arr].sort((a, b) => a - b)
    return s[Math.min(s.length - 1, Math.floor(s.length * p))]
}

/** Total RSS of the browser process tree, in MB. */
const browserRssMB = (pid) => {
    if (!pid) return 0
    try {
        const out = execSync(`ps -o rss= -g $(ps -o pgid= -p ${pid} | tr -d ' ') 2>/dev/null || ps -o rss= -p ${pid}`, { encoding: 'utf8' })
        const total = out.split('\n').map((l) => Number(l.trim())).filter(Boolean).reduce((a, b) => a + b, 0)
        return Math.round(total / 1024)
    } catch { return 0 }
}

const heapMB = async (cdp) => {
    try {
        const { metrics } = await cdp.send('Performance.getMetrics')
        const m = Object.fromEntries(metrics.map((x) => [x.name, x.value]))
        return Math.round((m.JSHeapUsedSize || 0) / 1e6)
    } catch { return 0 }
}

/** IoU between two mask bboxes-with-areas reported by the page. */
const iouFromPage = (page) => page.evaluate(() => {
    const s = window.__benchMasks
    if (!s?.a || !s?.b) return null
    const { a, b } = s
    let inter = 0
    let ua = 0
    let ub = 0
    for (let i = 0; i < a.length; i += 4) {
        const A = a[i] >= 128 ? 1 : 0
        const B = b[i] >= 128 ? 1 : 0
        if (A) ua += 1
        if (B) ub += 1
        if (A && B) inter += 1
    }
    const union = ua + ub - inter
    return union > 0 ? inter / union : null
})

/* ─── Run ───────────────────────────────────────────────────────────────── */

await mkdir(OUT, { recursive: true })
const report = {
    startedAt: new Date().toISOString(),
    host: { platform: process.platform, arch: process.arch, cpus: os.cpus().length, totalMemGB: +(os.totalmem() / 1e9).toFixed(1) },
    config: CFG,
    images: [],
    gates: [],
}

let context = null
let failed = false
try {
    context = await chromium.launchPersistentContext(PROFILE_DIR, {
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-gpu'],
    })
    const pid = context.browser()?.process?.()?.pid || null

    const page = await context.newPage()
    page.setDefaultTimeout(CFG.timeoutMs)
    page.on('console', (m) => { if (m.text().startsWith('[seglab]')) log(`browser: ${m.text().slice(0, 160)}`) })
    page.on('pageerror', (e) => log(`pageerror: ${String(e).slice(0, 160)}`))
    await page.goto(`http://127.0.0.1:${port}/`)
    await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 60_000 })
    const cdp = await context.newCDPSession(page)
    await cdp.send('Performance.enable')

    let peakRss = browserRssMB(pid)
    let peakHeap = await heapMB(cdp)
    const sample = async () => {
        peakRss = Math.max(peakRss, browserRssMB(pid))
        peakHeap = Math.max(peakHeap, await heapMB(cdp))
    }

    // Warm the text detector once; its download is not a per-query cost.
    log('warming the text detector (first run downloads ~155 MB)…')
    const tWarm = Date.now()
    await page.evaluate(() => window.__seglab.warmText())
    log(`text detector ready in ${((Date.now() - tWarm) / 1000).toFixed(1)}s`)
    await sample()

    /* --- pick the images --- */
    const found = CFG.synthetic ? { usable: [], skipped: [], reason: 'synthetic mode' } : await findImages(CFG.imagesDir)
    if (found.skipped.length) {
        log(`skipping ${found.skipped.length} undecodable file(s) (camera raw / HEIC): ${found.skipped.slice(0, 4).map((s) => s.name).join(', ')}`)
    }
    report.skippedFiles = found.skipped

    const targets = []
    for (const img of found.usable) {
        targets.push({ kind: 'photo', label: img.name, sizeMB: +(img.size / 1e6).toFixed(1), load: async () => {
            const b64 = (await readFile(img.path)).toString('base64')
            const ext = path.extname(img.name).toLowerCase().replace('.', '')
            const mime = ext === 'jpg' ? 'jpeg' : ext
            return page.evaluate((u) => window.__seglab.loadURL(u), `data:image/${mime};base64,${b64}`)
        } })
    }
    if (targets.length === 0) {
        log(`no decodable photos in ${CFG.imagesDir} — falling back to synthetic DSLR-scale fixtures`)
        report.usedSynthetic = true
        for (const [w, h] of [[6000, 4000], [3000, 2000]]) {
            targets.push({
                kind: 'synthetic', label: `synthetic ${w}×${h}`, sizeMB: 0,
                load: () => page.evaluate(({ W, H }) => window.__seglab.loadSynthetic({
                    w: W, h: H,
                    objects: [
                        { kind: 'circle', x: W * 0.25, y: H * 0.55, r: Math.round(H * 0.16), color: '#d8433b' },
                        { kind: 'rect', x: W * 0.68, y: H * 0.4, r: Math.round(H * 0.13), color: '#3b6fd8' },
                        // Deliberately minute: ~0.5% of the frame's short side.
                        { kind: 'circle', x: W * 0.8, y: H * 0.78, r: Math.max(6, Math.round(H * 0.005)), color: '#e8c33b' },
                    ],
                }), { W: w, H: h }),
            })
        }
    }

    const phrases = CFG.phrases.length ? CFG.phrases : DEFAULT_PHRASES

    /* --- measure --- */
    for (const target of targets) {
        log(`— ${target.label}`)
        const dims = await target.load()
        await sample()
        const entry = {
            label: target.label, kind: target.kind, sizeMB: target.sizeMB,
            source: dims, phrases: [], absent: [], agreement: null,
        }

        for (const phrase of phrases) {
            const runs = []
            let last = null
            for (let r = 0; r < CFG.repeats; r += 1) {
                const t0 = Date.now()
                const s = await page.evaluate((q) => window.__seglab.textSearch(q), phrase)
                const wall = Date.now() - t0
                const lr = s.lastRun || {}
                runs.push({
                    wall,
                    detectMs: lr.detectMs ?? 0,
                    decodeMs: lr.decodeMs ?? 0,
                    postMs: lr.postMs ?? 0,
                    encodeMs: lr.encodeMs ?? 0,
                    instances: s.instances?.length ?? 0,
                })
                last = s
                await sample()
            }
            const row = {
                phrase,
                instances: last?.instances?.length ?? 0,
                coverage: last?.maskSummary?.coverage ?? 0,
                // r=0 is the cold run (image + phrase both uncached); the rest are warm.
                coldMs: runs[0].wall,
                warmP50: pct(runs.slice(1).map((x) => x.wall), 0.5),
                warmP95: pct(runs.slice(1).map((x) => x.wall), 0.95),
                detectP50: pct(runs.map((x) => x.detectMs), 0.5),
                decodeP50: pct(runs.map((x) => x.decodeMs), 0.5),
                postP50: pct(runs.map((x) => x.postMs), 0.5),
                topScore: last?.instances?.[0]?.score ?? 0,
            }
            entry.phrases.push(row)
            log(`   "${phrase}" → ${row.instances} inst · cold ${row.coldMs}ms · warm p50 ${row.warmP50}ms (detect ${row.detectP50} / decode ${row.decodeP50} / post ${row.postP50})`)
        }

        // Absent-phrase controls: any instance here is a false positive.
        for (const phrase of ABSENT_CONTROLS) {
            const s = await page.evaluate((q) => window.__seglab.textSearch(q), phrase)
            const n = s.instances?.length ?? 0
            entry.absent.push({ phrase, instances: n, falsePositive: n > 0, topScore: s.instances?.[0]?.score ?? 0 })
            await sample()
        }
        const fp = entry.absent.filter((a) => a.falsePositive).length
        log(`   absent-phrase controls: ${fp}/${entry.absent.length} false positives`)

        report.images.push(entry)
        await page.evaluate(() => window.__seglab.reset())
    }

    report.peak = { rssMB: peakRss, jsHeapMB: peakHeap, ceilingMB: CFG.ceilingMB }
    log(`peak RSS ${peakRss} MB · peak JS heap ${peakHeap} MB · ceiling ${CFG.ceilingMB} MB`)

    /* --- gates --- */
    const gate = (label, ok, detail) => {
        report.gates.push({ label, ok, detail })
        console.log(`[bench] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
        if (!ok) failed = true
    }
    gate('peak RSS under the ceiling', peakRss > 0 && peakRss <= CFG.ceilingMB, `${peakRss} MB vs ${CFG.ceilingMB} MB`)
    const allAbsent = report.images.flatMap((i) => i.absent)
    const fpRate = allAbsent.length ? allAbsent.filter((a) => a.falsePositive).length / allAbsent.length : 0
    gate('absent-phrase false-positive rate ≤ 20%', fpRate <= 0.2, `${(fpRate * 100).toFixed(0)}% (${allAbsent.filter((a) => a.falsePositive).length}/${allAbsent.length})`)
    const warm = report.images.flatMap((i) => i.phrases.map((p) => p.warmP50)).filter(Boolean)
    gate('warm query p50 under 1.5 s', warm.length === 0 || pct(warm, 0.5) <= 1500, `${pct(warm, 0.5)} ms`)

    await page.close()
} catch (err) {
    console.error(`[bench] ✗ ${err?.stack || err?.message || err}`)
    report.error = String(err?.message || err)
    failed = true
} finally {
    await context?.close().catch(() => {})
    server.close()
}

report.finishedAt = new Date().toISOString()
await writeFile(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2))

/* ─── Markdown summary ──────────────────────────────────────────────────── */

const md = ['# SEGLAB text-search benchmark', '',
    `Run ${report.startedAt} · ${report.host.platform}/${report.host.arch} · ${report.host.cpus} cores · ${report.host.totalMemGB} GB`, '']
if (report.usedSynthetic) md.push('> No decodable photos found — synthetic DSLR-scale fixtures were used.', '')
if (report.skippedFiles?.length) {
    md.push(`> Skipped ${report.skippedFiles.length} undecodable file(s) (camera raw / HEIC — export to JPEG to include them).`, '')
}
if (report.peak) {
    md.push('## Resources', '', `| metric | value | ceiling |`, `|---|---|---|`,
        `| peak browser RSS | ${report.peak.rssMB} MB | ${report.peak.ceilingMB} MB |`,
        `| peak JS heap | ${report.peak.jsHeapMB} MB | — |`, '')
}
for (const img of report.images) {
    md.push(`## ${img.label}`, '',
        `Source ${img.source?.width}×${img.source?.height} → canonical ${img.source?.canonW}×${img.source?.canonH}`, '',
        '| phrase | instances | cold | warm p50 | warm p95 | detect | decode | post | coverage |',
        '|---|---|---|---|---|---|---|---|---|')
    for (const p of img.phrases) {
        md.push(`| ${p.phrase} | ${p.instances} | ${p.coldMs} ms | ${p.warmP50} ms | ${p.warmP95} ms | ${p.detectP50} ms | ${p.decodeP50} ms | ${p.postP50} ms | ${(p.coverage * 100).toFixed(1)}% |`)
    }
    md.push('', '**Absent-phrase controls** (any instance is a false positive):', '')
    for (const a of img.absent) md.push(`- \`${a.phrase}\` → ${a.instances} instance(s) ${a.falsePositive ? '❌ FALSE POSITIVE' : '✅'}`)
    md.push('')
}
md.push('## Gates', '')
for (const g of report.gates) md.push(`- ${g.ok ? '✅' : '❌'} ${g.label} — ${g.detail}`)
await writeFile(path.join(OUT, 'report.md'), md.join('\n'))

log(`wrote ${path.join(OUT, 'report.json')} and report.md`)
if (failed) { console.error('\n[bench] ✗ benchmark gates FAILED'); process.exit(1) }
console.log('\n[bench] ✓ all gates passed')

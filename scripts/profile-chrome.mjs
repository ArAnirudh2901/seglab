#!/usr/bin/env node
/**
 * Real-Chrome memory-profiled drive of seglab.
 * Scenario 'gpu' (default): WebGPU lanes, auto tier.
 * Scenario 'wasm': --disable-gpu + ?force=wasm — the GPU-less low-end device.
 *
 * Flow: boot → import 51 MP JPEG (real upload path) → eager-encode settle →
 * click cat #1 (encode+decode) → click cat #2 (cached decode) →
 * repeat search (frame cache) → YOLO-World search (int8 on wasm) → cutout PNG.
 * Memory: usedJSHeapSize @250 ms + measureUserAgentSpecificMemory @2.5 s, maxima kept.
 * Verdict vs the 8 GB-class budgets (lite/standard8 memBudgetMB 1800/1900).
 */
import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { chromium } from 'playwright'

const SCENARIO = process.argv[2] === 'wasm' ? 'wasm' : 'gpu'
const ROOT = new URL('..', import.meta.url).pathname
const OUT = `/Users/andhetharuntej/seglab/.cache/chrome-${SCENARIO}`
const URL_BASE = 'http://127.0.0.1:8788/'
const FIXTURE = '/models/test-fixtures/dslr-45mp.jpg'

const now = () => performance.now()
const ms = (t) => Math.round(t)

const waitPort = async (url, tries = 60) => {
    for (let i = 0; i < tries; i += 1) {
        try { const r = await fetch(url, { method: 'HEAD' }); if (r.ok || r.status === 404) return } catch { /* not up */ }
        await new Promise((r) => setTimeout(r, 250))
    }
    throw new Error('dev server did not come up')
}

const server = spawn('node', ['scripts/dev-server.mjs'], { cwd: ROOT, stdio: 'ignore' })
try {
    await waitPort(URL_BASE)
    const report = { scenario: SCENARIO, milestones: {}, timings: {}, checks: {}, console: [] }
    const args = SCENARIO === 'wasm' ? ['--disable-gpu'] : []
    const browser = await chromium.launch({ channel: 'chrome', headless: true, args })
    const page = await browser.newPage()
    page.on('console', (m) => { const t = m.text(); if (/seglab|error/i.test(t)) report.console.push(t.slice(0, 300)) })
    page.on('pageerror', (e) => report.console.push(`PAGEERROR ${String(e).slice(0, 200)}`))

    await page.addInitScript(() => {
        window.__memlog = { jsHeapMax: 0, uasMax: 0, uasLast: 0, samples: 0 }
        setInterval(() => {
            const h = performance.memory?.usedJSHeapSize || 0
            if (h > window.__memlog.jsHeapMax) window.__memlog.jsHeapMax = h
        }, 250)
        const uas = () => {
            if (!crossOriginIsolated || !performance.measureUserAgentSpecificMemory) return
            performance.measureUserAgentSpecificMemory().then((m) => {
                window.__memlog.uasLast = m.bytes
                window.__memlog.samples += 1
                if (m.bytes > window.__memlog.uasMax) window.__memlog.uasMax = m.bytes
            }).catch(() => { })
        }
        setInterval(uas, 2500)
        setTimeout(uas, 500)
    })

    const snap = async (label) => {
        const s = await page.evaluate(async () => {
            let uasNow = null
            if (crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
                try { uasNow = (await performance.measureUserAgentSpecificMemory()).bytes } catch { /* rate */ }
            }
            return { ...window.__memlog, uasNow, jsHeapNow: performance.memory?.usedJSHeapSize || 0 }
        })
        report.milestones[label] = Object.fromEntries(Object.entries(s).map(([k, v]) => [k, typeof v === 'number' && k !== 'samples' ? Math.round(v / 1e6) : v]))
        return s
    }

    await page.goto(URL_BASE + (SCENARIO === 'wasm' ? '?force=wasm' : ''))
    await page.waitForFunction(() => window.__seglab, null, { timeout: 30000 })
    report.budget = await page.evaluate(async () => {
        const b = await window.__seglab.resourceBudget()
        return { profile: b.profile, source: b.profileSource, samWebGPU: b.samWebGPU, gpuTier: b.gpuTier, memBudgetMB: b.memBudgetMB, proxyMax: b.proxyMax, detectorScale: b.detectorScale, forceWasm: !!b.forceWasm }
    })
    await snap('boot')

    // Import through the REAL upload path.
    let t = now()
    const imp = await page.evaluate((u) => window.__seglab.importUrl(u), FIXTURE)
    report.timings.importMs = ms(now() - t)
    report.checks.import = imp
    report.transform = await page.evaluate(() => {
        const tf = window.__seglab.imageTransform()
        return { originalW: tf.originalW, originalH: tf.originalH, proxyW: tf.proxyW, proxyH: tf.proxyH, proxyActive: tf.proxyActive, working: !!tf.workingBlob || tf.workingActive || null }
    })
    t = now()
    const eager = await page.evaluate(() => window.__seglab.eagerEncode())
    report.timings.eagerSettleMs = ms(now() - t)
    report.checks.eagerEncode = eager ? { encoded: !!eager.encoded, skipped: eager.skipped || null } : null
    await snap('after-import')

    // Click cat #1 (left) — pays encode on first selection when eager skipped.
    const { proxyW, proxyH } = report.transform
    t = now()
    const c1 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: proxyW * 0.28, y: proxyH * 0.62 })
    report.timings.click1Ms = ms(now() - t)
    report.checks.click1 = { stale: !!c1?.stale, mask: await page.evaluate(() => window.__seglab.maskStats()) }
    await snap('after-click1')

    // Click cat #2 — embedding cached, decode-only.
    await page.evaluate(() => window.__seglab.reset())
    t = now()
    await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: proxyW * 0.75, y: proxyH * 0.42 })
    report.timings.click2Ms = ms(now() - t)
    report.checks.click2 = { mask: await page.evaluate(() => window.__seglab.maskStats()) }
    await snap('after-click2')

    // Cutout PNG through the real button.
    const dl = page.waitForEvent('download', { timeout: 60000 })
    await page.click('#cutout')
    const download = await dl
    const path = `${OUT}-cutout.png`
    await download.saveAs(path)
    report.checks.cutout = { file: path }
    await snap('final')

    // Measured-headroom climb: let the governor run real cycles (~15 s covers
    // 4+ decision cycles with real measureUserAgentSpecificMemory samples). On
    // a healthy GPU device this should raise standard8 → standard.
    await page.waitForTimeout(15000)
    report.checks.climb = await page.evaluate(async () => {
        const b = await window.__seglab.resourceBudget()
        return { profile: b.profile, source: b.profileSource, exportFullRes: b.exportFullRes, headroomFires: window.__seglabHeadroom || 0 }
    })
    await snap('after-climb-window')

    // Pressure-3 release: on the wasm lane the SAM worker must be recycled,
    // returning its grown wasm Memory to the OS — resident should collapse.
    report.checks.released = await page.evaluate(() => window.__seglab.releaseMemory())
    await page.waitForTimeout(5000)
    await snap('after-release')

    const verdictBudget = report.budget.memBudgetMB || 1900
    report.verdict = {
        uasMaxMB: report.milestones.final?.uasMax ?? null,
        jsHeapMaxMB: report.milestones.final?.jsHeapMax ?? null,
        budgetMB: verdictBudget,
        withinBudget: (report.milestones.final?.uasMax ?? 0) <= verdictBudget,
    }
    writeFileSync(`${OUT}-report.json`, JSON.stringify(report, null, 2))
    console.log(JSON.stringify({ scenario: SCENARIO, budget: report.budget, timings: report.timings, verdict: report.verdict, climb: report.checks.climb, residentAfterRelease: report.milestones['after-release'], released: report.checks.released, checks: { eager: report.checks.eagerEncode, click1cov: report.checks.click1?.mask?.coverage, click2cov: report.checks.click2?.mask?.coverage } }, null, 1))
    await browser.close()
} finally {
    server.kill()
}

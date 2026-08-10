#!/usr/bin/env bun
/**
 * Headless-browser verification of SEGLAB end-to-end.
 *
 * Serves this directory, drives the REAL app (index.html + workers + models
 * via CDN) with Playwright Chromium through the window.__seglab test hooks.
 *
 * Suites:
 *   T  — pure logic (text-core, capability, policy, sizing)
 *   Q  — heavy-job queue contracts (node-side, no browser)
 *   S  — memory-contract static source scans
 *   A  — lite (memory-locked) browser phases: 1024 proxy, unsafe-flag lockout,
 *        one embedding, no OPFS persistence, revision/cancel, export caps,
 *        wasm cv-refine gates, decode/model serialization
 *   H  — trusted-host (Phosmith pro) phases: HD export, escalation, working
 *        copy, OPFS revisit — the paths a locked browser never reaches
 *   O  — offline zero-cloud proof
 *   V  — vendored cold start: fresh cache-less browser, both CDNs blocked
 *   P  — power-cut persistence: durable while open, restored on reopen
 *   A11 — hardening: fallback, no-storage, concurrency, malformed uploads
 *   A12 — a browser with no WebGPU at all
 *
 * Two rules used to live in comments and cost a session each. They are
 * enforced now (scripts/harness/):
 *
 *   One run per machine. The phases below share ONE Chromium profile, and a
 *   second run launching onto it blocks inside Chrome's singleton handshake
 *   forever. A lease guards it — taken lazily at the browser boundary, so
 *   node-only phases never contend — and contention fails fast with a verdict
 *   naming the holder and its phase. It is the only lock here: one lock cannot
 *   form a cycle, and every wait on it has a deadline.
 *
 *   Order is a contract, not a convention. PLAN below is data: a phase that
 *   stands up its own browser declares the headroom it needs, and the gate
 *   reclaims the ~1.2 GB instance and checks that floor before letting it
 *   start. Enter a phase out of plan order and it throws. Every phase is
 *   time-boxed, so an exhausted machine fails with a diagnosis instead of
 *   hanging on a cold encode that never returns.
 *
 * Usage: bun verify.mjs [--fast] [--wait=10m] [--isolated] [--strict]
 *                       [--list] [--force] [--no-reap]
 *   --fast      node-only phases (T/Q/S); no browser, no lease, safe in parallel
 *   --wait=10m  queue behind a running suite instead of failing (bounded)
 *   --isolated  throwaway profile: a genuinely parallel run, cold cache
 *   --strict    a phase skipped for lack of headroom fails the run
 *   --list      print the plan and exit
 */

import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync, readFileSync, rmSync } from 'node:fs'
import path from 'node:path'
import {
  classifyPixelColor, clusterObjects, colorEvidenceForBox, degenerateScores, DETECTOR_INPUT, dominantColorForBox, letterboxPlan, normalizePhrase, nms, pruneContainers, rankDetections, scaleBox, shrinkFactor, tilePlans, unletterboxBox, YOLOE_INPUT,
} from './js/text-core.js'
import {
  buildFacets, expandQuery, labelMatchesQuery, regionOf, suggest,
} from './js/search-taxonomy.js'
import { classifyCapability, probeTextLane } from './js/capability.js'
import { applyMemoryPressure, resolveBudget, PROFILE_PRESETS } from './js/policy.js'
import { decidePressure } from './js/memory-governor.js'
import { boxFraction, chooseCandidate, cleanRegions, fieldArea, promptFit, stabilityScore } from './js/mask-select.js'
import { refineField } from './js/mask-refine.js'
import { getBoundedProxySize, displayPlan, decodeBudgetMP, interactionPlan } from './js/proxy-plan.js'
import {
  composeChannels, maskChannelCoverages, maskToChannel, pickBestMask, pointInMask, RUNAWAY_COVERAGE,
} from './js/sam-core.js'
import { enqueueHeavy, cancelHeavyBefore, STALE, getHeavyQueueState } from './js/heavy-job-queue.js'
import { extractRawPreview } from './js/image-raw.js'
import { acquireLease, humanMs } from './scripts/harness/lease.mjs'
import { formatMem, memorySnapshot } from './scripts/harness/machine.mjs'
import { createRunner, definePlan, guardContext, withDeadline } from './scripts/harness/phases.mjs'

const ROOT = path.resolve(import.meta.dir)
const CACHE_DIR = path.join(ROOT, '.cache')
const TIMEOUT_MS = Number(process.env.HARNESS_TIMEOUT_MS || 8 * 60 * 1000)
const RAW_FIXTURE = process.env.RAW_FIXTURE || ''
const LAUNCH_MS = 90_000

/* ─── Options ───────────────────────────────────────────────────────────── */
const argv = process.argv.slice(2)
const flag = (name) => argv.some((a) => a === `--${name}` || a.startsWith(`--${name}=`))
const value = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`))
  return hit ? hit.slice(name.length + 3) : fallback
}
const parseMs = (spec, fallback) => {
  const m = /^(\d+(?:\.\d+)?)(ms|s|m)?$/.exec(String(spec).trim())
  return m ? Number(m[1]) * ({ ms: 1, s: 1000, m: 60_000 }[m[2] || 's']) : fallback
}
const OPTS = {
  fast: flag('fast'),           // node-only phases — no browser, so no lease and no contention
  list: flag('list'),
  force: flag('force'),
  isolated: flag('isolated'),   // throwaway profile: parallel runs at the cost of a cold cache
  strict: flag('strict'),       // an unproven (skipped) phase fails the run
  reap: !flag('no-reap'),
  waitMs: flag('wait') ? parseMs(value('wait', '10m'), 10 * 60_000) : 0,
}
const RUN_TAG = `${process.pid.toString(36)}${Date.now().toString(36).slice(-4)}`
const PROFILE_DIR = OPTS.isolated
  ? path.join(CACHE_DIR, 'profiles', RUN_TAG)
  : path.join(CACHE_DIR, 'profile')
const LOCK_PATH = path.join(CACHE_DIR, 'verify.lock')

const log = (msg) => console.log(`[verify] ${msg}`)
const results = []
const check = (label, ok, detail) => {
  results.push({ label, ok })
  console.log(`[verify] ${ok ? 'ok' : '✗'} ${label} — ${detail}`)
}

let context = null
let failed = false

/* ─── Phase plan ─────────────────────────────────────────────────────────
   The order below is the contract, not a convention. `after:` carries the rule
   that used to live in a comment: a phase standing up its OWN browser needs a
   machine that an exhausting phase has not flattened yet. A plan that breaks it
   refuses to start; code entering phases out of this order throws on the spot;
   and a phase that declares a floor gets it reclaimed and checked rather than
   trusting the order. Every phase is time-boxed — a hang fails, it never
   waits. */
const SCALE = Number(process.env.HARNESS_BUDGET_SCALE || 1)
const MIN_FREE_MB = Number(process.env.HARNESS_MIN_FREE_MB || 1600)
const mins = (n) => Math.round(n * 60_000 * SCALE)
const PLAN = definePlan([
  { id: 'T', title: 'pure logic', budgetMs: mins(4) },
  { id: 'Q', title: 'heavy-job queue', budgetMs: mins(3) },
  { id: 'S', title: 'static memory scans', budgetMs: mins(3) },
  { id: 'A', title: 'lite contract', budgetMs: mins(30) },
  { id: 'H', title: 'trusted host', budgetMs: mins(12) },
  { id: 'O', title: 'offline proof', budgetMs: mins(10) },
  { id: 'V', title: 'vendored cold start', budgetMs: mins(12), coldStart: true, reclaimFirst: true, needsMB: MIN_FREE_MB },
  { id: 'P', title: 'power-cut persistence', budgetMs: mins(10) },
  { id: 'A11', title: 'hardening', budgetMs: mins(25), exhausts: true, after: ['V', 'P'] },
  { id: 'A12', title: 'no-WebGPU browser', budgetMs: mins(10), coldStart: true, reclaimFirst: true, needsMB: MIN_FREE_MB, after: ['A11'] },
])
const RUN_BUDGET_MS = Number(process.env.HARNESS_RUN_BUDGET_MS)
  || PLAN.reduce((n, p) => n + p.budgetMs, 0) + mins(5)

if (OPTS.list) {
  log(`plan — run budget ${humanMs(RUN_BUDGET_MS)}, floor ${(MIN_FREE_MB / 1024).toFixed(1)} GB, host ${formatMem()}`)
  for (const p of PLAN) {
    log(`  ${p.id.padEnd(4)}${p.title.padEnd(24)}${humanMs(p.budgetMs).padStart(6)}`
      + `${p.coldStart ? `  cold-start ≥${(p.needsMB / 1024).toFixed(1)}GB` : ''}`
      + `${p.exhausts ? '  exhausts' : ''}${p.after ? `  after ${p.after.join(',')}` : ''}`)
  }
  process.exit(0)
}

let lease = null
const run = createRunner({
  plan: PLAN,
  results,
  log,
  lease: { setPhase: (p) => lease?.setPhase(p) },
  releaseSync: () => lease?.releaseSync(),
  globalBudgetMs: RUN_BUDGET_MS,
  strict: OPTS.strict,
  // Bounded, best-effort: the watchdog fires precisely when things are wedged,
  // so cleanup gets a deadline of its own and then the process goes.
  onWatchdog: async () => {
    await context?.close().catch(() => {})
    server.close()
  },
})

/* ─── Playwright (local dev dependency) ─────────────────────────────────── */
let chromium
try {
  ({ chromium } = await import('playwright'))
} catch (err) {
  if (process.env.CI_SKIP_BROWSER === '1') {
    log('skip — CI_SKIP_BROWSER=1')
    process.exit(0)
  }
  console.error('[verify] ✗ playwright not installed — run: bun add -d playwright && bunx playwright install chromium')
  console.error(`[verify]   (${err?.message})`)
  process.exit(1)
}

/* ─── Static server ─────────────────────────────────────────────────────── */
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
}
// Cross-origin isolation — must match the dev/prod servers so the suite runs
// under the same crossOriginIsolated + threaded-WASM conditions as production.
const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
}
const server = createServer(async (req, res) => {
  const head = (status, extra = {}) => res.writeHead(status, { ...ISOLATION_HEADERS, ...extra })
  try {
    const url = new URL(req.url, 'http://localhost')
    // Serve the optional RAW fixture to the browser suite (the develop worker
    // fetches it same-origin instead of shuttling megabytes over evaluate()).
    if (url.pathname === '/__raw_fixture' && RAW_FIXTURE && existsSync(RAW_FIXTURE)) {
      head(200, { 'Content-Type': 'application/octet-stream' })
      res.end(await readFile(RAW_FIXTURE))
      return
    }
    const rel = url.pathname === '/' ? '/index.html' : url.pathname
    const file = path.join(ROOT, path.normalize(rel))
    if (!file.startsWith(ROOT) || !existsSync(file)) {
      head(404).end('not found')
      return
    }
    head(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
    res.end(await readFile(file))
  } catch (e) {
    head(500).end(String(e?.message || e))
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const port = server.address().port
log(`serving SEGLAB on http://127.0.0.1:${port}`)

/* ─── Drive the app ─────────────────────────────────────────────────────── */
// Demo-scene ground truth in the 900×620 logical space (buildDemoScene).
const DISC = { x: 230, y: 340, r: 105 }
const SQUARE = { x: 615, y: 245, half: 95 }
const DOT = { x: 700, y: 480, r: 9 }
const FRAME = 900 * 620

// Trusted Phosmith budgets used by the H phases (a plain browser is locked
// to lite; these are the only route to standard/pro behaviour).
const HOST_PRO = { memoryBudgetGB: 16, vramGB: 12 }
const HOST_ULTRA = { memoryBudgetGB: 24, vramGB: 24, gpuName: 'RTX 4090' }

let pageSeq = 0
const newAppPage = async (context, query, longSide, { host = null, restore = false, pin = null } = {}) => {
  const page = await context.newPage()
  const tag = `p${++pageSeq}`
  // Synthetic phases must not inherit a session an earlier phase persisted.
  // Phase P opts back in — it is the gate that exercises restore.
  if (!restore) await page.addInitScript(() => { window.__seglabNoRestore = true })
  // Pin a profile (via the manual-override localStorage key) so a phase testing
  // a specific tier is deterministic regardless of the CI machine's autoTier.
  if (pin) await page.addInitScript((p) => { try { localStorage.setItem('seglab.profileOverride', p) } catch { /* storage off */ } }, pin)
  page.on('console', (msg) => {
    const text = msg.text()
    if (text.startsWith('[seglab]') || msg.type() === 'error') log(`browser[${tag}]: ${text.slice(0, 180)}`)
  })
  page.on('pageerror', (err) => log(`pageerror: ${String(err).slice(0, 180)}`))
  page.setDefaultTimeout(TIMEOUT_MS)
  if (host) await page.addInitScript((h) => { window.__PHOSMITH_DEVICE_RESOURCES__ = h }, host)
  await page.goto(`http://127.0.0.1:${port}/${query}`)
  await page.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  if (longSide !== null) await page.evaluate((ls) => window.__seglab.loadDemo(ls), longSide ?? undefined)
  return page
}

/** Disc-quality bar in whatever frame the page uses; coords pre-scaled. */
const checkDisc = (tag, s, cx, cy, discFrac) => {
  const b = s.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    `${tag}: click selects the disc`,
    b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3],
    `bbox [${b.map((v) => Math.round(v))}] vs centre (${cx.toFixed(0)},${cy.toFixed(0)})`,
  )
  check(
    `${tag}: disc mask is object-sized, not a flood`,
    s.maskSummary && s.maskSummary.coverage > discFrac * 0.4 && s.maskSummary.coverage < discFrac * 3,
    `coverage ${((s.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(discFrac * 100).toFixed(1)}%`,
  )
}
const DISC_FRAC = (Math.PI * DISC.r * DISC.r) / FRAME

try {
  /* ─── Phase T: pure logic (no browser) ──────────────────────────────── */
  await run.enter('T')
  const np = normalizePhrase('all red cars')
  check(
    'text-core: phrase → bare cores + multi intent',
    np && np.multi === true && np.color === 'red' && np.core === 'red car' && np.objectCore === 'car',
    `${JSON.stringify(np)}`,
  )
  const colorFrame = { data: new Uint8ClampedArray(12), width: 4, height: 1, contentWidth: 4, contentHeight: 1 }
  // Two red pixels at left; the broad candidate includes two blue pixels too.
  colorFrame.data.set([230, 35, 30, 240, 45, 35, 30, 70, 220, 20, 60, 210])
  const tightRed = colorEvidenceForBox(colorFrame, [0, 0, 0.5, 1], 'red')
  const broadScene = colorEvidenceForBox(colorFrame, [0, 0, 1, 1], 'red')
  check(
    'text-core: requested-colour evidence favours a tight matching box',
    tightRed === 1 && broadScene > 0 && broadScene < tightRed,
    `tight=${tightRed.toFixed(2)} broad=${broadScene.toFixed(2)}`,
  )
  const irregular = normalizePhrase('leaves')
  check(
    'text-core: irregular plurals depluralize to the real noun ("leaves" → leaf)',
    irregular.core === 'leaf' && irregular.multi === true
      && normalizePhrase('all people').core === 'person',
    `${irregular.core} multi=${irregular.multi}`,
  )
  // The detector scores a phrase as a bag of words, so a phrase whose SETTING is
  // present matches even with its subject absent — measured on the canonical NEF,
  // "the dog sitting among the flowers" returned 5 boxes and "snow covering the
  // flowers" 6, every one of them a flower. headCore is what the subject gate in
  // detectCandidates checks, so these assert where the subject ends.
  {
    const head = (p) => normalizePhrase(p).headCore
    check(
      'text-core: a post-modifier ends the subject ("dog sitting among the flowers" → dog)',
      head('the dog sitting among the flowers') === 'dog'
        && head('snow covering the flowers') === 'snow'
        && head('a red sports car parked on grass') === 'sports car'
        && head('green leaves in the background') === 'leaf',
      'setting stripped, head depluralized',
    )
    check(
      // A gate that fires on ordinary phrases would reject every real search,
      // and "of" must not split or "a cluster of blue florets" loses its subject.
      'text-core: no post-modifier means no subject gate (null, not the whole phrase)',
      head('orange tulip') === null && head('flower') === null
        && head('muscari') === null
        && head('a cluster of tightly packed blue florets') === null,
      'plain phrases unchanged',
    )
  }
  // A cluster-sized box around two real instances is a group guess, not a
  // match; a box containing only one other stays (could be the real object).
  const grouped = rankDetections([
    { box: [0, 0, 500, 500], score: 0.5 }, // container around both leaves
    { box: [40, 40, 200, 200], score: 0.45 },
    { box: [260, 260, 460, 460], score: 0.4 },
  ], { threshold: 0.15, iou: 0.5, topK: 8 })
  const lone = pruneContainers([
    { box: [0, 0, 500, 500], score: 0.5 },
    { box: [40, 40, 200, 200], score: 0.45 },
  ])
  check(
    'text-core: group box around several matches is pruned, members kept',
    grouped.length === 2 && grouped.every((d) => d.box[2] <= 460) && lone.length === 2,
    `grouped kept ${grouped.length}, single containment kept ${lone.length}`,
  )
  // A collapsed q4f16 head fills top_k with a flat ~0.6 band; a healthy result
  // has spread (or few boxes) and must never be flagged.
  const flatBand = Array.from({ length: 64 }, (_, i) => ({ box: [i, 0, i + 1, 1], score: 0.59 + (i % 10) * 0.002 }))
  const healthy = Array.from({ length: 64 }, (_, i) => ({ box: [i, 0, i + 1, 1], score: 0.05 + i * 0.01 }))
  check(
    'text-core: flat top-k score band is degenerate; spread or sparse output is not',
    degenerateScores(flatBand) === true && degenerateScores(healthy) === false
      && degenerateScores(flatBand.slice(0, 8)) === false,
    `flat=${degenerateScores(flatBand)} healthy=${degenerateScores(healthy)}`,
  )
  const deduped = nms([
    { box: [0, 0, 100, 100], score: 0.9 },
    { box: [5, 5, 105, 105], score: 0.8 },
    { box: [400, 400, 500, 500], score: 0.7 },
  ], 0.5)
  check('text-core: NMS drops overlaps, keeps distinct', deduped.length === 2, `kept ${deduped.length}`)
  const ranked = rankDetections(
    [{ box: [0, 0, 10, 10], score: 0.05 }, { box: [20, 20, 30, 30], score: 0.4 }],
    { threshold: 0.15, topK: 8 },
  )
  const sb = scaleBox([10, 20, 30, 40], 2, 3)
  check(
    'text-core: rank filters by threshold; scaleBox maps coords',
    ranked.length === 1 && ranked[0].score === 0.4 && sb[0] === 20 && sb[3] === 120,
    `ranked=${ranked.length} scaled=[${sb}]`,
  )
  const plan = letterboxPlan(1200, 800, DETECTOR_INPUT)
  const full = unletterboxBox([0, 0, 1, 640 / 960], plan)
  check(
    'text-core: letterbox plan preserves aspect; boxes map back to source px',
    plan.dw === 960 && plan.dh === 640 && full[2] === 1200 && Math.round(full[3]) === 800,
    `plan=${plan.dw}x${plan.dh} full=[${full}]`,
  )

  // Singular phrase: a train split front/rear collapses to one box; a distinct
  // far object and a small sign stay out of it.
  const fragments = [
    { box: [560, 635, 815, 920], score: 0.42, label: 'train' }, // front
    { box: [1105, 690, 1420, 900], score: 0.31, label: 'train' }, // rear (gap)
    { box: [1780, 800, 1840, 880], score: 0.2, label: 'train' }, // far small object
  ]
  const [merged] = clusterObjects(fragments)
  const twoTrains = clusterObjects([
    { box: [0, 0, 300, 300], score: 0.5 }, { box: [1400, 0, 1700, 300], score: 0.4 },
  ])
  check(
    'text-core: fragments of one object merge; distinct instances are both kept',
    merged.box[0] === 560 && merged.box[2] === 1420 && merged.box[3] === 920
      && twoTrains.length === 2 && twoTrains[0].box[2] === 300 && twoTrains[1].box[0] === 1400,
    `merged=[${merged.box}] distinct kept ${twoTrains.length}`,
  )

  // Search taxonomy: main class → kind recall expansion.
  const flowerExp = expandQuery('flower')
  const roseExp = expandQuery('rose')
  check(
    'taxonomy: main class expands to its kinds; a kind stays specific; unknown → null',
    flowerExp?.main === 'flower' && flowerExp.labels.includes('rose') && flowerExp.labels.includes('tulip')
      && roseExp?.main === 'flower' && roseExp.labels.length === 1 && roseExp.labels[0] === 'rose'
      && expandQuery('spaceship') === null,
    `flower=${flowerExp?.labels.length} rose=[${roseExp?.labels}]`,
  )
  check(
    'taxonomy: label match is class-aware, else falls back to flat whole-word',
    labelMatchesQuery('flower', 'rose') && labelMatchesQuery('flower', 'tulip') && !labelMatchesQuery('flower', 'car')
      && labelMatchesQuery('rose', 'rose') && !labelMatchesQuery('rose', 'tulip')
      && labelMatchesQuery('bottle', 'water bottle') && !labelMatchesQuery('bottle', 'bottleneck'),
    'expansion + fallback',
  )

  /* CLIP BPE — the open-vocabulary path's silent-failure surface. A wrong token
   * id still yields a confident vector, just for a different string, so these
   * are known-answer tests against the reference tokenizer (openai/CLIP), not
   * self-consistency checks. The last case covers punctuation and a contraction,
   * which is where the byte-symbol ordering bug showed up. */
  {
    const mergesPath = path.join(ROOT, 'models', 'clip-text', 'merges.txt')
    if (!existsSync(mergesPath)) {
      check('clip-tokenizer: merges.txt present', false, 'run scripts/export-clip-text.py')
    } else {
      const merges = readFileSync(mergesPath, 'utf8')
      const prevFetch = globalThis.fetch
      globalThis.fetch = async () => ({ text: async () => merges })
      const { loadTokenizer, tokenize, CONTEXT, VOCAB } = await import('./js/clip-tokenizer.js')
      await loadTokenizer()
      globalThis.fetch = prevFetch
      const golden = [
        ['orange tulip', [49406, 4287, 28389, 49407]],
        ['muscari', [49406, 5696, 3681, 49407]],
        ["a weathered wooden fence post, don't you think?",
          [49406, 320, 598, 34091, 9057, 12679, 1549, 267, 847, 713, 592, 1331, 286, 49407]],
      ]
      const out = tokenize(golden.map(([p]) => p))
      const ok = golden.every(([, want], i) => want.every((v, k) => out[i * CONTEXT + k] === v))
      check('clip-tokenizer: matches the reference BPE on known phrases', ok, `vocab ${VOCAB}`)
      // Padding must be zeros AFTER the EOT, and the EOT must survive truncation
      // — the tower reads its output from the EOT slot.
      const long = tokenize([`${'word '.repeat(120)}`])
      check(
        'clip-tokenizer: over-long phrase truncates with EOT kept last',
        long[CONTEXT - 1] === 49407 && long.length === CONTEXT,
        `last=${long[CONTEXT - 1]}`,
      )
    }
  }

  /* Detector tiling. A 45 MP frame squeezed into 640² puts a 200 px subject
   * under 16 px — measured as muscari scoring 0.09 while a tulip in the SAME
   * frame scored 0.60. Tiles raise linear resolution; these pin the geometry
   * that maps a tile-local box back to the original. */
  {
    const W = 8256; const H = 5504 // the canonical NEF
    check(
      'tiling: shrink factor decides whether a second pass can help',
      Math.round(shrinkFactor(W, H, YOLOE_INPUT)) === 13 && shrinkFactor(1200, 800, YOLOE_INPUT) < 2.5,
      `${shrinkFactor(W, H, YOLOE_INPUT).toFixed(1)}x on the NEF`,
    )
    const cells = tilePlans(W, H, YOLOE_INPUT, { grid: 2, overlap: 0.15 })
    const covers = Math.min(...cells.map((c) => c.ox)) === 0
      && Math.min(...cells.map((c) => c.oy)) === 0
      && Math.max(...cells.map((c) => c.ox + c.ow)) === W
      && Math.max(...cells.map((c) => c.oy + c.oh)) === H
    check('tiling: a 2x2 grid covers the whole frame with no gap', cells.length === 4 && covers,
      `${cells.length} cells`)
    // Seam overlap is what stops a subject on a tile edge being cut in half.
    const [a, b] = cells
    check('tiling: adjacent tiles overlap', (a.ox + a.ow) - b.ox > 0,
      `${Math.round((a.ox + a.ow) - b.ox)} px`)
    check(
      'tiling: each tile sees the subject larger than the full-frame pass does',
      cells.every((c) => Math.max(c.ow, c.oh) / YOLOE_INPUT < shrinkFactor(W, H, YOLOE_INPUT)),
      `${(Math.max(cells[0].ow, cells[0].oh) / YOLOE_INPUT).toFixed(1)}x vs ${shrinkFactor(W, H, YOLOE_INPUT).toFixed(1)}x`,
    )
    // A box found in a tile must land back on the same pixels in the original.
    const cell = cells[3]
    const local = unletterboxBox([0.25, 0.25, 0.75, 0.75], cell.plan)
    const raw = [local[0] + cell.ox, local[1] + cell.oy, local[2] + cell.ox, local[3] + cell.oy]
    const mapped = [Math.min(Math.max(raw[0], 0), W), Math.min(Math.max(raw[1], 0), H),
      Math.min(Math.max(raw[2], 0), W), Math.min(Math.max(raw[3], 0), H)]
    check(
      'tiling: a tile-local box maps back inside the original frame',
      mapped[0] >= 0 && mapped[1] >= 0 && mapped[2] <= W && mapped[3] <= H && mapped[2] > mapped[0],
      `[${mapped.map((v) => Math.round(v))}] (raw y1 ${raw[3].toFixed(1)} needed the clamp)`,
    )
  }

  /* Class slots. The axis is DYNAMIC and exactly one slot is fed per phrase:
   * the head emits each anchor once per class into a fixed top-300, so padding a
   * short list by repetition spent the budget on duplicates (measured: 10 unique
   * boxes out of 300 for a one-phrase query). These assert the list that decides
   * nc — deduped, user's words first, capped. */
  {
    const { MAX_SLOTS, DIM } = await import('./js/yoloe-detect.js')
    const { slotPhrases } = await import('./js/text-ui.js')
    // 48, not the 32 the graph was traced at: the class axis is dynamic and a
    // live sweep ran nc up to 128 for +14% latency, so the cap only has to clear
    // the largest taxonomy expansion ("animal" → 45 labels + phrase + object form).
    check('yoloe: class-slot contract', MAX_SLOTS === 48 && DIM === 512, `${MAX_SLOTS}x${DIM}`)
    const animal = slotPhrases(normalizePhrase('animal'))
    check(
      'yoloe: the widest taxonomy expansion is no longer truncated',
      animal.length === new Set(animal).size && animal.length <= MAX_SLOTS && animal.length >= 45,
      `${animal.length} slots`,
    )

    const one = slotPhrases(normalizePhrase('muscari'))
    check(
      'yoloe: an unknown phrase feeds exactly ONE class slot, not a padded 32',
      one.length === 1 && one[0] === 'muscari',
      `[${one}]`,
    )
    const many = slotPhrases(normalizePhrase('flower'))
    check(
      'yoloe: taxonomy expansion is deduped, user-phrase first, capped at MAX_SLOTS',
      many[0] === 'flower' && many.length <= MAX_SLOTS && new Set(many).size === many.length,
      `${many.length} slots`,
    )
    const colored = slotPhrases(normalizePhrase('the red car'))
    check(
      'yoloe: a colour phrase keeps the full wording in slot 0',
      colored[0] === 'red car' && colored.includes('car'),
      `[${colored.slice(0, 3)}]`,
    )
  }

  // Region axis (size/position) from a proxy box in a 1000×1000 image.
  const big = regionOf([100, 100, 700, 700], 1000, 1000) // 36% area, centered
  const tiny = regionOf([10, 10, 90, 90], 1000, 1000) // 0.6% area, top-left
  const low = regionOf([300, 650, 800, 980], 1000, 1000) // large & low → foreground
  check(
    'taxonomy: regionOf buckets size + position, flags large-and-low as foreground',
    big.size === 'large' && big.where === 'center' && tiny.size === 'small' && tiny.where === 'left'
      && low.foreground === true,
    `${big.size}/${big.where} ${tiny.size}/${tiny.where} fg=${low.foreground}`,
  )

  // Facets from ranked candidates (colour tagged by caller, region derived here).
  const cands = [
    { box: [0, 0, 400, 400], label: 'rose', color: 'red' },
    { box: [500, 0, 900, 400], label: 'rose', color: 'red' },
    { box: [0, 500, 400, 900], label: 'tulip', color: 'purple' },
  ]
  const facets = buildFacets(cands, { width: 1000, height: 1000 })
  const redFacet = facets.colour.find((f) => f.value === 'red')
  check(
    'taxonomy: buildFacets groups colour + kind axes with correct member indices',
    facets.colour.length === 2 && redFacet.count === 2 && redFacet.idx.join() === '0,1'
      && facets.kind.length === 2 && facets.kind.find((f) => f.value === 'tulip').idx.join() === '2'
      && buildFacets([cands[0]]).colour.length === 0,
    `colour=${facets.colour.length} kind=${facets.kind.length}`,
  )

  // Autocomplete: main classes, kinds, colour combos; colour prefix carries.
  const acFlo = suggest('flo')
  const acRedFlo = suggest('red flo')
  const acRose = suggest('ros')
  check(
    'taxonomy: suggest surfaces categories, kinds, and colour combos; empty → []',
    acFlo.some((r) => r.text === 'flower' && r.group === 'category')
      && acRedFlo.some((r) => r.text === 'red flower')
      && acRose.some((r) => r.text === 'rose' && r.group === 'kind')
      && suggest('').length === 0,
    `flo=${acFlo.length} redflo=${acRedFlo.length} ros=${acRose.length}`,
  )

  // Pixel colour classifier + dominant-colour box sampling.
  const redFrame = { data: new Uint8ClampedArray(4 * 4 * 3), width: 4, height: 4, contentWidth: 4, contentHeight: 4 }
  for (let i = 0; i < redFrame.data.length; i += 3) { redFrame.data[i] = 220; redFrame.data[i + 1] = 20; redFrame.data[i + 2] = 20 }
  const dom = dominantColorForBox(redFrame, [0, 0, 1, 1])
  check(
    'taxonomy: classifyPixelColor + dominantColorForBox agree on a red field',
    classifyPixelColor(230, 20, 20) === 'red' && classifyPixelColor(248, 248, 248) === 'white'
      && classifyPixelColor(8, 8, 8) === 'black' && dom?.color === 'red',
    `dom=${dom?.color}`,
  )

  /* ── Policy: the memory-trust lock ── */
  const gpu = { webgpu: true, f16: true, textureLimit: 16384, storageBufferLimit: 256 * 1024 * 1024 }
  const fourGB = classifyCapability({ ...gpu, browserMemoryGB: 4 })
  const browserEightGB = classifyCapability({ ...gpu, browserMemoryGB: 8 })
  const unknownMemory = classifyCapability({ ...gpu, browserMemoryGB: 0 })
  const phosmith16GB = classifyCapability({ ...gpu, browserMemoryGB: 8, hostResources: HOST_PRO })
  const phosmith24GB = classifyCapability({ ...gpu, browserMemoryGB: 8, hostResources: HOST_ULTRA })
  check(
    'capability: memory evidence no longer selects a tier — one config regardless (§11)',
    browserEightGB.profile === 'standard8' && fourGB.profile === 'standard8'
      && unknownMemory.profile === 'standard8'
      && browserEightGB.memorySource === 'browser' && unknownMemory.memorySource === 'unknown',
    JSON.stringify({ eight: browserEightGB.profile, four: fourGB.profile, unknown: unknownMemory.profile }),
  )
  check(
    'capability: a trusted Phosmith budget is recorded but no longer buys a tier',
    phosmith16GB.profile === 'standard8' && phosmith24GB.profile === 'standard8'
      && phosmith16GB.memorySource === 'phosmith' && phosmith16GB.hostManaged === true,
    JSON.stringify({ pro: phosmith16GB.profile, ultra: phosmith24GB.profile, src: phosmith16GB.memorySource }),
  )

  /* ── Text lane: on everywhere by default, ?text=0 is the only opt-out ── */
  check(
    'capability: the text lane is on by default on every engine, including WebKit',
    probeTextLane('').ok === true && probeTextLane('').reason === 'ok',
    JSON.stringify(probeTextLane('')),
  )
  check(
    'capability: ?text=0 disables the lane',
    probeTextLane('?text=0').ok === false && probeTextLane('?text=0').reason === 'disabled',
    JSON.stringify(probeTextLane('?text=0')),
  )

  const liteDefault = resolveBudget('', browserEightGB)
  const unknownBudget = resolveBudget('', unknownMemory)
  const provisional = resolveBudget('', null)
  check(
    'policy: every device resolves to the SAME single config (§11 — no tiers)',
    liteDefault.profile === unknownBudget.profile && unknownBudget.profile === provisional.profile
      && liteDefault.profileSource === 'single' && liteDefault.proxyMax === 1024,
    JSON.stringify({ a: liteDefault.profile, b: unknownBudget.profile, c: provisional.profile, src: liteDefault.profileSource }),
  )
  check(
    'policy: single-config caps — one embedding, one heavy job, no auto-escalation, bounded export',
    liteDefault.draftCacheMax === 1 && liteDefault.flagshipCacheMax === 0
      && liteDefault.maxResidentHeavy === 1 && liteDefault.flagship === false
      && liteDefault.autoEscalate === false && liteDefault.samWebGPU === true
      && liteDefault.exportMaxMP === 12 && liteDefault.exportMaxSide === 5120
      && liteDefault.hdExportDecode === true && liteDefault.embedPersist === true
      && liteDefault.detectorDispose === 'idle' && liteDefault.detectorEvictOnEncode === true,
    JSON.stringify({ draft: liteDefault.draftCacheMax, heavy: liteDefault.maxResidentHeavy,
      escalate: liteDefault.autoEscalate, exportMP: liteDefault.exportMaxMP }),
  )
  const liteNoGpu = resolveBudget('', classifyCapability({ webgpu: false, browserMemoryGB: 8 }))
  const liteFallbackGpu = resolveBudget('', classifyCapability({ webgpu: true, fallback: true, browserMemoryGB: 8 }))
  check(
    'policy: SAM runs on the GPU whenever one is probed, independent of the memory tier',
    liteDefault.samWebGPU === true && resolveBudget('', unknownMemory).samWebGPU === true
      && liteNoGpu.samWebGPU === false && liteFallbackGpu.samWebGPU === false
      && resolveBudget('?force=wasm', browserEightGB).forceWasm === true,
    JSON.stringify({ liteGpu: liteDefault.samWebGPU, liteNoGpu: liteNoGpu.samWebGPU, liteFallback: liteFallbackGpu.samWebGPU }),
  )
  /* ── Policy: adaptive auto-tier + manual toggle (real signals, never a URL param) ── */
  const eightCoreBrowser = classifyCapability({ ...gpu, browserMemoryGB: 8, logicalProcessors: 8 })
  const eightCoreLowMem = classifyCapability({ ...gpu, browserMemoryGB: 4, logicalProcessors: 8 })
  const fourCoreBrowser = classifyCapability({ ...gpu, browserMemoryGB: 8, logicalProcessors: 4 })
  const mobileEightCore = classifyCapability({ ...gpu, browserMemoryGB: 8, logicalProcessors: 8, mobile: true })
  const noGpuEightCore = classifyCapability({ webgpu: false, browserMemoryGB: 8, logicalProcessors: 8 })
  check(
    'capability: classify reports the single config for every device shape (§11)',
    eightCoreBrowser.profile === 'standard8' && fourCoreBrowser.profile === 'standard8'
      && mobileEightCore.profile === 'standard8' && noGpuEightCore.profile === 'standard8'
      && eightCoreBrowser.autoTier === undefined,
    JSON.stringify({ eight: eightCoreBrowser.profile, mobile: mobileEightCore.profile, autoTier: eightCoreBrowser.autoTier }),
  )
  const autoTiered = resolveBudget('', eightCoreBrowser)     // capable device, no override → standard8
  const autoNoSignal = resolveBudget('', browserEightGB)     // no cores probed → lite floor
  const manualLite = resolveBudget('', eightCoreBrowser, 'lite')
  const manualStandard = resolveBudget('', eightCoreBrowser, 'standard')
  const manualUltra = resolveBudget('', unknownMemory, 'ultra')
  const overrideIgnoredWhenTrusted = resolveBudget('', phosmith16GB, 'lite')
  check(
    'policy: a profile override is no longer honoured — one config, always (§11)',
    autoTiered.profile === 'standard8'
      && resolveBudget('', eightCoreBrowser, 'ultra').profile === 'standard8'
      && resolveBudget('?profile=ultra', eightCoreBrowser).profile === 'standard8',
    JSON.stringify({ auto: autoTiered.profile,
      arg: resolveBudget('', eightCoreBrowser, 'ultra').profile,
      url: resolveBudget('?profile=ultra', eightCoreBrowser).profile }),
  )
  check(
    // Memory-close to lite by design: the NEF import+click peak with escalation
    // ON was ~2.1 GB (measured); OFF it is ~1.3 GB, equal to lite. So the auto
    // tier takes only the cheap wins (12 MP HD-decoded export, crisper preview)
    // and leaves the interaction-time native re-decode to manually-chosen tiers.
    'policy: the single config is memory-safe by construction — one embedding, one heavy, ≤12 MP export, HD decode on, native escalation OFF',
    autoTiered.draftCacheMax === 1 && autoTiered.maxResidentHeavy === 1
      && autoTiered.exportMaxMP === 12 && autoTiered.hdExportDecode === true
      && autoTiered.autoEscalate === false && autoTiered.samWebGPU === true
      // The CEILING, not the working set — the lane rests at ~1980 MB, so a
      // budget under that declared a healthy app to be in permanent pressure.
      && autoTiered.memBudgetMB === 2200,
    JSON.stringify({ cache: autoTiered.draftCacheMax, exportMP: autoTiered.exportMaxMP, hd: autoTiered.hdExportDecode, esc: autoTiered.autoEscalate, budget: autoTiered.memBudgetMB }),
  )
  check(
    'policy: the detector is evicted before an encode and idles out',
    autoTiered.detectorEvictOnEncode === true && autoTiered.detectorIdleMs === 120_000,
    JSON.stringify({ evict: autoTiered.detectorEvictOnEncode, idleMs: autoTiered.detectorIdleMs }),
  )
  const flagged = resolveBudget('?flagship=1', browserEightGB)
  const ultraReq = resolveBudget('?profile=ultra', browserEightGB)
  const proxyMax = resolveBudget('?proxy=max', browserEightGB)
  const proxyOff = resolveBudget('?proxy=off', unknownMemory)
  const workingForce = resolveBudget('?working=1', browserEightGB)
  check(
    'safety: ?flagship=1 cannot enable SAM3 on an unverified budget',
    flagged.flagship === false && resolveBudget('?flagship=1', unknownMemory).flagship === false,
    `flagship=${flagged.flagship}`,
  )
  check(
    'safety: ?profile=ultra / ?proxy=max / ?proxy=off cannot raise the 1024 px cap',
    ultraReq.profile === 'standard8' && ultraReq.proxyMax === 1024
      && proxyMax.proxyMax === 1024 && proxyMax.proxyMode === 'auto'
      && proxyOff.proxyMax === 1024 && proxyOff.proxyMode === 'auto',
    JSON.stringify({ ultra: ultraReq.proxyMax, max: proxyMax.proxyMax, off: proxyOff.proxyMax }),
  )
  check(
    'safety: ?working=1 refused on a locked budget; lowering params still work',
    workingForce.workingMode === undefined
      && resolveBudget('?proxy=512', browserEightGB).proxyMax === 512
      && resolveBudget('?escalate=0', phosmith16GB).autoEscalate === false,
    JSON.stringify({ working: workingForce.workingMode }),
  )
  check(
    // 4096: the re-decode SOURCE for escalation and export, so it must not sit
    // below its own consumers (exportMaxSide 5120). ~67 MB RGBA.
    'policy: the working copy is bounded',
    liteDefault.workingMaxSide <= 4096,
    `workingMaxSide=${liteDefault.workingMaxSide}`,
  )
  const dispNative = resolveBudget('?display=native', browserEightGB)
  const dispOff = resolveBudget('?display=off', browserEightGB)
  const dispCap = resolveBudget('?display=1600', browserEightGB)
  check(
    'display: ?display native/off/<px> honored on a locked budget (display-only, not the memory contract)',
    liteDefault.displayMax === 2560 && dispNative.displayMode === 'native'
      && dispOff.displayMode === 'off' && dispCap.displayMax === 1600,
    JSON.stringify({ base: liteDefault.displayMax, native: dispNative.displayMode, off: dispOff.displayMode, cap: dispCap.displayMax }),
  )
  const ultraBudget = resolveBudget('?flagship=1', phosmith24GB)
  const pressured = applyMemoryPressure(liteDefault, 2)
  const pressured3 = applyMemoryPressure(liteDefault, 3)
  check(
    'policy: SAM3 is never enabled; pressure only ever TIGHTENS (flat ladder, §11)',
    ultraBudget.flagship === false
      && pressured.cvRefine === false && pressured.hdExportDecode === false
      && pressured.cropMaxSide <= 1280 && pressured.eagerEncode === false
      && pressured3.exportMaxMP === 4 && pressured3.proxyMax === 768
      && pressured3.exportMaxSide <= 4096
      && applyMemoryPressure(ultraBudget, 3).exportMaxMP === 4,
    JSON.stringify({ p2: pressured.cvRefine, p3MP: pressured3.exportMaxMP, p3Side: pressured3.exportMaxSide }),
  )
  check(
    // Pressure keeps the mask lane on the GPU: the old WASM lane pinned ~3 GB
    // against the GPU's ~0.5 GB (measured), so demoting under memory pressure
    // made swap WORSE. There is no wasm EP left, so this now guards the budget
    // flag rather than a device choice.
    'policy: pressure keeps SAM on the memory-safe GPU lane and lowers the display ceiling',
    applyMemoryPressure(liteDefault, 3).samWebGPU === true
      && applyMemoryPressure(ultraBudget, 1).samWebGPU === ultraBudget.samWebGPU
      && pressured.displayMax === 1600 && pressured3.displayMax === 1280,
    JSON.stringify({ p3sam: applyMemoryPressure(liteDefault, 3).samWebGPU, p2disp: pressured.displayMax, p3disp: pressured3.displayMax }),
  )
  check(
    'policy: every tier declares a memBudgetMB the governor watches',
    [liteDefault, PROFILE_PRESETS.standard8]
      .every((b) => b.memBudgetMB >= 1000)
      && applyMemoryPressure(liteDefault, 3).memBudgetMB === liteDefault.memBudgetMB, // pressure carries it through
    JSON.stringify({ single: liteDefault.memBudgetMB }),
  )

  /* ── Memory governor: judges THIS APP's footprint, never the machine's RAM ── */
  check(
    // A runaway WASM heap (the 3 GB failure) still sheds to L3. Drift measures
    // the HOST, so it can no longer shed at any magnitude — a machine swapping
    // under everything else the user has open must not pause our WebGPU work.
    // It survives only as a climb veto.
    'governor: app footprint sheds; drift alone never does; deep headroom climbs',
    decidePressure({ bytesMB: 3000, budgetMB: 2200 }).level === 3
      && decidePressure({ bytesMB: 500, budgetMB: 2200 }).headroom === true
      && decidePressure({ bytesMB: 2150, budgetMB: 2200 }).level === 1
      && decidePressure({ bytesMB: 0, driftMs: 6000 }).level === 0   // the OS is swapping, not us
      && decidePressure({ bytesMB: 0, driftMs: 500 }).level === 0
      && decidePressure({ bytesMB: 500, budgetMB: 2200, driftMs: 6000 }).headroom === false // …but do not climb into it
      && decidePressure({ bytesMB: 1200, budgetMB: 2200 }).headroom === false, // under budget but not deep → no climb
    JSON.stringify({
      wasm: decidePressure({ bytesMB: 3000, budgetMB: 2200 }).level,
      headroom: decidePressure({ bytesMB: 500, budgetMB: 2200 }).headroom,
      driftDeep: decidePressure({ bytesMB: 0, driftMs: 6000 }).level,
      driftVetoesClimb: decidePressure({ bytesMB: 500, budgetMB: 2200, driftMs: 6000 }).headroom,
    }),
  )

  // The ledger is the ONLY signal that sees the whole app.
  // `measureUserAgentSpecificMemory` covers one agent cluster, and the SAM lane's
  // SharedWorker is a different one — measured 75 MB while the app held ~2 GB. So
  // a byte reading is a FLOOR: it may raise the estimate, never silence it. It
  // used to do exactly that (the caller skipped the ledger whenever bytes > 0),
  // which left Chrome with no footprint signal at all.
  //
  // And the bands sit above normal operation: with an encoder live the ledger
  // rests at ~1980 MB, which the old 0.85 warn band called pressure forever.
  check(
    'governor: the ledger is a floor a partial byte reading cannot silence; rest is calm',
    decidePressure({ bytesMB: 0, estimateMB: 2600, budgetMB: 2200 }).level === 3
      && decidePressure({ bytesMB: 0, estimateMB: 2300, budgetMB: 2200 }).level === 2
      && decidePressure({ bytesMB: 0, estimateMB: 2150, budgetMB: 2200 }).level === 1
      && decidePressure({ bytesMB: 0, estimateMB: 1980, budgetMB: 2200 }).level === 0 // the lane at rest
      && decidePressure({ bytesMB: 0, estimateMB: 400, budgetMB: 2200 }).headroom === true
      // The 75 MB page reading must not hide a 2.3 GB app.
      && decidePressure({ bytesMB: 75, estimateMB: 2300, budgetMB: 2200 }).level === 2
      // The heap floor stays reachable when there is no ledger either.
      && decidePressure({ bytesMB: 0, estimateMB: 0, heapMB: 700 }).level === 3,
    JSON.stringify({
      shedL3: decidePressure({ bytesMB: 0, estimateMB: 2600, budgetMB: 2200 }).level,
      atRest: decidePressure({ bytesMB: 0, estimateMB: 1980, budgetMB: 2200 }).level,
      partialBytes: decidePressure({ bytesMB: 75, estimateMB: 2300, budgetMB: 2200 }).level,
    }),
  )

  /* ── Candidate arbitration: the six ways one click goes wrong ──────────────
   * SAM returns three masks and three predicted IoUs per decode. Predicted IoU
   * answers "how well would this mask score against its own target", which is
   * NOT the question a click asks, and taking argmax of it — what this lane did
   * — is silently wrong in each case below. Synthetic fields, because the point
   * is the decision rule, not the model.
   */
  {
    const S = 64
    const mk = (hit, mag = 4) => {
      const f = new Float32Array(S * S)
      for (let y = 0; y < S; y += 1) for (let x = 0; x < S; x += 1) f[y * S + x] = hit(x, y) ? mag : -mag
      return f
    }
    const rect = (x0, y0, x1, y1) => (x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1
    const opts = { side: S, scale: 1 }
    const sub = mk(rect(28, 28, 35, 35))     // 8×8   — a cluster of petals
    const part = mk(rect(22, 22, 41, 41))    // 20×20 — the bloom
    const whole = mk(rect(12, 12, 51, 51))   // 40×40 — the whole flower
    const planes = [sub, part, whole]
    const centre = [{ x: 32, y: 32, label: 1 }]

    // 1. The rose. Highest predicted IoU IS the mask that leaves parts out, so
    //    with nothing else to go on the default must still be score-led — the
    //    arbitration is not allowed to invent a preference for "bigger".
    const led = chooseCandidate({ planes, scores: [0.95, 0.80, 0.70], ...opts, clicks: centre })
    // 2. Hierarchy drift. Second click, and the level is free to change under
    //    the user. Continuity pins it to what they were already holding even
    //    though the sub-part scores far higher.
    const pinned = chooseCandidate({ planes, scores: [0.95, 0.80, 0.70], ...opts, clicks: centre, previous: whole })
    // 3. …but continuity must not pin to something unrelated. A previous mask
    //    that overlaps nothing falls through to the score instead of dragging
    //    the selection across the frame.
    const stray = mk(rect(0, 0, 4, 4))
    const fell = chooseCandidate({ planes, scores: [0.95, 0.80, 0.70], ...opts, clicks: centre, previous: stray })
    // 4. Negative clicks. An exclude point inside a candidate disqualifies it
    //    outright, even when it is the confident one — an instruction, not a
    //    preference. Without the exclude click the same call picks it.
    const exc = [...centre, { x: 45, y: 45, label: 0 }]
    const before = chooseCandidate({ planes, scores: [0.70, 0.80, 0.95], ...opts, clicks: centre })
    const after = chooseCandidate({ planes, scores: [0.70, 0.80, 0.95], ...opts, clicks: exc })
    check(
      'select: score leads a first click, continuity pins a refinement, an exclude click overrules both',
      led.index === 0 && pinned.index === 2 && pinned.reason === 'agree'
        && fell.index === 0 && fell.reason === 'rank'
        && before.index === 2 && after.index !== 2 && after.pick.leak === 0,
      JSON.stringify({ led: led.index, pinned: pinned.index, fell: fell.index, before: before.index, after: after.index }),
    )

    // 5. Mushy fields. Two candidates, same shape, same predicted IoU — one has
    //    a real boundary and one does not. Predicted IoU cannot tell them
    //    apart; stability is the whole difference.
    const crisp = mk(rect(22, 22, 41, 41), 4)
    const mushy = mk(rect(22, 22, 41, 41), 0.5)
    const byStab = chooseCandidate({ planes: [mushy, crisp], scores: [0.9, 0.9], ...opts, clicks: centre })
    check(
      'select: stability breaks a tie predicted IoU cannot see',
      stabilityScore(crisp) === 1 && stabilityScore(mushy) === 0 && byStab.index === 1,
      JSON.stringify({ crisp: stabilityScore(crisp), mushy: stabilityScore(mushy), pick: byStab.index }),
    )

    // 6. Box prompts. Text search selects exclusively by box, so a candidate
    //    that spills outside the detector's box is answering about a different
    //    object. Corners ride as labels 2/3, exactly as the decoder expects.
    const boxed = [
      { x: 32, y: 32, label: 1 },
      { x: 20, y: 20, label: 2 },
      { x: 44, y: 44, label: 3 },
    ]
    const spill = mk((x, y) => rect(22, 22, 41, 41)(x, y) || rect(0, 50, 20, 63)(x, y))
    const byBox = chooseCandidate({ planes: [spill, part], scores: [0.9, 0.9], ...opts, clicks: boxed })
    check(
      'select: a candidate that spills outside the prompt box loses to one that fits',
      boxFraction(part, S, [20, 20, 44, 44], 1) === 1
        && boxFraction(spill, S, [20, 20, 44, 44], 1) < 0.8
        && byBox.index === 1,
      JSON.stringify({ fit: boxFraction(part, S, [20, 20, 44, 44], 1), spill: +boxFraction(spill, S, [20, 20, 44, 44], 1).toFixed(3), pick: byBox.index }),
    )

    // Prompt fit itself: positives are checked leniently because one grid cell
    // is 4 px at the proxy and a click on a thin structure legitimately lands a
    // cell off centre; negatives are checked strictly, because an exclude point
    // still inside the mask is not a rounding error.
    const edgeClick = [{ x: 42, y: 32, label: 1 }]   // one cell outside `part`
    check(
      'select: positive clicks tolerate a cell of slop, negative clicks do not',
      promptFit(part, S, edgeClick, 1).miss === 0
        && promptFit(part, S, [{ x: 44, y: 32, label: 1 }], 1).miss === 1
        && promptFit(part, S, [{ x: 42, y: 32, label: 0 }], 1).leak === 0
        && promptFit(part, S, [{ x: 41, y: 32, label: 0 }], 1).leak === 1,
      JSON.stringify(promptFit(part, S, edgeClick, 1)),
    )

    /* ── Region hygiene: upstream SAM's remove_small_regions, both modes ── */
    // Islands: speckle far from the subject goes; a second component the user
    // actually clicked stays, and so does one large enough to be real.
    const speckled = mk((x, y) => rect(22, 22, 41, 41)(x, y) || rect(2, 2, 3, 3)(x, y) || rect(58, 58, 59, 59)(x, y))
    const kept = new Float32Array(speckled)
    const r1 = cleanRegions(speckled, S, { clicks: [{ x: 32, y: 32, label: 1 }], scale: 1 })
    const r2 = cleanRegions(kept, S, { clicks: [{ x: 32, y: 32, label: 1 }, { x: 3, y: 3, label: 1 }], scale: 1 })
    check(
      'hygiene: islands with no click are speckle; an island the user clicked is the object',
      r1.islands === 2 && fieldArea(speckled) === 400
        && r2.islands === 1 && fieldArea(kept) === 400 + 4,
      JSON.stringify({ dropped: r1.islands, areaAfter: fieldArea(speckled), clicked: r2.islands, keptArea: fieldArea(kept) }),
    )

    // …but only on a mask that is already one blob. A 256² grid shatters wires
    // and thin arms into many small components, every one of which looks like
    // speckle by size — measured on the streetlight crop, the ungated rule cost
    // 0.050 IoU against the shipped pipeline (scratchpad/quality.mjs), and
    // 0.041 even with speckle deliberately injected. Severing a wire IS the
    // failure this module exists to fix, so a fragmented mask is left alone.
    const wires = mk((x, y) => rect(30, 8, 33, 56)(x, y)       // pole
      || rect(4, 20, 59, 22)(x, y)                             // wire
      || rect(10, 34, 24, 35)(x, y)                            // wire, detached
      || rect(2, 2, 4, 4)(x, y))                               // real speckle
    const wiresArea = fieldArea(wires)
    const rw = cleanRegions(wires, S, { clicks: [{ x: 32, y: 30, label: 1 }], scale: 1 })
    check(
      'hygiene: a fragmented subject (wires, thin arms) is never pruned, speckle or not',
      rw.islands === 0 && fieldArea(wires) === wiresArea,
      JSON.stringify({ dropped: rw.islands, area: fieldArea(wires), was: wiresArea }),
    )

    // Holes: an enclosed gap is the "parts left out" failure; a gap the user
    // explicitly excluded is deliberate; and the background is not a hole,
    // however the mask is shaped.
    const holed = mk((x, y) => rect(20, 20, 43, 43)(x, y) && !rect(28, 28, 32, 32)(x, y))
    const before2 = fieldArea(holed)
    const h1 = cleanRegions(holed, S, { clicks: [{ x: 22, y: 22, label: 1 }], scale: 1 })
    const spared = mk((x, y) => rect(20, 20, 43, 43)(x, y) && !rect(28, 28, 32, 32)(x, y))
    const h2 = cleanRegions(spared, S, { clicks: [{ x: 30, y: 30, label: 0 }], scale: 1 })
    check(
      'hygiene: an enclosed gap is filled, an excluded one is left alone, the background is never a hole',
      h1.holes === 1 && fieldArea(holed) === 24 * 24
        && h2.holes === 0 && fieldArea(spared) === before2,
      JSON.stringify({ filled: h1.holes, area: fieldArea(holed), solid: 24 * 24, spared: h2.holes }),
    )
  }

  /* ── Edge refinement: the guide must see colour, not only luma ── */
  {
    // A vertical step at x=32 that is INVISIBLE in Rec.601 luma: magenta vs a
    // green matched to the same Y. This is the rose-against-foliage case, and
    // it is the one the luma guide provably cannot solve.
    const S = 64
    const A = [210, 54, 180]                       // Y ≈ 0.4511
    const B = [20, 181, 20]                        // Y ≈ 0.4491 — matched to <1/255
    const yOf = (c) => (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255
    const px = new Uint8ClampedArray(S * S * 4)
    for (let y = 0; y < S; y += 1) for (let x = 0; x < S; x += 1) {
      const c = x < 32 ? A : B
      const j = (y * S + x) * 4
      px[j] = c[0]; px[j + 1] = c[1]; px[j + 2] = c[2]; px[j + 3] = 255
    }
    // A field whose crossing sits at x=36 — 4 px past the true edge, i.e. the
    // "spilled over" error to be corrected, and within a radius-8 filter's pull.
    const OFF = 36
    const mk = () => {
      const f = new Float32Array(S * S)
      for (let y = 0; y < S; y += 1) for (let x = 0; x < S; x += 1) f[y * S + x] = (OFF - x) * 0.6
      return f
    }
    const crossing = (f) => {          // mean x of the zero crossing, mid-rows
      let acc = 0; let n = 0
      for (let y = 16; y < 48; y += 1) {
        for (let x = 1; x < S; x += 1) {
          const a = f[y * S + x - 1]; const b = f[y * S + x]
          if (a > 0 && b <= 0) { acc += (x - 1) + a / (a - b); n += 1; break }
        }
      }
      return n ? acc / n : NaN
    }
    const bbox = [8, 8, 56, 56]
    const fl = mk(); refineField(fl, px, S, S, bbox, { radius: 8, eps: 1e-4, scale: 4, color: false })
    const fc = mk(); refineField(fc, px, S, S, bbox, { radius: 8, eps: 1e-4, scale: 4 })
    const dl = Math.abs(crossing(fl) - 32)
    const dc = Math.abs(crossing(fc) - 32)
    check(
      'refine: isoluminant edge — colour guide snaps to it, luma guide cannot',
      Math.abs(yOf(A) - yOf(B)) < 0.004 && dc < 1.5 && dl > 2.5 && dc < dl * 0.5,
      JSON.stringify({ dY: +(yOf(A) - yOf(B)).toFixed(4), lumaErrPx: +dl.toFixed(2), colourErrPx: +dc.toFixed(2) }),
    )

    // Greyscale guide: the colour path must degenerate to the luma path, not
    // wander. This is the guard on thin achromatic subjects (wires, poles).
    const grey = new Uint8ClampedArray(S * S * 4)
    for (let y = 0; y < S; y += 1) for (let x = 0; x < S; x += 1) {
      const v = x < 32 ? 60 : 200
      const j = (y * S + x) * 4
      grey[j] = grey[j + 1] = grey[j + 2] = v; grey[j + 3] = 255
    }
    const gl = mk(); refineField(gl, grey, S, S, bbox, { radius: 8, eps: 1e-4, scale: 4, color: false })
    const gc = mk(); refineField(gc, grey, S, S, bbox, { radius: 8, eps: 1e-4, scale: 4 })
    check(
      'refine: achromatic guide — colour path tracks the luma path (no drift)',
      Math.abs(crossing(gc) - crossing(gl)) < 0.5 && crossing(gc) < OFF - 1,
      JSON.stringify({ luma: +crossing(gl).toFixed(2), colour: +crossing(gc).toFixed(2), from: OFF }),
    )

    // Flat guide: nothing to snap to, so the field must survive unchanged
    // rather than be pulled by a near-singular solve.
    const flat = new Uint8ClampedArray(S * S * 4).fill(128)
    const ff = mk(); refineField(ff, flat, S, S, bbox, { radius: 8, eps: 1e-4, scale: 4 })
    check(
      'refine: flat guide leaves the crossing where it was',
      Math.abs(crossing(ff) - OFF) < 1.0,
      JSON.stringify({ crossing: +crossing(ff).toFixed(2) }),
    )
  }

  /* ── Sizing: the authoritative proxy function ── */
  const s1 = getBoundedProxySize(6000, 4000, 768)
  const s2 = getBoundedProxySize(4000, 6000, 768)
  const s3 = getBoundedProxySize(768, 512, 768)
  check(
    'sizing: 6000×4000 → 768×512, portrait → 512×768, small stays native',
    s1.width === 768 && s1.height === 512 && s1.proxyActive
      && s2.width === 512 && s2.height === 768 && s2.proxyActive
      && s3.width === 768 && s3.height === 512 && !s3.proxyActive && s3.scale === 1,
    JSON.stringify({ s1, s2, s3 }),
  )
  let sizingThrew = false
  try { getBoundedProxySize(0, 4000) } catch { sizingThrew = true }
  let sizingThrew2 = false
  try { getBoundedProxySize(NaN, 10) } catch { sizingThrew2 = true }
  check('sizing: invalid dimensions reject cleanly', sizingThrew && sizingThrew2, 'both threw')

  /* ── Per-axis proxy: the encoder eats a 1024² SQUARE, so the SHORT edge is
     what starves. Sizing by the long edge alone fed it 683 real rows stretched
     to 1024 — measured at −8.2 pt boundary IoU on the canonical NEF. ── */
  {
    const B = { ...PROFILE_PRESETS.standard8, proxyMode: 'auto' }
    const dims = (w, h, b = B) => {
      const p = interactionPlan(w, h, b)
      return [Math.round(w * p.scale), Math.round(h * p.scale)]
    }
    const [lw, lh] = dims(6000, 4000)
    const [pw, ph] = dims(4000, 6000)
    check(
      'proxy: short edge reaches the 1024 encoder edge in both orientations',
      Math.min(lw, lh) === 1024 && Math.min(pw, ph) === 1024 && lw === 1536 && ph === 1536,
      JSON.stringify({ landscape: [lw, lh], portrait: [pw, ph] }),
    )
    // A panorama must not turn the per-axis rule into an unbounded buffer.
    const [aw, ah] = dims(8000, 1000)
    const [bw, bh] = dims(6000, 2000)
    check(
      'proxy: panoramas stay inside the long-edge and pixel caps',
      aw <= B.proxyLongMax && bw <= B.proxyLongMax
        && aw * ah <= B.proxyPixelMax && bw * bh <= B.proxyPixelMax,
      JSON.stringify({ '8:1': [aw, ah], '3:1': [bw, bh], longMax: B.proxyLongMax, pxMax: B.proxyPixelMax }),
    )
    // Never invent detail that is not in the source.
    const small = interactionPlan(900, 600, B)
    check(
      'proxy: a source below the cap is still native (no upscale)',
      small.scale === 1 && !small.proxyActive,
      JSON.stringify(small),
    )
    // An explicit ?proxy= is the user's number; pressure gives the boost up first.
    const [mw] = dims(6000, 4000, { ...B, proxyMode: 'manual', proxyMax: 512 })
    const [rw, rh] = dims(6000, 4000, { ...B, proxyShortMax: 0 })
    check(
      'proxy: manual override and pressure L3 both drop the per-axis boost',
      mw === 512 && rw === 1024 && rh === 683,
      JSON.stringify({ manual: mw, pressured: [rw, rh] }),
    )
  }

  /* ── Display formula: viewport-anchored, decode-budget-gated ── */
  const liteB = { ...PROFILE_PRESETS.standard8 }
  const vp = { w: 1728, h: 1117, dpr: 2 }
  const dCeiling = displayPlan({ srcW: 8000, srcH: 6000, budget: liteB, viewport: vp, textureLimit: 8192 })
  const dSource = displayPlan({ srcW: 1600, srcH: 1000, budget: liteB, viewport: vp, textureLimit: 8192 })
  const dViewport = displayPlan({ srcW: 8000, srcH: 6000, budget: liteB, viewport: { w: 500, h: 400, dpr: 1 }, textureLimit: 8192 })
  const dTexture = displayPlan({ srcW: 8000, srcH: 6000, budget: { ...liteB, displayMode: 'native' }, viewport: vp, textureLimit: 4096 })
  const dOff = displayPlan({ srcW: 8000, srcH: 6000, budget: { ...liteB, displayMode: 'off' }, viewport: vp })
  check(
    'display formula: side = min(viewport·dpr·slack, config ceiling, texture cap, source) for any source',
    dCeiling.side === 2560 && !dCeiling.allowFullDecode // 48 MP > decode budget
      && dSource.side === 1600 && dSource.allowFullDecode // 1.6 MP fits
      && dViewport.side === 750 // 500·1·1.5 viewport-bound
      && dTexture.side === 4096 && dTexture.allowFullDecode // native opt-in, texture-capped
      && dOff.side === 0 && !dOff.allowFullDecode,
    JSON.stringify({ ceiling: dCeiling, source: dSource, viewport: dViewport, texture: dTexture, off: dOff }),
  )
  check(
    'display formula: decode budget reuses escalateMaxMP and ratchets down under pressure',
    decodeBudgetMP(liteB) === 12
      && decodeBudgetMP({ ...liteB, pressureLevel: 1 }) === 6
      && decodeBudgetMP({ ...liteB, pressureLevel: 2 }) === 4
      && decodeBudgetMP({ ...liteB, pressureLevel: 3 }) === 2,
    JSON.stringify([0, 1, 2, 3].map((level) => decodeBudgetMP({ ...liteB, pressureLevel: level }))),
  )

  /* ── Click-union composition: add/sub replay + peel ── */
  const chanOf = (bits) => Uint8Array.from(bits.map((v) => (v ? 255 : 0)))
  const rAt = (res, i) => (res ? res.rgba[i * 4] : null)
  const unionOps = [
    { op: 'add', chan: chanOf([1, 1, 0, 0]) },
    { op: 'add', chan: chanOf([0, 0, 1, 0]) },
    { op: 'sub', chan: chanOf([1, 0, 0, 0]) },
  ]
  const composed = composeChannels(unionOps, 4, 1)
  const peeled = composeChannels(unionOps.slice(0, 2), 4, 1)
  const netZero = composeChannels([unionOps[0], { op: 'sub', chan: chanOf([1, 1, 0, 0]) }], 4, 1)
  const floorKeeps = composeChannels([unionOps[2]], 4, 1, chanOf([1, 1, 1, 1]))
  check(
    'click-union: op stack replays in order (add∪add∖sub), peels, nets to empty, respects the floor',
    rAt(composed, 0) === 0 && rAt(composed, 1) === 255 && rAt(composed, 2) === 255 && rAt(composed, 3) === 0
      && rAt(peeled, 0) === 255 && netZero === null
      && rAt(floorKeeps, 0) === 0 && rAt(floorKeeps, 3) === 255,
    JSON.stringify({ composed: composed && [...composed.rgba].filter((_, i) => i % 4 === 0) }),
  )
  /* ── Candidate pick: the whole-scene mask never wins an ambiguous click ── */
  // Three 10x10 candidates: a 9% object, a 25% part, a 96% whole-scene guess
  // scored the way SAM scores a click into a field of near-identical objects
  // — the runaway outscores the object actually pointed at.
  const candidates = (fracs) => {
    const size = 100
    const data = new Uint8Array(size * fracs.length)
    fracs.forEach((f, c) => data.fill(1, c * size, c * size + Math.round(f * size)))
    return data
  }
  const fieldOfFlowers = candidates([0.09, 0.25, 0.96])
  const fieldCov = maskChannelCoverages(fieldOfFlowers, 10, 10, 3)
  const fieldScores = Float32Array.from([0.87, 0.71, 0.94])
  // A frame-filling close-up: every candidate is large and the biggest is the
  // real subject, so the argmax winner must survive the runaway test.
  const closeUpCov = maskChannelCoverages(candidates([0.91, 0.95, 0.98]), 10, 10, 3)
  const closeUpScores = Float32Array.from([0.6, 0.7, 0.9])
  check(
    'candidate pick: an ambiguous click rejects the whole-scene runaway, box/refine and all-large keep argmax',
    pickBestMask(fieldScores, { coverages: fieldCov, ambiguous: true }) === 0
      && pickBestMask(fieldScores, { coverages: fieldCov, ambiguous: false }) === 2
      && pickBestMask(fieldScores) === 2
      && pickBestMask(closeUpScores, { coverages: closeUpCov, ambiguous: true }) === 2,
    `coverages ${fieldCov.map((c) => `${(c * 100).toFixed(0)}%`).join('/')} · runaway ≥ ${RUNAWAY_COVERAGE * 100}%`,
  )

  const maskLike = { data: Uint8ClampedArray.from([255, 255, 255, 255, 0, 0, 0, 0]), width: 2, height: 1 }
  check(
    'click-union: maskToChannel + pointInMask (tolerance) agree with the RGBA mask',
    maskToChannel(maskLike)[0] === 255 && maskToChannel(maskLike)[1] === 0
      && pointInMask(maskLike, 0, 0, 0) === true && pointInMask(maskLike, 1, 0, 0) === false
      && pointInMask(maskLike, 1, 0, 1) === true && pointInMask(null, 0, 0) === false,
    'channel + hit tests agree',
  )

  // The optional fixture exercises the bounded RAW-container parser directly.
  // It deliberately avoids Playwright's slow multi-megabyte file-upload bridge;
  // after extraction, the JPEG preview follows the already-covered Blob decode
  // path in the browser suite below.
  if (RAW_FIXTURE) {
    if (!existsSync(RAW_FIXTURE)) throw new Error(`RAW_FIXTURE not found: ${RAW_FIXTURE}`)
    const rawFile = Bun.file(RAW_FIXTURE)
    Object.defineProperty(rawFile, 'name', { value: path.basename(RAW_FIXTURE) })
    const preview = await extractRawPreview(rawFile, { proxyMinEdge: 768 })
    check(
      'RAW: bounded parser extracts an embedded JPEG and a bounded proxy preview',
      !!preview && preview.width > 768 && preview.height > 0 && preview.blob.size > 0
        && (!preview.proxyBlob || preview.proxyBlob.size <= preview.blob.size),
      JSON.stringify({
        full: preview ? `${preview.width}x${preview.height}` : null,
        jpegBytes: preview?.blob.size || 0,
        proxyBytes: preview?.proxyBlob?.size || 0,
      }),
    )
  }

  /* ─── Phase Q: heavy-job queue contracts ────────────────────────────── */
  await run.enter('Q')
  {
    let active = 0
    let maxActive = 0
    const order = []
    const job = (name, ms) => async () => {
      active += 1
      maxActive = Math.max(maxActive, active)
      order.push(name)
      await new Promise((r) => setTimeout(r, ms))
      active -= 1
      return name
    }
    const p1 = enqueueHeavy('decode', job('decode', 30), { priority: 'import' })
    const pIdle = enqueueHeavy('prewarm', job('prewarm', 10), { priority: 'idle' })
    const pHigh = enqueueHeavy('segment', job('segment', 10), { priority: 'high' })
    await Promise.all([p1, pIdle, pHigh])
    check(
      'queue: one heavy job at a time; interaction preempts queued prewarm',
      maxActive === 1 && order[0] === 'decode' && order[1] === 'segment' && order[2] === 'prewarm',
      `maxActive=${maxActive} order=${order.join('→')}`,
    )
    const rejected = enqueueHeavy('boom', async () => { throw new Error('boom') })
    let threw = false
    await rejected.catch(() => { threw = true })
    const after = await enqueueHeavy('after', async () => 'ran')
    check('queue: a rejected job does not deadlock the scheduler', threw && after === 'ran', `threw=${threw} after=${after}`)

    // A job that never SETTLES — the case a rejection test cannot reach, and the
    // one that actually shipped: ownership was released in a `.finally` on the
    // task's own promise, so a decode worker killed without firing `onerror`
    // held the queue forever and every later import queued behind a selection
    // that could no longer finish (stuck "Selecting…" → import spinner forever).
    let wedgeSettled = null
    const wedged = enqueueHeavy('wedged', () => new Promise((r) => { wedgeSettled = r }), { timeoutMs: 150 })
    let wedgeErr = null
    await wedged.catch((e) => { wedgeErr = e })
    const afterWedge = await enqueueHeavy('after-wedge', async () => 'ran')
    check(
      'queue: a job that never settles cannot wedge the scheduler',
      /timed out/.test(String(wedgeErr?.message)) && afterWedge === 'ran',
      `err=${wedgeErr?.message} after=${afterWedge}`,
    )
    // …and its late settle must not evict whoever owns the slot by then.
    const successor = enqueueHeavy('successor', () => new Promise((r) => setTimeout(() => r('mine'), 60)))
    wedgeSettled?.('too late')
    const ownerDuring = getHeavyQueueState().activeLabel
    check(
      'queue: a late settle from an abandoned job cannot steal a successor’s slot',
      ownerDuring === 'successor' && (await successor) === 'mine',
      `owner=${ownerDuring}`,
    )

    const blocker = enqueueHeavy('blocker', () => new Promise((r) => setTimeout(r, 40)))
    const oldJob = enqueueHeavy('old', async () => 'old-ran', { revision: 1 })
    const staleJob = enqueueHeavy('stale', async () => 'stale-ran', { isCurrent: () => false })
    cancelHeavyBefore(2)
    const [oldRes, staleRes] = await Promise.all([oldJob, staleJob])
    await blocker
    check(
      'queue: cancelHeavyBefore + isCurrent stop queued jobs before they run',
      oldRes === STALE && staleRes === STALE && getHeavyQueueState().queuedCount === 0,
      `old=${String(oldRes?.stale)} stale=${String(staleRes?.stale)}`,
    )
  }

  /* ─── Phase S: memory-contract static scans ─────────────────────────── */
  await run.enter('S')
  {
    const jsFiles = ['app.js', 'asset-store.js', 'image-io.js', 'capability.js', 'policy.js',
      'sam-client.js', 'sam-core.js',
      'sam21-lane.js', 'sam21-host.js', 'sam21-client.js', 'sam21-adapter.js',
      'export-hd.js', 'yoloe-detect.js', 'detect-worker.js', 'embed-store.js', 'text-core.js',
      'ort-loader.js', 'search-taxonomy.js', 'mask-refine.js', 'sam21-store.js',
      'text-encode.js', 'clip-tokenizer.js', 'text-embed-store.js',
      'text-ui.js', 'image-raw.js', 'heavy-job-queue.js', 'decode-worker.js', 'decode-client.js',
      'decode-core.js', 'proxy-plan.js', 'cv-refine-client.js', 'cv-refine-worker.js',
      'raw-develop-client.js', 'raw-develop-worker.js']
    const sources = Object.fromEntries(jsFiles.map((f) => [f, readFileSync(path.join(ROOT, 'js', f), 'utf8')]))
    const all = Object.values(sources).join('\n') + readFileSync(path.join(ROOT, 'sw.js'), 'utf8')
      + readFileSync(path.join(ROOT, 'index.html'), 'utf8')
    check('static: no toDataURL in source', !/\btoDataURL\s*\(/.test(all), 'clean')
    check(
      'static: no Python runtime (Pyodide/PyScript/…) and no full OpenCV.js',
      !/pyodide|pyscript|micropython|brython|skulpt|transcrypt/i.test(all)
        && !/opencv[._-]?js|\bcv\.imread\b/i.test(all),
      'clean',
    )
    check(
      'static: no startup model warm — warm starts only after the proxy is shown',
      !/startupWarm/.test(sources['app.js'])
        && /ensureWarm\(\)/.test(sources['app.js'])
        && !/^bootProbe\.then\(\(\) => warmUp/m.test(sources['app.js']),
      'app.js warms post-import only',
    )
    // The four checks that used to live here all asserted properties of the
    // SlimSAM engine — that it stayed the only lane, that its OPFS persistence
    // was gated, that its device pick honoured the budget, that nothing
    // reintroduced a perf-based WASM demotion. That engine is gone, and with it
    // the states those checks were policing. What replaces them is stricter:
    // the same guarantees now hold by CONSTRUCTION, so what is asserted is the
    // absence of the machinery that could break them.
    check(
      // Structural, not textual: the historical comments that explain WHY the
      // WASM lane was abandoned (a ~3 GB ORT heap, measured) are worth keeping,
      // so this asserts nothing can LOAD it, not that nothing may mention it.
      'static: the SlimSAM stack is gone — no second engine, no transformers.js',
      !existsSync(path.join(ROOT, 'js', 'sam-engine.js'))
        && !existsSync(path.join(ROOT, 'js', 'sam-worker.js'))
        && !existsSync(path.join(ROOT, 'js', 'model-assets.js'))
        && !existsSync(path.join(ROOT, 'lib', 'transformers'))
        && !existsSync(path.join(ROOT, 'models', 'Xenova'))
        && !/from '\.\/(sam-engine|sam-worker|model-assets)\.js'/.test(all)
        && !/import\(['"]\.\/(sam-engine|model-assets)\.js['"]\)/.test(all)
        && !/lib\/transformers|Xenova\//.test(all),
      'one engine',
    )
    check(
      'static: one segmentation path — no lane switch, no transport choice',
      !/get\(['"]lane['"]\)|sam21Enabled\b/.test(all)
        && !/inlineEngine|workerBroken|MAX_WORKER_RESTARTS/.test(sources['sam-client.js'])
        && /sam21Segment/.test(sources['sam-client.js']),
      'sam-client routes to exactly one lane',
    )
    check(
      // The lane used to expose `configure(patch)` through the host and client,
      // so any tab could re-point the model or precision at runtime — while the
      // OPFS cache key that encodes precision was read once and memoised.
      'static: the lane is frozen — no runtime configure, no mutable model/precision',
      !/export const configure/.test(sources['sam21-lane.js'])
        && !/^\s*configure:/m.test(sources['sam21-host.js'])
        && /export const PRECISION = 'fp16'/.test(sources['sam21-lane.js'])
        && !/precisionPromise/.test(sources['sam21-client.js']),
      'model + precision are module constants',
    )
    check(
      // Measured: WASM SlimSAM held a ~3 GB ORT heap that never shrank, against
      // ~0.5 GB on WebGPU. The old defence was a policy rule — never demote a
      // working GPU for being slow. The new one is that there is no wasm
      // execution provider to demote TO: checkDevice refuses an adapter without
      // shader-f16 up front, so the lane is WebGPU or it does not run.
      'static: WebGPU is a precondition, not a preference — no wasm EP in the mask lane',
      /executionProviders: \[webgpuEP\(\)\]/.test(sources['sam21-lane.js'])
        && !/executionProviders[^\n]*wasm/.test(sources['sam21-lane.js'])
        && !/encoderEP|decoderEP/.test(sources['sam21-lane.js'])
        && /shader-f16'\)\) return \{ ok: false/.test(sources['sam21-lane.js']),
      'no device arbitration left to get wrong',
    )
    check(
      // One fallback, and it is the device-not-yet-built case after a worker
      // restart. Anything that adds a second branch here is a regression.
      'static: the encode has exactly one fallback (null device → CPU tensor)',
      /const preDev = captureDevice\(\)/.test(sources['sam21-lane.js'])
        && /built\?\.tensor \|\| toTensorCPU/.test(sources['sam21-lane.js']),
      'single fallback',
    )
    const assetGetImageData = (sources['asset-store.js'].match(/getImageData\(/g) || []).length
    check(
      'static: asset-store reads back only the 16×16 hash canvas, never a full frame',
      assetGetImageData === 1 && /getImageData\(0, 0, 16, 16\)/.test(sources['asset-store.js']),
      `getImageData sites=${assetGetImageData}`,
    )
    check(
      'static: wasm artifacts exist (cv-refine.js + cv-refine.wasm)',
      existsSync(path.join(ROOT, 'public/wasm/cv-refine.js')) && existsSync(path.join(ROOT, 'public/wasm/cv-refine.wasm')),
      'built',
    )
    check(
      'static: raw-develop wasm artifacts exist (raw-develop.js + raw-develop.wasm)',
      existsSync(path.join(ROOT, 'public/wasm/raw-develop.js')) && existsSync(path.join(ROOT, 'public/wasm/raw-develop.wasm')),
      'built',
    )
    check(
      'static: LibRaw develop is the preview-less fallback — lazy, disposed after use, off at pressure ≥ 2',
      /developRaw\(source, \{ budget: BUDGET \}\)/.test(sources['app.js'])
        && /worker\.terminate\(\)/.test(sources['raw-develop-client.js'])
        && /budget\.rawDevelop === false \|\| \(budget\.pressureLevel \|\| 0\) >= 2/.test(sources['raw-develop-client.js'])
        && /next\.rawDevelop = false/.test(sources['policy.js']),
      'wired only in the no-preview branch; terminates worker; pressure-gated',
    )
    check(
      'static: develop returns a JPEG Blob through the normal decode path (no full-res RGBA crosses threads)',
      /new Blob\(\[result\.jpeg\], \{ type: 'image\/jpeg' \}\)/.test(sources['raw-develop-client.js'])
        && !/getImageData|Uint8ClampedArray|RGBA/.test(sources['raw-develop-worker.js'])
        && /half_size/.test(readFileSync(path.join(ROOT, 'cpp/raw_develop.cpp'), 'utf8')),
      'jpeg blob only; half-size demosaic bounds the peak',
    )
    const sw = readFileSync(path.join(ROOT, 'sw.js'), 'utf8')
    check(
      'static: service worker caches only on demand and versions obsolete cache cleanup',
      /const CACHE_NAME = 'seglab-models-v5'/.test(sw)
        && /cache\.match\(request\)/.test(sw)
        && /cache\.put\(request, response\.clone\(\)\)/.test(sw)
        && !/cache\.addAll|event\.waitUntil\([^)]*fetch/i.test(sw),
      'cache-first after request; no install-time model/Wasm/detector prefetch',
    )
    check(
      // A CACHE_NAME bump alone does not update a model: /models/ is served
      // `immutable, max-age=31536000`, so the refill is answered from the HTTP
      // cache with the copy the bump exists to discard. The two must ship together.
      'static: a cache-generation bump actually refetches (reload bypasses the immutable HTTP cache)',
      /fetch\(request, \{ cache: 'reload' \}\)/.test(sw)
        && /seglab-models-\$\{?|startsWith\('seglab-models-'\)/.test(sw),
      'reload on refill + old generations evicted',
    )
    check(
      // Weights are keyed by a URL carrying the model's sha, so cache-first is
      // always right for them. model.json is where that sha COMES FROM: caching
      // it pins the app to whichever model was current when the entry landed.
      // Measured — a tiny export sat on disk while the app kept loading small.
      'static: version pointers are network-first, never served stale from the SW cache',
      /isVersionPointer/.test(sw)
        && /\(model\|manifest\)\\\.json\$/.test(sw)
        && /if \(isVersionPointer\(request\.url\)\)[\s\S]{0,300}fetch\(request, \{ cache: 'reload' \}\)/.test(sw)
        && /isVersionPointer\(request\.url\)[\s\S]{0,600}cache\.match\(request\)/.test(sw),
      'pointer bypasses cache-first; cache is the offline fallback only',
    )
    check(
      // ORT-Web 1.22 computes the WebGPU fp16 kernels WRONG and every model here
      // is fp16 or 4-bit, so a stray version literal is a silent-wrong-results
      // bug on whichever lane picks it up. Two lanes had already drifted back to
      // 1.22 on their CDN fallback, which only fires when the vendored copy is
      // missing — the one place nobody would look for a bad mask.
      'static: exactly one ORT version, and it is >= 1.24.3',
      (() => {
        const loader = sources['ort-loader.js']
        const v = loader.match(/export const ORT_VERSION = '([\d.]+)'/)?.[1]
        if (!v) return false
        const [maj, min, patch] = v.split('.').map(Number)
        const ok = maj > 1 || (maj === 1 && (min > 24 || (min === 24 && patch >= 3)))
        // Every other module must go through the loader, never name a version.
        const strays = Object.entries(sources)
          .filter(([f]) => f !== 'ort-loader.js')
          .filter(([, s]) => /onnxruntime-web@[\d]/.test(s))
        const vendored = readFileSync(path.join(ROOT, 'scripts', 'download-models.mjs'), 'utf8')
          .match(/const ORT_WEB_VERSION = '([\d.]+)'/)?.[1]
        return ok && strays.length === 0 && vendored === v
      })(),
      'one pin, shared by the loader and the vendoring script',
    )
    check(
      // The storage buffer cache mode is what keeps the app under the ceiling
      // (Bucket 1128 MB of GPU process vs 290 MB with lazyRelease, identical
      // logits), and ORT-Web will not forward it — the vendored bundle is
      // patched to. Nothing at runtime can tell the difference, so gate the
      // artifact, the patcher, and every WebGPU session that must ask for it.
      'static: the vendored ORT carries the epConfig patch, and only the mask lane uses it',
      (() => {
        const bundle = path.join(ROOT, 'lib', 'ort-web', 'ort.webgpu.bundle.min.mjs')
        if (!existsSync(bundle)) return 'skip'
        const patched = readFileSync(bundle, 'utf8').includes('S.epConfig')
        const vendor = readFileSync(path.join(ROOT, 'scripts', 'download-models.mjs'), 'utf8')
        const patcher = /S\.epConfig&&Object\.entries/.test(vendor) && /epConfig anchor missing/.test(vendor)
        const decl = /webgpuEP = \(\) => \(\{ name: 'webgpu', epConfig: \{ storageBufferCacheMode: 'lazyRelease' \} \}\)/
          .test(sources['ort-loader.js'])
        // Mask lane only, and provably so: the detector lane measured WORSE.
        const users = /webgpuEP\(\)/.test(sources['sam21-lane.js'])
          && !['yoloe-detect.js', 'text-encode.js'].some((f) => /webgpuEP/.test(sources[f]))
        return patched && patcher && decl && users
      })(),
      'the reclaim is invisible at runtime; the build is where it is provable',
    )
    check(
      // Releasing the encoder alone reclaims 37 MB; releasing every session
      // destroys the device and reclaims 976 MB, because ORT pools GPU memory
      // per device and any surviving session pins the pool. The decode ref is
      // what makes that safe — the idle release frees the embedding's buffers.
      'static: the idle release gives up the DEVICE, and cannot fire mid-decode',
      /const releaseIdle = \(\) => \{/.test(sources['sam21-lane.js'])
        && /releaseIdle[\s\S]{0,400}releaseAll\(\)/.test(sources['sam21-lane.js'])
        && /state\.decodeRefs \+= 1/.test(sources['sam21-lane.js'])
        && /state\.encoderRefs > 0 \|\| state\.decodeRefs > 0/.test(sources['sam21-lane.js'])
        && !/scheduleEncoderRelease/.test(sources['sam21-lane.js']),
      'releaseAll at idle, guarded by encode + decode refs',
    )
  }

  /* ─── The browser boundary ─────────────────────────────────────────────
     Everything above is node-only and shares nothing, so it runs lock-free —
     two --fast runs can sit side by side. Everything below wants the one
     Chromium profile on this machine, so the lease opens here: the shortest
     window that is still correct. */
  if (OPTS.fast) {
    run.finish('not requested (--fast)')
    log('--fast — stopping before the browser phases')
    console.log(run.summary())
    process.exit(results.some((r) => !r.ok) ? 1 : 0)
  }
  if (OPTS.isolated) {
    const snap = memorySnapshot(true)
    log(`--isolated — throwaway profile ${RUN_TAG}, no lease taken. ${formatMem(snap)}`)
    if (snap.availableMB < MIN_FREE_MB * 2) {
      run.note('isolated run on a machine that cannot hold two suites — cold-start phases will skip rather than swap')
    }
  } else {
    lease = await acquireLease({
      lockPath: LOCK_PATH,
      profileDir: PROFILE_DIR,
      cacheDir: CACHE_DIR,
      waitMs: OPTS.waitMs,
      force: OPTS.force,
      reap: OPTS.reap,
      log,
    })
  }

  await run.enter('A')
  context = guardContext(
    await withDeadline(
      chromium.launchPersistentContext(PROFILE_DIR, {
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-gpu'],
        timeout: LAUNCH_MS,
      }),
      LAUNCH_MS + 15_000,
      'chromium launch on the shared profile',
    ),
    { evaluateMs: TIMEOUT_MS, label: 'app' },
  )
  run.track(context, 'shared')

  // One reclaim path, used by every gate that needs the machine back. The
  // instance holds ~1.2 GB that only a teardown returns (§1.4), so "hand the
  // GPU back before a cold start" stops being something a phase must remember.
  run.setReclaim(() => withDeadline((async () => {
    const pg = await newAppPage(context, '', null)
    await pg.evaluate(async () => (await import('./js/sam21-client.js')).op('shutdown').catch(() => null))
    await pg.close()
    await new Promise((r) => { setTimeout(r, 1500) })
  })(), 90_000, 'instance teardown').catch((err) => log(`reclaim: ${err?.message || err}`)))

  // Cold embed store + session for the persistence phases. A stale session
  // would otherwise auto-restore into every later page load.
  const pageZ = await context.newPage()
  await pageZ.goto(`http://127.0.0.1:${port}/`)
  await pageZ.evaluate(async () => {
    const root = await navigator.storage.getDirectory()
    await root.removeEntry('seglab-embeds', { recursive: true }).catch(() => {})
    await root.removeEntry('seglab-session', { recursive: true }).catch(() => {})
    // The SAM 2.1 embedding cache lives in its OWN directory and was never
    // cleared, so it accumulated across runs and every "did this encode?"
    // assertion was reading uncontrolled state. It passed for two runs and then
    // failed once a previous run's entry survived the LRU into the next.
    await root.removeEntry('seglab-sam21', { recursive: true }).catch(() => {})
  })
  await pageZ.close()

  // Phosmith live-update contract (no model, no image).
  const pageHost = await context.newPage()
  await pageHost.goto(`http://127.0.0.1:${port}/`)
  await pageHost.waitForFunction(() => window.__seglabReady === true)
  const hostUpdate = await pageHost.evaluate(async (hostUltra) => {
    window.__PHOSMITH_DEVICE_RESOURCES__ = hostUltra
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', { detail: hostUltra }))
    const high = await window.__seglab.resourceBudget()
    window.__PHOSMITH_DEVICE_RESOURCES__ = { memoryBudgetGB: 4, mode: 'conservative' }
    window.dispatchEvent(new CustomEvent('phosmithresourceschange', {
      detail: window.__PHOSMITH_DEVICE_RESOURCES__,
    }))
    const low = await window.__seglab.resourceBudget()
    return { high, low }
  }, HOST_ULTRA)
  check(
    'Phosmith event: a live resource change is applied without enabling SAM3 (config is fixed)',
    hostUpdate.high.profile === 'standard8' && hostUpdate.high.exportMaxMP === 12
      && hostUpdate.high.flagship === false
      && hostUpdate.low.profile === 'standard8' && hostUpdate.low.exportMaxMP === 12
      && hostUpdate.low.flagship === false,
    JSON.stringify({ high: hostUpdate.high.profile, low: hostUpdate.low.profile }),
  )
  await pageHost.close()

  /* ─── Phase A: lite (memory-locked) — the bounded-memory contract ─────
     Pinned to lite so this strictest-floor contract is tested deterministically
     even on a capable CI machine that would otherwise autoTier to standard8. */
  const page = await newAppPage(context, '?flagship=0', null, { pin: 'lite' })
  const bootBudget = await page.evaluate(() => window.__seglab.resourceBudget())
  // Boot now builds BOTH sessions with no image on screen, so the first click
  // pays neither the download nor the shader compile. The encoder is the ~1 GB
  // resident; holding it from boot is the deliberate trade for an instant first
  // click, and the governor + host idle exit remain its bounds.
  const bootState = await page.evaluate(async () => {
    const t0 = Date.now()
    const lane = async () => (await window.__seglab.engineState().catch(() => null))?.lane || {}
    while (!(await lane()).encoder && Date.now() - t0 < 120_000) {
      await new Promise((r) => setTimeout(r, 250))
    }
    const s = window.__seglab.state()
    const l = await lane()
    return { ready: s.ready, mode: s.mode, hasImage: s.hasImage, encoder: l.encoder, decoder: l.decoder }
  })
  check(
    'boot: both sessions stand with no image; the single config is bounded',
    bootState.ready === true && bootState.mode !== null && bootState.hasImage === false
      && bootState.decoder === true && bootState.encoder === true
      && bootBudget.profile === 'standard8' && bootBudget.memoryLocked === true && bootBudget.proxyMax === 1024,
    JSON.stringify({ ...bootState, profile: bootBudget.profile, proxyMax: bootBudget.proxyMax }),
  )
  const swSmoke = await page.evaluate(async () => {
    const registration = await navigator.serviceWorker?.getRegistration()
    await navigator.serviceWorker?.ready
    return { registered: !!registration, active: !!registration?.active }
  })
  check(
    'service worker: the real app boots headless and registers the on-demand cache worker',
    swSmoke.registered && swSmoke.active,
    JSON.stringify(swSmoke),
  )
  // 3000, not 1400: the per-axis rule (§8.1a) sizes to the SHORT edge, so a
  // 1400 px source now lands under the cap and is native — no proxy to test.
  await page.evaluate(() => window.__seglab.loadDemo(3000))
  const liteFrame = await page.evaluate(() => window.__seglab.imageTransform())
  check(
    'lite: an oversize source is proxied within the per-axis bound (short ≤1024, long ≤2048, ≤2.1 MP)',
    liteFrame && liteFrame.proxyActive === true
      && Math.min(liteFrame.proxyW, liteFrame.proxyH) <= 1024
      && Math.max(liteFrame.proxyW, liteFrame.proxyH) <= 2048
      && liteFrame.proxyW * liteFrame.proxyH <= 2_100_000,
    JSON.stringify({ proxy: `${liteFrame?.proxyW}x${liteFrame?.proxyH}` }),
  )
  const disp = await page.evaluate(() => ({
    photo: document.getElementById('photo').width,
    view: document.getElementById('view').width,
    overlay: document.getElementById('overlay').width,
    bound: Math.round(Math.max(window.innerWidth, window.innerHeight) * Math.min(window.devicePixelRatio || 1, 3) * 1.5),
  }))
  check(
    'display: crisp preview out-resolves the proxy, stays within the viewport-anchored bound; overlay tracks the model buffer',
    disp.photo > disp.view && disp.overlay === disp.view
      && disp.photo <= Math.max(disp.bound, disp.view) && disp.photo <= 2048,
    JSON.stringify(disp),
  )
  const idleUi = await page.evaluate(() => ({
    prepHidden: document.getElementById('prep')?.hidden,
    stageVisible: document.getElementById('stage')?.classList.contains('visible'),
    pickEnabled: !document.getElementById('pick')?.disabled,
  }))
  check('lite: editor is usable while the model warms in the background',
    idleUi.prepHidden && idleUi.stageVisible && idleUi.pickEnabled, JSON.stringify(idleUi))
  const textModeUi = await page.evaluate(() => {
    document.getElementById('mode-text')?.click()
    const tolerance = document.getElementById('tolerance-wrap')
    const result = { hidden: tolerance?.hidden, display: getComputedStyle(tolerance).display }
    document.getElementById('mode-click')?.click()
    return result
  })
  check(
    'text mode: colour tolerance is hidden and cannot be mistaken for text confidence',
    textModeUi.hidden === true && textModeUi.display === 'none',
    JSON.stringify(textModeUi),
  )
  log('phase A (lite) — eager encode warms model + embedding at import…')
  const eager = await page.evaluate(() => window.__seglab.eagerEncode())
  const engEager = await page.evaluate(() => window.__seglab.engineState())
  const chipDevice = await page.evaluate(() => document.getElementById('chip-device')?.textContent || '')
  check(
    // On webgpu the prewarm lands (one embedding). On a wasm lane the engine
    // REFUSES the speculative encode on an unverified budget (gpuOnly guard:
    // a prewarm nobody asked for must not allocate the multi-GB WASM arena);
    // the first real click pays it knowingly.
    'boot: eager encode lands on webgpu',
    engEager.device === 'webgpu' && eager && eager.stale !== true && eager.encoded === true,
    JSON.stringify({ eager, cached: engEager?.cachedImages, device: engEager?.device }),
  )
  // Chrome ≥149 headless ships a real hardware adapter, so the probe may
  // legitimately land on webgpu; the contract is honesty — the chip reports
  // whichever lane actually runs (policy: a probed GPU is always allowed).
  check(
    'boot: SAM device is the probed lane and the chip reports it honestly',
    (engEager.device === 'wasm' || engEager.device === 'webgpu')
      && chipDevice.includes(`device: ${engEager.device}`),
    `device=${engEager.device} chip="${chipDevice}"`,
  )

  const geo = await page.evaluate(() => window.__seglab.demoGeometry())
  const p = geo.proxyScale
  const sA = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.disc.x * p, y: geo.disc.y * p })
  checkDisc('lite draft', sA, geo.disc.x * p, geo.disc.y * p, DISC_FRAC)
  check(
    // webgpu: the eager embedding serves it (decode-only). wasm: the gpuOnly
    // guard deferred the prewarm, so the first click pays the encode knowingly.
    'first click uses the eager embedding (webgpu) or pays the deferred encode (wasm)',
    sA.lastRun && sA.lastRun.encoded === (engEager.device === 'webgpu' ? false : true),
    `device=${engEager.device}, encoded=${sA.lastRun?.encoded}, decode ${sA.lastRun?.decodeMs}ms`,
  )
  const statsA = await page.evaluate(() => window.__seglab.maskStats())
  check('hygiene: single component, no crumbs', statsA && statsA.components === 1, `components=${statsA?.components}`)
  check('edge refinement: soft boundary band present', statsA && statsA.softPixels > 100, `softPixels=${statsA?.softPixels}`)

  const sA2 = await page.evaluate(() => window.__seglab.clickAt(50, 50, true))
  check(
    'repeat click skips the encoder (cache hit)',
    sA2.lastRun && sA2.lastRun.encoded === false,
    `encoded=${sA2.lastRun?.encoded}, decode ${sA2.lastRun?.decodeMs}ms`,
  )

  // Click-union: separate objects accumulate; exclude carves one out; Z peels.
  await page.evaluate(() => window.__seglab.reset())
  const sqc = { x: (geo.square.x + geo.square.w / 2) * p, y: (geo.square.y + geo.square.h / 2) * p }
  const u1 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.disc.x * p, y: geo.disc.y * p })
  const u2 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), sqc)
  const statsU2 = await page.evaluate(() => window.__seglab.maskStats())
  const covU1 = u1.maskSummary?.coverage || 0
  check(
    'click-union: clicking a second object ADDS it (commit + two components), never replaces',
    u2.baseOps === 1 && u2.clicks === 1 && statsU2?.components === 2
      && (u2.maskSummary?.coverage || 0) > covU1 * 1.3,
    `baseOps=${u2.baseOps} components=${statsU2?.components} coverage ${(covU1 * 100).toFixed(2)}%→${((u2.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const u3 = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y, true), { x: geo.disc.x * p, y: geo.disc.y * p })
  const statsU3 = await page.evaluate(() => window.__seglab.maskStats())
  check(
    'click-union: exclude on a committed object carves it out; the other object survives',
    u3.baseOps === 2 && statsU3?.components === 1
      && (u3.maskSummary?.coverage || 0) < (u2.maskSummary?.coverage || 0)
      && (u3.maskSummary?.coverage || 0) > 0,
    `baseOps=${u3.baseOps} components=${statsU3?.components} coverage ${((u3.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const z1 = await page.evaluate(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z' }))
    return window.__seglab.state()
  })
  const z2 = await page.evaluate(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z' }))
    return window.__seglab.state()
  })
  check(
    'click-union: Z discards the live object first, then peels committed ops in order',
    // z1: live square gone; base nets ~empty (add(disc)∖sub(disc) may leave
    // a sub-pixel residue where the two decodes disagree). z2: disc returns.
    z1.clicks === 0 && (z1.maskSummary?.coverage || 0) < covU1 * 0.15 && z1.baseOps === 2
      && z2.baseOps === 1 && Math.abs((z2.maskSummary?.coverage || 0) - covU1) < covU1 * 0.2,
    JSON.stringify({ z1: { clicks: z1.clicks, baseOps: z1.baseOps, coverage: z1.maskSummary?.coverage || 0 }, z2: { baseOps: z2.baseOps, coverage: z2.maskSummary?.coverage } }),
  )

  // Minute object.
  await page.evaluate(() => window.__seglab.reset())
  const sDot = await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geo.dot.x * p, y: geo.dot.y * p })
  const dotFrac = (Math.PI * DOT.r * DOT.r) / FRAME
  const bDot = sDot.maskSummary?.bbox || [0, 0, -1, -1]
  check(
    'minute object (r=9px logical) is selectable',
    sDot.maskSummary && bDot[0] <= geo.dot.x * p && geo.dot.x * p <= bDot[2] && sDot.maskSummary.coverage < dotFrac * 40,
    `coverage ${((sDot.maskSummary?.coverage || 0) * 100).toFixed(2)}% (dot ${(dotFrac * 100).toFixed(3)}%)`,
  )

  // Lasso clamp.
  await page.evaluate(() => window.__seglab.reset())
  const lassoR = SQUARE.half * 1.5 * (geo.originalW / 900) * p
  const sqx = geo.square.x + geo.square.w / 2
  const sqy = geo.square.y + geo.square.h / 2
  const sLasso = await page.evaluate(
    ({ x, y, r }) => window.__seglab.lassoCircle(x, y, r),
    { x: sqx * p, y: sqy * p, r: lassoR },
  )
  const bL = sLasso.maskSummary?.bbox || [0, 0, -1, -1]
  const clampR = lassoR + 30
  const inClamp = bL[0] >= sqx * p - clampR && bL[2] <= sqx * p + clampR
    && bL[1] >= sqy * p - clampR && bL[3] <= sqy * p + clampR
  check(
    'lasso snaps to the square and stays clamped',
    sLasso.maskSummary && inClamp && bL[0] <= sqx * p && sqx * p <= bL[2],
    `bbox [${bL.map((v) => Math.round(v))}] within ±${Math.round(clampR)} of (${(sqx * p).toFixed(0)},${(sqy * p).toFixed(0)})`,
  )

  // Manual/predictive modes (proxy coords).
  await page.evaluate(() => window.__seglab.reset())
  const region = await page.evaluate(
    ({ x, y, r }) => window.__seglab.manualRegionCircle(x, y, r),
    { x: geo.disc.x * p, y: geo.disc.y * p, r: geo.disc.r * 0.7 * p },
  )
  const regionArea = Math.PI * (DISC.r * 0.7) ** 2 / FRAME
  check(
    'manual region: drawn area is selected without object snapping',
    region.manual === 'region' && region.components === 1
      && region.maskSummary.coverage > regionArea * 0.85 && region.maskSummary.coverage < regionArea * 1.15,
    `coverage ${((region.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs drawn ${(regionArea * 100).toFixed(1)}%`,
  )
  await page.evaluate(() => window.__seglab.reset())
  const colour = await page.evaluate(({ x, y }) => window.__seglab.colorAt(x, y, 24), { x: geo.disc.x * p, y: geo.disc.y * p })
  check(
    'color range: matching pixels select without object inference',
    colour.manual === 'color' && colour.components === 1
      && colour.maskSummary.coverage > DISC_FRAC * 0.8 && colour.maskSummary.coverage < DISC_FRAC * 1.2,
    `coverage ${((colour.maskSummary?.coverage || 0) * 100).toFixed(1)}% vs disc ${(DISC_FRAC * 100).toFixed(1)}%`,
  )
  const manualExport = await page.evaluate(() => window.__seglab.exportCutout())
  check(
    'manual masks: export preserves the drawn boundary without AI refinement',
    manualExport && manualExport.decoded === false
      && Math.abs(manualExport.coverage - colour.maskSummary.coverage) < 0.01,
    `decoded=${manualExport?.decoded}, coverage ${(manualExport?.coverage * 100 || 0).toFixed(1)}%`,
  )

  // Include/exclude accumulation.
  await page.evaluate(() => window.__seglab.reset())
  const addA = await page.evaluate(() => window.__seglab.manualRect(50, 50, 150, 150))
  const addB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150))
  const areaA = addA.maskSummary?.coverage || 0
  check(
    'include: a second region unions into the mask (two components)',
    addB.components === 2 && Math.abs((addB.maskSummary?.coverage || 0) - areaA * 2) < areaA * 0.1,
    `A ${(areaA * 100).toFixed(2)}% → A∪B ${((addB.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  const subB = await page.evaluate(() => window.__seglab.manualRect(300, 50, 400, 150, true))
  check(
    'exclude: re-applying the region as negative carves it back out',
    subB.components === 1 && Math.abs((subB.maskSummary?.coverage || 0) - areaA) < areaA * 0.1,
    `A∪B∖B ${((subB.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )

  // One-embedding contract across imports (same page, second document).
  await page.evaluate(() => window.__seglab.loadDemo(1200))
  const geoB = await page.evaluate(() => window.__seglab.demoGeometry())
  await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoB.disc.x * geoB.proxyScale, y: geoB.disc.y * geoB.proxyScale })
  const engAfter = await page.evaluate(() => window.__seglab.engineState())
  check(
    'embedding: exactly one resident embedding after a second import + click',
    engAfter && engAfter.cachedImages === 1,
    JSON.stringify(engAfter),
  )
  const opfsFiles = await page.evaluate(async () => {
    try {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('seglab-embeds')
      const names = []
      for await (const [name] of dir.entries()) names.push(name)
      return names
    } catch { return [] }
  })
  check('embedding: nothing persisted to OPFS in the lite policy', opfsFiles.length === 0, `files=${JSON.stringify(opfsFiles)}`)

  // Queue serialization + decode path over the real Blob upload route.
  await page.evaluate((side) => window.__seglab.loadDemoBlob(side), 3000)
  const geoQ = await page.evaluate(() => window.__seglab.demoGeometry())
  await page.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoQ.disc.x * geoQ.proxyScale, y: geoQ.disc.y * geoQ.proxyScale })
  const qlog = await page.evaluate(() => window.__seglab.queueLog())
  const qframe = await page.evaluate(() => window.__seglab.imageTransform())
  check(
    'queue: blob import runs decode-proxy as a heavy job; proxy stays within the per-axis bound',
    qlog.some((e) => e.label === 'decode-proxy' && e.outcome === 'done')
      && qlog.some((e) => e.label === 'model-warm')
      && qframe && Math.min(qframe.proxyW, qframe.proxyH) <= 1024
      && Math.max(qframe.proxyW, qframe.proxyH) <= 2048
      && qframe.proxyW * qframe.proxyH <= 2_100_000,
    `labels=${[...new Set(qlog.map((e) => e.label))].join(',')} proxy=${qframe?.proxyW}x${qframe?.proxyH}`,
  )
  // Rapid double import: the first must never commit.
  const raced = await page.evaluate(async () => {
    const a = window.__seglab.loadDemoBlob(2600)
    await new Promise((r) => setTimeout(r, 120)) // A has begun; B supersedes it
    const b = window.__seglab.loadDemoBlob(2000)
    await Promise.all([a, b])
    return window.__seglab.imageTransform()
  })
  check(
    'queue: a second import invalidates the prior queued import',
    raced && raced.originalW === 2000,
    `final original=${raced?.originalW}`,
  )
  await page.close()

  // Unsafe URL flags in the real app (pinned lite so the lockout is tested on
  // the known floor: URL params must not raise a locked budget).
  const pageFlags = await newAppPage(context, '?flagship=1&profile=ultra&proxy=max&working=1', 4000, { pin: 'lite' })
  const flagBudget = await pageFlags.evaluate(() => window.__seglab.resourceBudget())
  const flagFrame = await pageFlags.evaluate(() => window.__seglab.imageTransform())
  check(
    'safety (live): unsafe URL flags are refused on a locked budget — no flagship, bounded proxy, not raised to ultra',
    // ?proxy=max asks for 4096; the lock must hold it to the per-axis bound.
    flagBudget.profile === 'standard8' && flagBudget.flagship === false && flagBudget.proxyMax === 1024
      && flagFrame && Math.min(flagFrame.proxyW, flagFrame.proxyH) <= 1024
      && Math.max(flagFrame.proxyW, flagFrame.proxyH) <= 2048
      && flagFrame.proxyW * flagFrame.proxyH <= 2_100_000,
    JSON.stringify({ profile: flagBudget.profile, flagship: flagBudget.flagship, proxy: `${flagFrame?.proxyW}x${flagFrame?.proxyH}` }),
  )
  await pageFlags.close()

  /* ── Revision/cancel: overlapping prompts, one commit ── */
  const pageC = await newAppPage(context, '?flagship=0', 920)
  log('phase A2 (revision/cancel) — second click lands while the first is in flight…')
  const geoC = await pageC.evaluate(() => window.__seglab.demoGeometry())
  const m0 = await pageC.evaluate(async ({ disc, sq, scale }) => {
    const a = window.__seglab.clickAt(disc.x * scale, disc.y * scale)
    await new Promise((r) => setTimeout(r, 350))
    const b = window.__seglab.clickAt(sq.x * scale, sq.y * scale)
    await Promise.all([a, b])
    return { log: window.__seglab.commitLog.slice(), revision: window.__seglab.revision() }
  }, {
    disc: { x: geoC.disc.x, y: geoC.disc.y },
    sq: { x: geoC.square.x + geoC.square.w / 2, y: geoC.square.y + geoC.square.h / 2 },
    scale: geoC.proxyScale,
  })
  const committed = m0.log.filter((e) => e.outcome === 'committed')
  check(
    'revision/cancel: only the newest prompt commits',
    committed.length === 1 && committed[0].revision === m0.revision
      && m0.log.some((e) => e.revision < m0.revision && (e.outcome === 'stale' || e.outcome === 'superseded')),
    `log ${JSON.stringify(m0.log)} (current revision ${m0.revision})`,
  )
  await pageC.close()

  /* ── Export caps by tier: lite bounds to ≤8 MP from the proxy (no re-decode);
       standard8 unlocks a bigger, HD-decoded cutout (the quality win). ── */
  const pageX = await newAppPage(context, '?flagship=0', 4200, { pin: 'lite' })
  const geoX = await pageX.evaluate(() => window.__seglab.demoGeometry())
  await pageX.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoX.disc.x * geoX.proxyScale, y: geoX.disc.y * geoX.proxyScale })
  const exLite = await pageX.evaluate(() => window.__seglab.exportCutout())
  check(
    'export: bounded to the single config (≤5120 px / ≤12 MP), coverage sane',
    exLite && exLite.w <= 5120 && exLite.h <= 5120 && (exLite.w * exLite.h) <= 12.05e6
      && exLite.w < geoX.originalW && exLite.coverage > 0.01,
    `export ${exLite?.w}×${exLite?.h} (${((exLite?.w * exLite?.h || 0) / 1e6).toFixed(1)} MP), decoded=${exLite?.decoded}`,
  )
  await pageX.close()

  // standard8 (pinned) exports the SAME 4200 px source larger and HD-decoded —
  // the adaptive quality unlock a capable device now reaches automatically.
  const pageX8 = await newAppPage(context, '?flagship=0', 4200, { pin: 'standard8' })
  const budX8 = await pageX8.evaluate(() => window.__seglab.resourceBudget())
  const geoX8 = await pageX8.evaluate(() => window.__seglab.demoGeometry())
  await pageX8.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoX8.disc.x * geoX8.proxyScale, y: geoX8.disc.y * geoX8.proxyScale })
  const exStd8 = await pageX8.evaluate(() => window.__seglab.exportCutout())
  check(
    // One config, so this no longer contrasts with a lower tier — it asserts the
    // single export bound and that HD decode is on (§11).
    // decoded=false is correct on the SAM 2.1 lane: the export alpha is derived
    // from the continuous score field at native resolution and guided-filtered
    // against the native crop (js/sam21-adapter.js), so there is nothing to
    // re-decode. The old native re-decode belonged to the SlimSAM crop path.
    'export: cutout bounded at ≤5120 px / ≤12 MP, identical on every device',
    budX8.profile === 'standard8' && budX8.exportMaxMP === 12
      && exStd8 && exStd8.w <= 5120 && (exStd8.w * exStd8.h) <= 12.1e6
      && (exStd8.w * exStd8.h) === (exLite.w * exLite.h) // same config → same bound
      && exStd8.coverage > 0.01,
    `std8 export ${exStd8?.w}×${exStd8?.h} (${((exStd8?.w * exStd8?.h || 0) / 1e6).toFixed(1)} MP, decoded=${exStd8?.decoded}) vs lite ${((exLite.w * exLite.h) / 1e6).toFixed(1)} MP`,
  )
  await pageX8.close()

  /* ── Wasm cv-refine gates (through the real worker + transferables) ── */
  const pageV = await newAppPage(context, '?flagship=0', 900)
  log('phase A5 (wasm cv-refine) — hole fill, min-area, seeded cleanup, bounds…')
  const mk = (w, h, fn) => Array.from({ length: w * h }, (_, i) => (fn(i % w, (i / w) | 0) ? 255 : 0))
  const holeIn = mk(64, 64, (x, y) => x > 10 && x < 54 && y > 10 && y < 54 && !(x >= 30 && x < 35 && y >= 30 && y < 35))
  const holeOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: holeIn, width: 64, height: 64, seeds: [[20, 20]], options: { minArea: 0 },
  })
  check(
    'wasm: hole fill closes an interior pinhole, keeps the outside empty',
    holeOut.available && holeOut.alpha && holeOut.alpha[32 * 64 + 32] === 255 && holeOut.alpha[0] === 0,
    `available=${holeOut.available} center=${holeOut.alpha?.[32 * 64 + 32]}`,
  )
  const crumbs = mk(96, 96, (x, y) => (x > 10 && x < 60 && y > 10 && y < 60) || (x >= 80 && x < 83 && y >= 80 && y < 83))
  const crumbOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: crumbs, width: 96, height: 96, seeds: [[30, 30]], options: { minArea: 20 },
  })
  check(
    'wasm: components below min-area are removed, the seeded object stays',
    crumbOut.alpha && crumbOut.alpha[81 * 96 + 81] === 0 && crumbOut.alpha[30 * 96 + 30] === 255,
    `crumb=${crumbOut.alpha?.[81 * 96 + 81]} seed=${crumbOut.alpha?.[30 * 96 + 30]}`,
  )
  const twoComps = mk(96, 96, (x, y) => (x > 5 && x < 40 && y > 5 && y < 40) || (x > 55 && x < 90 && y > 55 && y < 90))
  const seededOut = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: twoComps, width: 96, height: 96, seeds: [[20, 20]], options: { minArea: 0 },
  })
  check(
    'wasm: seeded cleanup retains the seeded component, removes the other',
    seededOut.alpha && seededOut.alpha[20 * 96 + 20] === 255 && seededOut.alpha[70 * 96 + 70] === 0,
    `seed=${seededOut.alpha?.[20 * 96 + 20]} other=${seededOut.alpha?.[70 * 96 + 70]}`,
  )
  const tooBig = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: [255], width: 2048, height: 1, seeds: [], options: {},
  })
  const badBuf = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: [255, 255], width: 5, height: 5, seeds: [], options: {},
  })
  const stillWorks = await pageV.evaluate((payload) => window.__seglab.cvRefine(payload), {
    alpha: mk(8, 8, (x, y) => x > 1 && y > 1), width: 8, height: 8, seeds: [], options: { minArea: 0 },
  })
  check(
    'wasm: >1024 px and mismatched buffers are rejected without leaking worker state',
    tooBig.alpha === null && badBuf.alpha === null && Array.isArray(stillWorks.alpha),
    `tooBig=${tooBig.alpha} badBuf=${badBuf.alpha} recovered=${Array.isArray(stillWorks.alpha)}`,
  )
  const detached = await pageV.evaluate(async () => {
    const { refineAlpha } = await import('./js/cv-refine-client.js')
    const alpha = new Uint8Array(16 * 16).fill(255)
    const out = await refineAlpha({ alpha, width: 16, height: 16, seeds: [], options: { minArea: 0 }, budget: { cvRefine: true } })
    return { detached: alpha.buffer.byteLength === 0, got: !!out }
  })
  check(
    'wasm: request buffers TRANSFER (input detached), result returns a new buffer',
    detached.detached === true && detached.got === true,
    JSON.stringify(detached),
  )
  await pageV.close()

  /* ── LibRaw develop fallback (fixture-gated): real sensor → JPEG on-device ── */
  if (RAW_FIXTURE) {
    const pageR = await newAppPage(context, '?flagship=0', null)
    log('phase A5b (raw develop) — LibRaw demosaic + libjpeg encode in a disposed worker…')
    const dev = await pageR.evaluate(() => window.__seglab.developRawUrl('/__raw_fixture'))
    check(
      'raw develop: preview-less fallback demosaics the sensor to a decodable JPEG on-device',
      dev.ok && dev.w > 0 && dev.h > 0 && dev.jpegBytes > 0
        && dev.decodedW === dev.w && dev.decodedH === dev.h,
      JSON.stringify(dev),
    )
    await pageR.close()
  }

  /* ── Text-select plumbing + brush (proxy coords) ── */
  const pageE = await newAppPage(context, '?flagship=0', 900)
  log('phase A6 (text plumbing + brush) — box → mask → union…')
  const geoE = await pageE.evaluate(() => window.__seglab.demoGeometry())
  const pE = geoE.proxyScale
  const discBox = [120 * pE, 230 * pE, 340 * pE, 450 * pE]
  const squareBox = [515 * pE, 145 * pE, 715 * pE, 345 * pE]
  const one = await pageE.evaluate((b) => window.__seglab.selectBoxes([b]), discBox)
  checkDisc('text', one, geoE.disc.x * pE, geoE.disc.y * pE, DISC_FRAC)
  await pageE.evaluate(() => window.__seglab.reset())
  const both = await pageE.evaluate(([a, b]) => window.__seglab.selectBoxes([a, b]), [discBox, squareBox])
  check('text select: "all" unions instances → 2 components', both && both.components === 2, `components=${both?.components}`)
  await pageE.evaluate(() => window.__seglab.reset())
  const brushAdd = await pageE.evaluate((points) => window.__seglab.brushStroke(points), [[333, 256], [418, 256], [478, 294]])
  const brushErase = await pageE.evaluate((points) => window.__seglab.brushStroke(points, true), [[333, 256], [371, 256]])
  check(
    'brush: canvas stroke commits a mask and erases incrementally',
    brushAdd?.manual === 'brush' && brushAdd.maskSummary?.coverage > 0
      && brushErase?.manual === 'brush' && brushErase.maskSummary?.coverage > 0
      && brushErase.maskSummary.coverage < brushAdd.maskSummary.coverage,
    `add=${((brushAdd?.maskSummary?.coverage || 0) * 100).toFixed(2)}% erase=${((brushErase?.maskSummary?.coverage || 0) * 100).toFixed(2)}%`,
  )
  // The phrase→boxes step used to be skipped "to avoid a 151 MB model pull".
  // The YOLOE and YOLO-World weights are in models/, so the pull never applied
  // and a whole user-facing feature simply had no coverage. It is gated now.
  // NOTE: a single match AUTO-SELECTS, which clears the candidate list — so the
  // MASK is the evidence, not the candidate count. Reading the count instead is
  // what made this look broken when it was not.
  const pageT = await newAppPage(context, '', null)
  await pageT.evaluate(async () => {
    const blob = await (await fetch('/spikes/yoloe/bus.jpg')).blob()
    const dt = new DataTransfer()
    dt.items.add(new File([blob], 'bus.jpg', { type: 'image/jpeg' }))
    const i = document.getElementById('file')
    i.files = dt.files
    i.dispatchEvent(new Event('change', { bubbles: true }))
  })
  await pageT.waitForFunction(() => window.__seglab.state().hasImage === true && !!window.__seglab.imageTransform(), null, { timeout: 120_000 })
  const phrase = async (q) => pageT.evaluate(async (t) => {
    await window.__seglab.reset()
    const d = await window.__seglab.detect(t)
    // A single match auto-selects and runDetect does NOT await it, so poll for
    // the mask rather than guessing a delay — the first call also pays the
    // detector's cold load, which a fixed wait cannot cover.
    const deadline = performance.now() + 30000
    let s = window.__seglab.state()
    // The early exit must mean "the app SAID it found nothing", which it says in
    // the status line. It used to key on `candidates === 0` — but a single match
    // auto-selects and clears the list, so that is true on the success path too,
    // and any phrase whose mask took longer than the 3 s grace (the first one
    // always does: cold detector session) was scored as a miss. Measured: "bus"
    // reported 0.0% while the very same click reached 30.2% at 965 ms.
    const saidNoMatch = () => (document.getElementById('status')?.textContent || '').startsWith('No matches')
    while (performance.now() < deadline && !(s?.maskSummary?.coverage > 0)) {
      await new Promise((r) => setTimeout(r, 250))
      s = window.__seglab.state()
      if (saidNoMatch() && !s?.running) break
    }
    // Backend and raw top score: a wasm-EP fallback scores this graph very
    // differently from WebGPU, and without them a miss looks like a lane bug.
    const raw = await window.__seglab.testDetectRaw(t, 0.001)
    return {
      cov: s?.maskSummary?.coverage || 0, score: s?.score || 0,
      cands: d?.candidates ?? -1, backend: raw?.backend || '?', top: raw?.top?.[0] ?? -1,
    }
  }, q)
  const tBus = await phrase('bus')
  const tOpen = await phrase('vehicle')
  const tNone = await phrase('unicorn')
  check(
    'text search: a real phrase detects and selects the object',
    tBus.cov > 0.05 && tBus.score > 0.5,
    `"bus" → ${(tBus.cov * 100).toFixed(1)}% of frame, score ${tBus.score.toFixed(2)},`
    + ` candidates=${tBus.cands} detector=${tBus.backend} raw=${tBus.top}`,
  )
  check(
    // The open-vocab lane's whole point: a word the baked vocab does not carry
    // still resolves, via the precomputed CLIP embeddings.
    'text search: an open-vocabulary synonym resolves to the same object',
    Math.abs(tOpen.cov - tBus.cov) < 0.02 && tOpen.cov > 0.05,
    `"vehicle" → ${(tOpen.cov * 100).toFixed(1)}% vs "bus" ${(tBus.cov * 100).toFixed(1)}%`,
  )
  check(
    'text search: a phrase with no match says so instead of selecting something',
    tNone.cov === 0,
    `"unicorn" → ${(tNone.cov * 100).toFixed(1)}%`,
  )
  await pageT.evaluate(async () => (await import('./js/sam21-client.js')).op('shutdown').catch(() => null))
  await pageT.close()
  await pageE.close()

  /* ── Weak device: everything still works on forced WASM ── */
  const pageW = await newAppPage(context, '?force=wasm', 760)
  log('phase A7 (weak-device) — full pipeline forced onto wasm…')
  const geoW = await pageW.evaluate(() => window.__seglab.demoGeometry())
  const pW = geoW.proxyScale
  const wClick = await pageW.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoW.disc.x * pW, y: geoW.disc.y * pW })
  await pageW.evaluate(() => window.__seglab.reset())
  const wText = await pageW.evaluate((b) => window.__seglab.selectBoxes([b]), [120 * pW, 230 * pW, 340 * pW, 450 * pW])
  const wExport = await pageW.evaluate(() => window.__seglab.exportCutout())
  check(
    // §4: WebGPU is a HARD requirement for the mask lane, so ?force=wasm no
    // longer produces a WASM segmentation lane — the point is that the rest of
    // the app (click, text, export) still completes.
    'forced-wasm flag: click + text + export all still complete',
    !!wClick.maskSummary && !!wText.maskSummary && wExport && wExport.coverage > 0,
    `device=${wClick.device} export=${wExport?.w}×${wExport?.h}`,
  )
  const freed = await pageW.evaluate(() => window.__seglab.relievePressure(3))
  check(
    // 'arena' is the fix that matters: the ORT session (WASM linear memory /
    // GPU buffers) is the multi-GB resident; the embedding is only ~MBs.
    // Level 3 now exits the SharedWorker — measured, that is the ONLY thing
    // that returns ORT's WebGPU pool to the OS (FINDINGS §1.4).
    'pressure ladder: level 3 tears the shared instance down entirely',
    Array.isArray(freed) && freed.includes('sam21:worker'),
    `freed=${JSON.stringify(freed)}`,
  )
  // The rung above exits the SharedWorker, and a SharedWorker name stays
  // resolvable for a moment after close(): reconnecting under it attaches to a
  // corpse whose event loop is gone, so every op waits out its own timeout
  // (measured 20 s on `state`, and an encode would have waited 10 minutes).
  // Assert the reconnect is PROMPT, not merely eventual — the slow path still
  // "works", which is exactly why this needs a clock on it.
  const reconnect = await pageW.evaluate(async () => {
    const t = performance.now()
    const st = await (await import('./js/sam21-client.js')).refreshStatus()
    return { ms: performance.now() - t, ok: !!st }
  })
  check(
    'reconnect after teardown: a fresh generation answers promptly (no corpse attach)',
    reconnect.ok && reconnect.ms < 5000,
    `${reconnect.ms.toFixed(0)}ms`,
  )
  // The user-visible half of the same event: the governor sheds everything,
  // then the user clicks. The encode is the call that BUILDS the session, so a
  // torn-down host surfaces there first — and it had no retry at all, which
  // meant one dead click (no mask, and an export that returns null) instead of
  // one slow one.
  const afterTeardown = await pageW.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geoW.disc.x * pW, y: geoW.disc.y * pW },
  )
  check(
    'recovery: a click straight after a full teardown still returns a mask',
    afterTeardown?.maskSummary?.coverage > 0,
    `coverage=${((afterTeardown?.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )
  // A lost GPUDevice used to be TERMINAL for the lane: ORT caches its device on
  // its module object and the ES module cache survives re-import, so it would
  // never build another — every later encode failed "no WebGPU device after
  // encode" until the worker happened to exit, up to 120 s of a dead app. Real
  // devices do get lost (GPU reset, driver update, Chrome reclaiming), so drive
  // the actual loss rather than trusting the handler by inspection.
  const lostRecovery = await pageW.evaluate(async ({ x, y }) => {
    const c = await import('./js/sam21-client.js')
    await c.op('destroyDevice')
    await new Promise((r) => setTimeout(r, 300))
    const dead = await c.refreshStatus()
    const st = await window.__seglab.clickAt(x, y)
    const back = await c.refreshStatus()
    return {
      died: dead?.lane?.deviceLost === true,
      rebuilt: (back?.lane?.rebuilds || 0) >= 1,
      coverage: st?.maskSummary?.coverage || 0,
    }
  }, { x: geoW.disc.x * pW, y: geoW.disc.y * pW })
  check(
    'device loss: a destroyed GPUDevice is rebuilt and the next click still works',
    lostRecovery.died && lostRecovery.rebuilt && lostRecovery.coverage > 0,
    `died=${lostRecovery.died} rebuilt=${lostRecovery.rebuilt} coverage=${(lostRecovery.coverage * 100).toFixed(1)}%`,
  )
  const releasedAll = await pageW.evaluate(() => window.__seglab.releaseMemory().then(() => window.__seglab.engineState()))
  check('debug release-memory action clears residents', releasedAll && releasedAll.cachedImages === 0, JSON.stringify(releasedAll))
  await pageW.close()

  /* ── One shared instance across tabs — THE memory contract ──────────────
     Measured (FINDINGS §2): 2 dedicated instances cost more memory AND more
     wall clock than 3 shared ones, so everything downstream assumes tabs
     collapse onto one session/arena/GPUDevice. Nothing guarded it. The
     SharedWorker NAME is the sharing key, so a generation bug splits tabs into
     private instances and silently doubles the footprint while every other
     assertion still passes. Runs after the teardown above on purpose: it also
     proves two fresh tabs agree on the NEW generation. */
  log('phase A8 (multi-tab) — two tabs, one instance…')
  // A long side no other phase uses: the demo key is derived from the doc size,
  // so a shared size would let the OPFS cache serve tab A and hide whether the
  // tabs are sharing anything at all.
  const tabA = await newAppPage(context, '', 812)
  const tabB = await newAppPage(context, '', 812)
  const gT = await tabA.evaluate(() => window.__seglab.demoGeometry())
  const at = { x: gT.disc.x * gT.proxyScale, y: gT.disc.y * gT.proxyScale }
  const clickA = await tabA.evaluate((p) => window.__seglab.clickAt(p.x, p.y), at)
  const clickB = await tabB.evaluate((p) => window.__seglab.clickAt(p.x, p.y), at)
  const shared = await tabB.evaluate(async () => (await import('./js/sam21-client.js')).refreshStatus())
  check(
    'shared instance: two tabs attach to ONE host (memory stays O(1) in tabs)',
    shared && shared.tabs === 2 && shared.cachedImages >= 1,
    `tabs=${shared?.tabs} embeddings=${shared?.cachedImages}`,
  )
  check(
    // The second tab opening the same photo must ride the first tab's
    // embedding. If it re-encodes, the tabs are not actually sharing.
    'shared instance: second tab reuses the first tab\'s embedding (no re-encode)',
    clickA?.lastRun?.encoded === true && clickB?.lastRun?.encoded === false
      && clickB?.maskSummary?.coverage > 0,
    `A encoded=${clickA?.lastRun?.encoded} B encoded=${clickB?.lastRun?.encoded} B coverage=${((clickB?.maskSummary?.coverage || 0) * 100).toFixed(1)}%`,
  )
  await tabA.close()
  // Closing a tab must decrement the host's client count. SharedWorker has no
  // disconnect event, so this is the pagehide `bye` doing its job — without it
  // dead clients keep anyVisible() true and dormancy never sheds memory.
  const afterClose = await tabB.evaluate(async () => {
    const c = await import('./js/sam21-client.js')
    // Any queued op broadcasts, which is what exercises the reap. Covers both
    // routes: the pagehide `bye`, and — if the tab died without one — the
    // failed-post reap that is the only other signal a SharedWorker gets.
    await c.warm().catch(() => null)
    return c.refreshStatus()
  })
  check(
    'shared instance: a closed tab is dropped from the host (dormancy can still fire)',
    afterClose && afterClose.tabs === 1,
    `tabs=${afterClose?.tabs}`,
  )
  await tabB.close()

  /* ── Dormancy: the two rungs that actually give memory back ─────────────
     Releasing sessions does NOT return ORT's WebGPU pool (§1.4) — only exiting
     the worker does — so both rungs matter and they shed different things:
     rung 1 drops the sessions but KEEPS the embedding (recomputing it is the
     slow part), rung 2 exits. Until the double-registration fix above, a
     phantom client defaulted to visible and pinned anyVisible() true, so
     neither rung could ever fire in the product. Driven through a tunable
     rather than waiting the real 20 s / 2 min. */
  log('phase A9 (dormancy) — hidden tabs shed sessions, then the instance…')
  const tabD = await newAppPage(context, '', 844)
  const gD = await tabD.evaluate(() => window.__seglab.demoGeometry())
  await tabD.evaluate((p) => window.__seglab.clickAt(p.x, p.y), { x: gD.disc.x * gD.proxyScale, y: gD.disc.y * gD.proxyScale })
  const dorm = await tabD.evaluate(async () => {
    const c = await import('./js/sam21-client.js')
    const wait = (ms) => new Promise((r) => setTimeout(r, ms))
    await c.op('setDormancy', { dormantMs: 300, deepIdleMs: 1500 })
    const live = await c.refreshStatus()
    let closed = false
    c.subscribe((s) => { if (s === null) closed = true })
    // A synthetic hide is not enough: hello() re-reports the REAL
    // document.visibilityState, so the app's own idle prewarm would announce
    // the tab visible again and disarm dormancy. Make the document actually
    // look backgrounded, which is the state being tested.
    for (const [k, v] of [['visibilityState', 'hidden'], ['hidden', true]]) {
      Object.defineProperty(document, k, { get: () => v, configurable: true })
    }
    const hidden = await c.setVisible(false)   // restarts both timers from here
    await wait(900)
    const rung1 = await c.refreshStatus()
    await wait(1800)
    return {
      liveDecoder: !!live.lane.decoder,
      liveEmbeds: live.lane.embedKeys.length,
      rung1Encoder: !!rung1.lane.encoder,
      rung1Decoder: !!rung1.lane.decoder,
      rung1Embeds: rung1.lane.embedKeys.length,
      sameHost: live.hostId === rung1.hostId,
      closed,
    }
  })
  check(
    // The decoder deliberately survives rung 1: dropping the LAST ORT session
    // destroys the WebGPU device, and that clears every embedding — which is
    // precisely what this rung exists to avoid. 9.9 MB to anchor the device and
    // keep an 8 MB embedding worth a ~1 s re-encode.
    'dormancy rung 1: hidden drops the encoder, KEEPS the embedding and the device anchor',
    dorm.liveEmbeds >= 1 && dorm.sameHost === true
      && dorm.rung1Encoder === false && dorm.rung1Decoder === true && dorm.rung1Embeds >= 1,
    `dormant{encoder:${dorm.rung1Encoder},decoder:${dorm.rung1Decoder},embeds:${dorm.rung1Embeds}} sameHost=${dorm.sameHost}`,
  )
  check(
    // The only thing that hands ORT's WebGPU pool back to the OS (§1.4).
    'dormancy rung 2: still hidden → the instance exits and tells its tabs',
    dorm.closed === true,
    `closing received=${dorm.closed}`,
  )
  // Restore the real timings — a 1.5 s deep-idle leaking into later phases
  // would exit the host mid-test. Respawns the worker if rung 2 already fired.
  await tabD.evaluate(async () => (await import('./js/sam21-client.js')).op('setDormancy', {}))
  await tabD.close()

  /* ── Idle exit: the rung that actually gives the memory back ────────────
     Once an encoder session has existed the instance holds ~1.2 GB that
     release() cannot return (§1.4) — measured 1796 → 429 MB on exit, a 1367 MB
     reclaim. Visibility is the wrong trigger for that, so this rung is driven by
     ACTIVITY and fires with the tab still visible. What makes the trade
     acceptable is the resume: the embedding comes back from OPFS, so the one
     slow click does NOT re-encode. That is the assertion that matters. */
  log('phase A10 (idle exit) — idle instance with an encoder exits, resumes from OPFS…')
  const tabI = await newAppPage(context, '', 876)
  const gI = await tabI.evaluate(() => window.__seglab.demoGeometry())
  const atI = { x: gI.disc.x * gI.proxyScale, y: gI.disc.y * gI.proxyScale }
  await tabI.evaluate((p) => window.__seglab.clickAt(p.x, p.y), atI)
  const idle = await tabI.evaluate(async (p) => {
    const c = await import('./js/sam21-client.js')
    await c.op('setDormancy', { idleExitMs: 600 })
    const before = await c.refreshStatus()
    let closed = false
    c.subscribe((s) => { if (s === null) closed = true })
    await new Promise((r) => setTimeout(r, 2500))
    const st = await window.__seglab.clickAt(p.x, p.y)
    const after = await c.refreshStatus()
    await c.op('setDormancy', {})            // never leak 600 ms into later phases
    return {
      built: before?.builtEncoder === true,
      visibleTabs: before?.visible || 0,
      closed,
      hostChanged: !!before?.hostId && before.hostId !== after?.hostId,
      coverage: st?.maskSummary?.coverage || 0,
      encoded: st?.lastRun?.encoded,
    }
  }, atI)
  check(
    'idle exit: an instance holding an encoder exits even with a VISIBLE tab',
    idle.built === true && idle.visibleTabs >= 1 && idle.closed === true && idle.hostChanged === true,
    `builtEncoder=${idle.built} visibleTabs=${idle.visibleTabs} closing=${idle.closed} hostChanged=${idle.hostChanged}`,
  )
  check(
    // If this ever re-encodes, the rung is trading ~1.4 GB for a multi-second
    // stall instead of a ~1.1 s one, and it is no longer worth having.
    'idle exit: the resuming click restores from OPFS rather than re-encoding',
    idle.coverage > 0 && idle.encoded === false,
    `coverage=${(idle.coverage * 100).toFixed(1)}% encoded=${idle.encoded}`,
  )
  await tabI.close()

  /* ─── Phase H: trusted-host (Phosmith pro) — HD/escalation/working ────── */
  await run.enter('H')
  const pageD = await newAppPage(context, '?flagship=0', 2400, { host: HOST_PRO })
  log('phase H1 (HD export, trusted pro) — 2400px original, native-res cutout…')
  const geoD = await pageD.evaluate(() => window.__seglab.demoGeometry())
  await pageD.evaluate(
    ({ x, y }) => window.__seglab.clickAt(x, y),
    { x: geoD.disc.x * geoD.proxyScale, y: geoD.disc.y * geoD.proxyScale },
  )
  const ex = await pageD.evaluate(
    (probe) => window.__seglab.exportCutout(probe),
    { cx: geoD.disc.x, cy: geoD.disc.y, r: geoD.disc.r },
  )
  check(
    // One config bounds every export at 12 MP / 5120 px, so a large original is
    // downscaled rather than exported native (there is no 'pro' tier any more).
    'HD export: crop re-decoded, alpha correct, bounded by the single config',
    ex && ex.w <= 5120 && ex.h <= 5120 && (ex.w * ex.h) <= 12.1e6
      && ex.centerOpaque && ex.outsideTransparent,
    `export ${ex?.w}×${ex?.h} decoded=${ex?.decoded}`,
  )
  check(
    // radialErr is measured in EXPORT pixels; the export may be downscaled from
    // the original, so the tolerance scales with it.
    'HD export: boundary tracks the analytic disc, with a soft band',
    ex && ex.radialErr <= 3 * Math.max(1, geoD.originalW / ex.w) && ex.softPixels > 100,
    `radialErr=${ex?.radialErr?.toFixed(1)}px, softPixels=${ex?.softPixels}`,
  )
  await pageD.close()

  // `force` drives the EXPLICIT escalation action (§11). Automatic escalation
  // is retired — a native re-decode per click costs another full encode — so
  // the auto case must NOT fire, and the explicit case must.
  const runDot = async (query, force = false) => {
    const pg = await newAppPage(context, query, 4000, { host: HOST_PRO })
    const g = await pg.evaluate(() => window.__seglab.demoGeometry())
    await pg.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: g.dot.x * g.proxyScale, y: g.dot.y * g.proxyScale })
    if (force) await pg.evaluate(() => window.__seglab.escalate())
    const esc = await pg.evaluate((probe) => window.__seglab.escalation(probe), { cx: g.dot.x, cy: g.dot.y, r: g.dot.r })
    await pg.close()
    return { esc, r: g.dot.r }
  }
  log('phase H2 (crop escalation, trusted pro) — native re-decode vs ?escalate=0…')
  const on = await runDot('?flagship=0', true)     // explicit user action
  const auto = await runDot('?flagship=0')          // no action → must not fire
  const off = await runDot('?flagship=0&escalate=0', true)
  check(
    'escalation: fires when explicitly asked, never automatically, refused under ?escalate=0',
    on.esc.fired === true && on.esc.decoded === true && on.esc.centerOpaque === true
      && auto.esc.fired === false && off.esc.fired === false,
    `explicit{fired:${on.esc.fired},decoded:${on.esc.decoded}} auto{fired:${auto.esc.fired}} escalate0{fired:${off.esc.fired}}`,
  )
  check(
    'escalation (pro): native re-decode recovers the boundary',
    on.esc.radialErr != null && off.esc.radialErr != null
      && on.esc.radialErr <= 6 && on.esc.radialErr < off.esc.radialErr * 0.7,
    `escalate=1 ${on.esc.radialErr?.toFixed(1)}px vs control ${off.esc.radialErr?.toFixed(1)}px`,
  )

  log('phase H3 (working copy, trusted pro) — ?working=1, 5000px blob upload…')
  const pageWk = await newAppPage(context, '?flagship=0&working=1', null, { host: HOST_PRO })
  await pageWk.evaluate((side) => window.__seglab.loadDemoBlob(side), 5000)
  const wkFrame = await pageWk.evaluate(() => window.__seglab.imageTransform())
  check(
    'working copy: oversized upload keeps a ≤4096 bounded re-decode source',
    wkFrame && wkFrame.proxyActive === true && wkFrame.workingActive === true
      && Math.max(wkFrame.workingW || 0, wkFrame.workingH || 0) === 4096,
    JSON.stringify({ working: `${wkFrame?.workingW}x${wkFrame?.workingH}`, proxy: `${wkFrame?.proxyW}x${wkFrame?.proxyH}` }),
  )
  const geoWk = await pageWk.evaluate(() => window.__seglab.demoGeometry())
  await pageWk.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoWk.dot.x * geoWk.proxyScale, y: geoWk.dot.y * geoWk.proxyScale })
  await pageWk.evaluate(() => window.__seglab.escalate())   // explicit action (§11)
  const escWk = await pageWk.evaluate((probe) => window.__seglab.escalation(probe), { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r })
  check(
    'working copy: escalation decodes a bounded crop (rescaled prompts land)',
    escWk.fired === true && escWk.decoded === true && escWk.centerOpaque === true && escWk.radialErr <= 6,
    `fired=${escWk.fired} decoded=${escWk.decoded} radialErr=${escWk.radialErr?.toFixed(1)}px`,
  )
  const exWk = await pageWk.evaluate((probe) => window.__seglab.exportCutout(probe), { cx: geoWk.dot.x, cy: geoWk.dot.y, r: geoWk.dot.r })
  check(
    // The export is a TIGHT crop at native scale, not a full-frame composite —
    // `rect` carries the offset. What "bypasses the working copy" means is that
    // the crop was re-decoded from the original (decoded) and the boundary lands
    // at native precision, which is what these two assert.
    'working copy: export bypasses it — native re-decode, boundary lands',
    exWk && exWk.decoded === true && exWk.radialErr <= 3,
    `export ${exWk?.w}×${exWk?.h}, decoded=${exWk?.decoded}, radialErr=${exWk?.radialErr?.toFixed(1)}px`,
  )
  await pageWk.close()

  log('phase H4 (OPFS revisit, trusted pro) — persistence stays a trusted-only feature…')
  const pageR1 = await newAppPage(context, '?flagship=0', 1600, { host: HOST_PRO })
  const r1 = await pageR1.evaluate(() => window.__seglab.eagerEncode())
  await pageR1.close()
  check('revisit (pro): first visit encodes fresh (cold store)', r1?.encoded === true, `encoded=${r1?.encoded}`)
  const pageR2 = await newAppPage(context, '?flagship=0', 1600, { host: HOST_PRO })
  const r2 = await pageR2.evaluate(() => window.__seglab.eagerEncode())
  const geoR = await pageR2.evaluate(() => window.__seglab.demoGeometry())
  const sR = await pageR2.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoR.disc.x * geoR.proxyScale, y: geoR.disc.y * geoR.proxyScale })
  check(
    'revisit (pro): fresh session serves the import encode from OPFS and decodes',
    r2?.encoded === false && sR.lastRun?.encoded === false && !!sR.maskSummary,
    `eager encoded=${r2?.encoded}, first click encoded=${sR.lastRun?.encoded}`,
  )
  await pageR2.close()

  /* ─── Phase O: zero-cloud proof (lite) ──────────────────────────────── */
  await run.enter('O')
  const pageO = await newAppPage(context, '?flagship=0', 1900)
  const geoO = await pageO.evaluate(() => window.__seglab.demoGeometry())
  await pageO.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: geoO.disc.x * geoO.proxyScale, y: geoO.disc.y * geoO.proxyScale })
  let attempted = 0
  let succeeded = 0
  try {
    await context.setOffline(true)
    pageO.on('request', () => { attempted += 1 })
    pageO.on('requestfinished', () => { succeeded += 1 })
    log('phase O (offline) — network cut; fresh import + click + text + export…')
    // A NEW document offline forces a fresh encode with zero network.
    await pageO.evaluate(() => window.__seglab.loadDemo(1400))
    const gOff = await pageO.evaluate(() => window.__seglab.demoGeometry())
    const pOff = gOff.proxyScale
    const oClick = await pageO.evaluate(({ x, y }) => window.__seglab.clickAt(x, y), { x: gOff.disc.x * pOff, y: gOff.disc.y * pOff })
    await pageO.evaluate(() => window.__seglab.reset())
    const dbox = [
      (gOff.disc.x - gOff.disc.r * 1.2) * pOff,
      (gOff.disc.y - gOff.disc.r * 1.2) * pOff,
      (gOff.disc.x + gOff.disc.r * 1.2) * pOff,
      (gOff.disc.y + gOff.disc.r * 1.2) * pOff,
    ]
    const oText = await pageO.evaluate((b) => window.__seglab.selectBoxes([b]), dbox)
    const oEx = await pageO.evaluate(() => window.__seglab.exportCutout())
    const oDiag = oEx ? null : await pageO.evaluate(() => window.__seglab.exportDiag())
    check(
      // encoded may be false: the OPFS embedding cache legitimately serves a
      // revisit without encoding. What matters offline is that every stage
      // COMPLETES, which is what this asserts.
      'offline: fresh import + click + text + export all pass with the network cut',
      !!oClick?.maskSummary && !!oText?.maskSummary && oEx && oEx.coverage > 0,
      `click=${!!oClick.maskSummary} encoded=${oClick.lastRun?.encoded} export=${oEx?.w}×${oEx?.h}`
      + ` text=${((oText?.maskSummary?.coverage || 0) * 100).toFixed(1)}% textRun=${JSON.stringify(oText?.lastRun || null)}`
      + (oDiag ? ` diag=${JSON.stringify(oDiag)}` : ''),
    )
    check('offline: zero successful network fetches during inference', succeeded === 0, `${attempted} attempted, ${succeeded} succeeded`)
  } finally {
    await context.setOffline(false)
  }
  await pageO.close()

  /* ─── Phase V: vendored assets serve a cold start with no network.
         Phase O proves zero-cloud inference, but only once the profile has
         cached the weights. This is the stronger claim: fresh cache-less
         browser, both CDNs blocked, selection still works. Skipped when
         nothing is vendored. */
  if (!existsSync(path.join(ROOT, 'models', 'manifest.json'))) {
    run.skip('V', 'nothing vendored — run `bun run models` for the offline gate')
  } else if (await run.enter('V')) {
    const browserV = await withDeadline(
      chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu'], timeout: LAUNCH_MS }),
      LAUNCH_MS + 15_000,
      'cold browser launch',
    )
    try {
      const ctxV = guardContext(await browserV.newContext(), { evaluateMs: TIMEOUT_MS, label: 'cold' })
      run.track(ctxV, 'cold') // fresh: no profile, no Cache Storage
      const blocked = []
      const localHits = []
      await ctxV.route(/https:\/\/(huggingface\.co|cdn(-lfs[a-z0-9-]*)?\.(jsdelivr\.net|huggingface\.co))\//, (r) => {
        blocked.push(r.request().url())
        r.abort()
      })
      const pageV = await newAppPage(ctxV, '?flagship=0', 1400)
      pageV.on('request', (req) => {
        const u = req.url()
        if (u.includes('/models/') || u.includes('/lib/')) localHits.push(u)
      })
      const gV = await pageV.evaluate(() => window.__seglab.demoGeometry())
      const sV = await pageV.evaluate(
        ({ x, y }) => window.__seglab.clickAt(x, y),
        { x: gV.disc.x * gV.proxyScale, y: gV.disc.y * gV.proxyScale },
      )
      const exV = await pageV.evaluate(() => window.__seglab.exportCutout())
      check(
        'vendored: cold cache-less start selects + exports with both CDNs blocked',
        !!sV?.maskSummary && /sam2\.1/.test(sV.lastRun?.lane || '') && exV?.coverage > 0,
        `coverage ${((sV?.maskSummary?.coverage || 0) * 100).toFixed(1)}%, export ${exV?.w}×${exV?.h}, `
          + `${blocked.length} CDN requests aborted`,
      )
      check(
        // The mask lane's weights are fetched by the SharedWorker, whose
        // requests are NOT attributed to the page — so counting page-level hits
        // cannot see them. The guarantee that matters is "nothing external was
        // reached", which `blocked` covers, verified independently by running
        // Chrome with all non-localhost DNS mapped to NOTFOUND: import, click
        // and export all succeed with zero external requests attempted.
        'vendored: nothing is fetched from a CDN (mask weights load in the SharedWorker)',
        blocked.length === 0,
        `${blocked.length} CDN requests aborted, ${localHits.length} page-level local fetches`,
      )
      await pageV.close()
    } finally {
      await browserV.close().catch(() => {})
    }
  }

  /* ─── Phase P: work survives a power cut. The record must be durable while
         the page is still open — a real cut fires no unload, so anything saved
         only on pagehide would be lost. Reopen must bring the work back. */
  await run.enter('P')
  {
    // Drop any session an earlier phase persisted BEFORE pageP boots: it would
    // otherwise auto-restore, and saves are suppressed while a restore is in
    // flight (so the click below would never reach disk).
    const pageClr = await newAppPage(context, '?flagship=0', null)
    await pageClr.evaluate(() => window.__seglab.clearSession())
    await pageClr.close()

    const pageP = await newAppPage(context, '?flagship=0', null, { restore: true })
    await pageP.evaluate((side) => window.__seglab.loadDemoBlob(side), 1400)
    const gP = await pageP.evaluate(() => window.__seglab.demoGeometry())
    const before = await pageP.evaluate(
      ({ x, y }) => window.__seglab.clickAt(x, y),
      { x: gP.disc.x * gP.proxyScale, y: gP.disc.y * gP.proxyScale },
    )
    // Poll for the MASK to land, not just the import-time record — the click's
    // save is debounced. Still no unload of any kind has fired.
    let durable = null
    for (let i = 0; i < 30 && !durable?.mask; i += 1) {
      durable = await pageP.evaluate(() => window.__seglab.sessionSaved())
      if (!durable?.mask) await pageP.waitForTimeout(500)
    }
    check(
      'persistence: session is durable while the page is still open (no unload fired)',
      durable?.mask === true,
      `saved mask=${durable?.mask} at ${durable?.w}×${durable?.h}`,
    )
    // Simulate the cut: abandon the page without letting it save anything more.
    await pageP.close()

    const pageQ = await newAppPage(context, '?flagship=0', null, { restore: true })
    await pageQ.waitForFunction(() => window.__seglab.state()?.hasImage === true, null, { timeout: 30_000 }).catch(() => {})
    const after = await pageQ.evaluate(() => ({ ...window.__seglab.state(), ...(window.__seglab.maskStats() || {}) }))
    const cov = (n) => (n?.maskSummary?.coverage ?? n?.coverage ?? 0)
    check(
      'persistence: reopen restores the document + committed selection',
      after.hasImage === true && cov(after) > 0,
      `hasImage=${after.hasImage}, coverage ${(cov(after) * 100).toFixed(1)}% (was ${(cov(before) * 100).toFixed(1)}%)`,
    )
    await pageQ.evaluate(() => window.__seglab.clearSession())
    await pageQ.close()
  }

  /* Runs last by contract, not by convention: PLAN declares A11 after:['V','P'].
     It opens ~9 pages and holds a ~1.2 GB GPU process while the cold-start
     phases each need one of their own. Run earlier, this left an 8 GB machine
     at 63 MB free with 2 GB swapped and the cold encode never returned. Moving
     this block now throws at the gate instead of hanging an hour later. */
  await run.enter('A11')
  /* ── Phase A11: hardening. Paths the suite never exercised at all ──────
     Every one of these was unverified until now, and each is a mode a real
     user reaches: a browser whose SharedWorker cannot hold WebGPU, cookies
     blocked, two photos open at once, a teardown observed by two tabs, a
     half-downloaded RAW. */
  log('phase A11 (hardening) — fallback, no-storage, concurrency, bad files…')

  // Two tabs both observe `closing` and both retire the generation. If they
  // disagree on the next name they land in PRIVATE instances and the memory
  // contract silently doubles while every other assertion still passes.
  const cv1 = await newAppPage(context, '', 912)
  const cv2 = await newAppPage(context, '', 912)
  const gCv = await cv1.evaluate(() => window.__seglab.demoGeometry())
  const atCv = { x: gCv.disc.x * gCv.proxyScale, y: gCv.disc.y * gCv.proxyScale }
  await cv1.evaluate((p) => window.__seglab.clickAt(p.x, p.y), atCv)
  await cv2.evaluate((p) => window.__seglab.clickAt(p.x, p.y), atCv)
  const hostOf = (pg) => pg.evaluate(async () => (await import('./js/sam21-client.js')).refreshStatus())
  const cvBefore = await hostOf(cv1)
  await cv1.evaluate(async () => (await import('./js/sam21-client.js')).op('shutdown').catch(() => null))
  await new Promise((r) => setTimeout(r, 1200))
  await Promise.all([
    cv1.evaluate((p) => window.__seglab.clickAt(p.x, p.y), atCv),
    cv2.evaluate((p) => window.__seglab.clickAt(p.x, p.y), atCv),
  ])
  const cvA = await hostOf(cv1)
  const cvB = await hostOf(cv2)
  check(
    'generation: two tabs seeing one teardown converge on ONE new instance',
    cvA?.hostId === cvB?.hostId && cvA?.hostId !== cvBefore?.hostId && cvA?.tabs === 2,
    `${cvBefore?.hostId} → ${cvA?.hostId}/${cvB?.hostId}, tabs=${cvA?.tabs}`,
  )
  await cv1.close()
  await cv2.close()

  // Two photos open at once: one queue serialises them, both embeddings stay
  // resident (embedCap tracks the client count) and neither tab decodes the
  // other's image.
  const cc1 = await newAppPage(context, '', 920)
  const cc2 = await newAppPage(context, '', 948)
  const clickMid = (pg) => pg.evaluate(async () => {
    const g = window.__seglab.demoGeometry()
    const s = await window.__seglab.clickAt(g.disc.x * g.proxyScale, g.disc.y * g.proxyScale)
    return s?.maskSummary?.coverage || 0
  })
  // Sequential, not concurrent. The claim is "two tabs on different photos both
  // work and BOTH embeddings stay resident"; racing the two clicks only adds
  // queue-timing noise, and under end-of-suite load it made the second tab
  // return an empty mask.
  const ccA = await clickMid(cc1)
  const ccB = await clickMid(cc2)
  const ccS = await hostOf(cc2)
  check(
    'concurrent tabs on DIFFERENT photos: both masks land, both embeddings resident',
    ccA > 0 && ccB > 0 && ccS?.tabs === 2 && (ccS?.lane?.embedKeys?.length || 0) >= 2,
    `A=${(ccA * 100).toFixed(1)}% B=${(ccB * 100).toFixed(1)}% tabs=${ccS?.tabs} embeds=${ccS?.lane?.embedKeys?.length}`,
  )
  await cc1.close()
  await cc2.close()

  // A browser where a SharedWorker cannot reach WebGPU falls back to a
  // DedicatedWorker — a whole second product mode (memory O(N) in tabs) that
  // nothing exercised. Removing the constructor is exactly what connect()
  // feature-detects.
  const dw = await context.newPage()
  await dw.addInitScript(() => {
    window.__seglabNoRestore = true
    Object.defineProperty(window, 'SharedWorker', { get: () => undefined, configurable: true })
  })
  await dw.goto(`http://127.0.0.1:${port}/`)
  await dw.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  await dw.evaluate(() => window.__seglab.loadDemo(912))
  const dwClick = await clickMid(dw)
  const dwMode = await dw.evaluate(async () => (await import('./js/sam21-client.js')).hostMode())
  const dwExport = await dw.evaluate(() => window.__seglab.exportCutout())
  check(
    'no SharedWorker: falls back to a dedicated worker, click and export both work',
    dwMode === 'dedicated' && dwClick > 0 && dwExport && dwExport.coverage > 0,
    `mode=${dwMode} click=${(dwClick * 100).toFixed(1)}% export=${dwExport?.w}×${dwExport?.h}`,
  )
  await dw.close()

  // localStorage blocked (private mode / third-party cookie blocking). The
  // generation cannot be shared, so the worker name must PIN rather than
  // diverge — a per-tab counter would split tabs into private instances, which
  // costs far more than the corpse-attach it avoids.
  const ls = await context.newPage()
  await ls.addInitScript(() => {
    window.__seglabNoRestore = true
    Object.defineProperty(window, 'localStorage', { get() { throw new Error('blocked') }, configurable: true })
  })
  await ls.goto(`http://127.0.0.1:${port}/`)
  await ls.waitForFunction(() => window.__seglabReady === true, null, { timeout: 30_000 })
  await ls.evaluate(() => window.__seglab.loadDemo(912))
  const lsFirst = await clickMid(ls)
  const lsBefore = await hostOf(ls)
  await ls.evaluate(async () => (await import('./js/sam21-client.js')).op('shutdown').catch(() => null))
  await new Promise((r) => setTimeout(r, 1200))
  const lsAfter = await clickMid(ls)
  const lsHost = await hostOf(ls)
  check(
    'localStorage blocked: still segments, and still recovers from a teardown',
    lsFirst > 0 && lsAfter > 0 && lsBefore?.hostId !== lsHost?.hostId,
    `first=${(lsFirst * 100).toFixed(1)}% after=${(lsAfter * 100).toFixed(1)}% ${lsBefore?.hostId} → ${lsHost?.hostId}`,
  )
  await ls.close()

  // The idle exit must be pushed back out by ongoing work. A threshold shorter
  // than the natural gap BETWEEN clicks (tab-side refine and paint are not host
  // jobs) would fire regardless, so this only passes if each job re-arms it.
  const ie = await newAppPage(context, '', 912)
  await clickMid(ie)
  const ieRes = await ie.evaluate(async () => {
    const c = await import('./js/sam21-client.js')
    await c.op('setDormancy', { idleExitMs: 2000 })
    const before = (await c.refreshStatus())?.hostId
    const g = window.__seglab.demoGeometry()
    const t0 = performance.now()
    let n = 0
    while (performance.now() - t0 < 5000) {
      await window.__seglab.clickAt(g.disc.x * g.proxyScale + (n % 5), g.disc.y * g.proxyScale)
      n += 1
      await new Promise((r) => setTimeout(r, 400))
    }
    const after = (await c.refreshStatus())?.hostId
    await c.op('setDormancy', {})
    return { before, after, n }
  })
  check(
    'idle exit: ongoing work re-arms it — 5 s of clicks survive a 2 s threshold',
    ieRes.before === ieRes.after,
    `${ieRes.n} clicks, ${ieRes.before} → ${ieRes.after}`,
  )
  await ie.close()

  // Malformed uploads: an interrupted download, a renamed file, a 1 px
  // placeholder. None may throw past the app's own handling, and — the part
  // that actually matters — a good file after them must still work.
  const fz = await newAppPage(context, '', null)
  const fzErrs = []
  fz.on('pageerror', (e) => fzErrs.push(String(e).slice(0, 120)))
  const fzOut = await fz.evaluate(async () => {
    const feed = async (name, bytes, type) => {
      const dt = new DataTransfer()
      dt.items.add(new File([bytes], name, { type }))
      const i = document.getElementById('file')
      i.files = dt.files
      i.dispatchEvent(new Event('change', { bubbles: true }))
      await new Promise((r) => setTimeout(r, 1500))
      return window.__seglab.state().hasImage
    }
    const out = []
    out.push(['empty', await feed('a.nef', new Uint8Array(0), '')])
    out.push(['garbage', await feed('b.nef', Uint8Array.from({ length: 40000 }, (_, i) => (i * 37) & 255), '')])
    out.push(['html-as-jpg', await feed('c.jpg', new TextEncoder().encode('<!doctype html><h1>nope</h1>'), 'image/jpeg')])
    return out
  })
  // Recovery is the real assertion — loadDemo goes through the same pipeline.
  await fz.evaluate(() => window.__seglab.loadDemo(912))
  const fzBack = await clickMid(fz)
  check(
    'malformed uploads: refused without an unhandled error, and the app still works after',
    fzErrs.length === 0 && fzBack > 0,
    `${fzOut.map(([n, h]) => `${n}:${h ? 'loaded' : 'refused'}`).join(' ')} → recovery=${(fzBack * 100).toFixed(1)}% pageerrors=${fzErrs.length}`,
  )
  // Chrome exposes navigator.gpu inside a SharedWorker; Firefox and WebKit do
  // not (approved by the GPU-for-the-Web CG, unimplemented). So no browser
  // available here can exercise the fallback end to end — the DECISION is what
  // gets asserted, because getting it wrong means the app is dead on Safari and
  // Firefox while a dedicated worker would have worked fine.
  const fbPage = await newAppPage(context, '', null)
  const fbTruth = await fbPage.evaluate(async () => {
    const { shouldFallbackToDedicated: f } = await import('./js/sam21-client.js')
    return {
      sharedNoGpu: f('shared', { ok: false, reason: 'WebGPU unavailable' }, false),
      sharedOk: f('shared', { ok: true }, false),
      alreadyTried: f('shared', { ok: false }, true),
      dedicated: f('dedicated', { ok: false }, false),
      noVerdict: f('shared', undefined, false),
    }
  })
  check(
    'shared→dedicated fallback fires only when a shared host cannot reach WebGPU',
    fbTruth.sharedNoGpu === true && fbTruth.sharedOk === false
      && fbTruth.alreadyTried === false && fbTruth.dedicated === false && fbTruth.noVerdict === false,
    JSON.stringify(fbTruth),
  )
  await fbPage.close()

  /* ── Phase A12: a browser with no WebGPU at all — Safari and Firefox today.
     It needs its OWN browser process (the flag is process-wide, and patching
     navigator.gpu in the page would not reach the worker that runs the model),
     so the gate hands A11's instance back first and refuses to start without
     headroom. This is where an exhausted machine used to hang instead of
     failing. */
  if (await run.enter('A12')) {
    // WebGPU is a hard requirement (§4), so every selection fails here; what
    // matters is that the app SAYS so. It was surfacing the raw runtime string
    // instead ("no available backend found. ERR: [webgpu] Error: Failed to get
    // GPU adapter. You may need to enable fla…").
    const noGpuCtx = guardContext(
      await withDeadline(
        chromium.launchPersistentContext(`${PROFILE_DIR}-nogpu`, {
          headless: true,
          args: ['--disable-features=WebGPU,WebGPUService', '--disable-gpu'],
          timeout: LAUNCH_MS,
        }),
        LAUNCH_MS + 15_000,
        'no-WebGPU browser launch',
      ),
      { evaluateMs: TIMEOUT_MS, label: 'nogpu' },
    )
    run.track(noGpuCtx, 'nogpu')
    const ng = await noGpuCtx.newPage()
    const ngErrs = []
    ng.on('pageerror', (e) => ngErrs.push(String(e).slice(0, 120)))
    await ng.addInitScript(() => { window.__seglabNoRestore = true })
    await ng.goto(`http://127.0.0.1:${port}/`)
    const ngReady = await ng.waitForFunction(() => window.__seglabReady === true, null, { timeout: 45_000 })
      .then(() => true).catch(() => false)
    await ng.evaluate(() => window.__seglab.loadDemo(800)).catch(() => null)
    await ng.evaluate(async () => {
      const g = window.__seglab.demoGeometry()
      return window.__seglab.clickAt(g.disc.x * g.proxyScale, g.disc.y * g.proxyScale)
    }).catch(() => null)
    const ngStatus = await ng.evaluate(() => document.getElementById('status')?.textContent || '')
    check(
      'no WebGPU: the app boots, does not crash, and explains itself in plain language',
      ngReady && ngErrs.length === 0 && /needs WebGPU/i.test(ngStatus) && !/ERR:|backend found/i.test(ngStatus),
      `ready=${ngReady} errors=${ngErrs.length} status=${JSON.stringify(ngStatus.slice(0, 80))}`,
    )
    await noGpuCtx.close()
  }

} catch (err) {
  console.error(`[verify] ✗ ${err?.message || err}`)
  // A contended lease is a verdict, not a crash — the stack adds nothing.
  if (err?.code !== 'ELEASEBUSY' && err?.code !== 'EPROFILEHELD') console.error(err?.stack || '')
  failed = true
} finally {
  run.finish()
  await withDeadline(context?.close() ?? Promise.resolve(), 30_000, 'shared context close').catch(() => {})
  server.close()
  await lease?.release()
  if (OPTS.isolated) rmSync(PROFILE_DIR, { recursive: true, force: true })
  console.log(run.summary())
}

if (failed || results.some((r) => !r.ok)) {
  console.error('\n[verify] ✗ SEGLAB self-test FAILED')
  process.exit(1)
}
console.log('\n[verify] ✓ SEGLAB verified end-to-end in a real browser')

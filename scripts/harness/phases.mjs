/**
 * Phase plan + gate for the verify harness.
 *
 * Two failure modes this exists to make impossible:
 *
 *  1. Order drift. The tuned order lived in a comment, so moving a phase broke
 *     the suite silently. The plan is data now: `after:` deps are checked
 *     before any work starts, and entering a phase out of plan order throws.
 *
 *  2. Hangs. A phase that exhausted the machine made the NEXT phase's cold
 *     encode block forever instead of failing. So: every phase is time-boxed,
 *     and a phase that launches its own browser declares the headroom it needs
 *     — the gate reclaims, waits (bounded), then runs it or skips it loudly.
 *     Ordering stops being load-bearing because the precondition is checked.
 *
 * Nothing here waits without a deadline.
 */

import { formatMem, memorySnapshot, waitForHeadroom } from './machine.mjs'
import { humanMs } from './lease.mjs'

const EMERGENCY_MS = 8_000

/** Bound a promise that has no timeout of its own (page.evaluate is the big one). */
export function withDeadline(promise, ms, label) {
  if (!Number.isFinite(ms) || ms <= 0) return promise
  let timer
  const alarm = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`deadline — ${label} did not settle in ${humanMs(ms)}`)), ms)
    timer.unref?.()
  })
  return Promise.race([promise, alarm]).finally(() => clearTimeout(timer))
}

/**
 * page.evaluate has no timeout in Playwright — an in-page hang hangs the
 * harness forever. Wrapping newPage is the one interception point that covers
 * every page the suite makes.
 */
export function guardContext(context, { evaluateMs, label = 'page' } = {}) {
  if (!context || context.__seglabGuarded) return context
  context.__seglabGuarded = true
  const rawNewPage = context.newPage.bind(context)
  context.newPage = async (...args) => {
    const page = await rawNewPage(...args)
    const rawEval = page.evaluate.bind(page)
    page.evaluate = (fn, ...rest) => withDeadline(
      rawEval(fn, ...rest),
      evaluateMs,
      `${label}.evaluate(${String(fn).replace(/\s+/g, ' ').slice(0, 90)}…)`,
    )
    return page
  }
  return context
}

/* ─── The plan ───────────────────────────────────────────────────────────── */

/**
 * Validates before any work runs. A plan that cannot hold its own constraints
 * should cost zero minutes to discover.
 */
export function definePlan(defs) {
  const seen = new Set()
  for (const d of defs) {
    if (!d.id) throw new Error('plan: a phase has no id')
    if (seen.has(d.id)) throw new Error(`plan: duplicate phase id ${d.id}`)
    seen.add(d.id)
    if (!Number.isFinite(d.budgetMs) || d.budgetMs <= 0) {
      throw new Error(`plan: phase ${d.id} has no budgetMs — an untimed phase is a hang waiting to happen`)
    }
    if (d.coldStart && !Number.isFinite(d.needsMB)) {
      throw new Error(`plan: phase ${d.id} launches its own browser but declares no needsMB floor`)
    }
  }
  const at = (id) => defs.findIndex((d) => d.id === id)
  for (const d of defs) {
    for (const dep of d.after || []) {
      if (at(dep) < 0) throw new Error(`plan: ${d.id} declares after:${dep}, which is not in the plan`)
      if (at(dep) > at(d.id)) {
        throw new Error(
          `plan: ${d.id} must run after ${dep}, but is scheduled before it.\n`
          + `  This ordering is load-bearing — ${dep} needs a machine ${d.id} has not exhausted yet.`,
        )
      }
    }
  }
  return defs
}

/* ─── The runner ─────────────────────────────────────────────────────────── */

export function createRunner({
  plan, results, log, lease = null, globalBudgetMs, onWatchdog = null, releaseSync = null,
  headroomWaitMs = 45_000, strict = false,
}) {
  let cursor = 0
  let active = null
  let finished = false
  let reclaim = null
  const ledger = []
  const notes = []
  const contexts = []

  const failures = (from) => results.slice(from).filter((r) => !r.ok).length

  const openPages = () => {
    const urls = []
    for (const { ctx, label } of contexts) {
      try {
        for (const p of ctx.pages()) urls.push(`${label}:${p.url().slice(-60)}`)
      } catch { /* context closed */ }
    }
    return urls
  }

  const diagnose = (headline) => {
    const snap = memorySnapshot(true)
    const pages = openPages()
    console.error(`\n[verify] ✗✗ ${headline}`)
    console.error(`[verify]    host: ${formatMem(snap)} (${snap.source})`)
    console.error(`[verify]    open pages: ${pages.length}${pages.length ? ` — ${pages.slice(0, 12).join(', ')}` : ''}`)
    if (active) {
      console.error(`[verify]    phase ${active.def.id} (${active.def.title}) ran ${humanMs(Date.now() - active.startedAt)}`
        + ` of ${humanMs(active.def.budgetMs)}, entered at ${formatMem(active.entryMem)}`)
    }
    if (snap.availableMB < 400 || snap.swapUsedMB > 1024) {
      console.error('[verify]    the machine is exhausted, not the code — this is the resource-debt failure,')
      console.error('[verify]    which the phase gate is supposed to catch before a cold start, not after.')
    }
  }

  const stopTimers = () => {
    if (active?.timer) clearTimeout(active.timer)
    if (globalTimer) clearTimeout(globalTimer)
  }

  const die = async (headline) => {
    stopTimers()
    diagnose(headline)
    // Cleanup must not become the new hang.
    if (onWatchdog) {
      await withDeadline(Promise.resolve().then(onWatchdog), EMERGENCY_MS, 'emergency cleanup').catch(() => {})
    }
    releaseSync?.()
    process.exit(1)
  }

  let globalTimer = setTimeout(
    () => { die(`WATCHDOG — the whole run exceeded ${humanMs(globalBudgetMs)}`) },
    globalBudgetMs,
  )
  globalTimer.unref?.()

  // An unsettled promise with an empty event loop exits 0 — a hang that reports
  // success. The only defence is to notice we never reached finish().
  process.on('beforeExit', () => {
    if (finished) return
    finished = true
    diagnose('the run ended without finishing — a promise never settled')
    releaseSync?.()
    process.exit(1)
  })

  const closeActive = (state = 'ran') => {
    if (!active) return
    if (active.timer) clearTimeout(active.timer)
    const exitMem = memorySnapshot(true)
    ledger.push({
      id: active.def.id,
      title: active.def.title,
      state,
      ms: Date.now() - active.startedAt,
      checks: results.length - active.checksAt,
      failed: failures(active.checksAt),
      entryMem: active.entryMem,
      exitMem,
      exhausts: !!active.def.exhausts,
    })
    active = null
  }

  const gate = async (def) => {
    // Only phases that stand up their own browser need a clean machine; the
    // rest inherit whatever the shared context already holds.
    if (!def.coldStart && !def.reclaimFirst && !Number.isFinite(def.needsMB)) return true
    if (def.reclaimFirst && reclaim) {
      // Unconditional: a cold start should not depend on how much the previous
      // phase happened to leave behind on this particular machine.
      log(`gate ${def.id}: handing the instance back before a cold start…`)
      try { await reclaim() } catch (err) { log(`gate ${def.id}: reclaim failed, continuing — ${err?.message || err}`) }
    }
    const before = memorySnapshot(true)
    if (!Number.isFinite(def.needsMB) || before.availableMB >= def.needsMB) return true
    log(`gate ${def.id}: ${formatMem(before)} — below the ${(def.needsMB / 1024).toFixed(1)} GB floor, reclaiming…`)
    const got = await waitForHeadroom({
      needMB: def.needsMB,
      timeoutMs: headroomWaitMs,
      reclaim: def.reclaimFirst ? null : reclaim, // already handed back above
      log: (m) => log(`gate ${def.id}: ${m}`),
    })
    if (got.ok) {
      log(`gate ${def.id}: recovered to ${formatMem(got.snap)} in ${humanMs(got.waitedMs)}`)
      return true
    }
    return got.snap
  }

  return {
    current: () => active?.def.id || (finished ? 'done' : 'startup'),

    track(ctx, label) {
      contexts.push({ ctx, label })
      return ctx
    },

    /** The gate's reclaim path — one implementation, used by every gate. */
    setReclaim(fn) { reclaim = fn },

    note(msg) {
      notes.push(msg)
      log(`note: ${msg}`)
    },

    /**
     * Open a phase. Returns false when the machine could not meet its floor —
     * a recorded skip, never a hang. Throws if the code and the plan disagree
     * about what runs next.
     */
    async enter(id) {
      const def = plan[cursor]
      if (!def || def.id !== id) {
        const known = plan.findIndex((p) => p.id === id)
        throw new Error(
          `phase order violation: entered ${id} but the plan expects `
          + `${def ? `${def.id} (${def.title})` : 'nothing — the plan is finished'}.\n`
          + (known >= 0 && known < cursor
            ? `  ${id} already ran or was skipped.\n`
            : '')
          + '  The plan in verify.mjs is the contract. If the reorder is intentional, move the\n'
          + '  entry there too — its after: deps encode which phases need an unexhausted machine.',
        )
      }
      closeActive()
      cursor += 1
      const verdict = await gate(def)
      if (verdict !== true) {
        log(`skip ${def.id} (${def.title}) — ${formatMem(verdict)}, needs ${(def.needsMB / 1024).toFixed(1)} GB`)
        ledger.push({
          id: def.id,
          title: def.title,
          state: 'skipped',
          reason: `machine below floor (${formatMem(verdict)})`,
          ms: 0,
          checks: 0,
          failed: 0,
        })
        if (strict) results.push({ label: `phase ${def.id} skipped — machine below its declared floor`, ok: false })
        return false
      }
      active = {
        def,
        startedAt: Date.now(),
        entryMem: memorySnapshot(true),
        checksAt: results.length,
        timer: null,
      }
      active.timer = setTimeout(
        () => { die(`WATCHDOG — phase ${def.id} (${def.title}) exceeded ${humanMs(def.budgetMs)}`) },
        def.budgetMs,
      )
      active.timer.unref?.()
      lease?.setPhase(`${def.id} (${def.title})`)
      log(`── phase ${def.id}: ${def.title} — ${formatMem(active.entryMem)}`)
      return true
    },

    /** Skip a phase the run legitimately cannot do (no fixtures, say). Idempotent. */
    skip(id, reason) {
      const at = plan.findIndex((p) => p.id === id)
      if (at < cursor) return false // already accounted for, including by a gate skip
      const def = plan[cursor]
      if (!def || def.id !== id) throw new Error(`phase order violation: skipped ${id}, plan expects ${def?.id || 'nothing'}`)
      closeActive()
      cursor += 1
      ledger.push({ id: def.id, title: def.title, state: 'skipped', reason, ms: 0, checks: 0, failed: 0 })
      log(`skip ${def.id} (${def.title}) — ${reason}`)
      return true
    },

    finish(reason = 'never entered') {
      closeActive()
      stopTimers()
      for (const def of plan.slice(cursor)) {
        ledger.push({ id: def.id, title: def.title, state: reason, ms: 0, checks: 0, failed: 0 })
      }
      cursor = plan.length
      finished = true
      return ledger
    },

    summary() {
      const pad = (s, n) => String(s).padEnd(n)
      const lines = ['', '[verify] ── phase ledger ──────────────────────────────────────────────']
      for (const row of ledger) {
        const mem = row.exitMem
          ? `${(row.entryMem.availableMB / 1024).toFixed(1)}→${(row.exitMem.availableMB / 1024).toFixed(1)} GB free`
          : (row.reason || '')
        const verdict = row.state === 'ran'
          ? `${row.checks - row.failed}/${row.checks} ok`
          : row.state
        lines.push(`[verify]   ${pad(row.id, 4)} ${pad(row.title.slice(0, 30), 31)} ${pad(humanMs(row.ms), 7)} ${pad(verdict, 12)} ${mem}`)
      }
      // The ordering rule stays honest only if it is measured. A phase that
      // leaves the machine flat but is not declared exhausting is drift.
      for (const row of ledger) {
        if (row.state !== 'ran' || row.exhausts || !row.exitMem) continue
        if (row.exitMem.availableMB < 1024) {
          lines.push(`[verify]   ! ${row.id} ended at ${formatMem(row.exitMem)} but is not marked exhausts:true — the plan is drifting`)
        }
      }
      for (const n of notes) lines.push(`[verify]   note: ${n}`)
      const skipped = ledger.filter((r) => r.state !== 'ran')
      if (skipped.length) {
        lines.push(`[verify]   ${skipped.length} phase(s) unproven: ${skipped.map((r) => r.id).join(', ')}`)
      }
      return lines.join('\n')
    },
  }
}

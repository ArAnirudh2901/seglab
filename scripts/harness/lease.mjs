/**
 * Exclusive run lease for the verify harness.
 *
 * `.cache/profile` is machine-global mutable state. A second run launching
 * Chromium on it blocks inside Chrome's singleton handshake — no output, no
 * timeout, indistinguishable from a slow suite. That produced two false
 * failures before anyone noticed the cause.
 *
 * This is the harness's ONLY lock: one lock cannot form a wait cycle, so the
 * deadlock precondition does not exist. Every wait on it is bounded, a holder
 * that dies is detected rather than waited on, and the lock is taken lazily —
 * the node-only phases never touch the profile, so they never contend.
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const HOST = os.hostname()
const HEARTBEAT_MS = 5_000
// ~9 missed beats. Long enough that a swapping machine is not called dead.
const STALE_MS = 45_000
const POLL_MS = 750

const sleep = (ms) => new Promise((r) => { setTimeout(r, ms) })

export const humanMs = (ms) => {
  if (ms < 10_000) return `${(Math.max(0, ms) / 1000).toFixed(1)}s`
  const s = Math.round(ms / 1000)
  return s < 90 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
}

const pidAlive = (pid) => {
  if (!Number.isInteger(pid) || pid <= 1) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (err) {
    return err?.code === 'EPERM' // alive, owned by someone else
  }
}

const readRecord = (lockPath) => {
  try { return JSON.parse(fs.readFileSync(lockPath, 'utf8')) } catch { return null }
}

const writeAtomic = (lockPath, rec) => {
  const tmp = `${lockPath}.${process.pid}.tmp`
  fs.writeFileSync(tmp, JSON.stringify(rec))
  fs.renameSync(tmp, lockPath) // atomic within one filesystem
}

/**
 * live | stale | dead | corrupt — deliberately never "unknown". An unknown
 * holder is one we would wait on forever, so a holder we cannot verify
 * (different host, no heartbeat) ages out through `stale`.
 */
const holderState = (rec) => {
  if (!rec || typeof rec.runId !== 'string') return 'corrupt'
  if (rec.host === HOST && !pidAlive(rec.pid)) return 'dead'
  if (Date.now() - (rec.heartbeatAt || 0) > STALE_MS) return 'stale'
  return 'live'
}

const contendedError = (rec, { lockPath, profileDir }) => {
  const age = humanMs(Date.now() - (rec.startedAt || Date.now()))
  const beat = humanMs(Date.now() - (rec.heartbeatAt || Date.now()))
  const err = new Error(
    `another verify run owns the Chromium profile\n`
    + `    run ${rec.runId}  pid ${rec.pid} on ${rec.host}  started ${age} ago\n`
    + `    phase ${rec.phase || '?'} — last heartbeat ${beat} ago\n`
    + `    profile ${profileDir}\n`
    + `    lease   ${lockPath}\n`
    + `  two suites on one profile is the silent-block failure. Pick one:\n`
    + `    bun verify.mjs --wait=10m   queue behind it (bounded, fails with a verdict)\n`
    + `    bun verify.mjs --fast       node-only phases; no browser, no lease\n`
    + `    bun verify.mjs --isolated   throwaway profile (cold cache, +1.5 GB RAM)\n`
    + `    bun verify.mjs --force      steal the lease (only if that run is gone)`,
  )
  err.code = 'ELEASEBUSY'
  err.holder = rec
  return err
}

/* ─── Chrome profile hygiene ─────────────────────────────────────────────── */

/** POSIX Chrome writes SingletonLock as a symlink to `host-pid`. */
export function profileSingleton(profileDir) {
  const lock = path.join(profileDir, 'SingletonLock')
  let target = null
  try {
    target = fs.readlinkSync(lock)
  } catch (err) {
    if (err?.code === 'EINVAL') target = '' // exists but not a symlink
    else return null
  }
  const m = /^(.*)-(\d+)$/.exec(target || '')
  return { lock, host: m?.[1] || null, pid: m ? Number(m[2]) : null, target }
}

/**
 * Clear a singleton left by a run that died. Returns 'clear' | 'cleared' |
 * 'held'. A held singleton is physical truth — --force cannot override it.
 */
export function clearStaleSingleton(profileDir, log = () => {}) {
  const sing = profileSingleton(profileDir)
  if (!sing) return 'clear'
  if (sing.pid && sing.host === HOST && pidAlive(sing.pid)) return 'held'
  for (const name of ['SingletonLock', 'SingletonSocket', 'SingletonCookie']) {
    try { fs.unlinkSync(path.join(profileDir, name)) } catch { /* already gone */ }
  }
  log(`cleared a stale Chrome singleton in ${path.basename(profileDir)} (pid ${sing.pid ?? '?'} is gone)`)
  return 'cleared'
}

/**
 * Chromium processes still pinned to our cache dir. We hold the lease, so by
 * definition no live run owns them — they are orphans from a killed suite and
 * they are exactly what makes the next run block.
 */
export function reapOrphans(cacheDir, log = () => {}) {
  let listing = ''
  try {
    listing = execFileSync('ps', ['-A', '-w', '-w', '-o', 'pid=,args='], {
      encoding: 'utf8', timeout: 5000, maxBuffer: 16 * 1024 * 1024,
    })
  } catch { return [] }
  const needle = `--user-data-dir=${cacheDir}`
  const victims = []
  for (const line of listing.split('\n')) {
    const m = /^\s*(\d+)\s+(.*)$/.exec(line)
    if (!m || !m[2].includes(needle)) continue
    const pid = Number(m[1])
    if (pid === process.pid) continue
    victims.push(pid)
  }
  for (const pid of victims) {
    try { process.kill(pid, 'SIGTERM') } catch { /* already gone */ }
  }
  if (victims.length) log(`reaped ${victims.length} orphaned chromium process(es) holding ${cacheDir}: ${victims.join(', ')}`)
  return victims
}

/* ─── The lease ──────────────────────────────────────────────────────────── */

export async function acquireLease({
  lockPath, profileDir, cacheDir, waitMs = 0, force = false, reap = true, log = () => {},
}) {
  fs.mkdirSync(path.dirname(lockPath), { recursive: true })
  const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  const startedAt = Date.now()
  let phase = 'startup'
  const record = () => ({
    v: 1, runId, pid: process.pid, host: HOST, startedAt, heartbeatAt: Date.now(), phase,
  })

  const deadline = Date.now() + Math.max(0, waitMs)
  let announced = false
  for (;;) {
    try {
      const fd = fs.openSync(lockPath, 'wx') // O_CREAT|O_EXCL — atomic
      fs.writeSync(fd, JSON.stringify(record()))
      fs.closeSync(fd)
      break
    } catch (err) {
      if (err?.code !== 'EEXIST') throw err
    }
    const held = readRecord(lockPath)
    const state = force ? 'forced' : holderState(held)
    if (state !== 'live') {
      // Steal, then re-read: two stealers both rename, the loser sees a runId
      // that is not its own and goes round again. No waiting, no cycle.
      writeAtomic(lockPath, record())
      await sleep(40 + Math.random() * 90)
      if (readRecord(lockPath)?.runId === runId) {
        if (held) log(`took over a ${state} lease from pid ${held.pid} (phase ${held.phase || '?'})`)
        break
      }
      continue
    }
    if (Date.now() >= deadline) throw contendedError(held, { lockPath, profileDir })
    if (!announced) {
      log(`lease held by pid ${held.pid} in phase ${held.phase || '?'} — waiting up to ${humanMs(waitMs)}`)
      announced = true
    }
    await sleep(POLL_MS + Math.random() * 250)
  }

  const beat = setInterval(() => {
    try { writeAtomic(lockPath, record()) } catch { /* fs hiccup; staleness covers it */ }
  }, HEARTBEAT_MS)
  beat.unref?.()

  let released = false
  const releaseSync = () => {
    if (released) return
    released = true
    clearInterval(beat)
    // Owner check: never delete a lease that reclaimed ours.
    if (readRecord(lockPath)?.runId === runId) {
      try { fs.unlinkSync(lockPath) } catch { /* already gone */ }
    }
  }

  process.on('exit', releaseSync)
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(sig, () => { releaseSync(); process.exit(130) })
  }

  // Physical state second: the lease says no run owns the profile, so anything
  // still holding it is wreckage from one that died.
  if (reap) reapOrphans(cacheDir, log)
  const singleton = clearStaleSingleton(profileDir, log)
  if (singleton === 'held') {
    const sing = profileSingleton(profileDir)
    releaseSync()
    const err = new Error(
      `a live Chrome (pid ${sing?.pid}) still holds ${profileDir}\n`
      + `  no verify run owns the lease, so this is a stray browser, not a suite.\n`
      + `  quit it, or run: bun verify.mjs --isolated`,
    )
    err.code = 'EPROFILEHELD'
    throw err
  }

  log(`lease ${runId} acquired — profile ${path.basename(profileDir)}`)
  return {
    runId,
    setPhase(next) {
      phase = next
      try { writeAtomic(lockPath, record()) } catch { /* heartbeat retries */ }
    },
    release: async () => releaseSync(),
    releaseSync,
  }
}

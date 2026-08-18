/**
 * Host memory truth for the verify harness.
 *
 * os.freemem() on darwin reports free pages, not available memory — it reads
 * ~100 MB on a machine with 4 GB reclaimable, which would make every admission
 * check refuse. vm_stat + swapusage is the number that actually predicted the
 * cold-encode hang (63 MB free, 2 GB swapped).
 */

import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import os from 'node:os'

const MB = 1024 * 1024
const CACHE_MS = 400

let cached = { at: 0, snap: null }

const sh = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8', timeout: 4000 })

const darwinSnapshot = () => {
  const vm = sh('vm_stat', [])
  const pageSize = Number(/page size of (\d+) bytes/.exec(vm)?.[1] || 4096)
  const pages = (label) => Number(new RegExp(`${label}:\\s+(\\d+)`).exec(vm)?.[1] || 0)
  // Inactive + speculative + purgeable are reclaimable on demand; the compressor
  // pool is not, so it stays out.
  const available = (pages('Pages free') + pages('Pages inactive')
    + pages('Pages speculative') + pages('Pages purgeable')) * pageSize
  let swapUsed = 0
  try {
    const swap = sh('sysctl', ['-n', 'vm.swapusage'])
    const m = /used\s*=\s*([\d.]+)([MGK])/i.exec(swap)
    if (m) {
      const unit = { K: 1 / 1024, M: 1, G: 1024 }[m[2].toUpperCase()] || 1
      swapUsed = Number(m[1]) * unit * MB
    }
  } catch { /* swap disabled */ }
  return { available, swapUsed, source: 'vm_stat' }
}

const linuxSnapshot = () => {
  const info = readFileSync('/proc/meminfo', 'utf8')
  const kb = (label) => Number(new RegExp(`^${label}:\\s+(\\d+) kB`, 'm').exec(info)?.[1] || 0) * 1024
  const swapTotal = kb('SwapTotal')
  return {
    available: kb('MemAvailable') || os.freemem(),
    swapUsed: Math.max(0, swapTotal - kb('SwapFree')),
    source: '/proc/meminfo',
  }
}

/** { availableMB, totalMB, swapUsedMB, source } — never throws; degrades to os.freemem(). */
export function memorySnapshot(fresh = false) {
  const now = Date.now()
  if (!fresh && cached.snap && now - cached.at < CACHE_MS) return cached.snap
  let raw
  try {
    if (process.platform === 'darwin') raw = darwinSnapshot()
    else if (process.platform === 'linux') raw = linuxSnapshot()
    else raw = { available: os.freemem(), swapUsed: 0, source: 'os.freemem' }
  } catch {
    raw = { available: os.freemem(), swapUsed: 0, source: 'os.freemem (probe failed)' }
  }
  const snap = {
    availableMB: Math.round(raw.available / MB),
    totalMB: Math.round(os.totalmem() / MB),
    swapUsedMB: Math.round(raw.swapUsed / MB),
    source: raw.source,
    at: now,
  }
  cached = { at: now, snap }
  return snap
}

const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`

export const formatMem = (snap = memorySnapshot()) =>
  `${gb(snap.availableMB)} available of ${gb(snap.totalMB)}`
  + (snap.swapUsedMB > 64 ? `, ${gb(snap.swapUsedMB)} swapped` : '')

/**
 * Wait for the host to come back above a floor. Always bounded: the caller gets
 * a verdict, never an indefinite block. `reclaim` runs once, up front.
 */
export async function waitForHeadroom({
  needMB, timeoutMs = 45_000, pollMs = 1000, reclaim = null, log = () => {},
}) {
  const startedAt = Date.now()
  const deadline = startedAt + timeoutMs
  if (reclaim) {
    try {
      await reclaim()
    } catch (err) {
      log(`reclaim hook failed (continuing): ${err?.message || err}`)
    }
  }
  let snap = memorySnapshot(true)
  let announced = false
  while (snap.availableMB < needMB) {
    if (Date.now() >= deadline) {
      return { ok: false, snap, waitedMs: Date.now() - startedAt }
    }
    if (!announced) {
      log(`waiting for headroom — need ${gb(needMB)}, have ${formatMem(snap)}`)
      announced = true
    }
    await new Promise((r) => { setTimeout(r, pollMs) })
    snap = memorySnapshot(true)
  }
  return { ok: true, snap, waitedMs: Date.now() - startedAt }
}

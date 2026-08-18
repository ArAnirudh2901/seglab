// Which Chrome the harness drives. Playwright's bundled Chrome-for-Testing is
// not the build anyone ships to, and its WebGPU/ORT revision differs from
// stable — so every launcher here (verify, the probes) prefers the installed
// Google Chrome and falls back to the bundled build only when there is none.
import { existsSync } from 'node:fs'

// Playwright resolves the channel itself; these paths only answer "is it
// installed", so a machine without Chrome (CI) still runs instead of failing
// at launch.
const CHROME_PATHS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/opt/google/chrome/chrome',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
]

/** '' means Playwright's bundled Chrome-for-Testing. `--cft` forces it. */
export const resolveChannel = (argv = process.argv.slice(2)) => {
  const pick = argv.find((a) => a.startsWith('--channel='))
  if (pick) return pick.slice('--channel='.length)
  if (argv.includes('--cft')) return ''
  return CHROME_PATHS.some((p) => existsSync(p)) ? 'chrome' : ''
}

/** Launch options plus the channel, when there is one. */
export const onChannel = (opts, channel) => (channel ? { ...opts, channel } : opts)

/**
 * A profile dir carries the build that wrote it, and stable Chrome can refuse
 * one Chrome-for-Testing wrote ("created by a newer version"), so channels
 * never share. Cost of the split is one cold model download per channel.
 */
export const profileDir = (base, channel) => (channel ? `${base}-${channel}` : base)

/** What to print so a transcript records which build proved the run. */
export const channelLabel = (channel) => (channel
  ? `Google Chrome (channel ${channel})`
  : "Playwright's bundled Chrome-for-Testing")

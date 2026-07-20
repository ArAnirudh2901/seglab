/**
 * dev-server — static file server for local development.
 *
 * Replaces `python3 -m http.server`: that server cannot set the COOP/COEP
 * response headers that make the page cross-origin isolated. Isolation unlocks
 * `crossOriginIsolated`, which the app needs for BOTH threaded ORT/WASM
 * inference (SharedArrayBuffer) and `performance.measureUserAgentSpecificMemory()`
 * (the real signal the memory governor watches). No dependencies — plain Node.
 *
 * `require-corp` is used (not `credentialless`) because it is the only variant
 * Safari honours, so isolation — and threaded WASM — works there too. The app
 * is fully self-contained and vendored-first (models/ + lib/ are same-origin),
 * so the strict policy blocks nothing it needs; the CDN model *fallback* gets
 * its cross-origin responses tagged with CORP by sw.js.
 */

import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const PORT = Number(process.env.PORT || 8788)
const HOST = process.env.HOST || '127.0.0.1'

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.webp': 'image/webp', '.gif': 'image/gif', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream', '.txt': 'text/plain', '.map': 'application/json',
}

// Cross-origin isolation. Must be present on the top-level document AND every
// same-origin subresource for `crossOriginIsolated` to become true.
const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
}

const server = createServer(async (req, res) => {
  const setHeaders = (extra = {}) => {
    for (const [k, v] of Object.entries(ISOLATION_HEADERS)) res.setHeader(k, v)
    // Dev: never cache, so edits always reflect on reload.
    res.setHeader('Cache-Control', 'no-store')
    for (const [k, v] of Object.entries(extra)) res.setHeader(k, v)
  }
  try {
    const url = new URL(req.url, `http://${HOST}`)
    const rel = url.pathname === '/' ? '/index.html' : decodeURIComponent(url.pathname)
    let file = path.join(ROOT, path.normalize(rel))
    if (!file.startsWith(ROOT)) { setHeaders(); res.writeHead(403).end('forbidden'); return }
    // Directory → its index.html.
    try { if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html') } catch { /* handled below */ }
    let body
    try { body = await readFile(file) } catch { setHeaders(); res.writeHead(404).end('not found'); return }
    setHeaders({ 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' })
    res.writeHead(200).end(body)
  } catch (e) {
    setHeaders(); res.writeHead(500).end(String(e?.message || e))
  }
})

server.listen(PORT, HOST, () => {
  console.log(`[seglab] dev server (cross-origin isolated) on http://${HOST}:${PORT}`)
})

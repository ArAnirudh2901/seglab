/**
 * sw — offline cache for static model/wasm/library assets
 * ----------------------------------------------------------
 * Cache-on-fetch ONLY — nothing is fetched ahead of use; a resource enters
 * the cache the first time the app legitimately loads it. Never caches
 * per-image pixels, blobs, embeddings, non-GET, or partial responses.
 * HuggingFace weight fetches are left to transformers.js's own Cache-API
 * store to avoid double-storing ~40 MB.
 */

const CACHE = 'seglab-static-v1'

const CACHEABLE = [
    /\/lib\//,
    /\/models\//,
    /\/public\/wasm\//,
    /^https:\/\/cdn\.jsdelivr\.net\/npm\/(@huggingface\/transformers|onnxruntime-web)@/,
]

self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', (event) => {
    event.waitUntil((async () => {
        for (const key of await caches.keys()) {
            if (key.startsWith('seglab-') && key !== CACHE) await caches.delete(key)
        }
        await self.clients.claim()
    })())
})

self.addEventListener('fetch', (event) => {
    const { request } = event
    if (request.method !== 'GET' || !request.url.startsWith('http')) return
    if (!CACHEABLE.some((re) => re.test(request.url))) return
    event.respondWith((async () => {
        const cache = await caches.open(CACHE)
        const hit = await cache.match(request)
        if (hit) return hit
        const response = await fetch(request)
        if (response.ok && response.status === 200) {
            cache.put(request, response.clone()).catch(() => { /* quota */ })
        }
        return response
    })())
})

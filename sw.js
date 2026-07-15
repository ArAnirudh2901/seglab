/**
 * sw.js — SEGLAB Service Worker: persistent model cache
 * -------------------------------------------------------
 * Intercepts fetches for ONNX model files and the transformers.js bundle from
 * HuggingFace and jsDelivr CDNs. After the first download, every subsequent
 * request is served from the 'seglab-models-v1' Cache Storage bucket —
 * eliminating the ~14 MB (draft) and ~300 MB (flagship) downloads on every
 * visit.
 *
 * Strategy: cache-first for model blobs; network-first (passthrough) for
 * everything else. A stale model is never a problem here because the app
 * pins exact transformers.js + model versions — the cache key is the full URL.
 *
 * Cache lifetime: perpetual (no TTL). The user can clear it from browser
 * DevTools > Application > Cache Storage > seglab-models-v1, or we can bump
 * CACHE_NAME to invalidate on a future app version bump.
 */

// v2: evicts the old Xenova OWLv2 fp16 blobs. The activate handler deletes any
// seglab-models-* bucket that isn't this one, so a stale heavy detector can't
// sit in Cache Storage and load instantly into a session we no longer build.
const CACHE_NAME = 'seglab-models-v2'

/**
 * URL prefixes that should be intercepted and cached.
 * - HuggingFace CDN (model ONNX files, config JSONs, tokenizers …)
 * - jsDelivr CDN (the pinned transformers.js ESM bundle)
 */
const MODEL_ORIGINS = [
    'https://huggingface.co/',
    'https://cdn-lfs.huggingface.co/',
    'https://cdn-lfs-us-1.huggingface.co/',
    'https://cdn.jsdelivr.net/npm/@huggingface/',
    // HuggingFace uses a content-delivery proxy for large blobs:
    'https://huggingface.co/resolve/',
]

const isModelFetch = (url) => MODEL_ORIGINS.some((prefix) => url.startsWith(prefix))

// ── Install: take control immediately, no page refresh needed ────────────────
self.addEventListener('install', (event) => {
    // Skip the waiting phase so the new SW activates right away.
    self.skipWaiting()
})

// ── Activate: claim all clients so existing tabs are covered at once ─────────
self.addEventListener('activate', (event) => {
    event.waitUntil(
        (async () => {
            // Remove any old cache versions if we ever bump CACHE_NAME.
            const keys = await caches.keys()
            await Promise.all(
                keys
                    .filter((k) => k.startsWith('seglab-models-') && k !== CACHE_NAME)
                    .map((k) => caches.delete(k)),
            )
            // Claim clients without waiting for a page reload.
            await self.clients.claim()
        })(),
    )
})

// ── Fetch: cache-first for model files, passthrough for everything else ───────
self.addEventListener('fetch', (event) => {
    const { request } = event
    // Only handle GET requests (POST/PUT/etc. are always passed through).
    if (request.method !== 'GET') return
    if (!isModelFetch(request.url)) return

    event.respondWith(
        (async () => {
            const cache = await caches.open(CACHE_NAME)

            // Cache hit → serve immediately (no network).
            const cached = await cache.match(request)
            if (cached) return cached

            // Cache miss → fetch from network, store, then return.
            try {
                const response = await fetch(request)
                // Only cache successful, non-partial responses.
                if (response.ok && response.status === 200) {
                    // clone() before consuming so we can both cache and return.
                    cache.put(request, response.clone())
                }
                return response
            } catch (err) {
                // Network failure and no cache entry — let the error propagate
                // so the app can show its own "model load failed" message.
                throw err
            }
        })(),
    )
})

/**
 * sw.js — SEGLAB Service Worker: persistent model cache
 * -------------------------------------------------------
 * Intercepts fetches for the vendored weights and the onnxruntime-web CDN
 * fallback. After the first download every subsequent request is served from
 * Cache Storage, so a deployed install pays the weight download once rather
 * than per visit. It never fetches resources itself: cache entries are created
 * only after the editor requests them, so prefetching cannot add a startup
 * memory peak.
 *
 * Strategy: cache-first for model blobs; network-first (passthrough) for
 * everything else. A stale model is never a problem here because every URL
 * carries an exact version — the cache key is the full URL.
 *
 * Cache lifetime: perpetual (no TTL). The user can clear it from browser
 * DevTools > Application > Cache Storage, or we can bump CACHE_NAME to
 * invalidate on a future app version bump.
 */

// v3 retires every cache bucket created while the automatic SAM3 upgrade was
// available. Activation deletes obsolete buckets before serving future loads.
// v4: the text lane replaced YOLOE prompt-free + YOLO-World, and
// yoloe-26l-text.fp16.onnx changed its txt_feats contract from a static 32
// classes to a dynamic axis. Same URL, incompatible tensor — a browser holding
// the v3 copy fails every search with "Got: 1 Expected: 32". Bumping the name
// drops the old generation; the `cache: 'reload'` below is what stops the
// refetch being answered by the immutable HTTP cache instead of the network.
// v5: SlimSAM and transformers.js are gone. Returning visitors hold ~90 MB of
// weights for a model the app can no longer load, and nothing would ever evict
// them — the entries are keyed by URLs nobody requests any more. The bump is
// the eviction.
const CACHE_NAME = 'seglab-models-v5'

/**
 * URL prefixes that should be intercepted and cached. Only the ORT runtime
 * remains: every model is same-origin and vendored (see isVendoredFetch).
 * The HuggingFace origins that used to be listed here served SlimSAM.
 */
const MODEL_ORIGINS = [
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@',
]

// Same-origin static wasm (cv-refine) is cached after first load — never
// prefetched, and per-image pixels/embeddings are never stored here.
const isWasmFetch = (url) => url.includes('/public/wasm/')

// Same-origin vendored assets (download-models.mjs → lib/ + models/).
// Cached on first fetch like the CDN copies, so a deployed install pays the
// weight download once rather than per visit.
const isVendoredFetch = (url) => {
    try {
        const { origin, pathname } = new URL(url)
        return origin === self.location.origin
            && (pathname.startsWith('/models/') || pathname.startsWith('/lib/'))
    } catch { return false }
}

const isModelFetch = (url) =>
    MODEL_ORIGINS.some((prefix) => url.startsWith(prefix)) || isWasmFetch(url) || isVendoredFetch(url)

/**
 * Version POINTERS must never be answered from this cache. Weights are keyed by
 * a URL carrying the model's sha, so serving them cache-first is always correct;
 * the pointer is what supplies that sha, and caching it pins the app to whatever
 * model was current when the entry was written.
 *
 * Measured: a re-export wrote sam2.1-tiny to disk (model.json scale=tiny, 67 MB
 * encoder) and the app went on loading the 81 MB small one — the lane fetches
 * model.json with `cache: 'no-cache'`, but that is an HTTP-cache directive and
 * does not reach past a Service Worker. Every model swap looked like a no-op.
 */
const isVersionPointer = (url) => {
    try { return /\/(model|manifest)\.json$/.test(new URL(url).pathname) } catch { return false }
}

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

// The page is cross-origin isolated (COEP: require-corp), which blocks any
// cross-origin subresource that lacks a CORP header. The CDN model *fallback*
// (HuggingFace/jsDelivr) does not always send one, so re-tag those responses
// here — the vendored same-origin path already carries CORP from the server.
const isCrossOrigin = (url) => {
    try { return new URL(url).origin !== self.location.origin } catch { return false }
}
const withCorp = async (response, url) => {
    if (!isCrossOrigin(url) || !response || !response.ok) return response
    const headers = new Headers(response.headers)
    headers.set('Cross-Origin-Resource-Policy', 'cross-origin')
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp')
    // Rebuild the response with the augmented headers (body is single-use, so
    // callers pass a response they no longer need to read themselves).
    const body = await response.blob()
    return new Response(body, { status: response.status, statusText: response.statusText, headers })
}

// ── Fetch: cache-first for model files, passthrough for everything else ───────
self.addEventListener('fetch', (event) => {
    const { request } = event
    // Only handle GET requests (POST/PUT/etc. are always passed through).
    if (request.method !== 'GET') return
    if (!isModelFetch(request.url)) return

    event.respondWith(
        (async () => {
            const cache = await caches.open(CACHE_NAME)

            // Pointers: network first, cache only as an offline fallback. They
            // are ~800 bytes, so the request costs nothing and it is the only
            // way a re-exported model ever becomes visible to the app.
            if (isVersionPointer(request.url)) {
                try {
                    const fresh = await fetch(request, { cache: 'reload' })
                    if (fresh.ok && fresh.status === 200) cache.put(request, fresh.clone())
                    return withCorp(fresh, request.url)
                } catch {
                    const stale = await cache.match(request)
                    if (stale) return withCorp(stale, request.url)
                    throw new Error('offline and no cached model pointer')
                }
            }

            // Cache hit → serve immediately (no network).
            const cached = await cache.match(request)
            if (cached) return withCorp(cached, request.url)

            // Cache miss → fetch from network, store, then return.
            try {
                // `reload` bypasses the HTTP cache. /models/ is served
                // `immutable, max-age=31536000`, so after a CACHE_NAME bump a
                // plain fetch would be answered from disk with the very copy
                // the bump exists to discard.
                const response = await fetch(request, { cache: 'reload' })
                // Only cache successful, non-partial responses.
                if (response.ok && response.status === 200) {
                    // clone() before consuming so we can both cache and return.
                    cache.put(request, response.clone())
                }
                return withCorp(response, request.url)
            } catch (err) {
                // Network failure and no cache entry — let the error propagate
                // so the app can show its own "model load failed" message.
                throw err
            }
        })(),
    )
})

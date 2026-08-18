/**
 * engine — the one engine sniff, shared by the main thread and the workers.
 * Three copies of it had drifted; a wrong copy is silent, because the branch
 * simply never fires on the engine it exists for.
 *
 * `vendor` is 'Apple Computer, Inc.' on every WebKit browser (iOS Chrome
 * included) and '' on Gecko, which has neither the Metal compile cost nor the
 * inference-count crash. Both properties are on NavigatorID, so this is
 * worker-safe.
 */
export const IS_WEBKIT = typeof navigator !== 'undefined'
    && !navigator.userAgentData
    && /apple/i.test(navigator.vendor || '')

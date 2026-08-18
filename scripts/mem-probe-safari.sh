#!/bin/bash
# Safari/WebKit footprint, per process. The Chrome probe cannot see WebKit, and
# Safari has no automation path that does not require enabling Remote Automation
# — so this is driven by hand: click through the app, run this at each stage.
#
# What matters here is ONE number, not the tree total: Safari runs a WebContent
# process per site, so the seglab tab is a single process, and Safari's ~1 GB
# per-tab ceiling applies to it alone. The tree total is context, not the gate.
#
# WebContent processes are XPC-launched (ppid 1), so they cannot be attributed to
# Safari by parentage. Quit other WebKit apps (Mail, Messages, any Electron-free
# WKWebView app) before trusting the total.
#
# Usage: scripts/mem-probe-safari.sh [label]
#        scripts/mem-probe-safari.sh --watch [label]   # sample until Ctrl-C, report peak
set -uo pipefail

WATCH=
[ "${1:-}" = "--watch" ] && { WATCH=1; shift; }
LABEL="${1:-}"

mb_of() {
    footprint -p "$1" 2>/dev/null | awk '
        /Footprint:/ {
            for (i = 1; i < NF; i++) if ($i == "Footprint:") {
                v = $(i+1); u = $(i+2)
                if (u ~ /^GB/) v *= 1024; else if (u ~ /^KB/) v /= 1024
                printf "%d", v; exit
            }
        }'
}

# Safari.app itself plus the three WebKit service roles. Nothing else in the
# Safari bundle (sync agents, safe-browsing, extensions) holds page memory.
webkit_pids() {
    pgrep -f 'Safari.app/Contents/MacOS/Safari|com.apple.WebKit.WebContent|com.apple.WebKit.GPU|com.apple.WebKit.Networking'
}

snapshot() {
    local total=0 rows=()
    for p in $(webkit_pids); do
        local mb; mb=$(mb_of "$p")
        [ -z "$mb" ] && continue
        local name; name=$(ps -p "$p" -o comm= 2>/dev/null)
        case "$name" in
            *WebContent*) name=WebContent ;;
            *WebKit.GPU*) name=WebKit-GPU ;;
            *Networking*) name=WebKit-Net ;;
            *Safari*)     name=Safari ;;
        esac
        rows+=("$(printf '  · pid %-7s %-12s %5d MB' "$p" "$name" "$mb")")
        total=$((total + mb))
    done
    [ ${#rows[@]} -eq 0 ] && { echo "no Safari/WebKit processes"; return 1; }
    printf '%s\n' "${rows[@]}" | sort -k5 -rn
    printf '  ─────────────────────────\n  %-20s %5d MB   %s\n' "ALL WEBKIT" "$total" "$LABEL"
    # The gate: the largest WebContent, which is the tab under test.
    local biggest
    biggest=$(printf '%s\n' "${rows[@]}" | grep WebContent | sort -k5 -rn | head -1 | awk '{print $5}')
    [ -n "$biggest" ] && printf '  %-20s %5d MB   (Safari kills a tab near ~1000)\n' "BIGGEST TAB" "$biggest"
    echo "$total" > /tmp/seglab-safari-last
}

if [ -z "$WATCH" ]; then
    snapshot
    exit $?
fi

echo "sampling — drive the app in Safari, Ctrl-C when done"
peak=0
trap 'printf "\n  PEAK (all WebKit) %5d MB   %s\n" "$peak" "$LABEL"; exit 0' INT
while :; do
    snapshot >/dev/null 2>&1 || true
    t=$(cat /tmp/seglab-safari-last 2>/dev/null || echo 0)
    [ "$t" -gt "$peak" ] && peak=$t
done

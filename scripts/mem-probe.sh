#!/bin/bash
# Chrome process-tree footprint, bucketed by --type=.
#
# measureUserAgentSpecificMemory is blind to GPU/unified memory, so the in-page
# number cannot answer "does the whole app fit". This walks the real process tree
# instead and sums phys_footprint, which is what the 2.0-2.2 GB ceiling is
# actually about. Pass the --user-data-dir the browser was launched with.
#
# Usage: scripts/mem-probe.sh [-v] <user-data-dir-fragment> [label]
#   -v  also list every process individually (which renderer holds the memory)
set -uo pipefail
VERBOSE=
[ "${1:-}" = "-v" ] && { VERBOSE=1; shift; }
FRAG="${1:?usage: mem-probe.sh [-v] <user-data-dir-fragment> [label]}"
LABEL="${2:-}"

# Match the profile path anywhere in the command line, and only Chrome helpers —
# the user's own everyday Chrome must never be folded into this total. The
# second filter matters more than it looks: any shell whose own command line
# mentions the pattern (this script's caller, a grep, an editor) matches the
# pgrep and would otherwise be counted as a browser process.
pids=$(pgrep -f "user-data-dir=[^ ]*$FRAG" | while read -r p; do
    ps -p "$p" -o command= 2>/dev/null | grep -q 'Google Chrome' && echo "$p"
done)
[ -z "$pids" ] && { echo "no Chrome with a user-data-dir matching '$FRAG'"; exit 1; }
total=0
rows=()
detail=()
for p in $pids; do
    # `footprint` prints e.g. "Footprint: 114 MB (16384 bytes per page)".
    mb=$(footprint -p "$p" 2>/dev/null | awk '
        /Footprint:/ {
            for (i = 1; i < NF; i++) if ($i == "Footprint:") {
                v = $(i+1); u = $(i+2)
                if (u ~ /^GB/) v *= 1024; else if (u ~ /^KB/) v /= 1024
                printf "%d", v; exit
            }
        }')
    [ -z "$mb" ] && continue
    kind=$(ps -p "$p" -o command= 2>/dev/null | grep -o -- '--type=[a-z-]*' | head -1)
    kind=${kind:---browser}
    rows+=("$mb ${kind#--type=}")
    [ -n "$VERBOSE" ] && detail+=("$(printf '  · pid %-7s %-12s %5d MB' "$p" "${kind#--type=}" "$mb")")
    total=$((total + mb))
done
[ ${#rows[@]} -eq 0 ] && { echo "footprint returned nothing for ${pids//$'\n'/ }"; exit 1; }

[ -n "$VERBOSE" ] && printf '%s\n' "${detail[@]}" | sort -k5 -rn
printf '%s\n' "${rows[@]}" | awk '{a[$2]+=$1} END {for (k in a) printf "  %-14s %6d MB\n", k, a[k]}' | sort -k2 -rn
echo "  ─────────────────────────"
printf "  %-14s %6d MB   %s\n" "ALL CHROME" "$total" "$LABEL"

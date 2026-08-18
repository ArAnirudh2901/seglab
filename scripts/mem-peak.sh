#!/bin/bash
# Continuous peak sampler around scripts/mem-probe.sh.
#
# A point probe cannot answer "did the app ever exceed the ceiling" — a cold
# encode holds its peak for a second or two and a probe taken after it reads
# steady state. This samples the whole process tree and reports the MAXIMUM.
#
# Usage: mem-peak.sh sample <user-data-dir-fragment>   # run this in background
#        mem-peak.sh stop [label]                      # kill it and report
#
# A previous run's Chrome still tearing down matches the SAME --user-data-dir and
# is summed into this run's baseline — up to ~700 MB of phantom memory, which is
# what made the text lane read 2.9-3.6 GB when it actually peaks at ~2.0. Before
# relaunching on a profile, spin until
# `pgrep -fc "user-data-dir=[^ ]*<frag>"` returns 0.
#
# `sample` runs in the FOREGROUND on purpose. A backgrounded loop dies with the
# shell that started it — a 30 s scenario then yields three samples, which reads
# as a low peak rather than as a broken sampler. macOS has no `setsid` to detach
# with (nor `timeout`), so the caller owns the lifetime.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/tmp/seglab-mem-peak.log

case "${1:?usage: mem-peak.sh sample|stop}" in
sample)
    FRAG="${2:?usage: mem-peak.sh sample <user-data-dir-fragment>}"
    : > "$OUT"
    while :; do
        "$HERE/mem-probe.sh" "$FRAG" "" 2>/dev/null | awk '/ALL CHROME/ {print $3}' >> "$OUT"
    done
    ;;
stop)
    LABEL="${2:-}"
    pkill -f 'mem-peak.sh sample' 2>/dev/null
    sleep 0.3
    awk -v label="$LABEL" '
        {n++; s+=$1; if ($1 > max) max = $1; if (min == 0 || $1 < min) min = $1}
        END {
            if (n == 0) { print "  no samples"; exit 1 }
            printf "  samples %-4d  min %5d MB  mean %5d MB  PEAK %5d MB   %s\n", n, min, s/n, max, label
        }' "$OUT"
    ;;
*) echo "usage: mem-peak.sh sample <frag> | stop [label]"; exit 1 ;;
esac

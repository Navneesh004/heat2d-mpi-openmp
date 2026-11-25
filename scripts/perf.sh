#!/usr/bin/env bash
# Hardware-counter profile (native Linux only; perf is unreliable in WSL2/containers).
#   sudo sysctl kernel.perf_event_paranoid=1   # once, lets a normal user read counters
#
# Writes results/perf_stat_<variant>.txt and results/perf_report.txt.
source "$(dirname "$0")/common.sh"

NX="${NX:-4096}"
STEPS="${STEPS:-500}"
BLOCK="${BLOCK:-512}"
THREADS="${THREADS:-$CORES}"
EVENTS="cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"

command -v perf >/dev/null || { echo "perf not installed (linux-tools-generic)" >&2; exit 1; }

for variant in naive blocked; do
    args=("$NX" "$STEPS")
    [[ $variant == blocked ]] && args+=(--block "$BLOCK")
    for t in 1 "$THREADS"; do
        out="$RESULTS/perf_stat_${variant}_t${t}.txt"
        echo "perf stat: $variant, $t thread(s) -> $out"
        OMP_NUM_THREADS=$t perf stat -e "$EVENTS" -o "$out" "$HEAT_BIN" "${args[@]}"
    done
done

echo "perf record (call graph) ..."
OMP_NUM_THREADS="$THREADS" perf record -g -o "$RESULTS/perf.data" "$HEAT_BIN" "$NX" "$STEPS" >/dev/null
perf report -i "$RESULTS/perf.data" --stdio --no-children --percent-limit 1 > "$RESULTS/perf_report.txt"
echo "top of report:"; grep -v '^#' "$RESULTS/perf_report.txt" | grep -v '^$' | head -15

python3 "$(dirname "$0")/perf_summary.py" "$RESULTS"

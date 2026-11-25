#!/usr/bin/env bash
# Mode comparison at the full core count: OpenMP-only vs MPI-only vs every
# hybrid ranks x threads split.
source "$(dirname "$0")/common.sh"

NX="${NX:-4096}"
STEPS="${STEPS:-500}"
P="${P:-$CORES}"
CSV="$RESULTS/modes.csv"
new_csv "$CSV"

cfgs=("1 $P $NX $STEPS" "$P 1 $NX $STEPS")
for (( r = 2; r < P; r++ )); do
    (( P % r == 0 )) && cfgs+=("$r $(( P / r )) $NX $STEPS")
done

echo "modes at $P cores: ${NX}x${NX}, $STEPS steps, ${#cfgs[@]} configs -> $CSV"
bench "$CSV" "${cfgs[@]}"

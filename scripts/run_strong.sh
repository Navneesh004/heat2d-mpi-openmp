#!/usr/bin/env bash
# Strong scaling: fixed global grid, increasing core count.
# Usage: scripts/run_strong.sh            (NX=4096 STEPS=500 by default)
source "$(dirname "$0")/common.sh"

NX="${NX:-4096}"
STEPS="${STEPS:-500}"
CSV="$RESULTS/strong.csv"
new_csv "$CSV"

cfgs=()   # "ranks threads args..."
for p in $(core_counts); do
    if (( p == 1 )); then cfgs+=("1 1 $NX $STEPS"); continue; fi
    cfgs+=("1 $p $NX $STEPS" "$p 1 $NX $STEPS")                  # openmp, mpi
    (( p >= 4 && p % 2 == 0 )) && cfgs+=("2 $((p / 2)) $NX $STEPS") # hybrid 2 x p/2
done

echo "strong scaling: ${NX}x${NX}, $STEPS steps, cores: $(core_counts), ${#cfgs[@]} configs x $REPS reps -> $CSV"
bench "$CSV" "${cfgs[@]}"

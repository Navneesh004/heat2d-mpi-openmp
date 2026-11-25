#!/usr/bin/env bash
# Before/after for the optimisations: cache blocking and comm/compute overlap.
# Runs at the full core count in MPI-only mode (where overlap matters most) and
# in OpenMP-only mode (where blocking is the only change).
source "$(dirname "$0")/common.sh"

NX="${NX:-4096}"
STEPS="${STEPS:-500}"
BLOCK="${BLOCK:-512}"
P="${P:-$CORES}"
CSV="$RESULTS/opt.csv"
new_csv "$CSV"

cfgs=(
    "$P 1 $NX $STEPS"
    "$P 1 $NX $STEPS --block $BLOCK"
    "$P 1 $NX $STEPS --overlap"
    "$P 1 $NX $STEPS --block $BLOCK --overlap"
    "1 $P $NX $STEPS"
    "1 $P $NX $STEPS --block $BLOCK"
)

echo "optimisations at $P cores: ${NX}x${NX}, $STEPS steps, block=$BLOCK -> $CSV"
bench "$CSV" "${cfgs[@]}"

#!/usr/bin/env bash
# Weak scaling: fixed work per core (NX columns x ROWS_PER_CORE rows), so the
# global grid grows with the core count. Ideal: flat time.
source "$(dirname "$0")/common.sh"

NX="${NX:-4096}"
ROWS_PER_CORE="${ROWS_PER_CORE:-1024}"
STEPS="${STEPS:-500}"
CSV="$RESULTS/weak.csv"
new_csv "$CSV"

cfgs=()
for p in $(core_counts); do
    rows=$(( p * ROWS_PER_CORE ))
    if (( p == 1 )); then cfgs+=("1 1 $NX $STEPS --rows $rows"); continue; fi
    cfgs+=("1 $p $NX $STEPS --rows $rows" "$p 1 $NX $STEPS --rows $rows")
done

echo "weak scaling: ${NX} cols x ${ROWS_PER_CORE} rows/core, $STEPS steps, cores: $(core_counts) -> $CSV"
bench "$CSV" "${cfgs[@]}"

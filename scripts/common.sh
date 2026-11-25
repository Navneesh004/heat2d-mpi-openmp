#!/usr/bin/env bash
# Shared helpers for the benchmark scripts. Source this file; don't run it.
#
# Environment overrides:
#   HEAT_BIN      path to the binary            (default: build/heat)
#   CORES         max cores to use              (default: physical core count)
#   REPS          repetitions per configuration (default: 5)
#   MPIRUN        launcher                      (default: mpirun)
#   MPIRUN_EXTRA  extra launcher flags, e.g. "--oversubscribe" in containers

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAT_BIN="${HEAT_BIN:-$ROOT/build/heat}"
RESULTS="${RESULTS:-$ROOT/results}"
REPS="${REPS:-5}"
MPIRUN="${MPIRUN:-mpirun}"
MPIRUN_EXTRA="${MPIRUN_EXTRA:-}"
mkdir -p "$RESULTS"

[[ -x "$HEAT_BIN" ]] || { echo "binary not found: $HEAT_BIN (build first, see README)" >&2; exit 1; }

# Physical cores = unique (core, socket) pairs. Hyperthreads are not real cores
# and would bend the scaling curve.
physical_cores() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l
    else
        nproc
    fi
}
CORES="${CORES:-$(physical_cores)}"

# 1 2 4 8 ... up to CORES, plus CORES itself if it is not a power of two.
core_counts() {
    local p=1 out=""
    while (( p <= CORES )); do out+="$p "; p=$(( p * 2 )); done
    (( CORES & (CORES - 1) )) && out+="$CORES"
    echo $out
}

# Pinning. In a VM/container whose vCPUs show up as hyperthreads, use
#   OMP_PLACES=threads MPI_BIND=hwthread MPIRUN_EXTRA=--use-hwthread-cpus CORES=<vCPUs>
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
MPI_BIND="${MPI_BIND:-core}"

# run_cfg RANKS THREADS heat-args...   -> prints one CSV line
# A single rank is launched directly (no mpirun): `mpirun -np 1` would bind the
# process to one core and serialise all its threads.
run_cfg() {
    local ranks=$1 threads=$2; shift 2
    if (( ranks == 1 )); then
        OMP_NUM_THREADS=$threads "$HEAT_BIN" "$@" --csv
    else
        # PE=threads gives each rank its own `threads` cores (default would be 1).
        OMP_NUM_THREADS=$threads "$MPIRUN" -np "$ranks" \
            --map-by "slot:PE=$threads" --bind-to "$MPI_BIND" \
            -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
            $MPIRUN_EXTRA "$HEAT_BIN" "$@" --csv
    fi
}

# bench CSV "RANKS THREADS heat-args..." ...   -> appends REPS lines per config
#
# Configurations are INTERLEAVED: repetition 1 of every config, then repetition
# 2 of every config, and so on. Slow drifts in the machine (thermal throttling,
# turbo budget, background load) then hit every config equally instead of
# biasing whichever one happened to run during the bad minute.
bench() {
    local csv=$1; shift
    local cfgs=("$@") r cfg line
    for (( r = 1; r <= REPS; r++ )); do
        for cfg in "${cfgs[@]}"; do
            # shellcheck disable=SC2086
            line=$(run_cfg $cfg)
            echo "$line" >> "$csv"
            echo "  [$r/$REPS] $line"
        done
    done
}

new_csv() {
    "$HEAT_BIN" --csv-header > "$1"
}

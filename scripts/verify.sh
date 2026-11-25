#!/usr/bin/env bash
# Correctness check beyond ctest: every mode on a medium grid must match the
# serial reference bit-for-bit (--verify exits non-zero otherwise), and the
# final grids written by different modes must be byte-identical files.
source "$(dirname "$0")/common.sh"

N="${N:-513}"
STEPS="${STEPS:-300}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

check() {
    local name=$1 ranks=$2 threads=$3; shift 3
    if run_cfg "$ranks" "$threads" "$N" "$STEPS" --verify --out "$TMP/$name.bin" "$@" >/dev/null 2>"$TMP/$name.log" \
       && grep -q PASS "$TMP/$name.log"; then
        echo "PASS  $name"
    else
        echo "FAIL  $name"; cat "$TMP/$name.log"; fail=1
    fi
}

check serial   1 1
check openmp   1 4
check mpi4     4 1
check mpi3     3 1
check hybrid   2 2
check blocked  2 2 --block 64
check overlap  4 1 --overlap
check both     2 2 --block 64 --overlap

for f in "$TMP"/*.bin; do
    cmp -s "$TMP/serial.bin" "$f" || { echo "FAIL  $(basename "$f") differs from serial.bin"; fail=1; }
done
(( fail == 0 )) && echo "all modes bit-identical to the serial reference"
exit $fail

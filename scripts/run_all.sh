#!/usr/bin/env bash
# Full pipeline: build, test, benchmark, plot, animate.
# Quick smoke run:  QUICK=1 scripts/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${QUICK:-0}" == 1 ]]; then
    export NX=${NX:-512} STEPS=${STEPS:-100} ROWS_PER_CORE=${ROWS_PER_CORE:-128} REPS=${REPS:-3}
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
(cd build && ctest --output-on-failure)

scripts/run_strong.sh
scripts/run_weak.sh
scripts/run_modes.sh
scripts/run_opt.sh
python3 scripts/plot.py
scripts/make_gif.sh
echo "done: see results/"

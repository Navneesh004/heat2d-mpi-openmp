#!/usr/bin/env bash
# Heatmap animation of heat spreading from the hot top edge.
# Penetration depth grows like sqrt(steps), so a small grid needs many steps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEAT_BIN="${HEAT_BIN:-$ROOT/build/heat}"
N="${GIF_N:-128}"
STEPS="${GIF_STEPS:-16000}"
EVERY="${GIF_EVERY:-160}"
DIR="$ROOT/snapshots"

rm -rf "$DIR"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" "$HEAT_BIN" "$N" "$STEPS" --snap "$EVERY" --snap-dir "$DIR"
python3 "$ROOT/scripts/animate.py" "$DIR" "$ROOT/results/heat.gif"

#!/usr/bin/env python3
"""Parse results/perf_stat_*.txt and print IPC / cache-miss table (also to perf_summary.md)."""
import re
import sys
from pathlib import Path

results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"


def parse(path):
    vals = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([\d,.]+)\s+(?:\S+\s+)?([A-Za-z][\w-]*)", line)
        if m and not line.strip().startswith("#"):
            try:
                vals[m.group(2)] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        m = re.match(r"\s*([\d.]+) seconds time elapsed", line)
        if m:
            vals["elapsed"] = float(m.group(1))
    return vals


rows = ["| run | time (s) | IPC | cache-miss rate | LLC-load miss rate |", "|---|---|---|---|---|"]
for f in sorted(results.glob("perf_stat_*.txt")):
    v = parse(f)
    ipc = v.get("instructions", 0) / v["cycles"] if v.get("cycles") else float("nan")
    cm = v.get("cache-misses", 0) / v["cache-references"] * 100 if v.get("cache-references") else float("nan")
    llc = v.get("LLC-load-misses", 0) / v["LLC-loads"] * 100 if v.get("LLC-loads") else float("nan")
    name = f.stem.replace("perf_stat_", "")
    rows.append(f"| {name} | {v.get('elapsed', float('nan')):.3f} | {ipc:.2f} | {cm:.1f}% | {llc:.1f}% |")

text = "\n".join(rows)
print(text)
(results / "perf_summary.md").write_text("# perf stat summary\n\n" + text + "\n")

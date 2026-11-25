#!/usr/bin/env python3
"""Turn results/*.csv into charts and a markdown summary.

Every configuration is run REPS times; we use the median wall time.
Outputs (whichever inputs exist):
  results/strong_scaling.png   speedup + efficiency vs cores
  results/weak_scaling.png     time + efficiency vs cores
  results/modes.png            OpenMP vs MPI vs hybrid splits, compute/halo stacked
  results/opt.png              before/after for blocking and overlap
  results/summary.md           the same numbers as tables
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"

# Fixed categorical order: colour follows the mode, never its rank.
COLORS = {"openmp": "#2a78d6", "mpi": "#eb6834", "hybrid": "#1baf7a", "serial": "#52514e"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
    "font.size": 10,
})

summary = []


def load(name):
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    keys = ["mode", "ranks", "threads", "cores", "nx", "ny", "steps", "kernel", "overlap"]

    # Use the MEDIAN RUN of each configuration (the run whose total time is the
    # median), not separate medians of total/compute/halo: those could come from
    # different runs and then compute + halo would not add up to the total.
    def median_run(g):
        g = g.sort_values("total_s")
        row = g.iloc[(len(g) - 1) // 2].copy()
        row["runs"] = len(g)
        row["spread_s"] = g.total_s.max() - g.total_s.min()
        return row

    rows = [median_run(g) for _, g in df.groupby(keys)]
    return pd.DataFrame(rows).reset_index(drop=True)


def label(row):
    return f"{row['ranks']}x{row['threads']}"


def strong():
    df = load("strong")
    if df is None:
        return
    t1 = df[df.cores == 1].total_s.min()
    df["speedup"] = t1 / df.total_s
    df["efficiency"] = df.speedup / df.cores
    maxp = df.cores.max()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot([1, maxp], [1, maxp], "--", color=MUTED, linewidth=1, label="ideal")
    for mode in ["openmp", "mpi", "hybrid"]:
        d = df[(df["mode"] == mode) | (df.cores == 1)].sort_values("cores")  # all start at serial
        if len(d) < 2:
            continue
        a1.plot(d.cores, d.speedup, "o-", color=COLORS[mode], label=mode)
        a2.plot(d.cores, d.efficiency * 100, "o-", color=COLORS[mode], label=mode)
    nx, ny, steps = df.nx.iloc[0], df.ny.iloc[0], df.steps.iloc[0]
    a1.set(title=f"Strong scaling - {ny}x{nx} grid, {steps} steps", xlabel="cores", ylabel="speedup vs 1 core")
    a2.axhline(100, ls="--", color=MUTED, lw=1)
    a2.set(title="Parallel efficiency", xlabel="cores", ylabel="efficiency (%)", ylim=(0, max(110, a2.get_ylim()[1])))
    for a in (a1, a2):
        a.set_xticks(sorted(df.cores.unique()))
        a.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "strong_scaling.png", dpi=150)
    plt.close(fig)

    summary.append(f"## Strong scaling ({ny}x{nx}, {steps} steps, median of {df.runs.max()} runs)\n")
    summary.append("| mode | ranks x threads | cores | time (s) | halo (s, mean/rank) | speedup | efficiency |")
    summary.append("|---|---|---|---|---|---|---|")
    for _, r in df.sort_values(["cores", "mode"]).iterrows():
        summary.append(f"| {r['mode']} | {label(r)} | {r.cores} | {r.total_s:.3f} | {r.halo_s:.3f} "
                       f"| {r.speedup:.2f}x | {r.efficiency * 100:.0f}% |")
    summary.append("")


def weak():
    df = load("weak")
    if df is None:
        return
    t1 = df[df.cores == 1].total_s.min()
    df["efficiency"] = t1 / df.total_s
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in ["openmp", "mpi"]:
        d = df[(df["mode"] == mode) | (df.cores == 1)].sort_values("cores")
        if len(d) < 2:
            continue
        a1.plot(d.cores, d.total_s, "o-", color=COLORS[mode], label=mode)
        a2.plot(d.cores, d.efficiency * 100, "o-", color=COLORS[mode], label=mode)
    a1.axhline(t1, ls="--", color=MUTED, lw=1, label="ideal (flat)")
    rows_per_core = df.ny.iloc[0] // df.cores.iloc[0]
    a1.set(title=f"Weak scaling - {df.nx.iloc[0]} cols x {rows_per_core} rows per core",
           xlabel="cores", ylabel="time (s)", ylim=(0, None))
    a2.axhline(100, ls="--", color=MUTED, lw=1)
    a2.set(title="Weak-scaling efficiency (T1 / Tp)", xlabel="cores", ylabel="efficiency (%)",
           ylim=(0, max(110, a2.get_ylim()[1])))
    for a in (a1, a2):
        a.set_xticks(sorted(df.cores.unique()))
        a.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "weak_scaling.png", dpi=150)
    plt.close(fig)

    summary.append(f"## Weak scaling ({df.nx.iloc[0]} cols x {rows_per_core} rows/core)\n")
    summary.append("| mode | cores | grid | time (s) | halo (s, mean/rank) | efficiency |")
    summary.append("|---|---|---|---|---|---|")
    for _, r in df.sort_values(["cores", "mode"]).iterrows():
        summary.append(f"| {r['mode']} | {r.cores} | {r.ny}x{r.nx} | {r.total_s:.3f} | {r.halo_s:.3f} "
                       f"| {r.efficiency * 100:.0f}% |")
    summary.append("")


def stacked_bars(df, names, title, path):
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(df) + 3.5), 4.2))
    x = range(len(df))
    colors = [COLORS[m] for m in df["mode"]]
    # compute = solid, halo = lighter cap; 2px surface gap via white edge
    ax.bar(x, df.compute_s, color=colors, edgecolor="#fcfcfb", linewidth=2, width=0.6, label="compute")
    ax.bar(x, df.halo_s, bottom=df.compute_s, color=colors, alpha=0.4, edgecolor="#fcfcfb",
           linewidth=2, width=0.6, label="halo exchange")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r.total_s * 1.02, f"{r.total_s:.2f}s", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set(title=title, ylabel="time (s) - median run", ylim=(0, df.total_s.max() * 1.12))
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label="compute"), Patch(color=MUTED, alpha=0.4, label="halo exchange + wait")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), title="mean per rank", alignment="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def modes():
    df = load("modes")
    if df is None:
        return
    order = {"openmp": 0, "hybrid": 1, "mpi": 2}
    df = df.assign(o=df["mode"].map(order)).sort_values(["o", "ranks"]).reset_index(drop=True)
    names = [f"{r['mode']}\n{label(r)}" for _, r in df.iterrows()]
    p = df.cores.iloc[0]
    stacked_bars(df, names, f"Mode comparison at {p} cores ({df.ny.iloc[0]}x{df.nx.iloc[0]})",
                 RESULTS / "modes.png")
    summary.append(f"## Mode comparison at {p} cores\n")
    summary.append("| mode | ranks x threads | time (s) | compute (s, mean/rank) | halo (s, mean/rank) | MLUP/s |")
    summary.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        summary.append(f"| {r['mode']} | {label(r)} | {r.total_s:.3f} | {r.compute_s:.3f} | {r.halo_s:.3f} | {r.mlups:.0f} |")
    summary.append("")


def opt():
    df = load("opt")
    if df is None:
        return
    df = df.sort_values(["mode", "kernel", "overlap"], ascending=[False, False, True]).reset_index(drop=True)

    def vname(r):
        parts = [p for p in [("blocked" if r.kernel == "blocked" else ""), ("overlap" if r.overlap else "")] if p]
        return "+".join(parts) or "baseline"

    names = [f"{r['mode']} {label(r)}\n{vname(r)}" for _, r in df.iterrows()]
    stacked_bars(df, names, "Optimisations: before / after", RESULTS / "opt.png")
    summary.append("## Optimisations (before / after)\n")
    summary.append("| config | variant | time (s) | halo (s, mean/rank) | change vs baseline |")
    summary.append("|---|---|---|---|---|")
    for mode, g in df.groupby("mode", sort=False):
        base = g[(g.kernel == "naive") & (g.overlap == 0)].total_s
        base = base.iloc[0] if len(base) else float("nan")
        for _, r in g.iterrows():
            summary.append(f"| {mode} {label(r)} | {vname(r)} | {r.total_s:.3f} | {r.halo_s:.3f} "
                           f"| {(r.total_s / base - 1) * 100:+.1f}% |")
    summary.append("")


if __name__ == "__main__":
    for f in (strong, weak, modes, opt):
        f()
    if summary:
        (RESULTS / "summary.md").write_text("# Benchmark results\n\n" + "\n".join(summary) + "\n")
        print("\n".join(summary))
        print(f"charts and summary.md written to {RESULTS}")
    else:
        print(f"no CSVs found in {RESULTS}; run the scripts/run_*.sh benchmarks first")

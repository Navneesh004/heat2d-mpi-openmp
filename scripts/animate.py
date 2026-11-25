#!/usr/bin/env python3
"""Turn snapshot files (written by `heat --snap K`) into a heatmap GIF.

usage: animate.py SNAPSHOT_DIR OUT.gif [--fps 15]
Snapshot format: int32 ny, int32 nx, int32 step, then ny*nx float32.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def read_snapshot(path):
    with open(path, "rb") as f:
        ny, nx, step = np.fromfile(f, dtype=np.int32, count=3)
        data = np.fromfile(f, dtype=np.float32, count=ny * nx).reshape(ny, nx)
    return int(step), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapdir")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()

    files = sorted(Path(a.snapdir).glob("snap_*.bin"))
    if not files:
        raise SystemExit(f"no snapshots in {a.snapdir}")
    frames = [read_snapshot(f) for f in files]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0][1], cmap="inferno", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="temperature (°C)")
    title = ax.set_title("")

    def update(k):
        step, data = frames[k]
        im.set_data(data)
        title.set_text(f"2D heat diffusion - step {step}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
    print(f"wrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

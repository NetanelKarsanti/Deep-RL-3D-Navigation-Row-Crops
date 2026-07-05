"""
Top-down heatmap of the grid environment.
Each cell is colored by the height of its obstacle (0 = free).

Usage:
    python show_env.py                          # default 12x8x5
    python show_env.py --grid-l 100 --grid-w 100 --grid-h 20
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from grid3d_env import Grid3DEnv


def show(L, W, H, out_path: Path):
    env = Grid3DEnv(L=L, W=W, H=H)

    # height map: for each (x,y) how many z-levels are blocked
    height_map = env.occupancy.sum(axis=2).astype(float)   # shape (L, W)
    height_map[height_map == 0] = np.nan                   # free cells → transparent

    fig, ax = plt.subplots(figsize=(10, 8))

    # background — free cells in light gray
    ax.imshow(np.zeros((W, L)), vmin=0, vmax=H,
              cmap="Greys", origin="lower", aspect="equal", alpha=0.15)

    # obstacle height map (x=cols, y=rows → transpose)
    im = ax.imshow(height_map.T, vmin=1, vmax=H,
                   cmap="YlOrRd", origin="lower", aspect="equal", alpha=0.85)

    # goal
    gx, gy, _ = env.goal
    ax.scatter(gx, gy, s=200, marker="*", color="green",
               zorder=5, label=f"Goal {env.goal}")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Obstacle height (blocked z-levels)", fontsize=10)

    free_patch    = mpatches.Patch(color="#d0d0d0", label="Free cell")
    blocked_patch = mpatches.Patch(color="#d73027", label="Obstacle (tallest)")
    ax.legend(handles=[free_patch, blocked_patch,
                        mpatches.Patch(color="green", label=f"Goal {env.goal}")],
              loc="upper right", fontsize=9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Grid3DEnv {L}×{W}×{H} — top-down obstacle map")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, W - 0.5)

    # grid lines for small grids
    if L <= 20 and W <= 20:
        ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, W, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.4, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Goal: {env.goal}")
    print(f"Obstacle cells: {int(env.occupancy.sum())} / {L*W*H} ({100*env.occupancy.sum()/(L*W*H):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-l", type=int, default=50)
    parser.add_argument("--grid-w", type=int, default=50)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--out",    type=str, default=None)
    args = parser.parse_args()

    out = args.out or f"env_{args.grid_l}x{args.grid_w}x{args.grid_h}.png"
    root = Path(__file__).resolve().parent
    show(args.grid_l, args.grid_w, args.grid_h, root / out)


if __name__ == "__main__":
    main()

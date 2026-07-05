"""Quality two-panel visualization of Grid3DEnv.

Left  : the environment in 3D (voxel objects + agent + goal).
Right : a 2D height map — for every (x, y) ground cell, the height of the
        object stacked there (obstacles fill z = 0..h-1, so height = column sum).

Run:  python visualize_env.py   ->  saves env_overview.png
"""

import matplotlib
matplotlib.use("Agg")                      # headless: render straight to a file
import matplotlib.pyplot as plt
import numpy as np

from grid3d_env import Grid3DEnv, L, W, H


def main():
    env = Grid3DEnv()
    env.reset(seed=0)

    # Height of the object column at each (x, y). Objects fill z = 0..h-1
    # contiguously, so the column sum is exactly the object height.
    heightmap = env.occupancy.sum(axis=2)               # shape (L, W)

    # A representative color per (x, y) column (color of its top-most cell) so the
    # 3D and 2D panels read as the same scene.
    fig = plt.figure(figsize=(15, 6.2))
    fig.suptitle("Grid3DEnv  —  3D scene and 2D height map",
                 fontsize=15, fontweight="bold")

    # ---------------------------------------------------------------- 3D panel
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.voxels(env.occupancy, facecolors=env.facecolors,
                edgecolor="k", linewidth=0.3, alpha=0.75)
    ax3d.scatter(*[c + 0.5 for c in env.pos], color="red", s=220,
                 marker="o", depthshade=False, label="agent")
    ax3d.scatter(*[c + 0.5 for c in env.goal], color="green", s=340,
                 marker="*", depthshade=False, label="goal")
    ax3d.set_xlim(0, L); ax3d.set_ylim(0, W); ax3d.set_zlim(0, H)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z (height)")
    ax3d.set_title("3D environment", fontsize=12)
    ax3d.view_init(elev=28, azim=-52)
    ax3d.legend(loc="upper left")

    # ------------------------------------------------------------ height map
    ax2d = fig.add_subplot(1, 2, 2)
    # imshow expects (rows, cols) = (y, x); transpose and origin="lower" so the
    # axes match the 3D panel (x rightwards, y upwards).
    im = ax2d.imshow(heightmap.T, origin="lower", cmap="viridis",
                     vmin=0, vmax=H, extent=[0, L, 0, W], aspect="equal")

    # Annotate each cell with its height for exact readability.
    for x in range(L):
        for y in range(W):
            h = heightmap[x, y]
            ax2d.text(x + 0.5, y + 0.5, str(int(h)),
                      ha="center", va="center", fontsize=8,
                      color="white" if h < H * 0.6 else "black")

    # Mark agent and goal on the grid too.
    ax2d.scatter(env.pos[0] + 0.5, env.pos[1] + 0.5, color="red", s=180,
                 marker="o", edgecolor="white", linewidth=1.2, label="agent", zorder=3)
    ax2d.scatter(env.goal[0] + 0.5, env.goal[1] + 0.5, color="lime", s=280,
                 marker="*", edgecolor="black", linewidth=1.0, label="goal", zorder=3)

    ax2d.set_xticks(range(L + 1)); ax2d.set_yticks(range(W + 1))
    ax2d.grid(color="white", linewidth=0.4, alpha=0.3)
    ax2d.set_xlabel("x"); ax2d.set_ylabel("y")
    ax2d.set_title("2D height map (object height per ground cell)", fontsize=12)
    ax2d.legend(loc="upper left")
    cbar = fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
    cbar.set_label("height (occupied z-levels)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "env_overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved: {out}")
    print("height map (x rows, y cols):")
    print(heightmap.T[::-1])                # printed y-up to match the plot


if __name__ == "__main__":
    main()

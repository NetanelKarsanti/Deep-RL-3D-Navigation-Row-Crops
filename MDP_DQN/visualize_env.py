"""Quality two-panel visualization of the (larger) MDP_DQN Grid3DEnv.

Left  : the environment in 3D — a light-brown ground floor (z=0), voxel objects
        (green crop rows, distinctly-colored obstacles), agent and goal.
Right : a 2D top-down map using the SAME colors — light-brown floor, green crop
        rows, and each obstacle in its own color.

Scales to the env size read from the Grid3DEnv instance (default 50×50×4).

Run:  python visualize_env.py   ->  saves env_overview.png
"""

import matplotlib
matplotlib.use("Agg")                      # headless: render straight to a file
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

from grid3d_env import Grid3DEnv

FLOOR_COLOR = "#deb887"                     # light brown (burlywood) ground, z=0
CROP_COLOR  = "#90ee90"                     # crop-row green (matches the env)


def main():
    env = Grid3DEnv()                      # 50×50×4 by default
    env.reset(seed=0)
    L, W, H = env.L, env.W, env.H

    # Height of the object column at each (x, y). Objects fill z = 0..h-1
    # contiguously, so the column sum is exactly the object height.
    heightmap = env.occupancy.sum(axis=2)               # shape (L, W)

    fig = plt.figure(figsize=(15.5, 7))
    fig.suptitle(f"Grid3DEnv {L}×{W}×{H}  —  3D scene and 2D top-down map",
                 fontsize=15, fontweight="bold")

    # ---------------------------------------------------------------- 3D panel
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # Render the floor + obstacles in ONE voxels() call so they share the same
    # per-cube depth sorting — a separate plot_surface floor gets mis-sorted as a
    # single polygon (either washing out or hiding the obstacles). The floor is a
    # thin slab in z=[-0.3, 0]; obstacle level k keeps its true span z=[k-1, k].
    x_edges = np.arange(L + 1)
    y_edges = np.arange(W + 1)
    z_edges = np.concatenate([[-0.3], np.arange(H + 1)])      # thin floor + H levels
    xc, yc, zc = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    filled = np.zeros((L, W, H + 1), dtype=bool)
    filled[:, :, 0]  = True                                   # floor everywhere
    filled[:, :, 1:] = env.occupancy                          # obstacles on top

    fc = np.empty((L, W, H + 1), dtype=object)
    fc[:, :, 0]  = FLOOR_COLOR
    fc[:, :, 1:] = env.facecolors

    ax3d.voxels(xc, yc, zc, filled, facecolors=fc,
                edgecolor="k", linewidth=0.1, alpha=0.92)
    ax3d.scatter(*[c + 0.5 for c in env.pos], color="red", s=160,
                 marker="o", depthshade=False, label="agent")
    ax3d.scatter(*[c + 0.5 for c in env.goal], color="green", s=320,
                 marker="*", depthshade=False, label="goal")
    ax3d.set_xlim(0, L); ax3d.set_ylim(0, W); ax3d.set_zlim(-0.3, H)
    ax3d.set_box_aspect((L, W, H * 4))      # exaggerate z so short obstacles read
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z (height)")
    ax3d.set_title("3D environment", fontsize=12)
    ax3d.view_init(elev=42, azim=-58)
    ax3d.legend(loc="upper left")

    # ------------------------------------------------------------ 2D top-down
    # Build an RGB image colored exactly like the 3D scene: floor everywhere,
    # then each occupied cell painted with its object's color (top-most level).
    img = np.empty((W, L, 3), dtype=float)
    img[:] = to_rgb(FLOOR_COLOR)
    for x in range(L):
        for y in range(W):
            hv = heightmap[x, y]
            if hv > 0:
                img[y, x] = to_rgb(env.facecolors[x, y, hv - 1])   # object color

    ax2d = fig.add_subplot(1, 2, 2)
    # origin="lower" so the axes match the 3D panel (x rightwards, y upwards).
    ax2d.imshow(img, origin="lower", extent=[0, L, 0, W], aspect="equal",
                interpolation="nearest")

    ax2d.scatter(env.pos[0] + 0.5, env.pos[1] + 0.5, color="red", s=140,
                 marker="o", edgecolor="white", linewidth=1.2, label="agent", zorder=3)
    ax2d.scatter(env.goal[0] + 0.5, env.goal[1] + 0.5, color="green", s=320,
                 marker="*", edgecolor="white", linewidth=1.0, label="goal", zorder=3)

    ax2d.set_xticks(range(0, L + 1, max(1, round(L / 10 / 5) * 5) if L > 20 else 1))
    ax2d.set_yticks(range(0, W + 1, max(1, round(W / 10 / 5) * 5) if W > 20 else 1))
    ax2d.set_xlabel("x"); ax2d.set_ylabel("y")
    ax2d.set_title("2D top-down map (floor / crop rows / obstacles)", fontsize=12)

    legend_items = [
        Patch(facecolor=FLOOR_COLOR, edgecolor="k", label="floor (z=0)"),
        Patch(facecolor=CROP_COLOR, edgecolor="k", label="crop rows"),
        Patch(facecolor="#7f7f7f", edgecolor="k", label="obstacles (varied colors/heights)"),
        Line2D([], [], color="red", marker="o", linestyle="", markeredgecolor="white",
               markersize=9, label="agent"),
        Line2D([], [], color="green", marker="*", linestyle="", markeredgecolor="white",
               markersize=13, label="goal"),
    ]
    ax2d.legend(handles=legend_items, loc="upper left", fontsize=8,
                framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "env_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved: {out}   (grid {L}×{W}×{H})")
    occ_cells = int((heightmap > 0).sum())
    print(f"occupied ground cells: {occ_cells}/{L * W}  "
          f"({100 * occ_cells / (L * W):.1f}%)   max height: {int(heightmap.max())}")


if __name__ == "__main__":
    main()

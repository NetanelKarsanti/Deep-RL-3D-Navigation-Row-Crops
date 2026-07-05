"""
Run one greedy episode with the trained DQN agent and save it as a GIF.

Usage:
    python visualize_agent.py
    python visualize_agent.py --model dqn_results/online_q_network.pt --seed 7
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

import numpy as np
import torch

from grid3d_env import Grid3DEnv, ACTIONS
from train_dqn import TwoLayerQNetwork, normalize_obs, N_ACTIONS, OBS_DIM

ACTION_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def load_model(path: Path, device: str, hidden_dim: int = 128) -> TwoLayerQNetwork:
    net = TwoLayerQNetwork(input_dim=OBS_DIM, hidden_dim=hidden_dim, output_dim=N_ACTIONS).to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


def greedy_action(net, obs: np.ndarray, grid_max: np.ndarray, device: str) -> int:
    state = torch.as_tensor(normalize_obs(obs, grid_max), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(net(state), dim=1).item())


def draw_frame(ax, env, trajectory, total_reward, step):
    """Custom draw for large grids: stretched z-axis, path trail, scaled markers."""
    L, W, H = env.L, env.W, env.H

    # Stretch z-axis so the 4-level structure is clearly visible
    z_scale = max(2, L // (H * 3))

    ax.clear()

    # Obstacles — no edge lines (cleaner on large grid), semi-transparent
    ax.voxels(env.occupancy, facecolors=env.facecolors, edgecolor="none", alpha=0.45)

    # Path trail
    if len(trajectory) > 1:
        xs = [p[0] + 0.5 for p in trajectory]
        ys = [p[1] + 0.5 for p in trajectory]
        zs = [p[2] + 0.5 for p in trajectory]
        ax.plot(xs, ys, zs, color="white", linewidth=2.5, alpha=0.85, zorder=4)

    # Marker sizes scale with grid
    marker_base = max(60, 120_000 // (L * W))
    goal_size  = marker_base * 5
    agent_size = marker_base * 3

    # Goal — bright lime star, always on top
    gx, gy, gz = env.goal
    ax.scatter(gx + 0.5, gy + 0.5, gz + 0.5,
               color="lime", s=goal_size, marker="*",
               depthshade=False, label="goal", zorder=10)

    # Agent — red circle
    x, y, z = env.pos
    ax.scatter(x + 0.5, y + 0.5, z + 0.5,
               color="red", s=agent_size, marker="o",
               depthshade=False, label="agent", zorder=10)

    # Axis limits and stretched z-box
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect([L, W, H * z_scale])

    # Labels
    ax.set_xlabel("x", fontsize=9, labelpad=2)
    ax.set_ylabel("y", fontsize=9, labelpad=2)
    ax.set_zlabel("z", fontsize=9, labelpad=2)
    ax.tick_params(labelsize=7)

    # Camera angle: slight elevation to see z-layers, azimuth from back-left
    ax.view_init(elev=28, azim=225)

    ax.set_title(
        f"Grid3DEnv {L}×{W}×{H}   step {step}   reward {total_reward:.1f}",
        fontsize=10, pad=6
    )
    ax.legend(loc="upper right", fontsize=9, markerscale=0.8)


def run(model_path: Path, seed: int, out_gif: Path, out_png: Path, noise_prob: float,
        grid_l: int = 50, grid_w: int = 50, grid_h: int = 4, hidden_dim: int = 128):
    device = "cpu"
    net = load_model(model_path, device, hidden_dim=hidden_dim)

    env = Grid3DEnv(L=grid_l, W=grid_w, H=grid_h, noise_prob=noise_prob)
    grid_max = np.array([env.L - 1, env.W - 1, env.H - 1], dtype=np.float32)
    obs, _ = env.reset(seed=seed)

    print(f"Start: {env.pos}  |  Goal: {env.goal}")
    print(f"{'Step':>4}  {'Pos':>16}  {'Action':>5}  {'Reward':>7}  {'Done'}")
    print("-" * 55)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=4)

    total = 0.0
    step = 0
    trajectory = [env.pos]

    with writer.saving(fig, str(out_gif), dpi=100):
        draw_frame(ax, env, trajectory, total, step)
        writer.grab_frame()

        while True:
            action = greedy_action(net, obs, grid_max, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            step += 1
            trajectory.append(env.pos)

            print(f"{step:>4}  {str(env.pos):>16}  {ACTION_NAMES[action]:>5}  {reward:>7.1f}  "
                  f"{'GOAL' if terminated else 'trunc' if truncated else ''}")

            draw_frame(ax, env, trajectory, total, step)
            writer.grab_frame()

            obs = next_obs
            if terminated or truncated:
                break

        for _ in range(6):  # linger on final frame
            draw_frame(ax, env, trajectory, total, step)
            writer.grab_frame()

    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("-" * 55)
    print(f"Total reward: {total:.1f}  |  Steps: {step}  |  "
          f"{'Reached goal!' if env.pos == env.goal else 'Did not reach goal'}")
    print(f"Saved: {out_gif}  and  {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dqn_results/online_q_network.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--out-gif", type=str, default="agent_episode.gif")
    parser.add_argument("--out-png", type=str, default="agent_episode_final.png")
    parser.add_argument("--grid-l", type=int, default=50)
    parser.add_argument("--grid-w", type=int, default=50)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    run(
        model_path=root / args.model,
        seed=args.seed,
        out_gif=root / args.out_gif,
        out_png=root / args.out_png,
        noise_prob=args.noise_prob,
        grid_l=args.grid_l,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()

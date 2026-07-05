"""
Post-training analysis for Grid3DEnv DQN agent.

Generates three figures:
  analysis_training.png  — training curve from evaluation.csv
  analysis_metrics.png   — episode length, path efficiency, z-distribution, reward histogram
  analysis_heatmap.png   — per-cell success rate across all starting positions

Usage:
    python analyze_results.py
    python analyze_results.py --episodes 500 --no-heatmap
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from grid3d_env import Grid3DEnv
from train_dqn import TwoLayerQNetwork, normalize_obs, N_ACTIONS, OBS_DIM


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_model(path: Path, device: str, hidden_dim: int = 128) -> TwoLayerQNetwork:
    net = TwoLayerQNetwork(input_dim=OBS_DIM, hidden_dim=hidden_dim,
                           output_dim=N_ACTIONS).to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


def manhattan(a, b) -> int:
    return sum(abs(a[i] - b[i]) for i in range(3))


def run_episode(env, net, grid_max, device, start=None, max_steps=200) -> dict:
    """Greedy episode. If start is given, forces that starting cell."""
    obs, _ = env.reset()
    if start is not None:
        env.pos   = start
        env.steps = 0
        obs = env._get_obs()

    start_pos = env.pos
    trajectory = [start_pos]
    z_counts   = [0] * env.H
    total_reward = 0.0

    for _ in range(max_steps):
        state_t = torch.as_tensor(normalize_obs(obs, grid_max),
                                  dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(net(state_t), dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        trajectory.append(env.pos)
        z_counts[env.pos[2]] += 1

        if terminated or truncated:
            break

    steps   = len(trajectory) - 1
    success = env.pos == env.goal
    mdist   = max(1, manhattan(start_pos, env.goal))

    return {
        "start":          start_pos,
        "success":        success,
        "steps":          steps,
        "total_reward":   total_reward,
        "path_efficiency": steps / mdist,
        "manhattan_dist": mdist,
        "z_counts":       z_counts,
    }


# -----------------------------------------------------------------------------
# Plot 1 — Training curve
# -----------------------------------------------------------------------------

def plot_training_curve(csv_path: Path, out_path: Path) -> None:
    steps, means, stds = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["train_step"]))
            means.append(float(row["mean_return"]))
            stds.append(float(row["std_return"]))

    steps = np.array(steps)
    means = np.array(means)
    stds  = np.array(stds)

    w        = max(5, len(means) // 20)
    smoothed = np.convolve(means, np.ones(w) / w, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grid3D DQN — Training Curve", fontsize=13)

    for ax, y, title in zip(
        axes,
        [means, smoothed],
        ["Raw evaluation reward", f"Smoothed (rolling window={w})"],
    ):
        ax.fill_between(steps, means - stds, means + stds,
                        alpha=0.2, color="steelblue", label="±1 STD")
        ax.plot(steps, y, linewidth=1.8, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Episode reward")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Plot 2 — Episode metrics (2×2)
# -----------------------------------------------------------------------------

def plot_metrics(episodes: list, H: int, out_path: Path) -> None:
    succ = [e for e in episodes if e["success"]]
    fail = [e for e in episodes if not e["success"]]
    sr   = 100 * len(succ) / len(episodes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Agent Metrics  |  {len(episodes)} episodes  |  Success rate: {sr:.1f}%",
        fontsize=13,
    )

    # ── (0,0) Episode length distribution ────────────────────────────────────
    ax = axes[0, 0]
    if succ:
        ax.hist([e["steps"] for e in succ], bins=25, alpha=0.75,
                color="steelblue", label=f"Success ({len(succ)})")
    if fail:
        ax.hist([e["steps"] for e in fail], bins=25, alpha=0.75,
                color="salmon", label=f"Failure ({len(fail)})")
    ax.set_xlabel("Steps per episode")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── (0,1) Path efficiency scatter ────────────────────────────────────────
    ax = axes[0, 1]
    if succ:
        mdists = [e["manhattan_dist"] for e in succ]
        steps  = [e["steps"]          for e in succ]
        sc = ax.scatter(mdists, steps,
                        c=[e["total_reward"] for e in succ],
                        cmap="RdYlGn", alpha=0.6, s=30, zorder=3)
        plt.colorbar(sc, ax=ax, label="Total reward")
        mx = max(mdists)
        ax.plot([0, mx], [0, mx], "k--", linewidth=1.2, label="Optimal (ratio=1)")
    if fail:
        ax.scatter([e["manhattan_dist"] for e in fail],
                   [e["steps"]          for e in fail],
                   color="salmon", alpha=0.4, s=20, marker="x", label="Failure")
    ax.set_xlabel("Manhattan distance to goal")
    ax.set_ylabel("Actual steps taken")
    ax.set_title("Path Efficiency  (below dashed = better than optimal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) Z-level distribution ────────────────────────────────────────────
    ax = axes[1, 0]
    z_succ = np.zeros(H)
    z_fail = np.zeros(H)
    for e in succ: z_succ += np.array(e["z_counts"])
    for e in fail: z_fail += np.array(e["z_counts"])

    x = np.arange(H)
    bar_w = 0.35
    if z_succ.sum() > 0:
        ax.bar(x - bar_w / 2, 100 * z_succ / z_succ.sum(),
               bar_w, label="Success", color="steelblue", alpha=0.85)
    if z_fail.sum() > 0:
        ax.bar(x + bar_w / 2, 100 * z_fail / z_fail.sum(),
               bar_w, label="Failure", color="salmon", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z = {z}" for z in range(H)])
    ax.set_ylabel("% of steps")
    ax.set_title("Z-Level Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ── (1,1) Total reward distribution ──────────────────────────────────────
    ax = axes[1, 1]
    rewards = [e["total_reward"] for e in episodes]
    ax.hist(rewards, bins=30, color="mediumpurple", alpha=0.85, edgecolor="white")
    mean_r = np.mean(rewards)
    ax.axvline(mean_r, color="black", linewidth=1.8, linestyle="--",
               label=f"Mean: {mean_r:.1f}")
    ax.axvline(0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Total episode reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Plot 3 — Success heatmap
# -----------------------------------------------------------------------------

def plot_heatmap(env, net, grid_max, device, out_path: Path,
                 max_steps: int = 200) -> None:
    L, W = env.L, env.W
    success_map = np.full((L, W), np.nan)

    free_cells = [
        (x, y, 0) for x in range(L) for y in range(W)
        if not env._blocked(x, y, 0) and (x, y, 0) != env.goal
    ]

    print(f"Heatmap: running {len(free_cells)} starting positions...", flush=True)
    for i, start in enumerate(free_cells):
        ep = run_episode(env, net, grid_max, device,
                         start=start, max_steps=max_steps)
        success_map[start[0], start[1]] = float(ep["success"])
        if (i + 1) % 500 == 0:
            done_pct = 100 * (i + 1) / len(free_cells)
            print(f"  {i+1}/{len(free_cells)}  ({done_pct:.0f}%)", flush=True)

    # Obstacle height overlay
    height_map = env.occupancy.sum(axis=2).astype(float)
    height_map[height_map == 0] = np.nan

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(height_map.T, origin="lower", cmap="Greys",
              vmin=0, vmax=env.H, alpha=0.25, aspect="equal")
    im = ax.imshow(success_map.T, origin="lower", cmap="RdYlGn",
                   vmin=0, vmax=1, alpha=0.85, aspect="equal")
    plt.colorbar(im, ax=ax, label="Success (1=green) / Failure (0=red)",
                 fraction=0.03)

    gx, gy, _ = env.goal
    ax.scatter(gx, gy, s=350, marker="*", color="lime",
               zorder=5, label=f"Goal {env.goal}")

    total  = int(np.sum(~np.isnan(success_map)))
    n_succ = int(np.nansum(success_map))
    sr     = 100 * np.nanmean(success_map)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Success Rate by Starting Position\n"
        f"Overall: {sr:.1f}%  ({n_succ} / {total} cells)"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="dqn_results/online_q_network.pt")
    parser.add_argument("--results",    type=str, default="dqn_results")
    parser.add_argument("--episodes",   type=int, default=300)
    parser.add_argument("--max-steps",  type=int, default=200)
    parser.add_argument("--grid-l",     type=int, default=50)
    parser.add_argument("--grid-w",     type=int, default=50)
    parser.add_argument("--grid-h",     type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip the heatmap (saves ~60 seconds)")
    parser.add_argument("--random-goal", action="store_true",
                        help="Randomize the goal each episode (match a random-goal-trained policy).")
    args = parser.parse_args()

    root       = Path(__file__).resolve().parent
    model_path = root / args.model
    out_dir    = root / args.results

    device = "cpu"
    net    = load_model(model_path, device, args.hidden_dim)
    env    = Grid3DEnv(L=args.grid_l, W=args.grid_w, H=args.grid_h, noise_prob=0.0,
                       random_goal=args.random_goal)
    grid_max = np.array([env.L - 1, env.W - 1, env.H - 1], dtype=np.float32)

    # ── episodes ──────────────────────────────────────────────────────────────
    if args.random_goal:
        # Goal-conditioned task: let reset() pick BOTH a random goal and a random
        # start each episode (it guarantees start != goal; forcing a start could
        # collide with the random goal). Seed the env once so the stream is
        # reproducible; metrics are computed per-episode against that episode's goal.
        env.reset(seed=args.seed)
        print(f"Running {args.episodes} random-goal episodes...", flush=True)
        episodes = [
            run_episode(env, net, grid_max, device, start=None,
                        max_steps=args.max_steps)
            for _ in range(args.episodes)
        ]
    else:
        # Fixed goal: sweep random START positions against the single fixed goal.
        free = [
            (x, y, 0) for x in range(env.L) for y in range(env.W)
            if not env._blocked(x, y, 0) and (x, y, 0) != env.goal
        ]
        rng = np.random.default_rng(args.seed)
        print(f"Running {args.episodes} random-start episodes...", flush=True)
        episodes = [
            run_episode(env, net, grid_max, device,
                        start=free[int(rng.integers(len(free)))],
                        max_steps=args.max_steps)
            for _ in range(args.episodes)
        ]

    sr       = 100 * sum(e["success"] for e in episodes) / len(episodes)
    succ_eps = [e for e in episodes if e["success"]]
    avg_steps = np.mean([e["steps"] for e in succ_eps]) if succ_eps else float("nan")
    avg_eff   = np.mean([e["path_efficiency"] for e in succ_eps]) if succ_eps else float("nan")

    print(f"\nResults over {len(episodes)} episodes:")
    print(f"  Success rate:         {sr:.1f}%")
    print(f"  Avg steps (success):  {avg_steps:.1f}")
    print(f"  Avg path efficiency:  {avg_eff:.2f}x optimal")

    # ── generate plots ────────────────────────────────────────────────────────
    csv_path = out_dir / "evaluation.csv"
    if csv_path.exists():
        plot_training_curve(csv_path, out_dir / "analysis_training.png")

    plot_metrics(episodes, env.H, out_dir / "analysis_metrics.png")

    if args.random_goal:
        print("Skipping success heatmap (random goal: no single fixed goal to sweep).", flush=True)
    elif not args.no_heatmap:
        plot_heatmap(env, net, grid_max, device,
                     out_dir / "analysis_heatmap.png",
                     max_steps=args.max_steps)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

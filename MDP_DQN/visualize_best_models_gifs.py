"""Two 2x2 GIFs of the *best* trained models (no training — reuses checkpoints).

Static goal  : one fixed goal, 4 panels each rolling out from a DIFFERENT start.
Dynamic goal : 4 panels, each an EASY goal (dmap==1.0 from the best_models_2x2
               sweep) reached from a random start.

Same checkpoints / env config as paper_figures/plot_best_models_2x2.py.

Run (parl_gpu env):
    python visualize_best_models_gifs.py
Outputs -> paper_figures/gif_static_2x2.gif , paper_figures/gif_dynamic_2x2.gif
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from grid3d_env import Grid3DEnv                                  # noqa: E402
from Q3_DQN import TwoLayerQNetwork, normalize_obs, N_ACTIONS, OBS_DIM  # noqa: E402

EXP = ROOT.parent / "experiments"
STATIC_CK = EXP / "controlled_static_vs_dynamic_shaping/static/seed_001/online_q_network.pt"
DYNAMIC_CK = EXP / "06_random_goal_her/results/dqn/seed_001/online_q_network.pt"
CACHE = ("/tmp/claude-1716625076/-home-coder-project-PHD-PARL-Project/"
         "7b659fde-7807-445b-9b1a-e65545fa77f6/scratchpad/best_models_2x2.json")
OUT = ROOT.parent / "paper_figures"


def load_net(path):
    net = TwoLayerQNetwork(input_dim=OBS_DIM, hidden_dim=128, output_dim=N_ACTIONS)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def rollout(env, net, gmax, start, goal, max_steps=200):
    """Greedy rollout; returns (trajectory list of (x,y,z), reached_goal)."""
    env.goal = goal
    env.pos = start
    env.steps = 0
    obs = env._get_obs()
    traj = [env.pos]
    for _ in range(max_steps):
        s = torch.as_tensor(normalize_obs(obs, gmax), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = int(torch.argmax(net(s), dim=1).item())
        obs, _, term, trunc, _ = env.step(a)
        traj.append(env.pos)
        if term or trunc:
            break
    return traj, (env.pos == goal)


def nearest_free(free, target):
    """Free cell closest (Euclidean in x,y) to a target (x,y)."""
    tx, ty = target
    return min(free, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)


def draw(ax, env, traj_upto, pos, goal, title):
    L, W, H = env.L, env.W, env.H
    z_scale = max(2, L // (H * 3))
    ax.clear()
    ax.voxels(env.occupancy, facecolors=env.facecolors, edgecolor="none", alpha=0.45)
    if len(traj_upto) > 1:
        xs = [p[0] + 0.5 for p in traj_upto]
        ys = [p[1] + 0.5 for p in traj_upto]
        zs = [p[2] + 0.5 for p in traj_upto]
        ax.plot(xs, ys, zs, color="#c9a0ff", linewidth=2.8, alpha=0.95, zorder=4)
    marker_base = max(60, 120_000 // (L * W))
    ax.scatter(goal[0] + 0.5, goal[1] + 0.5, goal[2] + 0.5, color="lime",
               s=marker_base * 4, marker="*", depthshade=False, zorder=10)
    ax.scatter(pos[0] + 0.5, pos[1] + 0.5, pos[2] + 0.5, color="red",
               s=marker_base * 1.1, marker="o", depthshade=False, zorder=10)
    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_box_aspect([L, W, H * z_scale])
    ax.set_xlabel("x", fontsize=8, labelpad=1)
    ax.set_ylabel("y", fontsize=8, labelpad=1)
    ax.set_zlabel("z", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=28, azim=225)
    ax.set_title(title, fontsize=10, pad=2)


def animate(env, trajs, goals, titles, suptitle, out_gif, fps=4, hold=6):
    fig = plt.figure(figsize=(12, 10))
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    writer = PillowWriter(fps=fps)
    T = max(len(t) for t in trajs)
    with writer.saving(fig, str(out_gif), dpi=90):
        for f in range(T + hold):
            for i, ax in enumerate(axes):
                k = min(f, len(trajs[i]) - 1)
                draw(ax, env, trajs[i][:k + 1], trajs[i][k], goals[i], titles[i])
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            writer.grab_frame()
    plt.close(fig)
    print(f"saved: {out_gif}  ({T} frames + {hold} hold)")


def main():
    OUT.mkdir(exist_ok=True)
    env = Grid3DEnv(random_goal=False, noise_prob=0.0)
    L, W, H = env.L, env.W, env.H
    gmax = np.array([L - 1, W - 1, H - 1], dtype=np.float32)
    free = [(x, y, 0) for x in range(L) for y in range(W) if not env._blocked(x, y, 0)]
    rng = np.random.default_rng(0)
    corners = [(6, 6), (44, 6), (6, 44), (44, 44)]

    # ---------- STATIC : fixed goal, 4 different starts ----------
    net_s = load_net(STATIC_CK)
    fixed_goal = env.goal
    s_trajs, s_goals, s_titles = [], [], []
    for c in corners:
        start = nearest_free([f for f in free if f != fixed_goal], c)
        traj, ok = rollout(env, net_s, gmax, start, fixed_goal)
        s_trajs.append(traj); s_goals.append(fixed_goal)
        s_titles.append(f"start ({start[0]},{start[1]}) -> goal "
                        f"({fixed_goal[0]},{fixed_goal[1]})  |  {len(traj)-1} steps"
                        f"{'' if ok else '  (FAILED)'}")
        print(f"static  {start} -> {fixed_goal}  ok={ok}  steps={len(traj)-1}")
    animate(env, s_trajs, s_goals, s_titles,
            "Best static model — fixed goal, 4 start positions",
            OUT / "gif_static_2x2.gif")

    # ---------- DYNAMIC : 4 easy goals (dmap==1), random start each ----------
    net_d = load_net(DYNAMIC_CK)
    dmap = np.array(json.load(open(CACHE))["dmap"])
    easy = [tuple(c) + (0,) for c in np.argwhere(dmap == 1.0)]  # (x,y,0)
    easy_set = set(easy)
    d_trajs, d_goals, d_titles = [], [], []
    for c in corners:
        goal = nearest_free(easy, c)                     # easy goal near this corner
        # random start that the agent actually reaches this goal from
        traj, ok = None, False
        for _ in range(40):
            start = free[int(rng.integers(len(free)))]
            if start == goal:
                continue
            traj, ok = rollout(env, net_d, gmax, start, goal)
            if ok:
                break
        d_trajs.append(traj); d_goals.append(goal)
        d_titles.append(f"goal ({goal[0]},{goal[1]})  start ({start[0]},{start[1]})"
                        f"  |  {len(traj)-1} steps{'' if ok else '  (FAILED)'}")
        print(f"dynamic start {start} -> goal {goal}  ok={ok}  steps={len(traj)-1}")
    animate(env, d_trajs, d_goals, d_titles,
            "Best dynamic model (HER) — 4 easy goals, random starts",
            OUT / "gif_dynamic_2x2.gif")


if __name__ == "__main__":
    main()

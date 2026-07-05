"""
Air-time investigation for a trained Grid3D DQN agent.

Question: how much of the agent's time at z>=1 is *necessary* (it is flying over
an obstacle and has no choice) versus *gratuitous* (the cell directly below at
z=0 is free, so it could have walked on the ground instead)?

We roll out greedy episodes and classify every step:
  - ground          : agent at z == 0
  - air_necessary   : agent at z >= 1 AND the z=0 cell below is BLOCKED
                      (it is genuinely crossing an obstacle)
  - air_gratuitous  : agent at z >= 1 AND the z=0 cell below is FREE
                      (it could have descended and walked -> avoidable air time)

"climb only when it saves" == drive air_gratuitous toward 0 while keeping
air_necessary (obstacle crossings) and the success rate intact.

Usage:
    python investigate_airtime.py --model runs/run_07_truncation_fix/online_q_network.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from grid3d_env import Grid3DEnv
from train_dqn import TwoLayerQNetwork, normalize_obs


def rollout_stats(model, env, grid_max, n_episodes, max_steps, device):
    n_ground = n_air_nec = n_air_grat = 0
    total_steps = 0
    successes = 0
    ep_lengths = []
    # how the agent leaves altitude: did it descend before the goal, or ride high?
    final_air_grat_runs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = normalize_obs(obs, grid_max)
        ep_len = 0
        ep_grat = 0
        for _ in range(max_steps):
            with torch.no_grad():
                q = model(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            action = int(torch.argmax(q, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            state = normalize_obs(obs, grid_max)

            x, y, z = env.pos
            if z == 0:
                n_ground += 1
            elif env.occupancy[x, y, 0]:      # something solid directly below -> must be airborne
                n_air_nec += 1
            else:                              # free ground below -> avoidable air time
                n_air_grat += 1
                ep_grat += 1
            total_steps += 1
            ep_len += 1
            if terminated or truncated:
                break

        if terminated:
            successes += 1
        ep_lengths.append(ep_len)
        final_air_grat_runs.append(ep_grat)

    return {
        "episodes": n_episodes,
        "success_rate": 100.0 * successes / n_episodes,
        "avg_steps": float(np.mean(ep_lengths)),
        "total_steps": total_steps,
        "pct_ground": 100.0 * n_ground / total_steps,
        "pct_air_necessary": 100.0 * n_air_nec / total_steps,
        "pct_air_gratuitous": 100.0 * n_air_grat / total_steps,
        "avg_gratuitous_steps_per_ep": float(np.mean(final_air_grat_runs)),
    }


def main():
    p = argparse.ArgumentParser(description="Quantify gratuitous vs necessary air time.")
    p.add_argument("--model", type=str, default="runs/run_07_truncation_fix/online_q_network.pt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--noise-prob", type=float, default=0.0,
                   help="0.0 isolates the learned policy; raise to see behaviour under env noise.")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--grid-l", type=int, default=50)
    p.add_argument("--grid-w", type=int, default=50)
    p.add_argument("--grid-h", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = Grid3DEnv(L=args.grid_l, W=args.grid_w, H=args.grid_h,
                    max_steps=args.max_steps, noise_prob=args.noise_prob)
    env.reset(seed=args.seed)
    grid_max = np.array([env.L - 1, env.W - 1, env.H - 1], dtype=np.float32)

    model = TwoLayerQNetwork(hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(Path(args.model), map_location=device))
    model.eval()

    s = rollout_stats(model, env, grid_max, args.episodes, args.max_steps, device)

    print(f"\nAir-time investigation  |  model={args.model}")
    print(f"noise_prob={args.noise_prob}  episodes={s['episodes']}")
    print("-" * 56)
    print(f"  success rate            : {s['success_rate']:.1f}%")
    print(f"  avg steps / episode     : {s['avg_steps']:.1f}")
    print(f"  total steps analysed    : {s['total_steps']}")
    print("-" * 56)
    print(f"  on ground   (z=0)       : {s['pct_ground']:.1f}%")
    print(f"  air NECESSARY (obstacle): {s['pct_air_necessary']:.1f}%")
    print(f"  air GRATUITOUS (free)   : {s['pct_air_gratuitous']:.1f}%   <-- avoidable")
    print(f"  avg gratuitous steps/ep : {s['avg_gratuitous_steps_per_ep']:.1f}")
    print("-" * 56)
    air = s['pct_air_necessary'] + s['pct_air_gratuitous']
    if air > 0:
        print(f"  of all air time, {100.0 * s['pct_air_gratuitous'] / air:.0f}% is avoidable")


if __name__ == "__main__":
    main()

# PARL_Project — 3D Grid-World RL Environment

A minimal, dependency-light **3D grid-world** environment built on the
[Gymnasium](https://gymnasium.farama.org/) API, plus a demo that renders an
episode to an animated GIF.

![Episode demo](episode.gif)

## Overview

`Grid3DEnv` is a single-file Gymnasium environment where an agent navigates a
`12 × 8 × 5` voxel grid toward a goal, avoiding static obstacles of varying
heights (crop rows, full blockers, and colored obstacles). Dynamics use NumPy
only; matplotlib is imported lazily and used solely for rendering.

### Action space
`Discrete(6)` — move along `±x`, `±y`, `±z`.

### Observation space
`Box(12,)` — agent position `(x, y, z)`, goal-relative offset `(gx-x, gy-y, gz-z)`,
and 6 binary "is-the-next-cell-blocked" flags (one per action).

### Reward
| Event | Reward |
|-------|--------|
| Reached goal | `+50` |
| Blocked / out-of-bounds (no move) | `-5` |
| Move up (`+z`) | `-3` |
| Any other move | `-1` |
| Avoidable air (horizontal move at `z ≥ 1` over a free ground cell) | `-air_cost` |

The **avoidable-air penalty** adds `-air_cost` (default `0.2`) to a horizontal move
(`±x`, `±y`) performed at altitude `z ≥ 1` while the `z = 0` cell directly below is
free — the agent could have walked on the ground. Ascending/descending and flight
over an actual obstacle are exempt. Set `air_cost=0` to disable it.

Episodes terminate on reaching the goal and truncate after `max_steps` (default 200).

## Files
- `grid3d_env.py` — the `Grid3DEnv` environment (also runnable as a random-policy smoke test).
- `render_demo.py` — plans a shortest path with BFS, rolls it out, and saves `episode.gif` + `episode_final.png`.
- `episode.gif`, `episode_final.png` — example rendered output.

## Usage

```bash
# Dependencies
pip install gymnasium numpy matplotlib

# Run a random-policy episode (prints steps + total reward)
python grid3d_env.py

# Render a BFS-planned episode to episode.gif / episode_final.png
python render_demo.py
```

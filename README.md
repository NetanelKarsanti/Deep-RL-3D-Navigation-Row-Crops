# Deep Reinforcement Learning for 3D Navigation in Row-Crop Environments

A reinforcement-learning study of navigation in a **stochastic 3D grid-world**
modelled on a row-crop field. An agent learns to reach a goal from a random
start while avoiding static obstacles (crop rows and walls) of varying heights,
under an energy objective that discourages unnecessary flight. The repository
contains a shared Gymnasium environment and two agents (vanilla DQN and Double
DQN) that train and are compared on it, plus a multi-seed experiment runner.

**Authors:** Netanel Karsanti, Nisim Kachlon, Saar David
**Course supervisor:** Dr. Ayal Taitler

> This top-level README collects the parts common to every sub-project. Each
> sub-folder keeps its own README with the details specific to that variant.

---

## Repository Structure

| Folder | Contents |
|--------|----------|
| [`3D_env/`](3D_env/) | Minimal, dependency-light `Grid3DEnv` (`12×8×5`, deterministic) + a BFS-planned GIF render demo. The lightweight reference environment. |
| [`MDP_DQN/`](MDP_DQN/) | The full stochastic `Grid3DEnv` (`50×50×4`) + a **vanilla DQN** agent. |
| [`Double_DQN/`](Double_DQN/) | The same stochastic environment + a **Double DQN** agent and post-training analysis tooling. |
| [`experiments/`](experiments/) | `run_seed_sweep.py` — trains **both** agents over multiple seeds and produces a cross-seed comparison. |

Each agent folder shares the same core files: `grid3d_env.py` (environment),
`Q3_DQN.py` (training), `visualize_agent.py` (greedy rollout → GIF/PNG),
`show_env.py` (top-down obstacle heatmap), and `analyze_results.py` (plots).

> **Note:** training artifacts (`runs/`, `experiments/results/`, `*.pt`, `*.png`,
> `*.csv`, `*.gif`, `*.log`) are generated locally and intentionally excluded via
> `.gitignore` — the repository keeps code and documentation only.

---

## Environment: `Grid3DEnv`

A Gymnasium-compatible voxel grid of dimensions `L × W × H`
(default `50 × 50 × 4` for the DQN agents). Cells are indexed by integer
coordinates `(x, y, z)`. The goal sits at a fixed proportional position
`g = (⌊5L/12⌋, ⌊3W/8⌋, 0)`; the agent starts each episode at `z = 0` from a
uniformly random free cell (excluding the goal).

### Action Space — `Discrete(6)`

| Index | Direction | `(dx, dy, dz)` |
|-------|-----------|----------------|
| 0 | +x | (1, 0, 0) |
| 1 | −x | (−1, 0, 0) |
| 2 | +y | (0, 1, 0) |
| 3 | −y | (0, −1, 0) |
| 4 | +z | (0, 0, 1) |
| 5 | −z | (0, 0, −1) |

If the target cell is blocked (obstacle or out-of-bounds), the agent stays put
and receives the collision penalty.

### Observation Space — `Box(12,)`

A 12-D float32 vector: agent position `(x, y, z)`, goal-relative offset
`(gₓ−x, g_y−y, g_z−z)`, and 6 binary "next-cell-blocked" flags (one per action).
Positions and offsets are normalized by `(L−1, W−1, H−1)` before being fed to
the network.

### Transition Function

The DQN environment is **stochastic**: with probability `p_noise` (default `0.1`)
the chosen action is replaced by a uniformly random one; otherwise the move is
deterministic and succeeds only if the target cell is free.

### Reward Function

| Event | Reward |
|-------|--------|
| Reached goal (`s' = g`) | `+50` |
| Blocked / no movement (`s' = s`) | `−5` |
| Upward move (`+z`) | `−3` |
| Any other move | `−1` |
| **Avoidable-air penalty** (added to the move reward) | `−air_cost` |

The negative step cost encourages shortest paths; the elevated cost for upward
movement biases the agent toward ground-level navigation. Episodes **terminate**
on reaching the goal and **truncate** after `max_steps`.

**Avoidable-air penalty.** On top of the move reward, a horizontal cruise
(`±x`, `±y`) performed at altitude `z ≥ 1` while the `z = 0` cell directly below
is free incurs an extra `−air_cost` (default `air_cost = 0.2`) — the agent could
have walked on the ground. Ascending/descending (`±z`) and flight over an actual
obstacle are **exempt**, so a genuine obstacle crossing (climb → cross → descend)
is never penalised. This reflects the energy objective that air travel costs more
than ground travel. The penalty is **on by default**; set `air_cost = 0` to
disable it.

### Obstacle Layout

Obstacle heights scale with `H`: `h(α) = max(1, ⌊αH/5⌋)`. The layout combines
**crop rows** (horizontal barriers across the x-axis), **passage-blocking walls**
flanking the goal corridor, **scattered** rectangular obstacles (1×1 to 3×3), and
fixed **fill obstacles** that increase navigational complexity.

---

## Agents

All agents share a two-layer MLP `Q(s; θ): ℝ¹² → ℝ⁶` with ReLU and hidden
dimension 128, trained with experience replay and a target network.

| | `MDP_DQN` | `Double_DQN` |
|--|-----------|--------------|
| Target | `y = R + γ·maxₐ' Q_target(s', a')` | `y = R + γ·Q_target(s', argmaxₐ' Q_online(s', a'))` |
| Effect | baseline | decouples action **selection** from **evaluation**, removing Q-value overestimation bias (van Hasselt et al., 2016) |

### Shared Training Hyperparameters

- Replay buffer capacity: **50,000** transitions
- Batch size: **64**
- Optimizer: **Adam**, learning rate `η = 10⁻³`
- Discount factor: `γ = 0.99`
- Target-network update: every **250** gradient steps
- Gradient clipping: `‖∇θ‖ ≤ 10`
- ε-greedy exploration, linear decay from `ε = 1.0` to `ε = 0.1`

### Potential-Based Reward Shaping

To accelerate learning, both agents add a policy-invariant potential-based
shaping term `F(s, s') = γ·Φ(s') − Φ(s)` to the **stored** transitions only.
With `Φ(s) = −‖pos(s) − g‖₂` (and an optional z-aware term `−β_z·z` rewarding
descent toward ground level). Evaluation always uses the true reward `R`
(Ng, Harada & Russell, 1999).

---

## Setup

```bash
pip install gymnasium numpy matplotlib torch
```

---

## Reproducing the Experiments — `experiments/run_seed_sweep.py`

`run_seed_sweep.py` is the main entry point for the results in this project. It
trains **both** agents (vanilla DQN and Double DQN) over the same set of random
seeds, evaluates each run with `analyze_results.py`, and writes a cross-seed
comparison. Because both agents use the **same** reward, the only variable is the
Bellman target — a fair head-to-head.

Run it from the repository root:

```bash
# Full default sweep: both agents × 5 seeds × 500k steps each
python experiments/run_seed_sweep.py

# Preview the exact commands without running anything
python experiments/run_seed_sweep.py --dry-run

# A quick smoke run: one agent, one seed, short training
python experiments/run_seed_sweep.py --algos dqn --seeds 42 --steps 50000
```

### Available Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--seeds S [S ...]` | `1 2 42 7 123` | Random seeds to run. One training run per (algo, seed). |
| `--steps N` | `500000` | Training steps per run. |
| `--algos {dqn,double_dqn} [...]` | both | Which agent(s) to train. `dqn` → `MDP_DQN/`, `double_dqn` → `Double_DQN/`. |
| `--reward-mode {simple,energy}` | `simple` | Reward variant passed to **both** agents. `simple` = discrete table + air penalty (baseline); `energy` = height-graded avoidable-air penalty. |
| `--air-cost FLOAT` | agent default (`0.2`) | Override the avoidable-air penalty weight. `0` disables it. |
| `--random-goal` | off | Randomize the goal each episode (applied to both training **and** evaluation). |
| `--her` | off | Hindsight Experience Replay during training. Intended for use with `--random-goal`. |
| `--heatmap` | off | Also render the (slow) per-cell success heatmap during analysis. |
| `--force` | off | Retrain even if a checkpoint already exists (otherwise finished runs are skipped). |
| `--skip-analyze` | off | Train only; do not run `analyze_results.py`. |
| `--dry-run` | off | Print the train/analyze commands without executing anything. |

### Outputs

All outputs land under `experiments/results/` (git-ignored):

| Path | Contents |
|------|----------|
| `results/<algo>/seed_<NNN>/` | `online_q_network.pt`, `evaluation.csv`, plots |
| `results/<algo>/seed_<NNN>.train.log` | captured training stdout (`tail -f` to follow) |
| `results/summary.csv` | one row per (algo, seed) with metrics |
| `results/aggregate.csv` | per-algo mean & std across seeds |
| `results/summary.md` | human-readable per-run + aggregate tables |
| `results/sweep.log` | full orchestrator output (for headless/server runs) |

The runner is designed for headless execution: every progress line and both
summary tables are mirrored to disk, so nothing needs to be watched live.
Finished runs are cached — re-invoking skips any (algo, seed) that already has a
checkpoint unless `--force` is given.

---

## Working With a Single Agent

Each agent folder is self-contained. From inside `MDP_DQN/` or `Double_DQN/`:

```bash
# Train a single run
python Q3_DQN.py --output runs/run_01_baseline

# Visualize the environment layout (top-down obstacle heatmap)
python show_env.py

# Greedy rollout with a trained model → GIF + PNG
python visualize_agent.py --model runs/run_01_baseline/online_q_network.pt

# Post-training analysis and plots
python analyze_results.py --results runs/run_01_baseline \
                          --model runs/run_01_baseline/online_q_network.pt
```

For the lightweight demo environment, see [`3D_env/`](3D_env/):

```bash
python 3D_env/grid3d_env.py    # random-policy smoke test
python 3D_env/render_demo.py   # BFS-planned episode → episode.gif
```

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- van Hasselt, Guez & Silver (2016). *Deep reinforcement learning with double Q-learning.* AAAI.
- Ng, Harada & Russell (1999). *Policy invariance under reward transformations.* ICML.
- Andrychowicz et al. (2017). *Hindsight Experience Replay.* NeurIPS.
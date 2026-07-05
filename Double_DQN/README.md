# 3D Grid-World Navigation — Double DQN Agent

A Gymnasium-compatible **stochastic 3D grid-world** environment paired with a
**Double Deep Q-Network (Double DQN)** agent. The agent learns to navigate from a random starting
position to a fixed goal while avoiding static obstacles of varying heights.

---

## Environment: `Grid3DEnv`

### Grid Structure

The world is a discrete voxel grid of dimensions `L × W × H` (default `50 × 50 × 4`).
Cells are indexed by integer coordinates `(x, y, z)` where
`x ∈ {0,…,L−1}`, `y ∈ {0,…,W−1}`, `z ∈ {0,…,H−1}`.

The goal is placed at a fixed proportional position:

$$g = \left(\left\lfloor\frac{5L}{12}\right\rfloor,\ \left\lfloor\frac{3W}{8}\right\rfloor,\ 0\right)$$

The agent always starts at `z = 0` from a uniformly random free cell
(excluding the goal cell) at the beginning of each episode.

### Action Space

`Discrete(6)` — one step along each axis direction:

| Index | Direction | `(dx, dy, dz)` |
|-------|-----------|----------------|
| 0 | +x | (1, 0, 0) |
| 1 | −x | (−1, 0, 0) |
| 2 | +y | (0, 1, 0) |
| 3 | −y | (0, −1, 0) |
| 4 | +z | (0, 0, 1) |
| 5 | −z | (0, 0, −1) |

If the target cell is blocked (obstacle or out-of-bounds), the agent remains
in its current position and receives the collision penalty.

### Observation Space

`Box(12,)` — a 12-dimensional float32 vector:

$$o = \bigl(x,\ y,\ z,\ g_x - x,\ g_y - y,\ g_z - z,\ b_0,\ b_1,\ b_2,\ b_3,\ b_4,\ b_5\bigr)$$

where `bᵢ ∈ {0, 1}` is a binary flag indicating whether the cell adjacent in
direction `i` is blocked. Before being fed to the network, positions and
goal offsets are normalized:

$$\hat{o}_{0:3} = \frac{o_{0:3}}{(L-1,\ W-1,\ H-1)}, \qquad \hat{o}_{3:6} = \frac{o_{3:6}}{(L-1,\ W-1,\ H-1)}$$

yielding `ô ∈ [0,1]³ × [−1,1]³ × {0,1}⁶`.

### Transition Function

The environment is **stochastic**. Given a chosen action `a`, the executed
action `ã` is:

$$\tilde{a} = \begin{cases} a & \text{with probability } 1 - p_{\text{noise}} \\ \text{Uniform}\{0,\dots,5\} & \text{with probability } p_{\text{noise}} \end{cases}$$

with default `p_noise = 0.1`. The resulting state transition is then
deterministic: the agent moves to the target cell if and only if it is free.

### Reward Function

The immediate reward `R(s, a)` is defined as:

$$R(s, a) = \begin{cases} +50 & \text{if } s' = g \quad \text{(goal reached)} \\ -5 & \text{if } s' = s \quad \text{(blocked — no movement)} \\ -3 & \text{if } \tilde{a} = 4 \quad \text{(upward move, } +z\text{)} \\ -1 & \text{otherwise} \end{cases}$$

### Obstacle Layout

All obstacle heights scale proportionally with `H` via:

$$h(\alpha) = \max\!\left(1,\ \left\lfloor\frac{\alpha \cdot H}{5}\right\rfloor\right), \quad \alpha \in \{1, 2, 3, 4\}$$

With `H = 4`: `h(1) = 1`, `h(2) = 1`, `h(3) = 2`, `h(4) = 3`.

The layout consists of four obstacle categories:

1. **Crop rows** — horizontal barriers spanning ~90% of the x-axis at evenly
   spaced y-positions. Height `h(1)`.

2. **Passage-blocking walls** — two walls flanking the goal's y-corridor,
   forcing the agent to either climb or navigate around. Height `h(3)`.

3. **Scattered obstacles** — rectangular footprint obstacles (1×1 to 3×3)
   at proportional positions.

4. **Fill obstacles** — 4 fixed 2×2 obstacles of height `h(3) = 2`, placed in
   sparse inter-row bands.

---

## Agent: Double Deep Q-Network (Double DQN)

### Motivation

Standard DQN uses the same network to both **select** and **evaluate** the next action:

$$y = R + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')$$

This causes **Q-value overestimation** — the target network tends to pick the
highest (often noisy) Q-value, accumulating positive bias over time.

### Double DQN Fix

Double DQN (van Hasselt et al., 2016) decouples selection from evaluation:

$$y = R + \gamma \cdot Q_{\text{target}}\!\left(s',\ \arg\max_{a'} Q_{\text{online}}(s', a')\right)$$

The **online network** selects the best action; the **target network** evaluates it.
This eliminates the maximization bias while keeping the target network stable.

### Network Architecture

A two-layer MLP with ReLU activation:

$$Q(s; \theta) : \mathbb{R}^{12} \to \mathbb{R}^6$$

$$Q(s;\theta) = W_2 \cdot \text{ReLU}(W_1 \hat{o} + b_1) + b_2$$

with hidden dimension 128 (default).

### Training Algorithm

Double DQN with experience replay and a target network:

- **Replay buffer** capacity: 50,000 transitions
- **Batch size**: 64
- **Optimizer**: Adam, learning rate `η = 10⁻³`
- **Discount factor**: `γ = 0.99`
- **Target network** update frequency: every 250 gradient steps
- **Gradient clipping**: `‖∇θ‖ ≤ 10`

**ε-greedy exploration** with linear decay:

$$\varepsilon(t) = \varepsilon_{\text{start}} - \frac{t}{T_{\varepsilon}} \cdot (\varepsilon_{\text{start}} - \varepsilon_{\text{final}}), \quad \varepsilon_{\text{start}} = 1.0,\ \varepsilon_{\text{final}} = 0.1$$

### Potential-Based Reward Shaping

To accelerate learning, a **z-aware potential-based shaping** term is added to the
stored transitions:

$$\Phi(s) = -\|\text{pos}(s) - g\|_2 - \beta_z \cdot z, \quad \beta_z = 1.0$$

$$F(s, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

The z-component rewards descent toward ground level at every step.
This shaping is **policy-invariant** (Ng et al., 1999).

---

## Files

| File | Description |
|------|-------------|
| `grid3d_env.py` | `Grid3DEnv` Gymnasium environment |
| `train_dqn.py` | Double DQN training script |
| `analyze_results.py` | Post-training analysis and plots |
| `visualize_agent.py` | Greedy rollout with trained model — saves GIF and PNG |
| `show_env.py` | Top-down obstacle heatmap of the environment |

---

## Usage

```bash
# Train with default parameters
python train_dqn.py --output runs/run_01_baseline

# Train with custom parameters
python train_dqn.py --steps 500000 --lr 1e-3 --epsilon-decay 350000 \
                 --output runs/run_01_baseline

# Analyze results
python analyze_results.py --results runs/run_01_baseline \
                          --model runs/run_01_baseline/online_q_network.pt
```

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- van Hasselt, Guez & Silver (2016). *Deep reinforcement learning with double Q-learning.* AAAI.
- Ng, Harada & Russell (1999). *Policy invariance under reward transformations.* ICML.

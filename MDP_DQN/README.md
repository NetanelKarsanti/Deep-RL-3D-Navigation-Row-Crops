# 3D Grid-World Navigation — DQN Agent

A Gymnasium-compatible **stochastic 3D grid-world** environment paired with a
Deep Q-Network (DQN) agent. The agent learns to navigate from a random starting
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

The negative step cost encourages shortest-path solutions; the elevated cost
for upward movement biases the agent toward ground-level navigation.

### Obstacle Layout

All obstacle heights scale proportionally with `H` via:

$$h(\alpha) = \max\!\left(1,\ \left\lfloor\frac{\alpha \cdot H}{5}\right\rfloor\right), \quad \alpha \in \{1, 2, 3, 4\}$$

With `H = 4`: `h(1) = 1`, `h(2) = 1`, `h(3) = 2`, `h(4) = 3`.

The layout consists of four obstacle categories:

1. **Crop rows** — horizontal barriers spanning ~90% of the x-axis at evenly
   spaced y-positions. Count scales with grid width: `n_rows = max(2, ⌊W/8⌋)`.
   Height `h(1)`.

2. **Passage-blocking walls** — two walls flanking the goal's y-corridor,
   forcing the agent to either climb or navigate around:
   - *Eastern wall*: spans the full y-gap between the two crop rows enclosing
     `g_y`. Height `h(3)`.
   - *Western wall*: wide in x, one cell in y at `g_y`. Height `h(3)`.

3. **Scattered obstacles** — a set of rectangular footprint obstacles
   (1×1 to 3×3) at proportional positions. Count scales with `L`.

4. **Fill obstacles** — 4 fixed 2×2 obstacles of height `h(3) = 2`, placed in
   sparse inter-row bands to increase navigational complexity:

   | Position | x | y (W=50) |
   |----------|---|----------|
   | Right / low band | `7L/10` | `W/5` |
   | Left / mid band | `L/8` | `W/2` |
   | Right / upper-mid band | `4L/5` | `5W/8` |
   | Left / upper band | `L/10` | `3W/4` |

Episodes **terminate** upon reaching the goal and **truncate** after
`max_steps` steps (default 200).

---

## Agent: Deep Q-Network (DQN)

### Network Architecture

A two-layer MLP with ReLU activation:

$$Q(s; \theta) : \mathbb{R}^{12} \to \mathbb{R}^6$$

$$Q(s;\theta) = W_2 \cdot \text{ReLU}(W_1 \hat{o} + b_1) + b_2$$

with hidden dimension 128 (default).

### Training Algorithm

Standard DQN with experience replay and a target network:

- **Replay buffer** capacity: 50,000 transitions
- **Batch size**: 64
- **Optimizer**: Adam, learning rate `η = 10⁻³`
- **Discount factor**: `γ = 0.99`
- **Target network** update frequency: every 250 gradient steps
- **Gradient clipping**: `‖∇θ‖ ≤ 10`

**ε-greedy exploration** with linear decay:

$$\varepsilon(t) = \varepsilon_{\text{start}} - \frac{t}{T_{\varepsilon}} \cdot (\varepsilon_{\text{start}} - \varepsilon_{\text{final}}), \quad \varepsilon_{\text{start}} = 1.0,\ \varepsilon_{\text{final}} = 0.1$$

### Potential-Based Reward Shaping

To accelerate learning, a **potential-based shaping** term is added to the
stored transitions. Using the potential `Φ(s) = −‖\text{pos}(s) − g‖₂`:

$$F(s, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

The shaped reward stored in the replay buffer is:

$$\tilde{R}(s, a, s') = R(s, a) + F(s, s') = R(s, a) + \gamma \cdot (-\|s'-g\|_2) - (-\|s-g\|_2)$$

This shaping is **policy-invariant**: the optimal policy under `R̃` coincides
with the optimal policy under `R` (Ng et al., 1999). Evaluation uses the
true reward `R` exclusively.

---

## Files

| File | Description |
|------|-------------|
| `grid3d_env.py` | `Grid3DEnv` Gymnasium environment |
| `Q3_DQN.py` | DQN training script |
| `visualize_agent.py` | Greedy rollout with trained model — saves GIF and PNG |
| `show_env.py` | Top-down obstacle heatmap of the environment |

---

## Usage

```bash
pip install gymnasium numpy matplotlib torch

# Train on default 50×50×4 grid
python Q3_DQN.py

# Train with custom parameters
python Q3_DQN.py --grid-l 50 --grid-w 50 --grid-h 4 \
                 --steps 500000 --max-episode-steps 500 \
                 --epsilon-decay 350000 --noise-prob 0.1

# Visualize the environment layout
python show_env.py

# Run a greedy episode with a trained model
python visualize_agent.py --model dqn_results/online_q_network.pt

# Train on smaller grid for fast experimentation
python Q3_DQN.py --grid-l 12 --grid-w 8 --grid-h 4 --steps 200000
```

---

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Ng, Harada & Russell (1999). *Policy invariance under reward transformations.* ICML.

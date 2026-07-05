"""Minimal 3D grid-world Gymnasium environment (single file).

NumPy for the dynamics; matplotlib is used only for render_mode="human" and is
imported lazily so the env stays dependency-light when not rendering.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Default grid dimensions (kept for backward-compat imports in render_demo.py).
L, W, H = 50, 50, 4

# Action -> (dx, dy, dz): +x, -x, +y, -y, +z, -z.
ACTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
HORIZONTAL_ACTIONS = (0, 1, 2, 3)  # ±x, ±y indices into ACTIONS (4=+z up, 5=-z down)


class Grid3DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, L=50, W=50, H=4, max_steps=200, render_mode=None, noise_prob=0.0,
                 air_cost=0.2, reward_mode="simple", random_goal=False):
        super().__init__()
        assert render_mode in (None, *self.metadata["render_modes"])
        assert reward_mode in ("simple", "energy")
        self.render_mode = render_mode
        self._fig = None
        self._ax  = None

        self.L = L
        self.W = W
        self.H = H
        self.max_steps = max_steps
        self.noise_prob = noise_prob
        self.air_cost = air_cost
        self.reward_mode = reward_mode
        self.random_goal = random_goal

        # Goal: same proportional position as the original (5/12, 3/8, 0).
        self.goal = (5 * L // 12, 3 * W // 8, 0)

        # Obstacle heights scale with H so difficulty stays proportional.
        def h(orig): return max(1, orig * H // 5)

        self.occupancy  = np.zeros((L, W, H), dtype=bool)
        self.facecolors = np.empty((L, W, H), dtype=object)

        goal_x, goal_y = self.goal[0], self.goal[1]

        def add(x, y, height, color):
            # never overwrite the goal cell
            if (x, y) == (goal_x, goal_y):
                return
            if 0 <= x < L and 0 <= y < W:
                self.occupancy[x, y, 0:height] = True
                self.facecolors[x, y, 0:height] = color

        def add_rect(x0, y0, wx, wy, height, color):
            """Fill a wx × wy footprint of obstacle cells."""
            for dx in range(wx):
                for dy in range(wy):
                    add(x0 + dx, y0 + dy, height, color)

        # --- Crop rows ---
        n_crop_rows  = max(2, W // 8)
        crop_x_start = max(1, L // 20)
        crop_x_end   = min(L - 1, 19 * L // 20)
        for i in range(1, n_crop_rows + 1):
            y = i * W // (n_crop_rows + 1)
            if y == goal_y:
                y += 1
            for x in range(crop_x_start, crop_x_end):
                add(x, y, h(1), "#90ee90")

        # --- Passage-blocking walls ---
        # Find the two crop rows that flank goal_y, then span a wall across the full gap.
        passage_lo, passage_hi = 0, W
        for i in range(1, n_crop_rows + 1):
            row_y = i * W // (n_crop_rows + 1)
            if row_y == goal_y:
                row_y += 1
            if row_y < goal_y:
                passage_lo = row_y + 1
            elif row_y > goal_y and passage_hi == W:
                passage_hi = row_y
        passage_span = passage_hi - passage_lo   # full y-height of the gap

        # Eastern wall — spans the full passage height in y (forces climb or x-detour)
        wall_wx = max(2, L // 20)
        east_x  = goal_x + max(3, L // 10)
        add_rect(east_x, passage_lo, wall_wx, passage_span, h(3), "#7f7f7f")

        # Western wall — wide in x, 1 cell in y (original shape)
        wall2_w = max(3, L // 10)
        wall2_x = max(0, goal_x - wall2_w - max(2, L // 15))
        add_rect(wall2_x, goal_y, wall2_w, 1, h(3), "#7f7f7f")

        # --- Scattered obstacles (some wide, some tall) ---
        # Each entry: (x, y, footprint_wx, footprint_wy, h_orig, color)
        base_obstacles = [
            (8 * L // 12, 1 * W // 8, 2, 1, 2, "#ff7f0e"),   # 2×1
            (2 * L // 12, 3 * W // 8, 1, 2, 3, "#1f77b4"),   # 1×2
            (7 * L // 12, 6 * W // 8, 2, 2, 4, "#9467bd"),   # 2×2
            (4 * L // 12, 1 * W // 8, 3, 1, 1, "#8c564b"),   # 3×1
        ]
        extra_obstacles = [
            (3 * L // 12, 5 * W // 8, 2, 1, 2, "#e377c2"),
            (9 * L // 12, 4 * W // 8, 1, 2, 3, "#17becf"),
            (5 * L // 12, 7 * W // 8, 2, 2, 2, "#bcbd22"),
            (1 * L // 4,  1 * W // 4, 3, 1, 2, "#aec7e8"),
            (3 * L // 4,  3 * W // 4, 1, 3, 3, "#ffbb78"),
            (1 * L // 3,  2 * W // 3, 2, 2, 2, "#98df8a"),
            (2 * L // 3,  1 * W // 3, 3, 1, 3, "#c5b0d5"),
            (1 * L // 6,  1 * W // 2, 1, 2, 4, "#c49c94"),
            (5 * L // 6,  1 * W // 6, 2, 1, 2, "#f7b6d2"),
            (5 * L // 8,  5 * W // 8, 2, 2, 3, "#dbdb8d"),
        ]
        n_extra = max(0, L // 6 - 2)
        for ox, oy, wx, wy, oh, oc in base_obstacles + extra_obstacles[:n_extra]:
            add_rect(ox, oy, wx, wy, h(oh), oc)

        # 4 fixed fill obstacles (height 2), placed in sparse bands between crop rows
        fill_obstacles = [
            (7 * L // 10, 1 * W // 5,  2, 2, 3, "#4daf4a"),  # right side, low band
            (1 * L // 8,  1 * W // 2,  2, 2, 3, "#984ea3"),  # left side, mid band
            (4 * L // 5,  5 * W // 8,  2, 2, 3, "#ff7f00"),  # right side, upper-mid band
            (1 * L // 10, 3 * W // 4,  2, 2, 3, "#a65628"),  # left side, upper band
        ]
        for ox, oy, wx, wy, oh, oc in fill_obstacles:
            add_rect(ox, oy, wx, wy, h(oh), oc)

        self.action_space = spaces.Discrete(6)
        low  = np.array([0, 0, 0, -(L-1), -(W-1), -(H-1), 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([L-1, W-1, H-1,  L-1,  W-1,  H-1,  1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.pos   = (0, 0, 0)
        self.steps = 0

    # ------------------------------------------------------------------
    def _blocked(self, x, y, z):
        if not (0 <= x < self.L and 0 <= y < self.W and 0 <= z < self.H):
            return True
        return bool(self.occupancy[x, y, z])

    def _get_obs(self):
        x, y, z   = self.pos
        gx, gy, gz = self.goal
        flags = [float(self._blocked(x + dx, y + dy, z + dz)) for dx, dy, dz in ACTIONS]
        return np.array([x, y, z, gx - x, gy - y, gz - z, *flags], dtype=np.float32)

    def _compute_reward(self, action, reached, moved):
        if reached:   return  50.0
        if not moved: return  -5.0
        base = -3.0 if action == 4 else -1.0
        # Avoidable-air penalty: horizontal cruise (±x, ±y) at altitude while the
        # z=0 cell directly below is free, i.e. the agent could have walked on the
        # ground. Ascend/descend and flight over an obstacle are exempt, so a real
        # obstacle crossing (climb -> cross -> descend) is never penalised.
        # "simple": flat penalty (air_cost). "energy": graded by height — wasteful
        # altitude costs more the higher you fly (air_cost * z/(H-1)).
        if self.air_cost > 0.0 and action in HORIZONTAL_ACTIONS:
            x, y, z = self.pos
            if z >= 1 and not self.occupancy[x, y, 0]:
                if self.reward_mode == "energy":
                    z_norm = z / (self.H - 1) if self.H > 1 else 0.0
                    base -= self.air_cost * z_norm
                else:
                    base -= self.air_cost
        return base

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All free z=0 cells (pure list comp — does not consume the RNG).
        free_all = [(x, y, 0) for x in range(self.L) for y in range(self.W)
                    if not self._blocked(x, y, 0)]
        if self.random_goal:
            # Randomize the goal first, then sample a start != goal.
            self.goal = free_all[self.np_random.integers(len(free_all))]
        # When random_goal=False this `free_start` equals the legacy `free` list
        # (same cells/order) and a single draw reproduces the original start exactly.
        free_start = [c for c in free_all if c != self.goal]
        self.pos   = free_start[self.np_random.integers(len(free_start))]
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        if self.noise_prob > 0.0 and self.np_random.random() < self.noise_prob:
            action = int(self.np_random.integers(len(ACTIONS)))
        dx, dy, dz = ACTIONS[action]
        x, y, z    = self.pos
        target     = (x + dx, y + dy, z + dz)
        moved      = not self._blocked(*target)
        if moved:
            self.pos = target
        reached    = self.pos == self.goal
        reward     = self._compute_reward(action, reached, moved)
        self.steps += 1
        terminated = reached
        truncated  = self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _draw(self, ax):
        ax.clear()
        ax.voxels(self.occupancy, facecolors=self.facecolors, edgecolor="k", alpha=0.6)
        ax.scatter(*[c + 0.5 for c in self.pos],  color="red",   s=220, marker="o",
                   depthshade=False, label="agent")
        ax.scatter(*[c + 0.5 for c in self.goal], color="green", s=320, marker="*",
                   depthshade=False, label="goal")
        ax.set_xlim(0, self.L); ax.set_ylim(0, self.W); ax.set_zlim(0, self.H)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"Grid3DEnv {self.L}×{self.W}×{self.H}  —  step {self.steps}")
        ax.legend(loc="upper right")

    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt
        if self._fig is None:
            plt.ion()
            self._fig = plt.figure(figsize=(8, 6))
            self._ax  = self._fig.add_subplot(111, projection="3d")
        self._draw(self._ax)
        self._fig.canvas.draw_idle()
        plt.pause(1.0 / self.metadata["render_fps"])

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax  = None


if __name__ == "__main__":
    env = Grid3DEnv()
    obs, info = env.reset(seed=0)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
    print(f"episode finished in {env.steps} steps, total reward = {total}")

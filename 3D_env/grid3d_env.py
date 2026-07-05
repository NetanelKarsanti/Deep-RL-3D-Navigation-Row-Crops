"""Minimal 3D grid-world Gymnasium environment (single file).

NumPy for the dynamics; matplotlib is used only for render_mode="human" and is
imported lazily so the env stays dependency-light when not rendering.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Grid dimensions: x along rows (L), y across rows (W), z height (H).
L, W, H = 12, 8, 5

# Action -> (dx, dy, dz): +x, -x, +y, -y, +z, -z.
ACTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

HORIZONTAL_ACTIONS = (0, 1, 2, 3)  # ±x, ±y indices into ACTIONS (4=+z up, 5=-z down)


class Grid3DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_steps=200, render_mode=None, air_cost=0.2):
        super().__init__()
        assert render_mode in (None, *self.metadata["render_modes"])
        self.render_mode = render_mode
        self._fig = None        # lazily created matplotlib figure/axes
        self._ax = None
        self.max_steps = max_steps
        self.air_cost = air_cost
        self.goal = (5, 3, 0)

        # Build the static occupancy grid once. A height-h object at (x, y)
        # blocks z = 0..h-1.
        self.occupancy = np.zeros((L, W, H), dtype=bool)
        # Per-cell color for rendering (object array of color strings).
        self.facecolors = np.empty((L, W, H), dtype=object)

        def add(x, y, h, color):
            self.occupancy[x, y, 0:h] = True
            self.facecolors[x, y, 0:h] = color

        # Crop rows (h=1): y in {2, 5}, 2 <= x <= 9  -> light green.
        for y in (2, 5):
            for x in range(2, 10):
                add(x, y, 1, "#90ee90")
        # Full blocker (h=3) -> gray.
        add(6, 3, 3, "#7f7f7f")
        add(6, 4, 3, "#7f7f7f")
        # Obstacles with varying heights -> each a distinct color.
        add(8, 1, 2, "#ff7f0e")   # orange
        add(2, 3, 3, "#1f77b4")   # blue
        add(7, 6, 4, "#9467bd")   # purple
        add(4, 1, 1, "#8c564b")   # brown

        self.action_space = spaces.Discrete(6)
        low = np.array([0, 0, 0, -(L - 1), -(W - 1), -(H - 1), 0, 0, 0, 0, 0, 0],
                       dtype=np.float32)
        high = np.array([L - 1, W - 1, H - 1, L - 1, W - 1, H - 1, 1, 1, 1, 1, 1, 1],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.pos = (0, 0, 0)
        self.steps = 0

    def _blocked(self, x, y, z):
        """True if (x, y, z) is out-of-bounds or occupied. Single source of truth."""
        if not (0 <= x < L and 0 <= y < W and 0 <= z < H):
            return True
        return bool(self.occupancy[x, y, z])

    def _get_obs(self):
        x, y, z = self.pos
        gx, gy, gz = self.goal
        flags = [float(self._blocked(x + dx, y + dy, z + dz))
                 for dx, dy, dz in ACTIONS]
        return np.array([x, y, z, gx - x, gy - y, gz - z, *flags], dtype=np.float32)

    def _compute_reward(self, action, reached, moved):
        if reached:
            return 50.0
        if not moved:           # blocked / out-of-bounds -> stayed in place
            return -5.0
        base = -3.0 if action == 4 else -1.0   # up (+z) costs more than other moves
        # Avoidable-air penalty: horizontal cruise (±x, ±y) at altitude while the
        # z=0 cell directly below is free, i.e. the agent could have walked on the
        # ground. Ascend/descend and flight over an obstacle are exempt, so a real
        # obstacle crossing (climb -> cross -> descend) is never penalised.
        if self.air_cost > 0.0 and action in HORIZONTAL_ACTIONS:
            x, y, z = self.pos
            if z >= 1 and not self.occupancy[x, y, 0]:
                base -= self.air_cost
        return base

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Uniformly sample a free z=0 start cell that is not the goal.
        free = [(x, y, 0) for x in range(L) for y in range(W)
                if not self._blocked(x, y, 0) and (x, y, 0) != self.goal]
        self.pos = free[self.np_random.integers(len(free))]
        self.steps = 0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        dx, dy, dz = ACTIONS[action]
        x, y, z = self.pos
        target = (x + dx, y + dy, z + dz)
        moved = not self._blocked(*target)
        if moved: ## we wante to add a noise here ? 
            self.pos = target
        reached = self.pos == self.goal
        reward = self._compute_reward(action, reached, moved)
        self.steps += 1
        terminated = reached
        truncated = self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {}

    def _draw(self, ax):
        """Draw the current scene onto a 3D axes (shared by render & tests)."""
        ax.clear()
        # Objects as voxels; each cell uses its per-object color (crop rows light
        # green, each obstacle a distinct color). Cell (x,y,z) fills box [x,x+1]x...
        ax.voxels(self.occupancy, facecolors=self.facecolors, edgecolor="k",
                  alpha=0.6)
        ax.scatter(*[c + 0.5 for c in self.pos], color="red", s=220, marker="o",
                   depthshade=False, label="agent")
        ax.scatter(*[c + 0.5 for c in self.goal], color="green", s=320, marker="*",
                   depthshade=False, label="goal")
        ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"Grid3DEnv  -  step {self.steps}")
        ax.legend(loc="upper right")

    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt
        if self._fig is None:
            plt.ion()
            self._fig = plt.figure(figsize=(8, 6))
            self._ax = self._fig.add_subplot(111, projection="3d")
        self._draw(self._ax)
        self._fig.canvas.draw_idle()
        plt.pause(1.0 / self.metadata["render_fps"])

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


if __name__ == "__main__":
    env = Grid3DEnv()
    obs, info = env.reset(seed=0)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
    print(f"episode finished in {env.steps} steps, total reward = {total}")


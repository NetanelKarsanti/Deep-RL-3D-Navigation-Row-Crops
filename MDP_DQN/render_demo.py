"""Render a Grid3DEnv episode to an animated GIF you can open and view.

Plans a shortest path to the goal with BFS (so the rollout actually reaches it),
steps the env along that path, and saves each frame to episode.gif.
"""

from collections import deque

import matplotlib
matplotlib.use("Agg")                       # headless: render to file, no window
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from grid3d_env import Grid3DEnv, ACTIONS, L, W, H

DELTA_TO_ACTION = {d: a for a, d in enumerate(ACTIONS)}


def bfs_path(env, start, goal):
    """Shortest 6-connected path of free cells from start to goal (list of cells)."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y, z = cur
        for dx, dy, dz in ACTIONS:
            nxt = (x + dx, y + dy, z + dz)
            if nxt not in prev and not env._blocked(*nxt):
                prev[nxt] = cur
                q.append(nxt)
    if goal not in prev:
        return [start]
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def main():
    env = Grid3DEnv()
    env.reset(seed=0)
    # Pick the free z=0 cell farthest from the goal for a nice long path (a fixed
    # coordinate would risk landing inside an obstacle as the grid size changes).
    gx, gy, _ = env.goal
    start = max(
        ((x, y, 0) for x in range(env.L) for y in range(env.W)
         if not env._blocked(x, y, 0) and (x, y, 0) != env.goal),
        key=lambda c: (c[0] - gx) ** 2 + (c[1] - gy) ** 2,
    )
    env.pos = start
    path = bfs_path(env, start, env.goal)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    writer = PillowWriter(fps=3)
    total = 0.0
    with writer.saving(fig, "episode.gif", dpi=80):
        env._draw(ax)                        # initial frame
        writer.grab_frame()
        for cur, nxt in zip(path, path[1:]):
            delta = tuple(b - a for a, b in zip(cur, nxt))
            obs, r, term, trunc, info = env.step(DELTA_TO_ACTION[delta])
            total += r
            env._draw(ax)
            writer.grab_frame()
        for _ in range(4):                   # linger on the goal
            env._draw(ax)
            writer.grab_frame()

    fig.savefig("episode_final.png", dpi=90)  # also a single still image
    plt.close(fig)
    print(f"reached goal in {len(path) - 1} steps, total reward = {total}")
    print("saved: episode.gif  and  episode_final.png")


if __name__ == "__main__":
    main()

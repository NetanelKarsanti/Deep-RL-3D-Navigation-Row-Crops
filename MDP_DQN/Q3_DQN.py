"""
DQN implementation for Grid3DEnv — 3D grid navigation task.

The agent learns to navigate a 50x50x4 voxel grid (default) from a random start
at z=0 to the fixed goal at (5L/12, 3W/8, 0), avoiding static obstacles.

Observation: 12-dim float32 vector (agent pos, goal offset, 6 collision flags)
Actions:     6 discrete (±x, ±y, ±z movement)
Rewards:     +50 goal, -5 collision, -3 move up, -1 other move, -air_cost avoidable air

Default run:
    python Q3_DQN.py
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from grid3d_env import Grid3DEnv


# -----------------------------------------------------------------------------
# Domain constants
# -----------------------------------------------------------------------------
N_ACTIONS = 6
OBS_DIM = 12

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


def normalize_obs(obs: np.ndarray, grid_max: np.ndarray) -> np.ndarray:
    """Scale positions and goal offsets to [-1, 1]; collision flags stay as-is."""
    norm = obs.copy()
    norm[0:3] /= grid_max  # absolute position → [0, 1]
    norm[3:6] /= grid_max  # goal offset → [-1, 1]
    # norm[6:12] are binary collision flags, already in {0, 1}
    return norm


def goal_distance(obs: np.ndarray) -> float:
    """Euclidean distance to goal, read from the raw (unnormalized) goal-offset part of obs."""
    return float(np.linalg.norm(obs[3:6]))


def her_relabel_episode(episode, agent, her_env, grid_max, settings, her_rng) -> None:
    """Hindsight Experience Replay (future strategy).

    `episode` is a list of (raw_obs, action, raw_next_obs) for one finished episode.
    For each transition we add `her_k` extra transitions re-aimed at goals that were
    ACTUALLY achieved later in the same episode (positions visited at index >= i,
    including the immediate next state). Re-aiming only changes the goal: the
    collision flags (obs[6:12]) depend on position alone, and the reward is recomputed
    via her_env._compute_reward (goal enters only through `reached`). This manufactures
    dense success signal for a sparse goal-conditioned task.
    """
    T = len(episode)
    for i in range(T):
        obs_i, action_i, next_obs_i = episode[i]
        pos_i      = (int(obs_i[0]),      int(obs_i[1]),      int(obs_i[2]))
        next_pos_i = (int(next_obs_i[0]), int(next_obs_i[1]), int(next_obs_i[2]))
        moved_i = next_pos_i != pos_i
        flags_i      = obs_i[6:12]
        next_flags_i = next_obs_i[6:12]
        for j in her_rng.integers(i, T, size=min(settings.her_k, T - i)):
            g = (int(episode[j][2][0]), int(episode[j][2][1]), int(episode[j][2][2]))  # achieved next_pos
            her_env.pos = next_pos_i
            reached = next_pos_i == g
            r = her_env._compute_reward(action_i, reached, moved_i)
            obs_r = np.array([pos_i[0], pos_i[1], pos_i[2],
                              g[0] - pos_i[0], g[1] - pos_i[1], g[2] - pos_i[2], *flags_i],
                             dtype=np.float32)
            next_obs_r = np.array([next_pos_i[0], next_pos_i[1], next_pos_i[2],
                                   g[0] - next_pos_i[0], g[1] - next_pos_i[1], g[2] - next_pos_i[2],
                                   *next_flags_i], dtype=np.float32)
            # Match the main loop's shaping (applied for the discrete rewards, not "dist").
            if settings.shaping and settings.reward_mode in ("simple", "energy"):
                phi_b = -float(np.linalg.norm(obs_r[3:6]))      - settings.beta_z * float(obs_r[2])
                phi_a = -float(np.linalg.norm(next_obs_r[3:6])) - settings.beta_z * float(next_obs_r[2])
                shaped = r + settings.gamma * phi_a - phi_b
            else:
                shaped = r
            agent.save_transition(normalize_obs(obs_r, grid_max), action_i, shaped,
                                  normalize_obs(next_obs_r, grid_max), reached)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class DQNSettings:
    total_steps: int = 500_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 500
    learning_frequency: int = 4
    target_update_frequency: int = 250
    hidden_dim: int = 128

    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    epsilon_decay_steps: int = 350_000

    eval_frequency: int = 500
    eval_episodes: int = 10
    max_episode_steps: int = 500

    beta_z: float = 1.0
    air_cost: float = 0.2
    reward_mode: str = "simple"
    random_goal: bool = False
    her: bool = False
    her_k: int = 4
    shaping: bool = True   # potential-based reward shaping (Φ = -dist - β_z·z)

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    noise_prob: float = 0.1

    grid_l: int = 50
    grid_w: int = 50
    grid_h: int = 4


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------
class ExperienceReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.storage: Deque[Experience] = deque(maxlen=capacity)
        self.random_generator = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.storage.append(
            Experience(
                state.astype(np.float32),
                int(action),
                float(reward),
                next_state.astype(np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self.random_generator.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )


# -----------------------------------------------------------------------------
# Q-network: 2-layer MLP
# -----------------------------------------------------------------------------
class TwoLayerQNetwork(nn.Module):
    def __init__(self, input_dim: int = OBS_DIM, hidden_dim: int = 128, output_dim: int = N_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(state_tensor)


# -----------------------------------------------------------------------------
# DQN agent
# -----------------------------------------------------------------------------
class Grid3DDQNAgent:
    def __init__(self, settings: DQNSettings):
        self.settings = settings
        self.device = torch.device(settings.device)

        self.online_network = TwoLayerQNetwork(hidden_dim=settings.hidden_dim).to(self.device)
        self.target_network = TwoLayerQNetwork(hidden_dim=settings.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=settings.learning_rate)
        self.loss_function = nn.MSELoss()
        self.replay_buffer = ExperienceReplayBuffer(settings.replay_capacity, settings.seed)

        self.rng = np.random.default_rng(settings.seed)
        self.global_step = 0
        self.training_updates = 0
        self.epsilon = settings.epsilon_start

    def update_epsilon(self) -> None:
        fraction = min(1.0, self.global_step / max(1, self.settings.epsilon_decay_steps))
        self.epsilon = self.settings.epsilon_start + fraction * (
            self.settings.epsilon_final - self.settings.epsilon_start
        )

    def sample_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def save_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.store(state, action, reward, next_state, done)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def optimize_online_network(self):
        if len(self.replay_buffer) < self.settings.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.settings.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.online_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            best_next_q_values = self.target_network(next_states_t).max(dim=1, keepdim=True).values
            target_q_values = rewards_t + self.settings.gamma * (1.0 - dones_t) * best_next_q_values

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_updates += 1
        if self.training_updates % self.settings.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Evaluation, training and plotting
# -----------------------------------------------------------------------------
def evaluate_policy(eval_env: Grid3DEnv, agent: Grid3DDQNAgent, settings: DQNSettings, grid_max: np.ndarray) -> Tuple[float, float]:
    episode_returns: List[float] = []

    for _ in range(settings.eval_episodes):
        # Fresh random start each eval episode (no fixed seed), as in the original runs.
        obs, _ = eval_env.reset()
        state = normalize_obs(obs, grid_max)
        episode_return = 0.0

        for _ in range(settings.max_episode_steps):
            action = agent.sample_action(state, explore=False)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            state = normalize_obs(obs, grid_max)
            episode_return += reward  # true reward, no shaping
            if terminated or truncated:
                break

        episode_returns.append(episode_return)

    return float(np.mean(episode_returns)), float(np.std(episode_returns))


def train(settings: DQNSettings, output_dir: Path) -> List[Dict]:
    train_env = Grid3DEnv(L=settings.grid_l, W=settings.grid_w, H=settings.grid_h,
                          max_steps=settings.max_episode_steps, noise_prob=settings.noise_prob,
                          air_cost=settings.air_cost, reward_mode=settings.reward_mode,
                          random_goal=settings.random_goal)
    # Evaluation matches the historical airsweep protocol: same action noise as
    # training, and fresh random start positions each eval (see evaluate_policy).
    eval_env  = Grid3DEnv(L=settings.grid_l, W=settings.grid_w, H=settings.grid_h,
                          max_steps=settings.max_episode_steps, noise_prob=settings.noise_prob,
                          air_cost=settings.air_cost, reward_mode=settings.reward_mode,
                          random_goal=settings.random_goal)

    grid_max = np.array([train_env.L - 1, train_env.W - 1, train_env.H - 1], dtype=np.float32)

    agent = Grid3DDQNAgent(settings)

    # HER: a throwaway env (same layout) used only to recompute rewards for relabeled
    # goals, and a dedicated RNG so relabel sampling doesn't perturb the env stream.
    her_env = Grid3DEnv(L=settings.grid_l, W=settings.grid_w, H=settings.grid_h,
                        air_cost=settings.air_cost, reward_mode=settings.reward_mode) if settings.her else None
    her_rng = np.random.default_rng(settings.seed + 7)
    episode_experiences: List = []

    evaluation_log: List[Dict] = []
    # Seed the env RNG once (start positions + action noise). Gymnasium keeps
    # advancing this generator on subsequent seedless resets, so the whole run is
    # reproducible from `settings.seed` (set_reproducible_seeds only covers the
    # python/numpy/torch globals, not the env's separate Generator).
    obs, _ = train_env.reset(seed=settings.seed)
    state = normalize_obs(obs, grid_max)
    episode_number = 0
    episode_length = 0

    while agent.global_step < settings.total_steps:
        agent.update_epsilon()

        action = agent.sample_action(state, explore=True)
        dist_before = goal_distance(obs)
        z_before = float(obs[2])
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        dist_after = goal_distance(next_obs)
        z_after = float(next_obs[2])
        # Only `terminated` (goal reached) is a true MDP terminal state where the
        # bootstrap target must be zeroed. `truncated` (max_steps timeout) is an
        # artificial cutoff — the world continues, so we must keep bootstrapping
        # from next_state. Conflating the two would teach the agent that good
        # states near timeout have zero future value.
        episode_done = terminated or truncated

        # Potential-based reward shaping: F(s,s') = γ·Φ(s') - Φ(s)
        # Φ(s) = -dist - β_z·z  (z-aware: agent is rewarded for descending).
        # Disabled by --no-shaping (ablation: learn from the raw sparse reward only).
        if settings.shaping:
            phi_before = -dist_before - settings.beta_z * z_before
            phi_after  = -dist_after  - settings.beta_z * z_after
            shaped_reward = reward + settings.gamma * phi_after - phi_before
        else:
            shaped_reward = reward

        next_state = normalize_obs(next_obs, grid_max)
        agent.save_transition(state, action, shaped_reward, next_state, terminated)
        if settings.her:
            episode_experiences.append((obs.copy(), action, next_obs.copy()))
        agent.global_step += 1
        episode_length += 1

        if agent.global_step >= settings.learning_starts and agent.global_step % settings.learning_frequency == 0:
            agent.optimize_online_network()

        obs = next_obs
        state = next_state

        if episode_done or episode_length >= settings.max_episode_steps:
            if settings.her and episode_experiences:
                her_relabel_episode(episode_experiences, agent, her_env, grid_max, settings, her_rng)
                episode_experiences = []
            episode_number += 1
            episode_length = 0
            obs, _ = train_env.reset()
            state = normalize_obs(obs, grid_max)

        if agent.global_step % settings.eval_frequency == 0:
            mean_return, std_return = evaluate_policy(eval_env, agent, settings, grid_max)
            evaluation_log.append(
                {
                    "eval_index": len(evaluation_log) + 1,
                    "train_step": agent.global_step,
                    "episode": episode_number,
                    "mean_return": mean_return,
                    "std_return": std_return,
                }
            )

            if len(evaluation_log) % 25 == 0:
                print(
                    f"eval_point={len(evaluation_log)}, "
                    f"step={agent.global_step}, eval={mean_return:.3f} ± {std_return:.3f}, "
                    f"epsilon={agent.epsilon:.3f}"
                )

    train_env.close()
    eval_env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.online_network.state_dict(), output_dir / "online_q_network.pt")
    save_evaluation_csv(output_dir / "evaluation.csv", evaluation_log)
    plot_evaluation_rewards(evaluation_log, output_dir)

    return evaluation_log


def save_evaluation_csv(path: Path, evaluation_log: List[Dict]) -> None:
    if not evaluation_log:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluation_log[0].keys()))
        writer.writeheader()
        writer.writerows(evaluation_log)


def plot_evaluation_rewards(evaluation_log: List[Dict], output_dir: Path) -> None:
    if not evaluation_log:
        return

    eval_indices = np.asarray([row["eval_index"] for row in evaluation_log], dtype=float)
    means = np.asarray([row["mean_return"] for row in evaluation_log], dtype=float)
    stds = np.asarray([row["std_return"] for row in evaluation_log], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(eval_indices, means, linewidth=1.8, label="Mean Reward")
    plt.fill_between(eval_indices, means - stds, means + stds, alpha=0.3, label="±1 STD")
    plt.xlabel("Evaluation Point")
    plt.ylabel("Episode Reward")
    plt.title("Grid3D DQN Training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_reward.png", dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN for Grid3DEnv.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="dqn_results")
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--noise-prob", type=float, default=0.1)
    parser.add_argument("--grid-l", type=int, default=50)
    parser.add_argument("--grid-w", type=int, default=50)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--epsilon-decay", type=int, default=350_000)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-update-freq", type=int, default=250)
    parser.add_argument("--beta-z", type=float, default=1.0)
    parser.add_argument("--air-cost", type=float, default=0.2,
                        help="Avoidable-air penalty weight for horizontal cruise at z>=1 over free ground. "
                             "In 'energy' mode it is graded by height (air_cost * z/(H-1)).")
    parser.add_argument("--reward-mode", type=str, default="simple", choices=["simple", "energy"],
                        help="'simple'=flat air penalty (baseline); 'energy'=height-graded air penalty.")
    parser.add_argument("--random-goal", action="store_true",
                        help="Randomize the goal each episode (goal-conditioned task).")
    parser.add_argument("--her", action="store_true",
                        help="Hindsight Experience Replay: relabel failed trajectories with "
                             "achieved goals (use with --random-goal).")
    parser.add_argument("--her-k", type=int, default=4,
                        help="HER relabeled transitions added per real transition (default 4).")
    parser.add_argument("--no-shaping", dest="shaping", action="store_false",
                        help="Disable potential-based reward shaping (ablation: raw sparse reward only).")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    settings = DQNSettings(
        total_steps=args.steps,
        seed=args.seed,
        eval_frequency=args.eval_freq,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        device=args.device,
        noise_prob=args.noise_prob,
        grid_l=args.grid_l,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        max_episode_steps=args.max_episode_steps,
        epsilon_decay_steps=args.epsilon_decay,
        replay_capacity=args.replay_capacity,
        target_update_frequency=args.target_update_freq,
        beta_z=args.beta_z,
        air_cost=args.air_cost,
        reward_mode=args.reward_mode,
        random_goal=args.random_goal,
        her=args.her,
        her_k=args.her_k,
        shaping=args.shaping,
    )
    set_reproducible_seeds(settings.seed)

    output_dir = Path(__file__).resolve().parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_eval_points = settings.total_steps // settings.eval_frequency
    print(
        f"Training for {settings.total_steps} steps. "
        f"Evaluation every {settings.eval_frequency} steps -> "
        f"expected {expected_eval_points} eval points."
    )

    train(settings, output_dir)
    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

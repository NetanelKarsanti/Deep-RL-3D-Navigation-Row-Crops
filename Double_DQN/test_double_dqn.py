"""
Tests for the Double DQN implementation.

Run:
    python test_double_dqn.py
Expected output:
    Test 1 passed: optimize_online_network uses the Double DQN target.
    Test 2 passed: Double DQN target differs from the vanilla DQN target.
    Test 3 passed: Short training run completes and produces valid output.
    All tests passed.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure we import from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dqn import OBS_DIM, DQNSettings, Grid3DDQNAgent, train


class _ConstQNet(torch.nn.Module):
    """Network stub that ignores its input and returns a fixed Q-table.

    A zero-initialized `scale` parameter is broadcast onto the table so the output
    stays connected to an autograd parameter (the real optimize step can call
    backward / grad-clip on it) while leaving the returned values unchanged.
    """

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        return self.table.unsqueeze(0).repeat(n, 1) + self.scale


def _capture_target(reward, done):
    """Run the real `optimize_online_network` with stub networks and return the
    (current_q, target_q) tensors it actually computed, plus the online/target
    Q-tables used. online: best action = 2; target: action 2 -> 3.0."""
    online_table = [1.0, 2.0, 5.0, 0.5, 1.5, 0.1]   # argmax = action 2
    target_table = [4.0, 1.0, 3.0, 2.0, 0.5, 1.0]   # action 2 -> 3.0, max -> 4.0

    settings = DQNSettings(batch_size=1, gamma=0.99, seed=0, device="cpu")
    agent = Grid3DDQNAgent(settings)
    agent.online_network = _ConstQNet(online_table)
    agent.target_network = _ConstQNet(target_table)

    captured = {}
    real_loss = agent.loss_function

    def spy(current, target):
        captured["current"] = current.detach().clone()
        captured["target"] = target.detach().clone()
        return real_loss(current, target)

    agent.loss_function = spy

    # One known transition; the stored action (0) is deliberately NOT the argmax,
    # so a target that selects with the target net would pick a different action.
    state = np.zeros(OBS_DIM, dtype=np.float32)
    agent.save_transition(state, 0, reward, state, done)
    agent.optimize_online_network()

    return captured, online_table, target_table


# ---------------------------------------------------------------------------
# Test 1 — the production optimize step uses the Double DQN target
# ---------------------------------------------------------------------------
def test_optimize_uses_double_dqn_target():
    reward, gamma = 1.0, 0.99
    captured, online_table, target_table = _capture_target(reward, done=False)

    # Double DQN: online selects argmax (action 2), target evaluates it (3.0).
    best_action = int(np.argmax(online_table))
    expected = reward + gamma * target_table[best_action]
    assert abs(captured["target"].item() - expected) < 1e-5, \
        f"optimize target {captured['target'].item()} != Double DQN target {expected}"

    print("Test 1 passed: optimize_online_network uses the Double DQN target.")


# ---------------------------------------------------------------------------
# Test 2 — the Double DQN target differs from the vanilla DQN target,
#          and a terminal transition zeros the bootstrap
# ---------------------------------------------------------------------------
def test_double_differs_from_vanilla_and_handles_terminal():
    reward, gamma = 1.0, 0.99
    captured, _, target_table = _capture_target(reward, done=False)

    vanilla = reward + gamma * max(target_table)  # target net would max -> 4.0
    assert abs(captured["target"].item() - vanilla) > 1e-3, \
        "Double DQN target equals the vanilla DQN target — fix not applied."

    # Terminal transition: bootstrap must be zeroed -> target == reward.
    captured_terminal, _, _ = _capture_target(reward, done=True)
    assert abs(captured_terminal["target"].item() - reward) < 1e-5, \
        f"Terminal target {captured_terminal['target'].item()} != reward {reward}"

    print("Test 2 passed: Double DQN target differs from the vanilla DQN target.")


# ---------------------------------------------------------------------------
# Test 3 — Short training run completes and produces valid CSV
# ---------------------------------------------------------------------------
def test_short_training_run():
    out_dir = Path(__file__).resolve().parent / "runs" / "_test_run"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    settings = DQNSettings(
        total_steps=2_000,
        eval_frequency=500,
        eval_episodes=3,
        learning_starts=100,
        max_episode_steps=100,
        seed=0,
        noise_prob=0.0,
    )

    train(settings, out_dir)

    csv_path = out_dir / "evaluation.csv"
    assert csv_path.exists(), "evaluation.csv not created."

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    expected_points = settings.total_steps // settings.eval_frequency
    assert len(rows) == expected_points, \
        f"Expected {expected_points} eval points, got {len(rows)}"
    for row in rows:
        assert not np.isnan(float(row["mean_return"])), "NaN in mean_return"

    shutil.rmtree(out_dir)
    print("Test 3 passed: Short training run completes and produces valid output.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_optimize_uses_double_dqn_target()
    test_double_differs_from_vanilla_and_handles_terminal()
    test_short_training_run()
    print("All tests passed.")

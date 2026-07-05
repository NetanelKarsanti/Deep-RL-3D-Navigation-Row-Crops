#!/usr/bin/env python3
"""
Multi-seed sweep: train both the vanilla DQN (MDP_DQN) and the Double DQN
(Double_DQN) agents over the same set of random seeds, then collect a cross-seed
summary so the two algorithms can be compared on equal footing.

Both agents use the SAME default reward (discrete table + avoidable-air penalty,
reward_mode="simple") so the only variable is the Bellman target (vanilla vs
double). Each run trains for --steps and is evaluated with analyze_results.py.

Outputs (all under experiments/results/, which is git-ignored):
    results/<algo>/seed_<NNN>/            online_q_network.pt, evaluation.csv, plots
    results/<algo>/seed_<NNN>.train.log   captured training stdout
    results/<algo>/seed_<NNN>/analyze.log captured analysis stdout
    results/summary.csv                   one row per (algo, seed) + metrics
    results/aggregate.csv                 per-algo mean & std across seeds
    results/summary.md                    human-readable per-run + aggregate tables
    results/sweep.log                     full orchestrator output (for headless runs)

Designed for headless/server runs: every progress line and both summary tables are
written to results/sweep.log (and the .md/.csv files), so nothing needs to be
watched live. Tail any run with: tail -f results/<algo>/seed_<NNN>.train.log

Run it with the project's environment, e.g.:
    /home/ARO.local/netanelk/.conda/envs/parl_gpu/bin/python experiments/run_seed_sweep.py

Useful flags:
    --seeds 1 2 42 7 123     seeds to run (default)
    --steps 500000           training steps per run (default)
    --algos dqn double_dqn   which agents to run (default: both)
    --heatmap                also render the (slow) success heatmap in analysis
    --force                  retrain even if a checkpoint already exists
    --skip-analyze           train only, skip analyze_results.py
    --dry-run                print the commands without running anything
"""

import argparse
import csv
import datetime
import re
import statistics
import subprocess
import sys
from pathlib import Path

# experiments/ -> repo root is the parent
THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
RESULTS = THIS_DIR / "results"

# Logical algo name -> sub-project directory holding Q3_DQN.py / analyze_results.py
ALGO_DIRS = {
    "dqn": REPO / "MDP_DQN",
    "double_dqn": REPO / "Double_DQN",
}

DEFAULT_SEEDS = [1, 2, 42, 7, 123]
DEFAULT_STEPS = 500_000
# Match the training-time episode horizon so analysis evaluates like training did.
EVAL_MAX_STEPS = 500

SUMMARY_FIELDS = [
    "algo", "seed", "best_eval", "final_eval",
    "success_rate", "avg_steps", "path_efficiency",
]
AGG_METRICS = ["best_eval", "final_eval", "success_rate", "avg_steps", "path_efficiency"]

# Open file handle for results/sweep.log; set in main() for non-dry runs so every
# log() line is mirrored to disk (headless server runs have no live console).
_LOG_FH = None


def log(msg: str = "") -> None:
    """Print to stdout AND append to results/sweep.log (if logging is set up)."""
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def parse_eval_csv(path: Path):
    """Return (best mean_return, final mean_return) from a training evaluation.csv."""
    if not path.exists():
        return float("nan"), float("nan")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("nan"), float("nan")
    returns = [float(r["mean_return"]) for r in rows]
    return max(returns), returns[-1]


def parse_analyze_stdout(text: str):
    """Pull the headline metrics that analyze_results.py prints."""
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else float("nan")

    return {
        "success_rate": grab(r"Success rate:\s*([\d.]+)%"),
        "avg_steps": grab(r"Avg steps \(success\):\s*([\d.]+)"),
        "path_efficiency": grab(r"Avg path efficiency:\s*([\d.]+)x"),
    }


def run_one(algo: str, seed: int, steps: int, reward_mode: str, air_cost, random_goal: bool,
            her: bool, heatmap: bool, force: bool, skip_analyze: bool, dry_run: bool) -> dict:
    algo_dir = ALGO_DIRS[algo]
    out_dir = RESULTS / algo / f"seed_{seed:03d}"
    model = out_dir / "online_q_network.pt"
    train_log = RESULTS / algo / f"seed_{seed:03d}.train.log"

    # `-u` = unbuffered subprocess stdout, so the per-run train.log updates live
    # (block-buffering otherwise hides progress until the buffer fills / process ends,
    # which makes `tail -f` look frozen on a headless/SLURM run).
    # Absolute --output so Q3_DQN.py's `Path(__file__).parent / output` resolves
    # to OUR results dir rather than inside the sub-project folder.
    train_cmd = [
        sys.executable, "-u", str(algo_dir / "Q3_DQN.py"),
        "--output", str(out_dir),
        "--seed", str(seed),
        "--steps", str(steps),
        "--reward-mode", reward_mode,
    ]
    if air_cost is not None:
        train_cmd += ["--air-cost", str(air_cost)]
    if random_goal:
        train_cmd.append("--random-goal")
    if her:
        train_cmd.append("--her")

    analyze_cmd = [
        sys.executable, "-u", str(algo_dir / "analyze_results.py"),
        "--results", str(out_dir),
        "--model", str(model),
        "--max-steps", str(EVAL_MAX_STEPS),
        "--seed", str(seed),
    ]
    if not heatmap:
        analyze_cmd.append("--no-heatmap")
    if random_goal:
        analyze_cmd.append("--random-goal")

    if dry_run:
        log("    train:   " + " ".join(train_cmd))
        if not skip_analyze:
            log("    analyze: " + " ".join(analyze_cmd))
        return {"algo": algo, "seed": seed}

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- train (skip if a checkpoint already exists, unless --force) ----
    if model.exists() and not force:
        log(f"    [skip train] {model.name} already exists (use --force to retrain)")
    else:
        log(f"    [train] -> {train_log}")
        with open(train_log, "w") as logf:
            subprocess.run(train_cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)

    # ---- analyze ----
    metrics = {"success_rate": float("nan"), "avg_steps": float("nan"),
               "path_efficiency": float("nan")}
    if not skip_analyze and model.exists():
        log("    [analyze]")
        # Analysis is non-essential: never let an analyze hiccup discard a finished
        # (expensive) training run. On failure we keep the model + evaluation.csv,
        # record NaN metrics, and move on — re-run analyze_results.py later if needed.
        proc = subprocess.run(analyze_cmd, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out_dir / "analyze.log").write_text(proc.stdout)
        if proc.returncode == 0:
            metrics = parse_analyze_stdout(proc.stdout)
        else:
            log(f"    [warn] analyze failed (exit {proc.returncode}); "
                f"see {out_dir / 'analyze.log'} — continuing with NaN metrics")

    best, final = parse_eval_csv(out_dir / "evaluation.csv")
    row = {"algo": algo, "seed": seed, "best_eval": best, "final_eval": final}
    row.update(metrics)
    log(f"    [done] best_eval={best:.2f} final_eval={final:.2f} "
        f"success={metrics['success_rate']:.1f}%")
    return row


def _aggregate(rows: list) -> dict:
    """Per-algo {metric: (mean, std, n)} across seeds, NaNs dropped."""
    agg = {}
    for algo in ALGO_DIRS:
        algo_rows = [r for r in rows if r.get("algo") == algo]
        if not algo_rows:
            continue
        per_metric = {}
        for k in AGG_METRICS:
            vals = [r[k] for r in algo_rows
                    if isinstance(r.get(k), float) and r[k] == r[k]]  # drop NaN
            if not vals:
                per_metric[k] = (float("nan"), float("nan"), 0)
            else:
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                per_metric[k] = (statistics.mean(vals), std, len(vals))
        agg[algo] = per_metric
    return agg


def write_summary(rows: list):
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- per-run summary.csv ----
    summary_path = RESULTS / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})

    # ---- cross-seed aggregate (mean / std) ----
    agg = _aggregate(rows)

    # aggregate.csv: one row per algo, mean+std columns per metric
    agg_path = RESULTS / "aggregate.csv"
    agg_fields = ["algo", "n_seeds"] + [f"{m}_{s}" for m in AGG_METRICS for s in ("mean", "std")]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for algo, per_metric in agg.items():
            row = {"algo": algo, "n_seeds": max((v[2] for v in per_metric.values()), default=0)}
            for m in AGG_METRICS:
                mean, std, _ = per_metric[m]
                row[f"{m}_mean"] = f"{mean:.4f}"
                row[f"{m}_std"] = f"{std:.4f}"
            writer.writerow(row)

    # summary.md: human-readable per-run + aggregate tables
    md = ["# Seed-sweep results", "", "## Per run", "",
          "| algo | seed | best_eval | final_eval | success_% | avg_steps | path_eff |",
          "|------|------|-----------|------------|-----------|-----------|----------|"]
    for r in rows:
        md.append("| {algo} | {seed} | {best_eval:.2f} | {final_eval:.2f} | "
                  "{success_rate:.1f} | {avg_steps:.1f} | {path_efficiency:.2f} |".format(
                      algo=r["algo"], seed=r["seed"],
                      best_eval=r.get("best_eval", float("nan")),
                      final_eval=r.get("final_eval", float("nan")),
                      success_rate=r.get("success_rate", float("nan")),
                      avg_steps=r.get("avg_steps", float("nan")),
                      path_efficiency=r.get("path_efficiency", float("nan"))))
    md += ["", "## Cross-seed (mean ± std)", "",
           "| algo | n | best_eval | final_eval | success_% | avg_steps | path_eff |",
           "|------|---|-----------|------------|-----------|-----------|----------|"]
    for algo, per_metric in agg.items():
        n = max((v[2] for v in per_metric.values()), default=0)
        cells = " | ".join(f"{per_metric[m][0]:.2f} ± {per_metric[m][1]:.2f}" for m in AGG_METRICS)
        md.append(f"| {algo} | {n} | {cells} |")
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")

    # ---- echo the aggregate table to console + sweep.log ----
    log(f"\nPer-run summary : {summary_path}")
    log(f"Aggregate (csv) : {agg_path}")
    log(f"Aggregate (md)  : {RESULTS / 'summary.md'}")
    log("\n=== Cross-seed summary (mean +/- std) ===")
    header = f"{'algo':<12}" + "".join(f"{k:>18}" for k in AGG_METRICS)
    log(header)
    log("-" * len(header))
    for algo, per_metric in agg.items():
        cells = []
        for k in AGG_METRICS:
            mean, std, n = per_metric[k]
            if n == 0:
                cells.append(f"{'n/a':>18}")
            elif n == 1:
                cells.append(f"{mean:>18.2f}")
            else:
                cells.append(f"{mean:>11.2f}+/-{std:<4.2f}")
        log(f"{algo:<12}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--algos", nargs="+", choices=list(ALGO_DIRS),
                        default=list(ALGO_DIRS))
    parser.add_argument("--reward-mode", type=str, default="simple",
                        choices=["simple", "energy"],
                        help="Reward variant passed to BOTH algos (default: simple baseline).")
    parser.add_argument("--air-cost", type=float, default=None,
                        help="Override the air-penalty weight (default: leave each agent's default 0.2).")
    parser.add_argument("--random-goal", action="store_true",
                        help="Randomize the goal each episode for BOTH train and analyze (both algos).")
    parser.add_argument("--her", action="store_true",
                        help="Hindsight Experience Replay during training (both algos; use with --random-goal).")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also render the slow per-cell success heatmap.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a checkpoint already exists.")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Train only; do not run analyze_results.py.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running anything.")
    args = parser.parse_args()

    jobs = [(algo, seed) for algo in args.algos for seed in args.seeds]

    # Mirror all orchestrator output to results/sweep.log (append) for headless runs.
    global _LOG_FH
    if not args.dry_run:
        RESULTS.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(RESULTS / "sweep.log", "a")
        log(f"\n===== sweep started {datetime.datetime.now().isoformat(timespec='seconds')} =====")

    log(f"Sweep: {len(args.algos)} algo(s) x {len(args.seeds)} seed(s) = {len(jobs)} runs")
    log(f"  algos={args.algos}  seeds={args.seeds}  steps={args.steps}")
    log(f"  reward_mode={args.reward_mode}  air_cost={args.air_cost if args.air_cost is not None else 'default(0.2)'}  random_goal={args.random_goal}  her={args.her}")
    log(f"  python={sys.executable}")
    log(f"  results -> {RESULTS}\n")

    rows = []
    for i, (algo, seed) in enumerate(jobs, 1):
        log(f"[{i}/{len(jobs)}] algo={algo} seed={seed}")
        row = run_one(algo, seed, args.steps, args.reward_mode, args.air_cost, args.random_goal,
                      args.her, args.heatmap, args.force, args.skip_analyze, args.dry_run)
        if not args.dry_run:
            rows.append(row)

    if args.dry_run:
        log("\nDry run complete — no commands were executed.")
        return
    write_summary(rows)
    log(f"\n===== sweep finished {datetime.datetime.now().isoformat(timespec='seconds')} =====")
    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()

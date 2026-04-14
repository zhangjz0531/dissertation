
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_ABLATION_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\sticky_execution_ablation"
DEFAULT_OUT_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413"


# =========================================================
# Bootstrap confidence intervals on rebalance-level returns
#
# Uses moving block bootstrap on net_return from:
#   - static_test_actions.csv
#   - sticky_test_actions.csv
#
# Metrics bootstrapped:
#   - cumulative return
#   - annualized return
#   - Sharpe
#   - max drawdown
# plus:
#   - P(cumulative return > 0)
#   - P(Sharpe > 0)
# =========================================================


def annualized_return(final_equity: float, total_days: int) -> float:
    total_days = max(1, int(total_days))
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_ratio(period_returns: np.ndarray, horizon: int) -> float:
    if len(period_returns) <= 1:
        return 0.0
    std = period_returns.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((period_returns.mean() / std) * math.sqrt(252.0 / horizon))


def max_drawdown_from_returns(period_returns: np.ndarray) -> float:
    equity = [1.0]
    for r in period_returns:
        equity.append(equity[-1] * (1.0 + float(r)))
    eq = np.array(equity, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def moving_block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, max(1, n - block_length + 1)))
        block = list(range(start, min(n, start + block_length)))
        idx.extend(block)
    return np.array(idx[:n], dtype=int)


def bootstrap_strategy(
    actions_df: pd.DataFrame,
    horizon: int,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    returns = actions_df["net_return"].astype(float).values
    n = len(returns)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_bootstrap):
        idx = moving_block_bootstrap_indices(n=n, block_length=block_length, rng=rng)
        sample = returns[idx]

        equity = 1.0
        for r in sample:
            equity *= (1.0 + float(r))

        cumret = float(equity - 1.0)
        annret = annualized_return(equity, len(sample) * horizon)
        sharpe = sharpe_ratio(sample, horizon=horizon)
        mdd = max_drawdown_from_returns(sample)

        rows.append({
            "bootstrap_id": i + 1,
            "cumulative_return": cumret,
            "annualized_return": annret,
            "sharpe": sharpe,
            "max_drawdown": mdd,
        })

    boot_df = pd.DataFrame(rows)

    summary = {
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(block_length),
        "orig_periods": int(n),
        "cumret_mean": float(boot_df["cumulative_return"].mean()),
        "cumret_median": float(boot_df["cumulative_return"].median()),
        "cumret_ci_5": float(boot_df["cumulative_return"].quantile(0.05)),
        "cumret_ci_95": float(boot_df["cumulative_return"].quantile(0.95)),
        "annret_mean": float(boot_df["annualized_return"].mean()),
        "annret_ci_5": float(boot_df["annualized_return"].quantile(0.05)),
        "annret_ci_95": float(boot_df["annualized_return"].quantile(0.95)),
        "sharpe_mean": float(boot_df["sharpe"].mean()),
        "sharpe_median": float(boot_df["sharpe"].median()),
        "sharpe_ci_5": float(boot_df["sharpe"].quantile(0.05)),
        "sharpe_ci_95": float(boot_df["sharpe"].quantile(0.95)),
        "mdd_mean": float(boot_df["max_drawdown"].mean()),
        "mdd_ci_5": float(boot_df["max_drawdown"].quantile(0.05)),
        "mdd_ci_95": float(boot_df["max_drawdown"].quantile(0.95)),
        "prob_cumret_positive": float((boot_df["cumulative_return"] > 0).mean()),
        "prob_sharpe_positive": float((boot_df["sharpe"] > 0).mean()),
    }
    return boot_df, summary


def main():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for static vs sticky execution.")
    parser.add_argument("--ablation_dir", type=str, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--block_length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    static_actions = pd.read_csv(ablation_dir / "static_test_actions.csv")
    sticky_actions = pd.read_csv(ablation_dir / "sticky_test_actions.csv")

    out_root = Path(args.out_dir) / "bootstrap_confidence_intervals"
    out_root.mkdir(parents=True, exist_ok=True)

    static_boot, static_summary = bootstrap_strategy(
        actions_df=static_actions,
        horizon=args.horizon,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        seed=args.seed,
    )
    sticky_boot, sticky_summary = bootstrap_strategy(
        actions_df=sticky_actions,
        horizon=args.horizon,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        seed=args.seed + 1,
    )

    static_boot.to_csv(out_root / "bootstrap_static_test.csv", index=False)
    sticky_boot.to_csv(out_root / "bootstrap_sticky_test.csv", index=False)

    compare_rows = [
        {"strategy": "static_prediction_only", **static_summary},
        {"strategy": "final_sticky_execution", **sticky_summary},
    ]
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_root / "bootstrap_summary_comparison.csv", index=False)

    summary = {
        "static_prediction_only": static_summary,
        "final_sticky_execution": sticky_summary,
    }
    with open(out_root / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Bootstrap confidence intervals saved to: {out_root}")
    print("\nStatic bootstrap summary:")
    print(json.dumps(static_summary, ensure_ascii=False, indent=2))
    print("\nSticky bootstrap summary:")
    print(json.dumps(sticky_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

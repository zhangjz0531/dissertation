
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
# Bootstrap block-length sensitivity
#
# Reuses:
#   - static_test_actions.csv
#   - sticky_test_actions.csv
# from sticky_execution_ablation/
#
# Runs moving block bootstrap for multiple block lengths
# to check whether the bootstrap conclusion is stable.
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
            "block_length": int(block_length),
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
    parser = argparse.ArgumentParser(description="Bootstrap block-length sensitivity for static vs sticky execution.")
    parser.add_argument("--ablation_dir", type=str, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--block_lengths", type=str, default="3,5,10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    static_actions = pd.read_csv(ablation_dir / "static_test_actions.csv")
    sticky_actions = pd.read_csv(ablation_dir / "sticky_test_actions.csv")

    block_lengths = [int(x.strip()) for x in args.block_lengths.split(",") if x.strip()]
    out_root = Path(args.out_dir) / "bootstrap_block_length_sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    static_boot_list = []
    sticky_boot_list = []
    summary_json = {"static_prediction_only": {}, "final_sticky_execution": {}}

    for i, bl in enumerate(block_lengths):
        static_boot, static_summary = bootstrap_strategy(
            actions_df=static_actions,
            horizon=args.horizon,
            n_bootstrap=args.n_bootstrap,
            block_length=bl,
            seed=args.seed + i * 10,
        )
        sticky_boot, sticky_summary = bootstrap_strategy(
            actions_df=sticky_actions,
            horizon=args.horizon,
            n_bootstrap=args.n_bootstrap,
            block_length=bl,
            seed=args.seed + i * 10 + 1,
        )

        static_boot["strategy"] = "static_prediction_only"
        sticky_boot["strategy"] = "final_sticky_execution"
        static_boot_list.append(static_boot)
        sticky_boot_list.append(sticky_boot)

        all_summary_rows.append({"strategy": "static_prediction_only", **static_summary})
        all_summary_rows.append({"strategy": "final_sticky_execution", **sticky_summary})

        summary_json["static_prediction_only"][str(bl)] = static_summary
        summary_json["final_sticky_execution"][str(bl)] = sticky_summary

    pd.concat(static_boot_list, ignore_index=True).to_csv(out_root / "bootstrap_static_block_sensitivity.csv", index=False)
    pd.concat(sticky_boot_list, ignore_index=True).to_csv(out_root / "bootstrap_sticky_block_sensitivity.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(out_root / "bootstrap_block_sensitivity_summary.csv", index=False)

    with open(out_root / "bootstrap_block_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"Bootstrap block-length sensitivity saved to: {out_root}")
    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

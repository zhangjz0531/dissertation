from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import ABLATIONS_DIR, ROBUSTNESS_DIR, ensure_all_core_dirs, ensure_dir, timestamp_tag


# =========================================================
# Bootstrap confidence intervals for execution comparison
#
# Uses moving block bootstrap on rebalance-level net_return from:
#   - frozen_static_test_actions.csv (or canonical static_test_actions.csv)
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
#
# Also computes paired bootstrap differences:
#   sticky - static
# to quantify whether sticky robustly beats / loses to static.
# =========================================================


def latest_file_by_glob(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_default_ablation_dir() -> Path:
    base = ABLATIONS_DIR / "sticky_execution"
    p = latest_file_by_glob(base, "run_*")
    if p is None:
        raise FileNotFoundError(
            f"Could not auto-find latest sticky execution ablation run under {base}. "
            f"Please pass --ablation_dir explicitly."
        )
    return p


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


def load_actions_file(ablation_dir: Path, preferred_names: list[str]) -> pd.DataFrame:
    for name in preferred_names:
        p = ablation_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if "net_return" not in df.columns:
                raise ValueError(f"{p} is missing required column: net_return")
            return df
    raise FileNotFoundError(
        f"None of the candidate files exist under {ablation_dir}: {preferred_names}"
    )


def bootstrap_strategy(
    actions_df: pd.DataFrame,
    horizon: int,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    returns = actions_df["net_return"].astype(float).values
    n = len(returns)
    if n == 0:
        raise ValueError("actions_df contains zero rows.")

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


def bootstrap_paired_difference(
    static_actions_df: pd.DataFrame,
    sticky_actions_df: pd.DataFrame,
    horizon: int,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    static_returns = static_actions_df["net_return"].astype(float).values
    sticky_returns = sticky_actions_df["net_return"].astype(float).values

    n = min(len(static_returns), len(sticky_returns))
    if n == 0:
        raise ValueError("At least one action series is empty.")
    if len(static_returns) != len(sticky_returns):
        static_returns = static_returns[:n]
        sticky_returns = sticky_returns[:n]

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_bootstrap):
        idx = moving_block_bootstrap_indices(n=n, block_length=block_length, rng=rng)
        s0 = static_returns[idx]
        s1 = sticky_returns[idx]

        # Static
        eq_static = 1.0
        for r in s0:
            eq_static *= (1.0 + float(r))
        static_cumret = float(eq_static - 1.0)
        static_annret = annualized_return(eq_static, len(s0) * horizon)
        static_sharpe = sharpe_ratio(s0, horizon=horizon)
        static_mdd = max_drawdown_from_returns(s0)

        # Sticky
        eq_sticky = 1.0
        for r in s1:
            eq_sticky *= (1.0 + float(r))
        sticky_cumret = float(eq_sticky - 1.0)
        sticky_annret = annualized_return(eq_sticky, len(s1) * horizon)
        sticky_sharpe = sharpe_ratio(s1, horizon=horizon)
        sticky_mdd = max_drawdown_from_returns(s1)

        rows.append({
            "bootstrap_id": i + 1,
            "diff_cumret_sticky_minus_static": sticky_cumret - static_cumret,
            "diff_annret_sticky_minus_static": sticky_annret - static_annret,
            "diff_sharpe_sticky_minus_static": sticky_sharpe - static_sharpe,
            "diff_mdd_sticky_minus_static": sticky_mdd - static_mdd,
        })

    diff_df = pd.DataFrame(rows)

    # For MDD, "better" means LESS negative, i.e. sticky - static > 0 means sticky drawdown is shallower.
    summary = {
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(block_length),
        "paired_periods": int(n),

        "cumret_diff_mean": float(diff_df["diff_cumret_sticky_minus_static"].mean()),
        "cumret_diff_ci_5": float(diff_df["diff_cumret_sticky_minus_static"].quantile(0.05)),
        "cumret_diff_ci_95": float(diff_df["diff_cumret_sticky_minus_static"].quantile(0.95)),
        "prob_sticky_beats_static_cumret": float((diff_df["diff_cumret_sticky_minus_static"] > 0).mean()),
        "prob_static_beats_sticky_cumret": float((diff_df["diff_cumret_sticky_minus_static"] < 0).mean()),

        "sharpe_diff_mean": float(diff_df["diff_sharpe_sticky_minus_static"].mean()),
        "sharpe_diff_ci_5": float(diff_df["diff_sharpe_sticky_minus_static"].quantile(0.05)),
        "sharpe_diff_ci_95": float(diff_df["diff_sharpe_sticky_minus_static"].quantile(0.95)),
        "prob_sticky_beats_static_sharpe": float((diff_df["diff_sharpe_sticky_minus_static"] > 0).mean()),
        "prob_static_beats_sticky_sharpe": float((diff_df["diff_sharpe_sticky_minus_static"] < 0).mean()),

        "mdd_diff_mean": float(diff_df["diff_mdd_sticky_minus_static"].mean()),
        "mdd_diff_ci_5": float(diff_df["diff_mdd_sticky_minus_static"].quantile(0.05)),
        "mdd_diff_ci_95": float(diff_df["diff_mdd_sticky_minus_static"].quantile(0.95)),
        "prob_sticky_beats_static_mdd": float((diff_df["diff_mdd_sticky_minus_static"] > 0).mean()),
        "prob_static_beats_sticky_mdd": float((diff_df["diff_mdd_sticky_minus_static"] < 0).mean()),
    }
    return diff_df, summary


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for frozen static vs sticky execution.")
    parser.add_argument("--ablation_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(ROBUSTNESS_DIR / "bootstrap_ci"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--block_length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else resolve_default_ablation_dir()

    static_actions = load_actions_file(
        ablation_dir,
        preferred_names=["frozen_static_test_actions.csv", "static_test_actions.csv"],
    )
    sticky_actions = load_actions_file(
        ablation_dir,
        preferred_names=["sticky_test_actions.csv"],
    )

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
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
    diff_boot, diff_summary = bootstrap_paired_difference(
        static_actions_df=static_actions,
        sticky_actions_df=sticky_actions,
        horizon=args.horizon,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        seed=args.seed + 2,
    )

    static_boot.to_csv(out_root / "bootstrap_static_test.csv", index=False)
    sticky_boot.to_csv(out_root / "bootstrap_sticky_test.csv", index=False)
    diff_boot.to_csv(out_root / "bootstrap_difference_test.csv", index=False)

    compare_rows = [
        {"strategy": "frozen_static_baseline", **static_summary},
        {"strategy": "optimized_sticky_execution", **sticky_summary},
    ]
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_root / "bootstrap_summary_comparison.csv", index=False)

    summary = {
        "ablation_dir": str(ablation_dir.resolve()),
        "static_strategy": "frozen_static_baseline",
        "sticky_strategy": "optimized_sticky_execution",
        "static_summary": static_summary,
        "sticky_summary": sticky_summary,
        "paired_difference_summary": diff_summary,
    }
    with open(out_root / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Bootstrap confidence intervals saved to: {out_root}")
    print("\nFrozen static bootstrap summary:")
    print(json.dumps(static_summary, ensure_ascii=False, indent=2))
    print("\nSticky bootstrap summary:")
    print(json.dumps(sticky_summary, ensure_ascii=False, indent=2))
    print("\nPaired bootstrap difference summary (sticky - static):")
    print(json.dumps(diff_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
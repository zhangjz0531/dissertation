from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import (
    MAIN_H5_DATA,
    EXECUTION_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)


# =========================================================
# Benchmark + Cost Sensitivity
#
# What this script fills:
# 1) Buy-and-hold benchmark gap
#    - universe_equal_weight_buy_hold
#    - market_proxy_buy_hold (from mkt_return_* in feature dataset)
#
# 2) Cost sensitivity gap
#    - rerun frozen static / optimized sticky / buy-hold benchmarks
#      under multiple cost settings (default: 5,10,20,30 bps)
#
# Inputs:
#   - latest final_system_*/final_system_manifest.json by default
#   - source_predictions_path from manifest
#   - source_feature_data_path from manifest
#
# Outputs:
#   - benchmark_comparison_default_cost.csv
#   - cost_sensitivity_metrics.csv
#   - best_strategy_by_cost.csv
#   - benchmark_cost_summary.json
#   - cost_sensitivity_summary.json
#   - *_default_cost_actions.csv
# =========================================================


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def safe_read_json(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def latest_file_by_glob(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_default_final_system_manifest_path() -> Path:
    base = EXECUTION_DIR / "final_system"
    p = latest_file_by_glob(base, "final_system_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError(
            f"Could not auto-find latest final_system manifest under {base}. "
            f"Please pass --final_system_manifest_path explicitly."
        )
    return p


def annualized_return(final_equity: float, total_days: int) -> float:
    total_days = max(1, int(total_days))
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_ratio(period_returns: List[float], horizon: int) -> float:
    if len(period_returns) <= 1:
        return 0.0
    arr = np.asarray(period_returns, dtype=float)
    std = arr.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((arr.mean() / std) * math.sqrt(252.0 / horizon))


def max_drawdown_from_equity_curve(equity_curve: List[float]) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def infer_horizon_from_feature_df(feature_df: pd.DataFrame) -> int:
    return_cols = [c for c in feature_df.columns if c.startswith("future_return_")]
    if len(return_cols) != 1:
        return 5
    col = return_cols[0]
    if col.endswith("1d"):
        return 1
    if col.endswith("5d"):
        return 5
    return 5


def normalize_predictions(pred: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["date"] = pd.to_datetime(out["date"])

    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"])

    if "pred_prob" not in out.columns:
        cand = [c for c in out.columns if "prob" in c.lower()]
        if len(cand) == 0:
            raise ValueError("Predictions file must contain 'pred_prob' or equivalent.")
        out = out.rename(columns={cand[0]: "pred_prob"})

    if "split" not in out.columns:
        if "split" not in feat.columns:
            raise ValueError("Neither predictions nor feature data contains 'split'.")
        out = out.merge(
            feat[["date", "stock", "split"]],
            on=["date", "stock"],
            how="left",
        )

    if "future_return" not in out.columns:
        ret_cols = [c for c in feat.columns if c.startswith("future_return_")]
        if len(ret_cols) != 1:
            raise ValueError("Feature data must contain exactly one future_return_* column.")
        ret_col = ret_cols[0]
        out = out.merge(
            feat[["date", "stock", "split", ret_col]].rename(columns={ret_col: "future_return"}),
            on=["date", "stock", "split"],
            how="left",
        )

    required_cols = ["date", "stock", "split", "pred_prob", "future_return"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Predictions data missing required columns after normalization: {missing}")

    if out["future_return"].isna().any():
        raise ValueError("Predictions data contains missing future_return after normalization.")

    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def choose_portfolio_static(
    day_df: pd.DataFrame,
    top_k: int,
    min_prob: float,
) -> List[str]:
    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    chosen = day[day["pred_prob"] >= min_prob].head(top_k)
    return chosen["stock"].astype(str).tolist()


def choose_portfolio_sticky(
    day_df: pd.DataFrame,
    current_positions: Dict[str, Dict],
    top_k: int,
    entry_prob: float,
    exit_prob: float,
    switch_buffer: float,
    min_hold_periods: int,
    max_new_names_per_rebalance: int,
) -> Tuple[List[str], Dict]:
    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    prob_map = dict(zip(day["stock"], day["pred_prob"]))

    kept = []
    forced_keep = []

    for stock, info in current_positions.items():
        prob = float(prob_map.get(stock, -1.0))
        held_for = int(info.get("held_for", 0))

        if held_for < min_hold_periods:
            forced_keep.append(stock)
        elif prob >= exit_prob:
            kept.append(stock)

    portfolio = list(dict.fromkeys(forced_keep + kept))
    portfolio = portfolio[:top_k]

    current_set = set(current_positions.keys())
    new_candidates = [
        s for s in day["stock"].astype(str).tolist()
        if s not in current_set and float(prob_map.get(s, -1.0)) >= entry_prob
    ]

    added_new = 0

    for s in new_candidates:
        if len(portfolio) >= top_k:
            break
        if added_new >= max_new_names_per_rebalance:
            break
        portfolio.append(s)
        added_new += 1

    if len(portfolio) == top_k:
        for s in new_candidates:
            if s in portfolio:
                continue
            if added_new >= max_new_names_per_rebalance:
                break

            weakest_name = None
            weakest_prob = 1e9
            for held in portfolio:
                if held in forced_keep:
                    continue
                p = float(prob_map.get(held, -1.0))
                if p < weakest_prob:
                    weakest_prob = p
                    weakest_name = held

            if weakest_name is None:
                continue

            new_prob = float(prob_map.get(s, -1.0))
            if new_prob >= weakest_prob + switch_buffer and new_prob >= entry_prob:
                portfolio.remove(weakest_name)
                portfolio.append(s)
                added_new += 1

    portfolio = sorted(
        list(set(portfolio)),
        key=lambda x: float(prob_map.get(x, -1.0)),
        reverse=True,
    )[:top_k]

    debug = {
        "kept_existing": len([s for s in portfolio if s in current_set]),
        "newly_added": len([s for s in portfolio if s not in current_set]),
        "forced_keep_count": len([s for s in portfolio if s in forced_keep]),
        "entry_candidates_count": len(new_candidates),
    }
    return portfolio, debug


def compute_metrics_from_series(
    period_returns: List[float],
    turns: List[float],
    holds: List[int],
    exposures: List[float],
    horizon: int,
) -> Dict[str, float]:
    if len(period_returns) == 0:
        return {
            "periods": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "avg_holdings": 0.0,
            "avg_exposure": 0.0,
        }

    equity = 1.0
    equity_curve = [equity]
    for r in period_returns:
        equity *= (1.0 + float(r))
        equity_curve.append(equity)

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.asarray(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }


def run_static_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    top_k: int,
    min_prob: float,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    if pred_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        names = choose_portfolio_static(day, top_k=top_k, min_prob=min_prob)

        if len(names) == 0:
            new_w = {}
        else:
            w = 1.0 / len(names)
            new_w = {s: w for s in names}

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))

        rows.append({
            "date": pd.Timestamp(dt),
            "strategy": "frozen_static_baseline",
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
        })

        prev_w = new_w

    metrics = compute_metrics_from_series(period_returns, turns, holds, exposures, horizon)
    return metrics, pd.DataFrame(rows)


def run_sticky_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    top_k: int,
    entry_prob: float,
    exit_prob: float,
    switch_buffer: float,
    min_hold_periods: int,
    max_new_names_per_rebalance: int,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    if pred_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    current_positions: Dict[str, Dict] = {}
    prev_w = {}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        selected_names, debug = choose_portfolio_sticky(
            day_df=day,
            current_positions=current_positions,
            top_k=top_k,
            entry_prob=entry_prob,
            exit_prob=exit_prob,
            switch_buffer=switch_buffer,
            min_hold_periods=min_hold_periods,
            max_new_names_per_rebalance=max_new_names_per_rebalance,
        )

        if len(selected_names) == 0:
            new_w = {}
        else:
            w = 1.0 / len(selected_names)
            new_w = {s: w for s in selected_names}

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))

        rows.append({
            "date": pd.Timestamp(dt),
            "strategy": "optimized_sticky_execution",
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
            "kept_existing": debug["kept_existing"],
            "newly_added": debug["newly_added"],
            "forced_keep_count": debug["forced_keep_count"],
            "entry_candidates_count": debug["entry_candidates_count"],
        })

        next_positions = {}
        for s in new_w.keys():
            if s in current_positions:
                next_positions[s] = {"held_for": int(current_positions[s]["held_for"]) + 1}
            else:
                next_positions[s] = {"held_for": 1}
        current_positions = next_positions
        prev_w = new_w

    metrics = compute_metrics_from_series(period_returns, turns, holds, exposures, horizon)
    return metrics, pd.DataFrame(rows)


def run_equal_weight_buy_and_hold(
    pred_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Equal-weight buy-and-hold over the whole stock universe present on the first test rebalance date.
    Cost convention matches the main backtests:
      - initial buy incurs turnover cost
      - no terminal liquidation cost is charged
    """
    if pred_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    first_day = df[df["date"] == rebalance_dates[0]].copy().sort_values("stock").reset_index(drop=True)
    universe_names = first_day["stock"].astype(str).tolist()
    if len(universe_names) == 0:
        raise ValueError("No stocks found on the first test rebalance date.")

    initial_w = 1.0 / len(universe_names)

    prev_w: Dict[str, float] = {}
    current_drift_w: Dict[str, float] = {}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []

    for idx, dt in enumerate(rebalance_dates):
        day = df[df["date"] == dt].copy()
        ret_map = day.set_index("stock")["future_return"].to_dict()

        if idx == 0:
            new_w = {s: initial_w for s in universe_names}
        else:
            new_w = current_drift_w.copy()

        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))

        rows.append({
            "date": pd.Timestamp(dt),
            "strategy": "universe_equal_weight_buy_hold",
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
        })

        drift_values = {s: new_w[s] * (1.0 + ret_map.get(s, 0.0)) for s in new_w.keys()}
        total_val = float(sum(drift_values.values()))
        if total_val > 0:
            current_drift_w = {s: v / total_val for s, v in drift_values.items()}
        else:
            current_drift_w = {}

        prev_w = new_w

    metrics = compute_metrics_from_series(period_returns, turns, holds, exposures, horizon)
    return metrics, pd.DataFrame(rows)


def detect_market_return_col(feature_df: pd.DataFrame, horizon: int) -> Optional[str]:
    exact = f"mkt_return_{horizon}d"
    if exact in feature_df.columns:
        return exact

    cands = [c for c in feature_df.columns if c.startswith("mkt_return_")]
    if len(cands) == 0:
        return None

    # fallback preference order
    for suffix in [f"{horizon}d", "5d", "1d", "21d"]:
        for c in cands:
            if c.endswith(suffix):
                return c
    return cands[0]


def run_market_proxy_buy_and_hold(
    feature_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Market proxy buy-and-hold using mkt_return_* from the feature dataset.
    This is a proxy benchmark derived from the same dataset, not an external SPY download.
    """
    if feature_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()

    market_ret_col = detect_market_return_col(feature_df, horizon)
    if market_ret_col is None:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()

    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # one row per date is enough because market features are duplicated across stocks
    by_date = (
        df.sort_values(["date", "stock"])
          .groupby("date", as_index=False)
          .first()[["date", market_ret_col]]
          .sort_values("date")
          .reset_index(drop=True)
    )
    rebalance_dates = by_date["date"].tolist()[::horizon]
    day_map = by_date.set_index("date")[market_ret_col].to_dict()
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    current_drift_w = {}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []

    for idx, dt in enumerate(rebalance_dates):
        if idx == 0:
            new_w = {"MARKET_PROXY": 1.0}
        else:
            new_w = current_drift_w.copy() if current_drift_w else {"MARKET_PROXY": 1.0}

        gross = float(new_w["MARKET_PROXY"] * float(day_map.get(dt, 0.0)))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))

        rows.append({
            "date": pd.Timestamp(dt),
            "strategy": "market_proxy_buy_hold",
            "selected_stocks": "MARKET_PROXY",
            "n_holdings": 1,
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
            "market_return_col": market_ret_col,
        })

        drift_val = float(new_w["MARKET_PROXY"] * (1.0 + float(day_map.get(dt, 0.0))))
        current_drift_w = {"MARKET_PROXY": 1.0 if drift_val > 0 else 0.0}
        prev_w = new_w

    metrics = compute_metrics_from_series(period_returns, turns, holds, exposures, horizon)
    return metrics, pd.DataFrame(rows)


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def strategy_rank_table(metrics_df: pd.DataFrame, split_col: str = "cost_bps") -> pd.DataFrame:
    rows = []
    for cost_bps, sub in metrics_df.groupby(split_col):
        sub = sub.copy()
        best_cum = sub.sort_values("cumulative_return", ascending=False).iloc[0]
        best_shp = sub.sort_values("sharpe", ascending=False).iloc[0]
        best_mdd = sub.sort_values("max_drawdown", ascending=False).iloc[0]  # closer to 0 is better
        rows.append({
            split_col: cost_bps,
            "best_cumulative_return_strategy": best_cum["strategy"],
            "best_cumulative_return": float(best_cum["cumulative_return"]),
            "best_sharpe_strategy": best_shp["strategy"],
            "best_sharpe": float(best_shp["sharpe"]),
            "best_mdd_strategy": best_mdd["strategy"],
            "best_mdd": float(best_mdd["max_drawdown"]),
        })
    return pd.DataFrame(rows).sort_values(split_col).reset_index(drop=True)


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(
        description="Run buy-and-hold benchmarks and transaction-cost sensitivity on the latest final system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--final_system_manifest_path", type=str, default="")
    parser.add_argument("--predictions_path", type=str, default="")
    parser.add_argument("--feature_data_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(EXECUTION_DIR / "benchmark_cost_sensitivity"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--cost_grid_bps", type=str, default="5,10,20,30")
    parser.add_argument("--include_market_proxy", type=int, default=1)
    args = parser.parse_args()

    final_system_manifest_path = (
        Path(args.final_system_manifest_path)
        if args.final_system_manifest_path
        else resolve_default_final_system_manifest_path()
    )
    final_manifest = safe_read_json(final_system_manifest_path)

    predictions_path = (
        Path(args.predictions_path)
        if args.predictions_path
        else Path(final_manifest.get("source_predictions_path", ""))
    )
    feature_data_path = (
        Path(args.feature_data_path)
        if args.feature_data_path
        else Path(final_manifest.get("source_feature_data_path", str(MAIN_H5_DATA)))
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions path not found: {predictions_path}. "
            "Pass --predictions_path explicitly if needed."
        )
    if not feature_data_path.exists():
        raise FileNotFoundError(
            f"Feature data path not found: {feature_data_path}. "
            "Pass --feature_data_path explicitly if needed."
        )

    print(f"Using final system manifest: {final_system_manifest_path}")
    print(f"Using predictions: {predictions_path}")
    print(f"Using feature data: {feature_data_path}")

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)

    horizon = int(final_manifest.get("horizon", infer_horizon_from_feature_df(feat)))
    frozen_cfg = final_manifest.get("frozen_prediction_layer", {})
    sticky_cfg = final_manifest.get("execution_layer", {})
    default_cost_bps = float(final_manifest.get("transaction_cost_bps", 10.0))
    cost_grid_bps = parse_float_grid(args.cost_grid_bps)

    if not frozen_cfg:
        raise ValueError("final_system_manifest is missing 'frozen_prediction_layer'.")
    if not sticky_cfg:
        raise ValueError("final_system_manifest is missing 'execution_layer'.")

    pred_test = pred[pred["split"] == "test"].copy()
    feat_test = feat[feat["split"] == "test"].copy()

    if pred_test.empty or feat_test.empty:
        raise ValueError("Test split is empty in predictions or feature dataset.")

    # default-cost benchmark comparison
    static_metrics, static_actions = run_static_backtest(
        pred_df=pred_test,
        horizon=horizon,
        top_k=int(frozen_cfg["top_k"]),
        min_prob=float(frozen_cfg["min_prob"]),
        transaction_cost_bps=default_cost_bps,
    )
    sticky_metrics, sticky_actions = run_sticky_backtest(
        pred_df=pred_test,
        horizon=horizon,
        top_k=int(sticky_cfg["top_k"]),
        entry_prob=float(sticky_cfg["entry_prob"]),
        exit_prob=float(sticky_cfg["exit_prob"]),
        switch_buffer=float(sticky_cfg["switch_buffer"]),
        min_hold_periods=int(sticky_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(sticky_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=default_cost_bps,
    )
    ew_bh_metrics, ew_bh_actions = run_equal_weight_buy_and_hold(
        pred_df=pred_test,
        horizon=horizon,
        transaction_cost_bps=default_cost_bps,
    )

    benchmark_rows = [
        {"strategy": "frozen_static_baseline", "cost_bps": default_cost_bps, **static_metrics},
        {"strategy": "optimized_sticky_execution", "cost_bps": default_cost_bps, **sticky_metrics},
        {"strategy": "universe_equal_weight_buy_hold", "cost_bps": default_cost_bps, **ew_bh_metrics},
    ]

    market_proxy_metrics = {}
    market_proxy_actions = pd.DataFrame()
    if int(args.include_market_proxy) == 1:
        market_proxy_metrics, market_proxy_actions = run_market_proxy_buy_and_hold(
            feature_df=feat_test,
            horizon=horizon,
            transaction_cost_bps=default_cost_bps,
        )
        if len(market_proxy_actions) > 0:
            benchmark_rows.append({
                "strategy": "market_proxy_buy_hold",
                "cost_bps": default_cost_bps,
                **market_proxy_metrics,
            })

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(
        ["cumulative_return", "sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # cost sensitivity
    cost_rows = []
    for cost_bps in cost_grid_bps:
        s_metrics, _ = run_static_backtest(
            pred_df=pred_test,
            horizon=horizon,
            top_k=int(frozen_cfg["top_k"]),
            min_prob=float(frozen_cfg["min_prob"]),
            transaction_cost_bps=cost_bps,
        )
        t_metrics, _ = run_sticky_backtest(
            pred_df=pred_test,
            horizon=horizon,
            top_k=int(sticky_cfg["top_k"]),
            entry_prob=float(sticky_cfg["entry_prob"]),
            exit_prob=float(sticky_cfg["exit_prob"]),
            switch_buffer=float(sticky_cfg["switch_buffer"]),
            min_hold_periods=int(sticky_cfg["min_hold_periods"]),
            max_new_names_per_rebalance=int(sticky_cfg["max_new_names_per_rebalance"]),
            transaction_cost_bps=cost_bps,
        )
        b_metrics, _ = run_equal_weight_buy_and_hold(
            pred_df=pred_test,
            horizon=horizon,
            transaction_cost_bps=cost_bps,
        )

        cost_rows.append({"strategy": "frozen_static_baseline", "cost_bps": float(cost_bps), **s_metrics})
        cost_rows.append({"strategy": "optimized_sticky_execution", "cost_bps": float(cost_bps), **t_metrics})
        cost_rows.append({"strategy": "universe_equal_weight_buy_hold", "cost_bps": float(cost_bps), **b_metrics})

        if int(args.include_market_proxy) == 1:
            m_metrics, _ = run_market_proxy_buy_and_hold(
                feature_df=feat_test,
                horizon=horizon,
                transaction_cost_bps=cost_bps,
            )
            if m_metrics:
                cost_rows.append({"strategy": "market_proxy_buy_hold", "cost_bps": float(cost_bps), **m_metrics})

    cost_df = pd.DataFrame(cost_rows).sort_values(["cost_bps", "strategy"]).reset_index(drop=True)
    best_by_cost = strategy_rank_table(cost_df, split_col="cost_bps")

    # summaries
    default_row_sticky = benchmark_df[benchmark_df["strategy"] == "optimized_sticky_execution"].iloc[0].to_dict()
    default_row_static = benchmark_df[benchmark_df["strategy"] == "frozen_static_baseline"].iloc[0].to_dict()
    default_row_ew = benchmark_df[benchmark_df["strategy"] == "universe_equal_weight_buy_hold"].iloc[0].to_dict()
    default_row_market = (
        benchmark_df[benchmark_df["strategy"] == "market_proxy_buy_hold"].iloc[0].to_dict()
        if "market_proxy_buy_hold" in benchmark_df["strategy"].values
        else None
    )

    benchmark_cost_summary = {
        "source_final_system_manifest_path": str(final_system_manifest_path.resolve()),
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "horizon": int(horizon),
        "default_cost_bps": float(default_cost_bps),
        "frozen_static_cfg": frozen_cfg,
        "optimized_sticky_cfg": sticky_cfg,
        "default_cost_comparison": {
            "optimized_sticky_execution": default_row_sticky,
            "frozen_static_baseline": default_row_static,
            "universe_equal_weight_buy_hold": default_row_ew,
            "market_proxy_buy_hold": default_row_market,
        },
        "notes": [
            "Buy-and-hold benchmarks follow the same cost convention as the main backtests: initial entry cost is charged, no terminal liquidation cost is charged.",
            "universe_equal_weight_buy_hold uses the full stock universe present on the first test rebalance date.",
            "market_proxy_buy_hold is derived from mkt_return_* in the feature dataset and should be described as a market proxy benchmark.",
        ],
    }

    cost_sensitivity_summary = {
        "cost_grid_bps": [float(x) for x in cost_grid_bps],
        "best_strategy_by_cost": best_by_cost.to_dict(orient="records"),
        "sticky_minus_static_by_cost": (
            cost_df.pivot(index="cost_bps", columns="strategy", values="cumulative_return")
                  .reset_index()
                  .rename_axis(None, axis=1)
                  .to_dict(orient="records")
            if not cost_df.empty else []
        ),
    }

    # save
    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_df.to_csv(out_root / "benchmark_comparison_default_cost.csv", index=False)
    cost_df.to_csv(out_root / "cost_sensitivity_metrics.csv", index=False)
    best_by_cost.to_csv(out_root / "best_strategy_by_cost.csv", index=False)

    save_df_with_dates(static_actions, out_root / "frozen_static_default_cost_actions.csv")
    save_df_with_dates(sticky_actions, out_root / "optimized_sticky_default_cost_actions.csv")
    save_df_with_dates(ew_bh_actions, out_root / "universe_equal_weight_buy_hold_default_cost_actions.csv")
    if len(market_proxy_actions) > 0:
        save_df_with_dates(market_proxy_actions, out_root / "market_proxy_buy_hold_default_cost_actions.csv")

    with open(out_root / "benchmark_cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_cost_summary, f, ensure_ascii=False, indent=2)

    with open(out_root / "cost_sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(cost_sensitivity_summary, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["resolved_final_system_manifest_path"] = str(final_system_manifest_path.resolve())
    run_config["resolved_predictions_path"] = str(predictions_path.resolve())
    run_config["resolved_feature_data_path"] = str(feature_data_path.resolve())
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Benchmark + cost sensitivity saved to: {out_root}")
    print("\nDefault-cost benchmark comparison:")
    print(benchmark_df.to_string(index=False))
    print("\nBest strategy by cost:")
    print(best_by_cost.to_string(index=False))


if __name__ == "__main__":
    main()
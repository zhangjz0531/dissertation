
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Freeze final optimized Transformer system
#
# Final system definition
# -----------------------
# Base signal source:
#   Transformer v5 fixed predictions
#
# Final execution layer:
#   selected from validation-only execution optimization
#
# This script:
# 1) reads the optimized execution config
# 2) recomputes val / test / all-period actions and metrics
# 3) writes a final frozen manifest for the dissertation
# =========================================================


DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_OPTIMIZATION_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\final_execution_optimization_valonly"


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)


def safe_read_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def annualized_return(final_equity: float, total_days: int) -> float:
    total_days = max(1, int(total_days))
    if final_equity <= 0:
        return -1.0
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_ratio(period_returns: List[float], horizon: int) -> float:
    if len(period_returns) <= 1:
        return 0.0
    arr = np.array(period_returns, dtype=float)
    std = arr.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((arr.mean() / std) * math.sqrt(252.0 / horizon))


def max_drawdown_from_equity_curve(equity_curve: List[float]) -> float:
    eq = np.array(equity_curve, dtype=float)
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

    if "future_return" not in out.columns:
        ret_cols = [c for c in feat.columns if c.startswith("future_return_")]
        if len(ret_cols) != 1:
            raise ValueError("Feature data must contain exactly one future_return_* column.")
        ret_col = ret_cols[0]
        out = out.merge(
            feat[["date", "stock", "split", ret_col]].rename(columns={ret_col: "future_return"}),
            on=["date", "stock", "split"],
            how="left"
        )

    if out["future_return"].isna().any():
        raise ValueError("Predictions data contains missing future_return after normalization.")
    return out


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
    new_candidates = [s for s in day["stock"].tolist() if s not in current_set and float(prob_map.get(s, -1.0)) >= entry_prob]

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

    portfolio = sorted(list(set(portfolio)), key=lambda x: float(prob_map.get(x, -1.0)), reverse=True)[:top_k]

    debug = {
        "kept_existing": len([s for s in portfolio if s in current_set]),
        "newly_added": len([s for s in portfolio if s not in current_set]),
        "forced_keep_count": len([s for s in portfolio if s in forced_keep]),
        "entry_candidates_count": len(new_candidates),
    }
    return portfolio, debug


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
        }, pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    current_positions: Dict[str, Dict] = {}
    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    holds = []
    exposures = []
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

        top1_stock = day.iloc[0]["stock"] if len(day) > 0 else ""
        top1_prob = float(day.iloc[0]["pred_prob"]) if len(day) > 0 else np.nan

        rows.append({
            "date": pd.Timestamp(dt),
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "top1_stock": top1_stock,
            "top1_prob": top1_prob,
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

    metrics = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def latest_snapshot(pred_df: pd.DataFrame, feat_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    latest_date = pd.to_datetime(pred_df["date"]).max()
    latest_pred = pred_df[pred_df["date"] == latest_date].copy()
    latest_feat = feat_df.copy()
    latest_feat["date"] = pd.to_datetime(latest_feat["date"])
    latest_feat = latest_feat[latest_feat["date"] == latest_date].copy()

    cols = ["date", "stock"]
    for c in ["dc_trend", "mkt_dc_trend", "vix_z_60", "credit_stress"]:
        if c in latest_feat.columns:
            cols.append(c)

    latest = latest_pred.merge(latest_feat[cols], on=["date", "stock"], how="left")
    latest = latest.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    selected_names, _ = choose_portfolio_sticky(
        day_df=latest,
        current_positions={},
        top_k=int(cfg["top_k"]),
        entry_prob=float(cfg["entry_prob"]),
        exit_prob=float(cfg["exit_prob"]),
        switch_buffer=float(cfg["switch_buffer"]),
        min_hold_periods=int(cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(cfg["max_new_names_per_rebalance"]),
    )
    if len(selected_names) == 0:
        weights = {}
    else:
        w = 1.0 / len(selected_names)
        weights = {s: w for s in selected_names}

    latest["selected_in_final"] = latest["stock"].map(lambda s: 1 if s in weights else 0)
    latest["final_weight"] = latest["stock"].map(lambda s: float(weights.get(s, 0.0)))
    return latest


def compare_old_vs_new(
    pred_df: pd.DataFrame,
    horizon: int,
    old_cfg: Dict,
    new_cfg: Dict,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    rows = []
    for label, cfg in [("previous_frozen", old_cfg), ("final_optimized", new_cfg)]:
        for split_name in ["val", "test"]:
            split_df = pred_df[pred_df["split"] == split_name].copy()

            if label == "previous_frozen":
                metrics, _ = run_sticky_backtest(
                    split_df, horizon,
                    top_k=int(cfg["top_k"]),
                    entry_prob=float(cfg["min_prob"]),
                    exit_prob=float(cfg["min_prob"]),
                    switch_buffer=0.0,
                    min_hold_periods=0,
                    max_new_names_per_rebalance=int(cfg["top_k"]),
                    transaction_cost_bps=transaction_cost_bps,
                )
            else:
                metrics, _ = run_sticky_backtest(
                    split_df, horizon,
                    top_k=int(cfg["top_k"]),
                    entry_prob=float(cfg["entry_prob"]),
                    exit_prob=float(cfg["exit_prob"]),
                    switch_buffer=float(cfg["switch_buffer"]),
                    min_hold_periods=int(cfg["min_hold_periods"]),
                    max_new_names_per_rebalance=int(cfg["max_new_names_per_rebalance"]),
                    transaction_cost_bps=transaction_cost_bps,
                )

            rows.append({
                "config_label": label,
                "split": split_name,
                **metrics,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Freeze final optimized Transformer system.")
    parser.add_argument("--predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--optimization_dir", type=str, default=DEFAULT_OPTIMIZATION_DIR)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    optimization_dir = Path(args.optimization_dir)
    if not optimization_dir.exists():
        raise FileNotFoundError(f"Missing optimization dir: {optimization_dir}")

    pred = safe_read_csv(args.predictions_path)
    feat = safe_read_csv(args.feature_data_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    opt_summary = safe_read_json(str(optimization_dir / "optimization_summary.json"))
    final_cfg = opt_summary["optimized_execution_cfg"]

    previous_frozen_cfg = {
        "mode": "topk",
        "top_k": 3,
        "min_prob": 0.54,
        "threshold": 0.50,
    }

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()
    all_df = pred[pred["split"].isin(["train", "val", "test"])].copy()

    final_val_metrics, final_val_actions = run_sticky_backtest(
        val_df, horizon,
        top_k=int(final_cfg["top_k"]),
        entry_prob=float(final_cfg["entry_prob"]),
        exit_prob=float(final_cfg["exit_prob"]),
        switch_buffer=float(final_cfg["switch_buffer"]),
        min_hold_periods=int(final_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(final_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
    )
    final_test_metrics, final_test_actions = run_sticky_backtest(
        test_df, horizon,
        top_k=int(final_cfg["top_k"]),
        entry_prob=float(final_cfg["entry_prob"]),
        exit_prob=float(final_cfg["exit_prob"]),
        switch_buffer=float(final_cfg["switch_buffer"]),
        min_hold_periods=int(final_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(final_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
    )
    final_all_metrics, final_all_actions = run_sticky_backtest(
        all_df, horizon,
        top_k=int(final_cfg["top_k"]),
        entry_prob=float(final_cfg["entry_prob"]),
        exit_prob=float(final_cfg["exit_prob"]),
        switch_buffer=float(final_cfg["switch_buffer"]),
        min_hold_periods=int(final_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(final_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
    )

    out_root = Path(args.out_dir) / "final_transformer_optimized_system"
    out_root.mkdir(parents=True, exist_ok=True)

    final_val_actions.to_csv(out_root / "final_val_actions.csv", index=False)
    final_test_actions.to_csv(out_root / "final_test_actions.csv", index=False)
    final_all_actions.to_csv(out_root / "final_all_actions.csv", index=False)

    compare_df = compare_old_vs_new(pred, horizon, previous_frozen_cfg, final_cfg, args.transaction_cost_bps)
    compare_df.to_csv(out_root / "optimized_vs_previous_frozen.csv", index=False)

    latest_df = latest_snapshot(pred, feat, final_cfg)
    latest_df.to_csv(out_root / "final_snapshot_latest.csv", index=False)

    manifest = {
        "final_system_name": "H5 Transformer v5 final optimized deployable system",
        "final_decision": "freeze_validation_only_optimized_execution",
        "base_signal_source": "Transformer v5 fixed predictions",
        "execution_layer": final_cfg,
        "replaced_previous_frozen_config": previous_frozen_cfg,
        "transaction_cost_bps": args.transaction_cost_bps,
        "horizon": horizon,
        "optimization_dir": str(optimization_dir),
        "selection_policy": opt_summary.get("selection_policy", "validation_only"),
    }
    with open(out_root / "final_system_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    metrics_summary = {
        "final_execution_cfg": final_cfg,
        "validation_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "all_period_metrics": final_all_metrics,
    }
    with open(out_root / "final_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"Final optimized system frozen to: {out_root}")
    print("\nFinal execution config:")
    print(json.dumps(final_cfg, ensure_ascii=False, indent=2))
    print("\nValidation metrics:")
    print(json.dumps(final_val_metrics, ensure_ascii=False, indent=2))
    print("\nTest metrics:")
    print(json.dumps(final_test_metrics, ensure_ascii=False, indent=2))
    print("\nAll-period metrics:")
    print(json.dumps(final_all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

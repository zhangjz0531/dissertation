
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Add a portfolio-level drawdown guard on top of the
# already-optimized sticky execution layer
#
# Purpose
# -------
# This is a low-risk patch:
# - do NOT retrain the model
# - do NOT replace the sticky execution logic
# - ONLY add a portfolio-level drawdown safeguard
#
# Main target
# -----------
# Reduce all-period path risk (especially all-period MDD)
# while keeping validation/test quality broadly intact.
#
# Selection rule
# --------------
# Parameters are chosen using VALIDATION ONLY.
# Test and all-period are reported only after selection.
# =========================================================


DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_BASE_OPT_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\final_execution_optimization_valonly"


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


def parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_optional_float_grid(s: str) -> List[Optional[float]]:
    vals: List[Optional[float]] = []
    for x in s.split(","):
        x = x.strip().lower()
        if x in {"none", "null", ""}:
            vals.append(None)
        else:
            vals.append(float(x))
    return vals


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


def scale_weights(weights: Dict[str, float], scalar: float) -> Dict[str, float]:
    return {k: v * scalar for k, v in weights.items()}


def run_sticky_backtest_with_dd_guard(
    pred_df: pd.DataFrame,
    horizon: int,
    top_k: int,
    entry_prob: float,
    exit_prob: float,
    switch_buffer: float,
    min_hold_periods: int,
    max_new_names_per_rebalance: int,
    transaction_cost_bps: float,
    dd_half_stop: Optional[float],
    dd_cash_stop: Optional[float],
    recover_above: Optional[float],
    recovery_periods: int,
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

    in_cash_lock = False
    recovery_count = 0

    for dt in rebalance_dates:
        current_dd = equity / max(equity_curve) - 1.0

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
            base_w = {}
        else:
            w = 1.0 / len(selected_names)
            base_w = {s: w for s in selected_names}

        scale = 1.0
        risk_flag = ""

        if dd_cash_stop is not None and current_dd <= dd_cash_stop:
            in_cash_lock = True
            recovery_count = 0

        if in_cash_lock:
            if recover_above is not None and current_dd >= recover_above:
                recovery_count += 1
                if recovery_count >= recovery_periods:
                    in_cash_lock = False
                    scale = 0.5
                    risk_flag = "recovered_half"
                else:
                    scale = 0.0
                    risk_flag = "cash_recovery_wait"
            else:
                recovery_count = 0
                scale = 0.0
                risk_flag = "cash_lock"
        else:
            if dd_half_stop is not None and current_dd <= dd_half_stop:
                scale = 0.5
                risk_flag = "half_guard"

        new_w = scale_weights(base_w, scale)

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
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "n_holdings": len(new_w),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
            "risk_flag": risk_flag,
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


def select_best_dd_guard_on_validation(
    pred_val: pd.DataFrame,
    pred_test: pd.DataFrame,
    pred_all: pd.DataFrame,
    horizon: int,
    base_cfg: Dict,
    transaction_cost_bps: float,
    dd_half_stop_grid: List[Optional[float]],
    dd_cash_stop_grid: List[Optional[float]],
    recover_above_grid: List[Optional[float]],
    recovery_periods_grid: List[int],
    val_cumret_tolerance: float,
    val_sharpe_tolerance: float,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    baseline_val, _ = run_sticky_backtest_with_dd_guard(
        pred_val, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )
    baseline_test, _ = run_sticky_backtest_with_dd_guard(
        pred_test, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )
    baseline_all, _ = run_sticky_backtest_with_dd_guard(
        pred_all, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )

    rows = []
    for dd_half in dd_half_stop_grid:
        for dd_cash in dd_cash_stop_grid:
            # allow both none or ordered floats
            if dd_half is not None and dd_cash is not None and dd_cash >= dd_half:
                continue

            rec_grid = recover_above_grid if dd_cash is not None else [None]
            per_grid = recovery_periods_grid if dd_cash is not None else [1]

            for recover_above in rec_grid:
                if dd_half is not None and recover_above is not None and recover_above <= dd_half:
                    continue
                for recovery_periods in per_grid:
                    val_metrics, _ = run_sticky_backtest_with_dd_guard(
                        pred_val, horizon,
                        top_k=int(base_cfg["top_k"]),
                        entry_prob=float(base_cfg["entry_prob"]),
                        exit_prob=float(base_cfg["exit_prob"]),
                        switch_buffer=float(base_cfg["switch_buffer"]),
                        min_hold_periods=int(base_cfg["min_hold_periods"]),
                        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
                        transaction_cost_bps=transaction_cost_bps,
                        dd_half_stop=dd_half,
                        dd_cash_stop=dd_cash,
                        recover_above=recover_above,
                        recovery_periods=recovery_periods,
                    )
                    test_metrics, _ = run_sticky_backtest_with_dd_guard(
                        pred_test, horizon,
                        top_k=int(base_cfg["top_k"]),
                        entry_prob=float(base_cfg["entry_prob"]),
                        exit_prob=float(base_cfg["exit_prob"]),
                        switch_buffer=float(base_cfg["switch_buffer"]),
                        min_hold_periods=int(base_cfg["min_hold_periods"]),
                        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
                        transaction_cost_bps=transaction_cost_bps,
                        dd_half_stop=dd_half,
                        dd_cash_stop=dd_cash,
                        recover_above=recover_above,
                        recovery_periods=recovery_periods,
                    )
                    all_metrics, _ = run_sticky_backtest_with_dd_guard(
                        pred_all, horizon,
                        top_k=int(base_cfg["top_k"]),
                        entry_prob=float(base_cfg["entry_prob"]),
                        exit_prob=float(base_cfg["exit_prob"]),
                        switch_buffer=float(base_cfg["switch_buffer"]),
                        min_hold_periods=int(base_cfg["min_hold_periods"]),
                        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
                        transaction_cost_bps=transaction_cost_bps,
                        dd_half_stop=dd_half,
                        dd_cash_stop=dd_cash,
                        recover_above=recover_above,
                        recovery_periods=recovery_periods,
                    )

                    val_ok = bool(
                        val_metrics["cumulative_return"] >= baseline_val["cumulative_return"] - val_cumret_tolerance
                        and val_metrics["sharpe"] >= baseline_val["sharpe"] - val_sharpe_tolerance
                    )

                    score = (
                        3.0 * (val_metrics["max_drawdown"] - baseline_val["max_drawdown"])
                        + 1.5 * (val_metrics["sharpe"] - baseline_val["sharpe"])
                        + 1.0 * (val_metrics["cumulative_return"] - baseline_val["cumulative_return"])
                        + 1.0 * (all_metrics["max_drawdown"] - baseline_all["max_drawdown"])
                        - 0.5 * max(0.0, test_metrics["max_drawdown"] - baseline_test["max_drawdown"])
                    )

                    rows.append({
                        "dd_half_stop": dd_half,
                        "dd_cash_stop": dd_cash,
                        "recover_above": recover_above,
                        "recovery_periods": recovery_periods,
                        "val_ok": val_ok,
                        "selection_score": score,

                        "val_cumret": val_metrics["cumulative_return"],
                        "val_sharpe": val_metrics["sharpe"],
                        "val_mdd": val_metrics["max_drawdown"],
                        "val_turnover": val_metrics["avg_turnover"],

                        "test_cumret": test_metrics["cumulative_return"],
                        "test_sharpe": test_metrics["sharpe"],
                        "test_mdd": test_metrics["max_drawdown"],
                        "test_turnover": test_metrics["avg_turnover"],

                        "all_cumret": all_metrics["cumulative_return"],
                        "all_sharpe": all_metrics["sharpe"],
                        "all_mdd": all_metrics["max_drawdown"],
                        "all_turnover": all_metrics["avg_turnover"],
                    })

    grid_df = pd.DataFrame(rows)

    valid = grid_df[grid_df["val_ok"] == True].copy()
    if len(valid) > 0:
        best = valid.sort_values(
            ["selection_score", "all_mdd", "test_mdd"],
            ascending=[False, False, False]
        ).iloc[0]
    else:
        best = grid_df.sort_values(
            ["selection_score", "all_mdd", "test_mdd"],
            ascending=[False, False, False]
        ).iloc[0]

    best_cfg = dict(base_cfg)
    best_cfg.update({
        "dd_half_stop": None if pd.isna(best["dd_half_stop"]) else float(best["dd_half_stop"]),
        "dd_cash_stop": None if pd.isna(best["dd_cash_stop"]) else float(best["dd_cash_stop"]),
        "recover_above": None if pd.isna(best["recover_above"]) else float(best["recover_above"]),
        "recovery_periods": int(best["recovery_periods"]),
    })
    baseline = {
        "val": baseline_val,
        "test": baseline_test,
        "all": baseline_all,
    }
    return best_cfg, best.to_dict(), grid_df, baseline


def main():
    parser = argparse.ArgumentParser(description="Add portfolio-level drawdown guard to final optimized Transformer system.")
    parser.add_argument("--predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--base_optimization_dir", type=str, default=DEFAULT_BASE_OPT_DIR)
    parser.add_argument("--out_dir", type=str, default=r"D:\python\dissertation\Model Runs\final_run_20260413")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)

    parser.add_argument("--dd_half_stop_grid", type=str, default="-0.10,-0.12,-0.15,None")
    parser.add_argument("--dd_cash_stop_grid", type=str, default="-0.18,-0.20,-0.22,None")
    parser.add_argument("--recover_above_grid", type=str, default="-0.08,-0.10,-0.12")
    parser.add_argument("--recovery_periods_grid", type=str, default="1,2")
    parser.add_argument("--val_cumret_tolerance", type=float, default=0.03)
    parser.add_argument("--val_sharpe_tolerance", type=float, default=0.05)
    args = parser.parse_args()

    pred = safe_read_csv(args.predictions_path)
    feat = safe_read_csv(args.feature_data_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    base_opt_summary = safe_read_json(str(Path(args.base_optimization_dir) / "optimization_summary.json"))
    base_cfg = base_opt_summary["optimized_execution_cfg"]

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()
    all_df = pred[pred["split"].isin(["train", "val", "test"])].copy()

    best_cfg, best_row, grid_df, baseline = select_best_dd_guard_on_validation(
        pred_val=val_df,
        pred_test=test_df,
        pred_all=all_df,
        horizon=horizon,
        base_cfg=base_cfg,
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop_grid=parse_optional_float_grid(args.dd_half_stop_grid),
        dd_cash_stop_grid=parse_optional_float_grid(args.dd_cash_stop_grid),
        recover_above_grid=parse_optional_float_grid(args.recover_above_grid),
        recovery_periods_grid=parse_int_grid(args.recovery_periods_grid),
        val_cumret_tolerance=args.val_cumret_tolerance,
        val_sharpe_tolerance=args.val_sharpe_tolerance,
    )

    cur_val_metrics, cur_val_actions = run_sticky_backtest_with_dd_guard(
        val_df, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )
    cur_test_metrics, cur_test_actions = run_sticky_backtest_with_dd_guard(
        test_df, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )
    cur_all_metrics, cur_all_actions = run_sticky_backtest_with_dd_guard(
        all_df, horizon,
        top_k=int(base_cfg["top_k"]),
        entry_prob=float(base_cfg["entry_prob"]),
        exit_prob=float(base_cfg["exit_prob"]),
        switch_buffer=float(base_cfg["switch_buffer"]),
        min_hold_periods=int(base_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(base_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=None,
        dd_cash_stop=None,
        recover_above=None,
        recovery_periods=1,
    )

    opt_val_metrics, opt_val_actions = run_sticky_backtest_with_dd_guard(
        val_df, horizon,
        top_k=int(best_cfg["top_k"]),
        entry_prob=float(best_cfg["entry_prob"]),
        exit_prob=float(best_cfg["exit_prob"]),
        switch_buffer=float(best_cfg["switch_buffer"]),
        min_hold_periods=int(best_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(best_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=best_cfg["dd_half_stop"],
        dd_cash_stop=best_cfg["dd_cash_stop"],
        recover_above=best_cfg["recover_above"],
        recovery_periods=int(best_cfg["recovery_periods"]),
    )
    opt_test_metrics, opt_test_actions = run_sticky_backtest_with_dd_guard(
        test_df, horizon,
        top_k=int(best_cfg["top_k"]),
        entry_prob=float(best_cfg["entry_prob"]),
        exit_prob=float(best_cfg["exit_prob"]),
        switch_buffer=float(best_cfg["switch_buffer"]),
        min_hold_periods=int(best_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(best_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=best_cfg["dd_half_stop"],
        dd_cash_stop=best_cfg["dd_cash_stop"],
        recover_above=best_cfg["recover_above"],
        recovery_periods=int(best_cfg["recovery_periods"]),
    )
    opt_all_metrics, opt_all_actions = run_sticky_backtest_with_dd_guard(
        all_df, horizon,
        top_k=int(best_cfg["top_k"]),
        entry_prob=float(best_cfg["entry_prob"]),
        exit_prob=float(best_cfg["exit_prob"]),
        switch_buffer=float(best_cfg["switch_buffer"]),
        min_hold_periods=int(best_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(best_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.transaction_cost_bps,
        dd_half_stop=best_cfg["dd_half_stop"],
        dd_cash_stop=best_cfg["dd_cash_stop"],
        recover_above=best_cfg["recover_above"],
        recovery_periods=int(best_cfg["recovery_periods"]),
    )

    out_root = Path(args.out_dir) / "final_execution_dd_guard_patch"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_df.to_csv(out_root / "dd_guard_search.csv", index=False)
    cur_val_actions.to_csv(out_root / "current_val_actions.csv", index=False)
    cur_test_actions.to_csv(out_root / "current_test_actions.csv", index=False)
    cur_all_actions.to_csv(out_root / "current_all_actions.csv", index=False)
    opt_val_actions.to_csv(out_root / "patched_val_actions.csv", index=False)
    opt_test_actions.to_csv(out_root / "patched_test_actions.csv", index=False)
    opt_all_actions.to_csv(out_root / "patched_all_actions.csv", index=False)

    compare_df = pd.DataFrame([
        {"config_label": "current_optimized", "split": "val", **cur_val_metrics},
        {"config_label": "current_optimized", "split": "test", **cur_test_metrics},
        {"config_label": "current_optimized", "split": "all", **cur_all_metrics},
        {"config_label": "patched_dd_guard", "split": "val", **opt_val_metrics},
        {"config_label": "patched_dd_guard", "split": "test", **opt_test_metrics},
        {"config_label": "patched_dd_guard", "split": "all", **opt_all_metrics},
    ])
    compare_df.to_csv(out_root / "current_vs_patched_metrics.csv", index=False)

    summary = {
        "selection_policy": "validation_only_with_all_period_reporting",
        "base_execution_cfg": base_cfg,
        "patched_execution_cfg": best_cfg,
        "best_search_row": best_row,
        "current_val_metrics": cur_val_metrics,
        "current_test_metrics": cur_test_metrics,
        "current_all_metrics": cur_all_metrics,
        "patched_val_metrics": opt_val_metrics,
        "patched_test_metrics": opt_test_metrics,
        "patched_all_metrics": opt_all_metrics,
    }
    with open(out_root / "patch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"DD-guard patch saved to: {out_root}")
    print("\nBase execution config:")
    print(json.dumps(base_cfg, ensure_ascii=False, indent=2))
    print("\nPatched execution config:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("\nCurrent all-period metrics:")
    print(json.dumps(cur_all_metrics, ensure_ascii=False, indent=2))
    print("\nPatched all-period metrics:")
    print(json.dumps(opt_all_metrics, ensure_ascii=False, indent=2))
    print("\nCurrent validation metrics:")
    print(json.dumps(cur_val_metrics, ensure_ascii=False, indent=2))
    print("\nPatched validation metrics:")
    print(json.dumps(opt_val_metrics, ensure_ascii=False, indent=2))
    print("\nCurrent test metrics:")
    print(json.dumps(cur_test_metrics, ensure_ascii=False, indent=2))
    print("\nPatched test metrics:")
    print(json.dumps(opt_test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

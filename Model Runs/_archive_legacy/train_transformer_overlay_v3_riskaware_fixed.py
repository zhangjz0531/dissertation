
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Transformer risk-aware overlay (v3)
#
# Purpose
# -------
# This script builds a practical, risk-aware execution overlay on top of
# Transformer predictions.
#
# Main ideas
# ----------
# 1) Start from the base Transformer strategy signals (top-k + min_prob).
# 2) Add explicit transaction-cost-aware backtesting.
# 3) Add risk-aware exposure control:
#      - full exposure
#      - half exposure
#      - cash
# 4) Add drawdown kill-switch logic.
# 5) Select parameters on validation using excess-return + risk penalties.
#
# This is designed to be more defensible in a dissertation than an
# unconstrained aggressive overlay.
#
# Typical workflow
# ----------------
# Step 1: rebuild Transformer predictions (existing script)
#
# python rebuild_full_split_predictions.py ^
#   --data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --checkpoint_path "D:\python\dissertation\Model Runs\final_run_20260413\main_experiment_h5_v5_transformer_seq30_lr0.0003_wd0.0005_do0.2\transformer_best_strategy_model.pt" ^
#   --model_type transformer ^
#   --out_dir "D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5"
#
# Step 2: run this overlay
#
# python train_transformer_overlay_v3_riskaware.py ^
#   --predictions_path "D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv" ^
#   --feature_data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs\final_run_20260413"
# =========================================================


# -----------------------------
# Helpers
# -----------------------------
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


def infer_horizon_from_feature_path(feature_data_path: str) -> int:
    name = Path(feature_data_path).stem.lower()
    if "h1" in name:
        return 1
    if "h5" in name:
        return 5
    return 5


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)


def parse_optional_float_grid(grid_str: str) -> List[Optional[float]]:
    vals = []
    for x in grid_str.split(","):
        x = x.strip().lower()
        if x in {"none", "null", ""}:
            vals.append(None)
        else:
            vals.append(float(x))
    return vals


def parse_bool_grid(grid_str: str) -> List[bool]:
    vals = []
    for x in grid_str.split(","):
        x = x.strip().lower()
        if x in {"1", "true", "t", "yes", "y"}:
            vals.append(True)
        elif x in {"0", "false", "f", "no", "n"}:
            vals.append(False)
        else:
            raise ValueError(f"Cannot parse boolean grid value: {x}")
    return vals


# -----------------------------
# Data preparation
# -----------------------------
def load_and_merge_inputs(predictions_path: str, feature_data_path: str) -> pd.DataFrame:
    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)

    pred["date"] = pd.to_datetime(pred["date"])
    feat["date"] = pd.to_datetime(feat["date"])

    required_pred_cols = {"date", "stock", "split"}
    if not required_pred_cols.issubset(pred.columns):
        raise ValueError(f"Predictions file must contain at least {required_pred_cols}")

    # support both pred_prob and probability naming
    if "pred_prob" not in pred.columns:
        cand = [c for c in pred.columns if "prob" in c.lower()]
        if len(cand) == 0:
            raise ValueError("Predictions file must contain 'pred_prob' or a probability-like column.")
        pred = pred.rename(columns={cand[0]: "pred_prob"})

    return_cols = [c for c in feat.columns if c.startswith("future_return_")]
    if len(return_cols) != 1:
        raise ValueError("Feature dataset must contain exactly one future_return_* column.")
    return_col = return_cols[0]

    merge_cols = ["date", "stock", "split", return_col]
    for c in ["mkt_dc_trend", "dc_trend", "vix_z_60", "credit_stress", "mkt_return_5d", "mkt_return_1d"]:
        if c in feat.columns:
            merge_cols.append(c)

    feat_small = feat[merge_cols].copy()
    feat_small = feat_small.drop_duplicates(["date", "stock", "split"])

    # Avoid duplicating future_return if predictions file already carries it.
    if "future_return" in pred.columns:
        pred_ret = pred[["date", "stock", "split", "future_return"]].copy()
        feat_ret = feat_small[["date", "stock", "split", return_col]].rename(columns={return_col: "future_return_from_feat"}).copy()

        ret_check = pred_ret.merge(feat_ret, on=["date", "stock", "split"], how="left")
        if ret_check["future_return_from_feat"].isna().any():
            raise ValueError("Feature-side future_return has missing values after merge. Check date/stock alignment.")

        diff = (ret_check["future_return"].astype(float) - ret_check["future_return_from_feat"].astype(float)).abs()
        max_diff = float(diff.max()) if len(diff) > 0 else 0.0
        if max_diff > 1e-9:
            print(f"Warning: prediction-side future_return and feature-side {return_col} differ. max_abs_diff={max_diff:.12f}. Using prediction-side future_return.")

        feat_small = feat_small.drop(columns=[return_col])
        merged = pred.merge(feat_small, on=["date", "stock", "split"], how="left")
    else:
        merged = pred.merge(feat_small, on=["date", "stock", "split"], how="left")
        merged = merged.rename(columns={return_col: "future_return"})

    if "future_return" not in merged.columns:
        raise ValueError("Merged data does not contain future_return after alignment.")
    if merged["future_return"].isna().any():
        raise ValueError("Merged future_return contains missing values. Check date/stock alignment.")

    return merged


# -----------------------------
# Base strategy
# -----------------------------
def compute_base_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day_df = day_df.sort_values("pred_prob", ascending=False).copy()

    if mode == "topk":
        chosen = day_df[day_df["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day_df[day_df["pred_prob"] >= threshold].copy()
    else:
        raise ValueError(f"Unknown base mode: {mode}")

    if len(chosen) == 0:
        return {}

    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def backtest_base_strategy(
    df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    actions = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        new_w = compute_base_weights(day, mode=mode, top_k=top_k, min_prob=min_prob, threshold=threshold)

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)

        chosen = sorted(new_w.keys())
        actions.append({
            "date": pd.Timestamp(dt),
            "action": "base",
            "exposure": float(sum(new_w.values())),
            "selected_stocks": "|".join(chosen),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
        })
        prev_w = new_w

    metrics = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_action_exposure_proxy": float(np.mean([a["exposure"] for a in actions])) if actions else 0.0,
    }
    return metrics, pd.DataFrame(actions)


# -----------------------------
# Risk-aware overlay
# -----------------------------
def compute_overlay_weights_for_day(
    day_df: pd.DataFrame,
    confidence_margin: float,
    use_align_filter: bool,
    vix_z_max: Optional[float],
    credit_stress_max: Optional[float],
    dd_half_stop: float,
    dd_cash_stop: float,
    current_drawdown: float,
    min_prob_for_entry: float,
) -> Tuple[Dict[str, float], Dict]:
    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)

    top1 = day.iloc[0]
    top2 = day.iloc[1] if len(day) > 1 else day.iloc[0]
    top3 = day.iloc[2] if len(day) > 2 else day.iloc[min(1, len(day) - 1)]

    top1_prob = float(top1["pred_prob"])
    top2_prob = float(top2["pred_prob"])
    top3_prob = float(top3["pred_prob"])
    score_gap_23 = float(top2_prob - top3_prob)

    confidence_ok = bool(top1_prob >= min_prob_for_entry and top2_prob >= min_prob_for_entry and score_gap_23 >= confidence_margin)

    regime_ok = True
    risk_flags = []

    if use_align_filter and {"mkt_dc_trend", "dc_trend"}.issubset(day.columns):
        mdc = float(top1["mkt_dc_trend"]) if pd.notna(top1.get("mkt_dc_trend")) else np.nan
        tdc = float(top1["dc_trend"]) if pd.notna(top1.get("dc_trend")) else np.nan
        if np.isfinite(mdc) and np.isfinite(tdc):
            if not (mdc * tdc > 0):
                regime_ok = False
                risk_flags.append("dc_misaligned")

    if vix_z_max is not None and "vix_z_60" in day.columns and pd.notna(top1.get("vix_z_60")):
        if float(top1["vix_z_60"]) > vix_z_max:
            regime_ok = False
            risk_flags.append("high_vix")

    if credit_stress_max is not None and "credit_stress" in day.columns and pd.notna(top1.get("credit_stress")):
        if float(top1["credit_stress"]) > credit_stress_max:
            regime_ok = False
            risk_flags.append("credit_stress")

    # base desired exposure from risk gates
    if not confidence_ok:
        exposure = 0.0
        state = "cash"
    elif confidence_ok and regime_ok:
        exposure = 1.0
        state = "top2_full"
    else:
        exposure = 0.5
        state = "top2_half"

    # drawdown kill-switch
    if current_drawdown <= dd_cash_stop:
        exposure = 0.0
        state = "cash_dd_kill"
        risk_flags.append("dd_cash_stop")
    elif current_drawdown <= dd_half_stop:
        exposure = min(exposure, 0.5)
        if exposure > 0:
            state = "top2_half_dd_guard"
        risk_flags.append("dd_half_stop")

    if exposure <= 0:
        weights = {}
    else:
        chosen = day.head(2)
        w_each = exposure / len(chosen)
        weights = {row["stock"]: w_each for _, row in chosen.iterrows()}

    info = {
        "state": state,
        "top1_stock": str(top1["stock"]),
        "top2_stock": str(top2["stock"]),
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "top3_prob": top3_prob,
        "score_gap_23": score_gap_23,
        "confidence_ok": confidence_ok,
        "regime_ok": regime_ok,
        "risk_flags": "|".join(risk_flags) if risk_flags else "",
        "exposure": exposure,
    }
    return weights, info


def backtest_overlay_strategy(
    df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    confidence_margin: float,
    use_align_filter: bool,
    vix_z_max: Optional[float],
    credit_stress_max: Optional[float],
    dd_half_stop: float,
    dd_cash_stop: float,
    min_prob_for_entry: float,
) -> Tuple[Dict, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    actions = []

    for dt in rebalance_dates:
        current_drawdown = equity / max(equity_curve) - 1.0
        day = df[df["date"] == dt].copy()

        new_w, info = compute_overlay_weights_for_day(
            day_df=day,
            confidence_margin=confidence_margin,
            use_align_filter=use_align_filter,
            vix_z_max=vix_z_max,
            credit_stress_max=credit_stress_max,
            dd_half_stop=dd_half_stop,
            dd_cash_stop=dd_cash_stop,
            current_drawdown=current_drawdown,
            min_prob_for_entry=min_prob_for_entry,
        )

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - (transaction_cost_bps / 10000.0) * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)

        actions.append({
            "date": pd.Timestamp(dt),
            "action": info["state"],
            "top1_stock": info["top1_stock"],
            "top2_stock": info["top2_stock"],
            "top1_prob": info["top1_prob"],
            "top2_prob": info["top2_prob"],
            "top3_prob": info["top3_prob"],
            "score_gap_23": info["score_gap_23"],
            "confidence_ok": info["confidence_ok"],
            "regime_ok": info["regime_ok"],
            "risk_flags": info["risk_flags"],
            "exposure": info["exposure"],
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
        })

        prev_w = new_w

    metrics = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_action_exposure_proxy": float(np.mean([a["exposure"] for a in actions])) if actions else 0.0,
    }
    return metrics, pd.DataFrame(actions)


# -----------------------------
# Validation search
# -----------------------------
def select_overlay_params_on_validation(
    val_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    base_metrics: Dict,
    confidence_margin_grid: List[float],
    use_align_filter_grid: List[bool],
    vix_z_max_grid: List[Optional[float]],
    credit_quantile_grid: List[Optional[float]],
    dd_half_stop_grid: List[float],
    dd_cash_stop_grid: List[float],
    min_prob_for_entry_grid: List[float],
    score_weights: Dict[str, float],
) -> Tuple[Dict, Dict, pd.DataFrame]:
    rows = []

    credit_values = val_df["credit_stress"].dropna().values.astype(float) if "credit_stress" in val_df.columns else np.array([])

    for cm in confidence_margin_grid:
        for align in use_align_filter_grid:
            for vix_max in vix_z_max_grid:
                for cq in credit_quantile_grid:
                    if cq is None or len(credit_values) == 0:
                        credit_max = None
                    else:
                        credit_max = float(np.quantile(credit_values, cq))
                    for dd_half in dd_half_stop_grid:
                        for dd_cash in dd_cash_stop_grid:
                            if dd_cash >= dd_half:
                                continue
                            for mp in min_prob_for_entry_grid:
                                overlay_metrics, _ = backtest_overlay_strategy(
                                    df=val_df,
                                    horizon=horizon,
                                    transaction_cost_bps=transaction_cost_bps,
                                    confidence_margin=cm,
                                    use_align_filter=align,
                                    vix_z_max=vix_max,
                                    credit_stress_max=credit_max,
                                    dd_half_stop=dd_half,
                                    dd_cash_stop=dd_cash,
                                    min_prob_for_entry=mp,
                                )
                                excess_cumret = overlay_metrics["cumulative_return"] - base_metrics["cumulative_return"]
                                excess_sharpe = overlay_metrics["sharpe"] - base_metrics["sharpe"]

                                score = (
                                    score_weights["excess_cumret"] * excess_cumret
                                    + score_weights["excess_sharpe"] * excess_sharpe
                                    - score_weights["mdd_penalty"] * max(0.0, abs(overlay_metrics["max_drawdown"]) - score_weights["mdd_limit"])
                                    - score_weights["turnover_penalty"] * overlay_metrics["avg_turnover"]
                                )

                                rows.append({
                                    "confidence_margin": cm,
                                    "use_align_filter": align,
                                    "vix_z_max": vix_max,
                                    "credit_quantile": cq,
                                    "credit_stress_max": credit_max,
                                    "dd_half_stop": dd_half,
                                    "dd_cash_stop": dd_cash,
                                    "min_prob_for_entry": mp,
                                    "overlay_cumulative_return": overlay_metrics["cumulative_return"],
                                    "overlay_sharpe": overlay_metrics["sharpe"],
                                    "overlay_max_drawdown": overlay_metrics["max_drawdown"],
                                    "overlay_avg_turnover": overlay_metrics["avg_turnover"],
                                    "overlay_avg_exposure": overlay_metrics["avg_action_exposure_proxy"],
                                    "excess_cumret": excess_cumret,
                                    "excess_sharpe": excess_sharpe,
                                    "selection_score": score,
                                })

    grid_df = pd.DataFrame(rows)
    best = grid_df.sort_values(
        ["selection_score", "excess_cumret", "excess_sharpe"],
        ascending=[False, False, False]
    ).iloc[0]

    best_params = {
        "confidence_margin": float(best["confidence_margin"]),
        "use_align_filter": bool(best["use_align_filter"]),
        "vix_z_max": None if pd.isna(best["vix_z_max"]) else float(best["vix_z_max"]),
        "credit_stress_max": None if pd.isna(best["credit_stress_max"]) else float(best["credit_stress_max"]),
        "credit_quantile": None if pd.isna(best["credit_quantile"]) else float(best["credit_quantile"]),
        "dd_half_stop": float(best["dd_half_stop"]),
        "dd_cash_stop": float(best["dd_cash_stop"]),
        "min_prob_for_entry": float(best["min_prob_for_entry"]),
    }
    return best_params, best.to_dict(), grid_df


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Risk-aware overlay on top of Transformer predictions.")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--feature_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # Base strategy from Transformer v5 main system
    parser.add_argument("--base_mode", type=str, default="topk")
    parser.add_argument("--base_top_k", type=int, default=2)
    parser.add_argument("--base_min_prob", type=float, default=0.60)
    parser.add_argument("--base_threshold", type=float, default=0.50)

    # transaction costs
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)

    # validation search grids
    parser.add_argument("--confidence_margin_grid", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--use_align_filter_grid", type=str, default="true,false")
    parser.add_argument("--vix_z_max_grid", type=str, default="1.5,2.0,None")
    parser.add_argument("--credit_quantile_grid", type=str, default="0.8,None")
    parser.add_argument("--dd_half_stop_grid", type=str, default="-0.08,-0.10")
    parser.add_argument("--dd_cash_stop_grid", type=str, default="-0.15,-0.20")
    parser.add_argument("--min_prob_for_entry_grid", type=str, default="0.56,0.58,0.60")

    # selection score weights
    parser.add_argument("--score_excess_cumret_weight", type=float, default=4.0)
    parser.add_argument("--score_excess_sharpe_weight", type=float, default=0.5)
    parser.add_argument("--score_mdd_penalty_weight", type=float, default=3.0)
    parser.add_argument("--score_turnover_penalty_weight", type=float, default=0.3)
    parser.add_argument("--score_mdd_limit", type=float, default=0.25)

    args = parser.parse_args()

    args.confidence_margin_grid_list = [float(x.strip()) for x in args.confidence_margin_grid.split(",") if x.strip()]
    args.use_align_filter_grid_list = parse_bool_grid(args.use_align_filter_grid)
    args.vix_z_max_grid_list = parse_optional_float_grid(args.vix_z_max_grid)
    args.credit_quantile_grid_list = parse_optional_float_grid(args.credit_quantile_grid)
    args.dd_half_stop_grid_list = [float(x.strip()) for x in args.dd_half_stop_grid.split(",") if x.strip()]
    args.dd_cash_stop_grid_list = [float(x.strip()) for x in args.dd_cash_stop_grid.split(",") if x.strip()]
    args.min_prob_for_entry_grid_list = [float(x.strip()) for x in args.min_prob_for_entry_grid.split(",") if x.strip()]

    horizon = infer_horizon_from_feature_path(args.feature_data_path)
    print(f"Horizon inferred: {horizon}")

    merged = load_and_merge_inputs(args.predictions_path, args.feature_data_path)
    print(f"Merged rows: {len(merged)}")
    print(merged["split"].value_counts().to_string())

    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    # Base strategy metrics
    val_base_metrics, val_base_actions = backtest_base_strategy(
        val_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    test_base_metrics, test_base_actions = backtest_base_strategy(
        test_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    # Validation search
    best_params, best_row, val_grid = select_overlay_params_on_validation(
        val_df=val_df,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        base_metrics=val_base_metrics,
        confidence_margin_grid=args.confidence_margin_grid_list,
        use_align_filter_grid=args.use_align_filter_grid_list,
        vix_z_max_grid=args.vix_z_max_grid_list,
        credit_quantile_grid=args.credit_quantile_grid_list,
        dd_half_stop_grid=args.dd_half_stop_grid_list,
        dd_cash_stop_grid=args.dd_cash_stop_grid_list,
        min_prob_for_entry_grid=args.min_prob_for_entry_grid_list,
        score_weights={
            "excess_cumret": args.score_excess_cumret_weight,
            "excess_sharpe": args.score_excess_sharpe_weight,
            "mdd_penalty": args.score_mdd_penalty_weight,
            "turnover_penalty": args.score_turnover_penalty_weight,
            "mdd_limit": args.score_mdd_limit,
        },
    )

    val_overlay_metrics, val_actions = backtest_overlay_strategy(
        df=val_df,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        confidence_margin=best_params["confidence_margin"],
        use_align_filter=best_params["use_align_filter"],
        vix_z_max=best_params["vix_z_max"],
        credit_stress_max=best_params["credit_stress_max"],
        dd_half_stop=best_params["dd_half_stop"],
        dd_cash_stop=best_params["dd_cash_stop"],
        min_prob_for_entry=best_params["min_prob_for_entry"],
    )
    test_overlay_metrics, test_actions = backtest_overlay_strategy(
        df=test_df,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        confidence_margin=best_params["confidence_margin"],
        use_align_filter=best_params["use_align_filter"],
        vix_z_max=best_params["vix_z_max"],
        credit_stress_max=best_params["credit_stress_max"],
        dd_half_stop=best_params["dd_half_stop"],
        dd_cash_stop=best_params["dd_cash_stop"],
        min_prob_for_entry=best_params["min_prob_for_entry"],
    )

    overlay_dir = Path(args.out_dir) / f"{Path(args.predictions_path).stem}_transformer_overlay_v3_riskaware"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    val_grid.to_csv(overlay_dir / "validation_grid_search.csv", index=False)
    val_actions.to_csv(overlay_dir / "val_actions.csv", index=False)
    test_actions.to_csv(overlay_dir / "test_actions.csv", index=False)
    val_base_actions.to_csv(overlay_dir / "val_base_actions.csv", index=False)
    test_base_actions.to_csv(overlay_dir / "test_base_actions.csv", index=False)

    summary = {
        "overlay_type": "transformer_riskaware_v3",
        "base_strategy": {
            "mode": args.base_mode,
            "top_k": args.base_top_k,
            "min_prob": args.base_min_prob,
            "threshold": args.base_threshold,
            "transaction_cost_bps": args.transaction_cost_bps,
        },
        "selected_overlay_params": best_params,
        "validation_best_row": best_row,
        "validation_overlay_metrics": val_overlay_metrics,
        "validation_base_metrics": val_base_metrics,
        "validation_excess_cumulative_return": float(val_overlay_metrics["cumulative_return"] - val_base_metrics["cumulative_return"]),
        "validation_excess_sharpe": float(val_overlay_metrics["sharpe"] - val_base_metrics["sharpe"]),
        "test_overlay_metrics": test_overlay_metrics,
        "test_base_metrics": test_base_metrics,
        "test_excess_cumulative_return": float(test_overlay_metrics["cumulative_return"] - test_base_metrics["cumulative_return"]),
        "test_excess_sharpe": float(test_overlay_metrics["sharpe"] - test_base_metrics["sharpe"]),
    }

    with open(overlay_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nFinished Transformer risk-aware overlay.")
    print(f"Saved outputs to: {overlay_dir}")
    print("\nSelected overlay params:")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    print("\nValidation overlay metrics:")
    print(json.dumps(val_overlay_metrics, ensure_ascii=False, indent=2))
    print("\nValidation base metrics:")
    print(json.dumps(val_base_metrics, ensure_ascii=False, indent=2))
    print("\nValidation excess cumulative return:")
    print(val_overlay_metrics["cumulative_return"] - val_base_metrics["cumulative_return"])
    print("\nTest overlay metrics:")
    print(json.dumps(test_overlay_metrics, ensure_ascii=False, indent=2))
    print("\nTest base metrics:")
    print(json.dumps(test_base_metrics, ensure_ascii=False, indent=2))
    print("\nTest excess cumulative return:")
    print(test_overlay_metrics["cumulative_return"] - test_base_metrics["cumulative_return"])


if __name__ == "__main__":
    main()

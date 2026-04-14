
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Transformer base-weight risk-scaler overlay
#
# Core design
# -----------
# This overlay does NOT re-select assets.
# It ONLY rescales the existing base Transformer portfolio:
#
#   full: keep base weights unchanged
#   half: multiply base weights by 0.5
#   cash: zero weights
#
# The goal is to make the overlay a true risk-management layer rather
# than a second trading strategy.
#
# Key features
# ------------
# 1) Transaction-cost-aware backtest
# 2) Relative-to-base hard constraints in validation
# 3) Drawdown kill-switch
# 4) Recovery hysteresis
# 5) Dynamic exposure scaling
#
# Typical usage
# -------------
# python train_transformer_overlay_risk_scaler.py ^
#   --predictions_path "D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv" ^
#   --feature_data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs\final_run_20260413"
# =========================================================


# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)


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
    stem = Path(feature_data_path).stem.lower()
    if "h1" in stem:
        return 1
    if "h5" in stem:
        return 5
    return 5


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


def parse_int_grid(grid_str: str) -> List[int]:
    return [int(x.strip()) for x in grid_str.split(",") if x.strip()]


# -----------------------------
# Data loading / merge
# -----------------------------
def load_and_merge_inputs(predictions_path: str, feature_data_path: str) -> pd.DataFrame:
    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)

    pred["date"] = pd.to_datetime(pred["date"])
    feat["date"] = pd.to_datetime(feat["date"])

    required_pred_cols = {"date", "stock", "split"}
    if not required_pred_cols.issubset(pred.columns):
        raise ValueError(f"Predictions file must contain at least {required_pred_cols}")

    # support both pred_prob and probability-like naming
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
    for c in ["mkt_dc_trend", "dc_trend", "vix_z_60", "credit_stress"]:
        if c in feat.columns:
            merge_cols.append(c)

    feat_small = feat[merge_cols].copy().drop_duplicates(["date", "stock", "split"])

    # Avoid duplicate future_return columns if predictions already include one.
    if "future_return" in pred.columns:
        feat_ret = feat_small[["date", "stock", "split", return_col]].rename(columns={return_col: "future_return_from_feat"})
        chk = pred[["date", "stock", "split", "future_return"]].merge(feat_ret, on=["date", "stock", "split"], how="left")
        if chk["future_return_from_feat"].isna().any():
            raise ValueError("Feature-side future_return missing after merge.")
        diff = (chk["future_return"].astype(float) - chk["future_return_from_feat"].astype(float)).abs()
        max_diff = float(diff.max()) if len(diff) > 0 else 0.0
        if max_diff > 1e-9:
            print(f"Warning: prediction-side and feature-side future_return differ; max_abs_diff={max_diff:.12f}. Using prediction-side future_return.")
        feat_small = feat_small.drop(columns=[return_col])
        merged = pred.merge(feat_small, on=["date", "stock", "split"], how="left")
    else:
        merged = pred.merge(feat_small, on=["date", "stock", "split"], how="left")
        merged = merged.rename(columns={return_col: "future_return"})

    if "future_return" not in merged.columns:
        raise ValueError("Merged data does not contain future_return.")
    if merged["future_return"].isna().any():
        raise ValueError("Merged future_return contains missing values.")

    return merged


# -----------------------------
# Base strategy
# -----------------------------
def compute_base_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day = day_df.sort_values("pred_prob", ascending=False).copy()

    if mode == "topk":
        chosen = day[day["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day[day["pred_prob"] >= threshold].copy()
    else:
        raise ValueError(f"Unknown base mode: {mode}")

    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def build_base_daily_frames(
    df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
) -> List[Dict]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]

    daily = []
    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        base_w = compute_base_weights(day, mode=mode, top_k=top_k, min_prob=min_prob, threshold=threshold)

        top1 = day.iloc[0] if len(day) > 0 else None
        top2 = day.iloc[1] if len(day) > 1 else top1
        top3 = day.iloc[2] if len(day) > 2 else top2

        info = {
            "date": pd.Timestamp(dt),
            "day_df": day,
            "base_weights": base_w,
            "base_exposure": float(sum(base_w.values())),
            "top1_stock": None if top1 is None else str(top1["stock"]),
            "top1_prob": np.nan if top1 is None else float(top1["pred_prob"]),
            "top2_prob": np.nan if top2 is None else float(top2["pred_prob"]),
            "top3_prob": np.nan if top3 is None else float(top3["pred_prob"]),
            "score_gap_23": np.nan if top2 is None or top3 is None else float(top2["pred_prob"] - top3["pred_prob"]),
            "mkt_dc_trend": np.nan if top1 is None or "mkt_dc_trend" not in day.columns or pd.isna(top1.get("mkt_dc_trend")) else float(top1["mkt_dc_trend"]),
            "dc_trend": np.nan if top1 is None or "dc_trend" not in day.columns or pd.isna(top1.get("dc_trend")) else float(top1["dc_trend"]),
            "vix_z_60": np.nan if top1 is None or "vix_z_60" not in day.columns or pd.isna(top1.get("vix_z_60")) else float(top1["vix_z_60"]),
            "credit_stress": np.nan if top1 is None or "credit_stress" not in day.columns or pd.isna(top1.get("credit_stress")) else float(top1["credit_stress"]),
        }
        daily.append(info)
    return daily


def backtest_base_strategy(
    df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    daily = build_base_daily_frames(df, horizon, mode, top_k, min_prob, threshold)
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    actions = []

    for info in daily:
        day = info["day_df"]
        new_w = info["base_weights"]

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)

        actions.append({
            "date": info["date"],
            "action": "base",
            "top1_stock": info["top1_stock"],
            "top1_prob": info["top1_prob"],
            "score_gap_23": info["score_gap_23"],
            "exposure": info["base_exposure"],
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
# Risk-scaler overlay
# -----------------------------
def scale_weights(weights: Dict[str, float], scalar: float) -> Dict[str, float]:
    return {k: v * scalar for k, v in weights.items()}


def compute_overlay_scale(
    info: Dict,
    current_drawdown: float,
    in_cash_lock: bool,
    recovery_counter: int,
    use_align_filter: bool,
    vix_z_max: Optional[float],
    credit_stress_max: Optional[float],
    dd_half_stop: float,
    dd_cash_stop: float,
    recovery_periods: int,
    half_on_regime_break: bool,
) -> Tuple[float, str, bool, int, Dict]:
    """
    Returns:
      scale in {0.0, 0.5, 1.0}
      action label
      updated cash_lock
      updated recovery_counter
      debug info
    """
    base_has_position = len(info["base_weights"]) > 0

    if not base_has_position:
        return 0.0, "cash_no_base", False, 0, {
            "base_has_position": False,
            "full_eligible": False,
            "risk_flags": "no_base",
        }

    risk_flags = []

    full_eligible = True

    if use_align_filter:
        mdc = info["mkt_dc_trend"]
        tdc = info["dc_trend"]
        if np.isfinite(mdc) and np.isfinite(tdc):
            if not (mdc * tdc > 0):
                full_eligible = False
                risk_flags.append("dc_misaligned")

    if vix_z_max is not None and np.isfinite(info["vix_z_60"]):
        if info["vix_z_60"] > vix_z_max:
            full_eligible = False
            risk_flags.append("high_vix")

    if credit_stress_max is not None and np.isfinite(info["credit_stress"]):
        if info["credit_stress"] > credit_stress_max:
            full_eligible = False
            risk_flags.append("credit_stress")

    # Severe drawdown kill-switch
    if current_drawdown <= dd_cash_stop:
        return 0.0, "cash_dd_kill", True, 0, {
            "base_has_position": True,
            "full_eligible": full_eligible,
            "risk_flags": "|".join(risk_flags + ["dd_cash_stop"]),
        }

    # If previously in cash lock, require consecutive full-eligible periods to unlock
    if in_cash_lock:
        if full_eligible:
            recovery_counter += 1
            if recovery_counter >= recovery_periods:
                # unlock and go full
                return 1.0, "full_recovered", False, 0, {
                    "base_has_position": True,
                    "full_eligible": True,
                    "risk_flags": "|".join(risk_flags + ["recovered"]),
                }
            else:
                # partial re-entry during recovery
                return 0.5, "half_recovery", True, recovery_counter, {
                    "base_has_position": True,
                    "full_eligible": True,
                    "risk_flags": "|".join(risk_flags + ["recovery_wait"]),
                }
        else:
            # remain cautious
            return 0.0, "cash_recovery_wait", True, 0, {
                "base_has_position": True,
                "full_eligible": False,
                "risk_flags": "|".join(risk_flags + ["recovery_blocked"]),
            }

    # Soft drawdown guard
    if current_drawdown <= dd_half_stop:
        return 0.5, "half_dd_guard", False, 0, {
            "base_has_position": True,
            "full_eligible": full_eligible,
            "risk_flags": "|".join(risk_flags + ["dd_half_stop"]),
        }

    # Regime logic
    if full_eligible:
        return 1.0, "full", False, 0, {
            "base_has_position": True,
            "full_eligible": True,
            "risk_flags": "|".join(risk_flags),
        }
    else:
        if half_on_regime_break:
            return 0.5, "half_regime", False, 0, {
                "base_has_position": True,
                "full_eligible": False,
                "risk_flags": "|".join(risk_flags),
            }
        else:
            return 0.0, "cash_regime", False, 0, {
                "base_has_position": True,
                "full_eligible": False,
                "risk_flags": "|".join(risk_flags),
            }


def backtest_overlay_risk_scaler(
    df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float,
    use_align_filter: bool,
    vix_z_max: Optional[float],
    credit_stress_max: Optional[float],
    dd_half_stop: float,
    dd_cash_stop: float,
    recovery_periods: int,
    half_on_regime_break: bool,
) -> Tuple[Dict, pd.DataFrame]:
    daily = build_base_daily_frames(df, horizon, mode, top_k, min_prob, threshold)
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    actions = []

    in_cash_lock = False
    recovery_counter = 0

    for info in daily:
        current_drawdown = equity / max(equity_curve) - 1.0

        scale, action_name, in_cash_lock, recovery_counter, dbg = compute_overlay_scale(
            info=info,
            current_drawdown=current_drawdown,
            in_cash_lock=in_cash_lock,
            recovery_counter=recovery_counter,
            use_align_filter=use_align_filter,
            vix_z_max=vix_z_max,
            credit_stress_max=credit_stress_max,
            dd_half_stop=dd_half_stop,
            dd_cash_stop=dd_cash_stop,
            recovery_periods=recovery_periods,
            half_on_regime_break=half_on_regime_break,
        )

        new_w = scale_weights(info["base_weights"], scale)

        day = info["day_df"]
        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)

        actions.append({
            "date": info["date"],
            "action": action_name,
            "top1_stock": info["top1_stock"],
            "top1_prob": info["top1_prob"],
            "score_gap_23": info["score_gap_23"],
            "base_exposure": info["base_exposure"],
            "overlay_exposure": float(sum(new_w.values())),
            "selected_stocks": "|".join(sorted(new_w.keys())),
            "gross_return": gross,
            "net_return": net,
            "turnover": turn,
            "equity": equity,
            "drawdown": equity / max(equity_curve) - 1.0,
            "base_has_position": dbg["base_has_position"],
            "full_eligible": dbg["full_eligible"],
            "risk_flags": dbg["risk_flags"],
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
        "avg_action_exposure_proxy": float(np.mean([a["overlay_exposure"] for a in actions])) if actions else 0.0,
    }
    return metrics, pd.DataFrame(actions)


# -----------------------------
# Validation selection
# -----------------------------
def select_overlay_params_on_validation(
    val_df: pd.DataFrame,
    horizon: int,
    base_mode: str,
    base_top_k: int,
    base_min_prob: float,
    base_threshold: float,
    transaction_cost_bps: float,
    base_metrics: Dict,
    use_align_filter_grid: List[bool],
    vix_z_max_grid: List[Optional[float]],
    credit_quantile_grid: List[Optional[float]],
    dd_half_stop_grid: List[float],
    dd_cash_stop_grid: List[float],
    recovery_periods_grid: List[int],
    half_on_regime_break_grid: List[bool],
    return_tolerance: float,
    turnover_cap_mult: float,
    exposure_cap_add: float,
    sharpe_tolerance: float,
    score_weights: Dict[str, float],
) -> Tuple[Optional[Dict], Optional[Dict], pd.DataFrame]:
    credit_values = val_df["credit_stress"].dropna().values.astype(float) if "credit_stress" in val_df.columns else np.array([])
    rows = []

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
                        for rp in recovery_periods_grid:
                            for half_regime in half_on_regime_break_grid:
                                overlay_metrics, _ = backtest_overlay_risk_scaler(
                                    df=val_df,
                                    horizon=horizon,
                                    mode=base_mode,
                                    top_k=base_top_k,
                                    min_prob=base_min_prob,
                                    threshold=base_threshold,
                                    transaction_cost_bps=transaction_cost_bps,
                                    use_align_filter=align,
                                    vix_z_max=vix_max,
                                    credit_stress_max=credit_max,
                                    dd_half_stop=dd_half,
                                    dd_cash_stop=dd_cash,
                                    recovery_periods=rp,
                                    half_on_regime_break=half_regime,
                                )

                                excess_cumret = overlay_metrics["cumulative_return"] - base_metrics["cumulative_return"]
                                excess_sharpe = overlay_metrics["sharpe"] - base_metrics["sharpe"]
                                mdd_improvement = overlay_metrics["max_drawdown"] - base_metrics["max_drawdown"]  # higher is better
                                turnover_delta = overlay_metrics["avg_turnover"] - base_metrics["avg_turnover"]

                                constraints_ok = bool(
                                    overlay_metrics["max_drawdown"] >= base_metrics["max_drawdown"]
                                    and overlay_metrics["avg_turnover"] <= base_metrics["avg_turnover"] * turnover_cap_mult
                                    and overlay_metrics["avg_action_exposure_proxy"] <= base_metrics["avg_action_exposure_proxy"] + exposure_cap_add
                                    and overlay_metrics["cumulative_return"] >= base_metrics["cumulative_return"] - return_tolerance
                                    and overlay_metrics["sharpe"] >= base_metrics["sharpe"] - sharpe_tolerance
                                )

                                score = (
                                    score_weights["excess_sharpe"] * excess_sharpe
                                    + score_weights["excess_cumret"] * excess_cumret
                                    + score_weights["mdd_improvement"] * mdd_improvement
                                    - score_weights["turnover_penalty"] * max(0.0, turnover_delta)
                                )

                                rows.append({
                                    "use_align_filter": align,
                                    "vix_z_max": vix_max,
                                    "credit_quantile": cq,
                                    "credit_stress_max": credit_max,
                                    "dd_half_stop": dd_half,
                                    "dd_cash_stop": dd_cash,
                                    "recovery_periods": rp,
                                    "half_on_regime_break": half_regime,
                                    "constraints_ok": constraints_ok,
                                    "selection_score": score,
                                    "overlay_cumulative_return": overlay_metrics["cumulative_return"],
                                    "overlay_sharpe": overlay_metrics["sharpe"],
                                    "overlay_max_drawdown": overlay_metrics["max_drawdown"],
                                    "overlay_avg_turnover": overlay_metrics["avg_turnover"],
                                    "overlay_avg_exposure": overlay_metrics["avg_action_exposure_proxy"],
                                    "base_cumulative_return": base_metrics["cumulative_return"],
                                    "base_sharpe": base_metrics["sharpe"],
                                    "base_max_drawdown": base_metrics["max_drawdown"],
                                    "base_avg_turnover": base_metrics["avg_turnover"],
                                    "base_avg_exposure": base_metrics["avg_action_exposure_proxy"],
                                    "excess_cumret": excess_cumret,
                                    "excess_sharpe": excess_sharpe,
                                    "mdd_improvement": mdd_improvement,
                                    "turnover_delta": turnover_delta,
                                })

    grid_df = pd.DataFrame(rows)

    valid = grid_df[grid_df["constraints_ok"] == True].copy()
    if len(valid) == 0:
        return None, None, grid_df

    best = valid.sort_values(
        ["selection_score", "mdd_improvement", "excess_sharpe", "excess_cumret"],
        ascending=[False, False, False, False]
    ).iloc[0]

    best_params = {
        "use_align_filter": bool(best["use_align_filter"]),
        "vix_z_max": None if pd.isna(best["vix_z_max"]) else float(best["vix_z_max"]),
        "credit_stress_max": None if pd.isna(best["credit_stress_max"]) else float(best["credit_stress_max"]),
        "credit_quantile": None if pd.isna(best["credit_quantile"]) else float(best["credit_quantile"]),
        "dd_half_stop": float(best["dd_half_stop"]),
        "dd_cash_stop": float(best["dd_cash_stop"]),
        "recovery_periods": int(best["recovery_periods"]),
        "half_on_regime_break": bool(best["half_on_regime_break"]),
    }
    return best_params, best.to_dict(), grid_df


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Transformer base-weight risk-scaler overlay.")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--feature_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # Base strategy should match Transformer v5 selected config
    parser.add_argument("--base_mode", type=str, default="topk")
    parser.add_argument("--base_top_k", type=int, default=2)
    parser.add_argument("--base_min_prob", type=float, default=0.60)
    parser.add_argument("--base_threshold", type=float, default=0.50)

    # Costs
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)

    # Overlay validation search
    parser.add_argument("--use_align_filter_grid", type=str, default="true,false")
    parser.add_argument("--vix_z_max_grid", type=str, default="1.5,2.0,None")
    parser.add_argument("--credit_quantile_grid", type=str, default="0.8,None")
    parser.add_argument("--dd_half_stop_grid", type=str, default="-0.08,-0.10")
    parser.add_argument("--dd_cash_stop_grid", type=str, default="-0.15,-0.20")
    parser.add_argument("--recovery_periods_grid", type=str, default="1,2,3")
    parser.add_argument("--half_on_regime_break_grid", type=str, default="true,false")

    # Relative-to-base hard constraints
    parser.add_argument("--return_tolerance", type=float, default=0.05)
    parser.add_argument("--turnover_cap_mult", type=float, default=1.05)
    parser.add_argument("--exposure_cap_add", type=float, default=0.00)
    parser.add_argument("--sharpe_tolerance", type=float, default=0.05)

    # Selection score
    parser.add_argument("--score_excess_sharpe_weight", type=float, default=2.0)
    parser.add_argument("--score_excess_cumret_weight", type=float, default=1.0)
    parser.add_argument("--score_mdd_improvement_weight", type=float, default=2.0)
    parser.add_argument("--score_turnover_penalty_weight", type=float, default=0.25)

    args = parser.parse_args()

    args.use_align_filter_grid_list = parse_bool_grid(args.use_align_filter_grid)
    args.vix_z_max_grid_list = parse_optional_float_grid(args.vix_z_max_grid)
    args.credit_quantile_grid_list = parse_optional_float_grid(args.credit_quantile_grid)
    args.dd_half_stop_grid_list = [float(x.strip()) for x in args.dd_half_stop_grid.split(",") if x.strip()]
    args.dd_cash_stop_grid_list = [float(x.strip()) for x in args.dd_cash_stop_grid.split(",") if x.strip()]
    args.recovery_periods_grid_list = parse_int_grid(args.recovery_periods_grid)
    args.half_on_regime_break_grid_list = parse_bool_grid(args.half_on_regime_break_grid)

    horizon = infer_horizon_from_feature_path(args.feature_data_path)
    print(f"Horizon inferred: {horizon}")

    merged = load_and_merge_inputs(args.predictions_path, args.feature_data_path)
    print(f"Merged rows: {len(merged)}")
    print(merged["split"].value_counts().to_string())

    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    # Base metrics
    val_base_metrics, val_base_actions = backtest_base_strategy(
        df=val_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    test_base_metrics, test_base_actions = backtest_base_strategy(
        df=test_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    # Validation search
    best_params, best_row, grid_df = select_overlay_params_on_validation(
        val_df=val_df,
        horizon=horizon,
        base_mode=args.base_mode,
        base_top_k=args.base_top_k,
        base_min_prob=args.base_min_prob,
        base_threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
        base_metrics=val_base_metrics,
        use_align_filter_grid=args.use_align_filter_grid_list,
        vix_z_max_grid=args.vix_z_max_grid_list,
        credit_quantile_grid=args.credit_quantile_grid_list,
        dd_half_stop_grid=args.dd_half_stop_grid_list,
        dd_cash_stop_grid=args.dd_cash_stop_grid_list,
        recovery_periods_grid=args.recovery_periods_grid_list,
        half_on_regime_break_grid=args.half_on_regime_break_grid_list,
        return_tolerance=args.return_tolerance,
        turnover_cap_mult=args.turnover_cap_mult,
        exposure_cap_add=args.exposure_cap_add,
        sharpe_tolerance=args.sharpe_tolerance,
        score_weights={
            "excess_sharpe": args.score_excess_sharpe_weight,
            "excess_cumret": args.score_excess_cumret_weight,
            "mdd_improvement": args.score_mdd_improvement_weight,
            "turnover_penalty": args.score_turnover_penalty_weight,
        },
    )

    out_dir = Path(args.out_dir) / f"{Path(args.predictions_path).stem}_transformer_overlay_risk_scaler"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_dir / "validation_grid_search.csv", index=False)
    val_base_actions.to_csv(out_dir / "val_base_actions.csv", index=False)
    test_base_actions.to_csv(out_dir / "test_base_actions.csv", index=False)

    if best_params is None:
        summary = {
            "overlay_type": "transformer_risk_scaler",
            "selected_overlay_params": None,
            "selection_result": "no_valid_overlay_found_use_base",
            "validation_base_metrics": val_base_metrics,
            "test_base_metrics": test_base_metrics,
            "notes": "No overlay satisfied the relative-to-base validation constraints. Base strategy should be used."
        }
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nNo valid overlay satisfied the relative-to-base constraints.")
        print("Recommendation: use base Transformer v5 strategy directly.")
        print(f"Saved outputs to: {out_dir}")
        return

    val_overlay_metrics, val_actions = backtest_overlay_risk_scaler(
        df=val_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
        use_align_filter=best_params["use_align_filter"],
        vix_z_max=best_params["vix_z_max"],
        credit_stress_max=best_params["credit_stress_max"],
        dd_half_stop=best_params["dd_half_stop"],
        dd_cash_stop=best_params["dd_cash_stop"],
        recovery_periods=best_params["recovery_periods"],
        half_on_regime_break=best_params["half_on_regime_break"],
    )
    test_overlay_metrics, test_actions = backtest_overlay_risk_scaler(
        df=test_df,
        horizon=horizon,
        mode=args.base_mode,
        top_k=args.base_top_k,
        min_prob=args.base_min_prob,
        threshold=args.base_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
        use_align_filter=best_params["use_align_filter"],
        vix_z_max=best_params["vix_z_max"],
        credit_stress_max=best_params["credit_stress_max"],
        dd_half_stop=best_params["dd_half_stop"],
        dd_cash_stop=best_params["dd_cash_stop"],
        recovery_periods=best_params["recovery_periods"],
        half_on_regime_break=best_params["half_on_regime_break"],
    )

    val_actions.to_csv(out_dir / "val_actions.csv", index=False)
    test_actions.to_csv(out_dir / "test_actions.csv", index=False)

    summary = {
        "overlay_type": "transformer_risk_scaler",
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
        "validation_mdd_improvement": float(val_overlay_metrics["max_drawdown"] - val_base_metrics["max_drawdown"]),
        "test_overlay_metrics": test_overlay_metrics,
        "test_base_metrics": test_base_metrics,
        "test_excess_cumulative_return": float(test_overlay_metrics["cumulative_return"] - test_base_metrics["cumulative_return"]),
        "test_excess_sharpe": float(test_overlay_metrics["sharpe"] - test_base_metrics["sharpe"]),
        "test_mdd_improvement": float(test_overlay_metrics["max_drawdown"] - test_base_metrics["max_drawdown"]),
    }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nFinished Transformer base-weight risk-scaler overlay.")
    print(f"Saved outputs to: {out_dir}")
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

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
    EXPERIMENTS_DIR,
    EVALUATION_DIR,
    EXECUTION_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)


# =========================================================
# Optimize final deployable execution on top of fixed
# Transformer predictions (VALIDATION-ONLY SELECTION)
#
# Loose Sticky Version
# -------------------
# Goal:
#   Keep the same prediction layer, but re-search a LESS conservative
#   sticky execution family.
#
# Key changes vs prior cleaned version:
#   1) Still NO test leakage in grid search
#   2) Looser default sticky search space:
#        - lower entry_prob
#        - smaller switch_buffer
#        - allow max_new_names_per_rebalance = 2
#        - min_hold fixed to 0 by default
#   3) Selection score penalizes turnover less aggressively and gives
#      a small reward to fuller investment / fuller holdings
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


def latest_file_by_glob(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_default_predictions_path() -> Path:
    base = EXPERIMENTS_DIR / "main_transformer_h5"
    p = latest_file_by_glob(base, "run_*/transformer_predictions_all_splits.csv")
    if p is None:
        raise FileNotFoundError(
            "Could not auto-find transformer_predictions_all_splits.csv under "
            f"{base}. Please pass --predictions_path explicitly."
        )
    return p


def resolve_default_eval_manifest_path() -> Path:
    base = EVALUATION_DIR / "transformer_deployable"
    p = latest_file_by_glob(base, "run_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError(
            "Could not auto-find final_system_manifest.json under "
            f"{base}. Please pass --deployable_manifest_path explicitly."
        )
    return p


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


def run_static_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    top_k: int,
    min_prob: float,
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
        })

        prev_w = new_w

    metrics = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }
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
        "sharpe": sharpe_ratio(period_returns, horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def optimize_execution_on_validation(
    val_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    current_frozen_cfg: Dict,
    target_val_mdd_floor: float,
    target_val_turnover_cap: float,
    topk_grid: List[int],
    entry_prob_grid: List[float],
    exit_gap_grid: List[float],
    switch_buffer_grid: List[float],
    min_hold_grid: List[int],
    max_new_names_grid: List[int],
    turnover_penalty: float,
    exposure_reward: float,
    holdings_reward: float,
) -> Tuple[Dict, Dict, pd.DataFrame, Dict]:
    baseline_val, _ = run_static_backtest(
        pred_df=val_df,
        horizon=horizon,
        top_k=current_frozen_cfg["top_k"],
        min_prob=current_frozen_cfg["min_prob"],
        transaction_cost_bps=transaction_cost_bps,
    )

    rows = []
    for top_k in topk_grid:
        for entry_prob in entry_prob_grid:
            for exit_gap in exit_gap_grid:
                exit_prob = max(0.0, entry_prob - exit_gap)
                for switch_buffer in switch_buffer_grid:
                    for min_hold_periods in min_hold_grid:
                        for max_new_names_per_rebalance in max_new_names_grid:
                            val_metrics, _ = run_sticky_backtest(
                                pred_df=val_df,
                                horizon=horizon,
                                top_k=top_k,
                                entry_prob=entry_prob,
                                exit_prob=exit_prob,
                                switch_buffer=switch_buffer,
                                min_hold_periods=min_hold_periods,
                                max_new_names_per_rebalance=max_new_names_per_rebalance,
                                transaction_cost_bps=transaction_cost_bps,
                            )

                            mdd_ok = bool(val_metrics["max_drawdown"] >= target_val_mdd_floor)
                            turnover_ok = bool(val_metrics["avg_turnover"] <= target_val_turnover_cap)
                            all_constraints_ok = bool(mdd_ok and turnover_ok)

                            holdings_fill_ratio = 0.0
                            if top_k > 0:
                                holdings_fill_ratio = float(val_metrics["avg_holdings"]) / float(top_k)

                            selection_score = (
                                2.5 * float(val_metrics["sharpe"])
                                + 2.25 * float(val_metrics["cumulative_return"])
                                + 1.0 * float(val_metrics["max_drawdown"])
                                - float(turnover_penalty) * float(val_metrics["avg_turnover"])
                                + 0.25 * float(val_metrics["win_rate"])
                                + float(exposure_reward) * float(val_metrics["avg_exposure"])
                                + float(holdings_reward) * float(holdings_fill_ratio)
                            )

                            rows.append({
                                "top_k": int(top_k),
                                "entry_prob": float(entry_prob),
                                "exit_prob": float(exit_prob),
                                "exit_gap": float(exit_gap),
                                "switch_buffer": float(switch_buffer),
                                "min_hold_periods": int(min_hold_periods),
                                "max_new_names_per_rebalance": int(max_new_names_per_rebalance),

                                "mdd_ok": mdd_ok,
                                "turnover_ok": turnover_ok,
                                "all_constraints_ok": all_constraints_ok,
                                "selection_score": float(selection_score),

                                "val_cumret": float(val_metrics["cumulative_return"]),
                                "val_sharpe": float(val_metrics["sharpe"]),
                                "val_mdd": float(val_metrics["max_drawdown"]),
                                "val_turnover": float(val_metrics["avg_turnover"]),
                                "val_avg_holdings": float(val_metrics["avg_holdings"]),
                                "val_win_rate": float(val_metrics["win_rate"]),
                                "val_annualized_return": float(val_metrics["annualized_return"]),
                                "val_avg_exposure": float(val_metrics["avg_exposure"]),
                                "val_holdings_fill_ratio": float(holdings_fill_ratio),
                            })

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        raise RuntimeError("Validation search grid is empty. Check your search space.")

    valid = grid_df[grid_df["all_constraints_ok"] == True].copy()
    if len(valid) > 0:
        best = valid.sort_values(
            [
                "selection_score",
                "val_cumret",
                "val_sharpe",
                "val_avg_holdings",
                "val_avg_exposure",
                "val_turnover",
            ],
            ascending=[False, False, False, False, False, True],
        ).iloc[0]
    else:
        best = grid_df.sort_values(
            [
                "mdd_ok",
                "turnover_ok",
                "selection_score",
                "val_cumret",
                "val_avg_holdings",
                "val_turnover",
            ],
            ascending=[False, False, False, False, False, True],
        ).iloc[0]

    best_cfg = {
        "mode": "topk",
        "top_k": int(best["top_k"]),
        "entry_prob": float(best["entry_prob"]),
        "exit_prob": float(best["exit_prob"]),
        "threshold": 0.50,
        "switch_buffer": float(best["switch_buffer"]),
        "min_hold_periods": int(best["min_hold_periods"]),
        "max_new_names_per_rebalance": int(best["max_new_names_per_rebalance"]),
    }

    baseline = {
        "current_frozen_cfg": current_frozen_cfg,
        "baseline_val": baseline_val,
    }
    return best_cfg, best.to_dict(), grid_df, baseline


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(
        description="Optimize final deployable execution on fixed Transformer predictions (validation-only, loose sticky).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_path", type=str, default="")
    parser.add_argument("--feature_data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--deployable_manifest_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(EXECUTION_DIR / "final_system"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)

    parser.add_argument("--current_top_k", type=int, default=2)
    parser.add_argument("--current_min_prob", type=float, default=0.54)

    # Looser than prior version
    parser.add_argument("--target_val_mdd_floor", type=float, default=-0.25)
    parser.add_argument("--target_val_turnover_cap", type=float, default=1.10)

    # Looser sticky search space
    parser.add_argument("--topk_grid", type=str, default="2")
    parser.add_argument("--entry_prob_grid", type=str, default="0.54,0.55,0.56,0.57")
    parser.add_argument("--exit_gap_grid", type=str, default="0.00,0.01,0.02")
    parser.add_argument("--switch_buffer_grid", type=str, default="0.00,0.005,0.01")
    parser.add_argument("--min_hold_grid", type=str, default="0")
    parser.add_argument("--max_new_names_grid", type=str, default="2")

    # Gentler turnover penalty + mild reward for being fuller / less underinvested
    parser.add_argument("--turnover_penalty", type=float, default=0.20)
    parser.add_argument("--exposure_reward", type=float, default=0.20)
    parser.add_argument("--holdings_reward", type=float, default=0.10)

    args = parser.parse_args()

    predictions_path = Path(args.predictions_path) if args.predictions_path else resolve_default_predictions_path()
    feature_data_path = Path(args.feature_data_path)
    deployable_manifest_path = (
        Path(args.deployable_manifest_path)
        if args.deployable_manifest_path
        else resolve_default_eval_manifest_path()
    )

    print(f"Using predictions: {predictions_path}")
    print(f"Using feature data: {feature_data_path}")
    print(f"Using deployable manifest: {deployable_manifest_path}")

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    deployable_manifest = safe_read_json(deployable_manifest_path)
    current_cfg = deployable_manifest.get("frozen_config", {})
    if not current_cfg:
        current_cfg = {
            "mode": "topk",
            "top_k": args.current_top_k,
            "min_prob": args.current_min_prob,
            "threshold": 0.50,
        }

    current_cfg = {
        "mode": "topk",
        "top_k": int(current_cfg.get("top_k", args.current_top_k)),
        "min_prob": float(current_cfg.get("min_prob", args.current_min_prob)),
        "threshold": float(current_cfg.get("threshold", 0.50)),
    }

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Predictions file must contain non-empty val and test splits.")

    best_cfg, best_row, grid_df, baseline = optimize_execution_on_validation(
        val_df=val_df,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        current_frozen_cfg=current_cfg,
        target_val_mdd_floor=args.target_val_mdd_floor,
        target_val_turnover_cap=args.target_val_turnover_cap,
        topk_grid=parse_int_grid(args.topk_grid),
        entry_prob_grid=parse_float_grid(args.entry_prob_grid),
        exit_gap_grid=parse_float_grid(args.exit_gap_grid),
        switch_buffer_grid=parse_float_grid(args.switch_buffer_grid),
        min_hold_grid=parse_int_grid(args.min_hold_grid),
        max_new_names_grid=parse_int_grid(args.max_new_names_grid),
        turnover_penalty=args.turnover_penalty,
        exposure_reward=args.exposure_reward,
        holdings_reward=args.holdings_reward,
    )

    cur_val_metrics, cur_val_actions = run_static_backtest(
        pred_df=val_df,
        horizon=horizon,
        top_k=current_cfg["top_k"],
        min_prob=current_cfg["min_prob"],
        transaction_cost_bps=args.transaction_cost_bps,
    )
    cur_test_metrics, cur_test_actions = run_static_backtest(
        pred_df=test_df,
        horizon=horizon,
        top_k=current_cfg["top_k"],
        min_prob=current_cfg["min_prob"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    opt_val_metrics, opt_val_actions = run_sticky_backtest(
        pred_df=val_df,
        horizon=horizon,
        top_k=best_cfg["top_k"],
        entry_prob=best_cfg["entry_prob"],
        exit_prob=best_cfg["exit_prob"],
        switch_buffer=best_cfg["switch_buffer"],
        min_hold_periods=best_cfg["min_hold_periods"],
        max_new_names_per_rebalance=best_cfg["max_new_names_per_rebalance"],
        transaction_cost_bps=args.transaction_cost_bps,
    )
    opt_test_metrics, opt_test_actions = run_sticky_backtest(
        pred_df=test_df,
        horizon=horizon,
        top_k=best_cfg["top_k"],
        entry_prob=best_cfg["entry_prob"],
        exit_prob=best_cfg["exit_prob"],
        switch_buffer=best_cfg["switch_buffer"],
        min_hold_periods=best_cfg["min_hold_periods"],
        max_new_names_per_rebalance=best_cfg["max_new_names_per_rebalance"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"final_system_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_df.to_csv(out_root / "validation_search_grid.csv", index=False)
    save_df_with_dates(cur_val_actions, out_root / "current_val_actions.csv")
    save_df_with_dates(cur_test_actions, out_root / "current_test_actions.csv")
    save_df_with_dates(opt_val_actions, out_root / "optimized_val_actions.csv")
    save_df_with_dates(opt_test_actions, out_root / "optimized_test_actions.csv")

    save_df_with_dates(opt_val_actions, out_root / "final_val_actions.csv")
    save_df_with_dates(opt_test_actions, out_root / "final_test_actions.csv")

    compare_df = pd.DataFrame([
        {"config_label": "current_frozen", "split": "val", **cur_val_metrics},
        {"config_label": "current_frozen", "split": "test", **cur_test_metrics},
        {"config_label": "optimized_execution", "split": "val", **opt_val_metrics},
        {"config_label": "optimized_execution", "split": "test", **opt_test_metrics},
    ])
    compare_df.to_csv(out_root / "current_vs_optimized_metrics.csv", index=False)

    search_space_profile = {
        "profile_name": "loose_sticky_v1",
        "topk_grid": parse_int_grid(args.topk_grid),
        "entry_prob_grid": parse_float_grid(args.entry_prob_grid),
        "exit_gap_grid": parse_float_grid(args.exit_gap_grid),
        "switch_buffer_grid": parse_float_grid(args.switch_buffer_grid),
        "min_hold_grid": parse_int_grid(args.min_hold_grid),
        "max_new_names_grid": parse_int_grid(args.max_new_names_grid),
        "turnover_penalty": float(args.turnover_penalty),
        "exposure_reward": float(args.exposure_reward),
        "holdings_reward": float(args.holdings_reward),
        "target_val_mdd_floor": float(args.target_val_mdd_floor),
        "target_val_turnover_cap": float(args.target_val_turnover_cap),
    }

    final_manifest = {
        "final_system_name": "H5 Transformer v5 fixed predictions + optimized loose sticky execution",
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "source_deployable_manifest_path": str(deployable_manifest_path.resolve()),
        "horizon": int(horizon),
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "frozen_prediction_layer": current_cfg,
        "execution_layer": best_cfg,
        "selection_policy": "validation_only_cleaned_no_test_leakage_loose_sticky",
        "selection_constraints": {
            "target_val_mdd_floor": float(args.target_val_mdd_floor),
            "target_val_turnover_cap": float(args.target_val_turnover_cap),
        },
        "search_space_profile": search_space_profile,
        "validation_baseline": baseline,
        "best_search_row": best_row,
    }

    final_metrics_summary = {
        "selection_policy": "validation_only_cleaned_no_test_leakage_loose_sticky",
        "current_frozen_cfg": current_cfg,
        "optimized_execution_cfg": best_cfg,
        "search_space_profile": search_space_profile,
        "validation_baseline": baseline,
        "best_search_row": best_row,
        "current_val_metrics": cur_val_metrics,
        "current_test_metrics": cur_test_metrics,
        "optimized_val_metrics": opt_val_metrics,
        "optimized_test_metrics": opt_test_metrics,
    }

    with open(out_root / "final_system_manifest.json", "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, ensure_ascii=False, indent=2)

    with open(out_root / "final_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics_summary, f, ensure_ascii=False, indent=2)

    with open(out_root / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics_summary, f, ensure_ascii=False, indent=2)

    with open(out_root / "search_space_profile.json", "w", encoding="utf-8") as f:
        json.dump(search_space_profile, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["resolved_predictions_path"] = str(predictions_path.resolve())
    run_config["resolved_feature_data_path"] = str(feature_data_path.resolve())
    run_config["resolved_deployable_manifest_path"] = str(deployable_manifest_path.resolve())
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Execution optimization saved to: {out_root}")
    print("\nCurrent frozen config:")
    print(json.dumps(current_cfg, ensure_ascii=False, indent=2))
    print("\nOptimized execution config:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("\nCurrent validation metrics:")
    print(json.dumps(cur_val_metrics, ensure_ascii=False, indent=2))
    print("\nOptimized validation metrics:")
    print(json.dumps(opt_val_metrics, ensure_ascii=False, indent=2))
    print("\nCurrent test metrics:")
    print(json.dumps(cur_test_metrics, ensure_ascii=False, indent=2))
    print("\nOptimized test metrics:")
    print(json.dumps(opt_test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
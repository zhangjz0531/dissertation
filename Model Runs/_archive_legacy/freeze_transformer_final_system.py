
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Freeze final Transformer system using recommended
# deployable config from diagnostics
#
# What this script does
# ---------------------
# 1) Reads the diagnostics output (recommended deployable config).
# 2) Applies that fixed config to the final Transformer v5 predictions.
# 3) Recomputes validation / test / all-period actions and metrics.
# 4) Writes a final frozen manifest that you can reference in the dissertation.
#
# Intended usage
# --------------
# python freeze_transformer_final_system.py ^
#   --predictions_path "D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv" ^
#   --feature_data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --diagnostics_dir "D:\python\dissertation\Model Runs\final_run_20260413\main_experiment_h5_transformer_v5_diagnostics_expanding_tr1260_va126_te126_st126" ^
#   --out_dir "D:\python\dissertation\Model Runs\final_run_20260413"
#
# Outputs
# -------
# - final_transformer_system/final_system_manifest.json
# - final_transformer_system/final_metrics_summary.json
# - final_transformer_system/final_val_actions.csv
# - final_transformer_system/final_test_actions.csv
# - final_transformer_system/final_all_actions.csv
# - final_transformer_system/final_snapshot_latest.csv
# - final_transformer_system/final_config_vs_old_config.csv
#
# This script does NOT retrain anything.
# It freezes the final deployable version from the already-evaluated model.
# =========================================================


DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_DIAGNOSTICS_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\main_experiment_h5_transformer_v5_diagnostics_expanding_tr1260_va126_te126_st126"


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


def select_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day = day_df.sort_values("pred_prob", ascending=False).copy()

    if mode == "topk":
        chosen = day[day["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day[day["pred_prob"] >= threshold].copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if len(chosen) == 0:
        return {}

    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def run_strategy_backtest_with_actions(
    pred_df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
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
        new_w = select_weights(day, mode=mode, top_k=top_k, min_prob=min_prob, threshold=threshold)
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
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def compare_configs(
    pred_df: pd.DataFrame,
    horizon: int,
    old_cfg: Dict,
    new_cfg: Dict,
    cost_bps: float
) -> pd.DataFrame:
    rows = []
    for label, cfg in [("old_current_fixed", old_cfg), ("new_recommended_deployable", new_cfg)]:
        for split_name in ["val", "test"]:
            split_df = pred_df[pred_df["split"] == split_name].copy()
            metrics, _ = run_strategy_backtest_with_actions(
                split_df,
                horizon=horizon,
                mode=cfg["mode"],
                top_k=int(cfg["top_k"]),
                min_prob=float(cfg["min_prob"]),
                threshold=float(cfg["threshold"]),
                transaction_cost_bps=cost_bps,
            )
            rows.append({
                "config_label": label,
                "split": split_name,
                "mode": cfg["mode"],
                "top_k": int(cfg["top_k"]),
                "min_prob": float(cfg["min_prob"]),
                "threshold": float(cfg["threshold"]),
                **metrics,
            })
    return pd.DataFrame(rows)


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

    weights = select_weights(
        latest,
        mode=cfg["mode"],
        top_k=int(cfg["top_k"]),
        min_prob=float(cfg["min_prob"]),
        threshold=float(cfg["threshold"]),
    )
    latest["selected_in_final"] = latest["stock"].map(lambda s: 1 if s in weights else 0)
    latest["final_weight"] = latest["stock"].map(lambda s: float(weights.get(s, 0.0)))
    return latest


def main():
    parser = argparse.ArgumentParser(description="Freeze final Transformer system using recommended deployable config.")
    parser.add_argument("--predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--diagnostics_dir", type=str, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    diagnostics_dir = Path(args.diagnostics_dir)
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Missing diagnostics dir: {diagnostics_dir}")

    pred = safe_read_csv(args.predictions_path)
    feat = safe_read_csv(args.feature_data_path)
    pred = normalize_predictions(pred, feat)

    horizon = infer_horizon_from_feature_df(feat)

    recommended_cfg = safe_read_json(str(diagnostics_dir / "recommended_deployable_config.json"))
    overall_summary = safe_read_json(str(diagnostics_dir / "overall_summary.json"))
    coverage_summary = safe_read_json(str(diagnostics_dir / "coverage_summary.json"))

    old_cfg = {
        "mode": "topk",
        "top_k": 2,
        "min_prob": 0.60,
        "threshold": 0.50,
    }
    new_cfg = {
        "mode": recommended_cfg["mode"],
        "top_k": int(recommended_cfg["top_k"]),
        "min_prob": float(recommended_cfg["min_prob"]),
        "threshold": float(recommended_cfg["threshold"]),
    }

    out_root = Path(args.out_dir) / "final_transformer_system"
    out_root.mkdir(parents=True, exist_ok=True)

    # Recompute final actions/metrics with the frozen final config
    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()
    all_df = pred[pred["split"].isin(["train", "val", "test"])].copy()

    final_val_metrics, final_val_actions = run_strategy_backtest_with_actions(
        val_df, horizon, new_cfg["mode"], new_cfg["top_k"], new_cfg["min_prob"], new_cfg["threshold"], args.transaction_cost_bps
    )
    final_test_metrics, final_test_actions = run_strategy_backtest_with_actions(
        test_df, horizon, new_cfg["mode"], new_cfg["top_k"], new_cfg["min_prob"], new_cfg["threshold"], args.transaction_cost_bps
    )
    final_all_metrics, final_all_actions = run_strategy_backtest_with_actions(
        all_df, horizon, new_cfg["mode"], new_cfg["top_k"], new_cfg["min_prob"], new_cfg["threshold"], args.transaction_cost_bps
    )

    final_val_actions.to_csv(out_root / "final_val_actions.csv", index=False)
    final_test_actions.to_csv(out_root / "final_test_actions.csv", index=False)
    final_all_actions.to_csv(out_root / "final_all_actions.csv", index=False)

    compare_df = compare_configs(pred, horizon, old_cfg, new_cfg, args.transaction_cost_bps)
    compare_df.to_csv(out_root / "final_config_vs_old_config.csv", index=False)

    latest_df = latest_snapshot(pred, feat, new_cfg)
    latest_df.to_csv(out_root / "final_snapshot_latest.csv", index=False)

    manifest = {
        "final_system_name": "H5 Transformer v5 final deployable system",
        "final_decision": "freeze_recommended_deployable_config",
        "frozen_config": new_cfg,
        "replaced_previous_config": old_cfg,
        "transaction_cost_bps": args.transaction_cost_bps,
        "horizon": horizon,
        "diagnostics_dir": str(diagnostics_dir),
        "coverage_summary": coverage_summary,
        "walkforward_overall_summary": overall_summary,
    }
    with open(out_root / "final_system_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    metrics_summary = {
        "final_config": new_cfg,
        "validation_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "all_period_metrics": final_all_metrics,
    }
    with open(out_root / "final_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"Final system frozen to: {out_root}")
    print("\nFrozen config:")
    print(json.dumps(new_cfg, ensure_ascii=False, indent=2))
    print("\nValidation metrics:")
    print(json.dumps(final_val_metrics, ensure_ascii=False, indent=2))
    print("\nTest metrics:")
    print(json.dumps(final_test_metrics, ensure_ascii=False, indent=2))
    print("\nAll-period metrics:")
    print(json.dumps(final_all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

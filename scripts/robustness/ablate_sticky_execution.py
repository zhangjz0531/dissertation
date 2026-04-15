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
    ABLATIONS_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)


# =========================================================
# Sticky execution ablation
#
# Compares on the SAME fixed Transformer predictions:
#   A) frozen static prediction-only baseline
#   B) optimized sticky execution overlay
#
# No retraining happens here.
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


def resolve_default_deployable_manifest_path() -> Path:
    base = EVALUATION_DIR / "transformer_deployable"
    p = latest_file_by_glob(base, "run_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError(
            "Could not auto-find final_system_manifest.json under "
            f"{base}. Please pass --deployable_manifest_path explicitly."
        )
    return p


def resolve_default_final_system_manifest_path() -> Path:
    base = EXECUTION_DIR / "final_system"
    p = latest_file_by_glob(base, "final_system_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError(
            "Could not auto-find final_system_manifest.json under "
            f"{base}. Please pass --optimized_manifest_path explicitly."
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
        out = out.merge(feat[["date", "stock", "split"]], on=["date", "stock"], how="left")

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


def default_sector_map() -> Dict[str, str]:
    return {
        "AAPL": "Technology",
        "AMZN": "Consumer",
        "GOOGL": "Technology",
        "JNJ": "Healthcare",
        "JPM": "Financials",
        "META": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "UNH": "Healthcare",
        "V": "Financials",
    }


def load_sector_map(stocks: List[str], sector_map_path: Optional[str]) -> Dict[str, str]:
    sector_map = default_sector_map()
    if sector_map_path:
        p = Path(sector_map_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {sector_map_path}")
        extra = pd.read_csv(p)
        extra.columns = [str(c).strip().lower() for c in extra.columns]
        if "stock" not in extra.columns or "sector" not in extra.columns:
            raise ValueError("sector_map_path CSV must contain columns: stock, sector")
        for _, row in extra.iterrows():
            sector_map[str(row["stock"]).strip()] = str(row["sector"]).strip()
    return {s: sector_map.get(s, "Other") for s in stocks}


def select_static_weights(day_df: pd.DataFrame, top_k: int, min_prob: float) -> Dict[str, float]:
    chosen = day_df.sort_values("pred_prob", ascending=False)
    chosen = chosen[chosen["pred_prob"] >= min_prob].head(top_k)
    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {str(row["stock"]): w for _, row in chosen.iterrows()}


def run_static_backtest(
    pred_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
    horizon: int,
    top_k: int,
    min_prob: float,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    if pred_df.empty:
        return {}, pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns, turns, holds, exposures = [], [], [], []
    max_sector_ws, distinct_sectors, nvda_flags = [], [], []
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        new_w = select_static_weights(day, top_k=top_k, min_prob=min_prob)
        ret_map = day.set_index("stock")["future_return"].to_dict()

        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        sector_weights = {}
        for s, w in new_w.items():
            sec = stock_to_sector.get(s, "Other")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + w

        equity *= (1.0 + net)
        equity_curve.append(equity)

        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))
        max_sector_ws.append(max(sector_weights.values()) if sector_weights else 0.0)
        distinct_sectors.append(float(len(sector_weights)))
        nvda_flags.append(1.0 if "NVDA" in new_w else 0.0)

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
            "max_sector_weight": max(sector_weights.values()) if sector_weights else 0.0,
            "n_distinct_sectors": len(sector_weights),
            "nvda_in_portfolio": int("NVDA" in new_w),
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
        "avg_max_sector_weight": float(np.mean(max_sector_ws)) if max_sector_ws else 0.0,
        "avg_distinct_sectors": float(np.mean(distinct_sectors)) if distinct_sectors else 0.0,
        "nvda_presence_rate": float(np.mean(nvda_flags)) if nvda_flags else 0.0,
    }
    return metrics, pd.DataFrame(rows)


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

    portfolio = sorted(list(set(portfolio)), key=lambda x: float(prob_map.get(x, -1.0)), reverse=True)[:top_k]
    debug = {
        "n_holdings": len(portfolio),
        "kept_existing": len([s for s in portfolio if s in current_set]),
        "newly_added": len([s for s in portfolio if s not in current_set]),
    }
    return portfolio, debug


def run_sticky_backtest(
    pred_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
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
        return {}, pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    current_positions = {}
    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns, turns, holds, exposures = [], [], [], []
    max_sector_ws, distinct_sectors, nvda_flags = [], [], []
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

        sector_weights = {}
        for s, w in new_w.items():
            sec = stock_to_sector.get(s, "Other")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + w

        equity *= (1.0 + net)
        equity_curve.append(equity)

        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        exposures.append(float(sum(new_w.values())))
        max_sector_ws.append(max(sector_weights.values()) if sector_weights else 0.0)
        distinct_sectors.append(float(len(sector_weights)))
        nvda_flags.append(1.0 if "NVDA" in new_w else 0.0)

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
            "max_sector_weight": max(sector_weights.values()) if sector_weights else 0.0,
            "n_distinct_sectors": len(sector_weights),
            "nvda_in_portfolio": int("NVDA" in new_w),
            "kept_existing": debug["kept_existing"],
            "newly_added": debug["newly_added"],
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
        "avg_max_sector_weight": float(np.mean(max_sector_ws)) if max_sector_ws else 0.0,
        "avg_distinct_sectors": float(np.mean(distinct_sectors)) if distinct_sectors else 0.0,
        "nvda_presence_rate": float(np.mean(nvda_flags)) if nvda_flags else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def search_static_baseline(
    val_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
    horizon: int,
    cost_bps: float,
    topk_grid: List[int],
    min_prob_grid: List[float],
) -> Tuple[Dict, pd.DataFrame]:
    rows = []
    for top_k in topk_grid:
        for min_prob in min_prob_grid:
            metrics, _ = run_static_backtest(
                pred_df=val_df,
                stock_to_sector=stock_to_sector,
                horizon=horizon,
                top_k=top_k,
                min_prob=min_prob,
                transaction_cost_bps=cost_bps,
            )
            score = (
                2.5 * metrics["sharpe"]
                + 2.0 * metrics["cumulative_return"]
                + 1.0 * metrics["max_drawdown"]
                - 0.5 * metrics["avg_turnover"]
            )
            rows.append({
                "top_k": int(top_k),
                "min_prob": float(min_prob),
                "selection_score": float(score),
                **metrics,
            })

    grid_df = pd.DataFrame(rows).sort_values(
        ["selection_score", "cumulative_return", "sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = grid_df.iloc[0]
    best_cfg = {
        "mode": "static_topk",
        "top_k": int(best["top_k"]),
        "min_prob": float(best["min_prob"]),
        "threshold": 0.50,
    }
    return best_cfg, grid_df


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(description="Sticky execution ablation on top of fixed Transformer predictions.")
    parser.add_argument("--predictions_path", type=str, default="")
    parser.add_argument("--feature_data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--deployable_manifest_path", type=str, default="")
    parser.add_argument("--optimized_manifest_path", type=str, default="")
    parser.add_argument("--sector_map_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(ABLATIONS_DIR / "sticky_execution"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--cost_bps", type=float, default=10.0)

    parser.add_argument("--static_topk_grid", type=str, default="2,3,4")
    parser.add_argument("--static_min_prob_grid", type=str, default="0.54,0.56,0.58,0.60")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path) if args.predictions_path else resolve_default_predictions_path()
    feature_data_path = Path(args.feature_data_path)
    deployable_manifest_path = Path(args.deployable_manifest_path) if args.deployable_manifest_path else resolve_default_deployable_manifest_path()
    optimized_manifest_path = Path(args.optimized_manifest_path) if args.optimized_manifest_path else resolve_default_final_system_manifest_path()

    print(f"Using predictions: {predictions_path}")
    print(f"Using feature data: {feature_data_path}")
    print(f"Using deployable manifest: {deployable_manifest_path}")
    print(f"Using optimized execution manifest: {optimized_manifest_path}")

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()

    universe = sorted(pred["stock"].astype(str).unique().tolist())
    stock_to_sector = load_sector_map(universe, args.sector_map_path)

    deployable_manifest = safe_read_json(deployable_manifest_path)
    optimized_manifest = safe_read_json(optimized_manifest_path)

    frozen_cfg = deployable_manifest.get("frozen_config", {})
    if not frozen_cfg:
        raise ValueError("Deployable manifest missing frozen_config.")

    execution_cfg = optimized_manifest.get("execution_layer", {})
    if not execution_cfg:
        raise ValueError("Optimized execution manifest missing execution_layer.")

    frozen_cfg = {
        "mode": "topk",
        "top_k": int(frozen_cfg["top_k"]),
        "min_prob": float(frozen_cfg["min_prob"]),
        "threshold": float(frozen_cfg.get("threshold", 0.50)),
    }

    validation_static_cfg, static_grid = search_static_baseline(
        val_df=val_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        cost_bps=args.cost_bps,
        topk_grid=[int(x.strip()) for x in args.static_topk_grid.split(",") if x.strip()],
        min_prob_grid=[float(x.strip()) for x in args.static_min_prob_grid.split(",") if x.strip()],
    )

    frozen_val_metrics, frozen_val_actions = run_static_backtest(
        pred_df=val_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=frozen_cfg["top_k"],
        min_prob=frozen_cfg["min_prob"],
        transaction_cost_bps=args.cost_bps,
    )
    frozen_test_metrics, frozen_test_actions = run_static_backtest(
        pred_df=test_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=frozen_cfg["top_k"],
        min_prob=frozen_cfg["min_prob"],
        transaction_cost_bps=args.cost_bps,
    )

    static_val_metrics, static_val_actions = run_static_backtest(
        pred_df=val_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=validation_static_cfg["top_k"],
        min_prob=validation_static_cfg["min_prob"],
        transaction_cost_bps=args.cost_bps,
    )
    static_test_metrics, static_test_actions = run_static_backtest(
        pred_df=test_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=validation_static_cfg["top_k"],
        min_prob=validation_static_cfg["min_prob"],
        transaction_cost_bps=args.cost_bps,
    )

    sticky_val_metrics, sticky_val_actions = run_sticky_backtest(
        pred_df=val_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=int(execution_cfg["top_k"]),
        entry_prob=float(execution_cfg["entry_prob"]),
        exit_prob=float(execution_cfg["exit_prob"]),
        switch_buffer=float(execution_cfg.get("switch_buffer", 0.0)),
        min_hold_periods=int(execution_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(execution_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.cost_bps,
    )
    sticky_test_metrics, sticky_test_actions = run_sticky_backtest(
        pred_df=test_df,
        stock_to_sector=stock_to_sector,
        horizon=horizon,
        top_k=int(execution_cfg["top_k"]),
        entry_prob=float(execution_cfg["entry_prob"]),
        exit_prob=float(execution_cfg["exit_prob"]),
        switch_buffer=float(execution_cfg.get("switch_buffer", 0.0)),
        min_hold_periods=int(execution_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(execution_cfg["max_new_names_per_rebalance"]),
        transaction_cost_bps=args.cost_bps,
    )

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    static_grid.to_csv(out_root / "static_baseline_validation_search.csv", index=False)

    save_df_with_dates(frozen_val_actions, out_root / "frozen_static_val_actions.csv")
    save_df_with_dates(frozen_test_actions, out_root / "frozen_static_test_actions.csv")
    save_df_with_dates(static_val_actions, out_root / "validation_static_val_actions.csv")
    save_df_with_dates(static_test_actions, out_root / "validation_static_test_actions.csv")
    save_df_with_dates(sticky_val_actions, out_root / "sticky_val_actions.csv")
    save_df_with_dates(sticky_test_actions, out_root / "sticky_test_actions.csv")

    # Canonical files for downstream bootstrap scripts
    save_df_with_dates(frozen_test_actions, out_root / "static_test_actions.csv")

    rows = [
        {"strategy": "frozen_static_baseline", "split": "val", **frozen_val_metrics},
        {"strategy": "frozen_static_baseline", "split": "test", **frozen_test_metrics},
        {"strategy": "validation_selected_static", "split": "val", **static_val_metrics},
        {"strategy": "validation_selected_static", "split": "test", **static_test_metrics},
        {"strategy": "optimized_sticky_execution", "split": "val", **sticky_val_metrics},
        {"strategy": "optimized_sticky_execution", "split": "test", **sticky_test_metrics},
    ]
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(out_root / "ablation_metrics.csv", index=False)

    primary_compare_df = pd.DataFrame([
        {"strategy": "frozen_static_baseline", "split": "val", **frozen_val_metrics},
        {"strategy": "frozen_static_baseline", "split": "test", **frozen_test_metrics},
        {"strategy": "optimized_sticky_execution", "split": "val", **sticky_val_metrics},
        {"strategy": "optimized_sticky_execution", "split": "test", **sticky_test_metrics},
    ])
    primary_compare_df.to_csv(out_root / "primary_ablation_metrics.csv", index=False)

    summary = {
        "cost_bps": float(args.cost_bps),
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "source_deployable_manifest_path": str(deployable_manifest_path.resolve()),
        "source_optimized_manifest_path": str(optimized_manifest_path.resolve()),
        "frozen_static_cfg": frozen_cfg,
        "validation_selected_static_cfg": validation_static_cfg,
        "optimized_sticky_cfg": execution_cfg,
        "frozen_static_val_metrics": frozen_val_metrics,
        "frozen_static_test_metrics": frozen_test_metrics,
        "validation_selected_static_val_metrics": static_val_metrics,
        "validation_selected_static_test_metrics": static_test_metrics,
        "sticky_val_metrics": sticky_val_metrics,
        "sticky_test_metrics": sticky_test_metrics,
    }
    with open(out_root / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Sticky execution ablation saved to: {out_root}")
    print("\nFrozen static cfg:")
    print(json.dumps(frozen_cfg, ensure_ascii=False, indent=2))
    print("\nValidation-selected static cfg:")
    print(json.dumps(validation_static_cfg, ensure_ascii=False, indent=2))
    print("\nOptimized sticky cfg:")
    print(json.dumps(execution_cfg, ensure_ascii=False, indent=2))

    print("\nFrozen static validation metrics:")
    print(json.dumps(frozen_val_metrics, ensure_ascii=False, indent=2))
    print("\nFrozen static test metrics:")
    print(json.dumps(frozen_test_metrics, ensure_ascii=False, indent=2))

    print("\nOptimized sticky validation metrics:")
    print(json.dumps(sticky_val_metrics, ensure_ascii=False, indent=2))
    print("\nOptimized sticky test metrics:")
    print(json.dumps(sticky_test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
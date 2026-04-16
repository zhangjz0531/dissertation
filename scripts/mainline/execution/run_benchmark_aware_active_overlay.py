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


def resolve_default_final_system_manifest_path() -> Path:
    base = EXECUTION_DIR / "final_system"
    p = latest_file_by_glob(base, "final_system_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError(
            f"Could not auto-find latest final_system manifest under {base}. "
            f"Please pass --final_system_manifest_path explicitly."
        )
    return p


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
            on=["date", "stock", "split"], how="left"
        )

    required_cols = ["date", "stock", "split", "pred_prob", "future_return"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Predictions data missing required columns after normalization: {missing}")
    if out["future_return"].isna().any():
        raise ValueError("Predictions data contains missing future_return after normalization.")

    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def choose_portfolio_static(day_df: pd.DataFrame, top_k: int, min_prob: float) -> List[str]:
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

    portfolio = list(dict.fromkeys(forced_keep + kept))[:top_k]
    current_set = set(current_positions.keys())
    new_candidates = [
        s for s in day["stock"].astype(str).tolist()
        if s not in current_set and float(prob_map.get(s, -1.0)) >= entry_prob
    ]

    added_new = 0
    for s in new_candidates:
        if len(portfolio) >= top_k or added_new >= max_new_names_per_rebalance:
            break
        portfolio.append(s)
        added_new += 1

    if len(portfolio) == top_k:
        for s in new_candidates:
            if s in portfolio or added_new >= max_new_names_per_rebalance:
                continue
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


def run_static_backtest(pred_df: pd.DataFrame, horizon: int, top_k: int, min_prob: float, transaction_cost_bps: float):
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
        new_w = {} if len(names) == 0 else {s: 1.0 / len(names) for s in names}
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
            "date": pd.Timestamp(dt), "strategy": "frozen_static_baseline",
            "selected_stocks": "|".join(sorted(new_w.keys())), "n_holdings": len(new_w),
            "gross_return": gross, "net_return": net, "turnover": turn,
            "equity": equity, "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
        })
        prev_w = new_w

    return compute_metrics_from_series(period_returns, turns, holds, exposures, horizon), pd.DataFrame(rows)


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
):
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
            day_df=day, current_positions=current_positions, top_k=top_k,
            entry_prob=entry_prob, exit_prob=exit_prob, switch_buffer=switch_buffer,
            min_hold_periods=min_hold_periods, max_new_names_per_rebalance=max_new_names_per_rebalance,
        )
        new_w = {} if len(selected_names) == 0 else {s: 1.0 / len(selected_names) for s in selected_names}
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
            "date": pd.Timestamp(dt), "strategy": "optimized_sticky_execution",
            "selected_stocks": "|".join(sorted(new_w.keys())), "n_holdings": len(new_w),
            "gross_return": gross, "net_return": net, "turnover": turn,
            "equity": equity, "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(new_w.values())),
            "kept_existing": debug["kept_existing"], "newly_added": debug["newly_added"],
            "forced_keep_count": debug["forced_keep_count"], "entry_candidates_count": debug["entry_candidates_count"],
        })
        next_positions = {}
        for s in new_w.keys():
            next_positions[s] = {"held_for": int(current_positions[s]["held_for"]) + 1} if s in current_positions else {"held_for": 1}
        current_positions = next_positions
        prev_w = new_w

    return compute_metrics_from_series(period_returns, turns, holds, exposures, horizon), pd.DataFrame(rows)


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(w.values()))
    if total <= 0:
        return {k: 0.0 for k in w.keys()}
    return {k: float(v) / total for k, v in w.items()}


def drift_weights(weights: Dict[str, float], ret_map: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    grown = {k: float(v) * (1.0 + float(ret_map.get(k, 0.0))) for k, v in weights.items()}
    total = float(sum(grown.values()))
    if total <= 0:
        return {k: 0.0 for k in grown.keys()}
    return {k: float(v) / total for k, v in grown.items()}


def allocate_budget_equal(cap_map: Dict[str, float], total_budget: float) -> Dict[str, float]:
    remaining = float(total_budget)
    caps = {k: max(0.0, float(v)) for k, v in cap_map.items()}
    alloc = {k: 0.0 for k in cap_map.keys()}
    active = [k for k, v in caps.items() if v > 1e-12]
    while remaining > 1e-12 and active:
        share = remaining / len(active)
        next_active = []
        used = 0.0
        for k in active:
            add = min(share, caps[k])
            alloc[k] += add
            caps[k] -= add
            used += add
            if caps[k] > 1e-12:
                next_active.append(k)
        if used <= 1e-12:
            break
        remaining -= used
        active = next_active
    return alloc


def build_active_overlay_target(
    benchmark_w: Dict[str, float],
    day_df: pd.DataFrame,
    n_overweight: int,
    n_underweight: int,
    tilt_budget: float,
    min_weight: float,
    max_weight: float,
):
    if not benchmark_w:
        return {}, {"effective_budget": 0.0, "top_names": [], "bottom_names": [], "over_cap_total": 0.0, "under_cap_total": 0.0}

    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    universe = [s for s in day["stock"].astype(str).tolist() if s in benchmark_w]
    if len(universe) == 0:
        return benchmark_w.copy(), {"effective_budget": 0.0, "top_names": [], "bottom_names": [], "over_cap_total": 0.0, "under_cap_total": 0.0}

    base_w = {s: float(benchmark_w.get(s, 0.0)) for s in universe}
    n = len(universe)
    n_over = max(0, min(int(n_overweight), n))
    n_under = max(0, min(int(n_underweight), n))
    ordered = universe
    top_names = ordered[:n_over]
    bottom_names = [s for s in ordered[-n_under:] if s not in set(top_names)] if n_under > 0 else []

    if len(top_names) == 0 or len(bottom_names) == 0 or tilt_budget <= 1e-12:
        return normalize_weights(base_w), {"effective_budget": 0.0, "top_names": top_names, "bottom_names": bottom_names, "over_cap_total": 0.0, "under_cap_total": 0.0}

    over_caps = {s: max(0.0, float(max_weight) - base_w.get(s, 0.0)) for s in top_names}
    under_caps = {s: max(0.0, base_w.get(s, 0.0) - float(min_weight)) for s in bottom_names}
    over_cap_total = float(sum(over_caps.values()))
    under_cap_total = float(sum(under_caps.values()))
    effective_budget = float(min(float(tilt_budget), over_cap_total, under_cap_total))

    if effective_budget <= 1e-12:
        return normalize_weights(base_w), {
            "effective_budget": 0.0, "top_names": top_names, "bottom_names": bottom_names,
            "over_cap_total": over_cap_total, "under_cap_total": under_cap_total,
        }

    over_alloc = allocate_budget_equal(over_caps, effective_budget)
    under_alloc = allocate_budget_equal(under_caps, effective_budget)
    target = base_w.copy()
    for s, v in over_alloc.items():
        target[s] = target.get(s, 0.0) + float(v)
    for s, v in under_alloc.items():
        target[s] = target.get(s, 0.0) - float(v)

    target = {k: min(float(max_weight), max(float(min_weight), float(v))) for k, v in target.items()}
    target = normalize_weights(target)
    return target, {
        "effective_budget": effective_budget, "top_names": top_names, "bottom_names": bottom_names,
        "over_cap_total": over_cap_total, "under_cap_total": under_cap_total,
    }


def apply_turnover_cap(pretrade_w: Dict[str, float], target_w: Dict[str, float], turnover_cap: float) -> Dict[str, float]:
    if turnover_cap is None or turnover_cap <= 0:
        return target_w.copy()
    current_turn = turnover(pretrade_w, target_w)
    if current_turn <= turnover_cap + 1e-12 or current_turn <= 1e-12:
        return target_w.copy()
    scale = float(turnover_cap) / float(current_turn)
    names = set(pretrade_w.keys()) | set(target_w.keys())
    out = {n: float(pretrade_w.get(n, 0.0)) + scale * (float(target_w.get(n, 0.0)) - float(pretrade_w.get(n, 0.0))) for n in names}
    out = {k: max(0.0, float(v)) for k, v in out.items()}
    return normalize_weights(out)


def run_universe_equal_weight_buy_and_hold(pred_df: pd.DataFrame, horizon: int, transaction_cost_bps: float):
    if pred_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0
    first_day = df[df["date"] == rebalance_dates[0]].copy().sort_values("stock").reset_index(drop=True)
    universe_names = first_day["stock"].astype(str).tolist()
    if len(universe_names) == 0:
        raise ValueError("No stocks found on the first rebalance date.")
    current_w = {s: 1.0 / len(universe_names) for s in universe_names}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []
    initialized = False

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("stock").reset_index(drop=True)
        ret_map = day.set_index("stock")["future_return"].to_dict()
        pretrade_w = current_w.copy()
        turn = float(sum(current_w.values())) if not initialized else 0.0
        initialized = True
        gross = float(sum(pretrade_w.get(s, 0.0) * ret_map.get(s, 0.0) for s in pretrade_w.keys()))
        net = gross - tc * turn
        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(pretrade_w))
        exposures.append(float(sum(pretrade_w.values())))
        rows.append({
            "date": pd.Timestamp(dt), "strategy": "universe_equal_weight_buy_hold",
            "selected_stocks": "|".join(sorted(pretrade_w.keys())), "n_holdings": len(pretrade_w),
            "gross_return": gross, "net_return": net, "turnover": turn,
            "equity": equity, "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(pretrade_w.values())),
        })
        current_w = drift_weights(pretrade_w, ret_map)

    return compute_metrics_from_series(period_returns, turns, holds, exposures, horizon), pd.DataFrame(rows)


def run_benchmark_aware_active_overlay(
    pred_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    n_overweight: int,
    n_underweight: int,
    tilt_budget: float,
    min_weight: float,
    max_weight: float,
    turnover_cap: float,
):
    if pred_df.empty:
        return compute_metrics_from_series([], [], [], [], horizon), pd.DataFrame()
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0
    first_day = df[df["date"] == rebalance_dates[0]].copy().sort_values("stock").reset_index(drop=True)
    universe_names = first_day["stock"].astype(str).tolist()
    if len(universe_names) == 0:
        raise ValueError("No stocks found on the first rebalance date.")
    current_benchmark_w = {s: 1.0 / len(universe_names) for s in universe_names}
    current_overlay_w: Dict[str, float] = {}
    period_returns, turns, holds, exposures = [], [], [], []
    equity = 1.0
    equity_curve = [equity]
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        ret_map = day.set_index("stock")["future_return"].to_dict()
        benchmark_start_w = current_benchmark_w.copy()
        pretrade_overlay_w = current_overlay_w.copy() if len(current_overlay_w) > 0 else {}
        target_w_raw, debug = build_active_overlay_target(
            benchmark_w=benchmark_start_w, day_df=day, n_overweight=n_overweight,
            n_underweight=n_underweight, tilt_budget=tilt_budget,
            min_weight=min_weight, max_weight=max_weight,
        )
        posttrade_w = target_w_raw.copy() if len(pretrade_overlay_w) == 0 else apply_turnover_cap(pretrade_w=pretrade_overlay_w, target_w=target_w_raw, turnover_cap=turnover_cap)
        turn = turnover(pretrade_overlay_w, posttrade_w)
        gross = float(sum(posttrade_w.get(s, 0.0) * ret_map.get(s, 0.0) for s in posttrade_w.keys()))
        net = gross - tc * turn
        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len([s for s, w in posttrade_w.items() if w > 1e-12]))
        exposures.append(float(sum(posttrade_w.values())))
        active_share = 0.5 * turnover(benchmark_start_w, posttrade_w)
        rows.append({
            "date": pd.Timestamp(dt), "strategy": "benchmark_aware_active_overlay",
            "selected_stocks": "|".join(sorted([s for s, w in posttrade_w.items() if w > 1e-12])),
            "n_holdings": len([s for s, w in posttrade_w.items() if w > 1e-12]),
            "gross_return": gross, "net_return": net, "turnover": turn,
            "equity": equity, "drawdown": equity / max(equity_curve) - 1.0,
            "exposure": float(sum(posttrade_w.values())),
            "active_share_proxy": active_share, "effective_budget": float(debug["effective_budget"]),
            "top_names": "|".join(debug["top_names"]), "bottom_names": "|".join(debug["bottom_names"]),
        })
        current_overlay_w = drift_weights(posttrade_w, ret_map)
        current_benchmark_w = drift_weights(benchmark_start_w, ret_map)

    metrics = compute_metrics_from_series(period_returns, turns, holds, exposures, horizon)
    metrics["avg_active_share_proxy"] = float(pd.DataFrame(rows)["active_share_proxy"].mean()) if len(rows) > 0 else 0.0
    return metrics, pd.DataFrame(rows)


def compute_active_relative_metrics(overlay_actions: pd.DataFrame, benchmark_actions: pd.DataFrame, horizon: int) -> Dict[str, float]:
    a = overlay_actions.copy()
    b = benchmark_actions.copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    merged = a[["date", "net_return"]].merge(
        b[["date", "net_return"]], on="date", how="inner", suffixes=("_overlay", "_benchmark")
    ).sort_values("date").reset_index(drop=True)
    if merged.empty:
        return {"active_cumret_gap": 0.0, "active_win_rate": 0.0, "information_ratio": 0.0}
    active_returns = (merged["net_return_overlay"] - merged["net_return_benchmark"]).astype(float).values
    info_ratio = sharpe_ratio(active_returns.tolist(), horizon)
    overlay_eq = float((1.0 + merged["net_return_overlay"]).prod() - 1.0)
    bench_eq = float((1.0 + merged["net_return_benchmark"]).prod() - 1.0)
    return {
        "active_cumret_gap": float(overlay_eq - bench_eq),
        "active_win_rate": float((active_returns > 0).mean()),
        "information_ratio": float(info_ratio),
    }


def search_overlay_on_validation(
    val_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    tilt_budget_grid: List[float],
    n_overweight_grid: List[int],
    n_underweight_grid: List[int],
    min_weight_grid: List[float],
    max_weight_grid: List[float],
    turnover_cap_grid: List[float],
    mdd_floor: float,
):
    benchmark_metrics, benchmark_actions = run_universe_equal_weight_buy_and_hold(
        pred_df=val_df, horizon=horizon, transaction_cost_bps=transaction_cost_bps
    )
    rows = []
    for tilt_budget in tilt_budget_grid:
        for n_overweight in n_overweight_grid:
            for n_underweight in n_underweight_grid:
                for min_weight in min_weight_grid:
                    for max_weight in max_weight_grid:
                        if max_weight <= min_weight:
                            continue
                        for turnover_cap in turnover_cap_grid:
                            overlay_metrics, overlay_actions = run_benchmark_aware_active_overlay(
                                pred_df=val_df, horizon=horizon, transaction_cost_bps=transaction_cost_bps,
                                n_overweight=n_overweight, n_underweight=n_underweight,
                                tilt_budget=tilt_budget, min_weight=min_weight,
                                max_weight=max_weight, turnover_cap=turnover_cap,
                            )
                            rel = compute_active_relative_metrics(
                                overlay_actions=overlay_actions, benchmark_actions=benchmark_actions, horizon=horizon
                            )
                            beats_benchmark = bool(rel["active_cumret_gap"] > 0)
                            mdd_ok = bool(overlay_metrics["max_drawdown"] >= mdd_floor)
                            selection_score = (
                                2.5 * float(rel["active_cumret_gap"])
                                + 1.0 * float(rel["information_ratio"])
                                + 0.5 * float(overlay_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"])
                                + 0.2 * float(rel["active_win_rate"])
                                - 0.15 * float(overlay_metrics["avg_turnover"])
                            )
                            rows.append({
                                "tilt_budget": float(tilt_budget),
                                "n_overweight": int(n_overweight),
                                "n_underweight": int(n_underweight),
                                "min_weight": float(min_weight),
                                "max_weight": float(max_weight),
                                "turnover_cap": float(turnover_cap),
                                "beats_benchmark": beats_benchmark,
                                "mdd_ok": mdd_ok,
                                "selection_score": float(selection_score),
                                "val_overlay_cumret": float(overlay_metrics["cumulative_return"]),
                                "val_overlay_sharpe": float(overlay_metrics["sharpe"]),
                                "val_overlay_mdd": float(overlay_metrics["max_drawdown"]),
                                "val_overlay_turnover": float(overlay_metrics["avg_turnover"]),
                                "val_overlay_avg_holdings": float(overlay_metrics["avg_holdings"]),
                                "val_overlay_active_share_proxy": float(overlay_metrics.get("avg_active_share_proxy", 0.0)),
                                "val_benchmark_cumret": float(benchmark_metrics["cumulative_return"]),
                                "val_benchmark_sharpe": float(benchmark_metrics["sharpe"]),
                                "val_benchmark_mdd": float(benchmark_metrics["max_drawdown"]),
                                "active_cumret_gap": float(rel["active_cumret_gap"]),
                                "information_ratio": float(rel["information_ratio"]),
                                "active_win_rate": float(rel["active_win_rate"]),
                            })

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        raise RuntimeError("Overlay validation search grid is empty.")

    valid = grid_df[(grid_df["beats_benchmark"] == True) & (grid_df["mdd_ok"] == True)].copy()
    if len(valid) > 0:
        best = valid.sort_values(
            ["selection_score", "active_cumret_gap", "information_ratio", "val_overlay_sharpe", "val_overlay_turnover"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
    else:
        best = grid_df.sort_values(
            ["beats_benchmark", "mdd_ok", "selection_score", "active_cumret_gap", "information_ratio", "val_overlay_turnover"],
            ascending=[False, False, False, False, False, True],
        ).iloc[0]

    best_cfg = {
        "mode": "benchmark_aware_active_overlay",
        "tilt_budget": float(best["tilt_budget"]),
        "n_overweight": int(best["n_overweight"]),
        "n_underweight": int(best["n_underweight"]),
        "min_weight": float(best["min_weight"]),
        "max_weight": float(best["max_weight"]),
        "turnover_cap": float(best["turnover_cap"]),
    }
    baseline = {"validation_benchmark_metrics": benchmark_metrics}
    return best_cfg, best.to_dict(), grid_df, baseline


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def main():
    ensure_all_core_dirs()
    parser = argparse.ArgumentParser(
        description="Run validation-selected benchmark-aware active overlay on top of the latest final system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--final_system_manifest_path", type=str, default="")
    parser.add_argument("--predictions_path", type=str, default="")
    parser.add_argument("--feature_data_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(EXECUTION_DIR / "benchmark_aware_overlay"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--transaction_cost_bps", type=float, default=-1.0)
    parser.add_argument("--tilt_budget_grid", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--n_overweight_grid", type=str, default="2,3,4")
    parser.add_argument("--n_underweight_grid", type=str, default="2,3,4")
    parser.add_argument("--min_weight_grid", type=str, default="0.00,0.02")
    parser.add_argument("--max_weight_grid", type=str, default="0.15,0.20")
    parser.add_argument("--turnover_cap_grid", type=str, default="0.10,0.20,0.40,1.00")
    parser.add_argument("--mdd_floor", type=float, default=-0.30)
    args = parser.parse_args()

    final_system_manifest_path = Path(args.final_system_manifest_path) if args.final_system_manifest_path else resolve_default_final_system_manifest_path()
    final_manifest = safe_read_json(final_system_manifest_path)
    predictions_path = Path(args.predictions_path) if args.predictions_path else Path(final_manifest.get("source_predictions_path", ""))
    feature_data_path = Path(args.feature_data_path) if args.feature_data_path else Path(final_manifest.get("source_feature_data_path", str(MAIN_H5_DATA)))

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions path not found: {predictions_path}. Pass --predictions_path explicitly if needed.")
    if not feature_data_path.exists():
        raise FileNotFoundError(f"Feature data path not found: {feature_data_path}. Pass --feature_data_path explicitly if needed.")

    print(f"Using final system manifest: {final_system_manifest_path}")
    print(f"Using predictions: {predictions_path}")
    print(f"Using feature data: {feature_data_path}")

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)

    horizon = int(final_manifest.get("horizon", infer_horizon_from_feature_df(feat)))
    default_cost_bps = float(final_manifest.get("transaction_cost_bps", 10.0))
    tc_bps = float(args.transaction_cost_bps) if float(args.transaction_cost_bps) >= 0 else default_cost_bps

    frozen_cfg = final_manifest.get("frozen_prediction_layer", {})
    sticky_cfg = final_manifest.get("execution_layer", {})
    if not frozen_cfg:
        raise ValueError("final_system_manifest is missing 'frozen_prediction_layer'.")
    if not sticky_cfg:
        raise ValueError("final_system_manifest is missing 'execution_layer'.")

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()
    if val_df.empty or test_df.empty:
        raise ValueError("Predictions file must contain non-empty val and test splits.")

    best_cfg, best_row, grid_df, baseline = search_overlay_on_validation(
        val_df=val_df,
        horizon=horizon,
        transaction_cost_bps=tc_bps,
        tilt_budget_grid=parse_float_grid(args.tilt_budget_grid),
        n_overweight_grid=parse_int_grid(args.n_overweight_grid),
        n_underweight_grid=parse_int_grid(args.n_underweight_grid),
        min_weight_grid=parse_float_grid(args.min_weight_grid),
        max_weight_grid=parse_float_grid(args.max_weight_grid),
        turnover_cap_grid=parse_float_grid(args.turnover_cap_grid),
        mdd_floor=float(args.mdd_floor),
    )

    val_bh_metrics, val_bh_actions = run_universe_equal_weight_buy_and_hold(val_df, horizon, tc_bps)
    test_bh_metrics, test_bh_actions = run_universe_equal_weight_buy_and_hold(test_df, horizon, tc_bps)
    val_overlay_metrics, val_overlay_actions = run_benchmark_aware_active_overlay(
        val_df, horizon, tc_bps,
        n_overweight=int(best_cfg["n_overweight"]), n_underweight=int(best_cfg["n_underweight"]),
        tilt_budget=float(best_cfg["tilt_budget"]), min_weight=float(best_cfg["min_weight"]),
        max_weight=float(best_cfg["max_weight"]), turnover_cap=float(best_cfg["turnover_cap"]),
    )
    test_overlay_metrics, test_overlay_actions = run_benchmark_aware_active_overlay(
        test_df, horizon, tc_bps,
        n_overweight=int(best_cfg["n_overweight"]), n_underweight=int(best_cfg["n_underweight"]),
        tilt_budget=float(best_cfg["tilt_budget"]), min_weight=float(best_cfg["min_weight"]),
        max_weight=float(best_cfg["max_weight"]), turnover_cap=float(best_cfg["turnover_cap"]),
    )
    val_static_metrics, val_static_actions = run_static_backtest(val_df, horizon, int(frozen_cfg["top_k"]), float(frozen_cfg["min_prob"]), tc_bps)
    test_static_metrics, test_static_actions = run_static_backtest(test_df, horizon, int(frozen_cfg["top_k"]), float(frozen_cfg["min_prob"]), tc_bps)
    val_sticky_metrics, val_sticky_actions = run_sticky_backtest(
        val_df, horizon, int(sticky_cfg["top_k"]), float(sticky_cfg["entry_prob"]), float(sticky_cfg["exit_prob"]),
        float(sticky_cfg["switch_buffer"]), int(sticky_cfg["min_hold_periods"]), int(sticky_cfg["max_new_names_per_rebalance"]), tc_bps,
    )
    test_sticky_metrics, test_sticky_actions = run_sticky_backtest(
        test_df, horizon, int(sticky_cfg["top_k"]), float(sticky_cfg["entry_prob"]), float(sticky_cfg["exit_prob"]),
        float(sticky_cfg["switch_buffer"]), int(sticky_cfg["min_hold_periods"]), int(sticky_cfg["max_new_names_per_rebalance"]), tc_bps,
    )

    comparison_df = pd.DataFrame([
        {"strategy": "universe_equal_weight_buy_hold", "split": "val", **val_bh_metrics},
        {"strategy": "benchmark_aware_active_overlay", "split": "val", **val_overlay_metrics},
        {"strategy": "optimized_sticky_execution", "split": "val", **val_sticky_metrics},
        {"strategy": "frozen_static_baseline", "split": "val", **val_static_metrics},
        {"strategy": "universe_equal_weight_buy_hold", "split": "test", **test_bh_metrics},
        {"strategy": "benchmark_aware_active_overlay", "split": "test", **test_overlay_metrics},
        {"strategy": "optimized_sticky_execution", "split": "test", **test_sticky_metrics},
        {"strategy": "frozen_static_baseline", "split": "test", **test_static_metrics},
    ])

    val_rel = compute_active_relative_metrics(val_overlay_actions, val_bh_actions, horizon)
    test_rel = compute_active_relative_metrics(test_overlay_actions, test_bh_actions, horizon)

    manifest = {
        "mode": "benchmark_aware_active_overlay",
        "source_final_system_manifest_path": str(final_system_manifest_path.resolve()),
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "horizon": int(horizon),
        "transaction_cost_bps": float(tc_bps),
        "frozen_static_cfg": frozen_cfg,
        "optimized_sticky_cfg": sticky_cfg,
        "overlay_cfg": best_cfg,
        "selection_policy": "validation_only_vs_equal_weight_buy_hold_benchmark",
        "validation_benchmark_metrics": baseline["validation_benchmark_metrics"],
        "best_search_row": best_row,
        "relative_results": {
            "validation_overlay_vs_benchmark": val_rel,
            "test_overlay_vs_benchmark": test_rel,
        },
    }

    summary = {
        "overlay_cfg": best_cfg,
        "validation_overlay_vs_benchmark": val_rel,
        "test_overlay_vs_benchmark": test_rel,
        "strategy_comparison_val_test": comparison_df.to_dict(orient="records"),
    }

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_df.to_csv(out_root / "overlay_validation_search_grid.csv", index=False)
    comparison_df.to_csv(out_root / "strategy_comparison_val_test.csv", index=False)
    save_df_with_dates(val_bh_actions, out_root / "val_benchmark_buy_hold_actions.csv")
    save_df_with_dates(test_bh_actions, out_root / "test_benchmark_buy_hold_actions.csv")
    save_df_with_dates(val_overlay_actions, out_root / "val_benchmark_aware_overlay_actions.csv")
    save_df_with_dates(test_overlay_actions, out_root / "test_benchmark_aware_overlay_actions.csv")
    save_df_with_dates(val_sticky_actions, out_root / "val_sticky_reference_actions.csv")
    save_df_with_dates(test_sticky_actions, out_root / "test_sticky_reference_actions.csv")
    save_df_with_dates(val_static_actions, out_root / "val_static_reference_actions.csv")
    save_df_with_dates(test_static_actions, out_root / "test_static_reference_actions.csv")

    with open(out_root / "benchmark_aware_overlay_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(out_root / "benchmark_aware_overlay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["resolved_final_system_manifest_path"] = str(final_system_manifest_path.resolve())
    run_config["resolved_predictions_path"] = str(predictions_path.resolve())
    run_config["resolved_feature_data_path"] = str(feature_data_path.resolve())
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Benchmark-aware active overlay saved to: {out_root}")
    print("\nSelected overlay config:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("\nValidation overlay vs benchmark:")
    print(json.dumps(val_rel, ensure_ascii=False, indent=2))
    print("\nTest overlay vs benchmark:")
    print(json.dumps(test_rel, ensure_ascii=False, indent=2))
    print("\nStrategy comparison (val/test):")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()

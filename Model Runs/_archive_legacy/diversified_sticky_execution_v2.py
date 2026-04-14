
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURE_DATA = r"D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv"
DEFAULT_FIXED_PREDICTIONS = r"D:\python\dissertation\Model Runs\final_run_20260413\fixed_predictions_transformer_v5\transformer_predictions_all_splits_fixed.csv"
DEFAULT_BASE_OPT_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413\final_execution_optimization_valonly"
DEFAULT_OUT_DIR = r"D:\python\dissertation\Model Runs\final_run_20260413"


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


def parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_grid(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def default_sector_map() -> Dict[str, str]:
    return {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
        "GOOG": "Technology", "META": "Technology", "AMZN": "Consumer", "TSLA": "Consumer",
        "NFLX": "Communication", "CRM": "Technology", "ORCL": "Technology", "ADBE": "Technology",
        "INTC": "Technology", "AMD": "Technology", "CSCO": "Technology", "IBM": "Technology",
        "QCOM": "Technology", "TXN": "Technology", "AVGO": "Technology",
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
        "MS": "Financials", "V": "Financials", "MA": "Financials", "BRK.B": "Financials",
        "BRK-B": "Financials", "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
        "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare",
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer", "MCD": "Consumer", "NKE": "Consumer",
        "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples", "PM": "Consumer Staples",
        "DIS": "Communication", "CMCSA": "Communication", "T": "Communication", "VZ": "Communication",
        "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "BA": "Industrials", "UPS": "Industrials",
        "LIN": "Materials", "APD": "Materials", "AMT": "Real Estate", "PLD": "Real Estate",
        "NEE": "Utilities", "DUK": "Utilities",
    }


def load_sector_map(stocks: List[str], sector_map_path: Optional[str]) -> Dict[str, str]:
    sector_map = default_sector_map()
    if sector_map_path:
        p = Path(sector_map_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing sector map file: {sector_map_path}")
        extra = pd.read_csv(p)
        extra.columns = [str(c).strip().lower() for c in extra.columns]
        if "stock" not in extra.columns or "sector" not in extra.columns:
            raise ValueError("sector_map_path CSV must contain columns: stock, sector")
        for _, row in extra.iterrows():
            sector_map[str(row["stock"]).strip()] = str(row["sector"]).strip()
    return {s: sector_map.get(s, "Other") for s in stocks}


def choose_portfolio_diversified(
    day_df: pd.DataFrame,
    current_positions: Dict[str, Dict],
    recent_history: List[List[str]],
    stock_to_sector: Dict[str, str],
    top_k: int,
    entry_prob: float,
    exit_prob: float,
    switch_buffer: float,
    min_hold_periods: int,
    max_new_names_per_rebalance: int,
    max_turnover_per_rebalance: float,
    no_trade_band: float,
    sector_penalty: float,
    repeated_name_penalty: float,
    cooldown_lookback: int,
    cooldown_threshold: int,
    strong_signal_override_margin: float,
) -> Tuple[Dict[str, float], Dict]:
    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True).copy()
    prob_map = dict(zip(day["stock"], day["pred_prob"]))

    recent_flat = [name for basket in recent_history[-cooldown_lookback:] for name in basket]
    recent_count = pd.Series(recent_flat).value_counts().to_dict() if len(recent_flat) > 0 else {}

    forced_keep = []
    normal_keep = []
    current_names = list(current_positions.keys())

    for stock, info in current_positions.items():
        prob = float(prob_map.get(stock, -1.0))
        held_for = int(info.get("held_for", 0))
        if held_for < min_hold_periods:
            forced_keep.append(stock)
        elif prob >= exit_prob:
            normal_keep.append(stock)

    portfolio = list(dict.fromkeys(forced_keep + normal_keep))[:top_k]
    current_set = set(current_names)

    def sector_count(names: List[str]) -> Dict[str, int]:
        out = {}
        for n in names:
            sec = stock_to_sector.get(n, "Other")
            out[sec] = out.get(sec, 0) + 1
        return out

    current_sector_count = sector_count(portfolio)

    candidate_rows = []
    for _, row in day.iterrows():
        s = str(row["stock"])
        raw_prob = float(row["pred_prob"])
        sec = stock_to_sector.get(s, "Other")
        same_sector_pen = sector_penalty * current_sector_count.get(sec, 0)
        rep_pen = 0.0
        if recent_count.get(s, 0) >= cooldown_threshold:
            rep_pen = repeated_name_penalty * recent_count.get(s, 0)
        adjusted_score = raw_prob - same_sector_pen - rep_pen
        candidate_rows.append({
            "stock": s,
            "raw_prob": raw_prob,
            "sector": sec,
            "adjusted_score": adjusted_score,
        })

    cand_df = pd.DataFrame(candidate_rows).sort_values(
        ["adjusted_score", "raw_prob"], ascending=[False, False]
    ).reset_index(drop=True)

    max_changes_left = max_new_names_per_rebalance
    approx_turnover_used = 0.0

    def equal_weight(names: List[str]) -> Dict[str, float]:
        if len(names) == 0:
            return {}
        w = 1.0 / len(names)
        return {x: w for x in names}

    for _, row in cand_df.iterrows():
        s = row["stock"]
        raw_prob = float(row["raw_prob"])
        if len(portfolio) >= top_k:
            break
        if s in portfolio or raw_prob < entry_prob or s in current_set or max_changes_left <= 0:
            continue
        test_port = portfolio + [s]
        inc_turn = turnover(equal_weight(portfolio), equal_weight(test_port))
        if approx_turnover_used + inc_turn > max_turnover_per_rebalance + 1e-12:
            continue
        portfolio.append(s)
        current_sector_count = sector_count(portfolio)
        max_changes_left -= 1
        approx_turnover_used += inc_turn

    if len(portfolio) == top_k and max_changes_left > 0:
        for _, row in cand_df.iterrows():
            s = row["stock"]
            raw_prob = float(row["raw_prob"])
            adj_score = float(row["adjusted_score"])
            if s in portfolio or raw_prob < entry_prob or max_changes_left <= 0:
                continue

            weakest_name = None
            weakest_raw = 1e9
            weakest_adj = 1e9
            for held in portfolio:
                if held in forced_keep:
                    continue
                held_row = cand_df[cand_df["stock"] == held]
                held_raw = float(held_row.iloc[0]["raw_prob"]) if len(held_row) > 0 else float(prob_map.get(held, -1.0))
                held_adj = float(held_row.iloc[0]["adjusted_score"]) if len(held_row) > 0 else held_raw
                if held_adj < weakest_adj or (held_adj == weakest_adj and held_raw < weakest_raw):
                    weakest_adj = held_adj
                    weakest_raw = held_raw
                    weakest_name = held

            if weakest_name is None:
                continue

            replace = False
            if adj_score >= weakest_adj + no_trade_band and raw_prob >= weakest_raw + switch_buffer:
                replace = True
            if raw_prob >= weakest_raw + strong_signal_override_margin:
                replace = True
            if not replace:
                continue

            test_port = [x for x in portfolio if x != weakest_name] + [s]
            rep_turn = turnover(equal_weight(portfolio), equal_weight(test_port))
            if approx_turnover_used + rep_turn > max_turnover_per_rebalance + 1e-12:
                continue

            portfolio = test_port
            current_sector_count = sector_count(portfolio)
            max_changes_left -= 1
            approx_turnover_used += rep_turn

    portfolio = sorted(list(set(portfolio)), key=lambda x: float(prob_map.get(x, -1.0)), reverse=True)[:top_k]
    final_w = equal_weight(portfolio)

    sector_weights = {}
    for s, w in final_w.items():
        sec = stock_to_sector.get(s, "Other")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + w

    debug = {
        "n_holdings": len(final_w),
        "max_sector_weight": max(sector_weights.values()) if sector_weights else 0.0,
        "n_distinct_sectors": len(sector_weights),
        "turnover_budget_used_proxy": approx_turnover_used,
        "new_names_used": max_new_names_per_rebalance - max_changes_left,
        "sector_weights": sector_weights,
    }
    return final_w, debug


def run_diversified_backtest(
    pred_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
    horizon: int,
    top_k: int,
    entry_prob: float,
    exit_prob: float,
    switch_buffer: float,
    min_hold_periods: int,
    max_new_names_per_rebalance: int,
    max_turnover_per_rebalance: float,
    no_trade_band: float,
    sector_penalty: float,
    repeated_name_penalty: float,
    cooldown_lookback: int,
    cooldown_threshold: int,
    strong_signal_override_margin: float,
    transaction_cost_bps: float,
) -> Tuple[Dict, pd.DataFrame]:
    if pred_df.empty:
        return {
            "periods": 0, "cumulative_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0, "avg_turnover": 0.0, "avg_holdings": 0.0,
            "avg_exposure": 0.0, "avg_max_sector_weight": 0.0, "avg_distinct_sectors": 0.0,
            "nvda_presence_rate": 0.0,
        }, pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    current_positions = {}
    recent_history = []
    prev_w = {}

    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    holds = []
    exposures = []
    max_sector_ws = []
    n_distinct_sector_list = []
    nvda_flags = []
    rows = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
        new_w, debug = choose_portfolio_diversified(
            day_df=day,
            current_positions=current_positions,
            recent_history=recent_history,
            stock_to_sector=stock_to_sector,
            top_k=top_k,
            entry_prob=entry_prob,
            exit_prob=exit_prob,
            switch_buffer=switch_buffer,
            min_hold_periods=min_hold_periods,
            max_new_names_per_rebalance=max_new_names_per_rebalance,
            max_turnover_per_rebalance=max_turnover_per_rebalance,
            no_trade_band=no_trade_band,
            sector_penalty=sector_penalty,
            repeated_name_penalty=repeated_name_penalty,
            cooldown_lookback=cooldown_lookback,
            cooldown_threshold=cooldown_threshold,
            strong_signal_override_margin=strong_signal_override_margin,
        )

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
        max_sector_ws.append(float(debug["max_sector_weight"]))
        n_distinct_sector_list.append(float(debug["n_distinct_sectors"]))
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
            "max_sector_weight": float(debug["max_sector_weight"]),
            "n_distinct_sectors": float(debug["n_distinct_sectors"]),
            "sector_weights_json": json.dumps(debug["sector_weights"], ensure_ascii=False),
            "nvda_in_portfolio": int("NVDA" in new_w),
        })

        next_positions = {}
        for s in new_w.keys():
            if s in current_positions:
                next_positions[s] = {"held_for": int(current_positions[s]["held_for"]) + 1}
            else:
                next_positions[s] = {"held_for": 1}
        current_positions = next_positions
        prev_w = new_w
        recent_history.append(sorted(list(new_w.keys())))

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
        "avg_max_sector_weight": float(np.mean(max_sector_ws)) if max_sector_ws else 0.0,
        "avg_distinct_sectors": float(np.mean(n_distinct_sector_list)) if n_distinct_sector_list else 0.0,
        "nvda_presence_rate": float(np.mean(nvda_flags)) if nvda_flags else 0.0,
    }
    return metrics, pd.DataFrame(rows)


def evaluate_config_across_costs(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
    horizon: int,
    costs_bps: List[float],
    cfg: Dict,
) -> Tuple[Dict, Dict]:
    val_metrics_by_cost = {}
    test_metrics_by_cost = {}
    for c in costs_bps:
        val_m, _ = run_diversified_backtest(
            pred_df=val_df, stock_to_sector=stock_to_sector, horizon=horizon,
            top_k=int(cfg["top_k"]), entry_prob=float(cfg["entry_prob"]), exit_prob=float(cfg["exit_prob"]),
            switch_buffer=float(cfg["switch_buffer"]), min_hold_periods=int(cfg["min_hold_periods"]),
            max_new_names_per_rebalance=int(cfg["max_new_names_per_rebalance"]),
            max_turnover_per_rebalance=float(cfg["max_turnover_per_rebalance"]),
            no_trade_band=float(cfg["no_trade_band"]), sector_penalty=float(cfg["sector_penalty"]),
            repeated_name_penalty=float(cfg["repeated_name_penalty"]),
            cooldown_lookback=int(cfg["cooldown_lookback"]), cooldown_threshold=int(cfg["cooldown_threshold"]),
            strong_signal_override_margin=float(cfg["strong_signal_override_margin"]),
            transaction_cost_bps=float(c),
        )
        test_m, _ = run_diversified_backtest(
            pred_df=test_df, stock_to_sector=stock_to_sector, horizon=horizon,
            top_k=int(cfg["top_k"]), entry_prob=float(cfg["entry_prob"]), exit_prob=float(cfg["exit_prob"]),
            switch_buffer=float(cfg["switch_buffer"]), min_hold_periods=int(cfg["min_hold_periods"]),
            max_new_names_per_rebalance=int(cfg["max_new_names_per_rebalance"]),
            max_turnover_per_rebalance=float(cfg["max_turnover_per_rebalance"]),
            no_trade_band=float(cfg["no_trade_band"]), sector_penalty=float(cfg["sector_penalty"]),
            repeated_name_penalty=float(cfg["repeated_name_penalty"]),
            cooldown_lookback=int(cfg["cooldown_lookback"]), cooldown_threshold=int(cfg["cooldown_threshold"]),
            strong_signal_override_margin=float(cfg["strong_signal_override_margin"]),
            transaction_cost_bps=float(c),
        )
        val_metrics_by_cost[str(int(c))] = val_m
        test_metrics_by_cost[str(int(c))] = test_m
    return val_metrics_by_cost, test_metrics_by_cost


def mean_metric(metrics_by_cost: Dict[str, Dict], key: str) -> float:
    vals = [float(v[key]) for v in metrics_by_cost.values()]
    return float(np.mean(vals)) if vals else 0.0


def select_best_diversified_execution(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stock_to_sector: Dict[str, str],
    horizon: int,
    base_cfg: Dict,
    costs_bps: List[float],
    topk_grid: List[int],
    entry_prob_grid: List[float],
    exit_gap_grid: List[float],
    min_hold_grid: List[int],
    max_new_names_grid: List[int],
    max_turnover_budget_grid: List[float],
    no_trade_band_grid: List[float],
    sector_penalty_grid: List[float],
    repeated_name_penalty_grid: List[float],
    cooldown_lookback_grid: List[int],
    cooldown_threshold_grid: List[int],
    strong_signal_override_margin_grid: List[float],
    val_return_tolerance: float,
    val_sharpe_tolerance: float,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    base_search_cfg = {
        "top_k": int(base_cfg["top_k"]),
        "entry_prob": float(base_cfg["entry_prob"]),
        "exit_prob": float(base_cfg["exit_prob"]),
        "switch_buffer": float(base_cfg["switch_buffer"]),
        "min_hold_periods": int(base_cfg["min_hold_periods"]),
        "max_new_names_per_rebalance": int(base_cfg["max_new_names_per_rebalance"]),
        "max_turnover_per_rebalance": 999.0,
        "no_trade_band": 0.0,
        "sector_penalty": 0.0,
        "repeated_name_penalty": 0.0,
        "cooldown_lookback": 3,
        "cooldown_threshold": 99,
        "strong_signal_override_margin": 999.0,
    }
    base_val_by_cost, base_test_by_cost = evaluate_config_across_costs(
        val_df=val_df, test_df=test_df, stock_to_sector=stock_to_sector,
        horizon=horizon, costs_bps=costs_bps, cfg=base_search_cfg
    )
    base_summary = {
        "val_mean_cumret": mean_metric(base_val_by_cost, "cumulative_return"),
        "val_mean_sharpe": mean_metric(base_val_by_cost, "sharpe"),
        "val_mean_mdd": mean_metric(base_val_by_cost, "max_drawdown"),
        "val_mean_turnover": mean_metric(base_val_by_cost, "avg_turnover"),
        "val_mean_max_sector_weight": mean_metric(base_val_by_cost, "avg_max_sector_weight"),
        "val_mean_nvda_presence": mean_metric(base_val_by_cost, "nvda_presence_rate"),
    }

    rows = []
    for top_k in topk_grid:
        for entry_prob in entry_prob_grid:
            for exit_gap in exit_gap_grid:
                exit_prob = max(0.0, entry_prob - exit_gap)
                for min_hold in min_hold_grid:
                    for max_new in max_new_names_grid:
                        for max_turn_budget in max_turnover_budget_grid:
                            for no_trade_band in no_trade_band_grid:
                                for sector_penalty in sector_penalty_grid:
                                    for rep_pen in repeated_name_penalty_grid:
                                        for lookback in cooldown_lookback_grid:
                                            for threshold in cooldown_threshold_grid:
                                                for override_margin in strong_signal_override_margin_grid:
                                                    cfg = {
                                                        "top_k": top_k,
                                                        "entry_prob": entry_prob,
                                                        "exit_prob": exit_prob,
                                                        "switch_buffer": 0.0,
                                                        "min_hold_periods": min_hold,
                                                        "max_new_names_per_rebalance": max_new,
                                                        "max_turnover_per_rebalance": max_turn_budget,
                                                        "no_trade_band": no_trade_band,
                                                        "sector_penalty": sector_penalty,
                                                        "repeated_name_penalty": rep_pen,
                                                        "cooldown_lookback": lookback,
                                                        "cooldown_threshold": threshold,
                                                        "strong_signal_override_margin": override_margin,
                                                    }
                                                    val_by_cost, test_by_cost = evaluate_config_across_costs(
                                                        val_df=val_df, test_df=test_df, stock_to_sector=stock_to_sector,
                                                        horizon=horizon, costs_bps=costs_bps, cfg=cfg
                                                    )
                                                    val_mean_cumret = mean_metric(val_by_cost, "cumulative_return")
                                                    val_mean_sharpe = mean_metric(val_by_cost, "sharpe")
                                                    val_mean_mdd = mean_metric(val_by_cost, "max_drawdown")
                                                    val_mean_turnover = mean_metric(val_by_cost, "avg_turnover")
                                                    val_mean_max_sector_weight = mean_metric(val_by_cost, "avg_max_sector_weight")
                                                    val_mean_distinct_sectors = mean_metric(val_by_cost, "avg_distinct_sectors")
                                                    val_mean_nvda_presence = mean_metric(val_by_cost, "nvda_presence_rate")
                                                    test_mean_cumret = mean_metric(test_by_cost, "cumulative_return")
                                                    test_mean_sharpe = mean_metric(test_by_cost, "sharpe")
                                                    test_mean_mdd = mean_metric(test_by_cost, "max_drawdown")
                                                    test_mean_turnover = mean_metric(test_by_cost, "avg_turnover")
                                                    test_mean_max_sector_weight = mean_metric(test_by_cost, "avg_max_sector_weight")
                                                    test_mean_nvda_presence = mean_metric(test_by_cost, "nvda_presence_rate")

                                                    val_ok = bool(
                                                        val_mean_cumret >= base_summary["val_mean_cumret"] - val_return_tolerance
                                                        and val_mean_sharpe >= base_summary["val_mean_sharpe"] - val_sharpe_tolerance
                                                    )

                                                    selection_score = (
                                                        3.0 * (val_mean_cumret - base_summary["val_mean_cumret"])
                                                        + 2.5 * (val_mean_sharpe - base_summary["val_mean_sharpe"])
                                                        + 1.5 * (val_mean_mdd - base_summary["val_mean_mdd"])
                                                        + 1.2 * (base_summary["val_mean_turnover"] - val_mean_turnover)
                                                        + 1.5 * (base_summary["val_mean_max_sector_weight"] - val_mean_max_sector_weight)
                                                        + 1.2 * (base_summary["val_mean_nvda_presence"] - val_mean_nvda_presence)
                                                        + 0.8 * val_mean_distinct_sectors
                                                    )

                                                    rows.append({
                                                        **cfg,
                                                        "val_ok": val_ok,
                                                        "selection_score": selection_score,
                                                        "val_mean_cumret": val_mean_cumret,
                                                        "val_mean_sharpe": val_mean_sharpe,
                                                        "val_mean_mdd": val_mean_mdd,
                                                        "val_mean_turnover": val_mean_turnover,
                                                        "val_mean_max_sector_weight": val_mean_max_sector_weight,
                                                        "val_mean_distinct_sectors": val_mean_distinct_sectors,
                                                        "val_mean_nvda_presence": val_mean_nvda_presence,
                                                        "test_mean_cumret": test_mean_cumret,
                                                        "test_mean_sharpe": test_mean_sharpe,
                                                        "test_mean_mdd": test_mean_mdd,
                                                        "test_mean_turnover": test_mean_turnover,
                                                        "test_mean_max_sector_weight": test_mean_max_sector_weight,
                                                        "test_mean_nvda_presence": test_mean_nvda_presence,
                                                    })

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df["val_ok"] == True].copy()
    if len(valid) > 0:
        best = valid.sort_values(["selection_score", "val_mean_cumret", "val_mean_sharpe"], ascending=[False, False, False]).iloc[0]
    else:
        best = grid_df.sort_values(["selection_score", "val_mean_cumret", "val_mean_sharpe"], ascending=[False, False, False]).iloc[0]

    best_cfg = {
        "mode": "topk_diversified_v2",
        "top_k": int(best["top_k"]),
        "entry_prob": float(best["entry_prob"]),
        "exit_prob": float(best["exit_prob"]),
        "threshold": 0.50,
        "switch_buffer": float(best["switch_buffer"]),
        "min_hold_periods": int(best["min_hold_periods"]),
        "max_new_names_per_rebalance": int(best["max_new_names_per_rebalance"]),
        "max_turnover_per_rebalance": float(best["max_turnover_per_rebalance"]),
        "no_trade_band": float(best["no_trade_band"]),
        "sector_penalty": float(best["sector_penalty"]),
        "repeated_name_penalty": float(best["repeated_name_penalty"]),
        "cooldown_lookback": int(best["cooldown_lookback"]),
        "cooldown_threshold": int(best["cooldown_threshold"]),
        "strong_signal_override_margin": float(best["strong_signal_override_margin"]),
    }
    return best_cfg, grid_df, {
        "baseline_validation_summary": base_summary,
        "baseline_validation_by_cost": base_val_by_cost,
        "baseline_test_by_cost": base_test_by_cost,
        "best_search_row": best.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Search diversified + cost-robust + turnover-budgeted sticky execution v2.")
    parser.add_argument("--predictions_path", type=str, default=DEFAULT_FIXED_PREDICTIONS)
    parser.add_argument("--feature_data_path", type=str, default=DEFAULT_FEATURE_DATA)
    parser.add_argument("--base_optimization_dir", type=str, default=DEFAULT_BASE_OPT_DIR)
    parser.add_argument("--sector_map_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--cost_grid_bps", type=str, default="10,20,30")
    parser.add_argument("--topk_grid", type=str, default="3,4")
    parser.add_argument("--entry_prob_grid", type=str, default="0.58,0.60,0.62")
    parser.add_argument("--exit_gap_grid", type=str, default="0.02,0.04")
    parser.add_argument("--min_hold_grid", type=str, default="2,3")
    parser.add_argument("--max_new_names_grid", type=str, default="1,2")
    parser.add_argument("--max_turnover_budget_grid", type=str, default="0.75,1.00,1.25")
    parser.add_argument("--no_trade_band_grid", type=str, default="0.00,0.01,0.02")
    parser.add_argument("--sector_penalty_grid", type=str, default="0.00,0.01,0.02,0.03")
    parser.add_argument("--repeated_name_penalty_grid", type=str, default="0.00,0.005,0.01,0.02")
    parser.add_argument("--cooldown_lookback_grid", type=str, default="2,3,4")
    parser.add_argument("--cooldown_threshold_grid", type=str, default="2,3")
    parser.add_argument("--strong_signal_override_margin_grid", type=str, default="0.02,0.04,0.06")
    parser.add_argument("--val_return_tolerance", type=float, default=0.05)
    parser.add_argument("--val_sharpe_tolerance", type=float, default=0.10)
    args = parser.parse_args()

    pred = safe_read_csv(args.predictions_path)
    feat = safe_read_csv(args.feature_data_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    base_opt_summary = safe_read_json(str(Path(args.base_optimization_dir) / "optimization_summary.json"))
    base_cfg = base_opt_summary["optimized_execution_cfg"]

    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()

    universe = sorted(pred["stock"].astype(str).unique().tolist())
    stock_to_sector = load_sector_map(universe, args.sector_map_path)

    costs_bps = parse_float_grid(args.cost_grid_bps)

    best_cfg, grid_df, debug_info = select_best_diversified_execution(
        val_df=val_df, test_df=test_df, stock_to_sector=stock_to_sector, horizon=horizon, base_cfg=base_cfg,
        costs_bps=costs_bps,
        topk_grid=parse_int_grid(args.topk_grid),
        entry_prob_grid=parse_float_grid(args.entry_prob_grid),
        exit_gap_grid=parse_float_grid(args.exit_gap_grid),
        min_hold_grid=parse_int_grid(args.min_hold_grid),
        max_new_names_grid=parse_int_grid(args.max_new_names_grid),
        max_turnover_budget_grid=parse_float_grid(args.max_turnover_budget_grid),
        no_trade_band_grid=parse_float_grid(args.no_trade_band_grid),
        sector_penalty_grid=parse_float_grid(args.sector_penalty_grid),
        repeated_name_penalty_grid=parse_float_grid(args.repeated_name_penalty_grid),
        cooldown_lookback_grid=parse_int_grid(args.cooldown_lookback_grid),
        cooldown_threshold_grid=parse_int_grid(args.cooldown_threshold_grid),
        strong_signal_override_margin_grid=parse_float_grid(args.strong_signal_override_margin_grid),
        val_return_tolerance=args.val_return_tolerance,
        val_sharpe_tolerance=args.val_sharpe_tolerance,
    )

    val_by_cost, test_by_cost = evaluate_config_across_costs(
        val_df=val_df, test_df=test_df, stock_to_sector=stock_to_sector,
        horizon=horizon, costs_bps=costs_bps, cfg=best_cfg,
    )

    default_cost = costs_bps[0]
    final_val_metrics, final_val_actions = run_diversified_backtest(
        pred_df=val_df, stock_to_sector=stock_to_sector, horizon=horizon,
        top_k=int(best_cfg["top_k"]), entry_prob=float(best_cfg["entry_prob"]), exit_prob=float(best_cfg["exit_prob"]),
        switch_buffer=float(best_cfg["switch_buffer"]), min_hold_periods=int(best_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(best_cfg["max_new_names_per_rebalance"]),
        max_turnover_per_rebalance=float(best_cfg["max_turnover_per_rebalance"]),
        no_trade_band=float(best_cfg["no_trade_band"]), sector_penalty=float(best_cfg["sector_penalty"]),
        repeated_name_penalty=float(best_cfg["repeated_name_penalty"]),
        cooldown_lookback=int(best_cfg["cooldown_lookback"]), cooldown_threshold=int(best_cfg["cooldown_threshold"]),
        strong_signal_override_margin=float(best_cfg["strong_signal_override_margin"]),
        transaction_cost_bps=float(default_cost),
    )
    final_test_metrics, final_test_actions = run_diversified_backtest(
        pred_df=test_df, stock_to_sector=stock_to_sector, horizon=horizon,
        top_k=int(best_cfg["top_k"]), entry_prob=float(best_cfg["entry_prob"]), exit_prob=float(best_cfg["exit_prob"]),
        switch_buffer=float(best_cfg["switch_buffer"]), min_hold_periods=int(best_cfg["min_hold_periods"]),
        max_new_names_per_rebalance=int(best_cfg["max_new_names_per_rebalance"]),
        max_turnover_per_rebalance=float(best_cfg["max_turnover_per_rebalance"]),
        no_trade_band=float(best_cfg["no_trade_band"]), sector_penalty=float(best_cfg["sector_penalty"]),
        repeated_name_penalty=float(best_cfg["repeated_name_penalty"]),
        cooldown_lookback=int(best_cfg["cooldown_lookback"]), cooldown_threshold=int(best_cfg["cooldown_threshold"]),
        strong_signal_override_margin=float(best_cfg["strong_signal_override_margin"]),
        transaction_cost_bps=float(default_cost),
    )

    out_root = Path(args.out_dir) / "final_execution_diversified_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_df.to_csv(out_root / "diversified_execution_search.csv", index=False)
    final_val_actions.to_csv(out_root / "diversified_val_actions_10bps.csv", index=False)
    final_test_actions.to_csv(out_root / "diversified_test_actions_10bps.csv", index=False)

    rows = []
    for c in costs_bps:
        key = str(int(c))
        rows.append({"split": "val", "cost_bps": c, **val_by_cost[key]})
        rows.append({"split": "test", "cost_bps": c, **test_by_cost[key]})
    pd.DataFrame(rows).to_csv(out_root / "diversified_metrics_by_cost.csv", index=False)

    summary = {
        "selection_policy": "validation_only_multi_cost_diversified_execution",
        "base_current_execution_cfg": base_cfg,
        "chosen_diversified_execution_cfg": best_cfg,
        "baseline_debug": debug_info,
        "validation_metrics_by_cost": val_by_cost,
        "test_metrics_by_cost": test_by_cost,
        "default_cost_bps_for_action_files": default_cost,
        "default_cost_validation_metrics": final_val_metrics,
        "default_cost_test_metrics": final_test_metrics,
    }
    with open(out_root / "diversified_execution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Diversified sticky execution v2 saved to: {out_root}")
    print("\nBase current execution cfg:")
    print(json.dumps(base_cfg, ensure_ascii=False, indent=2))
    print("\nChosen diversified execution cfg:")
    print(json.dumps(best_cfg, ensure_ascii=False, indent=2))
    print("\nBaseline validation summary:")
    print(json.dumps(debug_info["baseline_validation_summary"], ensure_ascii=False, indent=2))
    print("\nChosen validation metrics by cost:")
    print(json.dumps(val_by_cost, ensure_ascii=False, indent=2))
    print("\nChosen test metrics by cost:")
    print(json.dumps(test_by_cost, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

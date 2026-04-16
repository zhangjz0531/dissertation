from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_H5_DATA = PROJECT_ROOT / "data_acquisition" / "cleaned_datasets" / "main_experiment_h5.csv"
EXECUTION_DIR = PROJECT_ROOT / "Model Runs" / "execution"

CANDD_CONFIG = {
    "strategy_name": "fixed_overlay_balanced_candD",
    "tilt_budget": 0.20,
    "n_overweight": 4,
    "n_underweight": 4,
    "min_weight": 0.00,
    "max_weight": 0.14,
    "turnover_cap": 0.30,
}
CANDB_CONFIG = {
    "strategy_name": "fixed_overlay_return_candB",
    "tilt_budget": 0.18,
    "n_overweight": 4,
    "n_underweight": 4,
    "min_weight": 0.00,
    "max_weight": 0.15,
    "turnover_cap": 0.30,
}

RAW_STATE_FEATURES = [
    "mkt_dc_trend",
    "mkt_return_5d",
    "vix_z_60",
    "credit_stress",
    "dc_trend",
    "dc_event",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_tag() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


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
    xs = list(root.glob(pattern))
    if not xs:
        return None
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def resolve_default_final_system_manifest_path() -> Path:
    p = latest_file_by_glob(EXECUTION_DIR / "final_system", "final_system_*/final_system_manifest.json")
    if p is None:
        raise FileNotFoundError("Could not auto-find final_system_manifest.json")
    return p


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def parse_action_levels(levels_str: str) -> List[float]:
    vals = sorted(set(float(x.strip()) for x in levels_str.split(",") if x.strip()))
    if len(vals) < 2:
        raise ValueError("Need at least two exposure levels")
    if min(vals) < 0 or max(vals) > 1:
        raise ValueError("Exposure levels must be within [0,1]")
    return vals


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
    return float((arr.mean() / std) * np.sqrt(252.0 / horizon))


def max_drawdown_from_equity_curve(equity_curve: List[float]) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def compute_metrics_from_series(
    period_returns: List[float],
    turns: List[float],
    holds: List[int],
    exposures: List[float],
    horizon: int,
) -> Dict[str, float]:
    if not period_returns:
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
    curve = [equity]
    for r in period_returns:
        equity *= (1.0 + float(r))
        curve.append(equity)

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": float(curve[-1] - 1.0),
        "annualized_return": annualized_return(curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon),
        "max_drawdown": max_drawdown_from_equity_curve(curve),
        "win_rate": float(np.mean(np.asarray(period_returns) > 0)),
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
    }


def normalize_predictions(pred: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["date"] = pd.to_datetime(out["date"])

    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"])

    if "pred_prob" not in out.columns:
        cand = [c for c in out.columns if "prob" in c.lower()]
        if not cand:
            raise ValueError("pred_prob missing")
        out = out.rename(columns={cand[0]: "pred_prob"})

    if "split" not in out.columns:
        if "split" not in feat.columns:
            raise ValueError("split missing")
        out = out.merge(
            feat[["date", "stock", "split"]].drop_duplicates(),
            on=["date", "stock"],
            how="left",
        )

    if "future_return" not in out.columns:
        ret_cols = [c for c in feat.columns if c.startswith("future_return_")]
        if len(ret_cols) != 1:
            raise ValueError("Need exactly one future_return_* column in feature data")
        rc = ret_cols[0]
        out = out.merge(
            feat[["date", "stock", "split", rc]].rename(columns={rc: "future_return"}),
            on=["date", "stock", "split"],
            how="left",
        )

    missing_state_cols = [c for c in RAW_STATE_FEATURES if c in feat.columns and c not in out.columns]
    if missing_state_cols:
        out = out.merge(
            feat[["date", "stock", "split"] + missing_state_cols].drop_duplicates(subset=["date", "stock", "split"]),
            on=["date", "stock", "split"],
            how="left",
        )

    req = ["date", "stock", "split", "pred_prob", "future_return"]
    miss = [c for c in req if c not in out.columns]
    if miss:
        raise ValueError(f"missing columns after normalization: {miss}")
    if out["future_return"].isna().any():
        raise ValueError("future_return has NaNs after normalization")

    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(w.values()))
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: float(v) / s for k, v in w.items()}


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w) | set(new_w)
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def drift_weights(weights: Dict[str, float], ret_map: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    grown = {k: float(v) * (1.0 + float(ret_map.get(k, 0.0))) for k, v in weights.items()}
    s = float(sum(grown.values()))
    if s <= 0:
        return {k: 0.0 for k in grown}
    return {k: float(v) / s for k, v in grown.items()}


def allocate_budget_equal(cap_map: Dict[str, float], total_budget: float) -> Dict[str, float]:
    rem = float(total_budget)
    caps = {k: max(0.0, float(v)) for k, v in cap_map.items()}
    alloc = {k: 0.0 for k in cap_map}
    active = [k for k, v in caps.items() if v > 1e-12]
    while rem > 1e-12 and active:
        share = rem / len(active)
        nxt, used = [], 0.0
        for k in active:
            add = min(share, caps[k])
            alloc[k] += add
            caps[k] -= add
            used += add
            if caps[k] > 1e-12:
                nxt.append(k)
        if used <= 1e-12:
            break
        rem -= used
        active = nxt
    return alloc


def build_active_overlay_target(
    benchmark_w: Dict[str, float],
    day_df: pd.DataFrame,
    n_overweight: int,
    n_underweight: int,
    tilt_budget: float,
    min_weight: float,
    max_weight: float,
) -> Tuple[Dict[str, float], Dict]:
    if not benchmark_w:
        return {}, {"effective_budget": 0.0, "top_names": [], "bottom_names": []}

    day = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    universe = [s for s in day["stock"].astype(str).tolist() if s in benchmark_w]
    if not universe:
        return benchmark_w.copy(), {"effective_budget": 0.0, "top_names": [], "bottom_names": []}

    base_w = {s: float(benchmark_w.get(s, 0.0)) for s in universe}
    n = len(universe)
    n_over = max(0, min(int(n_overweight), n))
    n_under = max(0, min(int(n_underweight), n))
    top_names = universe[:n_over]
    bottom_names = [s for s in universe[-n_under:] if s not in set(top_names)] if n_under > 0 else []

    if not top_names or not bottom_names or tilt_budget <= 1e-12:
        return normalize_weights(base_w), {"effective_budget": 0.0, "top_names": top_names, "bottom_names": bottom_names}

    over_caps = {s: max(0.0, float(max_weight) - base_w.get(s, 0.0)) for s in top_names}
    under_caps = {s: max(0.0, base_w.get(s, 0.0) - float(min_weight)) for s in bottom_names}
    eff = float(min(float(tilt_budget), float(sum(over_caps.values())), float(sum(under_caps.values()))))
    if eff <= 1e-12:
        return normalize_weights(base_w), {"effective_budget": 0.0, "top_names": top_names, "bottom_names": bottom_names}

    over_alloc = allocate_budget_equal(over_caps, eff)
    under_alloc = allocate_budget_equal(under_caps, eff)
    tgt = base_w.copy()
    for s, v in over_alloc.items():
        tgt[s] = tgt.get(s, 0.0) + float(v)
    for s, v in under_alloc.items():
        tgt[s] = tgt.get(s, 0.0) - float(v)

    tgt = {k: min(float(max_weight), max(float(min_weight), float(v))) for k, v in tgt.items()}
    return normalize_weights(tgt), {"effective_budget": eff, "top_names": top_names, "bottom_names": bottom_names}


def apply_turnover_cap(pretrade_w: Dict[str, float], target_w: Dict[str, float], turnover_cap: float) -> Dict[str, float]:
    if turnover_cap is None or turnover_cap <= 0:
        return target_w.copy()
    cur = turnover(pretrade_w, target_w)
    if cur <= turnover_cap + 1e-12 or cur <= 1e-12:
        return target_w.copy()
    scale = float(turnover_cap) / float(cur)
    names = set(pretrade_w) | set(target_w)
    out = {}
    for n in names:
        p = float(pretrade_w.get(n, 0.0))
        t = float(target_w.get(n, 0.0))
        out[n] = p + scale * (t - p)
    out = {k: max(0.0, float(v)) for k, v in out.items()}
    return normalize_weights(out)


def compute_stress_score(smoothed_row: pd.Series) -> Tuple[float, Dict[str, float]]:
    vals = {
        "vix_z_60_smoothed": float(smoothed_row.get("vix_z_60_smoothed", 0.0)),
        "credit_stress_smoothed": float(smoothed_row.get("credit_stress_smoothed", 0.0)),
        "mkt_dc_trend_smoothed": float(smoothed_row.get("mkt_dc_trend_smoothed", 0.0)),
        "mkt_return_5d_smoothed": float(smoothed_row.get("mkt_return_5d_smoothed", 0.0)),
    }
    return (
        0.40 * max(0.0, vals["vix_z_60_smoothed"])
        + 0.35 * max(0.0, vals["credit_stress_smoothed"])
        + 0.15 * max(0.0, -vals["mkt_dc_trend_smoothed"])
        + 0.10 * max(0.0, -vals["mkt_return_5d_smoothed"]),
        vals,
    )


def compute_stress_tier(
    smoothed_row: pd.Series,
    vix_warn: float,
    vix_high: float,
    credit_warn: float,
    credit_high: float,
    mkt_dc_warn: float,
    mkt_dc_high: float,
    mkt_ret_warn: float,
    mkt_ret_high: float,
) -> Tuple[int, Dict[str, float]]:
    score, vals = compute_stress_score(smoothed_row)

    mild = (
        (vals["vix_z_60_smoothed"] >= float(vix_warn))
        or (vals["credit_stress_smoothed"] >= float(credit_warn))
        or (vals["mkt_dc_trend_smoothed"] <= float(mkt_dc_warn))
        or (vals["mkt_return_5d_smoothed"] <= float(mkt_ret_warn))
    )
    high = (
        (vals["vix_z_60_smoothed"] >= float(vix_high))
        or (vals["credit_stress_smoothed"] >= float(credit_high))
        or (vals["mkt_dc_trend_smoothed"] <= float(mkt_dc_high))
        or (vals["mkt_return_5d_smoothed"] <= float(mkt_ret_high))
    )

    tier = 2 if high else (1 if mild else 0)
    vals["stress_score"] = float(score)
    vals["stress_tier"] = int(tier)
    return tier, vals


def make_risk_budget_state(
    day: pd.DataFrame,
    smoothed_row: pd.Series,
    base_drawdown: float,
    module_drawdown: float,
    base_recent_returns: Deque[float],
    module_recent_returns: Deque[float],
    current_exposure: float,
    hold_progress: float,
    stress_tier: int,
) -> np.ndarray:
    day = day.sort_values("pred_prob", ascending=False).reset_index(drop=True)
    probs = day["pred_prob"].astype(float).values
    top1 = float(probs[0]) if len(probs) >= 1 else 0.0
    top3 = float(np.mean(probs[:3])) if len(probs) >= 3 else (float(np.mean(probs)) if len(probs) > 0 else 0.0)
    top5 = float(np.mean(probs[:5])) if len(probs) >= 5 else (float(np.mean(probs)) if len(probs) > 0 else 0.0)
    prob_std = float(np.std(probs)) if len(probs) > 1 else 0.0
    gap12 = float(probs[0] - probs[1]) if len(probs) >= 2 else 0.0

    feats = [float(smoothed_row.get(f"{c}_smoothed", 0.0)) for c in RAW_STATE_FEATURES]
    base_vol = float(np.std(np.asarray(list(base_recent_returns), dtype=float))) if len(base_recent_returns) >= 2 else 0.0
    module_vol = float(np.std(np.asarray(list(module_recent_returns), dtype=float))) if len(module_recent_returns) >= 2 else 0.0

    return np.array(
        [
            top1,
            top3,
            top5,
            prob_std,
            gap12,
            *feats,
            float(base_drawdown),
            float(module_drawdown),
            float(base_vol),
            float(module_vol),
            float(current_exposure),
            float(hold_progress),
            float(stress_tier),
        ],
        dtype=np.float32,
    )


class RiskBudgetEpisodeCoreV3:
    """
    Fixed alpha sleeve = candD.
    Optional RL post-module only controls exposure multiplier.
    V3 mixed scheme:
    - keeps exploit fix from v2 (single-period downside protection only)
    - stronger braking action space by default (0.6 / 1.0)
    - higher switch penalty
    - two-tier stress gate
    """

    def __init__(
        self,
        pred_df: pd.DataFrame,
        horizon: int,
        transaction_cost_bps: float,
        base_config: Dict,
        action_levels: List[float],
        smooth_window: int = 3,
        min_hold_periods: int = 2,
        switch_penalty_raw: float = 0.005,
        cash_return_per_period: float = 0.0,
        downside_protection_coef: float = 0.30,
        drawdown_breach_coef: float = 0.04,
        drawdown_tolerance: float = 0.20,
        vol_penalty_coef: float = 0.02,
        vol_tolerance: float = 0.025,
        nonstress_cash_penalty_raw: float = 0.002,
        vix_warn: float = 0.90,
        vix_high: float = 1.30,
        credit_warn: float = 0.90,
        credit_high: float = 1.30,
        mkt_dc_warn: float = -0.12,
        mkt_dc_high: float = -0.25,
        mkt_ret_warn: float = -0.025,
        mkt_ret_high: float = -0.05,
        default_start_exposure: Optional[float] = None,
    ):
        if pred_df.empty:
            raise ValueError("pred_df empty")

        self.df = pred_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.rebalance_dates = sorted(self.df["date"].drop_duplicates().tolist())[::horizon]
        self.day_map = {
            dt: self.df[self.df["date"] == dt].copy().sort_values("pred_prob", ascending=False).reset_index(drop=True)
            for dt in self.rebalance_dates
        }

        self.horizon = int(horizon)
        self.tc = float(transaction_cost_bps) / 10000.0
        self.base_config = dict(base_config)
        self.action_levels = list(action_levels)
        self.smooth_window = int(max(1, smooth_window))
        self.min_hold_periods = int(max(1, min_hold_periods))
        self.switch_penalty_raw = float(max(0.0, switch_penalty_raw))
        self.cash_return_per_period = float(cash_return_per_period)
        self.downside_protection_coef = float(downside_protection_coef)
        self.drawdown_breach_coef = float(drawdown_breach_coef)
        self.drawdown_tolerance = float(drawdown_tolerance)
        self.vol_penalty_coef = float(vol_penalty_coef)
        self.vol_tolerance = float(vol_tolerance)
        self.nonstress_cash_penalty_raw = float(max(0.0, nonstress_cash_penalty_raw))

        self.vix_warn = float(vix_warn)
        self.vix_high = float(vix_high)
        self.credit_warn = float(credit_warn)
        self.credit_high = float(credit_high)
        self.mkt_dc_warn = float(mkt_dc_warn)
        self.mkt_dc_high = float(mkt_dc_high)
        self.mkt_ret_warn = float(mkt_ret_warn)
        self.mkt_ret_high = float(mkt_ret_high)

        self.default_start_exposure = float(default_start_exposure) if default_start_exposure is not None else float(max(action_levels))

        first_day = self.day_map[self.rebalance_dates[0]].copy().sort_values("stock").reset_index(drop=True)
        self.universe_names = first_day["stock"].astype(str).tolist()
        if not self.universe_names:
            raise ValueError("no stocks")

        rows = []
        for dt in self.rebalance_dates:
            day = self.day_map[dt]
            row = {"date": pd.Timestamp(dt)}
            for c in RAW_STATE_FEATURES:
                row[c] = float(day[c].mean()) if c in day.columns else 0.0
            rows.append(row)

        self.date_feature_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for c in RAW_STATE_FEATURES:
            self.date_feature_df[f"{c}_smoothed"] = self.date_feature_df[c].rolling(window=self.smooth_window, min_periods=1).mean()
        self.date_feature_map = {pd.Timestamp(r["date"]): r for _, r in self.date_feature_df.iterrows()}

        self.num_actions = len(self.action_levels)
        self.state_dim = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.benchmark_w = {s: 1.0 / len(self.universe_names) for s in self.universe_names}
        self.base_w = {}
        self.module_w = {"CASH": 1.0}

        self.base_equity = 1.0
        self.base_peak = 1.0
        self.base_drawdown = 0.0

        self.module_equity = 1.0
        self.module_peak = 1.0
        self.module_drawdown = 0.0

        self.current_exposure = self.default_start_exposure
        self.base_recent_returns: Deque[float] = deque(maxlen=12)
        self.module_recent_returns: Deque[float] = deque(maxlen=12)

        self.prev_action_idx = self.action_levels.index(self.default_start_exposure) if self.default_start_exposure in self.action_levels else None
        self.current_hold_periods = 0

        self.records: List[Dict] = []
        state = self._build_state()
        if self.state_dim is None:
            self.state_dim = int(len(state))
        return state

    def _current_day(self) -> pd.DataFrame:
        return self.day_map[self.rebalance_dates[self.t]]

    def _current_smoothed_row(self) -> pd.Series:
        return self.date_feature_map[pd.Timestamp(self.rebalance_dates[self.t])]

    def _current_stress_tier(self) -> Tuple[int, Dict[str, float]]:
        return compute_stress_tier(
            self._current_smoothed_row(),
            vix_warn=self.vix_warn,
            vix_high=self.vix_high,
            credit_warn=self.credit_warn,
            credit_high=self.credit_high,
            mkt_dc_warn=self.mkt_dc_warn,
            mkt_dc_high=self.mkt_dc_high,
            mkt_ret_warn=self.mkt_ret_warn,
            mkt_ret_high=self.mkt_ret_high,
        )

    def allowed_action_indices(self) -> List[int]:
        stress_tier, _ = self._current_stress_tier()
        max_idx = int(np.argmax(self.action_levels))
        min_idx = int(np.argmin(self.action_levels))

        if self.prev_action_idx is None:
            return [max_idx] if stress_tier == 0 else ([max_idx] if stress_tier == 1 else [min_idx, max_idx])

        if self.current_hold_periods < self.min_hold_periods:
            return [int(self.prev_action_idx)]

        if stress_tier == 0:
            return [max_idx]
        if stress_tier == 1:
            # mild stress: stay full, do not cut yet
            return [max_idx]
        # high stress only
        return [min_idx, max_idx]

    def _build_base_target(self, benchmark_start_w: Dict[str, float], day: pd.DataFrame) -> Dict[str, float]:
        target_raw, _ = build_active_overlay_target(
            benchmark_w=benchmark_start_w,
            day_df=day,
            n_overweight=int(self.base_config["n_overweight"]),
            n_underweight=int(self.base_config["n_underweight"]),
            tilt_budget=float(self.base_config["tilt_budget"]),
            min_weight=float(self.base_config["min_weight"]),
            max_weight=float(self.base_config["max_weight"]),
        )
        if not self.base_w:
            return target_raw.copy()
        return apply_turnover_cap(self.base_w, target_raw, float(self.base_config["turnover_cap"]))

    def _build_module_weights(self, exposure: float, sleeve_w: Dict[str, float]) -> Dict[str, float]:
        out = {"CASH": max(0.0, 1.0 - float(exposure))}
        for s, w in sleeve_w.items():
            out[s] = float(exposure) * float(w)
        return out

    def _build_state(self) -> np.ndarray:
        stress_tier, _ = self._current_stress_tier()
        hold_progress = min(float(self.current_hold_periods), float(self.min_hold_periods)) / float(self.min_hold_periods)
        return make_risk_budget_state(
            day=self._current_day(),
            smoothed_row=self._current_smoothed_row(),
            base_drawdown=self.base_drawdown,
            module_drawdown=self.module_drawdown,
            base_recent_returns=self.base_recent_returns,
            module_recent_returns=self.module_recent_returns,
            current_exposure=self.current_exposure,
            hold_progress=hold_progress,
            stress_tier=stress_tier,
        )

    def step(self, action_idx: int):
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError("invalid action_idx")
        if action_idx not in self.allowed_action_indices():
            raise ValueError("action not allowed under stress gate / hold rule")

        exposure = float(self.action_levels[action_idx])
        stress_tier, stress_info = self._current_stress_tier()

        day = self._current_day()
        benchmark_start_w = self.benchmark_w.copy()
        base_target_w = self._build_base_target(benchmark_start_w, day)
        ret_map = day.set_index("stock")["future_return"].to_dict()

        prev_base_w = self.base_w.copy()
        prev_module_w = self.module_w.copy()

        base_turn = turnover(prev_base_w, base_target_w)
        base_gross = float(sum(base_target_w.get(s, 0.0) * ret_map.get(s, 0.0) for s in base_target_w))
        base_net = base_gross - self.tc * base_turn

        module_target_w = self._build_module_weights(exposure, base_target_w)
        module_turn = turnover(prev_module_w, module_target_w)
        module_gross = float(exposure * base_gross + (1.0 - exposure) * self.cash_return_per_period)
        module_net = module_gross - self.tc * module_turn

        self.base_equity *= (1.0 + base_net)
        self.base_peak = max(self.base_peak, self.base_equity)
        self.base_drawdown = self.base_equity / max(self.base_peak, 1e-12) - 1.0

        self.module_equity *= (1.0 + module_net)
        self.module_peak = max(self.module_peak, self.module_equity)
        self.module_drawdown = self.module_equity / max(self.module_peak, 1e-12) - 1.0

        self.base_recent_returns.append(float(base_net))
        self.module_recent_returns.append(float(module_net))

        module_vol = float(np.std(np.asarray(list(self.module_recent_returns), dtype=float))) if len(self.module_recent_returns) >= 2 else 0.0
        drawdown_breach = max(0.0, -float(self.module_drawdown) - float(self.drawdown_tolerance))
        vol_breach = max(0.0, float(module_vol) - float(self.vol_tolerance))
        switched = int(self.prev_action_idx is not None and action_idx != self.prev_action_idx)

        relative_return = float(module_net - base_net)
        downside_save = max(0.0, max(0.0, -base_net) - max(0.0, -module_net))

        nonstress_cash_penalty = 0.0
        if stress_tier == 0:
            nonstress_cash_penalty = float(self.nonstress_cash_penalty_raw) * max(0.0, 1.0 - exposure)

        reward = (
            relative_return
            + float(self.downside_protection_coef) * float(downside_save)
            - float(self.drawdown_breach_coef) * float(drawdown_breach)
            - float(self.vol_penalty_coef) * float(vol_breach)
            - float(self.switch_penalty_raw) * float(switched)
            - float(nonstress_cash_penalty)
        )

        self.records.append({
            "date": pd.Timestamp(self.rebalance_dates[self.t]),
            "action_idx": int(action_idx),
            "chosen_exposure": float(exposure),
            "base_net_return": float(base_net),
            "module_net_return": float(module_net),
            "relative_return": float(relative_return),
            "downside_save": float(downside_save),
            "nonstress_cash_penalty": float(nonstress_cash_penalty),
            "base_turnover": float(base_turn),
            "module_turnover": float(module_turn),
            "base_equity": float(self.base_equity),
            "module_equity": float(self.module_equity),
            "base_drawdown": float(self.base_drawdown),
            "module_drawdown": float(self.module_drawdown),
            "module_vol": float(module_vol),
            "cash_weight": float(module_target_w.get("CASH", 0.0)),
            "switched_exposure": float(switched),
            "stress_tier": int(stress_tier),
            **stress_info,
        })

        self.base_w = drift_weights(base_target_w, ret_map)
        self.module_w = self._build_module_weights(exposure, self.base_w)
        self.benchmark_w = drift_weights(benchmark_start_w, ret_map)
        self.current_exposure = exposure

        self.current_hold_periods = 1 if (self.prev_action_idx is None or action_idx != self.prev_action_idx) else self.current_hold_periods + 1
        self.prev_action_idx = int(action_idx)

        self.t += 1
        done = bool(self.t >= len(self.rebalance_dates))
        next_state = np.zeros(self.state_dim if self.state_dim is not None else 1, dtype=np.float32) if done else self._build_state()
        info = {
            "module_net_return": float(module_net),
            "base_net_return": float(base_net),
            "module_turnover": float(module_turn),
            "chosen_exposure": float(exposure),
            "module_drawdown": float(self.module_drawdown),
            "switched_exposure": int(switched),
            "stress_tier": int(stress_tier),
            "downside_save": float(downside_save),
        }
        return next_state, float(reward), done, info


def summarize_risk_budget_episode(records: List[Dict], horizon: int, label: str):
    df = pd.DataFrame(records)
    if df.empty:
        empty = compute_metrics_from_series([], [], [], [], horizon)
        rel = {"cumret_gap_vs_base": 0.0, "sharpe_gap_vs_base": 0.0, "mdd_improvement_vs_base": 0.0}
        return empty, empty, rel, df

    module_metrics = compute_metrics_from_series(
        df["module_net_return"].astype(float).tolist(),
        df["module_turnover"].astype(float).tolist(),
        [10] * len(df),
        df["chosen_exposure"].astype(float).tolist(),
        horizon,
    )
    module_metrics["avg_cash_weight"] = float(df["cash_weight"].mean())
    module_metrics["switch_rate"] = float(df["switched_exposure"].mean())
    module_metrics["high_stress_rate"] = float((df["stress_tier"] == 2).mean())
    module_metrics["any_stress_rate"] = float((df["stress_tier"] >= 1).mean())
    module_metrics["avg_downside_save"] = float(df["downside_save"].mean())

    base_metrics = compute_metrics_from_series(
        df["base_net_return"].astype(float).tolist(),
        df["base_turnover"].astype(float).tolist(),
        [10] * len(df),
        [1.0] * len(df),
        horizon,
    )

    rel = {
        "cumret_gap_vs_base": float(module_metrics["cumulative_return"] - base_metrics["cumulative_return"]),
        "sharpe_gap_vs_base": float(module_metrics["sharpe"] - base_metrics["sharpe"]),
        "mdd_improvement_vs_base": float(abs(base_metrics["max_drawdown"]) - abs(module_metrics["max_drawdown"])),
    }

    actions_df = df.copy()
    actions_df["strategy"] = label
    return module_metrics, base_metrics, rel, actions_df


def run_fixed_post_module_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    base_config: Dict,
    fixed_exposure: float,
    label: str,
):
    core = RiskBudgetEpisodeCoreV3(
        pred_df=pred_df,
        horizon=horizon,
        transaction_cost_bps=transaction_cost_bps,
        base_config=base_config,
        action_levels=[fixed_exposure, fixed_exposure + 1e-9],
        smooth_window=1,
        min_hold_periods=1,
        switch_penalty_raw=0.0,
        cash_return_per_period=0.0,
        downside_protection_coef=0.0,
        drawdown_breach_coef=0.0,
        drawdown_tolerance=1.0,
        vol_penalty_coef=0.0,
        vol_tolerance=1.0,
        nonstress_cash_penalty_raw=0.0,
        vix_warn=999.0,
        vix_high=999.0,
        credit_warn=999.0,
        credit_high=999.0,
        mkt_dc_warn=-999.0,
        mkt_dc_high=-999.0,
        mkt_ret_warn=-999.0,
        mkt_ret_high=-999.0,
        default_start_exposure=fixed_exposure,
    )
    core.action_levels = [fixed_exposure]
    core.num_actions = 1
    core.reset()
    done = False
    while not done:
        _, _, done, _ = core.step(0)
    return summarize_risk_budget_episode(core.records, horizon, label)

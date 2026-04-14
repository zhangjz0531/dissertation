
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# H5 ranking model v3.1
#
# Focus:
#   Transformer + top2 + confidence/regime/exposure gates
#
# What this version adds on top of v3
# -----------------------------------
# 1) Keep the v3 grouped-by-date ranking training:
#    - pointwise regression loss
#    - pairwise ranking loss
#
# 2) Add execution-layer filters in validation/test backtest:
#    - confidence gate:
#         require top2 - top3 predicted-score margin >= threshold
#    - regime gate:
#         require market / top1 DC alignment and risk proxies not too bad
#    - exposure gate:
#         * confidence false -> cash
#         * confidence true, regime false -> half exposure to top2
#         * confidence true, regime true  -> full exposure to top2
#
# 3) Validation-time grid search over execution filters:
#    - confidence margin
#    - VIX z-score threshold
#    - credit stress quantile threshold
#    - alignment on/off
#
# 4) This script is intentionally STATIC-ONLY.
#    Reason: right now the best next step is to stabilize the validated
#    main system first. After that, rolling can be added on top.
#
# Recommended run:
#
# python train_h5_ranking_v31.py ^
#   --data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs" ^
#   --model transformer
# =========================================================


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Helpers
# -----------------------------
def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    target_cols = [c for c in df.columns if c.startswith("target_up_")]
    return_cols = [c for c in df.columns if c.startswith("future_return_")]
    if len(target_cols) != 1 or len(return_cols) != 1:
        raise ValueError("Dataset must contain exactly one target_up_* and one future_return_* column.")
    return target_cols[0], return_cols[0]


def infer_horizon(return_col: str) -> int:
    if return_col.endswith("1d"):
        return 1
    if return_col.endswith("5d"):
        return 5
    raise ValueError(f"Cannot infer horizon from {return_col}")


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


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


# -----------------------------
# Target engineering
# -----------------------------
class TargetScaler:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_ = None
        self.upper_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        self.lower_ = float(np.quantile(y, self.lower_q))
        self.upper_ = float(np.quantile(y, self.upper_q))
        y_w = np.clip(y, self.lower_, self.upper_)
        self.mean_ = float(y_w.mean())
        self.std_ = float(y_w.std())
        if self.std_ < 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y_w = np.clip(y, self.lower_, self.upper_)
        return ((y_w - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled, dtype=float)
        return (y_scaled * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "lower_q": self.lower_q,
            "upper_q": self.upper_q,
            "lower_": self.lower_,
            "upper_": self.upper_,
            "mean_": self.mean_,
            "std_": self.std_,
        }


def auto_target_mode(df: pd.DataFrame, return_col: str) -> str:
    if return_col.endswith("5d") and "mkt_return_5d" in df.columns:
        return "excess_vs_market"
    if return_col.endswith("1d") and "mkt_return_1d" in df.columns:
        return "excess_vs_market"
    return "cross_sectional_demean"


def build_target_column(df: pd.DataFrame, return_col: str, target_mode: str) -> pd.DataFrame:
    out = df.copy()
    if target_mode == "raw":
        out["target_value_raw"] = out[return_col].astype(float)
    elif target_mode == "excess_vs_market":
        if return_col.endswith("5d") and "mkt_return_5d" in out.columns:
            out["target_value_raw"] = out[return_col].astype(float) - out["mkt_return_5d"].astype(float)
        elif return_col.endswith("1d") and "mkt_return_1d" in out.columns:
            out["target_value_raw"] = out[return_col].astype(float) - out["mkt_return_1d"].astype(float)
        else:
            raise ValueError("Requested excess_vs_market but matching market return column not found.")
    elif target_mode == "cross_sectional_demean":
        out["target_value_raw"] = out[return_col].astype(float) - out.groupby("date")[return_col].transform("mean").astype(float)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return out


# -----------------------------
# Feature engineering
# -----------------------------
def build_v31_feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    excluded = {"date", "stock", "split", target_col, return_col, "target_value_raw", "target_value_scaled"}
    numeric_cols = [c for c in out.columns if c not in excluded and pd.api.types.is_numeric_dtype(out[c])]

    rank_bases = []
    for c in ["close", "return_5d", "return_21d", "volatility_20d", "rsi_14", "macd_hist_pct", "dc_tmv"]:
        if c in out.columns:
            rank_bases.append(c)

    for c in rank_bases:
        rank_col = f"{c}_rank_pct"
        if rank_col not in out.columns:
            out[rank_col] = out.groupby("date")[c].rank(pct=True)
        numeric_cols.append(rank_col)

    if "mkt_dc_trend" in out.columns and "dc_trend" in out.columns:
        out["dc_trend_align"] = out["mkt_dc_trend"].astype(float) * out["dc_trend"].astype(float)
        numeric_cols.append("dc_trend_align")

    seen = set()
    feats = []
    for c in numeric_cols:
        if c not in seen and pd.api.types.is_numeric_dtype(out[c]):
            feats.append(c)
            seen.add(c)

    return out, feats


def fit_feature_standardizer(train_df: pd.DataFrame, feats: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train_df[feats].mean()
    std = train_df[feats].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_feature_standardizer(df: pd.DataFrame, feats: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feats] = (out[feats] - mean) / std
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# -----------------------------
# Models
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


# -----------------------------
# Grouped-by-date dataset
# -----------------------------
class DateGroupDataset(Dataset):
    def __init__(self, groups: List[Dict]):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        return {
            "X": torch.tensor(g["X"], dtype=torch.float32),
            "y_scaled": torch.tensor(g["y_scaled"], dtype=torch.float32),
            "y_raw": torch.tensor(g["y_raw"], dtype=torch.float32),
            "date": g["date"],
            "stocks": g["stocks"],
            "future_return": torch.tensor(g["future_return"], dtype=torch.float32),
        }


def group_collate(batch):
    X = torch.cat([b["X"] for b in batch], dim=0)
    y_scaled = torch.cat([b["y_scaled"] for b in batch], dim=0)
    y_raw = torch.cat([b["y_raw"] for b in batch], dim=0)
    future_return = torch.cat([b["future_return"] for b in batch], dim=0)
    group_sizes = [len(b["stocks"]) for b in batch]
    dates = []
    stocks = []
    for b in batch:
        dates.extend([b["date"]] * len(b["stocks"]))
        stocks.extend(list(b["stocks"]))
    return {
        "X": X,
        "y_scaled": y_scaled,
        "y_raw": y_raw,
        "future_return": future_return,
        "group_sizes": group_sizes,
        "dates": dates,
        "stocks": stocks,
    }


def build_grouped_sequences(df: pd.DataFrame, feats: List[str], seq_len: int, split_col: str) -> List[Dict]:
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    rows = []
    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].values.astype(np.float32)
        y_scaled = g["target_value_scaled"].values.astype(np.float32)
        y_raw = g["target_value_raw"].values.astype(np.float32)
        fut = g[[c for c in g.columns if c.startswith("future_return_")][0]].values.astype(np.float32)
        split_vals = g[split_col].values

        for i in range(seq_len - 1, len(g)):
            rows.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "split": split_vals[i],
                "X": Xg[i - seq_len + 1:i + 1],
                "y_scaled": y_scaled[i],
                "y_raw": y_raw[i],
                "future_return": fut[i],
            })

    rdf = pd.DataFrame([{
        "date": r["date"],
        "stock": r["stock"],
        "split": r["split"],
        "X": r["X"],
        "y_scaled": r["y_scaled"],
        "y_raw": r["y_raw"],
        "future_return": r["future_return"],
    } for r in rows])

    groups = []
    for (dt, sp), g in rdf.groupby(["date", "split"], sort=True):
        g = g.sort_values("stock").reset_index(drop=True)
        groups.append({
            "date": pd.Timestamp(dt),
            "split": sp,
            "stocks": g["stock"].tolist(),
            "X": np.stack(g["X"].tolist()).astype(np.float32),
            "y_scaled": g["y_scaled"].values.astype(np.float32),
            "y_raw": g["y_raw"].values.astype(np.float32),
            "future_return": g["future_return"].values.astype(np.float32),
        })
    return groups


def split_group_lists(groups: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for g in groups:
        out[g["split"]].append(g)
    return out


# -----------------------------
# Losses
# -----------------------------
def pairwise_rank_loss(scores: torch.Tensor, targets: torch.Tensor, group_sizes: List[int], min_diff: float = 1e-6) -> torch.Tensor:
    losses = []
    start = 0
    for n in group_sizes:
        s = scores[start:start+n]
        t = targets[start:start+n]
        start += n

        if n < 2:
            continue

        diff_t = t.unsqueeze(1) - t.unsqueeze(0)
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)
        mask = diff_t > min_diff
        if mask.any():
            losses.append(F.softplus(-diff_s[mask]).mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(losses).mean()


# -----------------------------
# Prediction frames and metrics
# -----------------------------
def groups_to_prediction_df(groups: List[Dict], pred_scaled: np.ndarray, target_scaler: TargetScaler) -> pd.DataFrame:
    pred_raw = target_scaler.inverse_transform(pred_scaled)
    rows = []
    ptr = 0
    for g in groups:
        n = len(g["stocks"])
        for i in range(n):
            rows.append({
                "date": pd.Timestamp(g["date"]),
                "stock": g["stocks"][i],
                "target_value_raw": float(g["y_raw"][i]),
                "future_return": float(g["future_return"][i]),
                "pred_score": float(pred_raw[ptr + i]),
            })
        ptr += n
    return pd.DataFrame(rows)


def regression_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    err = y_pred_raw - y_true_raw
    mse = float(np.mean(err ** 2)) if len(err) > 0 else np.nan
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    mae = float(np.mean(np.abs(err))) if len(err) > 0 else np.nan
    return {"rmse": rmse, "mae": mae}


def cross_sectional_rank_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    ics = []
    for _, g in pred_df.groupby("date"):
        if len(g) < 2:
            continue
        ic = spearman_rank_corr(g["pred_score"].values.astype(float), g["target_value_raw"].values.astype(float))
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) == 0:
        return {"mean_rank_ic": np.nan, "rank_ic_ir": np.nan, "n_ic_dates": 0}

    arr = np.array(ics, dtype=float)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ic_ir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
    return {"mean_rank_ic": ic_mean, "rank_ic_ir": ic_ir, "n_ic_dates": int(len(arr))}


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


# -----------------------------
# Execution-layer filters
# -----------------------------
def per_date_context_table(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-date context used for execution filters.
    Uses top1 stock after sorting by pred_score.
    """
    rows = []
    for dt, g in full_df.groupby("date", sort=True):
        g = g.sort_values("pred_score", ascending=False).reset_index(drop=True)

        top1 = g.iloc[0]
        top2 = g.iloc[1] if len(g) > 1 else g.iloc[0]
        top3 = g.iloc[2] if len(g) > 2 else g.iloc[min(1, len(g) - 1)]

        rows.append({
            "date": pd.Timestamp(dt),
            "top1_stock": top1["stock"],
            "top1_score": float(top1["pred_score"]),
            "top2_score": float(top2["pred_score"]),
            "top3_score": float(top3["pred_score"]),
            "score_gap_23": float(top2["pred_score"] - top3["pred_score"]),
            "top1_dc_trend": float(top1["dc_trend"]) if "dc_trend" in g.columns and pd.notna(top1.get("dc_trend")) else np.nan,
            "mkt_dc_trend": float(top1["mkt_dc_trend"]) if "mkt_dc_trend" in g.columns and pd.notna(top1.get("mkt_dc_trend")) else np.nan,
            "vix_z_60": float(top1["vix_z_60"]) if "vix_z_60" in g.columns and pd.notna(top1.get("vix_z_60")) else np.nan,
            "credit_stress": float(top1["credit_stress"]) if "credit_stress" in g.columns and pd.notna(top1.get("credit_stress")) else np.nan,
        })
    return pd.DataFrame(rows)


def attach_context_columns(pred_df: pd.DataFrame, source_df_with_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge feature columns needed for execution filters onto prediction df.
    """
    use_cols = ["date", "stock"]
    for c in ["dc_trend", "mkt_dc_trend", "vix_z_60", "credit_stress"]:
        if c in source_df_with_features.columns:
            use_cols.append(c)

    feat = source_df_with_features[use_cols].drop_duplicates(["date", "stock"]).copy()
    out = pred_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    feat["date"] = pd.to_datetime(feat["date"])
    out = out.merge(feat, on=["date", "stock"], how="left")
    return out


def execution_backtest_with_filters(
    pred_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    confidence_margin: float,
    use_align_filter: bool,
    vix_z_max: Optional[float],
    credit_stress_max: Optional[float],
) -> Dict[str, float]:
    """
    Fixed top2 execution with three states:
      - confidence false -> cash
      - confidence true, regime false -> half exposure to top2
      - confidence true, regime true  -> full exposure to top2
    """
    if pred_df.empty:
        return {
            "periods": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "avg_exposure": 0.0,
            "full_exposure_rate": 0.0,
            "half_exposure_rate": 0.0,
            "cash_rate": 0.0,
        }

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turns = []
    exposures = []
    states = []

    ctx = per_date_context_table(df).set_index("date")

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy().sort_values("pred_score", ascending=False).reset_index(drop=True)
        if len(day) == 0:
            continue

        ctx_row = ctx.loc[pd.Timestamp(dt)]
        confidence_ok = bool(ctx_row["score_gap_23"] >= confidence_margin)

        regime_ok = True
        if use_align_filter:
            tdc = ctx_row["top1_dc_trend"]
            mdc = ctx_row["mkt_dc_trend"]
            if pd.notna(tdc) and pd.notna(mdc):
                regime_ok = regime_ok and (tdc * mdc > 0)

        if vix_z_max is not None and pd.notna(ctx_row["vix_z_60"]):
            regime_ok = regime_ok and (ctx_row["vix_z_60"] <= vix_z_max)

        if credit_stress_max is not None and pd.notna(ctx_row["credit_stress"]):
            regime_ok = regime_ok and (ctx_row["credit_stress"] <= credit_stress_max)

        top2 = day.head(2)
        if confidence_ok and regime_ok:
            exposure = 1.0
            state = "full"
        elif confidence_ok and not regime_ok:
            exposure = 0.5
            state = "half"
        else:
            exposure = 0.0
            state = "cash"

        if exposure > 0 and len(top2) > 0:
            w_each = exposure / len(top2)
            new_w = {row["stock"]: w_each for _, row in top2.iterrows()}
        else:
            new_w = {}

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        exposures.append(exposure)
        states.append(state)
        prev_w = new_w

    arr_states = np.array(states, dtype=object)
    return {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_exposure": float(np.mean(exposures)) if exposures else 0.0,
        "full_exposure_rate": float(np.mean(arr_states == "full")) if len(arr_states) > 0 else 0.0,
        "half_exposure_rate": float(np.mean(arr_states == "half")) if len(arr_states) > 0 else 0.0,
        "cash_rate": float(np.mean(arr_states == "cash")) if len(arr_states) > 0 else 0.0,
    }


def choose_best_execution_filters(
    val_pred_df: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    confidence_margin_grid: List[float],
    use_align_filter_grid: List[bool],
    vix_z_max_grid: List[Optional[float]],
    credit_quantile_grid: List[Optional[float]],
    max_val_drawdown: float,
    min_active_periods: int,
    min_rank_ic: float,
    min_avg_exposure: float,
) -> Tuple[Dict, Dict, bool, pd.DataFrame]:
    val_rank = cross_sectional_rank_metrics(val_pred_df)
    mean_rank_ic = 0.0 if not np.isfinite(val_rank["mean_rank_ic"]) else val_rank["mean_rank_ic"]

    # quantile-based thresholds computed on validation only
    credit_values = val_pred_df["credit_stress"].dropna().values.astype(float) if "credit_stress" in val_pred_df.columns else np.array([])
    rows = []

    for conf_margin in confidence_margin_grid:
        for use_align in use_align_filter_grid:
            for vix_max in vix_z_max_grid:
                for cq in credit_quantile_grid:
                    if cq is None or len(credit_values) == 0:
                        credit_max = None
                    else:
                        credit_max = float(np.quantile(credit_values, cq))

                    stats = execution_backtest_with_filters(
                        pred_df=val_pred_df,
                        horizon=horizon,
                        transaction_cost_bps=transaction_cost_bps,
                        confidence_margin=conf_margin,
                        use_align_filter=use_align,
                        vix_z_max=vix_max,
                        credit_stress_max=credit_max,
                    )
                    constraints_ok = bool(
                        stats["periods"] >= min_active_periods
                        and stats["max_drawdown"] >= max_val_drawdown
                        and mean_rank_ic >= min_rank_ic
                        and stats["avg_exposure"] >= min_avg_exposure
                    )
                    rows.append({
                        "confidence_margin": conf_margin,
                        "use_align_filter": use_align,
                        "vix_z_max": vix_max,
                        "credit_quantile": cq,
                        "credit_stress_max": credit_max,
                        "mean_rank_ic": mean_rank_ic,
                        "constraints_ok": constraints_ok,
                        **stats,
                    })

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df["constraints_ok"] == True].copy()

    if len(valid) > 0:
        best = valid.sort_values(
            ["mean_rank_ic", "sharpe", "cumulative_return", "max_drawdown"],
            ascending=[False, False, False, False]
        ).iloc[0]
        best_filter = {
            "confidence_margin": float(best["confidence_margin"]),
            "use_align_filter": bool(best["use_align_filter"]),
            "vix_z_max": None if pd.isna(best["vix_z_max"]) else float(best["vix_z_max"]),
            "credit_stress_max": None if pd.isna(best["credit_stress_max"]) else float(best["credit_stress_max"]),
            "credit_quantile": None if pd.isna(best["credit_quantile"]) else float(best["credit_quantile"]),
        }
        return best_filter, best.to_dict(), True, grid_df

    grid_df["fallback_score"] = (
        3.0 * grid_df["mean_rank_ic"].fillna(0.0)
        + 0.5 * grid_df["sharpe"]
        + 2.0 * grid_df["cumulative_return"]
        - 4.0 * np.maximum(0.0, max_val_drawdown - grid_df["max_drawdown"])
        - 2.0 * np.maximum(0.0, min_avg_exposure - grid_df["avg_exposure"])
    )
    best = grid_df.sort_values(["fallback_score", "mean_rank_ic", "sharpe"], ascending=[False, False, False]).iloc[0]
    best_filter = {
        "confidence_margin": float(best["confidence_margin"]),
        "use_align_filter": bool(best["use_align_filter"]),
        "vix_z_max": None if pd.isna(best["vix_z_max"]) else float(best["vix_z_max"]),
        "credit_stress_max": None if pd.isna(best["credit_stress_max"]) else float(best["credit_stress_max"]),
        "credit_quantile": None if pd.isna(best["credit_quantile"]) else float(best["credit_quantile"]),
    }
    return best_filter, best.to_dict(), False, grid_df


# -----------------------------
# Training loop
# -----------------------------
def predict_group_scores(model: nn.Module, groups: List[Dict], device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for g in groups:
            xb = torch.tensor(g["X"], dtype=torch.float32, device=device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([])


def evaluate_group_loader(model: nn.Module, loader: DataLoader, device: torch.device, pointwise_criterion, args) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    point_losses = []
    rank_losses = []
    ys, preds = [], []

    with torch.no_grad():
        for batch in loader:
            xb = batch["X"].to(device)
            yb = batch["y_scaled"].to(device)
            pred = model(xb)

            point_loss = pointwise_criterion(pred, yb)
            rank_loss = pairwise_rank_loss(pred, yb, batch["group_sizes"], min_diff=args.rank_min_diff)

            point_losses.append(point_loss.item())
            rank_losses.append(rank_loss.item())
            ys.append(yb.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(preds) if preds else np.array([])
    return (
        float(np.mean(point_losses)) if point_losses else np.nan,
        float(np.mean(rank_losses)) if rank_losses else np.nan,
        y_true,
        y_pred,
    )


def train_grouped_model(
    model: nn.Module,
    train_groups: List[Dict],
    val_groups: List[Dict],
    target_scaler: TargetScaler,
    horizon: int,
    device: torch.device,
    args,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    train_loader = DataLoader(
        DateGroupDataset(train_groups),
        batch_size=args.group_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=group_collate,
    )
    val_loader = DataLoader(
        DateGroupDataset(val_groups),
        batch_size=args.group_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=group_collate,
    )

    pointwise_criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4)

    model = model.to(device)
    best_state = None
    best_epoch = -1
    best_score = -1e18
    wait = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_point_losses = []
        train_rank_losses = []

        for batch in train_loader:
            xb = batch["X"].to(device)
            yb = batch["y_scaled"].to(device)

            optimizer.zero_grad()
            pred = model(xb)

            point_loss = pointwise_criterion(pred, yb)
            rank_loss = pairwise_rank_loss(pred, yb, batch["group_sizes"], min_diff=args.rank_min_diff)
            loss = args.point_loss_weight * point_loss + args.rank_loss_weight * rank_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_point_losses.append(point_loss.item())
            train_rank_losses.append(rank_loss.item())

        train_point = float(np.mean(train_point_losses)) if train_point_losses else np.nan
        train_rank = float(np.mean(train_rank_losses)) if train_rank_losses else np.nan

        val_point, val_rank, val_y_scaled, val_pred_scaled = evaluate_group_loader(
            model=model,
            loader=val_loader,
            device=device,
            pointwise_criterion=pointwise_criterion,
            args=args,
        )

        val_pred_df = groups_to_prediction_df(val_groups, val_pred_scaled, target_scaler)
        val_reg = regression_metrics(val_pred_df["target_value_raw"].values, val_pred_df["pred_score"].values)
        val_rank_metrics = cross_sectional_rank_metrics(val_pred_df)

        # attach filter context for execution selection
        val_pred_df_ctx = attach_context_columns(val_pred_df, args.df_for_context)

        best_filter, best_filter_stats, constraints_ok, filter_grid_df = choose_best_execution_filters(
            val_pred_df=val_pred_df_ctx,
            horizon=horizon,
            transaction_cost_bps=args.transaction_cost_bps,
            confidence_margin_grid=args.confidence_margin_grid_list,
            use_align_filter_grid=args.use_align_filter_grid_list,
            vix_z_max_grid=args.vix_z_max_grid_list,
            credit_quantile_grid=args.credit_quantile_grid_list,
            max_val_drawdown=args.max_val_drawdown,
            min_active_periods=args.min_active_periods,
            min_rank_ic=args.min_rank_ic,
            min_avg_exposure=args.min_avg_exposure,
        )

        val_score = (
            args.selection_ic_weight * (0.0 if not np.isfinite(val_rank_metrics["mean_rank_ic"]) else val_rank_metrics["mean_rank_ic"])
            + args.selection_sharpe_weight * best_filter_stats["sharpe"]
            + args.selection_return_weight * best_filter_stats["cumulative_return"]
            - args.selection_mdd_penalty * max(0.0, args.max_val_drawdown - best_filter_stats["max_drawdown"])
        )
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_point_loss_scaled": train_point,
            "train_rank_loss": train_rank,
            "val_point_loss_scaled": val_point,
            "val_rank_loss": val_rank,
            "val_rmse_raw": val_reg["rmse"],
            "val_mae_raw": val_reg["mae"],
            "val_mean_rank_ic": val_rank_metrics["mean_rank_ic"],
            "val_rank_ic_ir": val_rank_metrics["rank_ic_ir"],
            "best_confidence_margin": best_filter["confidence_margin"],
            "best_use_align_filter": best_filter["use_align_filter"],
            "best_vix_z_max": best_filter["vix_z_max"],
            "best_credit_quantile": best_filter["credit_quantile"],
            "best_val_sharpe": best_filter_stats["sharpe"],
            "best_val_cumret": best_filter_stats["cumulative_return"],
            "best_val_mdd": best_filter_stats["max_drawdown"],
            "best_val_avg_exposure": best_filter_stats["avg_exposure"],
            "val_selection_score": val_score,
            "val_constraints_ok": constraints_ok,
        })

        print(
            f"[transformer_v31] epoch {epoch:03d} | lr={lr_now:.6f} | "
            f"train_point={train_point:.4f} | train_rank={train_rank:.4f} | "
            f"val_point={val_point:.4f} | val_rank={val_rank:.4f} | "
            f"val_rmse_raw={val_reg['rmse']:.5f} | val_ic={val_rank_metrics['mean_rank_ic']:.4f} | "
            f"val_sharpe={best_filter_stats['sharpe']:.4f} | val_cumret={best_filter_stats['cumulative_return']:.4f} | "
            f"val_mdd={best_filter_stats['max_drawdown']:.4f} | score={val_score:.4f}"
        )

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= args.min_epochs:
                wait += 1

        if epoch >= args.min_epochs and wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history), {
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
    }


# -----------------------------
# Main experiment
# -----------------------------
def run_static_experiment(df: pd.DataFrame, feats: List[str], horizon: int, device: torch.device, args, out_root: Path):
    train_raw = df[df["split"] == "train"].copy()

    target_scaler = TargetScaler(lower_q=args.target_lower_q, upper_q=args.target_upper_q).fit(train_raw["target_value_raw"].values)
    df = df.copy()
    df["target_value_scaled"] = target_scaler.transform(df["target_value_raw"].values)

    feat_mean, feat_std = fit_feature_standardizer(train_raw, feats)
    df_std = apply_feature_standardizer(df, feats, feat_mean, feat_std)

    groups = build_grouped_sequences(df_std, feats, seq_len=args.seq_len, split_col="split")
    split_groups = split_group_lists(groups)

    model = TransformerRegressor(
        input_dim=len(feats),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.transformer_layers,
        dropout=args.dropout,
    )

    # pass original feature df for context merge during validation selection
    args.df_for_context = df.copy()

    model, history_df, train_summary = train_grouped_model(
        model=model,
        train_groups=split_groups["train"],
        val_groups=split_groups["val"],
        target_scaler=target_scaler,
        horizon=horizon,
        device=device,
        args=args,
    )

    train_pred_scaled = predict_group_scores(model, split_groups["train"], device=device)
    val_pred_scaled = predict_group_scores(model, split_groups["val"], device=device)
    test_pred_scaled = predict_group_scores(model, split_groups["test"], device=device)

    train_pred_df = groups_to_prediction_df(split_groups["train"], train_pred_scaled, target_scaler)
    val_pred_df = groups_to_prediction_df(split_groups["val"], val_pred_scaled, target_scaler)
    test_pred_df = groups_to_prediction_df(split_groups["test"], test_pred_scaled, target_scaler)

    val_pred_df_ctx = attach_context_columns(val_pred_df, df.copy())
    test_pred_df_ctx = attach_context_columns(test_pred_df, df.copy())

    val_reg = regression_metrics(val_pred_df["target_value_raw"].values, val_pred_df["pred_score"].values)
    test_reg = regression_metrics(test_pred_df["target_value_raw"].values, test_pred_df["pred_score"].values)
    val_rank = cross_sectional_rank_metrics(val_pred_df)
    test_rank = cross_sectional_rank_metrics(test_pred_df)

    best_filter, best_filter_stats, constraints_ok, filter_grid_df = choose_best_execution_filters(
        val_pred_df=val_pred_df_ctx,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        confidence_margin_grid=args.confidence_margin_grid_list,
        use_align_filter_grid=args.use_align_filter_grid_list,
        vix_z_max_grid=args.vix_z_max_grid_list,
        credit_quantile_grid=args.credit_quantile_grid_list,
        max_val_drawdown=args.max_val_drawdown,
        min_active_periods=args.min_active_periods,
        min_rank_ic=args.min_rank_ic,
    )

    test_exec = execution_backtest_with_filters(
        pred_df=test_pred_df_ctx,
        horizon=horizon,
        transaction_cost_bps=args.transaction_cost_bps,
        confidence_margin=best_filter["confidence_margin"],
        use_align_filter=best_filter["use_align_filter"],
        vix_z_max=best_filter["vix_z_max"],
        credit_stress_max=best_filter["credit_stress_max"],
    )

    model_dir = out_root / "transformer_static_v31"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    history_df.to_csv(model_dir / "training_history.csv", index=False)
    filter_grid_df.to_csv(model_dir / "val_execution_filter_grid.csv", index=False)

    pred_all = pd.concat([
        train_pred_df.assign(split="train"),
        val_pred_df.assign(split="val"),
        test_pred_df.assign(split="test"),
    ], ignore_index=True)
    pred_all["date"] = pd.to_datetime(pred_all["date"]).dt.strftime("%Y-%m-%d")
    pred_all.to_csv(model_dir / "predictions_all_splits.csv", index=False)
    pred_all[pred_all["split"] == "test"].to_csv(model_dir / "predictions_test.csv", index=False)

    summary = {
        "mode": "static",
        "train_summary": train_summary,
        "target_scaler": target_scaler.to_dict(),
        "feature_count": len(feats),
        "val_regression_metrics_raw": val_reg,
        "test_regression_metrics_raw": test_reg,
        "val_rank_metrics": val_rank,
        "test_rank_metrics": test_rank,
        "selected_execution_filter": {
            **best_filter,
            "constraints_satisfied": bool(constraints_ok),
            "val_strategy": best_filter_stats,
            "test_strategy": test_exec,
        },
    }
    with open(model_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame([{
        "model": "transformer_v31",
        "mode": "static",
        "best_epoch": train_summary["best_epoch"],
        "val_rmse_raw": val_reg["rmse"],
        "test_rmse_raw": test_reg["rmse"],
        "val_mean_rank_ic": val_rank["mean_rank_ic"],
        "test_mean_rank_ic": test_rank["mean_rank_ic"],
        "selected_constraints_satisfied": bool(constraints_ok),
        "confidence_margin": best_filter["confidence_margin"],
        "use_align_filter": best_filter["use_align_filter"],
        "vix_z_max": best_filter["vix_z_max"],
        "credit_quantile": best_filter["credit_quantile"],
        "val_strategy_sharpe": best_filter_stats["sharpe"],
        "val_strategy_cumret": best_filter_stats["cumulative_return"],
        "val_strategy_mdd": best_filter_stats["max_drawdown"],
        "val_strategy_avg_exposure": best_filter_stats["avg_exposure"],
        "test_strategy_sharpe": test_exec["sharpe"],
        "test_strategy_cumret": test_exec["cumulative_return"],
        "test_strategy_mdd": test_exec["max_drawdown"],
        "test_strategy_avg_exposure": test_exec["avg_exposure"],
    }])
    comparison.to_csv(model_dir / "model_comparison.csv", index=False)

    print("\nFinished.")
    print(f"Saved outputs to: {model_dir}")
    print(comparison.to_string(index=False))


# -----------------------------
# CLI
# -----------------------------
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


def main():
    parser = argparse.ArgumentParser(description="Transformer ranking v3.1 with confidence/regime/exposure gates.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_mode", type=str, default="auto", choices=["auto", "raw", "excess_vs_market", "cross_sectional_demean"])

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--group_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--huber_beta", type=float, default=0.25)
    parser.add_argument("--point_loss_weight", type=float, default=0.4)
    parser.add_argument("--rank_loss_weight", type=float, default=0.6)
    parser.add_argument("--rank_min_diff", type=float, default=0.03)

    parser.add_argument("--target_lower_q", type=float, default=0.01)
    parser.add_argument("--target_upper_q", type=float, default=0.99)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)

    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)
    parser.add_argument("--min_active_periods", type=int, default=12)
    parser.add_argument("--min_rank_ic", type=float, default=0.01)

    parser.add_argument("--selection_ic_weight", type=float, default=3.0)
    parser.add_argument("--selection_sharpe_weight", type=float, default=0.5)
    parser.add_argument("--selection_return_weight", type=float, default=2.0)
    parser.add_argument("--selection_mdd_penalty", type=float, default=3.0)

    # execution filter grids
    parser.add_argument("--confidence_margin_grid", type=str, default="0.0,0.005,0.01")
    parser.add_argument("--use_align_filter_grid", type=str, default="true,false")
    parser.add_argument("--vix_z_max_grid", type=str, default="1.5,None")
    parser.add_argument("--credit_quantile_grid", type=str, default="0.8,None")
    parser.add_argument("--min_avg_exposure", type=float, default=0.25)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    args.confidence_margin_grid_list = [float(x.strip()) for x in args.confidence_margin_grid.split(",") if x.strip()]
    args.use_align_filter_grid_list = parse_bool_grid(args.use_align_filter_grid)
    args.vix_z_max_grid_list = parse_optional_float_grid(args.vix_z_max_grid)
    args.credit_quantile_grid_list = parse_optional_float_grid(args.credit_quantile_grid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    horizon = infer_horizon(return_col)
    actual_target_mode = auto_target_mode(df, return_col) if args.target_mode == "auto" else args.target_mode
    df = build_target_column(df, return_col, actual_target_mode)
    df, feats = build_v31_feature_columns(df, target_col, return_col)

    out_root = Path(args.out_dir) / f"{Path(args.data_path).stem}_ranking_v31_static_{actual_target_mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["return_col"] = return_col
    run_config["target_mode_actual"] = actual_target_mode
    run_config["horizon"] = horizon
    run_config["feature_columns"] = feats
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Detected return column: {return_col}")
    print(f"Horizon: {horizon}")
    print(f"Target mode: {actual_target_mode}")
    print(f"Feature count: {len(feats)}")
    print(df["split"].value_counts().to_string())

    run_static_experiment(df=df, feats=feats, horizon=horizon, device=device, args=args, out_root=out_root)


if __name__ == "__main__":
    main()


import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# H5 excess-return ranking v2
#
# What this version changes
# -------------------------
# 1) Target is no longer raw future_return_5d by default.
#    Recommended target:
#       excess_return_5d = future_return_5d - mkt_return_5d
#    Fallback:
#       cross-sectional demeaned future_return_5d
#
# 2) Target is train-only winsorized + z-scored before training.
#    This makes optimization more stable and makes loss values more
#    interpretable. Small raw decimal returns no longer create the
#    illusion of "too-small loss".
#
# 3) Static mode:
#    - fit on train
#    - choose fixed top-k on official validation
#    - evaluate on official test
#
# 4) Rolling mode:
#    - first choose a FIXED top-k on official validation
#    - then only update model parameters during the test period
#    - no repeated top-k tuning inside tiny inner windows
#
# 5) Output:
#    - regression metrics
#    - rank IC metrics
#    - strategy backtest metrics
#    - predictions files
#
# Recommended first run:
#
# python train_h5_excess_return_ranking_v2.py ^
#   --data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs" ^
#   --model both ^
#   --eval_mode static
#
# Then:
#
# python train_h5_excess_return_ranking_v2.py ^
#   --data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs" ^
#   --model both ^
#   --eval_mode rolling
# =========================================================


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Basic helpers
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


def feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> List[str]:
    excluded = {"date", "stock", "split", target_col, return_col, "target_value_raw"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def fit_standardizer(train_df: pd.DataFrame, feats: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train_df[feats].mean()
    std = train_df[feats].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_standardizer(df: pd.DataFrame, feats: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feats] = (out[feats] - mean) / std
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


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


def list_to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


# -----------------------------
# Target engineering
# -----------------------------
class TargetScaler:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_: Optional[float] = None
        self.upper_: Optional[float] = None
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

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


def build_target_column(df: pd.DataFrame, return_col: str, target_mode: str) -> pd.DataFrame:
    out = df.copy()
    if target_mode == "raw":
        out["target_value_raw"] = out[return_col].astype(float)
    elif target_mode == "excess_vs_market":
        if "mkt_return_5d" in out.columns:
            out["target_value_raw"] = out[return_col].astype(float) - out["mkt_return_5d"].astype(float)
        elif "mkt_return_1d" in out.columns and return_col.endswith("1d"):
            out["target_value_raw"] = out[return_col].astype(float) - out["mkt_return_1d"].astype(float)
        else:
            raise ValueError("Requested target_mode=excess_vs_market but matching market return column was not found.")
    elif target_mode == "cross_sectional_demean":
        out["target_value_raw"] = (
            out[return_col].astype(float)
            - out.groupby("date")[return_col].transform("mean").astype(float)
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return out


def auto_target_mode(df: pd.DataFrame, return_col: str) -> str:
    if return_col.endswith("5d") and "mkt_return_5d" in df.columns:
        return "excess_vs_market"
    if return_col.endswith("1d") and "mkt_return_1d" in df.columns:
        return "excess_vs_market"
    return "cross_sectional_demean"


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
    def __init__(self, input_dim: int, d_model: int = 48, nhead: int = 4, num_layers: int = 1, dropout: float = 0.2):
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


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


def make_model(model_name: str, input_dim: int, args):
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
        )
    if model_name == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.transformer_layers,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# Dataset
# -----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(
    df: pd.DataFrame,
    feats: List[str],
    target_col_name: str,
    seq_len: int,
    split_col: str = "split",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list, y_list, meta = [], [], []

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].values.astype(np.float32)
        yg = g[target_col_name].values.astype(np.float32)

        for i in range(seq_len - 1, len(g)):
            X_list.append(Xg[i - seq_len + 1:i + 1])
            y_list.append(yg[i])
            meta.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "split": g.loc[i, split_col],
                "target_value_raw": float(g.loc[i, "target_value_raw"]),
                "future_return": float(g.loc[i, [c for c in g.columns if c.startswith("future_return_")][0]]),
            })

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def split_indices(meta_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "train": np.where(meta_df["split"].values == "train")[0],
        "val": np.where(meta_df["split"].values == "val")[0],
        "test": np.where(meta_df["split"].values == "test")[0],
    }


# -----------------------------
# Metrics
# -----------------------------
def regression_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    err = y_pred_raw - y_true_raw
    mse = float(np.mean(err ** 2)) if len(err) > 0 else np.nan
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    mae = float(np.mean(np.abs(err))) if len(err) > 0 else np.nan
    return {"rmse": rmse, "mae": mae}


def cross_sectional_rank_metrics(pred_df: pd.DataFrame, pred_col: str = "pred_score", target_col: str = "target_value_raw") -> Dict[str, float]:
    ics = []
    for _, g in pred_df.groupby("date"):
        if len(g) < 2:
            continue
        ic = spearman_rank_corr(g[pred_col].values.astype(float), g[target_col].values.astype(float))
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) == 0:
        return {"mean_rank_ic": np.nan, "rank_ic_ir": np.nan, "n_ic_dates": 0}

    arr = np.array(ics, dtype=float)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ic_ir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
    return {"mean_rank_ic": ic_mean, "rank_ic_ir": ic_ir, "n_ic_dates": int(len(arr))}


# -----------------------------
# Backtest
# -----------------------------
def select_topk_weights(day_df: pd.DataFrame, k: int) -> Dict[str, float]:
    day_df = day_df.sort_values("pred_score", ascending=False).copy()
    chosen = day_df.head(k)
    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def topk_backtest(pred_df: pd.DataFrame, horizon: int, top_k: int, transaction_cost_bps: float = 10.0) -> Dict[str, float]:
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
    holds = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        new_w = select_topk_weights(day, k=top_k)
        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))
        prev_w = new_w

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
    }


def choose_best_topk(
    val_pred_df: pd.DataFrame,
    horizon: int,
    topk_grid: List[int],
    transaction_cost_bps: float,
    max_val_drawdown: float,
    min_active_periods: int,
) -> Tuple[int, Dict[str, float], bool, pd.DataFrame]:
    rows = []
    for k in topk_grid:
        stats = topk_backtest(val_pred_df, horizon=horizon, top_k=k, transaction_cost_bps=transaction_cost_bps)
        constraints_ok = bool(stats["periods"] >= min_active_periods and stats["max_drawdown"] >= max_val_drawdown)
        rows.append({"top_k": int(k), "constraints_ok": constraints_ok, **stats})

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df["constraints_ok"] == True].copy()

    if len(valid) > 0:
        best = valid.sort_values(["sharpe", "cumulative_return", "max_drawdown"], ascending=[False, False, False]).iloc[0]
        return int(best["top_k"]), best.to_dict(), True, grid_df

    grid_df["fallback_score"] = (
        grid_df["sharpe"]
        - 4.0 * np.maximum(0.0, max_val_drawdown - grid_df["max_drawdown"])
        - 0.10 * np.maximum(0.0, min_active_periods - grid_df["periods"])
    )
    best = grid_df.sort_values(["fallback_score", "sharpe", "cumulative_return"], ascending=[False, False, False]).iloc[0]
    return int(best["top_k"]), best.to_dict(), False, grid_df


# -----------------------------
# Training
# -----------------------------
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys, yhats = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(loss.item() * len(xb))
            ys.append(yb.detach().cpu().numpy())
            yhats.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(yhats) if yhats else np.array([])
    avg_loss = float(sum(losses) / max(1, len(loader.dataset)))
    return avg_loss, y_true, y_pred


def predict_scaled(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([])


def train_regression_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_meta_df: pd.DataFrame,
    target_scaler: TargetScaler,
    horizon: int,
    device: torch.device,
    args,
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4
    )

    model = model.to(device)
    history = []
    best_state = None
    best_epoch = -1
    best_score = -1e18
    wait = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item() * len(xb))

        train_loss = float(sum(train_losses) / max(1, len(train_loader.dataset)))

        val_loss, val_y_scaled, val_pred_scaled = evaluate_loader(model, val_loader, device, criterion)
        val_y_raw = target_scaler.inverse_transform(val_y_scaled)
        val_pred_raw = target_scaler.inverse_transform(val_pred_scaled)

        val_pred_df = val_meta_df.copy()
        val_pred_df["pred_score"] = val_pred_raw

        val_reg = regression_metrics(val_y_raw, val_pred_raw)
        val_rank = cross_sectional_rank_metrics(val_pred_df)
        best_k, best_k_stats, constraints_ok, _ = choose_best_topk(
            val_pred_df=val_pred_df,
            horizon=horizon,
            topk_grid=args.topk_grid_list,
            transaction_cost_bps=args.transaction_cost_bps,
            max_val_drawdown=args.max_val_drawdown,
            min_active_periods=args.min_active_periods,
        )

        val_score = (
            args.selection_ic_weight * (0.0 if not np.isfinite(val_rank["mean_rank_ic"]) else val_rank["mean_rank_ic"])
            + args.selection_sharpe_weight * best_k_stats["sharpe"]
            + args.selection_return_weight * best_k_stats["cumulative_return"]
            - args.selection_mdd_penalty * max(0.0, args.max_val_drawdown - best_k_stats["max_drawdown"])
        )

        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "model": model_name,
            "epoch": epoch,
            "lr": lr_now,
            "train_loss_scaled": train_loss,
            "val_loss_scaled": val_loss,
            "val_rmse_raw": val_reg["rmse"],
            "val_mae_raw": val_reg["mae"],
            "val_mean_rank_ic": val_rank["mean_rank_ic"],
            "val_rank_ic_ir": val_rank["rank_ic_ir"],
            "val_selected_topk": int(best_k),
            "val_selected_sharpe": best_k_stats["sharpe"],
            "val_selected_cumret": best_k_stats["cumulative_return"],
            "val_selected_mdd": best_k_stats["max_drawdown"],
            "val_selection_score": val_score,
            "val_constraints_ok": constraints_ok,
        })

        print(
            f"[{model_name}] epoch {epoch:03d} | lr={lr_now:.6f} | "
            f"train_loss_scaled={train_loss:.5f} | val_loss_scaled={val_loss:.5f} | "
            f"val_rmse_raw={val_reg['rmse']:.5f} | val_ic={val_rank['mean_rank_ic']:.4f} | "
            f"val_sharpe={best_k_stats['sharpe']:.4f} | val_cumret={best_k_stats['cumulative_return']:.4f} | "
            f"score={val_score:.4f}"
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
# Static experiment
# -----------------------------
def run_static_experiment(
    df: pd.DataFrame,
    feats: List[str],
    return_col: str,
    horizon: int,
    model_name: str,
    device: torch.device,
    args,
    out_root: Path,
):
    train_raw = df[df["split"] == "train"].copy()

    # target scaler on train only
    target_scaler = TargetScaler(lower_q=args.target_lower_q, upper_q=args.target_upper_q).fit(train_raw["target_value_raw"].values)
    df = df.copy()
    df["target_value_scaled"] = np.nan
    df.loc[:, "target_value_scaled"] = target_scaler.transform(df["target_value_raw"].values)

    # feature scaler on train only
    mean, std = fit_standardizer(train_raw, feats)
    df_std = apply_standardizer(df, feats, mean, std)

    X, y_scaled, meta_df = build_sequences(df_std, feats, "target_value_scaled", seq_len=args.seq_len, split_col="split")
    idx = split_indices(meta_df)

    train_ds = SequenceDataset(X[idx["train"]], y_scaled[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y_scaled[idx["val"]])
    test_ds = SequenceDataset(X[idx["test"]], y_scaled[idx["test"]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(model_name, input_dim=len(feats), args=args)
    val_meta_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)

    model, history_df, train_summary = train_regression_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_meta_df=val_meta_df,
        target_scaler=target_scaler,
        horizon=horizon,
        device=device,
        args=args,
    )

    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    _, val_y_scaled, val_pred_scaled = evaluate_loader(model, val_loader, device, criterion)
    _, test_y_scaled, test_pred_scaled = evaluate_loader(model, test_loader, device, criterion)

    val_y_raw = target_scaler.inverse_transform(val_y_scaled)
    val_pred_raw = target_scaler.inverse_transform(val_pred_scaled)
    test_y_raw = target_scaler.inverse_transform(test_y_scaled)
    test_pred_raw = target_scaler.inverse_transform(test_pred_scaled)

    val_pred_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    val_pred_df["pred_score"] = val_pred_raw
    test_pred_df = meta_df.iloc[idx["test"]].copy().reset_index(drop=True)
    test_pred_df["pred_score"] = test_pred_raw

    val_reg = regression_metrics(val_y_raw, val_pred_raw)
    test_reg = regression_metrics(test_y_raw, test_pred_raw)
    val_rank = cross_sectional_rank_metrics(val_pred_df)
    test_rank = cross_sectional_rank_metrics(test_pred_df)

    best_k, best_k_stats, constraints_ok, grid_df = choose_best_topk(
        val_pred_df=val_pred_df,
        horizon=horizon,
        topk_grid=args.topk_grid_list,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        min_active_periods=args.min_active_periods,
    )
    test_bt = topk_backtest(test_pred_df, horizon=horizon, top_k=best_k, transaction_cost_bps=args.transaction_cost_bps)

    # train/val/test predictions for inspection
    train_pred_scaled = predict_scaled(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False), device)
    train_pred_raw = target_scaler.inverse_transform(train_pred_scaled)

    pred_all = meta_df.copy()
    pred_all["pred_score"] = np.nan
    pred_all.loc[idx["train"], "pred_score"] = train_pred_raw
    pred_all.loc[idx["val"], "pred_score"] = val_pred_raw
    pred_all.loc[idx["test"], "pred_score"] = test_pred_raw

    model_dir = out_root / f"{model_name}_static"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "best_model.pt")
    history_df.to_csv(model_dir / "training_history.csv", index=False)
    grid_df.to_csv(model_dir / "val_topk_grid.csv", index=False)
    pred_out = pred_all.copy()
    pred_out["date"] = pred_out["date"].dt.strftime("%Y-%m-%d")
    pred_out.to_csv(model_dir / "predictions_all_splits.csv", index=False)
    pred_out[pred_out["split"] == "test"].to_csv(model_dir / "predictions_test.csv", index=False)

    summary = {
        "mode": "static",
        "train_summary": train_summary,
        "target_scaler": target_scaler.to_dict(),
        "feature_count": len(feats),
        "val_regression_metrics_raw": val_reg,
        "test_regression_metrics_raw": test_reg,
        "val_rank_metrics": val_rank,
        "test_rank_metrics": test_rank,
        "selected_topk": {
            "top_k": int(best_k),
            "constraints_satisfied": bool(constraints_ok),
            "val_strategy": best_k_stats,
            "test_strategy": test_bt,
        },
    }
    with open(model_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "model": model_name,
        "mode": "static",
        "best_epoch": train_summary["best_epoch"],
        "val_rmse_raw": val_reg["rmse"],
        "test_rmse_raw": test_reg["rmse"],
        "val_mean_rank_ic": val_rank["mean_rank_ic"],
        "test_mean_rank_ic": test_rank["mean_rank_ic"],
        "selected_top_k": int(best_k),
        "selected_constraints_satisfied": bool(constraints_ok),
        "val_strategy_sharpe": best_k_stats["sharpe"],
        "val_strategy_cumret": best_k_stats["cumulative_return"],
        "val_strategy_mdd": best_k_stats["max_drawdown"],
        "test_strategy_sharpe": test_bt["sharpe"],
        "test_strategy_cumret": test_bt["cumulative_return"],
        "test_strategy_mdd": test_bt["max_drawdown"],
    }


# -----------------------------
# Rolling experiment
# -----------------------------
def assign_temp_split(hist_df: pd.DataFrame, val_days: int) -> pd.DataFrame:
    out = hist_df.copy().sort_values(["date", "stock"]).reset_index(drop=True)
    unique_dates = sorted(pd.to_datetime(out["date"]).drop_duplicates().tolist())
    if len(unique_dates) <= val_days + 20:
        raise ValueError("Not enough history for rolling inner train/val split.")
    val_date_set = set(unique_dates[-val_days:])
    out["split_temp"] = np.where(out["date"].isin(val_date_set), "val", "train")
    return out


def build_prediction_sequences_for_date(
    df_std: pd.DataFrame,
    feats: List[str],
    target_date: pd.Timestamp,
    seq_len: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    X_list, meta = [], []
    df_std = df_std.sort_values(["stock", "date"]).reset_index(drop=True)

    for stock, g in df_std.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(g["date"]).tolist()
        if target_date not in dates:
            continue
        i = dates.index(target_date)
        if i < seq_len - 1:
            continue
        X_list.append(g[feats].values.astype(np.float32)[i - seq_len + 1:i + 1])
        meta.append({
            "date": pd.Timestamp(target_date),
            "stock": stock,
            "target_value_raw": float(g.loc[i, "target_value_raw"]),
            "future_return": float(g.loc[i, [c for c in g.columns if c.startswith("future_return_")][0]]),
            "split": "test",
        })

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feats)), dtype=np.float32), pd.DataFrame()
    return np.stack(X_list), pd.DataFrame(meta)


def cutoff_date_for_target(unique_dates: List[pd.Timestamp], target_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    idx = unique_dates.index(target_date)
    if idx - horizon < 0:
        raise ValueError("Not enough past dates to avoid label leakage.")
    return unique_dates[idx - horizon]


def train_inner_rolling_fit(
    hist_df: pd.DataFrame,
    feats: List[str],
    horizon: int,
    model_name: str,
    device: torch.device,
    args,
):
    hist_df = assign_temp_split(hist_df, val_days=args.rolling_val_days)

    train_raw = hist_df[hist_df["split_temp"] == "train"].copy()
    target_scaler = TargetScaler(lower_q=args.target_lower_q, upper_q=args.target_upper_q).fit(train_raw["target_value_raw"].values)
    hist_df = hist_df.copy()
    hist_df["target_value_scaled"] = target_scaler.transform(hist_df["target_value_raw"].values)

    mean, std = fit_standardizer(train_raw, feats)
    hist_std = apply_standardizer(hist_df, feats, mean, std)

    X, y_scaled, meta_df = build_sequences(hist_std, feats, "target_value_scaled", seq_len=args.seq_len, split_col="split_temp")
    idx = split_indices(meta_df)

    train_ds = SequenceDataset(X[idx["train"]], y_scaled[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y_scaled[idx["val"]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(model_name, input_dim=len(feats), args=args)
    val_meta_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)

    inner_args = argparse.Namespace(**vars(args))
    inner_args.max_epochs = args.rolling_retrain_epochs
    inner_args.min_epochs = min(args.min_epochs, max(3, args.rolling_retrain_epochs // 2))
    inner_args.patience = min(args.patience, max(2, args.rolling_retrain_epochs // 2))

    model, history_df, fit_summary = train_regression_model(
        model_name=f"{model_name}_rolling_fit",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_meta_df=val_meta_df,
        target_scaler=target_scaler,
        horizon=horizon,
        device=device,
        args=inner_args,
    )

    return model, target_scaler, mean, std, fit_summary


def choose_fixed_topk_from_official_validation(
    df: pd.DataFrame,
    feats: List[str],
    horizon: int,
    model_name: str,
    device: torch.device,
    args,
) -> Tuple[int, Dict[str, float]]:
    """
    One-time outer validation selection on official train/val split.
    This fixed top-k is then used throughout rolling test evaluation.
    """
    train_raw = df[df["split"] == "train"].copy()
    target_scaler = TargetScaler(lower_q=args.target_lower_q, upper_q=args.target_upper_q).fit(train_raw["target_value_raw"].values)
    df = df.copy()
    df["target_value_scaled"] = target_scaler.transform(df["target_value_raw"].values)

    mean, std = fit_standardizer(train_raw, feats)
    df_std = apply_standardizer(df, feats, mean, std)

    X, y_scaled, meta_df = build_sequences(df_std, feats, "target_value_scaled", seq_len=args.seq_len, split_col="split")
    idx = split_indices(meta_df)

    train_ds = SequenceDataset(X[idx["train"]], y_scaled[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y_scaled[idx["val"]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(model_name, input_dim=len(feats), args=args)
    val_meta_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)

    outer_args = argparse.Namespace(**vars(args))
    outer_args.max_epochs = args.fixed_topk_selection_epochs
    outer_args.min_epochs = min(args.min_epochs, max(3, args.fixed_topk_selection_epochs // 2))
    outer_args.patience = min(args.patience, max(2, args.fixed_topk_selection_epochs // 2))

    model, _, _ = train_regression_model(
        model_name=f"{model_name}_fixed_topk_selector",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_meta_df=val_meta_df,
        target_scaler=target_scaler,
        horizon=horizon,
        device=device,
        args=outer_args,
    )

    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    _, val_y_scaled, val_pred_scaled = evaluate_loader(model, val_loader, device, criterion)
    val_pred_raw = target_scaler.inverse_transform(val_pred_scaled)

    val_pred_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    val_pred_df["pred_score"] = val_pred_raw

    best_k, best_k_stats, constraints_ok, grid_df = choose_best_topk(
        val_pred_df=val_pred_df,
        horizon=horizon,
        topk_grid=args.topk_grid_list,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        min_active_periods=args.min_active_periods,
    )
    return best_k, {
        "constraints_ok": bool(constraints_ok),
        "best_k_stats": best_k_stats,
    }


def run_rolling_experiment(
    df: pd.DataFrame,
    feats: List[str],
    horizon: int,
    model_name: str,
    device: torch.device,
    args,
    out_root: Path,
):
    # choose fixed top-k once from official validation
    fixed_top_k, fixed_topk_meta = choose_fixed_topk_from_official_validation(
        df=df,
        feats=feats,
        horizon=horizon,
        model_name=model_name,
        device=device,
        args=args,
    )

    df = df.copy().sort_values(["date", "stock"]).reset_index(drop=True)
    test_dates = sorted(pd.to_datetime(df[df["split"] == "test"]["date"]).drop_duplicates().tolist())[::horizon]
    all_dates = sorted(pd.to_datetime(df["date"]).drop_duplicates().tolist())

    pred_rows = []
    retrain_rows = []

    current_model = None
    current_target_scaler = None
    current_feat_mean = None
    current_feat_std = None

    for j, target_date in enumerate(test_dates):
        need_refit = (j == 0) or (j % args.retrain_every_n_rebalances == 0)

        if need_refit:
            try:
                cutoff_dt = cutoff_date_for_target(all_dates, target_date, horizon)
            except Exception:
                continue

            hist_df = df[df["date"] <= cutoff_dt].copy()
            if hist_df["date"].nunique() < args.seq_len + args.rolling_val_days + 30:
                continue

            current_model, current_target_scaler, current_feat_mean, current_feat_std, fit_summary = train_inner_rolling_fit(
                hist_df=hist_df,
                feats=feats,
                horizon=horizon,
                model_name=model_name,
                device=device,
                args=args,
            )

            retrain_rows.append({
                "retrain_for_target_date": str(pd.Timestamp(target_date).date()),
                "label_cutoff_date": str(pd.Timestamp(cutoff_dt).date()),
                "best_epoch": int(fit_summary["best_epoch"]),
                "fixed_top_k": int(fixed_top_k),
            })

        if current_model is None:
            continue

        hist_up_to_target = df[df["date"] <= target_date].copy()
        hist_up_to_target_std = apply_standardizer(hist_up_to_target, feats, current_feat_mean, current_feat_std)

        X_pred, meta_pred_df = build_prediction_sequences_for_date(
            df_std=hist_up_to_target_std,
            feats=feats,
            target_date=target_date,
            seq_len=args.seq_len,
        )
        if len(X_pred) == 0 or meta_pred_df.empty:
            continue

        pred_ds = SequenceDataset(X_pred, np.zeros(len(X_pred), dtype=np.float32))
        pred_loader = DataLoader(pred_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        pred_scaled = predict_scaled(current_model, pred_loader, device)
        pred_raw = current_target_scaler.inverse_transform(pred_scaled)

        meta_pred_df["pred_score"] = pred_raw
        meta_pred_df["chosen_top_k"] = int(fixed_top_k)
        pred_rows.append(meta_pred_df)

    if len(pred_rows) == 0:
        raise ValueError("Rolling experiment produced no test predictions.")

    pred_test_df = pd.concat(pred_rows, ignore_index=True)
    test_rank = cross_sectional_rank_metrics(pred_test_df)
    test_bt = topk_backtest(pred_test_df, horizon=horizon, top_k=fixed_top_k, transaction_cost_bps=args.transaction_cost_bps)

    model_dir = out_root / f"{model_name}_rolling"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_out = pred_test_df.copy()
    pred_out["date"] = pred_out["date"].dt.strftime("%Y-%m-%d")
    pred_out.to_csv(model_dir / "rolling_predictions_test.csv", index=False)
    pd.DataFrame(retrain_rows).to_csv(model_dir / "retrain_log.csv", index=False)

    summary = {
        "mode": "rolling",
        "fixed_top_k_selection": {
            "top_k": int(fixed_top_k),
            **fixed_topk_meta
        },
        "test_rank_metrics": test_rank,
        "test_strategy": test_bt,
    }
    with open(model_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "model": model_name,
        "mode": "rolling",
        "best_epoch": np.nan,
        "val_rmse_raw": np.nan,
        "test_rmse_raw": np.nan,
        "val_mean_rank_ic": np.nan,
        "test_mean_rank_ic": test_rank["mean_rank_ic"],
        "selected_top_k": int(fixed_top_k),
        "selected_constraints_satisfied": fixed_topk_meta["constraints_ok"],
        "val_strategy_sharpe": fixed_topk_meta["best_k_stats"]["sharpe"],
        "val_strategy_cumret": fixed_topk_meta["best_k_stats"]["cumulative_return"],
        "val_strategy_mdd": fixed_topk_meta["best_k_stats"]["max_drawdown"],
        "test_strategy_sharpe": test_bt["sharpe"],
        "test_strategy_cumret": test_bt["cumulative_return"],
        "test_strategy_mdd": test_bt["max_drawdown"],
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="H5 excess-return ranking v2.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="both", choices=["lstm", "transformer", "both"])
    parser.add_argument("--eval_mode", type=str, default="static", choices=["static", "rolling"])
    parser.add_argument("--target_mode", type=str, default="auto", choices=["auto", "raw", "excess_vs_market", "cross_sectional_demean"])

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--huber_beta", type=float, default=0.25)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)

    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--target_lower_q", type=float, default=0.01)
    parser.add_argument("--target_upper_q", type=float, default=0.99)

    parser.add_argument("--topk_grid", type=str, default="1,2,3")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)
    parser.add_argument("--min_active_periods", type=int, default=12)

    parser.add_argument("--selection_ic_weight", type=float, default=1.0)
    parser.add_argument("--selection_sharpe_weight", type=float, default=0.5)
    parser.add_argument("--selection_return_weight", type=float, default=2.0)
    parser.add_argument("--selection_mdd_penalty", type=float, default=2.0)

    # rolling options
    parser.add_argument("--rolling_val_days", type=int, default=120)
    parser.add_argument("--rolling_retrain_epochs", type=int, default=8)
    parser.add_argument("--retrain_every_n_rebalances", type=int, default=6)
    parser.add_argument("--fixed_topk_selection_epochs", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.topk_grid_list = [int(x.strip()) for x in args.topk_grid.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    horizon = infer_horizon(return_col)

    actual_target_mode = auto_target_mode(df, return_col) if args.target_mode == "auto" else args.target_mode
    df = build_target_column(df, return_col=return_col, target_mode=actual_target_mode)

    feats = feature_columns(df, target_col, return_col)

    out_root = Path(args.out_dir) / f"{Path(args.data_path).stem}_excess_return_ranking_v2_{args.eval_mode}_{actual_target_mode}"
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
    print(df['split'].value_counts().to_string())

    models = ["lstm", "transformer"] if args.model == "both" else [args.model]
    comparison_rows = []

    for model_name in models:
        print("\n" + "=" * 100)
        print(f"Running model={model_name} | eval_mode={args.eval_mode} | target_mode={actual_target_mode}")
        print("=" * 100)

        if args.eval_mode == "static":
            row = run_static_experiment(
                df=df,
                feats=feats,
                return_col=return_col,
                horizon=horizon,
                model_name=model_name,
                device=device,
                args=args,
                out_root=out_root,
            )
        else:
            row = run_rolling_experiment(
                df=df,
                feats=feats,
                horizon=horizon,
                model_name=model_name,
                device=device,
                args=args,
                out_root=out_root,
            )

        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out_root / "model_comparison.csv", index=False)

    print("\nFinished.")
    print(f"Saved outputs to: {out_root}")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()

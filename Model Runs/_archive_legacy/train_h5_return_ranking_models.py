
import os
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
# H5 return-regression + ranking backtest script
#
# Main idea
# ---------
# Instead of classifying target_up_5d, this script predicts future_return_5d
# directly, then uses the predicted score for cross-sectional ranking.
#
# What it supports
# ----------------
# 1) Sequence models:
#      - LSTM
#      - Transformer
#
# 2) Objective:
#      - regression on future_return_* (recommended: future_return_5d)
#
# 3) Strategy layer:
#      - cross-sectional top-k equal-weight backtest
#      - validation-time top-k selection
#
# 4) Evaluation modes:
#      - static: one fit on train, choose top-k on val, evaluate on test
#      - rolling: expanding-history re-fit during the test period
#
# Recommended first use
# ---------------------
# Static mode on main_experiment_h5.csv:
#
# python train_h5_return_ranking_models.py ^
#   --data_path "D:\python\dissertation\Data Acquisition\cleaned_datasets\main_experiment_h5.csv" ^
#   --out_dir "D:\python\dissertation\Model Runs" ^
#   --model both ^
#   --eval_mode static
#
# Then if useful, try:
#   --eval_mode rolling
# =========================================================


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    raise ValueError(f"Cannot infer horizon from return column: {return_col}")


def feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> List[str]:
    excluded = {"date", "stock", "split", target_col, return_col}
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
    return_col: str,
    seq_len: int,
    split_col: str = "split",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list, y_list, meta = [], [], []

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].values.astype(np.float32)
        yg = g[return_col].values.astype(np.float32)

        for i in range(seq_len - 1, len(g)):
            X_list.append(Xg[i - seq_len + 1:i + 1])
            y_list.append(yg[i])
            meta.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "split": g.loc[i, split_col],
                "future_return": float(g.loc[i, return_col]),
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
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2)) if len(err) > 0 else np.nan
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    mae = float(np.mean(np.abs(err))) if len(err) > 0 else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
    }


def cross_sectional_rank_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    ics = []
    for _, g in pred_df.groupby("date"):
        if len(g) < 2:
            continue
        ic = spearman_rank_corr(g["pred_score"].values.astype(float), g["future_return"].values.astype(float))
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) == 0:
        return {"mean_rank_ic": np.nan, "rank_ic_ir": np.nan, "n_ic_dates": 0}

    arr = np.array(ics, dtype=float)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ic_ir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
    return {
        "mean_rank_ic": ic_mean,
        "rank_ic_ir": ic_ir,
        "n_ic_dates": int(len(arr)),
    }


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
    holdings = []

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
        holdings.append(len(new_w))
        prev_w = new_w

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holdings)) if holdings else 0.0,
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
        rows.append({
            "top_k": int(k),
            "constraints_ok": constraints_ok,
            **stats
        })
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
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            yhat = model(xb).detach().cpu().numpy()
            preds.append(yhat)
    return np.concatenate(preds) if preds else np.array([])


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys, yhats = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            losses.append(loss.item() * len(xb))
            ys.append(yb.detach().cpu().numpy())
            yhats.append(yhat.detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(yhats) if yhats else np.array([])
    avg_loss = float(sum(losses) / max(1, len(loader.dataset)))
    return avg_loss, y_true, y_pred


def train_regression_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_meta_df: pd.DataFrame,
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
    best_val_score = -1e18
    wait = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item() * len(xb))

        train_loss = float(sum(train_losses) / max(1, len(train_loader.dataset)))

        val_loss, val_y, val_pred = evaluate_loader(model, val_loader, device, criterion)
        val_pred_df = val_meta_df.copy()
        val_pred_df["pred_score"] = val_pred

        val_reg = regression_metrics(val_y, val_pred)
        val_rank = cross_sectional_rank_metrics(val_pred_df)

        # economic signal selection on validation
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
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_reg["rmse"],
            "val_mae": val_reg["mae"],
            "val_mean_rank_ic": val_rank["mean_rank_ic"],
            "val_rank_ic_ir": val_rank["rank_ic_ir"],
            "val_selected_topk": best_k,
            "val_selected_sharpe": best_k_stats["sharpe"],
            "val_selected_cumret": best_k_stats["cumulative_return"],
            "val_selected_mdd": best_k_stats["max_drawdown"],
            "val_selection_score": val_score,
            "val_constraints_ok": constraints_ok,
        })

        print(
            f"[{model_name}] epoch {epoch:03d} | lr={lr_now:.6f} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"val_ic={val_rank['mean_rank_ic']:.4f} | val_sharpe={best_k_stats['sharpe']:.4f} | "
            f"val_cumret={best_k_stats['cumulative_return']:.4f} | score={val_score:.4f}"
        )

        if val_score > best_val_score + 1e-6:
            best_val_score = val_score
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

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_val_score),
    }
    return model, pd.DataFrame(history), summary


# -----------------------------
# Static pipeline
# -----------------------------
def run_static_experiment(df: pd.DataFrame, feats: List[str], return_col: str, horizon: int, model_name: str, device: torch.device, args, out_root: Path):
    train_raw = df[df["split"] == "train"].copy()
    mean, std = fit_standardizer(train_raw, feats)
    df_std = apply_standardizer(df, feats, mean, std)

    X, y, meta_df = build_sequences(df_std, feats, return_col, seq_len=args.seq_len, split_col="split")
    idx = split_indices(meta_df)

    train_ds = SequenceDataset(X[idx["train"]], y[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y[idx["val"]])
    test_ds = SequenceDataset(X[idx["test"]], y[idx["test"]])

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
        horizon=horizon,
        device=device,
        args=args,
    )

    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    val_loss, val_y, val_pred = evaluate_loader(model, val_loader, device, criterion)
    test_loss, test_y, test_pred = evaluate_loader(model, test_loader, device, criterion)

    val_pred_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    val_pred_df["pred_score"] = val_pred
    test_pred_df = meta_df.iloc[idx["test"]].copy().reset_index(drop=True)
    test_pred_df["pred_score"] = test_pred

    val_reg = regression_metrics(val_y, val_pred)
    test_reg = regression_metrics(test_y, test_pred)
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
    test_bt = topk_backtest(
        pred_df=test_pred_df,
        horizon=horizon,
        top_k=best_k,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    model_dir = out_root / f"{model_name}_static"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "best_model.pt")
    history_df.to_csv(model_dir / "training_history.csv", index=False)
    grid_df.to_csv(model_dir / "val_topk_grid.csv", index=False)

    pred_all = meta_df.copy()
    pred_all["pred_score"] = np.nan
    pred_all.loc[idx["train"], "pred_score"] = predict_scores(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False), device)
    pred_all.loc[idx["val"], "pred_score"] = val_pred
    pred_all.loc[idx["test"], "pred_score"] = test_pred
    pred_all["date"] = pred_all["date"].dt.strftime("%Y-%m-%d")
    pred_all.to_csv(model_dir / "predictions_all_splits.csv", index=False)
    pred_all[pred_all["split"] == "test"].to_csv(model_dir / "predictions_test.csv", index=False)

    summary = {
        "train_summary": train_summary,
        "val_regression_metrics": val_reg,
        "test_regression_metrics": test_reg,
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

    comparison_row = {
        "model": model_name,
        "mode": "static",
        "best_epoch": train_summary["best_epoch"],
        "val_rmse": val_reg["rmse"],
        "test_rmse": test_reg["rmse"],
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
    return comparison_row


# -----------------------------
# Rolling pipeline
# -----------------------------
def assign_temp_split(df_hist: pd.DataFrame, val_days: int) -> pd.DataFrame:
    out = df_hist.copy().sort_values(["date", "stock"]).reset_index(drop=True)
    unique_dates = sorted(pd.to_datetime(out["date"]).drop_duplicates().tolist())
    if len(unique_dates) <= val_days + 5:
        raise ValueError("Not enough history for rolling train/val split.")
    val_date_set = set(unique_dates[-val_days:])
    out["split_temp"] = np.where(out["date"].isin(val_date_set), "val", "train")
    return out


def build_prediction_sequences_for_date(df_std: pd.DataFrame, feats: List[str], target_date: pd.Timestamp, seq_len: int) -> Tuple[np.ndarray, pd.DataFrame]:
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
            "future_return": float(g.loc[i, [c for c in g.columns if c.startswith("future_return_")][0]]),
        })

    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feats)), dtype=np.float32), pd.DataFrame()
    return np.stack(X_list), pd.DataFrame(meta)


def train_single_rolling_fit(
    df_hist: pd.DataFrame,
    feats: List[str],
    return_col: str,
    horizon: int,
    model_name: str,
    device: torch.device,
    args,
) -> Tuple[nn.Module, Dict[str, float]]:
    df_hist = assign_temp_split(df_hist, val_days=args.rolling_val_days)
    train_raw = df_hist[df_hist["split_temp"] == "train"].copy()
    mean, std = fit_standardizer(train_raw, feats)
    df_std = apply_standardizer(df_hist, feats, mean, std)

    X, y, meta_df = build_sequences(df_std, feats, return_col, seq_len=args.seq_len, split_col="split_temp")
    idx = split_indices(meta_df.rename(columns={"split": "split"}))

    train_ds = SequenceDataset(X[idx["train"]], y[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y[idx["val"]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(model_name, input_dim=len(feats), args=args)
    val_meta_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)

    rolling_args = argparse.Namespace(**vars(args))
    rolling_args.max_epochs = args.rolling_retrain_epochs
    rolling_args.min_epochs = min(args.min_epochs, max(2, args.rolling_retrain_epochs // 2))
    rolling_args.patience = min(args.patience, max(2, args.rolling_retrain_epochs // 2))

    model, _, _ = train_regression_model(
        model_name=f"{model_name}_rolling_fit",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_meta_df=val_meta_df,
        horizon=horizon,
        device=device,
        args=rolling_args,
    )

    # choose top-k on temporary validation
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    _, val_y, val_pred = evaluate_loader(model, val_loader, device, criterion)
    val_pred_df = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    val_pred_df["pred_score"] = val_pred
    best_k, best_k_stats, constraints_ok, _ = choose_best_topk(
        val_pred_df=val_pred_df,
        horizon=horizon,
        topk_grid=args.topk_grid_list,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        min_active_periods=max(3, args.min_active_periods // 2),
    )

    rolling_meta = {
        "best_k": int(best_k),
        "constraints_ok": bool(constraints_ok),
        "val_sharpe": best_k_stats["sharpe"],
        "val_cumret": best_k_stats["cumulative_return"],
        "mean": mean,
        "std": std,
    }
    return model, rolling_meta


def run_rolling_experiment(df: pd.DataFrame, feats: List[str], return_col: str, horizon: int, model_name: str, device: torch.device, args, out_root: Path):
    df = df.copy().sort_values(["date", "stock"]).reset_index(drop=True)
    test_dates = sorted(pd.to_datetime(df[df["split"] == "test"]["date"]).drop_duplicates().tolist())[::horizon]

    pred_rows = []
    retrain_logs = []
    current_model = None
    current_mean = None
    current_std = None
    current_best_k = args.topk_grid_list[0]

    for j, target_date in enumerate(test_dates):
        need_refit = (j == 0) or (j % args.retrain_every_n_rebalances == 0)

        if need_refit:
            hist_df = df[df["date"] < target_date].copy()
            if len(hist_df["date"].drop_duplicates()) < args.seq_len + args.rolling_val_days + 20:
                continue

            current_model, meta = train_single_rolling_fit(
                df_hist=hist_df,
                feats=feats,
                return_col=return_col,
                horizon=horizon,
                model_name=model_name,
                device=device,
                args=args,
            )
            current_mean = meta["mean"]
            current_std = meta["std"]
            current_best_k = meta["best_k"]
            retrain_logs.append({
                "retrain_at_date": str(pd.Timestamp(target_date).date()),
                "selected_top_k": int(current_best_k),
                "temp_val_sharpe": float(meta["val_sharpe"]),
                "temp_val_cumret": float(meta["val_cumret"]),
            })

        if current_model is None:
            continue

        hist_up_to_date = df[df["date"] <= target_date].copy()
        hist_std = apply_standardizer(hist_up_to_date, feats, current_mean, current_std)
        X_pred, meta_pred_df = build_prediction_sequences_for_date(hist_std, feats, target_date, seq_len=args.seq_len)
        if len(X_pred) == 0 or meta_pred_df.empty:
            continue

        ds = SequenceDataset(X_pred, np.zeros(len(X_pred), dtype=np.float32))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        pred_scores = predict_scores(current_model, loader, device)

        meta_pred_df["pred_score"] = pred_scores
        meta_pred_df["split"] = "test"
        meta_pred_df["chosen_top_k_for_date"] = int(current_best_k)
        pred_rows.append(meta_pred_df)

    if len(pred_rows) == 0:
        raise ValueError("Rolling experiment produced no test predictions.")

    pred_test_df = pd.concat(pred_rows, ignore_index=True)
    # Use modal selected k across retrain points for summary, but strategy uses per-date chosen k in logs only if needed.
    modal_k = int(pd.Series([r["selected_top_k"] for r in retrain_logs]).mode().iloc[0]) if retrain_logs else args.topk_grid_list[0]

    # For simplicity of final portfolio metric, use modal_k on predicted scores
    test_bt = topk_backtest(pred_df=pred_test_df, horizon=horizon, top_k=modal_k, transaction_cost_bps=args.transaction_cost_bps)
    test_rank = cross_sectional_rank_metrics(pred_test_df)

    model_dir = out_root / f"{model_name}_rolling"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_out = pred_test_df.copy()
    pred_out["date"] = pred_out["date"].dt.strftime("%Y-%m-%d")
    pred_out.to_csv(model_dir / "rolling_predictions_test.csv", index=False)
    pd.DataFrame(retrain_logs).to_csv(model_dir / "retrain_log.csv", index=False)

    summary = {
        "mode": "rolling",
        "test_rank_metrics": test_rank,
        "rolling_strategy": {
            "modal_selected_top_k": modal_k,
            "test_strategy": test_bt,
        },
    }
    with open(model_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    comparison_row = {
        "model": model_name,
        "mode": "rolling",
        "best_epoch": np.nan,
        "val_rmse": np.nan,
        "test_rmse": np.nan,
        "val_mean_rank_ic": np.nan,
        "test_mean_rank_ic": test_rank["mean_rank_ic"],
        "selected_top_k": modal_k,
        "selected_constraints_satisfied": np.nan,
        "val_strategy_sharpe": np.nan,
        "val_strategy_cumret": np.nan,
        "val_strategy_mdd": np.nan,
        "test_strategy_sharpe": test_bt["sharpe"],
        "test_strategy_cumret": test_bt["cumulative_return"],
        "test_strategy_mdd": test_bt["max_drawdown"],
    }
    return comparison_row


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train H5 return-regression ranking models.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="both", choices=["lstm", "transformer", "both"])
    parser.add_argument("--eval_mode", type=str, default="static", choices=["static", "rolling"])

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--huber_beta", type=float, default=0.01)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)

    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--topk_grid", type=str, default="1,2,3")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)
    parser.add_argument("--min_active_periods", type=int, default=12)

    parser.add_argument("--selection_ic_weight", type=float, default=1.0)
    parser.add_argument("--selection_sharpe_weight", type=float, default=0.5)
    parser.add_argument("--selection_return_weight", type=float, default=2.0)
    parser.add_argument("--selection_mdd_penalty", type=float, default=2.0)

    # rolling options
    parser.add_argument("--rolling_val_days", type=int, default=60)
    parser.add_argument("--rolling_retrain_epochs", type=int, default=8)
    parser.add_argument("--retrain_every_n_rebalances", type=int, default=6)

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
    feats = feature_columns(df, target_col, return_col)

    out_root = Path(args.out_dir) / f"{Path(args.data_path).stem}_return_ranking_{args.eval_mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["target_col"] = target_col
    run_config["return_col"] = return_col
    run_config["horizon"] = horizon
    run_config["feature_columns"] = feats
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Detected target: {target_col}")
    print(f"Detected return: {return_col}")
    print(f"Horizon: {horizon}")
    print(f"Feature count: {len(feats)}")
    print(df["split"].value_counts().to_string())

    models = ["lstm", "transformer"] if args.model == "both" else [args.model]
    comparison_rows = []

    for model_name in models:
        print("\n" + "=" * 100)
        print(f"Running model={model_name} | eval_mode={args.eval_mode}")
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
                return_col=return_col,
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


import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Robust extension ablation: rolling walk-forward evaluation
#
# Why this version:
#   The previous extension experiment used a single 2025+ split.
#   For H5, that produced only ~13 non-overlapping evaluation periods,
#   making strategy results extremely unstable.
#
# What this script does:
#   1) Uses the extension dataset (2025+ with valid fundamentals)
#   2) Compares:
#        - no_fundamentals
#        - with_fundamentals
#   3) Runs multiple expanding walk-forward folds on dates
#   4) Aggregates classification and strategy metrics across folds
#
# Recommended use:
#   Run this as your robustness / ablation experiment.
# =========================================================


# -----------------------------
# Model definitions
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
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


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 24, nhead: int = 4, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout)
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
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 24, num_layers: int = 1, dropout: float = 0.2):
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


# -----------------------------
# Data helpers
# -----------------------------
FUNDAMENTAL_COLS = [
    "net_margin",
    "operating_margin",
    "revenue_growth_qoq",
    "debt_to_equity",
    "asset_turnover",
    "has_fundamental_data",
]


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def get_feature_sets(df: pd.DataFrame, target_col: str, return_col: str) -> Dict[str, List[str]]:
    excluded = {"date", "stock", "split", target_col, return_col}
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    fund_cols = [c for c in FUNDAMENTAL_COLS if c in numeric_cols]
    base_cols = [c for c in numeric_cols if c not in fund_cols]
    full_cols = base_cols + fund_cols

    return {
        "no_fundamentals": base_cols,
        "with_fundamentals": full_cols,
    }


def fit_standardizer(train_df: pd.DataFrame, feats: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train_df[feats].mean()
    std = train_df[feats].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_standardizer(df: pd.DataFrame, feats: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feats] = (out[feats] - mean) / std
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_sequences(df: pd.DataFrame, feats: List[str], target_col: str, return_col: str, seq_len: int):
    X_list, y_list, meta = [], [], []
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].values.astype(np.float32)
        yg = g[target_col].values.astype(np.float32)

        for i in range(seq_len - 1, len(g)):
            X_list.append(Xg[i - seq_len + 1:i + 1])
            y_list.append(yg[i])
            meta.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "future_return": float(g.loc[i, return_col]),
                "target": float(g.loc[i, target_col]),
            })

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = float("nan")
    return out


# -----------------------------
# Strategy layer
# -----------------------------
@dataclass
class StrategyCandidate:
    mode: str
    top_k: int = 0
    min_prob: float = 0.5
    threshold: float = 0.5


def generate_candidates(threshold_grid: List[float], topk_grid: List[int]) -> List[StrategyCandidate]:
    out = []
    for thr in threshold_grid:
        out.append(StrategyCandidate(mode="threshold", threshold=float(thr)))
    for k in topk_grid:
        for mp in threshold_grid:
            out.append(StrategyCandidate(mode="topk", top_k=int(k), min_prob=float(mp)))
    return out


def select_positions(day_df: pd.DataFrame, cand: StrategyCandidate) -> Dict[str, float]:
    day_df = day_df.sort_values("pred_prob", ascending=False).copy()

    if cand.mode == "threshold":
        chosen = day_df[day_df["pred_prob"] >= cand.threshold].copy()
    elif cand.mode == "topk":
        chosen = day_df[day_df["pred_prob"] >= cand.min_prob].head(cand.top_k).copy()
    else:
        raise ValueError(f"Unknown mode: {cand.mode}")

    if len(chosen) == 0:
        return {}

    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def non_overlapping_backtest(pred_df: pd.DataFrame, horizon: int, cand: StrategyCandidate, transaction_cost_bps: float = 10.0):
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
    unique_dates = sorted(df["date"].drop_duplicates().tolist())
    rebalance_dates = unique_dates[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w = {}
    equity = 1.0
    eq_curve = [equity]
    period_returns = []
    turnovers = []
    holdings = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        new_w = select_positions(day, cand)

        gross_ret = 0.0
        if len(new_w) > 0:
            ret_map = day.set_index("stock")["future_return"].to_dict()
            gross_ret = sum(new_w[s] * ret_map.get(s, 0.0) for s in new_w.keys())

        t = turnover(prev_w, new_w)
        net_ret = gross_ret - tc * t

        equity *= (1.0 + net_ret)
        eq_curve.append(equity)
        period_returns.append(net_ret)
        turnovers.append(t)
        holdings.append(len(new_w))
        prev_w = new_w

    period_returns = np.array(period_returns, dtype=float)
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
        }

    eq = np.array(eq_curve)
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())

    cumulative_return = float(eq[-1] - 1.0)
    avg_ret = float(np.mean(period_returns))
    std_ret = float(np.std(period_returns, ddof=1)) if len(period_returns) > 1 else 0.0
    sharpe = float((avg_ret / std_ret) * math.sqrt(252.0 / horizon)) if std_ret > 1e-12 else 0.0

    total_days = max(1, len(period_returns) * horizon)
    annualized_return = float((eq[-1] ** (252.0 / total_days)) - 1.0) if eq[-1] > 0 else -1.0

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float(np.mean(period_returns > 0)),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "avg_holdings": float(np.mean(holdings)) if holdings else 0.0,
    }


def choose_candidate(
    val_pred_df: pd.DataFrame,
    horizon: int,
    candidates: List[StrategyCandidate],
    transaction_cost_bps: float,
    max_val_drawdown: float,
    min_active_periods: int,
):
    rows = []
    for cand in candidates:
        stats = non_overlapping_backtest(
            pred_df=val_pred_df,
            horizon=horizon,
            cand=cand,
            transaction_cost_bps=transaction_cost_bps,
        )
        active_ok = stats["periods"] >= min_active_periods and stats["avg_holdings"] > 0
        mdd_ok = stats["max_drawdown"] >= max_val_drawdown
        ok = bool(active_ok and mdd_ok)

        rows.append({
            "mode": cand.mode,
            "top_k": cand.top_k,
            "min_prob": cand.min_prob,
            "threshold": cand.threshold,
            "constraints_ok": ok,
            **stats,
        })

    grid_df = pd.DataFrame(rows)
    valid_df = grid_df[grid_df["constraints_ok"] == True].copy()

    if len(valid_df) > 0:
        best = valid_df.sort_values(
            ["sharpe", "cumulative_return", "max_drawdown"],
            ascending=[False, False, False]
        ).iloc[0]
        chosen = StrategyCandidate(
            mode=str(best["mode"]),
            top_k=int(best["top_k"]),
            min_prob=float(best["min_prob"]),
            threshold=float(best["threshold"]),
        )
        return chosen, best.to_dict(), True, grid_df

    grid_df["fallback_score"] = (
        grid_df["sharpe"]
        - 4.0 * np.maximum(0.0, max_val_drawdown - grid_df["max_drawdown"])
        - 0.10 * np.maximum(0.0, min_active_periods - grid_df["periods"])
    )
    best = grid_df.sort_values(
        ["fallback_score", "sharpe", "cumulative_return"],
        ascending=[False, False, False]
    ).iloc[0]
    chosen = StrategyCandidate(
        mode=str(best["mode"]),
        top_k=int(best["top_k"]),
        min_prob=float(best["min_prob"]),
        threshold=float(best["threshold"]),
    )
    return chosen, best.to_dict(), False, grid_df


# -----------------------------
# Walk-forward folds
# -----------------------------
def build_walkforward_folds(unique_dates: List[pd.Timestamp], train_dates: int, val_dates: int, test_dates: int, step_dates: int):
    """
    Expanding training window:
      train: [0 : train_end)
      val:   [train_end : val_end)
      test:  [val_end : test_end)
    Then move forward by step_dates.
    """
    folds = []
    n = len(unique_dates)
    start_train = 0

    while True:
        train_end = start_train + train_dates
        val_end = train_end + val_dates
        test_end = val_end + test_dates

        if test_end > n:
            break

        fold = {
            "train_start": unique_dates[start_train],
            "train_end": unique_dates[train_end - 1],
            "val_start": unique_dates[train_end],
            "val_end": unique_dates[val_end - 1],
            "test_start": unique_dates[val_end],
            "test_end": unique_dates[test_end - 1],
        }
        folds.append(fold)

        # expanding train origin
        start_train += step_dates

    return folds


def assign_fold_split(df: pd.DataFrame, fold: Dict) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out["split_fold"] = np.where(
        (out["date"] >= fold["train_start"]) & (out["date"] <= fold["train_end"]),
        "train",
        np.where(
            (out["date"] >= fold["val_start"]) & (out["date"] <= fold["val_end"]),
            "val",
            np.where(
                (out["date"] >= fold["test_start"]) & (out["date"] <= fold["test_end"]),
                "test",
                "ignore"
            )
        )
    )
    out = out[out["split_fold"] != "ignore"].copy()
    return out


# -----------------------------
# Training
# -----------------------------
def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_prob, all_y = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(Xb)

            prob = torch.sigmoid(logits).detach().cpu().numpy()
            all_prob.append(prob)
            all_y.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_prob = np.concatenate(all_prob) if all_prob else np.array([])
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, y_true, y_prob


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    min_epochs: int,
    patience: int,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4
    )
    model = model.to(device)

    history = []
    best_state = None
    best_epoch = -1
    best_val_auc = -np.inf
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(Xb)

        train_loss = running / max(1, len(train_loader.dataset))
        val_loss, val_y, val_prob = evaluate_model(model, val_loader, device, criterion)
        val_metrics = classification_metrics(val_y, val_prob, threshold=0.5)
        val_auc = val_metrics["auc"]
        scheduler.step(val_auc if np.isfinite(val_auc) else -1e9)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "model": model_name,
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        })

        improved = np.isfinite(val_auc) and (val_auc > best_val_auc + 1e-5)
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= min_epochs:
                wait += 1

        if epoch >= min_epochs and wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc) if np.isfinite(best_val_auc) else None,
    }
    return model, pd.DataFrame(history), summary


def make_model(model_name: str, input_dim: int, args):
    if model_name == "transformer":
        return TransformerClassifier(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.transformer_layers,
            dropout=args.dropout,
        )
    elif model_name == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_one_fold(
    df: pd.DataFrame,
    feats: List[str],
    target_col: str,
    return_col: str,
    horizon: int,
    fold: Dict,
    model_name: str,
    args,
    device,
):
    df_fold = assign_fold_split(df, fold)

    train_df = df_fold[df_fold["split_fold"] == "train"].copy()
    val_df = df_fold[df_fold["split_fold"] == "val"].copy()
    test_df = df_fold[df_fold["split_fold"] == "test"].copy()

    mean, std = fit_standardizer(train_df, feats)
    df_fold_std = apply_standardizer(df_fold, feats, mean, std)
    df_fold_std = df_fold_std.rename(columns={"split_fold": "split"})

    X, y, meta_df = build_sequences(df_fold_std, feats, target_col, return_col, seq_len=args.seq_len)

    # map splits from meta dates
    date_to_split = (
        df_fold_std[["date", "stock", "split"]]
        .drop_duplicates()
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    meta_df = meta_df.merge(date_to_split, on=["date", "stock"], how="left", suffixes=("", "_from_df"))

    train_idx = np.where(meta_df["split"] == "train")[0]
    val_idx = np.where(meta_df["split"] == "val")[0]
    test_idx = np.where(meta_df["split"] == "test")[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        return None

    train_ds = SequenceDataset(X[train_idx], y[train_idx])
    val_ds = SequenceDataset(X[val_idx], y[val_idx])
    test_ds = SequenceDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(model_name=model_name, input_dim=len(feats), args=args)
    model, history_df, train_summary = train_single_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
    )

    criterion = nn.BCEWithLogitsLoss()
    _, val_y, val_prob = evaluate_model(model, val_loader, device, criterion)
    _, test_y, test_prob = evaluate_model(model, test_loader, device, criterion)

    val_default = classification_metrics(val_y, val_prob, threshold=0.5)
    test_default = classification_metrics(test_y, test_prob, threshold=0.5)

    pred_df = meta_df.copy()
    pred_df["pred_prob"] = np.nan
    pred_df.loc[val_idx, "pred_prob"] = val_prob
    pred_df.loc[test_idx, "pred_prob"] = test_prob

    threshold_grid = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    topk_grid = [int(x.strip()) for x in args.topk_grid.split(",") if x.strip()]
    candidates = generate_candidates(threshold_grid=threshold_grid, topk_grid=topk_grid)

    val_pred_df = pred_df.iloc[val_idx].copy().reset_index(drop=True)
    test_pred_df = pred_df.iloc[test_idx].copy().reset_index(drop=True)

    chosen_cand, chosen_val_stats, constraints_satisfied, grid_df = choose_candidate(
        val_pred_df=val_pred_df,
        horizon=horizon,
        candidates=candidates,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        min_active_periods=args.min_active_periods,
    )
    test_stats = non_overlapping_backtest(
        pred_df=test_pred_df,
        horizon=horizon,
        cand=chosen_cand,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    return {
        "fold_train_end": str(fold["train_end"].date()),
        "fold_val_end": str(fold["val_end"].date()),
        "fold_test_end": str(fold["test_end"].date()),
        "train_dates": int(train_df["date"].nunique()),
        "val_dates": int(val_df["date"].nunique()),
        "test_dates": int(test_df["date"].nunique()),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "best_epoch": train_summary["best_epoch"],
        "best_val_auc": train_summary["best_val_auc"],
        "val_auc_thr_0_5": val_default["auc"],
        "val_f1_thr_0_5": val_default["f1"],
        "test_auc_thr_0_5": test_default["auc"],
        "test_f1_thr_0_5": test_default["f1"],
        "selected_strategy_mode": chosen_cand.mode,
        "selected_top_k": chosen_cand.top_k,
        "selected_min_prob": chosen_cand.min_prob,
        "selected_threshold": chosen_cand.threshold,
        "selected_constraints_satisfied": constraints_satisfied,
        "val_strategy_sharpe": chosen_val_stats["sharpe"],
        "val_strategy_max_drawdown": chosen_val_stats["max_drawdown"],
        "test_strategy_sharpe": test_stats["sharpe"],
        "test_strategy_cumulative_return": test_stats["cumulative_return"],
        "test_strategy_max_drawdown": test_stats["max_drawdown"],
    }, history_df


def summarize_folds(df_folds: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "best_val_auc",
        "val_auc_thr_0_5",
        "val_f1_thr_0_5",
        "test_auc_thr_0_5",
        "test_f1_thr_0_5",
        "val_strategy_sharpe",
        "val_strategy_max_drawdown",
        "test_strategy_sharpe",
        "test_strategy_cumulative_return",
        "test_strategy_max_drawdown",
        "selected_constraints_satisfied",
    ]

    rows = []
    for keys, g in df_folds.groupby(group_cols):
        row = {}
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                row[k] = v
        else:
            row[group_cols[0]] = keys

        row["n_folds"] = len(g)
        for c in metric_cols:
            if c == "selected_constraints_satisfied":
                row[f"{c}_rate"] = float(g[c].mean())
            else:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Rolling walk-forward fundamentals ablation.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="both", choices=["transformer", "lstm", "both"])

    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--min_epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--d_model", type=int, default=24)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--lstm_hidden", type=int, default=24)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--train_dates", type=int, default=120)
    parser.add_argument("--val_dates", type=int, default=40)
    parser.add_argument("--test_dates", type=int, default=40)
    parser.add_argument("--step_dates", type=int, default=20)

    parser.add_argument("--threshold_grid", type=str, default="0.50,0.52,0.54,0.56,0.58,0.60")
    parser.add_argument("--topk_grid", type=str, default="1,2")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.30)
    parser.add_argument("--min_active_periods", type=int, default=6)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    horizon = infer_horizon(return_col)

    feature_sets = get_feature_sets(df, target_col, return_col)

    print(f"Detected target: {target_col}")
    print(f"Detected return: {return_col}")
    print(f"Horizon: {horizon}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"no_fundamentals feature count: {len(feature_sets['no_fundamentals'])}")
    print(f"with_fundamentals feature count: {len(feature_sets['with_fundamentals'])}")

    unique_dates = sorted(pd.to_datetime(df["date"]).drop_duplicates().tolist())
    folds = build_walkforward_folds(
        unique_dates=unique_dates,
        train_dates=args.train_dates,
        val_dates=args.val_dates,
        test_dates=args.test_dates,
        step_dates=args.step_dates,
    )

    if len(folds) == 0:
        raise ValueError("No folds generated. Reduce train_dates/val_dates/test_dates or use more data.")

    print(f"Generated folds: {len(folds)}")
    for i, f in enumerate(folds, start=1):
        print(
            f"Fold {i}: train[{f['train_start'].date()} -> {f['train_end'].date()}], "
            f"val[{f['val_start'].date()} -> {f['val_end'].date()}], "
            f"test[{f['test_start'].date()} -> {f['test_end'].date()}]"
        )

    out_root = Path(args.out_dir) / f"{Path(args.data_path).stem}_walkforward_ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["target_col"] = target_col
    run_config["return_col"] = return_col
    run_config["horizon"] = horizon
    run_config["folds"] = [
        {k: str(v.date()) for k, v in fold.items()}
        for fold in folds
    ]
    run_config["feature_sets"] = feature_sets
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    if args.model == "both":
        models = ["transformer", "lstm"]
    else:
        models = [args.model]

    all_fold_rows = []
    all_histories = []

    for model_name in models:
        for setting_name, feats in feature_sets.items():
            print("\n" + "=" * 100)
            print(f"Model={model_name} | Setting={setting_name}")
            print("=" * 100)

            for fold_idx, fold in enumerate(folds, start=1):
                print(
                    f"Running fold {fold_idx}/{len(folds)} | "
                    f"train_end={fold['train_end'].date()} | test_end={fold['test_end'].date()}"
                )

                result = run_one_fold(
                    df=df,
                    feats=feats,
                    target_col=target_col,
                    return_col=return_col,
                    horizon=horizon,
                    fold=fold,
                    model_name=model_name,
                    args=args,
                    device=device,
                )

                if result is None:
                    print("Skipped fold due to empty train/val/test sequences.")
                    continue

                row, hist = result
                row["fold_idx"] = fold_idx
                row["model"] = model_name
                row["setting"] = setting_name
                row["feature_count"] = len(feats)
                all_fold_rows.append(row)
                hist["fold_idx"] = fold_idx
                hist["model"] = model_name
                hist["setting"] = setting_name
                all_histories.append(hist)

    folds_df = pd.DataFrame(all_fold_rows)
    folds_df.to_csv(out_root / "fold_level_results.csv", index=False)

    if all_histories:
        pd.concat(all_histories, ignore_index=True).to_csv(out_root / "training_history_all_folds.csv", index=False)

    summary_df = summarize_folds(folds_df, group_cols=["model", "setting"])
    summary_df.to_csv(out_root / "summary_by_model_setting.csv", index=False)

    # incremental effect of fundamentals
    delta_rows = []
    for model_name in models:
        sub = summary_df[summary_df["model"] == model_name].copy()
        if set(sub["setting"]) >= {"no_fundamentals", "with_fundamentals"}:
            base = sub[sub["setting"] == "no_fundamentals"].iloc[0]
            full = sub[sub["setting"] == "with_fundamentals"].iloc[0]
            delta_rows.append({
                "model": model_name,
                "delta_test_auc_mean": full["test_auc_thr_0_5_mean"] - base["test_auc_thr_0_5_mean"],
                "delta_test_f1_mean": full["test_f1_thr_0_5_mean"] - base["test_f1_thr_0_5_mean"],
                "delta_test_strategy_sharpe_mean": full["test_strategy_sharpe_mean"] - base["test_strategy_sharpe_mean"],
                "delta_test_strategy_cumulative_return_mean": full["test_strategy_cumulative_return_mean"] - base["test_strategy_cumulative_return_mean"],
                "delta_test_strategy_max_drawdown_mean": full["test_strategy_max_drawdown_mean"] - base["test_strategy_max_drawdown_mean"],
                "delta_constraints_rate": full["selected_constraints_satisfied_rate"] - base["selected_constraints_satisfied_rate"],
            })

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_root / "fundamental_increment_summary_walkforward.csv", index=False)

    print("\nFinished walk-forward ablation.")
    print(f"Saved outputs to: {out_root}")
    print("\nSummary by model/setting:")
    print(summary_df.to_string(index=False))
    if len(delta_df) > 0:
        print("\nIncrement from adding fundamentals:")
        print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()

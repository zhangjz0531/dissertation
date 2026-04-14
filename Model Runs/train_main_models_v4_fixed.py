
import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# v4: safer main-experiment training script
#
# Key fixes compared with earlier versions:
# 1) Keep ordinary BCE (pos_weight fixed at 1.0 by default)
# 2) Use moderate-capacity models and stronger early stopping
# 3) IMPORTANT: fix strategy evaluation for horizon>1
#    - use NON-OVERLAPPING rebalance every horizon days
#    - avoid row-by-row overlapping use of future_return_5d
# 4) IMPORTANT: use hard validation constraints when selecting strategy
#    - max drawdown filter is truly enforced
#    - if no candidate passes, explicitly report fallback
# 5) Use cross-sectional long-only top-k strategy, which matches a stock panel
#    better than a global absolute threshold-only rule
# =========================================================


# -----------------------------
# Models
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 48, nhead: int = 4, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class LSTMClassifier(nn.Module):
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


# -----------------------------
# Dataset helpers
# -----------------------------
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
    future_cols = [c for c in df.columns if c.startswith("future_return_")]

    if len(target_cols) != 1 or len(future_cols) != 1:
        raise ValueError("Dataset must contain exactly one target_up_* column and one future_return_* column.")

    return target_cols[0], future_cols[0]


def infer_horizon_from_return_col(return_col: str) -> int:
    if return_col.endswith("1d"):
        return 1
    if return_col.endswith("5d"):
        return 5
    raise ValueError(f"Unable to infer horizon from return column: {return_col}")


def feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> List[str]:
    excluded = {"date", "stock", "split", target_col, return_col}
    numeric_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def fit_standardizer(train_df: pd.DataFrame, feats: List[str]) -> Tuple[pd.Series, pd.Series]:
    mean = train_df[feats].mean()
    std = train_df[feats].std().replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_standardizer(df: pd.DataFrame, feats: List[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feats] = (out[feats] - mean) / std
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_sequences(
    df: pd.DataFrame,
    feats: List[str],
    target_col: str,
    return_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Build rolling sequences per stock.
    metadata[i] holds the prediction date row info for the sequence sample.
    """
    X_list, y_list, meta = [], [], []
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        Xg = g[feats].values.astype(np.float32)
        yg = g[target_col].values.astype(np.float32)

        for i in range(seq_len - 1, len(g)):
            start = i - seq_len + 1
            seq_x = Xg[start:i + 1]
            seq_y = yg[i]

            X_list.append(seq_x)
            y_list.append(seq_y)
            meta.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "split": g.loc[i, "split"],
                "future_return": float(g.loc[i, return_col]),
                "target": float(g.loc[i, target_col]),
            })

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta


def split_indices_from_meta(meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = np.array([m["split"] for m in meta])
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    return train_idx, val_idx, test_idx


# -----------------------------
# Metrics
# -----------------------------
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
# Strategy selection and backtest
# -----------------------------
@dataclass
class StrategyCandidate:
    mode: str                    # "topk" or "threshold"
    top_k: int = 0
    min_prob: float = 0.5
    threshold: float = 0.5


def generate_strategy_candidates(threshold_grid: List[float], topk_grid: List[int]) -> List[StrategyCandidate]:
    out = []

    # threshold-only candidates
    for thr in threshold_grid:
        out.append(StrategyCandidate(mode="threshold", threshold=float(thr)))

    # top-k cross-sectional candidates with min prob filter
    for k in topk_grid:
        for mp in threshold_grid:
            out.append(StrategyCandidate(mode="topk", top_k=int(k), min_prob=float(mp)))

    return out


def select_positions_for_date(day_df: pd.DataFrame, cand: StrategyCandidate) -> Dict[str, float]:
    """
    Long-only portfolio on a single rebalance date.
    Returns weights by stock.
    """
    day_df = day_df.sort_values("pred_prob", ascending=False).copy()

    if cand.mode == "threshold":
        chosen = day_df[day_df["pred_prob"] >= cand.threshold].copy()
    elif cand.mode == "topk":
        chosen = day_df[day_df["pred_prob"] >= cand.min_prob].head(cand.top_k).copy()
    else:
        raise ValueError(f"Unknown strategy mode: {cand.mode}")

    if len(chosen) == 0:
        return {}

    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def turnover_between(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def non_overlapping_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    cand: StrategyCandidate,
    transaction_cost_bps: float = 10.0,
) -> Dict[str, float]:
    """
    IMPORTANT FIX:
    For horizon > 1, we rebalance every horizon days on NON-OVERLAPPING dates.
    This avoids overlapping use of future_return_5d and reduces performance inflation.
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
            "avg_holdings": 0.0,
        }

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = sorted(df["date"].drop_duplicates().tolist())

    # non-overlapping rebalance dates
    rebalance_dates = unique_dates[::horizon]

    tc = transaction_cost_bps / 10000.0
    prev_w = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    turnovers = []
    holdings = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        new_w = select_positions_for_date(day, cand)

        gross_ret = 0.0
        if len(new_w) > 0:
            future_map = day.set_index("stock")["future_return"].to_dict()
            gross_ret = sum(new_w[s] * future_map.get(s, 0.0) for s in new_w.keys())

        turnover = turnover_between(prev_w, new_w)
        net_ret = gross_ret - tc * turnover

        equity *= (1.0 + net_ret)
        equity_curve.append(equity)
        period_returns.append(net_ret)
        turnovers.append(turnover)
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

    eq = np.array(equity_curve)
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


def choose_best_candidate(
    val_pred_df: pd.DataFrame,
    horizon: int,
    candidates: List[StrategyCandidate],
    transaction_cost_bps: float,
    max_val_drawdown: float,
    min_active_periods: int,
) -> Tuple[StrategyCandidate, Dict[str, float], bool, pd.DataFrame]:
    """
    Hard constraints are truly enforced here.
    If no candidate passes, we explicitly fall back and return constraints_satisfied=False.
    """
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
        constraints_ok = bool(active_ok and mdd_ok)

        row = {
            "mode": cand.mode,
            "top_k": cand.top_k,
            "min_prob": cand.min_prob,
            "threshold": cand.threshold,
            "constraints_ok": constraints_ok,
            **stats,
        }
        rows.append(row)

    grid_df = pd.DataFrame(rows)

    valid_df = grid_df[grid_df["constraints_ok"] == True].copy()
    if len(valid_df) > 0:
        best = valid_df.sort_values(
            ["sharpe", "cumulative_return", "max_drawdown"],
            ascending=[False, False, False],
        ).iloc[0]
        chosen = StrategyCandidate(
            mode=str(best["mode"]),
            top_k=int(best["top_k"]),
            min_prob=float(best["min_prob"]),
            threshold=float(best["threshold"]),
        )
        return chosen, best.to_dict(), True, grid_df

    # explicit fallback with penalty if nothing passes constraints
    grid_df["fallback_score"] = (
        grid_df["sharpe"]
        - 4.0 * np.maximum(0.0, max_val_drawdown - grid_df["max_drawdown"])   # penalty for drawdown breach
        - 0.10 * np.maximum(0.0, min_active_periods - grid_df["periods"])      # penalty for too few periods
    )

    best = grid_df.sort_values(
        ["fallback_score", "sharpe", "cumulative_return"],
        ascending=[False, False, False],
    ).iloc[0]
    chosen = StrategyCandidate(
        mode=str(best["mode"]),
        top_k=int(best["top_k"]),
        min_prob=float(best["min_prob"]),
        threshold=float(best["threshold"]),
    )
    return chosen, best.to_dict(), False, grid_df


# -----------------------------
# Training
# -----------------------------
def evaluate_model(model, loader, device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
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
) -> Tuple[nn.Module, pd.DataFrame, Dict]:
    criterion = nn.BCEWithLogitsLoss()   # fixed ordinary BCE
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

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "model": model_name,
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        })

        print(
            f"[{model_name}] epoch {epoch:02d} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['auc']}"
        )

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
            print(
                f"[{model_name}] early stopping triggered at epoch {epoch}. "
                f"Best epoch={best_epoch}, best_val_auc={best_val_auc:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc) if np.isfinite(best_val_auc) else None,
    }
    return model, pd.DataFrame(history), summary


# -----------------------------
# Main pipeline
# -----------------------------
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


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="v4 main-model training with safer backtest.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--min_epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)

    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--threshold_grid", type=str, default="0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64")
    parser.add_argument("--topk_grid", type=str, default="1,2,3")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)
    parser.add_argument("--min_active_periods", type=int, default=20)

    args = parser.parse_args()

    out_root = os.path.join(
        args.out_dir,
        f"{Path(args.data_path).stem}_v4_seq{args.seq_len}_lr{args.lr}_wd{args.weight_decay}_do{args.dropout}"
    )
    os.makedirs(out_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    horizon = infer_horizon_from_return_col(return_col)

    print(f"Detected target: {target_col}")
    print(f"Detected return: {return_col}")

    feats = feature_columns(df, target_col, return_col)
    print(f"Feature count: {len(feats)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(df["split"].value_counts().to_string())

    # standardize on train only
    train_df_raw = df[df["split"] == "train"].copy()
    mean, std = fit_standardizer(train_df_raw, feats)
    df_std = apply_standardizer(df, feats, mean, std)

    # build sequences
    X, y, meta = build_sequences(df_std, feats, target_col, return_col, seq_len=args.seq_len)
    train_idx, val_idx, test_idx = split_indices_from_meta(meta)

    print(f"Sequence samples: {len(X)}")
    print(f"Train/Val/Test sequence counts: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"Train positive rate: {y[train_idx].mean():.4f}" if len(train_idx) else "Train positive rate: N/A")
    auto_pw = (1.0 - y[train_idx].mean()) / max(y[train_idx].mean(), 1e-8) if len(train_idx) else 1.0
    print(f"auto pos_weight from data: {auto_pw:.4f}")
    print("using fixed pos_weight for BCE: 1.0000")

    threshold_grid = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    topk_grid = [int(x.strip()) for x in args.topk_grid.split(",") if x.strip()]
    print(f"Threshold grid: {threshold_grid}")
    print(f"Top-k grid: {topk_grid}")
    print(f"Max validation drawdown filter: {args.max_val_drawdown}")

    train_ds = SequenceDataset(X[train_idx], y[train_idx])
    val_ds = SequenceDataset(X[val_idx], y[val_idx])
    test_ds = SequenceDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    all_histories = []
    comparison_rows = []

    run_config = vars(args).copy()
    run_config["feature_columns"] = feats
    run_config["target_col"] = target_col
    run_config["return_col"] = return_col
    run_config["horizon"] = horizon
    save_json(run_config, os.path.join(out_root, "run_config.json"))

    for model_name in ["transformer", "lstm"]:
        print("\n" + "=" * 80)
        print(f"Training {model_name.upper()}")
        print("=" * 80)

        model = make_model(model_name, input_dim=len(feats), args=args)
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
        all_histories.append(history_df)

        # save model
        torch.save(model.state_dict(), os.path.join(out_root, f"{model_name}_best_model.pt"))

        criterion = nn.BCEWithLogitsLoss()
        val_loss, val_y, val_prob = evaluate_model(model, val_loader, device, criterion)
        test_loss, test_y, test_prob = evaluate_model(model, test_loader, device, criterion)

        val_default = classification_metrics(val_y, val_prob, threshold=0.5)
        test_default = classification_metrics(test_y, test_prob, threshold=0.5)

        print(f"[{model_name}] default val metrics (thr=0.5): {val_default}")
        print(f"[{model_name}] default test metrics (thr=0.5): {test_default}")

        # prediction frames
        meta_df = pd.DataFrame(meta)
        pred_df = meta_df.copy()
        pred_df["pred_prob"] = np.nan
        pred_df.loc[val_idx, "pred_prob"] = val_prob
        pred_df.loc[test_idx, "pred_prob"] = test_prob

        # validation candidate search
        val_pred_df = pred_df.iloc[val_idx].copy().reset_index(drop=True)
        test_pred_df = pred_df.iloc[test_idx].copy().reset_index(drop=True)

        candidates = generate_strategy_candidates(threshold_grid=threshold_grid, topk_grid=topk_grid)
        chosen_cand, chosen_val_stats, constraints_satisfied, grid_df = choose_best_candidate(
            val_pred_df=val_pred_df,
            horizon=horizon,
            candidates=candidates,
            transaction_cost_bps=args.transaction_cost_bps,
            max_val_drawdown=args.max_val_drawdown,
            min_active_periods=args.min_active_periods,
        )

        grid_df.to_csv(os.path.join(out_root, f"{model_name}_val_strategy_grid.csv"), index=False)

        test_stats = non_overlapping_backtest(
            pred_df=test_pred_df,
            horizon=horizon,
            cand=chosen_cand,
            transaction_cost_bps=args.transaction_cost_bps,
        )

        print(
            f"[{model_name}] selected strategy: mode={chosen_cand.mode}, "
            f"top_k={chosen_cand.top_k}, min_prob={chosen_cand.min_prob:.2f}, "
            f"threshold={chosen_cand.threshold:.2f}, "
            f"constraints_satisfied={constraints_satisfied}"
        )
        print(f"[{model_name}] val strategy @selected candidate: {chosen_val_stats}")
        print(f"[{model_name}] test strategy @selected candidate: {test_stats}")

        # selected-threshold classification metrics where relevant
        selected_thr = chosen_cand.threshold if chosen_cand.mode == "threshold" else chosen_cand.min_prob
        val_selected_cls = classification_metrics(val_y, val_prob, threshold=selected_thr)
        test_selected_cls = classification_metrics(test_y, test_prob, threshold=selected_thr)

        # save predictions
        pred_out = pred_df.copy()
        pred_out["date"] = pred_out["date"].dt.strftime("%Y-%m-%d")
        pred_out.to_csv(os.path.join(out_root, f"{model_name}_predictions_all_splits.csv"), index=False)
        pred_out[pred_out["split"] == "test"].to_csv(
            os.path.join(out_root, f"{model_name}_predictions_test.csv"), index=False
        )

        metrics_summary = {
            "train_summary": train_summary,
            "val_default_threshold_0_5": val_default,
            "test_default_threshold_0_5": test_default,
            "selected_strategy": {
                "mode": chosen_cand.mode,
                "top_k": chosen_cand.top_k,
                "min_prob": chosen_cand.min_prob,
                "threshold": chosen_cand.threshold,
                "constraints_satisfied": constraints_satisfied,
                "val_stats": chosen_val_stats,
                "test_stats": test_stats,
            },
            "val_selected_threshold_classification": val_selected_cls,
            "test_selected_threshold_classification": test_selected_cls,
        }
        save_json(metrics_summary, os.path.join(out_root, f"{model_name}_metrics_summary.json"))

        comparison_rows.append({
            "model": model_name,
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
        })

    history_all = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    history_all.to_csv(os.path.join(out_root, "training_history.csv"), index=False)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(out_root, "model_comparison.csv"), index=False)

    print("\nFinished.")
    print(f"Saved outputs to: {out_root}")
    if len(comp_df) > 0:
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()

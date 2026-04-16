from __future__ import annotations

import json
import math
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import (
    MAIN_H5_DATA,
    EXPERIMENTS_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)


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


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    vals = pd.Series(y_score)
    avg_ranks = vals.rank(method="average").values
    sum_pos = avg_ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    auc = roc_auc_binary(y_true, prob)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
    }


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


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        norm_first: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
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
    target_col: str,
    return_col: str,
    seq_len: int,
    split_col: str = "split",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def select_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day_df = day_df.sort_values("pred_prob", ascending=False).copy()
    if mode == "topk":
        chosen = day_df[day_df["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day_df[day_df["pred_prob"] >= threshold].copy()
    else:
        raise ValueError(f"Unknown strategy mode: {mode}")
    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {row["stock"]: w for _, row in chosen.iterrows()}


def run_strategy_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float,
) -> Dict[str, float]:
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
    period_returns, turns, holds = [], [], []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
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


def choose_strategy_on_validation(
    val_pred_df: pd.DataFrame,
    horizon: int,
    mode_grid: List[str],
    topk_grid: List[int],
    min_prob_grid: List[float],
    threshold_grid: List[float],
    transaction_cost_bps: float,
    max_val_drawdown: float,
    max_val_turnover: float,
    score_weights: Dict[str, float],
    auc_for_epoch: float,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    rows = []
    for mode in mode_grid:
        for top_k in topk_grid:
            for min_prob in min_prob_grid:
                for threshold in threshold_grid:
                    stats = run_strategy_backtest(
                        pred_df=val_pred_df,
                        horizon=horizon,
                        mode=mode,
                        top_k=top_k,
                        min_prob=min_prob,
                        threshold=threshold,
                        transaction_cost_bps=transaction_cost_bps,
                    )
                    dd_ok = bool(stats["max_drawdown"] >= max_val_drawdown)
                    to_ok = bool(stats["avg_turnover"] <= max_val_turnover)
                    constraints_ok = bool(dd_ok and to_ok)

                    auc_edge = max(0.0, float(auc_for_epoch) - 0.5) if np.isfinite(auc_for_epoch) else 0.0
                    score = (
                        score_weights["auc"] * auc_edge
                        + score_weights["sharpe"] * stats["sharpe"]
                        + score_weights["cumret"] * stats["cumulative_return"]
                        - score_weights["mdd_penalty"] * max(0.0, max_val_drawdown - stats["max_drawdown"])
                        - score_weights["turnover_penalty"] * stats["avg_turnover"]
                    )

                    rows.append({
                        "mode": mode,
                        "top_k": int(top_k),
                        "min_prob": float(min_prob),
                        "threshold": float(threshold),
                        "drawdown_ok": dd_ok,
                        "turnover_ok": to_ok,
                        "constraints_ok": constraints_ok,
                        "strategy_score": score,
                        **stats,
                    })

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df["constraints_ok"] == True].copy()

    if len(valid) > 0:
        best = valid.sort_values(
            ["strategy_score", "sharpe", "cumulative_return"],
            ascending=[False, False, False]
        ).iloc[0]
    else:
        best = grid_df.sort_values(
            ["strategy_score", "sharpe", "cumulative_return"],
            ascending=[False, False, False]
        ).iloc[0]

    best_cfg = {
        "mode": str(best["mode"]),
        "top_k": int(best["top_k"]),
        "min_prob": float(best["min_prob"]),
        "threshold": float(best["threshold"]),
        "constraints_satisfied": bool(best["constraints_ok"]),
        "drawdown_ok": bool(best["drawdown_ok"]),
        "turnover_ok": bool(best["turnover_ok"]),
    }
    return best_cfg, best.to_dict(), grid_df


def pick_epoch_candidate(best_cfg_epoch: Dict, auc_gate_pass: bool, require_constraints: bool) -> bool:
    if not auc_gate_pass:
        return False
    if require_constraints and not bool(best_cfg_epoch["constraints_satisfied"]):
        return False
    return True


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, ys, logits_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item() * len(xb))
            ys.append(yb.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    logits = np.concatenate(logits_list) if logits_list else np.array([])
    avg_loss = float(sum(losses) / max(1, len(loader.dataset)))
    return avg_loss, y_true, logits


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def save_scaler_stats(save_dir: Path, mean: pd.Series, std: pd.Series, feats: List[str]) -> None:
    np.savez(
        save_dir / "scaler_stats.npz",
        feature_columns=np.array(feats, dtype=object),
        mean=mean.values.astype(np.float32),
        std=std.values.astype(np.float32),
    )


def train_transformer_main(data_path: str, out_dir: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {data_path}")

    raw_df = pd.read_csv(data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    target_col, return_col = infer_columns(raw_df)
    horizon = infer_horizon(return_col)
    feats = feature_columns(raw_df, target_col, return_col)

    print(f"Detected target: {target_col}")
    print(f"Detected return: {return_col}")
    print(f"Feature count: {len(feats)}")
    print(f"Date range: {raw_df['date'].min().date()} -> {raw_df['date'].max().date()}")
    print(raw_df["split"].value_counts().to_string())

    train_raw = raw_df[raw_df["split"] == "train"].copy()
    mean, std = fit_standardizer(train_raw, feats)
    df = apply_standardizer(raw_df, feats, mean, std)

    X, y, meta_df = build_sequences(df, feats, target_col, return_col, seq_len=args.seq_len, split_col="split")
    idx = split_indices(meta_df)

    print(f"Sequence samples: {len(X)}")
    print(f"Train/Val/Test sequence counts: {len(idx['train'])} / {len(idx['val'])} / {len(idx['test'])}")

    y_train = y[idx["train"]]
    train_positive_rate = float(y_train.mean()) if len(y_train) > 0 else np.nan
    auto_pos_weight = float((len(y_train) - y_train.sum()) / max(1.0, y_train.sum())) if len(y_train) > 0 else 1.0
    effective_pos_weight = auto_pos_weight if args.pos_weight <= 0 else float(args.pos_weight)

    print(f"Train positive rate: {train_positive_rate:.4f}")
    print(f"auto pos_weight from data: {auto_pos_weight:.4f}")
    print(f"using effective pos_weight for BCE: {effective_pos_weight:.4f}")
    print(f"Threshold grid: {args.threshold_grid_list}")
    print(f"Top-k grid: {args.topk_grid_list}")
    print(f"Mode grid: {args.mode_grid_list}")
    print(f"Min-prob grid: {args.min_prob_grid_list}")
    print(f"Max validation drawdown filter: {args.max_val_drawdown}")
    print(f"Max validation turnover filter: {args.max_val_turnover}")
    print(f"d_model={args.d_model}, dropout={args.dropout}, lr={args.lr}, wd={args.weight_decay}, norm_first={bool(args.norm_first)}")
    print(
        f"AUC gate: enabled={bool(args.use_auc_gate_for_strategy_selection)}, "
        f"abs_floor={args.auc_gate_abs_floor}, slack={args.auc_gate_slack}, "
        f"require_constraints={bool(args.require_strategy_constraints_for_checkpoint)}"
    )

    train_ds = SequenceDataset(X[idx["train"]], y[idx["train"]])
    val_ds = SequenceDataset(X[idx["val"]], y[idx["val"]])
    test_ds = SequenceDataset(X[idx["test"]], y[idx["test"]])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TransformerClassifier(
        input_dim=len(feats),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.transformer_layers,
        dropout=args.dropout,
        norm_first=bool(args.norm_first),
    ).to(device)

    pos_weight_tensor = torch.tensor([effective_pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, threshold=1e-4
    )

    val_meta = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    test_meta = meta_df.iloc[idx["test"]].copy().reset_index(drop=True)

    best_auc_epoch, best_auc, best_auc_state = -1, -1e18, None
    best_strategy_epoch, best_strategy_score, best_strategy_state = -1, -1e18, None
    best_strategy_cfg, best_strategy_val_stats = None, None

    best_relaxed_epoch, best_relaxed_score, best_relaxed_state = -1, -1e18, None
    best_relaxed_cfg, best_relaxed_val_stats = None, None

    best_raw_strategy_epoch, best_raw_strategy_score = -1, -1e18
    best_raw_strategy_cfg, best_raw_strategy_val_stats = None, None

    history = []
    wait = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item() * len(xb))

        train_loss = float(sum(train_losses) / max(1, len(train_loader.dataset)))
        val_loss, val_y_true, val_logits = evaluate_model(model, val_loader, device, criterion)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics_default = classification_metrics(val_y_true, val_prob, threshold=0.5)

        val_pred_df = val_meta.copy()
        val_pred_df["pred_prob"] = val_prob

        best_cfg_epoch, best_stats_epoch, _ = choose_strategy_on_validation(
            val_pred_df=val_pred_df,
            horizon=horizon,
            mode_grid=args.mode_grid_list,
            topk_grid=args.topk_grid_list,
            min_prob_grid=args.min_prob_grid_list,
            threshold_grid=args.threshold_grid_list,
            transaction_cost_bps=args.transaction_cost_bps,
            max_val_drawdown=args.max_val_drawdown,
            max_val_turnover=args.max_val_turnover,
            score_weights={
                "auc": args.auc_score_weight,
                "sharpe": args.strategy_score_sharpe_weight,
                "cumret": args.strategy_score_cumret_weight,
                "mdd_penalty": args.strategy_score_mdd_penalty,
                "turnover_penalty": args.strategy_score_turnover_penalty,
            },
            auc_for_epoch=val_metrics_default["auc"],
        )

        strategy_score_epoch = float(best_stats_epoch["strategy_score"])
        lr_now = optimizer.param_groups[0]["lr"]

        if np.isfinite(val_metrics_default["auc"]) and val_metrics_default["auc"] > best_auc + 1e-6:
            best_auc = float(val_metrics_default["auc"])
            best_auc_epoch = epoch
            best_auc_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if strategy_score_epoch > best_raw_strategy_score + 1e-6:
            best_raw_strategy_score = strategy_score_epoch
            best_raw_strategy_epoch = epoch
            best_raw_strategy_cfg = dict(best_cfg_epoch)
            best_raw_strategy_val_stats = dict(best_stats_epoch)

        auc_gate_floor = max(float(args.auc_gate_abs_floor), float(best_auc) - float(args.auc_gate_slack))
        auc_gate_pass = True
        if bool(args.use_auc_gate_for_strategy_selection):
            auc_gate_pass = bool(np.isfinite(val_metrics_default["auc"]) and val_metrics_default["auc"] >= auc_gate_floor)

        constraints_pass = bool(best_cfg_epoch["constraints_satisfied"])
        strict_eligible = pick_epoch_candidate(best_cfg_epoch, auc_gate_pass, bool(args.require_strategy_constraints_for_checkpoint))
        relaxed_eligible = auc_gate_pass

        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics_default["f1"],
            "val_auc": val_metrics_default["auc"],
            "auc_gate_floor": auc_gate_floor,
            "auc_gate_pass": auc_gate_pass,
            "constraints_pass": constraints_pass,
            "strict_eligible": strict_eligible,
            "relaxed_eligible": relaxed_eligible,
            "val_strategy_mode": best_cfg_epoch["mode"],
            "val_strategy_top_k": best_cfg_epoch["top_k"],
            "val_strategy_min_prob": best_cfg_epoch["min_prob"],
            "val_strategy_threshold": best_cfg_epoch["threshold"],
            "val_strategy_constraints_satisfied": best_cfg_epoch["constraints_satisfied"],
            "val_strategy_drawdown_ok": best_cfg_epoch["drawdown_ok"],
            "val_strategy_turnover_ok": best_cfg_epoch["turnover_ok"],
            "val_strategy_sharpe": best_stats_epoch["sharpe"],
            "val_strategy_cumret": best_stats_epoch["cumulative_return"],
            "val_strategy_mdd": best_stats_epoch["max_drawdown"],
            "val_strategy_turnover": best_stats_epoch["avg_turnover"],
            "val_strategy_score": strategy_score_epoch,
        })

        print(
            f"[transformer_main] epoch {epoch:02d} | lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_metrics_default['f1']:.4f} | val_auc={val_metrics_default['auc']:.6f} | "
            f"auc_gate_floor={auc_gate_floor:.6f} | auc_gate_pass={auc_gate_pass} | "
            f"constraints_pass={constraints_pass} | strict_eligible={strict_eligible}"
        )
        print(
            f"                 best_val_strategy: mode={best_cfg_epoch['mode']}, "
            f"top_k={best_cfg_epoch['top_k']}, min_prob={best_cfg_epoch['min_prob']:.2f}, "
            f"thr={best_cfg_epoch['threshold']:.2f}, constraints={best_cfg_epoch['constraints_satisfied']}, "
            f"dd_ok={best_cfg_epoch['drawdown_ok']}, to_ok={best_cfg_epoch['turnover_ok']}, "
            f"sharpe={best_stats_epoch['sharpe']:.4f}, cumret={best_stats_epoch['cumulative_return']:.4f}, "
            f"mdd={best_stats_epoch['max_drawdown']:.4f}, turnover={best_stats_epoch['avg_turnover']:.4f}, "
            f"score={strategy_score_epoch:.4f}"
        )

        if strict_eligible and strategy_score_epoch > best_strategy_score + 1e-6:
            best_strategy_score = strategy_score_epoch
            best_strategy_epoch = epoch
            best_strategy_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_strategy_cfg = dict(best_cfg_epoch)
            best_strategy_val_stats = dict(best_stats_epoch)
            wait = 0
        else:
            if epoch >= args.min_epochs:
                wait += 1

        if relaxed_eligible and strategy_score_epoch > best_relaxed_score + 1e-6:
            best_relaxed_score = strategy_score_epoch
            best_relaxed_epoch = epoch
            best_relaxed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_relaxed_cfg = dict(best_cfg_epoch)
            best_relaxed_val_stats = dict(best_stats_epoch)

        scheduler.step(strategy_score_epoch)

        if epoch >= args.min_epochs and wait >= args.patience:
            print(
                f"[transformer_main] early stopping triggered at epoch {epoch}. "
                f"Best strict strategy epoch={best_strategy_epoch}, best_strategy_score={best_strategy_score:.6f}, "
                f"best_val_auc_epoch={best_auc_epoch}, best_val_auc={best_auc:.6f}"
            )
            break

    if best_auc_state is None:
        raise RuntimeError("best_auc_state is None. Training did not produce a valid AUC checkpoint.")

    base_out_dir = ensure_dir(Path(out_dir))
    run_name = args.run_name.strip() if args.run_name else ""
    if run_name == "":
        run_name = f"run_{timestamp_tag()}"
    save_dir = base_out_dir / run_name
    if save_dir.exists():
        save_dir = base_out_dir / f"{run_name}_{timestamp_tag()}"
    save_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "training_history.csv", index=False)
    np.savez(
        save_dir / "scaler_stats.npz",
        feature_columns=np.array(feats, dtype=object),
        mean=mean.values.astype(np.float32),
        std=std.values.astype(np.float32),
    )

    model.load_state_dict(best_auc_state)
    auc_val_prob = predict_probs(model, val_loader, device)
    auc_test_prob = predict_probs(model, test_loader, device)

    auc_val_metrics = classification_metrics(y[idx["val"]], auc_val_prob, threshold=0.5)
    auc_test_metrics = classification_metrics(y[idx["test"]], auc_test_prob, threshold=0.5)

    auc_val_pred_df = val_meta.copy()
    auc_val_pred_df["pred_prob"] = auc_val_prob
    auc_test_pred_df = test_meta.copy()
    auc_test_pred_df["pred_prob"] = auc_test_prob

    auc_best_cfg, auc_best_stats, auc_grid_df = choose_strategy_on_validation(
        val_pred_df=auc_val_pred_df,
        horizon=horizon,
        mode_grid=args.mode_grid_list,
        topk_grid=args.topk_grid_list,
        min_prob_grid=args.min_prob_grid_list,
        threshold_grid=args.threshold_grid_list,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        max_val_turnover=args.max_val_turnover,
        score_weights={
            "auc": args.auc_score_weight,
            "sharpe": args.strategy_score_sharpe_weight,
            "cumret": args.strategy_score_cumret_weight,
            "mdd_penalty": args.strategy_score_mdd_penalty,
            "turnover_penalty": args.strategy_score_turnover_penalty,
        },
        auc_for_epoch=auc_val_metrics["auc"],
    )
    auc_test_stats = run_strategy_backtest(
        pred_df=auc_test_pred_df,
        horizon=horizon,
        mode=auc_best_cfg["mode"],
        top_k=auc_best_cfg["top_k"],
        min_prob=auc_best_cfg["min_prob"],
        threshold=auc_best_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    strategy_source = "strict_eligible_checkpoint"
    if best_strategy_state is None or best_strategy_cfg is None or best_strategy_val_stats is None:
        if best_relaxed_state is not None and best_relaxed_cfg is not None and best_relaxed_val_stats is not None:
            print("[transformer_main] No strict eligible checkpoint found. Falling back to best relaxed AUC-gated checkpoint.")
            best_strategy_epoch = best_relaxed_epoch
            best_strategy_score = best_relaxed_score
            best_strategy_state = best_relaxed_state
            best_strategy_cfg = best_relaxed_cfg
            best_strategy_val_stats = best_relaxed_val_stats
            strategy_source = "relaxed_auc_gated_checkpoint"
        else:
            print("[transformer_main] No relaxed AUC-gated checkpoint found. Falling back to best AUC checkpoint.")
            best_strategy_epoch = best_auc_epoch
            best_strategy_score = float("-inf")
            best_strategy_state = {k: v.clone() for k, v in best_auc_state.items()}
            best_strategy_cfg = dict(auc_best_cfg)
            best_strategy_val_stats = dict(auc_best_stats)
            strategy_source = "best_auc_checkpoint"

    model.load_state_dict(best_strategy_state)
    strat_val_prob = predict_probs(model, val_loader, device)
    strat_test_prob = predict_probs(model, test_loader, device)
    strat_train_prob = predict_probs(
        model,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False),
        device
    )

    strat_val_metrics = classification_metrics(y[idx["val"]], strat_val_prob, threshold=0.5)
    strat_test_metrics = classification_metrics(y[idx["test"]], strat_test_prob, threshold=0.5)

    strat_val_pred_df = val_meta.copy()
    strat_val_pred_df["pred_prob"] = strat_val_prob
    strat_test_pred_df = test_meta.copy()
    strat_test_pred_df["pred_prob"] = strat_test_prob

    selected_test_stats = run_strategy_backtest(
        pred_df=strat_test_pred_df,
        horizon=horizon,
        mode=best_strategy_cfg["mode"],
        top_k=best_strategy_cfg["top_k"],
        min_prob=best_strategy_cfg["min_prob"],
        threshold=best_strategy_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    _, _, final_grid_df = choose_strategy_on_validation(
        val_pred_df=strat_val_pred_df,
        horizon=horizon,
        mode_grid=args.mode_grid_list,
        topk_grid=args.topk_grid_list,
        min_prob_grid=args.min_prob_grid_list,
        threshold_grid=args.threshold_grid_list,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
        max_val_turnover=args.max_val_turnover,
        score_weights={
            "auc": args.auc_score_weight,
            "sharpe": args.strategy_score_sharpe_weight,
            "cumret": args.strategy_score_cumret_weight,
            "mdd_penalty": args.strategy_score_mdd_penalty,
            "turnover_penalty": args.strategy_score_turnover_penalty,
        },
        auc_for_epoch=strat_val_metrics["auc"],
    )

    final_grid_df.to_csv(save_dir / "val_strategy_grid_selected_checkpoint.csv", index=False)
    auc_grid_df.to_csv(save_dir / "val_strategy_grid_auc_checkpoint.csv", index=False)

    torch.save(best_auc_state, save_dir / "transformer_best_auc_model.pt")
    torch.save(best_strategy_state, save_dir / "transformer_best_strategy_model.pt")

    pred_all = meta_df.copy()
    pred_all["pred_prob"] = np.nan
    pred_all.loc[idx["train"], "pred_prob"] = strat_train_prob
    pred_all.loc[idx["val"], "pred_prob"] = strat_val_prob
    pred_all.loc[idx["test"], "pred_prob"] = strat_test_prob

    pred_all_out = pred_all.copy()
    pred_all_out["date"] = pred_all_out["date"].dt.strftime("%Y-%m-%d")
    pred_all_out.to_csv(save_dir / "transformer_predictions_all_splits.csv", index=False)
    pred_all_out[pred_all_out["split"] == "test"].to_csv(save_dir / "transformer_predictions_test.csv", index=False)

    run_config = vars(args).copy()
    run_config["effective_pos_weight"] = effective_pos_weight
    run_config["target_col"] = target_col
    run_config["return_col"] = return_col
    run_config["feature_columns"] = feats
    run_config["feature_count"] = len(feats)
    run_config["horizon"] = horizon
    run_config["data_path"] = str(Path(data_path).resolve())
    run_config["save_dir"] = str(save_dir.resolve())
    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    summary = {
        "model": "transformer_main",
        "strategy_checkpoint_source": strategy_source,
        "best_auc_epoch": int(best_auc_epoch),
        "best_val_auc": float(best_auc),
        "best_strategy_epoch": int(best_strategy_epoch),
        "best_strategy_score": float(best_strategy_score) if np.isfinite(best_strategy_score) else None,
        "best_relaxed_epoch": int(best_relaxed_epoch),
        "best_relaxed_score": float(best_relaxed_score) if np.isfinite(best_relaxed_score) else None,
        "best_raw_strategy_epoch_without_auc_gate": int(best_raw_strategy_epoch),
        "best_raw_strategy_score_without_auc_gate": float(best_raw_strategy_score) if np.isfinite(best_raw_strategy_score) else None,
        "feature_count": int(len(feats)),
        "auc_gate": {
            "enabled": bool(args.use_auc_gate_for_strategy_selection),
            "abs_floor": float(args.auc_gate_abs_floor),
            "slack": float(args.auc_gate_slack),
            "require_constraints_for_checkpoint": bool(args.require_strategy_constraints_for_checkpoint),
        },
        "best_auc_checkpoint": {
            "val_metrics_thr_0_5": auc_val_metrics,
            "test_metrics_thr_0_5": auc_test_metrics,
            "selected_strategy": {
                **auc_best_cfg,
                "val_strategy": auc_best_stats,
                "test_strategy": auc_test_stats,
            },
        },
        "best_strategy_checkpoint": {
            "val_metrics_thr_0_5": strat_val_metrics,
            "test_metrics_thr_0_5": strat_test_metrics,
            "selected_strategy": {
                **best_strategy_cfg,
                "val_strategy": best_strategy_val_stats,
                "test_strategy": selected_test_stats,
            },
        },
    }
    with open(save_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    comparison = pd.DataFrame([{
        "model": "transformer_main",
        "strategy_checkpoint_source": strategy_source,
        "best_auc_epoch": best_auc_epoch,
        "best_val_auc": best_auc,
        "best_strategy_epoch": best_strategy_epoch,
        "best_strategy_score": best_strategy_score if np.isfinite(best_strategy_score) else np.nan,
        "best_relaxed_epoch": best_relaxed_epoch,
        "best_raw_strategy_epoch_without_auc_gate": best_raw_strategy_epoch,
        "val_auc_thr_0_5": strat_val_metrics["auc"],
        "val_f1_thr_0_5": strat_val_metrics["f1"],
        "test_auc_thr_0_5": strat_test_metrics["auc"],
        "test_f1_thr_0_5": strat_test_metrics["f1"],
        "selected_strategy_mode": best_strategy_cfg["mode"],
        "selected_top_k": best_strategy_cfg["top_k"],
        "selected_min_prob": best_strategy_cfg["min_prob"],
        "selected_threshold": best_strategy_cfg["threshold"],
        "selected_constraints_satisfied": best_strategy_cfg["constraints_satisfied"],
        "selected_drawdown_ok": best_strategy_cfg["drawdown_ok"],
        "selected_turnover_ok": best_strategy_cfg["turnover_ok"],
        "val_strategy_sharpe": best_strategy_val_stats["sharpe"],
        "val_strategy_cumulative_return": best_strategy_val_stats["cumulative_return"],
        "val_strategy_max_drawdown": best_strategy_val_stats["max_drawdown"],
        "val_strategy_avg_turnover": best_strategy_val_stats["avg_turnover"],
        "test_strategy_sharpe": selected_test_stats["sharpe"],
        "test_strategy_cumulative_return": selected_test_stats["cumulative_return"],
        "test_strategy_max_drawdown": selected_test_stats["max_drawdown"],
        "test_strategy_avg_turnover": selected_test_stats["avg_turnover"],
    }])
    comparison.to_csv(save_dir / "model_comparison.csv", index=False)

    print("\\nFinished.")
    print(f"Saved outputs to: {save_dir}")
    print(comparison.to_string(index=False))


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(description="Train main Transformer system for the dissertation.")
    parser.add_argument("--data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--out_dir", type=str, default=str(EXPERIMENTS_DIR / "main_transformer_h5"))
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)

    # Mainline training configuration
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--norm_first", type=int, default=0)

    parser.add_argument("--pos_weight", type=float, default=0.75)

    # Mainline strategy search space
    parser.add_argument("--mode_grid", type=str, default="topk")
    parser.add_argument("--topk_grid", type=str, default="3,4")
    parser.add_argument("--min_prob_grid", type=str, default="0.56,0.58")
    parser.add_argument("--threshold_grid", type=str, default="0.50")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.20)
    parser.add_argument("--max_val_turnover", type=float, default=0.90)

    parser.add_argument("--auc_score_weight", type=float, default=1.0)
    parser.add_argument("--strategy_score_sharpe_weight", type=float, default=2.0)
    parser.add_argument("--strategy_score_cumret_weight", type=float, default=2.0)
    parser.add_argument("--strategy_score_mdd_penalty", type=float, default=3.0)
    parser.add_argument("--strategy_score_turnover_penalty", type=float, default=0.45)

    parser.add_argument("--use_auc_gate_for_strategy_selection", type=int, default=1)
    parser.add_argument("--auc_gate_abs_floor", type=float, default=0.500)
    parser.add_argument("--auc_gate_slack", type=float, default=0.02)
    parser.add_argument("--require_strategy_constraints_for_checkpoint", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by nhead ({args.nhead}).")

    set_seed(args.seed)
    args.mode_grid_list = [x.strip() for x in args.mode_grid.split(",") if x.strip()]
    args.topk_grid_list = [int(x.strip()) for x in args.topk_grid.split(",") if x.strip()]
    args.min_prob_grid_list = [float(x.strip()) for x in args.min_prob_grid.split(",") if x.strip()]
    args.threshold_grid_list = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]

    train_transformer_main(data_path=args.data_path, out_dir=args.out_dir, args=args)


if __name__ == "__main__":
    main()

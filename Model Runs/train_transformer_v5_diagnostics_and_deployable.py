
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
# Final Transformer v5 walk-forward + deployment diagnostics
#
# This script extends the earlier walk-forward evaluator to address:
# 1) tail coverage into the latest sample with an optional partial last fold
# 2) fixed-config representativeness via validation-aggregated robust config selection
# 3) negative test-fold / weak-validation diagnostics
# 4) turnover / transaction-cost sensitivity
#
# It does NOT change preprocessing or the core model architecture.
# It strengthens the evaluation layer around the final Transformer system.
# =========================================================


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
    split_col: str = "wf_split",
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
                "wf_split": g.loc[i, split_col],
                "future_return": float(g.loc[i, return_col]),
            })

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def split_indices(meta_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "train": np.where(meta_df["wf_split"].values == "train")[0],
        "val": np.where(meta_df["wf_split"].values == "val")[0],
        "test": np.where(meta_df["wf_split"].values == "test")[0],
    }


def make_walkforward_folds(
    unique_dates: List[pd.Timestamp],
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    walk_mode: str = "expanding",
    max_folds: int = 0,
    allow_partial_last_test: bool = True,
    min_partial_test_days: int = 60,
) -> List[Dict]:
    n = len(unique_dates)
    folds = []
    fold_id = 1

    if walk_mode not in {"expanding", "rolling"}:
        raise ValueError("walk_mode must be one of {'expanding', 'rolling'}")

    train_start = 0
    train_end = train_days

    while True:
        val_start = train_end
        val_end = val_start + val_days
        test_start = val_end
        test_end = test_start + test_days

        if test_end > n:
            break

        folds.append({
            "fold_id": fold_id,
            "is_partial_tail": False,
            "train_start_date": unique_dates[train_start],
            "train_end_date": unique_dates[train_end - 1],
            "val_start_date": unique_dates[val_start],
            "val_end_date": unique_dates[val_end - 1],
            "test_start_date": unique_dates[test_start],
            "test_end_date": unique_dates[test_end - 1],
        })

        fold_id += 1
        if max_folds > 0 and len(folds) >= max_folds:
            return folds

        if walk_mode == "expanding":
            train_end += step_days
        else:
            train_start += step_days
            train_end += step_days

    if allow_partial_last_test and (max_folds == 0 or len(folds) < max_folds):
        val_start = train_end
        val_end = val_start + val_days
        test_start = val_end
        test_end = n

        remaining_test_days = n - test_start
        if val_end < n and remaining_test_days >= min_partial_test_days:
            folds.append({
                "fold_id": fold_id,
                "is_partial_tail": True,
                "train_start_date": unique_dates[train_start],
                "train_end_date": unique_dates[train_end - 1],
                "val_start_date": unique_dates[val_start],
                "val_end_date": unique_dates[val_end - 1],
                "test_start_date": unique_dates[test_start],
                "test_end_date": unique_dates[test_end - 1],
            })

    return folds


def assign_fold_splits(df: pd.DataFrame, fold: Dict) -> pd.DataFrame:
    out = df.copy()
    out["wf_split"] = "unused"

    train_mask = (out["date"] >= fold["train_start_date"]) & (out["date"] <= fold["train_end_date"])
    val_mask = (out["date"] >= fold["val_start_date"]) & (out["date"] <= fold["val_end_date"])
    test_mask = (out["date"] >= fold["test_start_date"]) & (out["date"] <= fold["test_end_date"])

    out.loc[train_mask, "wf_split"] = "train"
    out.loc[val_mask, "wf_split"] = "val"
    out.loc[test_mask, "wf_split"] = "test"

    return out[out["wf_split"].isin(["train", "val", "test"])].copy()


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
    period_returns = []
    turns = []
    holds = []

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


def expand_candidate_grid(
    mode_grid: List[str],
    topk_grid: List[int],
    min_prob_grid: List[float],
    threshold_grid: List[float],
) -> List[Dict]:
    grid = []
    for mode in mode_grid:
        for top_k in topk_grid:
            for min_prob in min_prob_grid:
                for threshold in threshold_grid:
                    grid.append({
                        "mode": mode,
                        "top_k": int(top_k),
                        "min_prob": float(min_prob),
                        "threshold": float(threshold),
                    })
    return grid


def choose_strategy_on_validation(
    val_pred_df: pd.DataFrame,
    horizon: int,
    mode_grid: List[str],
    topk_grid: List[int],
    min_prob_grid: List[float],
    threshold_grid: List[float],
    transaction_cost_bps: float,
    max_val_drawdown: float,
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

                    constraints_ok = bool(stats["max_drawdown"] >= max_val_drawdown)
                    score = (
                        score_weights["auc"] * (0.0 if not np.isfinite(auc_for_epoch) else auc_for_epoch)
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
                        "constraints_ok": constraints_ok,
                        "strategy_score": score,
                        **stats,
                    })

    grid_df = pd.DataFrame(rows)
    valid = grid_df[grid_df["constraints_ok"] == True].copy()
    if len(valid) > 0:
        best = valid.sort_values(["strategy_score", "sharpe", "cumulative_return"], ascending=[False, False, False]).iloc[0]
    else:
        best = grid_df.sort_values(["strategy_score", "sharpe", "cumulative_return"], ascending=[False, False, False]).iloc[0]

    best_cfg = {
        "mode": str(best["mode"]),
        "top_k": int(best["top_k"]),
        "min_prob": float(best["min_prob"]),
        "threshold": float(best["threshold"]),
        "constraints_satisfied": bool(best["constraints_ok"]),
    }
    return best_cfg, best.to_dict(), grid_df


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys, logits_list = [], []

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


def collect_candidate_metrics_for_fold(
    fold_id: int,
    split_name: str,
    pred_df: pd.DataFrame,
    horizon: int,
    candidate_grid: List[Dict],
    transaction_cost_bps: float,
    max_val_drawdown: float,
) -> pd.DataFrame:
    rows = []
    for cfg in candidate_grid:
        stats = run_strategy_backtest(
            pred_df=pred_df,
            horizon=horizon,
            mode=cfg["mode"],
            top_k=cfg["top_k"],
            min_prob=cfg["min_prob"],
            threshold=cfg["threshold"],
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.append({
            "fold_id": fold_id,
            "split_name": split_name,
            "mode": cfg["mode"],
            "top_k": cfg["top_k"],
            "min_prob": cfg["min_prob"],
            "threshold": cfg["threshold"],
            "constraints_ok": bool(stats["max_drawdown"] >= max_val_drawdown) if split_name == "val" else np.nan,
            **stats,
        })
    return pd.DataFrame(rows)


def run_one_fold(
    raw_df: pd.DataFrame,
    feats: List[str],
    target_col: str,
    return_col: str,
    horizon: int,
    fold: Dict,
    device: torch.device,
    args,
    fold_dir: Path,
    candidate_grid: List[Dict],
) -> Tuple[Dict, pd.DataFrame]:
    fold_df = assign_fold_splits(raw_df, fold)

    train_raw = fold_df[fold_df["wf_split"] == "train"].copy()
    val_raw = fold_df[fold_df["wf_split"] == "val"].copy()
    test_raw = fold_df[fold_df["wf_split"] == "test"].copy()

    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise ValueError(f"Fold {fold['fold_id']} has empty train/val/test split.")

    mean, std = fit_standardizer(train_raw, feats)
    df = apply_standardizer(fold_df, feats, mean, std)

    X, y, meta_df = build_sequences(df, feats, target_col, return_col, seq_len=args.seq_len, split_col="wf_split")
    idx = split_indices(meta_df)

    if len(idx["train"]) == 0 or len(idx["val"]) == 0 or len(idx["test"]) == 0:
        raise ValueError(f"Fold {fold['fold_id']} has empty sequence split after sequence construction.")

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
    ).to(device)

    pos_weight_tensor = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4
    )

    val_meta = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    test_meta = meta_df.iloc[idx["test"]].copy().reset_index(drop=True)

    best_auc_epoch = -1
    best_auc = -1e18
    best_strategy_epoch = -1
    best_strategy_score = -1e18
    best_strategy_state = None
    best_strategy_cfg = None
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

        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics_default["f1"],
            "val_auc": val_metrics_default["auc"],
            "val_strategy_mode": best_cfg_epoch["mode"],
            "val_strategy_top_k": best_cfg_epoch["top_k"],
            "val_strategy_min_prob": best_cfg_epoch["min_prob"],
            "val_strategy_threshold": best_cfg_epoch["threshold"],
            "val_strategy_constraints_satisfied": best_cfg_epoch["constraints_satisfied"],
            "val_strategy_sharpe": best_stats_epoch["sharpe"],
            "val_strategy_cumret": best_stats_epoch["cumulative_return"],
            "val_strategy_mdd": best_stats_epoch["max_drawdown"],
            "val_strategy_turnover": best_stats_epoch["avg_turnover"],
            "val_strategy_score": strategy_score_epoch,
        })

        print(
            f"[fold {fold['fold_id']:02d}] epoch {epoch:02d} | lr={lr_now:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_metrics_default['f1']:.4f} | val_auc={val_metrics_default['auc']}"
        )
        print(
            f"                     best_val_strategy: mode={best_cfg_epoch['mode']}, "
            f"top_k={best_cfg_epoch['top_k']}, min_prob={best_cfg_epoch['min_prob']:.2f}, "
            f"thr={best_cfg_epoch['threshold']:.2f}, constraints={best_cfg_epoch['constraints_satisfied']}, "
            f"sharpe={best_stats_epoch['sharpe']:.4f}, cumret={best_stats_epoch['cumulative_return']:.4f}, "
            f"mdd={best_stats_epoch['max_drawdown']:.4f}, turnover={best_stats_epoch['avg_turnover']:.4f}, "
            f"score={strategy_score_epoch:.4f}"
        )

        if np.isfinite(val_metrics_default["auc"]) and val_metrics_default["auc"] > best_auc + 1e-6:
            best_auc = float(val_metrics_default["auc"])
            best_auc_epoch = epoch

        if strategy_score_epoch > best_strategy_score + 1e-6:
            best_strategy_score = strategy_score_epoch
            best_strategy_epoch = epoch
            best_strategy_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_strategy_cfg = dict(best_cfg_epoch)
            wait = 0
        else:
            if epoch >= args.min_epochs:
                wait += 1

        scheduler.step(strategy_score_epoch)

        if epoch >= args.min_epochs and wait >= args.patience:
            print(
                f"[fold {fold['fold_id']:02d}] early stop at epoch {epoch}. "
                f"best_strategy_epoch={best_strategy_epoch}, best_strategy_score={best_strategy_score:.6f}, "
                f"best_auc_epoch={best_auc_epoch}, best_auc={best_auc:.6f}"
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / "training_history.csv", index=False)

    model.load_state_dict(best_strategy_state)
    strat_val_prob = predict_probs(model, val_loader, device)
    strat_test_prob = predict_probs(model, test_loader, device)

    strat_val_metrics = classification_metrics(y[idx["val"]], strat_val_prob, threshold=0.5)
    strat_test_metrics = classification_metrics(y[idx["test"]], strat_test_prob, threshold=0.5)

    strat_val_pred_df = val_meta.copy()
    strat_val_pred_df["pred_prob"] = strat_val_prob
    strat_test_pred_df = test_meta.copy()
    strat_test_pred_df["pred_prob"] = strat_test_prob

    selected_val_stats = run_strategy_backtest(
        pred_df=strat_val_pred_df,
        horizon=horizon,
        mode=best_strategy_cfg["mode"],
        top_k=best_strategy_cfg["top_k"],
        min_prob=best_strategy_cfg["min_prob"],
        threshold=best_strategy_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )
    selected_test_stats = run_strategy_backtest(
        pred_df=strat_test_pred_df,
        horizon=horizon,
        mode=best_strategy_cfg["mode"],
        top_k=best_strategy_cfg["top_k"],
        min_prob=best_strategy_cfg["min_prob"],
        threshold=best_strategy_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    current_fixed_val_stats = run_strategy_backtest(
        pred_df=strat_val_pred_df,
        horizon=horizon,
        mode=args.current_fixed_mode,
        top_k=args.current_fixed_top_k,
        min_prob=args.current_fixed_min_prob,
        threshold=args.current_fixed_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    current_fixed_test_stats = run_strategy_backtest(
        pred_df=strat_test_pred_df,
        horizon=horizon,
        mode=args.current_fixed_mode,
        top_k=args.current_fixed_top_k,
        min_prob=args.current_fixed_min_prob,
        threshold=args.current_fixed_threshold,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    selected_cost_rows = []
    current_fixed_cost_rows = []
    for cost in args.cost_grid_list:
        sel = run_strategy_backtest(
            pred_df=strat_test_pred_df,
            horizon=horizon,
            mode=best_strategy_cfg["mode"],
            top_k=best_strategy_cfg["top_k"],
            min_prob=best_strategy_cfg["min_prob"],
            threshold=best_strategy_cfg["threshold"],
            transaction_cost_bps=cost,
        )
        cur = run_strategy_backtest(
            pred_df=strat_test_pred_df,
            horizon=horizon,
            mode=args.current_fixed_mode,
            top_k=args.current_fixed_top_k,
            min_prob=args.current_fixed_min_prob,
            threshold=args.current_fixed_threshold,
            transaction_cost_bps=cost,
        )
        selected_cost_rows.append({"fold_id": fold["fold_id"], "strategy_type": "selected", "cost_bps": cost, **sel})
        current_fixed_cost_rows.append({"fold_id": fold["fold_id"], "strategy_type": "current_fixed", "cost_bps": cost, **cur})

    # Candidate grid diagnostics on the trained checkpoint
    val_candidate_df = collect_candidate_metrics_for_fold(
        fold_id=fold["fold_id"],
        split_name="val",
        pred_df=strat_val_pred_df,
        horizon=horizon,
        candidate_grid=candidate_grid,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
    )
    test_candidate_df = collect_candidate_metrics_for_fold(
        fold_id=fold["fold_id"],
        split_name="test",
        pred_df=strat_test_pred_df,
        horizon=horizon,
        candidate_grid=candidate_grid,
        transaction_cost_bps=args.transaction_cost_bps,
        max_val_drawdown=args.max_val_drawdown,
    )
    fold_candidate_df = val_candidate_df.merge(
        test_candidate_df,
        on=["fold_id", "mode", "top_k", "min_prob", "threshold"],
        suffixes=("_val", "_test"),
        how="inner",
    )
    fold_candidate_df.to_csv(fold_dir / "candidate_grid_metrics.csv", index=False)

    summary = {
        "fold_id": int(fold["fold_id"]),
        "is_partial_tail": bool(fold.get("is_partial_tail", False)),
        "train_start_date": str(pd.Timestamp(fold["train_start_date"]).date()),
        "train_end_date": str(pd.Timestamp(fold["train_end_date"]).date()),
        "val_start_date": str(pd.Timestamp(fold["val_start_date"]).date()),
        "val_end_date": str(pd.Timestamp(fold["val_end_date"]).date()),
        "test_start_date": str(pd.Timestamp(fold["test_start_date"]).date()),
        "test_end_date": str(pd.Timestamp(fold["test_end_date"]).date()),

        "n_train_rows": int(len(train_raw)),
        "n_val_rows": int(len(val_raw)),
        "n_test_rows": int(len(test_raw)),
        "n_train_seq": int(len(idx["train"])),
        "n_val_seq": int(len(idx["val"])),
        "n_test_seq": int(len(idx["test"])),

        "best_auc_epoch": int(best_auc_epoch),
        "best_val_auc": float(best_auc),
        "best_strategy_epoch": int(best_strategy_epoch),
        "best_strategy_score": float(best_strategy_score),

        "selected_mode": best_strategy_cfg["mode"],
        "selected_top_k": best_strategy_cfg["top_k"],
        "selected_min_prob": best_strategy_cfg["min_prob"],
        "selected_threshold": best_strategy_cfg["threshold"],
        "selected_constraints_satisfied": bool(best_strategy_cfg["constraints_satisfied"]),

        "current_fixed_mode": args.current_fixed_mode,
        "current_fixed_top_k": args.current_fixed_top_k,
        "current_fixed_min_prob": args.current_fixed_min_prob,
        "current_fixed_threshold": args.current_fixed_threshold,
        "selected_matches_current_fixed": bool(
            best_strategy_cfg["mode"] == args.current_fixed_mode and
            best_strategy_cfg["top_k"] == args.current_fixed_top_k and
            abs(best_strategy_cfg["min_prob"] - args.current_fixed_min_prob) < 1e-12 and
            abs(best_strategy_cfg["threshold"] - args.current_fixed_threshold) < 1e-12
        ),

        "val_auc_thr_0_5": float(strat_val_metrics["auc"]) if np.isfinite(strat_val_metrics["auc"]) else np.nan,
        "val_f1_thr_0_5": float(strat_val_metrics["f1"]),
        "test_auc_thr_0_5": float(strat_test_metrics["auc"]) if np.isfinite(strat_test_metrics["auc"]) else np.nan,
        "test_f1_thr_0_5": float(strat_test_metrics["f1"]),

        "selected_val_cumulative_return": float(selected_val_stats["cumulative_return"]),
        "selected_val_sharpe": float(selected_val_stats["sharpe"]),
        "selected_val_max_drawdown": float(selected_val_stats["max_drawdown"]),
        "selected_val_avg_turnover": float(selected_val_stats["avg_turnover"]),

        "selected_test_cumulative_return": float(selected_test_stats["cumulative_return"]),
        "selected_test_sharpe": float(selected_test_stats["sharpe"]),
        "selected_test_max_drawdown": float(selected_test_stats["max_drawdown"]),
        "selected_test_avg_turnover": float(selected_test_stats["avg_turnover"]),

        "current_fixed_val_cumulative_return": float(current_fixed_val_stats["cumulative_return"]),
        "current_fixed_val_sharpe": float(current_fixed_val_stats["sharpe"]),
        "current_fixed_val_max_drawdown": float(current_fixed_val_stats["max_drawdown"]),
        "current_fixed_val_avg_turnover": float(current_fixed_val_stats["avg_turnover"]),

        "current_fixed_test_cumulative_return": float(current_fixed_test_stats["cumulative_return"]),
        "current_fixed_test_sharpe": float(current_fixed_test_stats["sharpe"]),
        "current_fixed_test_max_drawdown": float(current_fixed_test_stats["max_drawdown"]),
        "current_fixed_test_avg_turnover": float(current_fixed_test_stats["avg_turnover"]),
    }

    with open(fold_dir / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fold_cost_df = pd.DataFrame(selected_cost_rows + current_fixed_cost_rows)
    fold_cost_df.to_csv(fold_dir / "cost_sensitivity.csv", index=False)

    return summary, fold_candidate_df, fold_cost_df


def summarize_numeric_metrics(fold_df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        ser = pd.to_numeric(fold_df[metric], errors="coerce")
        rows.append({
            "metric": metric,
            "mean": float(ser.mean()),
            "std": float(ser.std(ddof=1)) if len(ser.dropna()) > 1 else 0.0,
            "median": float(ser.median()),
            "min": float(ser.min()),
            "max": float(ser.max()),
        })
    return pd.DataFrame(rows)


def build_parameter_frequency_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        fold_df.groupby(["selected_mode", "selected_top_k", "selected_min_prob", "selected_threshold"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "selected_top_k", "selected_min_prob"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    grp["share"] = grp["count"] / len(fold_df)
    return grp


def add_regime_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["test_start_date"] = pd.to_datetime(out["test_start_date"])
    out["test_year"] = out["test_start_date"].dt.year

    def bucket(dt: pd.Timestamp) -> str:
        if dt.year <= 2019:
            return "pre_2020"
        if 2020 <= dt.year <= 2022:
            return "2020_2022"
        return "2023_plus"

    out["regime_bucket"] = out["test_start_date"].apply(bucket)
    return out


def summarize_by_group(fold_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, g in fold_df.groupby(group_col):
        rows.append({
            group_col: group_value,
            "n_folds": int(len(g)),
            "selected_test_cumret_mean": float(g["selected_test_cumulative_return"].mean()),
            "selected_test_cumret_std": float(g["selected_test_cumulative_return"].std(ddof=1)) if len(g) > 1 else 0.0,
            "selected_test_sharpe_mean": float(g["selected_test_sharpe"].mean()),
            "selected_test_sharpe_std": float(g["selected_test_sharpe"].std(ddof=1)) if len(g) > 1 else 0.0,
            "selected_test_mdd_mean": float(g["selected_test_max_drawdown"].mean()),
            "selected_test_turnover_mean": float(g["selected_test_avg_turnover"].mean()),
            "current_fixed_test_cumret_mean": float(g["current_fixed_test_cumulative_return"].mean()),
            "current_fixed_test_sharpe_mean": float(g["current_fixed_test_sharpe"].mean()),
            "current_fixed_test_mdd_mean": float(g["current_fixed_test_max_drawdown"].mean()),
            "selected_matches_current_fixed_rate": float(g["selected_matches_current_fixed"].mean()),
        })
    return pd.DataFrame(rows)


def aggregate_candidate_grid(all_candidate_df: pd.DataFrame, max_val_drawdown: float, transaction_cost_bps: float) -> pd.DataFrame:
    key_cols = ["mode", "top_k", "min_prob", "threshold"]
    rows = []

    for key, g in all_candidate_df.groupby(key_cols):
        g = g.copy()
        rows.append({
            "mode": key[0],
            "top_k": int(key[1]),
            "min_prob": float(key[2]),
            "threshold": float(key[3]),

            "n_folds": int(g["fold_id"].nunique()),
            "val_constraints_rate": float(pd.to_numeric(g["constraints_ok_val"], errors="coerce").mean()),
            "val_positive_rate": float((pd.to_numeric(g["cumulative_return_val"], errors="coerce") > 0).mean()),
            "test_positive_rate": float((pd.to_numeric(g["cumulative_return_test"], errors="coerce") > 0).mean()),

            "val_cumret_mean": float(pd.to_numeric(g["cumulative_return_val"], errors="coerce").mean()),
            "val_cumret_std": float(pd.to_numeric(g["cumulative_return_val"], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0,
            "val_sharpe_mean": float(pd.to_numeric(g["sharpe_val"], errors="coerce").mean()),
            "val_sharpe_std": float(pd.to_numeric(g["sharpe_val"], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0,
            "val_mdd_mean": float(pd.to_numeric(g["max_drawdown_val"], errors="coerce").mean()),
            "val_turnover_mean": float(pd.to_numeric(g["avg_turnover_val"], errors="coerce").mean()),
            "val_worst_cumret": float(pd.to_numeric(g["cumulative_return_val"], errors="coerce").min()),

            "test_cumret_mean": float(pd.to_numeric(g["cumulative_return_test"], errors="coerce").mean()),
            "test_cumret_std": float(pd.to_numeric(g["cumulative_return_test"], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0,
            "test_sharpe_mean": float(pd.to_numeric(g["sharpe_test"], errors="coerce").mean()),
            "test_sharpe_std": float(pd.to_numeric(g["sharpe_test"], errors="coerce").std(ddof=1)) if len(g) > 1 else 0.0,
            "test_mdd_mean": float(pd.to_numeric(g["max_drawdown_test"], errors="coerce").mean()),
            "test_turnover_mean": float(pd.to_numeric(g["avg_turnover_test"], errors="coerce").mean()),
        })

    agg_df = pd.DataFrame(rows)
    if len(agg_df) == 0:
        return agg_df

    # Validation-only robust deployable score (no test leakage in selection)
    agg_df["deployable_score"] = (
        2.0 * agg_df["val_sharpe_mean"]
        + 2.0 * agg_df["val_cumret_mean"]
        + 0.5 * agg_df["val_positive_rate"]
        + 0.5 * agg_df["val_constraints_rate"]
        - 0.15 * agg_df["val_turnover_mean"]
        - 0.50 * agg_df["val_cumret_std"]
        - 0.25 * agg_df["val_sharpe_std"]
        - 1.00 * np.maximum(0.0, -agg_df["val_worst_cumret"])
        - 3.00 * np.maximum(0.0, max_val_drawdown - agg_df["val_mdd_mean"])
    )
    agg_df = agg_df.sort_values(
        ["deployable_score", "val_constraints_rate", "val_sharpe_mean", "val_cumret_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return agg_df


def summarize_cost_sensitivity(cost_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_type, cost_bps), g in cost_df.groupby(["strategy_type", "cost_bps"]):
        rows.append({
            "strategy_type": strategy_type,
            "cost_bps": float(cost_bps),
            "cumret_mean": float(g["cumulative_return"].mean()),
            "cumret_std": float(g["cumulative_return"].std(ddof=1)) if len(g) > 1 else 0.0,
            "sharpe_mean": float(g["sharpe"].mean()),
            "sharpe_std": float(g["sharpe"].std(ddof=1)) if len(g) > 1 else 0.0,
            "mdd_mean": float(g["max_drawdown"].mean()),
            "turnover_mean": float(g["avg_turnover"].mean()),
            "positive_rate": float((g["cumulative_return"] > 0).mean()),
        })
    return pd.DataFrame(rows).sort_values(["strategy_type", "cost_bps"]).reset_index(drop=True)


def build_deployable_comparison(
    candidate_summary_df: pd.DataFrame,
    current_fixed_cfg: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    def match_cfg(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
        return df[
            (df["mode"] == cfg["mode"])
            & (df["top_k"] == cfg["top_k"])
            & (np.isclose(df["min_prob"], cfg["min_prob"]))
            & (np.isclose(df["threshold"], cfg["threshold"]))
        ].copy()

    current_row = match_cfg(candidate_summary_df, current_fixed_cfg)
    if len(candidate_summary_df) == 0:
        return pd.DataFrame(), {}

    recommended = candidate_summary_df.iloc[0].to_dict()
    out_rows = []

    if len(current_row) > 0:
        row = current_row.iloc[0].to_dict()
        row["config_label"] = "current_fixed"
        out_rows.append(row)

    rec = dict(recommended)
    rec["config_label"] = "recommended_validation_robust"
    out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    recommended_cfg = {
        "mode": rec["mode"],
        "top_k": int(rec["top_k"]),
        "min_prob": float(rec["min_prob"]),
        "threshold": float(rec["threshold"]),
        "deployable_score": float(rec["deployable_score"]),
    }
    return out_df, recommended_cfg


def main():
    parser = argparse.ArgumentParser(description="Transformer v5 walk-forward diagnostics and deployable-config recommender.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # Fold plan
    parser.add_argument("--walk_mode", type=str, default="expanding", choices=["expanding", "rolling"])
    parser.add_argument("--train_days", type=int, default=1260)
    parser.add_argument("--val_days", type=int, default=126)
    parser.add_argument("--test_days", type=int, default=126)
    parser.add_argument("--step_days", type=int, default=126)
    parser.add_argument("--max_folds", type=int, default=0, help="0 means use all possible folds")
    parser.add_argument("--allow_partial_last_test", action="store_true")
    parser.add_argument("--min_partial_test_days", type=int, default=60)

    # Model / training
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--min_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    # Adaptive strategy grid
    parser.add_argument("--mode_grid", type=str, default="topk")
    parser.add_argument("--topk_grid", type=str, default="2,3")
    parser.add_argument("--min_prob_grid", type=str, default="0.54,0.56,0.58,0.60")
    parser.add_argument("--threshold_grid", type=str, default="0.50")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)

    # Strategy score weights during training selection
    parser.add_argument("--auc_score_weight", type=float, default=0.25)
    parser.add_argument("--strategy_score_sharpe_weight", type=float, default=2.0)
    parser.add_argument("--strategy_score_cumret_weight", type=float, default=2.0)
    parser.add_argument("--strategy_score_mdd_penalty", type=float, default=3.0)
    parser.add_argument("--strategy_score_turnover_penalty", type=float, default=0.15)

    # Current frozen final config (for comparison)
    parser.add_argument("--current_fixed_mode", type=str, default="topk")
    parser.add_argument("--current_fixed_top_k", type=int, default=2)
    parser.add_argument("--current_fixed_min_prob", type=float, default=0.60)
    parser.add_argument("--current_fixed_threshold", type=float, default=0.50)

    # Cost sensitivity
    parser.add_argument("--cost_grid", type=str, default="5,10,20")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.mode_grid_list = [x.strip() for x in args.mode_grid.split(",") if x.strip()]
    args.topk_grid_list = [int(x.strip()) for x in args.topk_grid.split(",") if x.strip()]
    args.min_prob_grid_list = [float(x.strip()) for x in args.min_prob_grid.split(",") if x.strip()]
    args.threshold_grid_list = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    args.cost_grid_list = [float(x.strip()) for x in args.cost_grid.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")

    raw_df = pd.read_csv(args.data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df = raw_df.sort_values(["date", "stock"]).reset_index(drop=True)

    target_col, return_col = infer_columns(raw_df)
    horizon = infer_horizon(return_col)
    feats = feature_columns(raw_df, target_col, return_col)
    unique_dates = sorted(pd.to_datetime(raw_df["date"].drop_duplicates()).tolist())

    folds = make_walkforward_folds(
        unique_dates=unique_dates,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        walk_mode=args.walk_mode,
        max_folds=args.max_folds,
        allow_partial_last_test=args.allow_partial_last_test,
        min_partial_test_days=args.min_partial_test_days,
    )
    if len(folds) == 0:
        raise ValueError("No valid folds were created. Adjust train/val/test windows.")

    candidate_grid = expand_candidate_grid(
        mode_grid=args.mode_grid_list,
        topk_grid=args.topk_grid_list,
        min_prob_grid=args.min_prob_grid_list,
        threshold_grid=args.threshold_grid_list,
    )

    run_name = (
        f"{Path(args.data_path).stem}_transformer_v5_diagnostics_"
        f"{args.walk_mode}_tr{args.train_days}_va{args.val_days}_te{args.test_days}_st{args.step_days}"
    )
    out_root = Path(args.out_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["target_col"] = target_col
    run_config["return_col"] = return_col
    run_config["feature_columns"] = feats
    run_config["horizon"] = horizon
    run_config["n_unique_dates"] = len(unique_dates)
    run_config["n_candidate_configs"] = len(candidate_grid)
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    fold_plan_df = pd.DataFrame([{
        "fold_id": f["fold_id"],
        "is_partial_tail": bool(f.get("is_partial_tail", False)),
        "train_start_date": str(pd.Timestamp(f["train_start_date"]).date()),
        "train_end_date": str(pd.Timestamp(f["train_end_date"]).date()),
        "val_start_date": str(pd.Timestamp(f["val_start_date"]).date()),
        "val_end_date": str(pd.Timestamp(f["val_end_date"]).date()),
        "test_start_date": str(pd.Timestamp(f["test_start_date"]).date()),
        "test_end_date": str(pd.Timestamp(f["test_end_date"]).date()),
    } for f in folds])
    fold_plan_df.to_csv(out_root / "fold_plan.csv", index=False)

    print(f"Detected target: {target_col}")
    print(f"Detected return: {return_col}")
    print(f"Horizon: {horizon}")
    print(f"Feature count: {len(feats)}")
    print(f"Date range: {raw_df['date'].min().date()} -> {raw_df['date'].max().date()}")
    print(f"Unique dates: {len(unique_dates)}")
    print(f"Candidate configs: {len(candidate_grid)}")
    print(f"Generated folds: {len(folds)}")
    print(fold_plan_df.to_string(index=False))

    fold_summaries = []
    candidate_dfs = []
    cost_dfs = []

    for fold in folds:
        print("\n" + "=" * 120)
        print(
            f"Running fold {fold['fold_id']} | "
            f"partial_tail={bool(fold.get('is_partial_tail', False))} | "
            f"train={pd.Timestamp(fold['train_start_date']).date()}->{pd.Timestamp(fold['train_end_date']).date()} | "
            f"val={pd.Timestamp(fold['val_start_date']).date()}->{pd.Timestamp(fold['val_end_date']).date()} | "
            f"test={pd.Timestamp(fold['test_start_date']).date()}->{pd.Timestamp(fold['test_end_date']).date()}"
        )
        print("=" * 120)

        fold_dir = out_root / f"fold_{fold['fold_id']:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_summary, fold_candidate_df, fold_cost_df = run_one_fold(
            raw_df=raw_df,
            feats=feats,
            target_col=target_col,
            return_col=return_col,
            horizon=horizon,
            fold=fold,
            device=device,
            args=args,
            fold_dir=fold_dir,
            candidate_grid=candidate_grid,
        )
        fold_summaries.append(fold_summary)
        candidate_dfs.append(fold_candidate_df)
        cost_dfs.append(fold_cost_df)

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(out_root / "fold_results.csv", index=False)

    all_candidate_df = pd.concat(candidate_dfs, ignore_index=True)
    all_candidate_df.to_csv(out_root / "candidate_grid_all_folds.csv", index=False)

    all_cost_df = pd.concat(cost_dfs, ignore_index=True)
    all_cost_df.to_csv(out_root / "cost_sensitivity_all_folds.csv", index=False)

    coverage_summary = {
        "data_start_date": str(pd.Timestamp(raw_df["date"].min()).date()),
        "data_end_date": str(pd.Timestamp(raw_df["date"].max()).date()),
        "n_folds": int(len(fold_df)),
        "n_partial_tail_folds": int(fold_df["is_partial_tail"].sum()),
        "first_test_start_date": str(pd.to_datetime(fold_df["test_start_date"]).min().date()),
        "last_test_end_date": str(pd.to_datetime(fold_df["test_end_date"]).max().date()),
        "covers_2024_or_later": bool((pd.to_datetime(fold_df["test_end_date"]).dt.year >= 2024).any()),
        "covers_2025_or_later": bool((pd.to_datetime(fold_df["test_end_date"]).dt.year >= 2025).any()),
        "covers_2026_or_later": bool((pd.to_datetime(fold_df["test_end_date"]).dt.year >= 2026).any()),
    }
    with open(out_root / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(coverage_summary, f, ensure_ascii=False, indent=2)

    overall = {
        "n_folds": int(len(fold_df)),
        "selected_constraints_satisfied_rate": float(fold_df["selected_constraints_satisfied"].mean()),
        "positive_selected_test_cumret_rate": float((fold_df["selected_test_cumulative_return"] > 0).mean()),
        "positive_current_fixed_test_cumret_rate": float((fold_df["current_fixed_test_cumulative_return"] > 0).mean()),
        "selected_test_cumret_mean": float(fold_df["selected_test_cumulative_return"].mean()),
        "selected_test_cumret_std": float(fold_df["selected_test_cumulative_return"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "selected_test_sharpe_mean": float(fold_df["selected_test_sharpe"].mean()),
        "selected_test_sharpe_std": float(fold_df["selected_test_sharpe"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "selected_test_mdd_mean": float(fold_df["selected_test_max_drawdown"].mean()),
        "selected_test_mdd_std": float(fold_df["selected_test_max_drawdown"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "selected_test_auc_mean": float(pd.to_numeric(fold_df["test_auc_thr_0_5"], errors="coerce").mean()),
        "selected_test_auc_std": float(pd.to_numeric(fold_df["test_auc_thr_0_5"], errors="coerce").std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "current_fixed_test_cumret_mean": float(fold_df["current_fixed_test_cumulative_return"].mean()),
        "current_fixed_test_sharpe_mean": float(fold_df["current_fixed_test_sharpe"].mean()),
        "current_fixed_test_mdd_mean": float(fold_df["current_fixed_test_max_drawdown"].mean()),
        "selected_matches_current_fixed_rate": float(fold_df["selected_matches_current_fixed"].mean()),
    }
    with open(out_root / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    metric_cols = [
        "val_auc_thr_0_5",
        "val_f1_thr_0_5",
        "test_auc_thr_0_5",
        "test_f1_thr_0_5",
        "selected_val_cumulative_return",
        "selected_val_sharpe",
        "selected_val_max_drawdown",
        "selected_val_avg_turnover",
        "selected_test_cumulative_return",
        "selected_test_sharpe",
        "selected_test_max_drawdown",
        "selected_test_avg_turnover",
        "current_fixed_val_cumulative_return",
        "current_fixed_val_sharpe",
        "current_fixed_val_max_drawdown",
        "current_fixed_val_avg_turnover",
        "current_fixed_test_cumulative_return",
        "current_fixed_test_sharpe",
        "current_fixed_test_max_drawdown",
        "current_fixed_test_avg_turnover",
    ]
    metric_summary_df = summarize_numeric_metrics(fold_df, metric_cols)
    metric_summary_df.to_csv(out_root / "metric_summary.csv", index=False)

    param_freq_df = build_parameter_frequency_table(fold_df)
    param_freq_df.to_csv(out_root / "parameter_frequency.csv", index=False)

    fold_df_regime = add_regime_bucket(fold_df)
    by_year_df = summarize_by_group(fold_df_regime, "test_year")
    by_year_df.to_csv(out_root / "dispersion_by_test_year.csv", index=False)
    by_regime_df = summarize_by_group(fold_df_regime, "regime_bucket")
    by_regime_df.to_csv(out_root / "dispersion_by_regime_bucket.csv", index=False)

    negative_selected_df = fold_df[fold_df["selected_test_cumulative_return"] <= 0].copy()
    negative_selected_df.to_csv(out_root / "negative_selected_test_folds.csv", index=False)

    weak_validation_df = fold_df[
        (fold_df["selected_val_cumulative_return"] <= 0) |
        (fold_df["selected_val_sharpe"] <= 0.5)
    ].copy()
    weak_validation_df.to_csv(out_root / "weak_validation_folds.csv", index=False)

    candidate_summary_df = aggregate_candidate_grid(
        all_candidate_df=all_candidate_df,
        max_val_drawdown=args.max_val_drawdown,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    candidate_summary_df.to_csv(out_root / "candidate_grid_summary.csv", index=False)

    current_fixed_cfg = {
        "mode": args.current_fixed_mode,
        "top_k": args.current_fixed_top_k,
        "min_prob": args.current_fixed_min_prob,
        "threshold": args.current_fixed_threshold,
    }
    deployable_comparison_df, recommended_cfg = build_deployable_comparison(candidate_summary_df, current_fixed_cfg)
    deployable_comparison_df.to_csv(out_root / "deployable_config_comparison.csv", index=False)
    with open(out_root / "recommended_deployable_config.json", "w", encoding="utf-8") as f:
        json.dump(recommended_cfg, f, ensure_ascii=False, indent=2)

    cost_summary_df = summarize_cost_sensitivity(all_cost_df)
    cost_summary_df.to_csv(out_root / "cost_sensitivity_summary.csv", index=False)

    print("\nFinished Transformer v5 diagnostics.")
    print(f"Saved outputs to: {out_root}")
    print("\nCoverage summary:")
    print(json.dumps(coverage_summary, ensure_ascii=False, indent=2))
    print("\nOverall summary:")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\nRecommended deployable config:")
    print(json.dumps(recommended_cfg, ensure_ascii=False, indent=2))
    print("\nParameter frequency:")
    print(param_freq_df.to_string(index=False))
    print("\nDeployable config comparison:")
    print(deployable_comparison_df.to_string(index=False))
    print("\nCost sensitivity summary:")
    print(cost_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

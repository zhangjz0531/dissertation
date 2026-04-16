from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------
# Project import path
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import (
    MAIN_H5_DATA,
    EVALUATION_DIR,
    ABLATIONS_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)

# =========================================================
# DC Feature Ablation (H5)
#
# Variants:
#   - full_features : all numeric features (baseline)
#   - no_dc         : remove dc_* and mkt_dc_* features
#   - dc_only       : keep only dc_* and mkt_dc_* (optional via --include_dc_only)
#
# Key points:
#   - Same model family (Transformer)
#   - Same fixed execution config (from deployable manifest if available)
#   - Best checkpoint chosen by VALIDATION fixed-execution score
#
# Outputs:
#   Model Runs/ablations/dc_features/run_*/<variant>/
#     - feature_list.json
#     - training_history.csv
#     - best_model.pt  (state_dict)
#     - val_predictions.csv / test_predictions.csv
#     - val_fixed_actions.csv / test_fixed_actions.csv
#     - variant_summary.json
#
#   run_*/
#     - run_manifest.json
#     - dc_ablation_comparison.csv
#     - dc_ablation_deltas_vs_full.csv
#     - dc_ablation_summary.json
# =========================================================


# ---------------------- utils ----------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latest_file_by_glob(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def resolve_default_deployable_manifest_path() -> Optional[Path]:
    base = EVALUATION_DIR / "transformer_deployable"
    return latest_file_by_glob(base, "run_*/final_system_manifest.json")


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


def numeric_feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> List[str]:
    excluded = {"date", "stock", "split", target_col, return_col}
    feats = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def detect_dc_columns(feats: List[str]) -> List[str]:
    # keep both stock-level and market-level DC columns
    return sorted([c for c in feats if c.startswith("dc_") or c.startswith("mkt_dc_")])


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
    arr = np.asarray(period_returns, dtype=float)
    std = arr.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((arr.mean() / std) * math.sqrt(252.0 / horizon))


def max_drawdown_from_equity(equity_curve: List[float]) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos = float(ranks[pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2.0 * prec * rec / max(1e-12, prec + rec)
    auc = roc_auc_binary(y_true, prob)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else float("nan"),
    }


# ---------------------- model ----------------------


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 48, nhead: int = 4, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_sequences(
    df: pd.DataFrame,
    feats: List[str],
    target_col: str,
    return_col: str,
    seq_len: int,
    split_col: str = "split",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta: List[Dict] = []

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].to_numpy(dtype=np.float32)
        yg = g[target_col].to_numpy(dtype=np.float32)

        for i in range(seq_len - 1, len(g)):
            X_list.append(Xg[i - seq_len + 1 : i + 1])
            y_list.append(float(yg[i]))
            meta.append(
                {
                    "date": pd.Timestamp(g.loc[i, "date"]),
                    "stock": str(stock),
                    "split": str(g.loc[i, split_col]),
                    "future_return": float(g.loc[i, return_col]),
                }
            )

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, y, meta_df


def split_indices(meta_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "train": np.where(meta_df["split"].to_numpy() == "train")[0],
        "val": np.where(meta_df["split"].to_numpy() == "val")[0],
        "test": np.where(meta_df["split"].to_numpy() == "test")[0],
    }


# ---------------------- execution evaluation (fixed config) ----------------------


def select_weights(day_df: pd.DataFrame, mode: str, top_k: int, min_prob: float, threshold: float) -> Dict[str, float]:
    day = day_df.sort_values("pred_prob", ascending=False)
    if mode == "topk":
        chosen = day[day["pred_prob"] >= min_prob].head(top_k)
    elif mode == "threshold":
        chosen = day[day["pred_prob"] >= threshold]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if len(chosen) == 0:
        return {}
    w = 1.0 / len(chosen)
    return {str(r["stock"]): w for _, r in chosen.iterrows()}


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def run_fixed_strategy_backtest(
    pred_df: pd.DataFrame,
    horizon: int,
    mode: str,
    top_k: int,
    min_prob: float,
    threshold: float,
    transaction_cost_bps: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if pred_df.empty:
        empty = {
            "periods": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "avg_holdings": 0.0,
        }
        return empty, pd.DataFrame()

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    rebalance_dates = sorted(df["date"].drop_duplicates().tolist())[::horizon]
    tc = transaction_cost_bps / 10000.0

    prev_w: Dict[str, float] = {}
    equity = 1.0
    equity_curve = [equity]
    period_returns: List[float] = []
    turns: List[float] = []
    holds: List[int] = []
    rows: List[Dict] = []

    for dt in rebalance_dates:
        day = df[df["date"] == dt].copy()
        new_w = select_weights(day, mode=mode, top_k=top_k, min_prob=min_prob, threshold=threshold)

        ret_map = day.set_index("stock")["future_return"].to_dict()
        gross = float(sum(new_w[s] * float(ret_map.get(s, 0.0)) for s in new_w.keys()))
        turn = turnover(prev_w, new_w)
        net = gross - tc * turn

        equity *= (1.0 + net)
        equity_curve.append(equity)
        period_returns.append(net)
        turns.append(turn)
        holds.append(len(new_w))

        rows.append(
            {
                "date": pd.Timestamp(dt),
                "selected_stocks": "|".join(sorted(new_w.keys())),
                "n_holdings": len(new_w),
                "gross_return": gross,
                "net_return": net,
                "turnover": turn,
                "equity": equity,
            }
        )
        prev_w = new_w

    stats = {
        "periods": int(len(period_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(period_returns) * horizon),
        "sharpe": sharpe_ratio(period_returns, horizon=horizon),
        "max_drawdown": max_drawdown_from_equity(equity_curve),
        "win_rate": float(np.mean(np.asarray(period_returns) > 0)) if period_returns else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_holdings": float(np.mean(holds)) if holds else 0.0,
    }
    return stats, pd.DataFrame(rows)


def fixed_strategy_score(stats: Dict[str, float], max_val_drawdown: float) -> float:
    penalty = max(0.0, max_val_drawdown - stats["max_drawdown"])
    score = 2.0 * stats["sharpe"] + 2.0 * stats["cumulative_return"] - 3.0 * penalty - 0.15 * stats["avg_turnover"]
    return float(score)


# ---------------------- train/eval variant ----------------------


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys: List[np.ndarray] = []
    logits_list: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()) * len(xb))
            ys.append(yb.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    logits = np.concatenate(logits_list) if logits_list else np.array([])
    avg_loss = float(sum(losses) / max(1, len(loader.dataset)))
    return avg_loss, y_true, logits


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def save_df_with_dates(df: pd.DataFrame, out_path: Path) -> None:
    tmp = df.copy()
    if "date" in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_path, index=False)


def train_and_evaluate_variant(
    variant_name: str,
    raw_df: pd.DataFrame,
    feats: List[str],
    dc_in_variant: List[str],
    target_col: str,
    return_col: str,
    horizon: int,
    device: torch.device,
    args,
    variant_dir: Path,
    fixed_cfg: Dict,
) -> Dict:
    if not feats:
        raise ValueError(f"[{variant_name}] empty feature list.")

    print("\n" + "=" * 78)
    print(f"[{variant_name}] n_features={len(feats)} | n_dc_features={len(dc_in_variant)}")

    # standardize by TRAIN only
    train_raw = raw_df[raw_df["split"] == "train"].copy()
    mean, std = fit_standardizer(train_raw, feats)
    df = apply_standardizer(raw_df, feats, mean, std)

    X, y, meta_df = build_sequences(df, feats, target_col, return_col, seq_len=args.seq_len, split_col="split")
    idx = split_indices(meta_df)

    if len(idx["train"]) == 0 or len(idx["val"]) == 0 or len(idx["test"]) == 0:
        raise RuntimeError(f"[{variant_name}] empty train/val/test after sequence building.")

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4)

    val_meta = meta_df.iloc[idx["val"]].copy().reset_index(drop=True)
    test_meta = meta_df.iloc[idx["test"]].copy().reset_index(drop=True)

    best_state = None
    best_epoch = -1
    best_score = -1e18
    wait = 0
    history: List[Dict] = []

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
            train_losses.append(float(loss.item()) * len(xb))

        train_loss = float(sum(train_losses) / max(1, len(train_loader.dataset)))
        val_loss, val_y_true, val_logits = evaluate_model(model, val_loader, device, criterion)
        val_prob = 1.0 / (1.0 + np.exp(-val_logits))

        val_cls = classification_metrics(val_y_true, val_prob, threshold=0.5)

        val_pred_df = val_meta.copy()
        val_pred_df["pred_prob"] = val_prob

        val_fixed_stats, _ = run_fixed_strategy_backtest(
            pred_df=val_pred_df,
            horizon=horizon,
            mode=fixed_cfg["mode"],
            top_k=fixed_cfg["top_k"],
            min_prob=fixed_cfg["min_prob"],
            threshold=fixed_cfg["threshold"],
            transaction_cost_bps=args.transaction_cost_bps,
        )
        score = fixed_strategy_score(val_fixed_stats, args.max_val_drawdown)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_cls["auc"],
                "val_f1": val_cls["f1"],
                "val_fixed_cumret": val_fixed_stats["cumulative_return"],
                "val_fixed_sharpe": val_fixed_stats["sharpe"],
                "val_fixed_mdd": val_fixed_stats["max_drawdown"],
                "val_fixed_turnover": val_fixed_stats["avg_turnover"],
                "val_fixed_score": score,
            }
        )

        print(
            f"[{variant_name}] ep{epoch:02d} lr={lr_now:.6f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_cls['auc']} val_f1={val_cls['f1']:.4f} "
            f"val_cumret={val_fixed_stats['cumulative_return']:.4f} "
            f"val_sharpe={val_fixed_stats['sharpe']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if epoch >= args.min_epochs:
                wait += 1

        scheduler.step(score)
        if epoch >= args.min_epochs and wait >= args.patience:
            print(f"[{variant_name}] early stop at ep{epoch}. best_ep={best_epoch} best_score={best_score:.6f}")
            break

    if best_state is None:
        raise RuntimeError(f"[{variant_name}] failed to produce a checkpoint.")

    # persist
    pd.DataFrame(history).to_csv(variant_dir / "training_history.csv", index=False)
    with open(variant_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    torch.save(best_state, variant_dir / "best_model.pt")

    # load best
    model.load_state_dict(best_state)

    # predictions
    val_prob = predict_probs(model, val_loader, device)
    test_prob = predict_probs(model, test_loader, device)

    val_cls = classification_metrics(y[idx["val"]], val_prob, threshold=0.5)
    test_cls = classification_metrics(y[idx["test"]], test_prob, threshold=0.5)

    val_pred_df = val_meta.copy()
    val_pred_df["pred_prob"] = val_prob
    test_pred_df = test_meta.copy()
    test_pred_df["pred_prob"] = test_prob

    save_df_with_dates(val_pred_df, variant_dir / "val_predictions.csv")
    save_df_with_dates(test_pred_df, variant_dir / "test_predictions.csv")

    # fixed strategy actions/stats
    val_stats, val_actions = run_fixed_strategy_backtest(
        pred_df=val_pred_df,
        horizon=horizon,
        mode=fixed_cfg["mode"],
        top_k=fixed_cfg["top_k"],
        min_prob=fixed_cfg["min_prob"],
        threshold=fixed_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )
    test_stats, test_actions = run_fixed_strategy_backtest(
        pred_df=test_pred_df,
        horizon=horizon,
        mode=fixed_cfg["mode"],
        top_k=fixed_cfg["top_k"],
        min_prob=fixed_cfg["min_prob"],
        threshold=fixed_cfg["threshold"],
        transaction_cost_bps=args.transaction_cost_bps,
    )

    save_df_with_dates(val_actions, variant_dir / "val_fixed_actions.csv")
    save_df_with_dates(test_actions, variant_dir / "test_fixed_actions.csv")

    summary = {
        "variant": variant_name,
        "n_features": int(len(feats)),
        "n_dc_features": int(len(dc_in_variant)),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "val_classification": val_cls,
        "test_classification": test_cls,
        "val_fixed_strategy": val_stats,
        "test_fixed_strategy": test_stats,
    }
    with open(variant_dir / "variant_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ---------------------- fixed config resolution ----------------------


def build_fixed_cfg_from_manifest_or_args(args) -> Tuple[Dict, Optional[Path]]:
    fixed_cfg = None
    manifest_path = Path(args.deployable_manifest_path) if args.deployable_manifest_path else resolve_default_deployable_manifest_path()

    if manifest_path is not None and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        frozen = manifest.get("frozen_config", {})
        tx_cost = manifest.get("transaction_cost_bps", args.transaction_cost_bps)

        if frozen:
            fixed_cfg = {
                "mode": str(frozen.get("mode", "topk")),
                "top_k": int(frozen.get("top_k", args.fixed_top_k)),
                "min_prob": float(frozen.get("min_prob", args.fixed_min_prob)),
                "threshold": float(frozen.get("threshold", args.fixed_threshold)),
            }
            args.transaction_cost_bps = float(tx_cost)

    if fixed_cfg is None:
        fixed_cfg = {
            "mode": args.fixed_mode,
            "top_k": int(args.fixed_top_k),
            "min_prob": float(args.fixed_min_prob),
            "threshold": float(args.fixed_threshold),
        }

    return fixed_cfg, manifest_path if manifest_path is not None and manifest_path.exists() else None


# ---------------------- main ----------------------


def main() -> None:
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(description="DC feature ablation under the same Transformer framework (H5).")
    parser.add_argument("--data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--deployable_manifest_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(ABLATIONS_DIR / "dc_features"))
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=16)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)

    parser.add_argument("--fixed_mode", type=str, default="topk")
    parser.add_argument("--fixed_top_k", type=int, default=2)
    parser.add_argument("--fixed_min_prob", type=float, default=0.54)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)

    parser.add_argument("--include_dc_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    fixed_cfg, used_manifest = build_fixed_cfg_from_manifest_or_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_df = pd.read_csv(args.data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    target_col, return_col = infer_columns(raw_df)
    horizon = infer_horizon(return_col)
    all_feats = numeric_feature_columns(raw_df, target_col, return_col)
    dc_cols = detect_dc_columns(all_feats)
    non_dc_cols = [c for c in all_feats if c not in dc_cols]

    if not dc_cols:
        raise RuntimeError("No dc_* or mkt_dc_* columns detected. DC ablation cannot run.")

    variants: List[Tuple[str, List[str]]] = [
        ("full_features", all_feats),
        ("no_dc", non_dc_cols),
    ]
    if args.include_dc_only:
        variants.append(("dc_only", dc_cols))

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "data_path": str(Path(args.data_path).resolve()),
        "used_deployable_manifest": str(used_manifest.resolve()) if used_manifest else "",
        "target_col": target_col,
        "return_col": return_col,
        "horizon": int(horizon),
        "fixed_strategy_config": fixed_cfg,
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "all_feature_count": int(len(all_feats)),
        "dc_feature_count": int(len(dc_cols)),
        "dc_features": dc_cols,
        "non_dc_feature_count": int(len(non_dc_cols)),
        "run_args": vars(args),
    }
    with open(out_root / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    print(f"Using device: {device}")
    print(f"Loading data: {args.data_path}")
    print(f"Horizon: {horizon}")
    print(f"All numeric features: {len(all_feats)} | DC features: {len(dc_cols)}")
    print(f"Fixed strategy config: {json.dumps(fixed_cfg, ensure_ascii=False)}")
    if used_manifest:
        print(f"Using deployable manifest: {used_manifest}")

    summaries: List[Dict] = []
    for variant_name, feats in variants:
        variant_dir = out_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        dc_in_variant = [c for c in feats if c in dc_cols]

        summary = train_and_evaluate_variant(
            variant_name=variant_name,
            raw_df=raw_df,
            feats=feats,
            dc_in_variant=dc_in_variant,
            target_col=target_col,
            return_col=return_col,
            horizon=horizon,
            device=device,
            args=args,
            variant_dir=variant_dir,
            fixed_cfg=fixed_cfg,
        )
        summaries.append(summary)

    compare_rows = []
    for s in summaries:
        compare_rows.append(
            {
                "variant": s["variant"],
                "n_features": s["n_features"],
                "n_dc_features": s["n_dc_features"],
                "best_epoch": s["best_epoch"],
                "best_score": s["best_score"],
                "val_auc": s["val_classification"]["auc"],
                "val_f1": s["val_classification"]["f1"],
                "test_auc": s["test_classification"]["auc"],
                "test_f1": s["test_classification"]["f1"],
                "val_fixed_cumret": s["val_fixed_strategy"]["cumulative_return"],
                "val_fixed_sharpe": s["val_fixed_strategy"]["sharpe"],
                "val_fixed_mdd": s["val_fixed_strategy"]["max_drawdown"],
                "val_fixed_turnover": s["val_fixed_strategy"]["avg_turnover"],
                "test_fixed_cumret": s["test_fixed_strategy"]["cumulative_return"],
                "test_fixed_sharpe": s["test_fixed_strategy"]["sharpe"],
                "test_fixed_mdd": s["test_fixed_strategy"]["max_drawdown"],
                "test_fixed_turnover": s["test_fixed_strategy"]["avg_turnover"],
            }
        )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_root / "dc_ablation_comparison.csv", index=False)

    # deltas vs full
    if (compare_df["variant"] == "full_features").any():
        base = compare_df[compare_df["variant"] == "full_features"].iloc[0]
        delta_rows = []
        for _, row in compare_df.iterrows():
            if row["variant"] == "full_features":
                continue
            delta_rows.append(
                {
                    "variant": row["variant"],
                    "delta_test_auc_vs_full": float(row["test_auc"] - base["test_auc"]),
                    "delta_test_f1_vs_full": float(row["test_f1"] - base["test_f1"]),
                    "delta_test_fixed_cumret_vs_full": float(row["test_fixed_cumret"] - base["test_fixed_cumret"]),
                    "delta_test_fixed_sharpe_vs_full": float(row["test_fixed_sharpe"] - base["test_fixed_sharpe"]),
                    "delta_test_fixed_mdd_vs_full": float(row["test_fixed_mdd"] - base["test_fixed_mdd"]),
                    "delta_test_fixed_turnover_vs_full": float(row["test_fixed_turnover"] - base["test_fixed_turnover"]),
                }
            )
        pd.DataFrame(delta_rows).to_csv(out_root / "dc_ablation_deltas_vs_full.csv", index=False)

    summary = {
        "fixed_strategy_config": fixed_cfg,
        "transaction_cost_bps": float(args.transaction_cost_bps),
        "dc_feature_count": int(len(dc_cols)),
        "dc_features": dc_cols,
        "variant_summaries": summaries,
    }
    with open(out_root / "dc_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("DC ablation finished.")
    print(f"Saved outputs to: {out_root}")
    print("\nComparison table:")
    print(compare_df.to_string(index=False))


if __name__ == "__main__":
    main()

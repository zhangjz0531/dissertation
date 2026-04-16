
import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Rebuild full-split predictions from a saved v4 checkpoint
#
# Why this script:
# ----------------
# The previous exported predictions file likely had pred_prob only for
# val/test, while train split pred_prob stayed NaN.
# That breaks downstream overlay training because the state includes
# pred_prob-based features such as top1_prob / top2_prob / mean_prob.
#
# This script:
#   1) Loads the cleaned experiment dataset
#   2) Recreates the exact sequence samples
#   3) Loads the saved checkpoint (Transformer or LSTM)
#   4) Runs inference on ALL splits: train / val / test
#   5) Exports a corrected predictions_all_splits.csv
#
# Recommended use right now:
# --------------------------
# Rebuild the H5 LSTM v4 predictions file, then use that fixed file as
# the input for the overlay stage.
# =========================================================


# -----------------------------
# Model definitions (match v4)
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


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 48, nhead: int = 4, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
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


class SequenceOnlyDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# -----------------------------
# Helpers
# -----------------------------
def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    target_cols = [c for c in df.columns if c.startswith("target_up_")]
    return_cols = [c for c in df.columns if c.startswith("future_return_")]
    if len(target_cols) != 1 or len(return_cols) != 1:
        raise ValueError("Dataset must contain exactly one target_up_* and one future_return_* column.")
    return target_cols[0], return_cols[0]


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
) -> Tuple[np.ndarray, pd.DataFrame]:
    X_list = []
    meta = []

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    for stock, g in df.groupby("stock", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        Xg = g[feats].values.astype(np.float32)

        for i in range(seq_len - 1, len(g)):
            start = i - seq_len + 1
            X_list.append(Xg[start:i + 1])
            meta.append({
                "date": pd.Timestamp(g.loc[i, "date"]),
                "stock": stock,
                "split": g.loc[i, "split"],
                "future_return": float(g.loc[i, return_col]),
                "target": int(g.loc[i, target_col]),
            })

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return X, meta_df


def make_model(model_type: str, input_dim: int, args):
    if model_type == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
        )
    elif model_type == "transformer":
        return TransformerClassifier(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.transformer_layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def batch_predict_prob(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    ds = SequenceOnlyDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    probs = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(prob)
    if len(probs) == 0:
        return np.array([])
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser(description="Rebuild corrected full-split predictions from a saved v4 checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to main_experiment_h1/h5.csv")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved *_best_model.pt checkpoint")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "transformer"])
    parser.add_argument("--out_dir", type=str, required=True)

    # v4 defaults
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)

    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading dataset: {args.data_path}")
    print(f"Loading checkpoint: {args.checkpoint_path}")

    df = pd.read_csv(args.data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    feats = feature_columns(df, target_col, return_col)

    train_df = df[df["split"] == "train"].copy()
    mean, std = fit_standardizer(train_df, feats)
    df_std = apply_standardizer(df, feats, mean, std)

    X, meta_df = build_sequences(df_std, feats, target_col, return_col, seq_len=args.seq_len)

    print(f"Feature count: {len(feats)}")
    print(f"Sequence samples: {len(X)}")
    print(meta_df['split'].value_counts().to_string())

    model = make_model(args.model_type, input_dim=len(feats), args=args).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)

    pred_prob = batch_predict_prob(model, X, batch_size=args.batch_size, device=device)

    out_df = meta_df.copy()
    out_df["pred_prob"] = pred_prob
    out_df["pred_logit"] = np.log(np.clip(pred_prob, 1e-7, 1 - 1e-7) / np.clip(1 - pred_prob, 1e-7, 1 - 1e-7))

    all_path = out_dir / f"{args.model_type}_predictions_all_splits_fixed.csv"
    test_path = out_dir / f"{args.model_type}_predictions_test_fixed.csv"
    out_df2 = out_df.copy()
    out_df2["date"] = out_df2["date"].dt.strftime("%Y-%m-%d")
    out_df2.to_csv(all_path, index=False)
    out_df2[out_df2["split"] == "test"].to_csv(test_path, index=False)

    sanity = (
        out_df.groupby("split")["pred_prob"]
        .agg(["count", "size", "min", "max", "mean"])
        .reset_index()
    )
    sanity["na_count"] = out_df.groupby("split")["pred_prob"].apply(lambda s: int(s.isna().sum())).values

    sanity_path = out_dir / f"{args.model_type}_prediction_sanity_check.csv"
    sanity.to_csv(sanity_path, index=False)

    config = vars(args).copy()
    config["target_col"] = target_col
    config["return_col"] = return_col
    config["feature_columns"] = feats
    with open(out_dir / f"{args.model_type}_rebuild_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(all_path)
    print(test_path)
    print(sanity_path)

    print("\nSanity check by split:")
    print(sanity.to_string(index=False))


if __name__ == "__main__":
    main()

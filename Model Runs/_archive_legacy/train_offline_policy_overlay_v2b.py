
import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Offline policy overlay v2b
#
# Changes vs previous offline overlay:
# 1) Shrink action space from 4 -> 3
#    - cash
#    - top1_full
#    - top2_equal
#    (drop top1_half because it was nearly absent in pseudo-labels)
#
# 2) Add class weights in CrossEntropy
#    - to reduce bias toward frequent pseudo-label classes
#
# 3) Select best checkpoint by VALIDATION ECONOMIC OBJECTIVE
#    instead of val_loss only
#    - primary: validation excess cumulative return over base
#    - secondary: validation excess Sharpe over base
#    - with drawdown penalty
#
# 4) Fix turnover metric calculation in overlay/base backtest
#
# This is still an offline policy-learning overlay, not online RL.
# It is designed to be practical and stable for your dissertation setting.
# =========================================================


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utility
# -----------------------------
def annualized_return_from_equity(final_equity: float, total_days: int) -> float:
    if final_equity <= 0:
        return -1.0
    total_days = max(1, int(total_days))
    return float(final_equity ** (252.0 / total_days) - 1.0)


def sharpe_from_returns(period_returns: List[float], horizon: int) -> float:
    if len(period_returns) <= 1:
        return 0.0
    arr = np.array(period_returns, dtype=float)
    std = arr.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float((arr.mean() / std) * math.sqrt(252.0 / horizon))


def max_drawdown_from_equity(equity_curve: List[float]) -> float:
    eq = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


# -----------------------------
# Data preparation
# -----------------------------
DEFAULT_STATE_COLS = [
    "mkt_dc_trend",
    "mkt_dc_event",
    "mkt_return_5d",
    "mkt_volatility_20d",
    "vix_z_60",
    "credit_stress",
    "dc_trend",
    "dc_event",
    "dc_tmv",
    "return_21d_cs_z",
    "rsi_14_cs_z",
    "macd_hist_pct_cs_z",
]

# 3-action space only
ACTION_SPACE = {
    0: "cash",
    1: "top1_full",
    2: "top2_equal",
}


def load_predictions(predictions_path: str) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    required = {"date", "stock", "split", "future_return", "target", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def maybe_merge_features(pred_df: pd.DataFrame, feature_df_path: Optional[str]) -> pd.DataFrame:
    if feature_df_path is None:
        return pred_df.copy()

    feat_df = pd.read_csv(feature_df_path)
    if "date" not in feat_df.columns or "stock" not in feat_df.columns:
        raise ValueError("Feature data must contain date and stock columns.")
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    feat_df = feat_df.drop(
        columns=[c for c in feat_df.columns if c in pred_df.columns and c not in {"date", "stock"}],
        errors="ignore",
    )
    merged = pred_df.merge(feat_df, on=["date", "stock"], how="left")
    return merged


def infer_horizon(predictions_path: str, feature_df_path: Optional[str]) -> int:
    if feature_df_path is not None:
        raw_df = pd.read_csv(feature_df_path)
        future_cols = [c for c in raw_df.columns if c.startswith("future_return_")]
        if len(future_cols) == 1:
            col = future_cols[0]
            if col.endswith("1d"):
                return 1
            if col.endswith("5d"):
                return 5

    stem = Path(predictions_path).stem.lower()
    if "h1" in stem or "1d" in stem:
        return 1
    return 5


def pick_available_state_cols(df: pd.DataFrame, requested_cols: List[str]) -> List[str]:
    return [c for c in requested_cols if c in df.columns]


def build_rebalance_dates(split_df: pd.DataFrame, horizon: int) -> List[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(split_df["date"]).drop_duplicates().tolist())
    return unique_dates[::horizon]


def action_to_weights(sorted_stocks: List[str], action: int) -> Dict[str, float]:
    if action == 0:
        return {}
    elif action == 1:
        return {sorted_stocks[0]: 1.0}
    elif action == 2:
        if len(sorted_stocks) == 1:
            return {sorted_stocks[0]: 1.0}
        return {sorted_stocks[0]: 0.5, sorted_stocks[1]: 0.5}
    else:
        raise ValueError(f"Unknown action {action}")


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def portfolio_return_from_weights(sorted_stocks: List[str], sorted_future_returns: List[float], weights: Dict[str, float]) -> float:
    ret_map = {s: r for s, r in zip(sorted_stocks, sorted_future_returns)}
    return float(sum(weights.get(s, 0.0) * ret_map.get(s, 0.0) for s in weights.keys()))


def base_strategy_weights(sorted_stocks: List[str], sorted_probs: List[float], base_min_prob: float) -> Dict[str, float]:
    if len(sorted_stocks) == 0:
        return {}
    if sorted_probs[0] >= base_min_prob:
        return {sorted_stocks[0]: 1.0}
    return {}


def summarize_day_state(day_df: pd.DataFrame, state_cols: List[str], base_min_prob: float) -> Tuple[np.ndarray, Dict]:
    g = day_df.sort_values("pred_prob", ascending=False).reset_index(drop=True).copy()

    probs = g["pred_prob"].astype(float).values
    top1 = g.iloc[0]
    top2 = g.iloc[1] if len(g) > 1 else g.iloc[0]
    top3 = g.iloc[2] if len(g) > 2 else g.iloc[min(1, len(g) - 1)]

    state = {
        "top1_prob": float(top1["pred_prob"]),
        "top2_prob": float(top2["pred_prob"]),
        "top3_prob": float(top3["pred_prob"]),
        "prob_gap_12": float(top1["pred_prob"] - top2["pred_prob"]),
        "prob_gap_13": float(top1["pred_prob"] - top3["pred_prob"]),
        "mean_prob": float(np.mean(probs)),
        "std_prob": float(np.std(probs)),
        "base_signal_active": float(top1["pred_prob"] >= base_min_prob),
    }

    for c in state_cols:
        state[f"top1_{c}"] = float(top1[c]) if c in top1.index and pd.notna(top1[c]) else 0.0

    for c in [x for x in ["return_21d_cs_z", "rsi_14_cs_z", "macd_hist_pct_cs_z"] if x in g.columns]:
        vals = pd.to_numeric(g[c], errors="coerce").fillna(0.0).values.astype(float)
        state[f"xs_mean_{c}"] = float(np.mean(vals))
        state[f"xs_std_{c}"] = float(np.std(vals))

    meta = {
        "date": pd.Timestamp(top1["date"]),
        "split": str(top1["split"]),
        "sorted_stocks": g["stock"].tolist(),
        "sorted_probs": g["pred_prob"].astype(float).tolist(),
        "sorted_future_returns": g["future_return"].astype(float).tolist(),
    }
    return np.array(list(state.values()), dtype=np.float32), meta


def prepare_overlay_steps(
    predictions_path: str,
    feature_df_path: Optional[str],
    base_min_prob: float,
) -> Tuple[pd.DataFrame, List[str], int]:
    pred_df = load_predictions(predictions_path)
    merged = maybe_merge_features(pred_df, feature_df_path)
    horizon = infer_horizon(predictions_path, feature_df_path)
    available_state_cols = pick_available_state_cols(merged, DEFAULT_STATE_COLS)

    rows = []
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name].copy()
        if split_df.empty:
            continue

        rebalance_dates = build_rebalance_dates(split_df, horizon=horizon)
        for dt in rebalance_dates:
            day = split_df[split_df["date"] == dt].copy()
            if day.empty:
                continue

            state_vec, meta = summarize_day_state(day, available_state_cols, base_min_prob=base_min_prob)
            rows.append({
                "date": meta["date"],
                "split": meta["split"],
                "state_vec": state_vec,
                "sorted_stocks": meta["sorted_stocks"],
                "sorted_probs": meta["sorted_probs"],
                "sorted_future_returns": meta["sorted_future_returns"],
            })

    df_steps = pd.DataFrame(rows)
    return df_steps, available_state_cols, horizon


def fit_state_standardizer(train_state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_state_matrix.mean(axis=0)
    std = train_state_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_state_standardizer(state_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state_matrix - mean) / std


# -----------------------------
# Offline pseudo-labels
# -----------------------------
def compute_action_reward(
    sorted_stocks: List[str],
    sorted_probs: List[float],
    sorted_future_returns: List[float],
    action: int,
    prev_overlay_w: Dict[str, float],
    prev_base_w: Dict[str, float],
    base_min_prob: float,
    transaction_cost_bps: float,
    inactivity_penalty: float,
) -> Tuple[float, Dict[str, float], Dict[str, float], float, float]:
    tc = transaction_cost_bps / 10000.0

    overlay_w = action_to_weights(sorted_stocks, action)
    overlay_turn = turnover(prev_overlay_w, overlay_w)
    overlay_net = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, overlay_w) - tc * overlay_turn

    base_w = base_strategy_weights(sorted_stocks, sorted_probs, base_min_prob)
    base_turn = turnover(prev_base_w, base_w)
    base_net = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, base_w) - tc * base_turn

    reward = overlay_net - base_net
    if action == 0 and len(sorted_probs) > 0 and sorted_probs[0] >= base_min_prob:
        reward -= inactivity_penalty

    return float(reward), overlay_w, base_w, float(overlay_net), float(base_net)


def build_pseudo_labels(
    df_steps: pd.DataFrame,
    base_min_prob: float,
    transaction_cost_bps: float,
    inactivity_penalty: float,
) -> pd.DataFrame:
    df = df_steps.copy().reset_index(drop=True)
    prev_overlay_w = {}
    prev_base_w = {}

    best_actions = []
    best_rewards = []
    all_action_rewards = []

    for i in range(len(df)):
        row = df.iloc[i]
        sorted_stocks = row["sorted_stocks"]
        sorted_probs = row["sorted_probs"]
        sorted_future_returns = row["sorted_future_returns"]

        rewards = {}
        overlay_ws = {}
        base_w_for_step = None

        for action in ACTION_SPACE.keys():
            r, overlay_w, base_w, _, _ = compute_action_reward(
                sorted_stocks=sorted_stocks,
                sorted_probs=sorted_probs,
                sorted_future_returns=sorted_future_returns,
                action=action,
                prev_overlay_w=prev_overlay_w,
                prev_base_w=prev_base_w,
                base_min_prob=base_min_prob,
                transaction_cost_bps=transaction_cost_bps,
                inactivity_penalty=inactivity_penalty,
            )
            rewards[action] = r
            overlay_ws[action] = overlay_w
            base_w_for_step = base_w

        best_action = max(rewards.items(), key=lambda x: x[1])[0]
        best_reward = rewards[best_action]

        prev_overlay_w = overlay_ws[best_action]
        prev_base_w = base_w_for_step if base_w_for_step is not None else {}

        best_actions.append(int(best_action))
        best_rewards.append(float(best_reward))
        all_action_rewards.append(rewards)

    df["best_action"] = best_actions
    df["best_action_reward"] = best_rewards
    for a in ACTION_SPACE.keys():
        df[f"reward_action_{a}"] = [r[a] for r in all_action_rewards]

    return df


# -----------------------------
# Policy network
# -----------------------------
class PolicyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_class_weights(y_train: np.ndarray, action_dim: int, power: float = 0.5) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=action_dim).astype(np.float32)
    counts[counts <= 0] = 1.0
    inv = 1.0 / np.power(counts, power)
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Backtest/eval
# -----------------------------
def run_overlay_backtest(
    df_steps: pd.DataFrame,
    horizon: int,
    base_min_prob: float,
    transaction_cost_bps: float,
    policy_model: Optional[nn.Module],
    device: Optional[torch.device],
    mode: str,
):
    tc = transaction_cost_bps / 10000.0

    overlay_equity = 1.0
    base_equity = 1.0
    overlay_peak = 1.0

    overlay_prev_w = {}
    base_prev_w = {}

    overlay_returns = []
    base_returns = []
    overlay_turnovers = []
    base_turnovers = []
    action_exposure = []
    logs = []

    for _, row in df_steps.iterrows():
        sorted_stocks = row["sorted_stocks"]
        sorted_probs = row["sorted_probs"]
        sorted_future_returns = row["sorted_future_returns"]

        base_w = base_strategy_weights(sorted_stocks, sorted_probs, base_min_prob)
        base_turn = turnover(base_prev_w, base_w)
        base_net = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, base_w) - tc * base_turn
        base_equity *= (1.0 + base_net)
        base_prev_w = base_w
        base_returns.append(base_net)
        base_turnovers.append(base_turn)

        if mode == "base":
            overlay_w = base_w.copy()
            action = 1 if len(base_w) == 1 and list(base_w.values())[0] == 1.0 else 0
        elif mode == "policy":
            x = torch.tensor(row["state_scaled"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy_model(x)
                action = int(torch.argmax(logits, dim=1).item())
            overlay_w = action_to_weights(sorted_stocks, action)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        overlay_turn = turnover(overlay_prev_w, overlay_w)
        overlay_net = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, overlay_w) - tc * overlay_turn
        overlay_equity *= (1.0 + overlay_net)
        overlay_prev_w = overlay_w
        overlay_returns.append(overlay_net)
        overlay_turnovers.append(overlay_turn)

        exposure = float(sum(overlay_w.values())) if len(overlay_w) > 0 else 0.0
        action_exposure.append(exposure)

        overlay_peak = max(overlay_peak, overlay_equity)
        overlay_dd = overlay_equity / max(overlay_peak, 1e-12) - 1.0

        logs.append({
            "date": str(pd.Timestamp(row["date"]).date()),
            "action": int(action),
            "action_name": ACTION_SPACE[int(action)],
            "overlay_turnover": float(overlay_turn),
            "base_turnover": float(base_turn),
            "overlay_net_return": float(overlay_net),
            "base_net_return": float(base_net),
            "overlay_equity": float(overlay_equity),
            "base_equity": float(base_equity),
            "overlay_drawdown": float(overlay_dd),
            "overlay_exposure": exposure,
        })

    overlay_metrics = {
        "periods": int(len(overlay_returns)),
        "cumulative_return": float(overlay_equity - 1.0),
        "annualized_return": annualized_return_from_equity(overlay_equity, len(overlay_returns) * horizon),
        "sharpe": sharpe_from_returns(overlay_returns, horizon),
        "max_drawdown": max_drawdown_from_equity([1.0] + list(np.cumprod(1 + np.array(overlay_returns)))),
        "win_rate": float(np.mean(np.array(overlay_returns) > 0)) if overlay_returns else 0.0,
        "avg_turnover": float(np.mean(overlay_turnovers)) if overlay_turnovers else 0.0,
        "avg_action_exposure_proxy": float(np.mean(action_exposure)) if action_exposure else 0.0,
    }

    base_metrics = {
        "periods": int(len(base_returns)),
        "cumulative_return": float(base_equity - 1.0),
        "annualized_return": annualized_return_from_equity(base_equity, len(base_returns) * horizon),
        "sharpe": sharpe_from_returns(base_returns, horizon),
        "max_drawdown": max_drawdown_from_equity([1.0] + list(np.cumprod(1 + np.array(base_returns)))),
        "win_rate": float(np.mean(np.array(base_returns) > 0)) if base_returns else 0.0,
        "avg_turnover": float(np.mean(base_turnovers)) if base_turnovers else 0.0,
    }

    return overlay_metrics, base_metrics, pd.DataFrame(logs)


# -----------------------------
# Training
# -----------------------------
def train_policy_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_df_steps: pd.DataFrame,
    horizon: int,
    base_min_prob: float,
    transaction_cost_bps: float,
    input_dim: int,
    class_weights: torch.Tensor,
    device: torch.device,
    args,
):
    model = PolicyNet(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, action_dim=len(ACTION_SPACE)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)

    best_state = None
    best_epoch = -1
    best_score = -1e18
    wait = 0
    history = []

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
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))

                pred = torch.argmax(logits, dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += int(len(yb))

        # economic selection on validation
        val_overlay, val_base, _ = run_overlay_backtest(
            df_steps=val_df_steps,
            horizon=horizon,
            base_min_prob=base_min_prob,
            transaction_cost_bps=transaction_cost_bps,
            policy_model=model,
            device=device,
            mode="policy",
        )
        val_excess_cumret = val_overlay["cumulative_return"] - val_base["cumulative_return"]
        val_excess_sharpe = val_overlay["sharpe"] - val_base["sharpe"]

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        val_acc = float(val_correct / max(1, val_total))

        # checkpoint score: economics first, loss second
        val_score = (
            args.selection_return_weight * val_excess_cumret
            + args.selection_sharpe_weight * val_excess_sharpe
            - args.selection_mdd_penalty * max(0.0, args.max_val_drawdown - val_overlay["max_drawdown"])
            - args.selection_exposure_penalty * max(0.0, args.min_avg_exposure - val_overlay["avg_action_exposure_proxy"])
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_action_acc": val_acc,
            "val_overlay_cumret": val_overlay["cumulative_return"],
            "val_base_cumret": val_base["cumulative_return"],
            "val_excess_cumret": val_excess_cumret,
            "val_overlay_sharpe": val_overlay["sharpe"],
            "val_base_sharpe": val_base["sharpe"],
            "val_excess_sharpe": val_excess_sharpe,
            "val_overlay_mdd": val_overlay["max_drawdown"],
            "val_overlay_exposure": val_overlay["avg_action_exposure_proxy"],
            "val_selection_score": val_score,
        })

        print(
            f"[policy] epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_action_acc={val_acc:.4f} | val_excess_cumret={val_excess_cumret:.4f} | "
            f"val_excess_sharpe={val_excess_sharpe:.4f} | val_mdd={val_overlay['max_drawdown']:.4f} | "
            f"score={val_score:.4f}"
        )

        improved = val_score > best_score + 1e-6
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= args.min_epochs and wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
    }
    return model, history_df, summary


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline policy overlay v2b (3-action, class-weighted, econ-selected).")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--feature_data_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--base_min_prob", type=float, default=0.62)
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--inactivity_penalty", type=float, default=0.002)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--min_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--class_weight_power", type=float, default=0.5)

    # validation economic selection
    parser.add_argument("--selection_return_weight", type=float, default=5.0)
    parser.add_argument("--selection_sharpe_weight", type=float, default=0.5)
    parser.add_argument("--selection_mdd_penalty", type=float, default=2.0)
    parser.add_argument("--selection_exposure_penalty", type=float, default=0.5)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)
    parser.add_argument("--min_avg_exposure", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_steps, used_state_cols, horizon = prepare_overlay_steps(
        predictions_path=args.predictions_path,
        feature_df_path=args.feature_data_path,
        base_min_prob=args.base_min_prob,
    )

    train_df = df_steps[df_steps["split"] == "train"].copy().reset_index(drop=True)
    val_df = df_steps[df_steps["split"] == "val"].copy().reset_index(drop=True)
    test_df = df_steps[df_steps["split"] == "test"].copy().reset_index(drop=True)

    print(f"Horizon inferred: {horizon}")
    print(f"Overlay train/val/test steps: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Using state columns: {used_state_cols}")

    train_state_matrix = np.stack(train_df["state_vec"].tolist())
    mean, std = fit_state_standardizer(train_state_matrix)

    for d in [train_df, val_df, test_df]:
        d["state_scaled"] = d["state_vec"].apply(lambda x: apply_state_standardizer(np.array([x]), mean, std)[0])

    train_labeled = build_pseudo_labels(
        df_steps=train_df,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        inactivity_penalty=args.inactivity_penalty,
    )
    val_labeled = build_pseudo_labels(
        df_steps=val_df,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        inactivity_penalty=args.inactivity_penalty,
    )

    X_train = np.stack(train_labeled["state_scaled"].tolist())
    y_train = train_labeled["best_action"].values.astype(np.int64)
    X_val = np.stack(val_labeled["state_scaled"].tolist())
    y_val = val_labeled["best_action"].values.astype(np.int64)

    print("Train pseudo-label distribution:")
    print(pd.Series(y_train).value_counts().sort_index().rename(index=ACTION_SPACE).to_string())
    print("Val pseudo-label distribution:")
    print(pd.Series(y_val).value_counts().sort_index().rename(index=ACTION_SPACE).to_string())

    class_weights = compute_class_weights(y_train, action_dim=len(ACTION_SPACE), power=args.class_weight_power)
    print("Class weights:")
    print({ACTION_SPACE[i]: float(class_weights[i].item()) for i in range(len(ACTION_SPACE))})

    train_ds = PolicyDataset(X_train, y_train)
    val_ds = PolicyDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    input_dim = X_train.shape[1]
    model, history_df, train_summary = train_policy_model(
        train_loader=train_loader,
        val_loader=val_loader,
        val_df_steps=val_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        input_dim=input_dim,
        class_weights=class_weights,
        device=device,
        args=args,
    )

    val_overlay, val_base, val_logs = run_overlay_backtest(
        df_steps=val_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        policy_model=model,
        device=device,
        mode="policy",
    )
    test_overlay, test_base, test_logs = run_overlay_backtest(
        df_steps=test_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        policy_model=model,
        device=device,
        mode="policy",
    )

    out_root = Path(args.out_dir) / f"{Path(args.predictions_path).stem}_offline_policy_overlay_v2b"
    out_root.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_root / "policy_overlay_best_model.pt")
    history_df.to_csv(out_root / "training_history.csv", index=False)
    train_labeled.to_csv(out_root / "train_pseudo_labels.csv", index=False)
    val_labeled.to_csv(out_root / "val_pseudo_labels.csv", index=False)
    val_logs.to_csv(out_root / "val_overlay_actions.csv", index=False)
    test_logs.to_csv(out_root / "test_overlay_actions.csv", index=False)

    summary = {
        "train_summary": train_summary,
        "horizon": int(horizon),
        "used_state_cols": used_state_cols,
        "state_dim": int(input_dim),
        "action_space": ACTION_SPACE,
        "base_strategy_definition": {
            "description": "full top-1 if top1_prob >= base_min_prob else cash",
            "base_min_prob": float(args.base_min_prob),
        },
        "class_weights": {ACTION_SPACE[i]: float(class_weights[i].item()) for i in range(len(ACTION_SPACE))},
        "val_overlay_metrics": val_overlay,
        "val_base_metrics": val_base,
        "val_excess_cumulative_return": float(val_overlay["cumulative_return"] - val_base["cumulative_return"]),
        "test_overlay_metrics": test_overlay,
        "test_base_metrics": test_base,
        "test_excess_cumulative_return": float(test_overlay["cumulative_return"] - test_base["cumulative_return"]),
        "notes": [
            "This is a 3-action offline policy-learning overlay.",
            "Checkpoint selection is based on validation economic performance, not validation loss alone.",
            "Use overlay vs base comparison as the dissertation result, not raw action accuracy.",
        ],
    }
    with open(out_root / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["horizon"] = int(horizon)
    run_config["used_state_cols"] = used_state_cols
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\nFinished offline policy overlay v2b.")
    print(f"Saved outputs to: {out_root}")
    print("\nValidation overlay metrics:")
    print(json.dumps(val_overlay, indent=2, ensure_ascii=False))
    print("\nValidation base metrics:")
    print(json.dumps(val_base, indent=2, ensure_ascii=False))
    print("\nValidation excess cumulative return:")
    print(val_overlay["cumulative_return"] - val_base["cumulative_return"])
    print("\nTest overlay metrics:")
    print(json.dumps(test_overlay, indent=2, ensure_ascii=False))
    print("\nTest base metrics:")
    print(json.dumps(test_base, indent=2, ensure_ascii=False))
    print("\nTest excess cumulative return:")
    print(test_overlay["cumulative_return"] - test_base["cumulative_return"])


if __name__ == "__main__":
    main()

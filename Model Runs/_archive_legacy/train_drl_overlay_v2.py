
import os
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# DRL overlay v2
#
# Core fix compared with v1:
# --------------------------
# The previous DRL overlay often learned the trivial all-cash policy.
# Why?
#   - cash always gives reward 0
#   - risky actions often give noisy / slightly negative reward
#   - validation selection did not explicitly discourage inactivity
#
# Main changes in v2:
# -------------------
# 1) Reward is defined RELATIVE TO THE BASE STRATEGY
#    reward = overlay_net_return - base_net_return
#             - drawdown_penalty * current_drawdown_penalty
#             - inactivity_penalty if cash while base signal active
#
# 2) We compute and save the base strategy metrics directly
#    so you can compare:
#       base predictor strategy vs DRL overlay strategy
#
# 3) Validation model selection uses:
#       excess cumulative return over base
#       + overlay Sharpe
#    under drawdown constraint
#
# Recommended use:
# ----------------
# Use H5 LSTM v4 predictions as the base predictor.
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
# Helpers
# -----------------------------
def infer_horizon_from_return_col(return_col: str) -> int:
    if return_col.endswith("1d"):
        return 1
    if return_col.endswith("5d"):
        return 5
    raise ValueError(f"Cannot infer horizon from return column: {return_col}")


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

    drop_cols = {"split"}
    feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c in pred_df.columns and c not in {"date", "stock"}], errors="ignore")

    merged = pred_df.merge(feat_df, on=["date", "stock"], how="left")
    return merged


def pick_available_state_cols(df: pd.DataFrame, requested_cols: List[str]) -> List[str]:
    return [c for c in requested_cols if c in df.columns]


def build_rebalance_dates(split_df: pd.DataFrame, horizon: int) -> List[pd.Timestamp]:
    unique_dates = sorted(pd.to_datetime(split_df["date"]).drop_duplicates().tolist())
    return unique_dates[::horizon]


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

    optional_cols = [c for c in ["return_21d_cs_z", "rsi_14_cs_z", "macd_hist_pct_cs_z"] if c in g.columns]
    for c in optional_cols:
        vals = pd.to_numeric(g[c], errors="coerce").fillna(0.0).values.astype(float)
        state[f"xs_mean_{c}"] = float(np.mean(vals))
        state[f"xs_std_{c}"] = float(np.std(vals))

    state_vec = np.array(list(state.values()), dtype=np.float32)
    meta = {
        "date": pd.Timestamp(top1["date"]),
        "split": str(top1["split"]),
        "sorted_stocks": g["stock"].tolist(),
        "sorted_probs": g["pred_prob"].astype(float).tolist(),
        "sorted_future_returns": g["future_return"].astype(float).tolist(),
    }
    return state_vec, meta


def prepare_overlay_dataset(
    predictions_path: str,
    feature_data_path: Optional[str],
    base_min_prob: float,
) -> Tuple[pd.DataFrame, List[str], int]:
    pred_df = load_predictions(predictions_path)
    merged = maybe_merge_features(pred_df, feature_data_path)

    horizon_guess = 5 if "5" in Path(predictions_path).stem else None
    if feature_data_path is not None:
        raw_df = pd.read_csv(feature_data_path)
        future_cols = [c for c in raw_df.columns if c.startswith("future_return_")]
        if len(future_cols) == 1:
            horizon_guess = infer_horizon_from_return_col(future_cols[0])

    if horizon_guess is None:
        horizon_guess = 5

    available_state_cols = pick_available_state_cols(merged, DEFAULT_STATE_COLS)

    rows = []
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name].copy()
        if split_df.empty:
            continue

        rebalance_dates = build_rebalance_dates(split_df, horizon=horizon_guess)
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

    overlay_df = pd.DataFrame(rows)
    return overlay_df, available_state_cols, horizon_guess


def fit_state_standardizer(train_state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_state_matrix.mean(axis=0)
    std = train_state_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_state_standardizer(state_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state_matrix - mean) / std


# -----------------------------
# Actions
# -----------------------------
ACTION_SPACE = {
    0: "cash",
    1: "top1_half",
    2: "top1_full",
    3: "top2_equal",
}


def action_to_weights(sorted_stocks: List[str], action: int) -> Dict[str, float]:
    if action == 0:
        return {}
    elif action == 1:
        return {sorted_stocks[0]: 0.5}
    elif action == 2:
        return {sorted_stocks[0]: 1.0}
    elif action == 3:
        if len(sorted_stocks) == 1:
            return {sorted_stocks[0]: 1.0}
        return {sorted_stocks[0]: 0.5, sorted_stocks[1]: 0.5}
    else:
        raise ValueError(f"Unknown action: {action}")


def turnover(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    names = set(prev_w.keys()) | set(new_w.keys())
    return float(sum(abs(prev_w.get(n, 0.0) - new_w.get(n, 0.0)) for n in names))


def portfolio_return_from_weights(sorted_stocks: List[str], sorted_future_returns: List[float], weights: Dict[str, float]) -> float:
    ret_map = {s: r for s, r in zip(sorted_stocks, sorted_future_returns)}
    return float(sum(weights.get(s, 0.0) * ret_map.get(s, 0.0) for s in weights.keys()))


def base_strategy_weights(sorted_stocks: List[str], sorted_probs: List[float], base_min_prob: float) -> Dict[str, float]:
    # Base strategy = validated H5 LSTM style: full top-1 only when top1 prob passes threshold, else cash.
    if len(sorted_stocks) == 0:
        return {}
    if sorted_probs[0] >= base_min_prob:
        return {sorted_stocks[0]: 1.0}
    return {}


# -----------------------------
# Environment
# -----------------------------
class OverlayEnv:
    def __init__(
        self,
        df_steps: pd.DataFrame,
        horizon: int,
        base_min_prob: float,
        transaction_cost_bps: float = 10.0,
        drawdown_penalty: float = 0.10,
        inactivity_penalty: float = 0.002,
    ):
        self.df_steps = df_steps.reset_index(drop=True).copy()
        self.horizon = int(horizon)
        self.base_min_prob = float(base_min_prob)
        self.tc = transaction_cost_bps / 10000.0
        self.drawdown_penalty = float(drawdown_penalty)
        self.inactivity_penalty = float(inactivity_penalty)
        self.reset()

    def reset(self):
        self.i = 0

        self.overlay_equity = 1.0
        self.overlay_peak = 1.0
        self.overlay_prev_weights = {}
        self.overlay_returns = []
        self.overlay_actions = []
        self.overlay_equity_curve = [1.0]

        self.base_equity = 1.0
        self.base_prev_weights = {}
        self.base_returns = []
        self.base_equity_curve = [1.0]

        return self._get_state()

    def _get_state(self):
        row = self.df_steps.iloc[self.i]
        return row["state_scaled"]

    def step(self, action: int):
        row = self.df_steps.iloc[self.i]

        sorted_stocks = row["sorted_stocks"]
        sorted_probs = row["sorted_probs"]
        sorted_future_returns = row["sorted_future_returns"]
        base_signal_active = bool(sorted_probs[0] >= self.base_min_prob) if len(sorted_probs) > 0 else False

        overlay_new_weights = action_to_weights(sorted_stocks, action)
        overlay_turnover = turnover(self.overlay_prev_weights, overlay_new_weights)
        overlay_gross = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, overlay_new_weights)
        overlay_net = overlay_gross - self.tc * overlay_turnover

        base_new_weights = base_strategy_weights(sorted_stocks, sorted_probs, self.base_min_prob)
        base_turn = turnover(self.base_prev_weights, base_new_weights)
        base_gross = portfolio_return_from_weights(sorted_stocks, sorted_future_returns, base_new_weights)
        base_net = base_gross - self.tc * base_turn

        # update overlay path
        self.overlay_equity *= (1.0 + overlay_net)
        self.overlay_peak = max(self.overlay_peak, self.overlay_equity)
        current_dd = self.overlay_equity / max(self.overlay_peak, 1e-12) - 1.0

        # update base path
        self.base_equity *= (1.0 + base_net)

        # reward = excess performance over base, plus activity shaping
        reward = overlay_net - base_net
        reward -= self.drawdown_penalty * max(0.0, -current_dd)
        if action == 0 and base_signal_active:
            reward -= self.inactivity_penalty

        self.overlay_prev_weights = overlay_new_weights
        self.base_prev_weights = base_new_weights

        self.overlay_returns.append(overlay_net)
        self.base_returns.append(base_net)
        self.overlay_actions.append(action)
        self.overlay_equity_curve.append(self.overlay_equity)
        self.base_equity_curve.append(self.base_equity)

        done = self.i >= len(self.df_steps) - 1
        info = {
            "date": str(pd.Timestamp(row["date"]).date()),
            "action": int(action),
            "action_name": ACTION_SPACE[int(action)],
            "base_signal_active": int(base_signal_active),
            "overlay_gross_return": float(overlay_gross),
            "overlay_net_return": float(overlay_net),
            "overlay_turnover": float(overlay_turnover),
            "base_gross_return": float(base_gross),
            "base_net_return": float(base_net),
            "base_turnover": float(base_turn),
            "excess_return_vs_base": float(overlay_net - base_net),
            "overlay_drawdown": float(current_dd),
            "overlay_equity": float(self.overlay_equity),
            "base_equity": float(self.base_equity),
        }

        if not done:
            self.i += 1
            next_state = self._get_state()
        else:
            next_state = np.zeros_like(row["state_scaled"], dtype=np.float32)

        return next_state, float(reward), bool(done), info

    def evaluate_greedy(self, policy_net: nn.Module, device: torch.device):
        state = self.reset()
        done = False
        logs = []

        while not done:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy_net(s)
                action = int(torch.argmax(q, dim=1).item())

            next_state, reward, done, info = self.step(action)
            info["reward"] = float(reward)
            logs.append(info)
            state = next_state

        overlay_metrics = summarize_episode(
            period_returns=self.overlay_returns,
            equity_curve=self.overlay_equity_curve,
            horizon=self.horizon,
            logs=logs,
            turnover_key="overlay_turnover",
        )
        base_metrics = summarize_episode(
            period_returns=self.base_returns,
            equity_curve=self.base_equity_curve,
            horizon=self.horizon,
            logs=logs,
            turnover_key="base_turnover",
        )
        return overlay_metrics, base_metrics, pd.DataFrame(logs)


def summarize_episode(period_returns: List[float], equity_curve: List[float], horizon: int, logs: List[Dict], turnover_key: str) -> Dict[str, float]:
    cumulative_return = float(equity_curve[-1] - 1.0)
    ann_return = annualized_return_from_equity(equity_curve[-1], total_days=max(1, len(period_returns) * horizon))
    sharpe = sharpe_from_returns(period_returns, horizon=horizon)
    mdd = max_drawdown_from_equity(equity_curve)

    turns = [x[turnover_key] for x in logs] if logs else []
    actions = [x["action"] for x in logs] if logs else []
    exposures = []
    for a in actions:
        if a == 0:
            exposures.append(0.0)
        elif a == 1:
            exposures.append(0.5)
        elif a == 2:
            exposures.append(1.0)
        elif a == 3:
            exposures.append(2.0)
        else:
            exposures.append(0.0)

    return {
        "periods": int(len(period_returns)),
        "cumulative_return": cumulative_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": float(np.mean(np.array(period_returns) > 0)) if len(period_returns) > 0 else 0.0,
        "avg_turnover": float(np.mean(turns)) if turns else 0.0,
        "avg_action_exposure_proxy": float(np.mean(exposures)) if exposures else 0.0,
    }


# -----------------------------
# DQN
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


def train_dqn_overlay(train_df: pd.DataFrame, val_df: pd.DataFrame, horizon: int, device: torch.device, args):
    state_dim = train_df.iloc[0]["state_scaled"].shape[0]
    action_dim = len(ACTION_SPACE)

    q_net = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    target_net = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(capacity=args.replay_size)

    train_env = OverlayEnv(
        df_steps=train_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        drawdown_penalty=args.drawdown_penalty,
        inactivity_penalty=args.inactivity_penalty,
    )
    val_env = OverlayEnv(
        df_steps=val_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        drawdown_penalty=args.drawdown_penalty,
        inactivity_penalty=args.inactivity_penalty,
    )

    epsilon = args.eps_start
    best_state = None
    best_episode = -1
    best_val_score = -1e18
    history_rows = []

    for episode in range(1, args.episodes + 1):
        state = train_env.reset()
        done = False
        td_losses = []

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = int(torch.argmax(q_net(s_t), dim=1).item())

            next_state, reward, done, _ = train_env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay) >= args.batch_size:
                s, a, r, ns, d = replay.sample(args.batch_size)

                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)

                q_values = q_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    next_actions = q_net(ns_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(ns_t).gather(1, next_actions)
                    target = r_t + args.gamma * next_q * (1.0 - d_t)

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
                td_losses.append(float(loss.item()))

        if episode % args.target_update_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(args.eps_end, epsilon * args.eps_decay)

        val_overlay, val_base, _ = val_env.evaluate_greedy(q_net, device=device)
        val_excess_cumret = val_overlay["cumulative_return"] - val_base["cumulative_return"]
        constraints_ok = val_overlay["max_drawdown"] >= args.max_val_drawdown

        # selection score favors improving over base, not all-cash inactivity
        val_score = (
            5.0 * val_excess_cumret
            + 0.50 * val_overlay["sharpe"]
            - 2.0 * max(0.0, args.max_val_drawdown - val_overlay["max_drawdown"])
            - 0.50 * max(0.0, 0.05 - val_overlay["avg_action_exposure_proxy"])
        )

        if constraints_ok and val_score > best_val_score + 1e-6:
            best_val_score = val_score
            best_episode = episode
            best_state = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}

        history_rows.append({
            "episode": int(episode),
            "epsilon": float(epsilon),
            "avg_td_loss": float(np.mean(td_losses)) if td_losses else np.nan,
            "train_overlay_cumulative_return": float(train_env.overlay_equity_curve[-1] - 1.0),
            "train_base_cumulative_return": float(train_env.base_equity_curve[-1] - 1.0),
            "val_overlay_sharpe": float(val_overlay["sharpe"]),
            "val_overlay_cumulative_return": float(val_overlay["cumulative_return"]),
            "val_overlay_max_drawdown": float(val_overlay["max_drawdown"]),
            "val_base_sharpe": float(val_base["sharpe"]),
            "val_base_cumulative_return": float(val_base["cumulative_return"]),
            "val_excess_cumret": float(val_excess_cumret),
            "val_score_for_selection": float(val_score),
            "constraints_ok": bool(constraints_ok),
        })

        print(
            f"[episode {episode:03d}] eps={epsilon:.4f} | "
            f"val_overlay_sharpe={val_overlay['sharpe']:.4f} | "
            f"val_overlay_cumret={val_overlay['cumulative_return']:.4f} | "
            f"val_base_cumret={val_base['cumulative_return']:.4f} | "
            f"val_excess_cumret={val_excess_cumret:.4f} | "
            f"val_mdd={val_overlay['max_drawdown']:.4f} | "
            f"selection_score={val_score:.4f}"
        )

    if best_state is None:
        # fallback: keep latest network if no constrained best was found
        best_state = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}
        best_episode = args.episodes

    q_net.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    summary = {
        "best_episode": int(best_episode),
        "best_val_score": float(best_val_score),
    }
    return q_net, history_df, summary


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DRL overlay v2 on top of a validated base predictor.")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to base predictor predictions_all_splits.csv (recommended: H5 LSTM v4).")
    parser.add_argument("--feature_data_path", type=str, default=None,
                        help="Optional path to experiment dataset, e.g. main_experiment_h5.csv.")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--base_min_prob", type=float, default=0.62,
                        help="Base strategy threshold from the validated H5 LSTM system.")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--drawdown_penalty", type=float, default=0.10)
    parser.add_argument("--inactivity_penalty", type=float, default=0.002)
    parser.add_argument("--max_val_drawdown", type=float, default=-0.25)

    parser.add_argument("--episodes", type=int, default=140)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_size", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--target_update_every", type=int, default=5)

    parser.add_argument("--eps_start", type=float, default=1.00)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.97)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    overlay_df, used_state_cols, horizon = prepare_overlay_dataset(
        predictions_path=args.predictions_path,
        feature_data_path=args.feature_data_path,
        base_min_prob=args.base_min_prob,
    )

    if overlay_df.empty:
        raise ValueError("No overlay steps created. Check input predictions file.")

    train_mask = overlay_df["split"] == "train"
    val_mask = overlay_df["split"] == "val"
    test_mask = overlay_df["split"] == "test"

    train_states = np.stack(overlay_df.loc[train_mask, "state_vec"].tolist())
    mean, std = fit_state_standardizer(train_states)
    overlay_df["state_scaled"] = overlay_df["state_vec"].apply(lambda x: apply_state_standardizer(np.array([x]), mean, std)[0])

    train_df = overlay_df.loc[train_mask].reset_index(drop=True)
    val_df = overlay_df.loc[val_mask].reset_index(drop=True)
    test_df = overlay_df.loc[test_mask].reset_index(drop=True)

    out_root = Path(args.out_dir) / f"{Path(args.predictions_path).stem}_drl_overlay_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Horizon inferred: {horizon}")
    print(f"Overlay train/val/test steps: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"State dimension: {len(train_df.iloc[0]['state_scaled'])}")
    print(f"Using state columns from feature merge: {used_state_cols}")
    print(f"Action space: {ACTION_SPACE}")

    q_net, history_df, train_summary = train_dqn_overlay(
        train_df=train_df,
        val_df=val_df,
        horizon=horizon,
        device=device,
        args=args,
    )

    val_env = OverlayEnv(
        df_steps=val_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        drawdown_penalty=args.drawdown_penalty,
        inactivity_penalty=args.inactivity_penalty,
    )
    test_env = OverlayEnv(
        df_steps=test_df,
        horizon=horizon,
        base_min_prob=args.base_min_prob,
        transaction_cost_bps=args.transaction_cost_bps,
        drawdown_penalty=args.drawdown_penalty,
        inactivity_penalty=args.inactivity_penalty,
    )

    val_overlay, val_base, val_logs = val_env.evaluate_greedy(q_net, device=device)
    test_overlay, test_base, test_logs = test_env.evaluate_greedy(q_net, device=device)

    torch.save(q_net.state_dict(), out_root / "drl_overlay_best_model.pt")
    history_df.to_csv(out_root / "training_history.csv", index=False)
    val_logs.to_csv(out_root / "val_episode_actions.csv", index=False)
    test_logs.to_csv(out_root / "test_episode_actions.csv", index=False)

    summary = {
        "train_summary": train_summary,
        "horizon": int(horizon),
        "used_state_cols": used_state_cols,
        "state_dim": int(len(train_df.iloc[0]["state_scaled"])),
        "action_space": ACTION_SPACE,
        "base_strategy_definition": {
            "description": "full top-1 if top1_prob >= base_min_prob else cash",
            "base_min_prob": float(args.base_min_prob),
        },
        "val_overlay_metrics": val_overlay,
        "val_base_metrics": val_base,
        "val_excess_cumulative_return": float(val_overlay["cumulative_return"] - val_base["cumulative_return"]),
        "test_overlay_metrics": test_overlay,
        "test_base_metrics": test_base,
        "test_excess_cumulative_return": float(test_overlay["cumulative_return"] - test_base["cumulative_return"]),
        "notes": [
            "This DRL overlay refines execution on top of the base predictor.",
            "The reward is defined relative to the base strategy, which prevents trivial all-cash dominance.",
            "Compare overlay metrics against base metrics, not against zero.",
        ],
    }
    with open(out_root / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_config = vars(args).copy()
    run_config["horizon"] = int(horizon)
    run_config["used_state_cols"] = used_state_cols
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\nFinished DRL overlay v2 training.")
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

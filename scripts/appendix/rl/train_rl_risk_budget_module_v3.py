from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Deque, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from risk_budget_module_common_v3 import (
    MAIN_H5_DATA,
    CANDD_CONFIG,
    ensure_dir,
    normalize_predictions,
    parse_action_levels,
    resolve_default_final_system_manifest_path,
    safe_read_csv,
    safe_read_json,
    save_df_with_dates,
    timestamp_tag,
    RiskBudgetEpisodeCoreV3,
    run_fixed_post_module_backtest,
    summarize_risk_budget_episode,
)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state.copy(), int(action), float(reward), next_state.copy(), float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_action(
    q_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    allowed_actions: List[int],
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return random.choice(allowed_actions)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = q_net(s).squeeze(0).cpu().numpy()
        masked = np.full_like(q, -1e18, dtype=float)
        for a in allowed_actions:
            masked[a] = q[a]
        return int(np.argmax(masked))


def evaluate_greedy_policy(env: RiskBudgetEpisodeCoreV3, q_net: QNetwork, device: torch.device, label: str):
    state = env.reset()
    done = False
    while not done:
        allowed = env.allowed_action_indices()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = q_net(s).squeeze(0).cpu().numpy()
            masked = np.full_like(q, -1e18, dtype=float)
            for a in allowed:
                masked[a] = q[a]
            action = int(np.argmax(masked))
        state, _, done, _ = env.step(action)
    return summarize_risk_budget_episode(env.records, env.horizon, label)


def build_checkpoint_payload(
    q_net,
    hidden_dim,
    train_env,
    action_levels,
    args_dict,
    final_system_manifest_path,
    predictions_path,
    feature_data_path,
    best_episode,
    best_selection_score,
    val_relative,
    val_module_metrics,
    val_base_metrics,
):
    return {
        "model_state_dict": copy.deepcopy(q_net.state_dict()),
        "state_dim": int(train_env.state_dim),
        "num_actions": int(train_env.num_actions),
        "hidden_dim": int(hidden_dim),
        "action_levels": list(action_levels),
        "base_config": copy.deepcopy(train_env.base_config),
        "train_args": args_dict,
        "source_final_system_manifest_path": str(final_system_manifest_path.resolve()),
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "best_episode": int(best_episode),
        "best_selection_score": float(best_selection_score),
        "best_val_relative": copy.deepcopy(val_relative),
        "best_val_module_metrics": copy.deepcopy(val_module_metrics),
        "best_val_base_metrics": copy.deepcopy(val_base_metrics),
    }


def main():
    ap = argparse.ArgumentParser(description="Train optional RL risk budget post-module v3 on top of fixed candD.")
    ap.add_argument("--final_system_manifest_path", type=str, default="")
    ap.add_argument("--predictions_path", type=str, default="")
    ap.add_argument("--feature_data_path", type=str, default=str(MAIN_H5_DATA))
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--buffer_size", type=int, default=8000)
    ap.add_argument("--min_buffer_to_learn", type=int, default=192)
    ap.add_argument("--target_update_every", type=int, default=5)
    ap.add_argument("--eps_start", type=float, default=0.80)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=float, default=0.985)
    ap.add_argument("--reward_scale", type=float, default=100.0)
    ap.add_argument("--min_select_episode", type=int, default=8)

    # mixed-scheme defaults
    ap.add_argument("--action_levels", type=str, default="0.6,1.0")
    ap.add_argument("--smooth_window", type=int, default=3)
    ap.add_argument("--min_hold_periods", type=int, default=2)
    ap.add_argument("--switch_penalty_raw", type=float, default=0.005)
    ap.add_argument("--cash_return_per_period", type=float, default=0.0)

    ap.add_argument("--downside_protection_coef", type=float, default=0.30)
    ap.add_argument("--drawdown_breach_coef", type=float, default=0.04)
    ap.add_argument("--drawdown_tolerance", type=float, default=0.20)
    ap.add_argument("--vol_penalty_coef", type=float, default=0.02)
    ap.add_argument("--vol_tolerance", type=float, default=0.025)
    ap.add_argument("--nonstress_cash_penalty_raw", type=float, default=0.002)

    ap.add_argument("--vix_warn", type=float, default=0.90)
    ap.add_argument("--vix_high", type=float, default=1.30)
    ap.add_argument("--credit_warn", type=float, default=0.90)
    ap.add_argument("--credit_high", type=float, default=1.30)
    ap.add_argument("--mkt_dc_warn", type=float, default=-0.12)
    ap.add_argument("--mkt_dc_high", type=float, default=-0.25)
    ap.add_argument("--mkt_ret_warn", type=float, default=-0.025)
    ap.add_argument("--mkt_ret_high", type=float, default=-0.05)
    args = ap.parse_args()

    set_seed(args.seed)

    final_system_manifest_path = Path(args.final_system_manifest_path) if args.final_system_manifest_path else resolve_default_final_system_manifest_path()
    final_manifest = safe_read_json(final_system_manifest_path)
    predictions_path = Path(args.predictions_path) if args.predictions_path else Path(final_manifest.get("source_predictions_path", ""))
    feature_data_path = Path(args.feature_data_path)

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)

    horizon = int(final_manifest.get("horizon", 5))
    tc_bps = float(final_manifest.get("transaction_cost_bps", 10.0))
    action_levels = parse_action_levels(args.action_levels)

    train_df = pred[pred["split"] == "train"].copy()
    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Need non-empty train/val/test splits.")

    common_kwargs = dict(
        horizon=horizon,
        transaction_cost_bps=tc_bps,
        base_config=CANDD_CONFIG,
        action_levels=action_levels,
        smooth_window=int(args.smooth_window),
        min_hold_periods=int(args.min_hold_periods),
        switch_penalty_raw=float(args.switch_penalty_raw),
        cash_return_per_period=float(args.cash_return_per_period),
        downside_protection_coef=float(args.downside_protection_coef),
        drawdown_breach_coef=float(args.drawdown_breach_coef),
        drawdown_tolerance=float(args.drawdown_tolerance),
        vol_penalty_coef=float(args.vol_penalty_coef),
        vol_tolerance=float(args.vol_tolerance),
        nonstress_cash_penalty_raw=float(args.nonstress_cash_penalty_raw),
        vix_warn=float(args.vix_warn),
        vix_high=float(args.vix_high),
        credit_warn=float(args.credit_warn),
        credit_high=float(args.credit_high),
        mkt_dc_warn=float(args.mkt_dc_warn),
        mkt_dc_high=float(args.mkt_dc_high),
        mkt_ret_warn=float(args.mkt_ret_warn),
        mkt_ret_high=float(args.mkt_ret_high),
        default_start_exposure=max(action_levels),
    )

    train_env = RiskBudgetEpisodeCoreV3(pred_df=train_df, **common_kwargs)
    val_env = RiskBudgetEpisodeCoreV3(pred_df=val_df, **common_kwargs)
    test_env = RiskBudgetEpisodeCoreV3(pred_df=test_df, **common_kwargs)

    val_base_metrics, _, _, _ = run_fixed_post_module_backtest(
        val_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=1.0, label="base_candD_full_val"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(train_env.state_dim, train_env.num_actions, hidden_dim=args.hidden_dim).to(device)
    target_net = QNetwork(train_env.state_dim, train_env.num_actions, hidden_dim=args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = Adam(q_net.parameters(), lr=args.lr, weight_decay=1e-5)
    replay = ReplayBuffer(args.buffer_size)

    base_out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (predictions_path.resolve().parents[3] / "execution" / "risk_budget_module_v3"))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_root / "best_risk_budget_module_v3.pt"

    epsilon = float(args.eps_start)
    best_val_score = -1e18
    history_rows = []

    for ep in range(1, int(args.episodes) + 1):
        state = train_env.reset()
        done = False
        train_reward_raw = 0.0
        train_reward_scaled = 0.0
        steps = 0

        while not done:
            allowed = train_env.allowed_action_indices()
            action = select_action(q_net, state, epsilon, allowed, device)
            next_state, raw_reward, done, _ = train_env.step(action)
            scaled_reward = float(raw_reward) * float(args.reward_scale)

            replay.push(state, action, scaled_reward, next_state, done)
            state = next_state
            train_reward_raw += float(raw_reward)
            train_reward_scaled += float(scaled_reward)
            steps += 1

            if len(replay) >= max(args.batch_size, args.min_buffer_to_learn):
                states, actions, rewards, next_states, dones = replay.sample(args.batch_size)
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_vals = q_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(dim=1, keepdim=True).values
                    target = rewards_t + (1.0 - dones_t) * float(args.gamma) * next_q

                loss = nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

        if ep % int(args.target_update_every) == 0:
            target_net.load_state_dict(q_net.state_dict())

        val_module_metrics, val_base_metrics_reloaded, val_rel, _ = evaluate_greedy_policy(
            val_env, q_net, device, label="risk_budget_module_v3_val"
        )

        cumret_gap = float(val_module_metrics["cumulative_return"]) - float(val_base_metrics["cumulative_return"])
        sharpe_gap = float(val_module_metrics["sharpe"]) - float(val_base_metrics["sharpe"])
        mdd_improve = abs(float(val_base_metrics["max_drawdown"])) - abs(float(val_module_metrics["max_drawdown"]))
        exposure_shortfall = max(0.0, 0.92 - float(val_module_metrics["avg_exposure"]))
        switch_pen = float(val_module_metrics.get("switch_rate", 0.0))

        score = (
            1.15 * sharpe_gap
            + 0.50 * mdd_improve
            + 0.30 * cumret_gap
            - 0.25 * exposure_shortfall
            - 0.12 * switch_pen
        )

        history_rows.append({
            "episode": ep,
            "epsilon": epsilon,
            "train_reward_sum_raw": train_reward_raw,
            "train_reward_sum_scaled": train_reward_scaled,
            "train_steps": steps,
            "val_cumret": float(val_module_metrics["cumulative_return"]),
            "val_sharpe": float(val_module_metrics["sharpe"]),
            "val_mdd": float(val_module_metrics["max_drawdown"]),
            "val_avg_exposure": float(val_module_metrics["avg_exposure"]),
            "val_avg_cash_weight": float(val_module_metrics.get("avg_cash_weight", 0.0)),
            "val_switch_rate": float(val_module_metrics.get("switch_rate", 0.0)),
            "val_any_stress_rate": float(val_module_metrics.get("any_stress_rate", 0.0)),
            "val_high_stress_rate": float(val_module_metrics.get("high_stress_rate", 0.0)),
            "val_cumret_gap_vs_base": float(cumret_gap),
            "val_sharpe_gap_vs_base": float(sharpe_gap),
            "val_mdd_improvement_vs_base": float(mdd_improve),
            "selection_score": float(score),
        })

        if ep >= int(args.min_select_episode) and score > best_val_score:
            best_val_score = score
            payload = build_checkpoint_payload(
                q_net=q_net,
                hidden_dim=int(args.hidden_dim),
                train_env=train_env,
                action_levels=action_levels,
                args_dict=vars(args).copy(),
                final_system_manifest_path=final_system_manifest_path,
                predictions_path=predictions_path,
                feature_data_path=feature_data_path,
                best_episode=ep,
                best_selection_score=float(score),
                val_relative=val_rel,
                val_module_metrics=val_module_metrics,
                val_base_metrics=val_base_metrics_reloaded,
            )
            torch.save(payload, best_ckpt_path)

        epsilon = max(float(args.eps_end), epsilon * float(args.eps_decay))
        print(
            f"[Episode {ep:03d}] epsilon={epsilon:.4f} "
            f"val_sharpe={val_module_metrics['sharpe']:.4f} "
            f"val_mdd={val_module_metrics['max_drawdown']:.4f} "
            f"val_exp={val_module_metrics['avg_exposure']:.3f} "
            f"score={score:.6f}"
        )

    if not best_ckpt_path.exists():
        # fallback: use last model if all episodes were before min_select_episode
        payload = build_checkpoint_payload(
            q_net=q_net,
            hidden_dim=int(args.hidden_dim),
            train_env=train_env,
            action_levels=action_levels,
            args_dict=vars(args).copy(),
            final_system_manifest_path=final_system_manifest_path,
            predictions_path=predictions_path,
            feature_data_path=feature_data_path,
            best_episode=int(args.episodes),
            best_selection_score=float(history_rows[-1]["selection_score"]),
            val_relative=val_rel,
            val_module_metrics=val_module_metrics,
            val_base_metrics=val_base_metrics_reloaded,
        )
        torch.save(payload, best_ckpt_path)

    saved_best = torch.load(best_ckpt_path, map_location=device)
    q_net.load_state_dict(saved_best["model_state_dict"])

    val_module_metrics, val_base_metrics_reloaded, val_rel, val_actions_df = evaluate_greedy_policy(
        val_env, q_net, device, label="risk_budget_module_v3_val"
    )
    test_module_metrics, test_base_metrics, test_rel, test_actions_df = evaluate_greedy_policy(
        test_env, q_net, device, label="risk_budget_module_v3_test"
    )

    pd.DataFrame(history_rows).to_csv(out_root / "training_history.csv", index=False)
    save_df_with_dates(val_actions_df, out_root / "val_risk_budget_actions.csv")
    save_df_with_dates(test_actions_df, out_root / "test_risk_budget_actions.csv")

    summary = {
        "best_episode": int(saved_best["best_episode"]),
        "best_selection_score": float(saved_best["best_selection_score"]),
        "best_val_relative_from_checkpoint": saved_best["best_val_relative"],
        "best_val_module_metrics_from_checkpoint": saved_best["best_val_module_metrics"],
        "best_val_base_metrics_from_checkpoint": saved_best["best_val_base_metrics"],
        "val_relative_reloaded": val_rel,
        "test_relative": test_rel,
        "val_module_metrics_reloaded": val_module_metrics,
        "test_module_metrics": test_module_metrics,
        "val_base_metrics_reloaded": val_base_metrics_reloaded,
        "test_base_metrics": test_base_metrics,
        "action_levels": action_levels,
        "base_config": CANDD_CONFIG,
    }
    with open(out_root / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Risk budget module v3 saved to: {out_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from risk_budget_module_common_v3 import (
    MAIN_H5_DATA,
    CANDD_CONFIG,
    ensure_dir,
    normalize_predictions,
    resolve_default_final_system_manifest_path,
    safe_read_csv,
    safe_read_json,
    save_df_with_dates,
    timestamp_tag,
    RiskBudgetEpisodeCoreV3,
    run_fixed_post_module_backtest,
)
from train_rl_risk_budget_module_v3 import QNetwork, evaluate_greedy_policy


def main():
    ap = argparse.ArgumentParser(description="Run switchable post-module backtest: base candD or candD + RL risk budget module v3.")
    ap.add_argument("--enable_module", type=int, default=1, help="1=enable RL risk budget module, 0=run base candD only")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--final_system_manifest_path", type=str, default="")
    ap.add_argument("--predictions_path", type=str, default="")
    ap.add_argument("--feature_data_path", type=str, default=str(MAIN_H5_DATA))
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--run_name", type=str, default="")
    args = ap.parse_args()

    final_system_manifest_path = Path(args.final_system_manifest_path) if args.final_system_manifest_path else resolve_default_final_system_manifest_path()
    final_manifest = safe_read_json(final_system_manifest_path)

    predictions_path = Path(args.predictions_path) if args.predictions_path else Path(final_manifest.get("source_predictions_path", ""))
    feature_data_path = Path(args.feature_data_path)

    pred = safe_read_csv(predictions_path)
    feat = safe_read_csv(feature_data_path)
    pred = normalize_predictions(pred, feat)

    split_df = pred[pred["split"] == args.split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={args.split}")

    horizon = int(final_manifest.get("horizon", 5))
    tc_bps = float(final_manifest.get("transaction_cost_bps", 10.0))

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (predictions_path.resolve().parents[3] / "execution" / "risk_budget_switchable_runs_v3"))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    if int(args.enable_module) == 0:
        metrics, _, _, actions = run_fixed_post_module_backtest(
            split_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=1.0, label=f"base_candD_only_{args.split}"
        )
        save_df_with_dates(actions, out_root / f"{args.split}_base_candD_actions.csv")
        summary = {
            "mode": "base_only",
            "split": args.split,
            "metrics": metrics,
            "module_enabled": False,
        }
        with open(out_root / "switchable_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        default_root = predictions_path.resolve().parents[3] / "execution" / "risk_budget_module_v3"
        xs = sorted(default_root.glob("run_*/best_risk_budget_module_v3.pt"))
        if not xs:
            raise FileNotFoundError("Could not auto-find best_risk_budget_module_v3.pt")
        model_path = xs[-1]

    checkpoint = torch.load(model_path, map_location="cpu")
    hidden_dim = int(checkpoint["hidden_dim"])
    state_dim = int(checkpoint["state_dim"])
    num_actions = int(checkpoint["num_actions"])
    action_levels = list(checkpoint["action_levels"])
    train_args = checkpoint.get("train_args", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(state_dim, num_actions, hidden_dim=hidden_dim).to(device)
    q_net.load_state_dict(checkpoint["model_state_dict"])
    q_net.eval()

    env = RiskBudgetEpisodeCoreV3(
        pred_df=split_df,
        horizon=horizon,
        transaction_cost_bps=tc_bps,
        base_config=CANDD_CONFIG,
        action_levels=action_levels,
        smooth_window=int(train_args.get("smooth_window", 3)),
        min_hold_periods=int(train_args.get("min_hold_periods", 2)),
        switch_penalty_raw=float(train_args.get("switch_penalty_raw", 0.005)),
        cash_return_per_period=float(train_args.get("cash_return_per_period", 0.0)),
        downside_protection_coef=float(train_args.get("downside_protection_coef", 0.30)),
        drawdown_breach_coef=float(train_args.get("drawdown_breach_coef", 0.04)),
        drawdown_tolerance=float(train_args.get("drawdown_tolerance", 0.20)),
        vol_penalty_coef=float(train_args.get("vol_penalty_coef", 0.02)),
        vol_tolerance=float(train_args.get("vol_tolerance", 0.025)),
        nonstress_cash_penalty_raw=float(train_args.get("nonstress_cash_penalty_raw", 0.002)),
        vix_warn=float(train_args.get("vix_warn", 0.90)),
        vix_high=float(train_args.get("vix_high", 1.30)),
        credit_warn=float(train_args.get("credit_warn", 0.90)),
        credit_high=float(train_args.get("credit_high", 1.30)),
        mkt_dc_warn=float(train_args.get("mkt_dc_warn", -0.12)),
        mkt_dc_high=float(train_args.get("mkt_dc_high", -0.25)),
        mkt_ret_warn=float(train_args.get("mkt_ret_warn", -0.025)),
        mkt_ret_high=float(train_args.get("mkt_ret_high", -0.05)),
        default_start_exposure=max(action_levels),
    )

    module_metrics, base_metrics, rel, actions = evaluate_greedy_policy(
        env, q_net, device, label=f"risk_budget_module_v3_{args.split}"
    )
    save_df_with_dates(actions, out_root / f"{args.split}_risk_budget_actions.csv")

    summary = {
        "mode": "base_plus_risk_budget_module_v3",
        "split": args.split,
        "module_enabled": True,
        "module_metrics": module_metrics,
        "base_metrics_shadow": base_metrics,
        "relative": rel,
        "model_path": str(model_path.resolve()),
    }
    with open(out_root / "switchable_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

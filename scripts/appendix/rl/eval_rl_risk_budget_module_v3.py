from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from risk_budget_module_common_v3 import (
    MAIN_H5_DATA,
    CANDD_CONFIG,
    CANDB_CONFIG,
    ensure_dir,
    normalize_predictions,
    resolve_default_final_system_manifest_path,
    safe_read_csv,
    safe_read_json,
    timestamp_tag,
    RiskBudgetEpisodeCoreV3,
    run_fixed_post_module_backtest,
)
from train_rl_risk_budget_module_v3 import QNetwork, evaluate_greedy_policy


def main():
    ap = argparse.ArgumentParser(description="Evaluate optional RL risk budget module v3 against base candD.")
    ap.add_argument("--model_path", type=str, default="")
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

    horizon = int(final_manifest.get("horizon", 5))
    tc_bps = float(final_manifest.get("transaction_cost_bps", 10.0))
    val_df = pred[pred["split"] == "val"].copy()
    test_df = pred[pred["split"] == "test"].copy()

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        default_root = predictions_path.resolve().parents[3] / "execution" / "risk_budget_module_v3"
        xs = sorted(default_root.glob("run_*/best_risk_budget_module_v3.pt"))
        if not xs:
            raise FileNotFoundError("best_risk_budget_module_v3.pt not found")
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

    common_kwargs = dict(
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

    val_env = RiskBudgetEpisodeCoreV3(pred_df=val_df, **common_kwargs)
    test_env = RiskBudgetEpisodeCoreV3(pred_df=test_df, **common_kwargs)

    module_val_metrics, _, module_val_rel, module_val_actions = evaluate_greedy_policy(val_env, q_net, device, label="risk_budget_module_v3")
    module_test_metrics, _, module_test_rel, module_test_actions = evaluate_greedy_policy(test_env, q_net, device, label="risk_budget_module_v3")

    base_val_metrics, _, _, _ = run_fixed_post_module_backtest(val_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=1.0, label="base_candD_full")
    base_test_metrics, _, _, _ = run_fixed_post_module_backtest(test_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=1.0, label="base_candD_full")
    candb_val_metrics, _, _, _ = run_fixed_post_module_backtest(val_df, horizon, tc_bps, CANDB_CONFIG, fixed_exposure=1.0, label="base_candB_full")
    candb_test_metrics, _, _, _ = run_fixed_post_module_backtest(test_df, horizon, tc_bps, CANDB_CONFIG, fixed_exposure=1.0, label="base_candB_full")
    base060_val_metrics, _, _, _ = run_fixed_post_module_backtest(val_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=0.6, label="base_candD_060")
    base060_test_metrics, _, _, _ = run_fixed_post_module_backtest(test_df, horizon, tc_bps, CANDD_CONFIG, fixed_exposure=0.6, label="base_candD_060")

    comparison_df = pd.DataFrame([
        {"strategy": "base_candD_full", "split": "val", **base_val_metrics},
        {"strategy": "base_candD_full", "split": "test", **base_test_metrics},
        {"strategy": "base_candD_060", "split": "val", **base060_val_metrics},
        {"strategy": "base_candD_060", "split": "test", **base060_test_metrics},
        {"strategy": "base_candB_full", "split": "val", **candb_val_metrics},
        {"strategy": "base_candB_full", "split": "test", **candb_test_metrics},
        {"strategy": "risk_budget_module_v3", "split": "val", **module_val_metrics},
        {"strategy": "risk_budget_module_v3", "split": "test", **module_test_metrics},
    ])
    relative_df = pd.DataFrame([
        {
            "strategy": "risk_budget_module_v3",
            "split": "val",
            "cumret_gap_vs_base": float(module_val_metrics["cumulative_return"] - base_val_metrics["cumulative_return"]),
            "sharpe_gap_vs_base": float(module_val_metrics["sharpe"] - base_val_metrics["sharpe"]),
            "mdd_improvement_vs_base": float(abs(base_val_metrics["max_drawdown"]) - abs(module_val_metrics["max_drawdown"])),
            "avg_exposure_gap_vs_full": float(module_val_metrics["avg_exposure"] - 1.0),
        },
        {
            "strategy": "risk_budget_module_v3",
            "split": "test",
            "cumret_gap_vs_base": float(module_test_metrics["cumulative_return"] - base_test_metrics["cumulative_return"]),
            "sharpe_gap_vs_base": float(module_test_metrics["sharpe"] - base_test_metrics["sharpe"]),
            "mdd_improvement_vs_base": float(abs(base_test_metrics["max_drawdown"]) - abs(module_test_metrics["max_drawdown"])),
            "avg_exposure_gap_vs_full": float(module_test_metrics["avg_exposure"] - 1.0),
        },
    ])

    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else (predictions_path.resolve().parents[3] / "execution" / "risk_budget_eval_v3"))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(out_root / "strategy_comparison_val_test.csv", index=False)
    relative_df.to_csv(out_root / "relative_vs_base_candD.csv", index=False)
    module_val_actions.to_csv(out_root / "val_risk_budget_actions.csv", index=False)
    module_test_actions.to_csv(out_root / "test_risk_budget_actions.csv", index=False)

    summary = {
        "model_path": str(model_path.resolve()),
        "module_val_relative": module_val_rel,
        "module_test_relative": module_test_rel,
        "module_test_vs_base_candD": relative_df[relative_df["split"] == "test"].iloc[0].to_dict(),
        "action_levels": action_levels,
        "base_config": CANDD_CONFIG,
    }
    with open(out_root / "risk_budget_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Risk budget module v3 evaluation saved to: {out_root}")
    print("\\nRelative vs base candD:")
    print(relative_df.to_string(index=False))
    print("\\nStrategy comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()

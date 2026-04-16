from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_seeds(seed_str: str):
    xs = [int(x.strip()) for x in seed_str.split(",") if x.strip()]
    if not xs:
        raise ValueError("No seeds")
    return xs


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_subprocess(cmd, cwd: Path):
    print("\\n[Running]")
    print(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}: {' '.join(cmd)}")


def extract_strategy_row(df: pd.DataFrame, strategy: str, split: str):
    sub = df[(df["strategy"] == strategy) & (df["split"] == split)]
    if len(sub) != 1:
        raise ValueError(f"Expected exactly 1 row for strategy={strategy}, split={split}, got {len(sub)}")
    return sub.iloc[0].to_dict()


def mean_safe(vals):
    return float(statistics.mean(vals)) if vals else 0.0


def stdev_safe(vals):
    return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Run multi-seed validation for optional RL risk budget module v3.")
    ap.add_argument("--seeds", type=str, default="7,21,42")
    ap.add_argument("--python_exe", type=str, default=sys.executable)
    ap.add_argument("--train_script", type=str, default="")
    ap.add_argument("--eval_script", type=str, default="")
    ap.add_argument("--final_system_manifest_path", type=str, default="")
    ap.add_argument("--predictions_path", type=str, default="")
    ap.add_argument("--feature_data_path", type=str, default="")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--train_base_run_name", type=str, default="multiseed_risk_budget_v3_train")
    ap.add_argument("--eval_base_run_name", type=str, default="multiseed_risk_budget_v3_eval")

    # forwarded train args
    ap.add_argument("--action_levels", type=str, default="0.6,1.0")
    ap.add_argument("--downside_protection_coef", type=float, default=0.30)
    ap.add_argument("--switch_penalty_raw", type=float, default=0.005)
    ap.add_argument("--nonstress_cash_penalty_raw", type=float, default=0.002)
    ap.add_argument("--smooth_window", type=int, default=3)
    ap.add_argument("--min_hold_periods", type=int, default=2)
    ap.add_argument("--drawdown_breach_coef", type=float, default=0.04)
    ap.add_argument("--drawdown_tolerance", type=float, default=0.20)
    ap.add_argument("--vol_penalty_coef", type=float, default=0.02)
    ap.add_argument("--vol_tolerance", type=float, default=0.025)
    ap.add_argument("--vix_warn", type=float, default=0.90)
    ap.add_argument("--vix_high", type=float, default=1.30)
    ap.add_argument("--credit_warn", type=float, default=0.90)
    ap.add_argument("--credit_high", type=float, default=1.30)
    ap.add_argument("--mkt_dc_warn", type=float, default=-0.12)
    ap.add_argument("--mkt_dc_high", type=float, default=-0.25)
    ap.add_argument("--mkt_ret_warn", type=float, default=-0.025)
    ap.add_argument("--mkt_ret_high", type=float, default=-0.05)
    ap.add_argument("--min_select_episode", type=int, default=8)

    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    root = find_project_root()
    train_script = Path(args.train_script) if args.train_script else root / "scripts" / "rl" / "train_rl_risk_budget_module_v3.py"
    eval_script = Path(args.eval_script) if args.eval_script else root / "scripts" / "rl" / "eval_rl_risk_budget_module_v3.py"
    if not train_script.exists() or not eval_script.exists():
        raise FileNotFoundError("train or eval script missing")
    out_dir = Path(args.out_dir) if args.out_dir else root / "Model Runs" / "execution" / "risk_budget_multiseed_v3"
    out_dir = ensure_dir(out_dir)

    rows = []
    for seed in seeds:
        print("\\n" + "=" * 80)
        print(f"Running optional RL risk budget v3 multi-seed validation for seed={seed}")
        print("=" * 80)

        train_run = f"{args.train_base_run_name}_seed{seed}"
        eval_run = f"{args.eval_base_run_name}_seed{seed}"

        train_cmd = [
            args.python_exe, str(train_script),
            "--seed", str(seed),
            "--episodes", str(args.episodes),
            "--run_name", train_run,
            "--action_levels", args.action_levels,
            "--downside_protection_coef", str(args.downside_protection_coef),
            "--switch_penalty_raw", str(args.switch_penalty_raw),
            "--nonstress_cash_penalty_raw", str(args.nonstress_cash_penalty_raw),
            "--smooth_window", str(args.smooth_window),
            "--min_hold_periods", str(args.min_hold_periods),
            "--drawdown_breach_coef", str(args.drawdown_breach_coef),
            "--drawdown_tolerance", str(args.drawdown_tolerance),
            "--vol_penalty_coef", str(args.vol_penalty_coef),
            "--vol_tolerance", str(args.vol_tolerance),
            "--vix_warn", str(args.vix_warn),
            "--vix_high", str(args.vix_high),
            "--credit_warn", str(args.credit_warn),
            "--credit_high", str(args.credit_high),
            "--mkt_dc_warn", str(args.mkt_dc_warn),
            "--mkt_dc_high", str(args.mkt_dc_high),
            "--mkt_ret_warn", str(args.mkt_ret_warn),
            "--mkt_ret_high", str(args.mkt_ret_high),
            "--min_select_episode", str(args.min_select_episode),
        ]
        if args.final_system_manifest_path:
            train_cmd += ["--final_system_manifest_path", args.final_system_manifest_path]
        if args.predictions_path:
            train_cmd += ["--predictions_path", args.predictions_path]
        if args.feature_data_path:
            train_cmd += ["--feature_data_path", args.feature_data_path]
        run_subprocess(train_cmd, root)

        train_dir = root / "Model Runs" / "execution" / "risk_budget_module_v3" / train_run
        model_path = train_dir / "best_risk_budget_module_v3.pt"
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        eval_cmd = [args.python_exe, str(eval_script), "--model_path", str(model_path), "--run_name", eval_run]
        if args.final_system_manifest_path:
            eval_cmd += ["--final_system_manifest_path", args.final_system_manifest_path]
        if args.predictions_path:
            eval_cmd += ["--predictions_path", args.predictions_path]
        if args.feature_data_path:
            eval_cmd += ["--feature_data_path", args.feature_data_path]
        run_subprocess(eval_cmd, root)

        eval_dir = root / "Model Runs" / "execution" / "risk_budget_eval_v3" / eval_run
        strategy_df = pd.read_csv(eval_dir / "strategy_comparison_val_test.csv")
        relative_df = pd.read_csv(eval_dir / "relative_vs_base_candD.csv")

        module_test = extract_strategy_row(strategy_df, "risk_budget_module_v3", "test")
        base_test = extract_strategy_row(strategy_df, "base_candD_full", "test")
        rel_test = relative_df[(relative_df["strategy"] == "risk_budget_module_v3") & (relative_df["split"] == "test")].iloc[0].to_dict()

        rows.append({
            "seed": seed,
            "module_test_cumret": module_test["cumulative_return"],
            "module_test_sharpe": module_test["sharpe"],
            "module_test_mdd": module_test["max_drawdown"],
            "module_test_avg_exposure": module_test["avg_exposure"],
            "module_test_switch_rate": module_test.get("switch_rate", 0.0),
            "module_test_any_stress_rate": module_test.get("any_stress_rate", 0.0),
            "module_test_high_stress_rate": module_test.get("high_stress_rate", 0.0),
            "base_test_cumret": base_test["cumulative_return"],
            "base_test_sharpe": base_test["sharpe"],
            "base_test_mdd": base_test["max_drawdown"],
            "cumret_gap_vs_base": rel_test["cumret_gap_vs_base"],
            "sharpe_gap_vs_base": rel_test["sharpe_gap_vs_base"],
            "mdd_improvement_vs_base": rel_test["mdd_improvement_vs_base"],
            "module_beats_base_on_sharpe": float(module_test["sharpe"] > base_test["sharpe"]),
            "module_beats_base_on_cumret": float(module_test["cumulative_return"] > base_test["cumulative_return"]),
            "module_reduces_abs_drawdown_vs_base": float(abs(module_test["max_drawdown"]) < abs(base_test["max_drawdown"])),
            "train_run_dir": str(train_dir),
            "eval_run_dir": str(eval_dir),
        })

    seed_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    seed_df.to_csv(out_dir / "risk_budget_seed_summary.csv", index=False)

    agg = {"n_seeds": int(len(seed_df)), "seeds": seed_df["seed"].astype(int).tolist()}
    for c in [
        "module_test_cumret",
        "module_test_sharpe",
        "module_test_mdd",
        "module_test_avg_exposure",
        "module_test_switch_rate",
        "module_test_any_stress_rate",
        "module_test_high_stress_rate",
        "base_test_cumret",
        "base_test_sharpe",
        "base_test_mdd",
        "cumret_gap_vs_base",
        "sharpe_gap_vs_base",
        "mdd_improvement_vs_base",
    ]:
        vals = seed_df[c].astype(float).tolist()
        agg[f"{c}_mean"] = mean_safe(vals)
        agg[f"{c}_std"] = stdev_safe(vals)

    for c in ["module_beats_base_on_sharpe", "module_beats_base_on_cumret", "module_reduces_abs_drawdown_vs_base"]:
        agg[f"{c}_rate"] = float(seed_df[c].mean())

    agg["reliability_interpretation"] = {
        "improves_sharpe_most_seeds": bool(agg["module_beats_base_on_sharpe_rate"] >= 0.67),
        "protects_drawdown_most_seeds": bool(agg["module_reduces_abs_drawdown_vs_base_rate"] >= 0.67),
        "improves_cumret_most_seeds": bool(agg["module_beats_base_on_cumret_rate"] >= 0.67),
    }

    with open(out_dir / "risk_budget_aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print("\\nSaved risk budget v3 multi-seed summary to:")
    print(out_dir)
    print("\\nPer-seed concise table:")
    print(
        seed_df[
            [
                "seed",
                "module_test_cumret",
                "module_test_sharpe",
                "module_test_mdd",
                "module_test_avg_exposure",
                "module_test_switch_rate",
                "module_test_any_stress_rate",
                "module_test_high_stress_rate",
                "cumret_gap_vs_base",
                "sharpe_gap_vs_base",
                "mdd_improvement_vs_base",
            ]
        ].to_string(index=False)
    )
    print("\\nAggregate summary:")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

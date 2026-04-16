from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import (
    MAIN_H5_DATA,
    EXPERIMENTS_DIR,
    EXECUTION_DIR,
    ensure_all_core_dirs,
    ensure_dir,
    timestamp_tag,
)


# =========================================================
# Diagnose why sticky execution underperformed static
#
# What this script answers:
# 1) Did sticky frequently miss top-1 / top-2 signals?
# 2) Was sticky often constrained by max_new_names_per_rebalance?
# 3) Did lower turnover actually come at the cost of missing alpha?
#
# Inputs:
#   - final_system_*/current_test_actions.csv
#   - final_system_*/optimized_test_actions.csv
#   - fixed predictions file (same one used by execution scripts)
#   - feature dataset (for split/future_return normalization if needed)
#
# Outputs:
#   - diagnostic_summary.json
#   - period_level_diagnostics.csv
#   - missed_signal_examples.csv
#   - sticky_failure_report.csv
# =========================================================


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def safe_read_json(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_file_by_glob(root: Path, pattern: str) -> Optional[Path]:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_default_final_system_dir() -> Path:
    base = EXECUTION_DIR / "final_system"
    p = latest_file_by_glob(base, "final_system_*")
    if p is None:
        raise FileNotFoundError(
            f"Could not auto-find latest final_system_* directory under {base}. "
            f"Please pass --final_system_dir explicitly."
        )
    return p


def resolve_default_predictions_path() -> Path:
    base = EXPERIMENTS_DIR / "main_transformer_h5"
    p = latest_file_by_glob(base, "run_*/transformer_predictions_all_splits.csv")
    if p is None:
        raise FileNotFoundError(
            "Could not auto-find transformer_predictions_all_splits.csv under "
            f"{base}. Please pass --predictions_path explicitly."
        )
    return p


def infer_horizon_from_feature_df(feature_df: pd.DataFrame) -> int:
    return_cols = [c for c in feature_df.columns if c.startswith("future_return_")]
    if len(return_cols) != 1:
        return 5
    col = return_cols[0]
    if col.endswith("1d"):
        return 1
    if col.endswith("5d"):
        return 5
    return 5


def normalize_predictions(pred: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["date"] = pd.to_datetime(out["date"])

    feat = feat.copy()
    feat["date"] = pd.to_datetime(feat["date"])

    if "pred_prob" not in out.columns:
        cand = [c for c in out.columns if "prob" in c.lower()]
        if len(cand) == 0:
            raise ValueError("Predictions file must contain 'pred_prob' or equivalent.")
        out = out.rename(columns={cand[0]: "pred_prob"})

    if "split" not in out.columns:
        if "split" not in feat.columns:
            raise ValueError("Neither predictions nor feature data contains 'split'.")
        out = out.merge(
            feat[["date", "stock", "split"]],
            on=["date", "stock"],
            how="left",
        )

    if "future_return" not in out.columns:
        ret_cols = [c for c in feat.columns if c.startswith("future_return_")]
        if len(ret_cols) != 1:
            raise ValueError("Feature data must contain exactly one future_return_* column.")
        ret_col = ret_cols[0]
        out = out.merge(
            feat[["date", "stock", "split", ret_col]].rename(columns={ret_col: "future_return"}),
            on=["date", "stock", "split"],
            how="left",
        )

    required_cols = ["date", "stock", "split", "pred_prob", "future_return"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Predictions data missing required columns after normalization: {missing}")

    if out["future_return"].isna().any():
        raise ValueError("Predictions data contains missing future_return after normalization.")

    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def parse_selected_stocks(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if s == "":
        return []
    return sorted([x for x in s.split("|") if x])


def to_set(value) -> Set[str]:
    return set(parse_selected_stocks(value))


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def compute_topk_maps(test_pred: pd.DataFrame, top_n: int = 2) -> Dict[pd.Timestamp, Dict]:
    out: Dict[pd.Timestamp, Dict] = {}
    for dt, day in test_pred.groupby("date"):
        day = day.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        top_names = day["stock"].astype(str).head(top_n).tolist()
        top_probs = day["pred_prob"].astype(float).head(top_n).tolist()
        top_rets = day["future_return"].astype(float).head(top_n).tolist()

        stock_prob = dict(zip(day["stock"].astype(str), day["pred_prob"].astype(float)))
        stock_ret = dict(zip(day["stock"].astype(str), day["future_return"].astype(float)))

        out[pd.Timestamp(dt)] = {
            "top_names": top_names,
            "top_probs": top_probs,
            "top_rets": top_rets,
            "stock_prob": stock_prob,
            "stock_ret": stock_ret,
            "day_df": day,
        }
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


def max_drawdown_from_equity_curve(equity_curve: List[float]) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())


def summarize_returns(net_returns: np.ndarray, horizon: int) -> Dict[str, float]:
    if len(net_returns) == 0:
        return {
            "periods": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    equity_curve = [1.0]
    for r in net_returns:
        equity_curve.append(equity_curve[-1] * (1.0 + float(r)))

    return {
        "periods": int(len(net_returns)),
        "cumulative_return": float(equity_curve[-1] - 1.0),
        "annualized_return": annualized_return(equity_curve[-1], len(net_returns) * horizon),
        "sharpe": sharpe_ratio(net_returns.tolist(), horizon),
        "max_drawdown": max_drawdown_from_equity_curve(equity_curve),
        "win_rate": float((net_returns > 0).mean()),
    }


def build_period_level_diagnostics(
    static_actions: pd.DataFrame,
    sticky_actions: pd.DataFrame,
    topk_maps: Dict[pd.Timestamp, Dict],
) -> pd.DataFrame:
    s = static_actions.copy()
    t = sticky_actions.copy()

    s["date"] = pd.to_datetime(s["date"])
    t["date"] = pd.to_datetime(t["date"])

    merged = s.merge(
        t,
        on="date",
        how="inner",
        suffixes=("_static", "_sticky"),
    ).sort_values("date").reset_index(drop=True)

    rows = []
    for _, row in merged.iterrows():
        dt = pd.Timestamp(row["date"])
        if dt not in topk_maps:
            continue

        info = topk_maps[dt]
        top_names = info["top_names"]
        top1 = top_names[0] if len(top_names) >= 1 else ""
        top2 = top_names[:2]
        stock_prob = info["stock_prob"]
        stock_ret = info["stock_ret"]

        static_set = to_set(row.get("selected_stocks_static"))
        sticky_set = to_set(row.get("selected_stocks_sticky"))

        static_has_top1 = int(top1 in static_set) if top1 else 0
        sticky_has_top1 = int(top1 in sticky_set) if top1 else 0

        static_top2_hits = int(sum(1 for x in top2 if x in static_set))
        sticky_top2_hits = int(sum(1 for x in top2 if x in sticky_set))

        static_only = sorted(list(static_set - sticky_set))
        sticky_only = sorted(list(sticky_set - static_set))

        static_net = safe_float(row.get("net_return_static"), 0.0)
        sticky_net = safe_float(row.get("net_return_sticky"), 0.0)
        static_turn = safe_float(row.get("turnover_static"), 0.0)
        sticky_turn = safe_float(row.get("turnover_sticky"), 0.0)

        forced_keep_count = int(safe_float(row.get("forced_keep_count"), 0.0)) if "forced_keep_count" in row.index else 0
        newly_added = int(safe_float(row.get("newly_added"), 0.0)) if "newly_added" in row.index else 0
        entry_candidates_count = int(safe_float(row.get("entry_candidates_count"), 0.0)) if "entry_candidates_count" in row.index else 0
        kept_existing = int(safe_float(row.get("kept_existing"), 0.0)) if "kept_existing" in row.index else 0

        capped_new_entry_flag = int(entry_candidates_count > newly_added)
        forced_keep_flag = int(forced_keep_count > 0)

        rows.append({
            "date": dt,
            "top1_stock": top1,
            "top1_prob": safe_float(stock_prob.get(top1), np.nan) if top1 else np.nan,
            "top1_future_return": safe_float(stock_ret.get(top1), np.nan) if top1 else np.nan,
            "top2_stocks": "|".join(top2),
            "top2_mean_future_return": float(np.mean([stock_ret.get(x, np.nan) for x in top2])) if len(top2) > 0 else np.nan,

            "selected_stocks_static": "|".join(sorted(static_set)),
            "selected_stocks_sticky": "|".join(sorted(sticky_set)),
            "static_only_names": "|".join(static_only),
            "sticky_only_names": "|".join(sticky_only),
            "portfolio_overlap_count": int(len(static_set & sticky_set)),

            "static_has_top1": static_has_top1,
            "sticky_has_top1": sticky_has_top1,
            "static_top2_hits": static_top2_hits,
            "sticky_top2_hits": sticky_top2_hits,
            "sticky_missed_top1": int(static_has_top1 == 1 and sticky_has_top1 == 0),
            "sticky_missed_any_top2_vs_static": int(static_top2_hits > sticky_top2_hits),
            "sticky_missed_both_top2": int(sticky_top2_hits == 0 and len(top2) > 0),

            "net_return_static": static_net,
            "net_return_sticky": sticky_net,
            "net_return_gap_static_minus_sticky": static_net - sticky_net,
            "turnover_static": static_turn,
            "turnover_sticky": sticky_turn,
            "turnover_gap_static_minus_sticky": static_turn - sticky_turn,

            "n_holdings_static": int(safe_float(row.get("n_holdings_static"), 0)),
            "n_holdings_sticky": int(safe_float(row.get("n_holdings_sticky"), 0)),
            "exposure_static": safe_float(row.get("exposure_static"), np.nan),
            "exposure_sticky": safe_float(row.get("exposure_sticky"), np.nan),

            "kept_existing": kept_existing,
            "newly_added": newly_added,
            "forced_keep_count": forced_keep_count,
            "entry_candidates_count": entry_candidates_count,
            "capped_new_entry_flag": capped_new_entry_flag,
            "forced_keep_flag": forced_keep_flag,

            "static_better_period": int(static_net > sticky_net),
            "sticky_better_period": int(sticky_net > static_net),
            "sticky_lower_turnover_period": int(sticky_turn < static_turn),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def build_summary(diag_df: pd.DataFrame, horizon: int) -> Dict:
    if diag_df.empty:
        return {"error": "period_level_diagnostics is empty"}

    static_rets = diag_df["net_return_static"].astype(float).values
    sticky_rets = diag_df["net_return_sticky"].astype(float).values

    static_perf = summarize_returns(static_rets, horizon)
    sticky_perf = summarize_returns(sticky_rets, horizon)

    summary = {
        "test_periods": int(len(diag_df)),

        "static_test_performance": static_perf,
        "sticky_test_performance": sticky_perf,

        "avg_net_return_static_minus_sticky": float(diag_df["net_return_gap_static_minus_sticky"].mean()),
        "median_net_return_static_minus_sticky": float(diag_df["net_return_gap_static_minus_sticky"].median()),
        "avg_turnover_static_minus_sticky": float(diag_df["turnover_gap_static_minus_sticky"].mean()),
        "median_turnover_static_minus_sticky": float(diag_df["turnover_gap_static_minus_sticky"].median()),

        "static_better_period_rate": float(diag_df["static_better_period"].mean()),
        "sticky_better_period_rate": float(diag_df["sticky_better_period"].mean()),
        "sticky_lower_turnover_rate": float(diag_df["sticky_lower_turnover_period"].mean()),

        "sticky_missed_top1_rate": float(diag_df["sticky_missed_top1"].mean()),
        "sticky_missed_any_top2_rate": float(diag_df["sticky_missed_any_top2_vs_static"].mean()),
        "sticky_missed_both_top2_rate": float(diag_df["sticky_missed_both_top2"].mean()),

        "sticky_forced_keep_rate": float(diag_df["forced_keep_flag"].mean()),
        "sticky_capped_new_entry_rate": float(diag_df["capped_new_entry_flag"].mean()),
        "sticky_underinvested_rate": float((diag_df["exposure_sticky"].astype(float) < 0.999).mean()),

        "avg_top2_hits_static": float(diag_df["static_top2_hits"].mean()),
        "avg_top2_hits_sticky": float(diag_df["sticky_top2_hits"].mean()),
        "avg_overlap_count": float(diag_df["portfolio_overlap_count"].mean()),
    }

    mask_static_adv = diag_df["net_return_gap_static_minus_sticky"] > 0
    summary["sticky_missed_top1_when_static_better_rate"] = float(
        diag_df.loc[mask_static_adv, "sticky_missed_top1"].mean()
    ) if mask_static_adv.any() else 0.0

    summary["sticky_capped_new_entry_when_static_better_rate"] = float(
        diag_df.loc[mask_static_adv, "capped_new_entry_flag"].mean()
    ) if mask_static_adv.any() else 0.0

    summary["sticky_forced_keep_when_static_better_rate"] = float(
        diag_df.loc[mask_static_adv, "forced_keep_flag"].mean()
    ) if mask_static_adv.any() else 0.0

    return summary


def build_failure_report(diag_df: pd.DataFrame) -> pd.DataFrame:
    df = diag_df.copy()

    conditions = []
    for _, row in df.iterrows():
        tags = []
        if int(row["sticky_missed_top1"]) == 1:
            tags.append("missed_top1")
        if int(row["sticky_missed_any_top2_vs_static"]) == 1:
            tags.append("missed_top2")
        if int(row["capped_new_entry_flag"]) == 1:
            tags.append("capped_new_entry")
        if int(row["forced_keep_flag"]) == 1:
            tags.append("forced_keep")
        if float(row["exposure_sticky"]) < 0.999:
            tags.append("underinvested")
        if int(row["sticky_lower_turnover_period"]) == 1:
            tags.append("lower_turnover")
        conditions.append("|".join(tags))

    df["diagnostic_tags"] = conditions

    report = df.sort_values(
        ["net_return_gap_static_minus_sticky", "turnover_gap_static_minus_sticky"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return report


def main():
    ensure_all_core_dirs()

    parser = argparse.ArgumentParser(description="Diagnose why sticky execution underperformed static.")
    parser.add_argument("--final_system_dir", type=str, default="")
    parser.add_argument("--predictions_path", type=str, default="")
    parser.add_argument("--feature_data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--out_dir", type=str, default=str(EXECUTION_DIR / "sticky_vs_static_diagnostics"))
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    final_system_dir = Path(args.final_system_dir) if args.final_system_dir else resolve_default_final_system_dir()
    predictions_path = Path(args.predictions_path) if args.predictions_path else resolve_default_predictions_path()
    feature_data_path = Path(args.feature_data_path)

    current_test_actions_path = final_system_dir / "current_test_actions.csv"
    optimized_test_actions_path = final_system_dir / "optimized_test_actions.csv"
    manifest_path = final_system_dir / "final_system_manifest.json"

    print(f"Using final system dir: {final_system_dir}")
    print(f"Using predictions: {predictions_path}")
    print(f"Using feature data: {feature_data_path}")

    static_actions = safe_read_csv(current_test_actions_path)
    sticky_actions = safe_read_csv(optimized_test_actions_path)
    feat = safe_read_csv(feature_data_path)
    pred = safe_read_csv(predictions_path)
    pred = normalize_predictions(pred, feat)
    horizon = infer_horizon_from_feature_df(feat)

    final_manifest = {}
    if manifest_path.exists():
        final_manifest = safe_read_json(manifest_path)

    test_pred = pred[pred["split"] == "test"].copy()
    topk_maps = compute_topk_maps(test_pred, top_n=2)

    diag_df = build_period_level_diagnostics(
        static_actions=static_actions,
        sticky_actions=sticky_actions,
        topk_maps=topk_maps,
    )
    if diag_df.empty:
        raise RuntimeError("No overlapping test dates found between action files and predictions.")

    summary = build_summary(diag_df, horizon=horizon)
    failure_report = build_failure_report(diag_df)

    missed_examples = failure_report[
        (failure_report["diagnostic_tags"].str.contains("missed_top1|missed_top2|capped_new_entry|forced_keep", regex=True))
        & (failure_report["net_return_gap_static_minus_sticky"] > 0)
    ].copy()

    keep_cols = [
        "date",
        "diagnostic_tags",
        "top1_stock",
        "top1_prob",
        "top1_future_return",
        "top2_stocks",
        "top2_mean_future_return",
        "selected_stocks_static",
        "selected_stocks_sticky",
        "static_only_names",
        "sticky_only_names",
        "net_return_static",
        "net_return_sticky",
        "net_return_gap_static_minus_sticky",
        "turnover_static",
        "turnover_sticky",
        "turnover_gap_static_minus_sticky",
        "kept_existing",
        "newly_added",
        "forced_keep_count",
        "entry_candidates_count",
    ]
    missed_examples = missed_examples[keep_cols].head(30)

    output_payload = {
        "source_final_system_dir": str(final_system_dir.resolve()),
        "source_predictions_path": str(predictions_path.resolve()),
        "source_feature_data_path": str(feature_data_path.resolve()),
        "horizon": int(horizon),
        "final_system_manifest_excerpt": {
            "frozen_prediction_layer": final_manifest.get("frozen_prediction_layer", {}),
            "execution_layer": final_manifest.get("execution_layer", {}),
            "selection_policy": final_manifest.get("selection_policy", ""),
        },
        "diagnostic_summary": summary,
    }

    base_out_dir = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    out_root = base_out_dir / run_name
    if out_root.exists():
        out_root = base_out_dir / f"{run_name}_{timestamp_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    diag_to_save = diag_df.copy()
    diag_to_save["date"] = pd.to_datetime(diag_to_save["date"]).dt.strftime("%Y-%m-%d")
    diag_to_save.to_csv(out_root / "period_level_diagnostics.csv", index=False)

    failure_to_save = failure_report.copy()
    failure_to_save["date"] = pd.to_datetime(failure_to_save["date"]).dt.strftime("%Y-%m-%d")
    failure_to_save.to_csv(out_root / "sticky_failure_report.csv", index=False)

    missed_to_save = missed_examples.copy()
    if not missed_to_save.empty:
        missed_to_save["date"] = pd.to_datetime(missed_to_save["date"]).dt.strftime("%Y-%m-%d")
    missed_to_save.to_csv(out_root / "missed_signal_examples.csv", index=False)

    with open(out_root / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"Sticky vs Static diagnostics saved to: {out_root}")
    print("\nDiagnostic summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
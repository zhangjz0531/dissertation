from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import MAIN_H5_DATA, ROBUSTNESS_DIR, ensure_all_core_dirs, ensure_dir, timestamp_tag


warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")


def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    target_cols = [c for c in df.columns if c.startswith("target_up_")]
    return_cols = [c for c in df.columns if c.startswith("future_return_")]
    if len(target_cols) != 1 or len(return_cols) != 1:
        raise ValueError("Dataset must contain exactly one target_up_* and one future_return_* column.")
    return target_cols[0], return_cols[0]


def feature_columns(df: pd.DataFrame, target_col: str, return_col: str) -> List[str]:
    excluded = {"date", "stock", "split", target_col, return_col}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(y_score).rank(method="average").values
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    xv = tmp.iloc[:, 0].values.astype(float)
    yv = tmp.iloc[:, 1].values.astype(float)
    if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    xr = tmp.iloc[:, 0].rank(method="average")
    yr = tmp.iloc[:, 1].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return np.nan
    return safe_pearson(xr, yr)


def pct_top_bottom_spread(df: pd.DataFrame, feature: str, return_col: str, q: float = 0.2) -> Tuple[float, float]:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[return_col], errors="coerce")
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 20 or tmp["x"].nunique() < 2:
        return np.nan, np.nan
    lo = tmp["x"].quantile(q)
    hi = tmp["x"].quantile(1 - q)
    top = tmp.loc[tmp["x"] >= hi, "y"]
    bot = tmp.loc[tmp["x"] <= lo, "y"]
    if len(top) < 5 or len(bot) < 5:
        return np.nan, np.nan
    spread = float(top.mean() - bot.mean())
    direction = 1.0 if spread >= 0 else -1.0
    return spread, direction


def categorize_feature(name: str) -> str:
    lower = name.lower()
    if lower.startswith("mkt_"):
        return "benchmark_market"
    if lower.endswith("_cs_z"):
        return "cross_sectional"
    if lower.startswith("dc_") or "dc_" in lower:
        return "dc"
    if "vix" in lower or "sentiment" in lower:
        return "sentiment_proxy"
    if "credit" in lower or "interest_rate" in lower or "macro" in lower or "hyg" in lower or "rate_" in lower:
        return "macro_proxy"
    if any(k in lower for k in ["net_margin", "operating_margin", "revenue_growth", "debt_to_equity", "asset_turnover", "fundamental"]):
        return "fundamental"
    if any(k in lower for k in ["rsi", "macd", "ma", "atr", "spread", "volume_z", "volatility", "return_", "log_return"]):
        return "technical"
    return "other"


def is_cross_sectional_eligible(feature: str) -> bool:
    group = categorize_feature(feature)
    # market-wide / date-constant features should NOT be scored via same-date cross-sectional IC
    if group in {"benchmark_market", "macro_proxy", "sentiment_proxy"}:
        return False
    return True


def datewise_cross_sectional_ic(df: pd.DataFrame, feature: str, return_col: str, min_names: int = 5) -> Dict[str, float]:
    vals = []
    eligible_dates = 0
    skipped_constant_dates = 0

    for _, g in df.groupby("date", sort=False):
        if len(g) < min_names:
            continue
        x = pd.to_numeric(g[feature], errors="coerce")
        y = pd.to_numeric(g[return_col], errors="coerce")
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(tmp) < min_names:
            continue
        eligible_dates += 1
        if tmp["x"].nunique() < 2 or tmp["x"].std() < 1e-12:
            skipped_constant_dates += 1
            continue
        ic = safe_spearman(tmp["x"], tmp["y"])
        if np.isfinite(ic):
            vals.append(ic)

    if len(vals) == 0:
        return {
            "cs_ic_mean": np.nan,
            "cs_ic_std": np.nan,
            "cs_ic_ir": np.nan,
            "cs_ic_hit_rate": np.nan,
            "cs_ic_n_dates": 0,
            "cs_ic_eligible_dates": int(eligible_dates),
            "cs_ic_skipped_constant_dates": int(skipped_constant_dates),
        }

    arr = np.asarray(vals, dtype=float)
    std = arr.std(ddof=1) if len(arr) > 1 else np.nan
    ir = arr.mean() / std if std and np.isfinite(std) and std > 1e-12 else np.nan
    return {
        "cs_ic_mean": float(arr.mean()),
        "cs_ic_std": float(std) if np.isfinite(std) else np.nan,
        "cs_ic_ir": float(ir) if np.isfinite(ir) else np.nan,
        "cs_ic_hit_rate": float((arr > 0).mean()),
        "cs_ic_n_dates": int(len(arr)),
        "cs_ic_eligible_dates": int(eligible_dates),
        "cs_ic_skipped_constant_dates": int(skipped_constant_dates),
    }


def global_directional_stats(df: pd.DataFrame, feature: str, target_col: str, return_col: str) -> Dict[str, float]:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    r = pd.to_numeric(df[return_col], errors="coerce")
    tmp = pd.DataFrame({"x": x, "y": y, "r": r}).dropna()
    if len(tmp) < 50 or tmp["x"].nunique() < 2:
        return {
            "global_auc": np.nan,
            "global_auc_edge": np.nan,
            "global_auc_direction": np.nan,
            "spearman_with_return": np.nan,
            "pearson_with_return": np.nan,
            "top_bottom_spread": np.nan,
            "top_bottom_direction": np.nan,
        }

    auc = roc_auc_binary(tmp["y"].values, tmp["x"].values)
    if np.isfinite(auc):
        auc_edge = abs(auc - 0.5)
        auc_direction = 1.0 if auc >= 0.5 else -1.0
    else:
        auc_edge, auc_direction = np.nan, np.nan

    spread, spread_direction = pct_top_bottom_spread(tmp.rename(columns={"x": feature, "r": return_col}), feature, return_col)
    return {
        "global_auc": float(auc) if np.isfinite(auc) else np.nan,
        "global_auc_edge": float(auc_edge) if np.isfinite(auc_edge) else np.nan,
        "global_auc_direction": float(auc_direction) if np.isfinite(auc_direction) else np.nan,
        "spearman_with_return": safe_spearman(tmp["x"], tmp["r"]),
        "pearson_with_return": safe_pearson(tmp["x"], tmp["r"]),
        "top_bottom_spread": float(spread) if np.isfinite(spread) else np.nan,
        "top_bottom_direction": float(spread_direction) if np.isfinite(spread_direction) else np.nan,
    }


def market_feature_time_series_stats(df: pd.DataFrame, feature: str, target_col: str, return_col: str) -> Dict[str, float]:
    # Collapse to one row per date for the feature, compare it to date-level average future returns
    g = df.groupby("date", as_index=False).agg(
        feature_value=(feature, "first"),
        mean_future_return=(return_col, "mean"),
        median_future_return=(return_col, "median"),
        pos_rate=(target_col, "mean"),
    )
    x = pd.to_numeric(g["feature_value"], errors="coerce")
    mean_ret = pd.to_numeric(g["mean_future_return"], errors="coerce")
    med_ret = pd.to_numeric(g["median_future_return"], errors="coerce")
    pos_rate = pd.to_numeric(g["pos_rate"], errors="coerce")
    pos_label = (pos_rate > 0.5).astype(float)

    auc = roc_auc_binary(pos_label.values, x.values)
    auc_edge = abs(auc - 0.5) if np.isfinite(auc) else np.nan

    return {
        "ts_spearman_with_mean_return": safe_spearman(x, mean_ret),
        "ts_pearson_with_mean_return": safe_pearson(x, mean_ret),
        "ts_spearman_with_median_return": safe_spearman(x, med_ret),
        "ts_auc_vs_date_positive_rate": float(auc) if np.isfinite(auc) else np.nan,
        "ts_auc_edge_vs_date_positive_rate": float(auc_edge) if np.isfinite(auc_edge) else np.nan,
        "ts_n_dates": int(len(g)),
    }


def compute_feature_diagnostics(df: pd.DataFrame, target_col: str, return_col: str, feats: List[str]) -> pd.DataFrame:
    rows = []
    for split_name, sdf in df.groupby("split", sort=False):
        for feat in feats:
            x = pd.to_numeric(sdf[feat], errors="coerce")
            missing_rate = float(x.isna().mean())
            finite_std = float(x.dropna().std()) if x.notna().sum() > 1 else np.nan
            fgroup = categorize_feature(feat)

            if is_cross_sectional_eligible(feat):
                cs = datewise_cross_sectional_ic(sdf[["date", feat, return_col]].copy(), feat, return_col)
            else:
                cs = {
                    "cs_ic_mean": np.nan,
                    "cs_ic_std": np.nan,
                    "cs_ic_ir": np.nan,
                    "cs_ic_hit_rate": np.nan,
                    "cs_ic_n_dates": 0,
                    "cs_ic_eligible_dates": 0,
                    "cs_ic_skipped_constant_dates": 0,
                }

            gl = global_directional_stats(sdf[[feat, target_col, return_col]].copy(), feat, target_col, return_col)

            if not is_cross_sectional_eligible(feat):
                ts = market_feature_time_series_stats(sdf[["date", feat, target_col, return_col]].copy(), feat, target_col, return_col)
            else:
                ts = {
                    "ts_spearman_with_mean_return": np.nan,
                    "ts_pearson_with_mean_return": np.nan,
                    "ts_spearman_with_median_return": np.nan,
                    "ts_auc_vs_date_positive_rate": np.nan,
                    "ts_auc_edge_vs_date_positive_rate": np.nan,
                    "ts_n_dates": 0,
                }

            rows.append({
                "split": split_name,
                "feature": feat,
                "feature_group": fgroup,
                "cross_sectional_eligible": bool(is_cross_sectional_eligible(feat)),
                "missing_rate": missing_rate,
                "std": finite_std,
                **cs,
                **gl,
                **ts,
            })

    diag = pd.DataFrame(rows)

    train_ref = diag[diag["split"] == "train"][[
        "feature", "cs_ic_mean", "global_auc_edge", "top_bottom_spread",
        "ts_spearman_with_mean_return", "ts_auc_edge_vs_date_positive_rate"
    ]].copy()
    train_ref = train_ref.rename(columns={
        "cs_ic_mean": "train_cs_ic_mean_ref",
        "global_auc_edge": "train_auc_edge_ref",
        "top_bottom_spread": "train_top_bottom_spread_ref",
        "ts_spearman_with_mean_return": "train_ts_spearman_ref",
        "ts_auc_edge_vs_date_positive_rate": "train_ts_auc_edge_ref",
    })
    diag = diag.merge(train_ref, on="feature", how="left")

    diag["generalization_ic_gap"] = (diag["cs_ic_mean"] - diag["train_cs_ic_mean_ref"]).abs()
    diag["generalization_auc_gap"] = (diag["global_auc_edge"] - diag["train_auc_edge_ref"]).abs()
    diag["generalization_spread_gap"] = (diag["top_bottom_spread"] - diag["train_top_bottom_spread_ref"]).abs()
    diag["generalization_ts_gap"] = (diag["ts_spearman_with_mean_return"] - diag["train_ts_spearman_ref"]).abs()

    def train_score_row(r):
        if bool(r["cross_sectional_eligible"]):
            return (
                3.0 * (abs(r["cs_ic_mean"]) if np.isfinite(r["cs_ic_mean"]) else 0.0)
                + 2.0 * (r["global_auc_edge"] if np.isfinite(r["global_auc_edge"]) else 0.0)
                + 1.5 * (abs(r["top_bottom_spread"]) if np.isfinite(r["top_bottom_spread"]) else 0.0)
                + 0.5 * (r["cs_ic_hit_rate"] if np.isfinite(r["cs_ic_hit_rate"]) else 0.0)
            )
        return (
            3.0 * (abs(r["ts_spearman_with_mean_return"]) if np.isfinite(r["ts_spearman_with_mean_return"]) else 0.0)
            + 2.0 * (r["ts_auc_edge_vs_date_positive_rate"] if np.isfinite(r["ts_auc_edge_vs_date_positive_rate"]) else 0.0)
            + 1.0 * (abs(r["spearman_with_return"]) if np.isfinite(r["spearman_with_return"]) else 0.0)
        )

    diag["train_rank_score"] = 0.0
    train_mask = diag["split"] == "train"
    diag.loc[train_mask, "train_rank_score"] = diag.loc[train_mask].apply(train_score_row, axis=1)

    return diag


def summarize_group(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    sub = df[df["split"] == split_name].copy()
    if sub.empty:
        return pd.DataFrame()
    out = sub.groupby("feature_group", as_index=False).agg(
        n_features=("feature", "count"),
        n_cross_sectional_eligible=("cross_sectional_eligible", "sum"),
        mean_abs_cs_ic=("cs_ic_mean", lambda s: float(np.nanmean(np.abs(s))) if len(s) else np.nan),
        mean_cs_ic_ir=("cs_ic_ir", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
        mean_auc_edge=("global_auc_edge", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
        mean_abs_top_bottom_spread=("top_bottom_spread", lambda s: float(np.nanmean(np.abs(s))) if len(s) else np.nan),
        mean_abs_ts_corr=("ts_spearman_with_mean_return", lambda s: float(np.nanmean(np.abs(s))) if len(s) else np.nan),
        mean_ts_auc_edge=("ts_auc_edge_vs_date_positive_rate", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
        mean_missing_rate=("missing_rate", lambda s: float(np.nanmean(s)) if len(s) else np.nan),
    )
    out["split"] = split_name
    return out.sort_values(
        ["mean_abs_cs_ic", "mean_abs_ts_corr", "mean_auc_edge", "mean_ts_auc_edge"],
        ascending=False
    )


def build_shortlists(diag: pd.DataFrame, top_n: int = 15) -> Dict[str, pd.DataFrame]:
    train = diag[diag["split"] == "train"].copy()
    val = diag[diag["split"] == "val"][[
        "feature", "cs_ic_mean", "global_auc_edge", "top_bottom_spread",
        "ts_spearman_with_mean_return", "ts_auc_edge_vs_date_positive_rate"
    ]].copy()
    test = diag[diag["split"] == "test"][[
        "feature", "cs_ic_mean", "global_auc_edge", "top_bottom_spread",
        "ts_spearman_with_mean_return", "ts_auc_edge_vs_date_positive_rate"
    ]].copy()

    val = val.rename(columns={
        "cs_ic_mean": "val_cs_ic_mean",
        "global_auc_edge": "val_auc_edge",
        "top_bottom_spread": "val_top_bottom_spread",
        "ts_spearman_with_mean_return": "val_ts_spearman",
        "ts_auc_edge_vs_date_positive_rate": "val_ts_auc_edge",
    })
    test = test.rename(columns={
        "cs_ic_mean": "test_cs_ic_mean",
        "global_auc_edge": "test_auc_edge",
        "top_bottom_spread": "test_top_bottom_spread",
        "ts_spearman_with_mean_return": "test_ts_spearman",
        "ts_auc_edge_vs_date_positive_rate": "test_ts_auc_edge",
    })

    merged = train.merge(val, on="feature", how="left").merge(test, on="feature", how="left")
    merged["stability_score"] = (
        2.0 * abs(merged["val_cs_ic_mean"]).fillna(0.0)
        + 2.0 * abs(merged["test_cs_ic_mean"]).fillna(0.0)
        + 1.0 * merged["val_auc_edge"].fillna(0.0)
        + 1.0 * merged["test_auc_edge"].fillna(0.0)
        + 1.5 * abs(merged["val_ts_spearman"]).fillna(0.0)
        + 1.5 * abs(merged["test_ts_spearman"]).fillna(0.0)
        - 1.0 * merged["generalization_ic_gap"].fillna(0.0)
        - 0.5 * merged["generalization_auc_gap"].fillna(0.0)
        - 0.5 * merged["generalization_ts_gap"].fillna(0.0)
    )
    merged["overall_signal_score"] = merged["train_rank_score"].fillna(0.0) + merged["stability_score"].fillna(0.0)

    top_overall = merged.sort_values(
        ["overall_signal_score", "train_rank_score", "stability_score"],
        ascending=False
    ).head(top_n)

    top_train_ic = train.sort_values("cs_ic_mean", ascending=False).head(top_n)
    bottom_train_ic = train.sort_values("cs_ic_mean", ascending=True).head(top_n)
    top_auc_edge = train.sort_values("global_auc_edge", ascending=False).head(top_n)
    top_spread = train.sort_values("top_bottom_spread", ascending=False).head(top_n)
    top_ts = train.sort_values("ts_spearman_with_mean_return", ascending=False).head(top_n)

    return {
        "top_overall": top_overall,
        "top_train_ic": top_train_ic,
        "bottom_train_ic": bottom_train_ic,
        "top_auc_edge": top_auc_edge,
        "top_spread": top_spread,
        "top_ts_signal": top_ts,
    }


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Diagnose signal strength of main experiment features before retraining Transformer.")
    parser.add_argument("--data_path", type=str, default=str(MAIN_H5_DATA))
    parser.add_argument("--out_dir", type=str, default=str(ROBUSTNESS_DIR / "signal_diagnosis_main"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    ensure_all_core_dirs()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    out_root = ensure_dir(Path(args.out_dir))
    run_name = args.run_name.strip() if args.run_name else f"run_{timestamp_tag()}"
    save_dir = ensure_dir(out_root / run_name)

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    target_col, return_col = infer_columns(df)
    feats = feature_columns(df, target_col, return_col)

    print("==================================================")
    print("Signal diagnosis started")
    print("==================================================")
    print(f"Dataset      : {data_path}")
    print(f"Target       : {target_col}")
    print(f"Return       : {return_col}")
    print(f"Feature count: {len(feats)}")
    print(f"Date range   : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(df["split"].value_counts().to_string())
    print("")

    diag = compute_feature_diagnostics(df, target_col, return_col, feats)
    diag.to_csv(save_dir / "feature_diagnostics_by_split.csv", index=False)

    group_frames = []
    for split_name in ["train", "val", "test"]:
        grp = summarize_group(diag, split_name)
        if not grp.empty:
            group_frames.append(grp)
    group_df = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    group_df.to_csv(save_dir / "feature_group_summary.csv", index=False)

    shortlists = build_shortlists(diag, top_n=args.top_n)
    for name, sdf in shortlists.items():
        sdf.to_csv(save_dir / f"{name}.csv", index=False)

    summary = {
        "data_path": str(data_path.resolve()),
        "target_col": target_col,
        "return_col": return_col,
        "feature_count": len(feats),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "top_n": int(args.top_n),
        "design_fix": {
            "cross_sectional_features_only_for_cs_ic": True,
            "market_features_use_time_series_diagnostics": True,
            "constant_input_protection": True,
            "warning_filter_enabled": True,
        },
    }
    save_json(summary, save_dir / "diagnosis_summary.json")

    print(f"Saved outputs to: {save_dir}")
    print("")
    print("Top overall shortlist:")
    display_cols = [
        "feature", "feature_group", "cross_sectional_eligible",
        "train_rank_score", "stability_score", "overall_signal_score",
        "cs_ic_mean", "global_auc_edge", "top_bottom_spread",
        "ts_spearman_with_mean_return", "ts_auc_edge_vs_date_positive_rate"
    ]
    top_overall = shortlists["top_overall"].copy()
    keep = [c for c in display_cols if c in top_overall.columns]
    print(top_overall[keep].head(10).to_string(index=False))

    if not group_df.empty:
        print("")
        print("Feature-group summary (train):")
        train_grp = group_df[group_df["split"] == "train"].copy()
        if not train_grp.empty:
            print(train_grp.to_string(index=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------
# Project path bootstrap
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from project_paths import ensure_dir  # type: ignore
except Exception:
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------
# Reduced feature definition
# ---------------------------
DEFAULT_KEEP_FEATURES: List[str] = [
    # technical / return state
    "return_1d",
    "return_5d",
    "return_21d",
    "volatility_20d",
    "rsi_14",
    "macd_hist",
    "macd_hist_pct",
    "atr_pct",
    "hl_spread",
    "volume_z_20",

    # DC features
    "dc_event",
    "dc_trend",
    "dc_tmv",
    "dc_event_density_20",

    # cross-sectional normalized features
    "return_21d_cs_z",
    "rsi_14_cs_z",
    "macd_hist_pct_cs_z",

    # sentiment / macro proxies
    "vix_change_5d",
    "vix_z_60",
    "sentiment_score",
    "interest_rate",
    "rate_change_5d",
    "credit_stress",
    "hyg_return_5d",
    "macro_news_pressure",

    # benchmark / market-state features
    "mkt_return_1d",
    "mkt_return_5d",
    "mkt_return_21d",
    "mkt_volatility_20d",
    "mkt_dc_event",
    "mkt_dc_trend",
    "mkt_dc_tmv",
]

DEFAULT_DROP_IF_PRESENT: List[str] = [
    # raw price levels / likely unstable or redundant variables
    "open", "high", "low", "close", "volume",
    "log_return_1d",
    "macd", "macd_signal",
    "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "atr_14",
    "dc_run_length", "dc_dist_extreme",
    "vix_level",
    "mkt_sentiment_score",
    "mkt_macro_news_pressure",
    "mkt_interest_rate",
    "mkt_credit_stress",
]


# ---------------------------
# Helpers
# ---------------------------
def infer_default_paths() -> Tuple[Path, Path, Path]:
    cleaned_dir = PROJECT_ROOT / "data_acquisition" / "cleaned_datasets"
    input_path = cleaned_dir / "main_experiment_h5.csv"
    output_path = cleaned_dir / "main_experiment_h5_reduced_v1.csv"
    feature_json_path = cleaned_dir / "reduced_feature_set_v1.json"
    return input_path, output_path, feature_json_path


def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    target_cols = [c for c in df.columns if c.startswith("target_up_")]
    return_cols = [c for c in df.columns if c.startswith("future_return_")]
    if len(target_cols) != 1 or len(return_cols) != 1:
        raise ValueError(
            "Dataset must contain exactly one target_up_* column and one future_return_* column."
        )
    return target_cols[0], return_cols[0]


def load_feature_list(path: str | None, fallback: List[str]) -> List[str]:
    if not path:
        return fallback

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature config not found: {p}")

    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "features" in obj:
            feats = obj["features"]
        else:
            feats = obj
    else:
        feats = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    if not isinstance(feats, list):
        raise ValueError("Feature config must be a list or a dict containing key 'features'.")

    return [str(x) for x in feats]


def basic_dataset_summary(df: pd.DataFrame, target_col: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "rows": int(len(df)),
        "stocks": int(df["stock"].nunique()) if "stock" in df.columns else None,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns else None,
        "target_positive_rate": float(pd.to_numeric(df[target_col], errors="coerce").mean()),
    }
    if "split" in df.columns:
        summary["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}
    return summary


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    default_input, default_output, default_feature_json = infer_default_paths()

    parser = argparse.ArgumentParser(
        description="Build reduced-feature main H5 dataset for the dissertation mainline."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(default_input),
        help="Path to the source main_experiment_h5.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(default_output),
        help="Path to save the reduced dataset CSV",
    )
    parser.add_argument(
        "--feature_json",
        type=str,
        default="",
        help="Optional custom feature list (.json or .txt). If omitted, uses DEFAULT_KEEP_FEATURES.",
    )
    parser.add_argument(
        "--write_feature_json_path",
        type=str,
        default=str(default_feature_json),
        help="Where to save the finalized reduced feature list JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    feature_json_out = Path(args.write_feature_json_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    ensure_dir(output_path.parent)
    ensure_dir(feature_json_out.parent)

    df = pd.read_csv(input_path)
    target_col, return_col = infer_columns(df)

    base_cols = ["date", "stock", "split", target_col, return_col]
    keep_features = load_feature_list(args.feature_json, DEFAULT_KEEP_FEATURES)

    existing_keep = [c for c in keep_features if c in df.columns]
    missing_requested = [c for c in keep_features if c not in df.columns]
    explicit_drop_detected = [c for c in DEFAULT_DROP_IF_PRESENT if c in df.columns and c not in existing_keep]

    final_cols = base_cols + existing_keep
    out_df = df[final_cols].copy()

    numeric_cols = [c for c in out_df.columns if c not in {"date", "stock", "split"}]
    out_df[numeric_cols] = out_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out_df = out_df.sort_values(["stock", "date"]).reset_index(drop=True)

    summary: Dict[str, object] = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "target_col": target_col,
        "return_col": return_col,
        "kept_feature_count": int(len(existing_keep)),
        "kept_features": existing_keep,
        "missing_requested_features": missing_requested,
        "explicit_drop_if_present_detected": explicit_drop_detected,
        "dataset_summary": basic_dataset_summary(out_df, target_col),
        "notes": [
            "This reduced dataset is the mainline H5 training input after signal diagnosis.",
            "It keeps a compact subset of technical, DC, cross-sectional, sentiment, macro, and benchmark-state features.",
            "It intentionally excludes raw OHLCV price-level features and several redundant/unstable variables.",
        ],
    }

    feature_json_payload = {
        "name": "reduced_feature_set_v1",
        "target_col": target_col,
        "return_col": return_col,
        "features": existing_keep,
        "feature_count": int(len(existing_keep)),
        "missing_requested_features": missing_requested,
    }

    output_path.write_text("", encoding="utf-8") if False else None
    out_df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".feature_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(feature_json_out, "w", encoding="utf-8") as f:
        json.dump(feature_json_payload, f, ensure_ascii=False, indent=2)

    print("==================================================")
    print("Reduced-feature dataset built")
    print("==================================================")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Feature JSON: {feature_json_out}")
    print(f"Summary JSON: {summary_path}")
    print(f"Rows  : {len(out_df)}")
    print(f"Target: {target_col}")
    print(f"Return: {return_col}")
    print(f"Kept feature count: {len(existing_keep)}")
    print("")
    print("Kept features:")
    for c in existing_keep:
        print(f"  - {c}")
    if missing_requested:
        print("")
        print("Missing requested features (not found in input):")
        for c in missing_requested:
            print(f"  - {c}")
    print("")
    ds = summary["dataset_summary"]
    print("Dataset summary:")
    print(json.dumps(ds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

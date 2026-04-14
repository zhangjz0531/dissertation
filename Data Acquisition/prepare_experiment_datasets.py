import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# =========================================================
# Dissertation dataset preparation script
# Main experiment:
#   price / return
#   technical indicators
#   DC features
#   benchmark market state
#   VIX / TNX / HYG macro-sentiment proxies
#
# Extension / ablation:
#   + fundamentals
#   only 2025+ sample
# =========================================================


# -----------------------------
# User-adjustable settings
# -----------------------------
WARMUP_DAYS = 60
EXTENSION_START_DATE = "2025-01-01"

# Main experiment chronological split
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"   # 2024-01-01 onward -> test

# Extension experiment split (2025+)
EXT_TRAIN_END = "2025-09-30"
EXT_VAL_END = "2025-12-31"   # 2026-01-01 onward -> extension test

# Default raw path found on your machine
DEFAULT_RAW_PATH = r"D:\python\dissertation\download_data\esg_data\stock_panel_raw.csv"

ID_COLS = ["date", "stock"]

FUNDAMENTAL_COLS = [
    "net_margin",
    "operating_margin",
    "revenue_growth_qoq",
    "debt_to_equity",
    "asset_turnover",
    "has_fundamental_data",
]

# Main experiment feature set
MAIN_FEATURES = [
    # price / volume
    "open", "high", "low", "close", "volume",

    # return features
    "return_1d", "log_return_1d", "return_5d", "return_21d",

    # technical indicators
    "volatility_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist", "macd_hist_pct",
    "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "atr_14", "atr_pct",
    "hl_spread",
    "volume_z_20",

    # stock-level DC features
    "dc_event", "dc_trend", "dc_run_length", "dc_dist_extreme", "dc_tmv", "dc_event_density_20",

    # cross-sectional features
    "return_21d_cs_z", "rsi_14_cs_z", "macd_hist_pct_cs_z",

    # macro / sentiment proxies
    "vix_level", "vix_change_5d", "vix_z_60", "sentiment_score",
    "interest_rate", "rate_change_5d",
    "credit_stress", "hyg_return_5d", "macro_news_pressure",

    # benchmark market state
    "mkt_return_1d", "mkt_return_5d", "mkt_return_21d", "mkt_volatility_20d",
    "mkt_dc_event", "mkt_dc_trend", "mkt_dc_run_length", "mkt_dc_tmv",
]

EXTENSION_FEATURES = MAIN_FEATURES + FUNDAMENTAL_COLS


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir / "cleaned_datasets"

    parser = argparse.ArgumentParser(
        description="Prepare cleaned datasets for dissertation main and extension experiments."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default=DEFAULT_RAW_PATH,
        help="Path to stock_panel_raw.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(default_out),
        help="Directory to save cleaned datasets"
    )
    return parser.parse_args()


def find_file_recursively(root: Path, filename: str) -> List[Path]:
    try:
        return list(root.rglob(filename))
    except Exception:
        return []


def resolve_raw_path(raw_arg: str) -> Path:
    """
    Resolve input CSV path robustly on Windows.
    1) Use given path if it exists
    2) Search near script
    3) Search under dissertation root
    """
    p = Path(raw_arg).expanduser()

    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir / raw_arg,
        script_dir.parent / raw_arg,
        script_dir / "cleaned_datasets" / Path(raw_arg).name,
        script_dir.parent / "cleaned_datasets" / Path(raw_arg).name,
        Path(DEFAULT_RAW_PATH),
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    search_roots = [
        script_dir,
        script_dir.parent,
        Path(r"D:\python\dissertation"),
    ]

    filename = Path(raw_arg).name
    all_matches = []
    for root in search_roots:
        if root.exists():
            all_matches.extend(find_file_recursively(root, filename))

    # deduplicate
    unique_matches = []
    seen = set()
    for m in all_matches:
        r = str(m.resolve())
        if r not in seen:
            seen.add(r)
            unique_matches.append(m.resolve())

    if len(unique_matches) == 1:
        print(f"Auto-found input file: {unique_matches[0]}")
        return unique_matches[0]

    if len(unique_matches) > 1:
        msg = "\n".join(str(x) for x in unique_matches[:20])
        raise FileNotFoundError(
            f"Found multiple candidate files for '{filename}'. "
            f"Please pass --raw_path explicitly.\nCandidates:\n{msg}"
        )

    raise FileNotFoundError(
        f"Input file not found: {p}\n"
        f"Searched near script and under D:\\python\\dissertation recursively."
    )


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"date", "stock"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["stock", "date"], keep="last").copy()
    return df


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def recompute_future_targets_one_stock(group: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute future returns and binary targets from close price.
    This avoids incorrect labels at the tail caused by blanket fillna(0).
    """
    group = group.sort_values("date").copy()

    group["future_return_1d"] = group["close"].shift(-1) / group["close"] - 1.0
    group["future_return_5d"] = group["close"].shift(-5) / group["close"] - 1.0

    group["target_up_1d"] = np.where(
        group["future_return_1d"].notna(),
        (group["future_return_1d"] > 0).astype(int),
        np.nan,
    )
    group["target_up_5d"] = np.where(
        group["future_return_5d"].notna(),
        (group["future_return_5d"] > 0).astype(int),
        np.nan,
    )
    return group


def drop_warmup_one_stock(group: pd.DataFrame, warmup_days: int) -> pd.DataFrame:
    """
    Remove the first N rows per stock to avoid unstable rolling-indicator warm-up region.
    """
    group = group.sort_values("date").copy()
    if len(group) <= warmup_days:
        return group.iloc[0:0].copy()
    return group.iloc[warmup_days:].copy()


def add_time_split(df: pd.DataFrame, train_end: str, val_end: str) -> pd.DataFrame:
    df = df.copy()
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    df["split"] = np.where(
        df["date"] <= train_end_ts,
        "train",
        np.where(df["date"] <= val_end_ts, "val", "test")
    )
    return df


def validate_feature_presence(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_by_stock(df: pd.DataFrame, func, **kwargs) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("stock", sort=False):
        parts.append(func(g.copy(), **kwargs) if kwargs else func(g.copy()))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_clean_base_dataset(df: pd.DataFrame, warmup_days: int) -> pd.DataFrame:
    """
    Base cleaning:
    - replace inf with nan
    - recompute future returns / labels
    - drop per-stock warm-up region
    - keep only rows with valid close
    """
    df = replace_inf_with_nan(df)
    df = apply_by_stock(df, recompute_future_targets_one_stock)
    df = apply_by_stock(df, drop_warmup_one_stock, warmup_days=warmup_days)
    df = df[df["close"].notna()].copy()
    return df.reset_index(drop=True)


def build_experiment_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    date_floor: Optional[str] = None,
    require_fundamentals: bool = False,
) -> pd.DataFrame:
    """
    Create a ready-to-train dataset:
    - optional date filter
    - optional has_fundamental_data filter
    - keep only selected features + target
    - remove rows with invalid future labels
    - drop rows with missing selected features
    """
    df = df.copy()

    if date_floor is not None:
        df = df[df["date"] >= pd.Timestamp(date_floor)].copy()

    if require_fundamentals:
        if "has_fundamental_data" not in df.columns:
            raise ValueError("Column 'has_fundamental_data' not found.")
        df = df[df["has_fundamental_data"] == 1].copy()

    validate_feature_presence(df, feature_cols)

    target_col = f"target_up_{horizon}d"
    future_ret_col = f"future_return_{horizon}d"

    if target_col not in df.columns or future_ret_col not in df.columns:
        raise ValueError(f"Missing target columns for horizon={horizon}")

    # Remove rows without valid future information
    df = df[df[future_ret_col].notna()].copy()
    df = df[df[target_col].notna()].copy()

    keep_cols = ID_COLS + feature_cols + [future_ret_col, target_col]
    df = df[keep_cols].copy()

    # For clean training data, selected features must be complete
    df = df.dropna(subset=feature_cols + [future_ret_col, target_col])

    df[target_col] = df[target_col].astype(int)
    return df.sort_values(["date", "stock"]).reset_index(drop=True)


def summarize_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> dict:
    pos_rate = float(df[target_col].mean()) if len(df) else None
    return {
        "rows": int(len(df)),
        "stocks": int(df["stock"].nunique()),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "feature_count": len(feature_cols),
        "target_positive_rate": pos_rate,
        "split_counts": df["split"].value_counts().to_dict() if "split" in df.columns else {},
        "stock_counts": df["stock"].value_counts().to_dict(),
        "missing_after_cleaning": int(df[feature_cols].isna().sum().sum()) if len(df) else 0,
    }


def save_dataset(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp.to_csv(out_csv, index=False)


def print_basic_check(df: pd.DataFrame, name: str, target_col: str):
    print(f"\n[{name}]")
    print(f"rows: {len(df)}")
    print(f"stocks: {df['stock'].nunique()}")
    print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"{target_col} positive rate: {df[target_col].mean():.4f}")
    print(df["split"].value_counts(dropna=False).to_string())


def main():
    args = parse_args()

    raw_path = resolve_raw_path(args.raw_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Using raw input :", raw_path)
    print("Saving outputs to:", out_dir)

    raw_df = load_raw_dataframe(raw_path)
    print(f"Loaded rows: {len(raw_df)}")
    print(f"Loaded stocks: {raw_df['stock'].nunique()}")
    print(f"Raw date range: {raw_df['date'].min().date()} -> {raw_df['date'].max().date()}")

    # -------------------------------------------------
    # Step 1: build clean base after removing warm-up
    # -------------------------------------------------
    clean_base = build_clean_base_dataset(raw_df, warmup_days=WARMUP_DAYS)
    clean_base = add_time_split(clean_base, train_end=TRAIN_END, val_end=VAL_END)

    # -------------------------------------------------
    # Step 2: main experiment datasets
    # -------------------------------------------------
    main_h1 = build_experiment_dataset(clean_base, MAIN_FEATURES, horizon=1)
    main_h1 = add_time_split(main_h1, train_end=TRAIN_END, val_end=VAL_END)

    main_h5 = build_experiment_dataset(clean_base, MAIN_FEATURES, horizon=5)
    main_h5 = add_time_split(main_h5, train_end=TRAIN_END, val_end=VAL_END)

    # -------------------------------------------------
    # Step 3: extension experiment datasets
    # fundamentals + 2025+
    # -------------------------------------------------
    ext_h1 = build_experiment_dataset(
        clean_base,
        EXTENSION_FEATURES,
        horizon=1,
        date_floor=EXTENSION_START_DATE,
        require_fundamentals=True,
    )
    ext_h1 = add_time_split(ext_h1, train_end=EXT_TRAIN_END, val_end=EXT_VAL_END)

    ext_h5 = build_experiment_dataset(
        clean_base,
        EXTENSION_FEATURES,
        horizon=5,
        date_floor=EXTENSION_START_DATE,
        require_fundamentals=True,
    )
    ext_h5 = add_time_split(ext_h5, train_end=EXT_TRAIN_END, val_end=EXT_VAL_END)

    # -------------------------------------------------
    # Step 4: save csv files
    # -------------------------------------------------
    save_dataset(clean_base, out_dir / "clean_base_after_warmup.csv")
    save_dataset(main_h1, out_dir / "main_experiment_h1.csv")
    save_dataset(main_h5, out_dir / "main_experiment_h5.csv")
    save_dataset(ext_h1, out_dir / "extension_fundamental_2025plus_h1.csv")
    save_dataset(ext_h5, out_dir / "extension_fundamental_2025plus_h5.csv")

    # -------------------------------------------------
    # Step 5: save config
    # -------------------------------------------------
    config = {
        "raw_input": str(raw_path),
        "warmup_days_removed_per_stock": WARMUP_DAYS,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_start": str((pd.Timestamp(VAL_END) + pd.Timedelta(days=1)).date()),
        "extension_train_end": EXT_TRAIN_END,
        "extension_val_end": EXT_VAL_END,
        "extension_test_start": str((pd.Timestamp(EXT_VAL_END) + pd.Timedelta(days=1)).date()),
        "main_features": MAIN_FEATURES,
        "extension_features": EXTENSION_FEATURES,
        "fundamental_features": FUNDAMENTAL_COLS,
        "notes": [
            "Main experiment excludes fundamentals to preserve long history and avoid pseudo-zero contamination.",
            "Extension experiment includes fundamentals and is restricted to 2025+ with has_fundamental_data == 1.",
            "Targets are recomputed from close to avoid invalid tail labels.",
            "Fit scalers on TRAIN split only. Do not normalize on the full sample.",
        ],
    }
    with open(out_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # Step 6: save summary
    # -------------------------------------------------
    summary = {
        "main_h1": summarize_dataset(main_h1, MAIN_FEATURES, "target_up_1d"),
        "main_h5": summarize_dataset(main_h5, MAIN_FEATURES, "target_up_5d"),
        "extension_h1": summarize_dataset(ext_h1, EXTENSION_FEATURES, "target_up_1d"),
        "extension_h5": summarize_dataset(ext_h5, EXTENSION_FEATURES, "target_up_5d"),
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------
    print_basic_check(main_h1, "main_h1", "target_up_1d")
    print_basic_check(main_h5, "main_h5", "target_up_5d")
    print_basic_check(ext_h1, "extension_h1", "target_up_1d")
    print_basic_check(ext_h5, "extension_h5", "target_up_5d")

    print("\nSaved files:")
    print(out_dir / "clean_base_after_warmup.csv")
    print(out_dir / "main_experiment_h1.csv")
    print(out_dir / "main_experiment_h5.csv")
    print(out_dir / "extension_fundamental_2025plus_h1.csv")
    print(out_dir / "extension_fundamental_2025plus_h5.csv")
    print(out_dir / "feature_config.json")
    print(out_dir / "dataset_summary.json")


if __name__ == "__main__":
    main()
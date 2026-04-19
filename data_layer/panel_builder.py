from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from data_layer.config import (
    DEFAULT_LABEL_COL,
    MAIN_PANEL_PATH,
    MACRO_STOCK_PATH,
    PANEL_DIR,
    PROCESSED_DIR,
    SPLIT_PANEL_PATH,
    TECH_DAILY_STOCK_PATH,
    DC_DAILY_STOCK_PATH,
    TARGET_HORIZONS,
    get_default_dc_ready_cols,
)

warnings.filterwarnings("ignore")

DEFAULT_BASE_PATH = SPLIT_PANEL_PATH
DEFAULT_TECH_PATH = TECH_DAILY_STOCK_PATH
DEFAULT_DC_PATH = DC_DAILY_STOCK_PATH
DEFAULT_MACRO_PATH = MACRO_STOCK_PATH
DEFAULT_OUTPUT_PATH = MAIN_PANEL_PATH


# ============================================================
# 通用工具
# ============================================================
def _load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _save_any(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ 已保存 parquet: {output_path}")
    print(f"  ✅ 已保存 csv    : {csv_path}")


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺失必要列: {missing}")


def _normalize_asset_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _normalize_time_series(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    bad = int(out[time_col].isna().sum())
    if bad > 0:
        raise ValueError(f"{time_col} 有 {bad} 个无法解析的值")
    return out


def _sort_asset_time(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["asset_id", "timestamp"])
        .drop_duplicates(subset=["asset_id", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _sort_time_only(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["timestamp"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _clean_asset_time_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, ["asset_id", "timestamp"], name)
    out = _normalize_time_series(out, "timestamp")
    out["asset_id"] = _normalize_asset_id(out["asset_id"])
    return _sort_asset_time(out)


def _clean_time_only_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, ["timestamp"], name)
    out = _normalize_time_series(out, "timestamp")
    if "asset_id" in out.columns:
        out = out.drop(columns=["asset_id"])
    return _sort_time_only(out)


def _pick_new_columns(incoming: pd.DataFrame, existing_cols: Iterable[str], key_cols: Iterable[str]) -> List[str]:
    existing = set(existing_cols)
    keys = set(key_cols)
    cols = []
    for c in incoming.columns:
        if c in keys or c in existing:
            continue
        cols.append(c)
    return cols


def _parse_required_feature_cols(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return []
    out = []
    for item in raw:
        for col in str(item).split(","):
            col = col.strip()
            if col:
                out.append(col)
    return out


# ============================================================
# merge
# ============================================================
def merge_asset_time_features(base: pd.DataFrame, feat: pd.DataFrame, source_name: str) -> pd.DataFrame:
    feat = _clean_asset_time_df(feat, source_name)
    new_cols = _pick_new_columns(feat, existing_cols=base.columns, key_cols=["asset_id", "timestamp"])
    if not new_cols:
        print(f"[INFO] {source_name}: 没有可新增列，跳过")
        return base
    merged = base.merge(feat[["asset_id", "timestamp", *new_cols]], on=["asset_id", "timestamp"], how="left")
    print(f"[merge] {source_name:<12} +{len(new_cols):>3} 列")
    return merged


def merge_time_only_features(base: pd.DataFrame, feat: pd.DataFrame, source_name: str) -> pd.DataFrame:
    feat = _clean_time_only_df(feat, source_name)
    new_cols = _pick_new_columns(feat, existing_cols=base.columns, key_cols=["timestamp"])
    if not new_cols:
        print(f"[INFO] {source_name}: 没有可新增列，跳过")
        return base
    merged = base.merge(feat[["timestamp", *new_cols]], on="timestamp", how="left")
    print(f"[merge] {source_name:<12} +{len(new_cols):>3} 列")
    return merged


# ============================================================
# usable recompute
# ============================================================
def recompute_usable_flag(
    panel: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    main_horizon: int = max(TARGET_HORIZONS),
    required_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = panel.copy()
    req_cols = []
    ret_col = f"target_ret_{int(main_horizon)}"
    if ret_col in out.columns:
        req_cols.append(ret_col)
    if label_col in out.columns:
        req_cols.append(label_col)
    if required_feature_cols:
        req_cols.extend([c for c in required_feature_cols if c in out.columns])

    if "is_warmup" not in out.columns:
        raise ValueError("recompute_usable_flag 需要 is_warmup 列")

    if not req_cols:
        out["is_usable_for_model"] = (~out["is_warmup"]).astype(int)
        return out

    valid = out[req_cols].notna().all(axis=1)
    out["is_usable_for_model"] = ((~out["is_warmup"].astype(bool)) & valid).astype(int)
    return out


# ============================================================
# main
# ============================================================
def build_main_panel(
    base_panel: pd.DataFrame,
    technical_df: Optional[pd.DataFrame] = None,
    dc_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    label_col: str = DEFAULT_LABEL_COL,
    main_horizon: int = max(TARGET_HORIZONS),
    recompute_usable: bool = True,
    required_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    base = _clean_asset_time_df(base_panel, "base_panel")
    if technical_df is not None:
        base = merge_asset_time_features(base, technical_df, "technical")
    if dc_df is not None:
        base = merge_asset_time_features(base, dc_df, "dc")
    if macro_df is not None:
        base = merge_time_only_features(base, macro_df, "macro")

    if recompute_usable and {"is_warmup", "split"}.issubset(base.columns):
        auto_required = list(required_feature_cols) if required_feature_cols else []
        if not auto_required:
            auto_required = [c for c in get_default_dc_ready_cols() if c in base.columns]
        base = recompute_usable_flag(
            base,
            label_col=label_col,
            main_horizon=main_horizon,
            required_feature_cols=auto_required,
        )

    return _sort_asset_time(base)


# ============================================================
# 诊断
# ============================================================
def _group_feature_counts(columns: List[str]) -> Dict[str, int]:
    groups = {
        "raw_price": 0,
        "technical": 0,
        "dc": 0,
        "macro": 0,
        "targets": 0,
        "split_meta": 0,
        "other": 0,
    }
    raw_price_cols = {"open", "high", "low", "close", "volume"}
    split_meta_cols = {"split", "is_warmup", "is_usable_for_model", "row_num_in_asset"}
    for c in columns:
        if c in {"asset_id", "timestamp"}:
            continue
        if c in raw_price_cols:
            groups["raw_price"] += 1
        elif c.startswith("target_") or c.startswith("benchmark_"):
            groups["targets"] += 1
        elif c.startswith("dc_"):
            groups["dc"] += 1
        elif c in split_meta_cols or c.startswith("wf_"):
            groups["split_meta"] += 1
        elif c.startswith(("vix_", "ust", "credit_", "macro_", "sentiment_", "dxy_", "hyg_")):
            groups["macro"] += 1
        elif c.startswith((
            "ret_", "return_", "logret_", "vol_", "volatility_", "rsi", "macd", "atr", "ma_", "bb_",
            "boll", "hl_", "volume_", "dollar_vol", "dollar_volume_",
        )):
            groups["technical"] += 1
        else:
            groups["other"] += 1
    return groups


def summarize_panel(df: pd.DataFrame) -> Dict:
    summary: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_assets": int(df["asset_id"].nunique()),
        "date_min": str(df["timestamp"].min().date()),
        "date_max": str(df["timestamp"].max().date()),
        "feature_group_counts": _group_feature_counts(list(df.columns)),
        "missing_ratio_top20": {},
        "split_counts": {},
        "usable_counts": {},
        "warmup_counts": {},
    }
    miss = df.isna().mean().sort_values(ascending=False)
    summary["missing_ratio_top20"] = {str(k): float(v) for k, v in miss.head(20).to_dict().items()}
    if "split" in df.columns:
        summary["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts(dropna=False).to_dict().items()}
    if "is_usable_for_model" in df.columns:
        summary["usable_counts"] = {str(k): int(v) for k, v in df["is_usable_for_model"].value_counts(dropna=False).to_dict().items()}
    if "is_warmup" in df.columns:
        summary["warmup_counts"] = {str(k): int(v) for k, v in df["is_warmup"].value_counts(dropna=False).to_dict().items()}
    return summary


def print_panel_report(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("  主线总面板概况")
    print("=" * 72)
    print(f"  行数: {len(df):,}")
    print(f"  列数: {df.shape[1]}")
    print(f"  资产数: {df['asset_id'].nunique()}")
    print(f"  日期: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

    group_counts = _group_feature_counts(list(df.columns))
    print("\n  特征组计数:")
    for k, v in group_counts.items():
        print(f"    {k:<12} {v}")

    if "split" in df.columns:
        print("\n  split 计数:")
        for k, v in df["split"].value_counts().to_dict().items():
            print(f"    {str(k):<8} {int(v):>8}")

    if "is_warmup" in df.columns:
        print("\n  warmup 计数:")
        for k, v in df["is_warmup"].value_counts().to_dict().items():
            print(f"    {str(k):<8} {int(v):>8}")

    if "is_usable_for_model" in df.columns:
        print("\n  usable 计数:")
        for k, v in df["is_usable_for_model"].value_counts().to_dict().items():
            print(f"    {str(k):<8} {int(v):>8}")

    miss = df.isna().mean().sort_values(ascending=False).head(15)
    print("\n  缺失率前 15 列:")
    for c, r in miss.items():
        print(f"    {c:<28} {r:.2%}")
    print()


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Build main stock panel without fundamental / ESG.")
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--technical", type=str, default=str(DEFAULT_TECH_PATH))
    parser.add_argument("--dc", type=str, default=str(DEFAULT_DC_PATH))
    parser.add_argument("--macro", type=str, default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--no_technical", action="store_true")
    parser.add_argument("--no_dc", action="store_true")
    parser.add_argument("--no_macro", action="store_true")
    parser.add_argument("--no_recompute_usable", action="store_true")
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--main_horizon", type=int, default=max(TARGET_HORIZONS))
    parser.add_argument("--required_feature_cols", type=str, nargs="*", default=[])
    args = parser.parse_args()

    base_path = Path(args.base)
    tech_path = Path(args.technical)
    dc_path = Path(args.dc)
    macro_path = Path(args.macro)
    output_path = Path(args.output)

    print("=" * 72)
    print("  构造主线总面板（股票日线 / 无 fundamental / 无 ESG）")
    print("=" * 72)
    print(f"  base      : {base_path}")
    print(f"  technical : {tech_path}  (enabled={not args.no_technical})")
    print(f"  dc        : {dc_path}  (enabled={not args.no_dc})")
    print(f"  macro     : {macro_path}  (enabled={not args.no_macro})")
    print(f"  output    : {output_path}")
    print(f"  label_col : {args.label_col}")
    print()

    base_df = _load_any(base_path)
    tech_df = None if args.no_technical or not tech_path.exists() else _load_any(tech_path)
    dc_df = None if args.no_dc or not dc_path.exists() else _load_any(dc_path)
    macro_df = None if args.no_macro or not macro_path.exists() else _load_any(macro_path)

    if not args.no_technical and tech_df is None:
        print(f"[WARN] technical 文件不存在，跳过: {tech_path}")
    if not args.no_dc and dc_df is None:
        print(f"[WARN] dc 文件不存在，跳过: {dc_path}")
    if not args.no_macro and macro_df is None:
        print(f"[WARN] macro 文件不存在，跳过: {macro_path}")

    required_feature_cols = _parse_required_feature_cols(args.required_feature_cols)
    panel = build_main_panel(
        base_panel=base_df,
        technical_df=tech_df,
        dc_df=dc_df,
        macro_df=macro_df,
        label_col=args.label_col,
        main_horizon=int(args.main_horizon),
        recompute_usable=not bool(args.no_recompute_usable),
        required_feature_cols=required_feature_cols,
    )

    print_panel_report(panel)
    _save_any(panel, output_path)

    summary = summarize_panel(panel)
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 已保存 summary: {summary_path}")


if __name__ == "__main__":
    main()

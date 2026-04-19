from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data_layer.config import (
    BURN_IN_DAYS,
    DAILY_TRAIN_END,
    DAILY_VAL_END,
    DEFAULT_LABEL_COL,
    MAIN_PANEL_PATH,
    STOCK_BENCHMARK,
    TARGET_HORIZONS,
    VALIDATION_ASSET_SUMMARY_CSV_PATH,
    VALIDATION_REPORT_JSON_PATH,
    get_default_dc_ready_cols,
)

warnings.filterwarnings("ignore")

DEFAULT_INPUT_PATH = MAIN_PANEL_PATH
DEFAULT_REPORT_JSON = VALIDATION_REPORT_JSON_PATH
DEFAULT_ASSET_SUMMARY_CSV = VALIDATION_ASSET_SUMMARY_CSV_PATH


# ============================================================
# 基础工具
# ============================================================
def _load_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _to_time(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def _safe_quantiles(s: pd.Series, qs=(0.05, 0.5, 0.95)) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    q = s.quantile(list(qs))
    return {
        "p5": float(q.iloc[0]),
        "median": float(q.iloc[1]),
        "p95": float(q.iloc[2]),
        "mean": float(s.mean()),
        "count": int(len(s)),
    }


def _group_columns(columns: List[str]) -> Dict[str, List[str]]:
    groups = {
        "raw_price": [],
        "technical": [],
        "dc": [],
        "macro": [],
        "targets": [],
        "split_meta": [],
        "other": [],
    }
    raw_price_cols = {"open", "high", "low", "close", "volume"}
    split_meta_cols = {"split", "is_warmup", "is_usable_for_model", "row_num_in_asset"}
    for c in columns:
        if c in {"asset_id", "timestamp"}:
            continue
        if c in raw_price_cols:
            groups["raw_price"].append(c)
        elif c.startswith("target_") or c.startswith("benchmark_"):
            groups["targets"].append(c)
        elif c.startswith("dc_"):
            groups["dc"].append(c)
        elif c in split_meta_cols or c.startswith("wf_"):
            groups["split_meta"].append(c)
        elif c.startswith(("vix_", "ust", "credit_", "macro_", "sentiment_", "dxy_", "hyg_")):
            groups["macro"].append(c)
        elif c.startswith((
            "ret_", "return_", "logret_", "vol_", "volatility_", "rsi", "macd", "atr", "ma_", "bb_",
            "boll", "hl_", "volume_", "dollar_vol", "dollar_volume_",
        )):
            groups["technical"].append(c)
        else:
            groups["other"].append(c)
    return groups


# ============================================================
# checks
# ============================================================
def check_required_columns(df: pd.DataFrame) -> Dict:
    required = [
        "asset_id", "timestamp", "open", "high", "low", "close", "volume",
        "split", "is_warmup", "is_usable_for_model",
    ]
    for h in TARGET_HORIZONS:
        required.extend([f"target_ret_{h}", f"target_dir_{h}"])

    missing = _require_columns(df, required)

    optional_target_cols = []
    for h in TARGET_HORIZONS:
        optional_target_cols.extend([
            f"benchmark_ret_{h}",
            f"target_excess_ret_{h}",
            f"target_excess_dir_{h}",
            f"target_band_dir_{h}",
            f"target_band_keep_{h}",
        ])
    optional_presence = {c: (c in df.columns) for c in optional_target_cols}

    return {
        "passed": len(missing) == 0,
        "missing_columns": missing,
        "optional_target_presence": optional_presence,
    }


def check_no_benchmark_asset(df: pd.DataFrame, benchmark_asset: str = STOCK_BENCHMARK) -> Dict:
    n = int((df["asset_id"].astype(str).str.upper().str.strip() == str(benchmark_asset).upper().strip()).sum())
    return {"passed": n == 0, "benchmark_asset": benchmark_asset, "n_rows_found": n}


def check_primary_key(df: pd.DataFrame) -> Dict:
    dup_mask = df.duplicated(subset=["asset_id", "timestamp"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    examples = []
    if n_dup_rows > 0:
        examples = (
            df.loc[dup_mask, ["asset_id", "timestamp"]]
            .sort_values(["asset_id", "timestamp"])
            .head(20)
            .astype(str)
            .to_dict(orient="records")
        )
    return {"passed": n_dup_rows == 0, "n_duplicate_rows": n_dup_rows, "examples": examples}


def check_timestamp_parsing(df: pd.DataFrame) -> Dict:
    bad = int(df["timestamp"].isna().sum())
    return {"passed": bad == 0, "n_bad_timestamp": bad}


def check_time_monotonicity(df: pd.DataFrame) -> Dict:
    bad_assets = []
    for asset_id, sub in df.groupby("asset_id", sort=False):
        if not sub["timestamp"].is_monotonic_increasing:
            bad_assets.append(str(asset_id))
    return {"passed": len(bad_assets) == 0, "bad_assets": bad_assets}


def check_basic_price_sanity(df: pd.DataFrame) -> Dict:
    details = {}
    for c in ["open", "high", "low", "close"]:
        details[f"{c}_nonpositive"] = int((pd.to_numeric(df[c], errors="coerce") <= 0).sum())
    bad_hl = int((pd.to_numeric(df["high"], errors="coerce") < pd.to_numeric(df["low"], errors="coerce")).sum())
    bad_vol = int((pd.to_numeric(df["volume"], errors="coerce") < 0).sum())
    passed = all(v == 0 for v in details.values()) and bad_hl == 0 and bad_vol == 0
    return {"passed": passed, "details": details, "high_lt_low_rows": bad_hl, "negative_volume_rows": bad_vol}


def check_split_boundaries(df: pd.DataFrame) -> Dict:
    train_end = pd.Timestamp(DAILY_TRAIN_END)
    val_end = pd.Timestamp(DAILY_VAL_END)
    split_bad = []
    for split_name, sub in df.groupby("split", dropna=False):
        ts = pd.to_datetime(sub["timestamp"])
        if str(split_name) == "train":
            bad = int((ts > train_end).sum())
        elif str(split_name) == "val":
            bad = int(((ts <= train_end) | (ts > val_end)).sum())
        elif str(split_name) == "test":
            bad = int((ts <= val_end).sum())
        else:
            bad = int(len(sub))
        if bad > 0:
            split_bad.append({"split": str(split_name), "bad_rows": bad})
    return {
        "passed": len(split_bad) == 0,
        "bad_split_rows": split_bad,
        "expected_train_end": str(train_end.date()),
        "expected_val_end": str(val_end.date()),
    }


def check_warmup_consistency(df: pd.DataFrame, burn_in_days: int = BURN_IN_DAYS) -> Dict:
    bad_assets = []
    per_asset = []
    for asset_id, sub in df.groupby("asset_id", sort=False):
        n_true = int(sub["is_warmup"].sum()) if "is_warmup" in sub.columns else 0
        per_asset.append({"asset_id": str(asset_id), "warmup_true": n_true})
        if n_true < burn_in_days:
            bad_assets.append(str(asset_id))
    return {
        "passed": len(bad_assets) == 0,
        "expected_min_per_asset": int(burn_in_days),
        "bad_assets": bad_assets,
        "per_asset": per_asset,
    }


def check_target_internal_gaps(df: pd.DataFrame, horizons: Optional[List[int]] = None) -> Dict:
    if horizons is None:
        horizons = list(TARGET_HORIZONS)
    problems = []
    per_asset = []
    for asset_id, sub in df.groupby("asset_id", sort=False):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        row_info = {"asset_id": str(asset_id)}
        for h in horizons:
            col = f"target_ret_{h}"
            if col not in sub.columns:
                continue
            expected_tail_nan = int(h)
            actual_nan_mask = sub[col].isna()
            actual_nan_count = int(actual_nan_mask.sum())
            internal_nan = max(actual_nan_count - expected_tail_nan, 0)
            row_info[f"{col}_nan_count"] = actual_nan_count
            row_info[f"{col}_internal_nan_est"] = internal_nan
            if internal_nan > 0:
                problems.append({"asset_id": str(asset_id), "col": col, "internal_nan_est": int(internal_nan)})
        per_asset.append(row_info)
    return {"passed": len(problems) == 0, "problems": problems[:50], "per_asset": per_asset}


def check_usable_rows(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    label_col: str = DEFAULT_LABEL_COL,
    required_feature_cols: Optional[List[str]] = None,
) -> Dict:
    if horizons is None:
        horizons = list(TARGET_HORIZONS)
    main_h = max(horizons)

    req_cols = []
    ret_col = f"target_ret_{main_h}"
    if ret_col in df.columns:
        req_cols.append(ret_col)
    if label_col in df.columns:
        req_cols.append(label_col)

    if required_feature_cols:
        req_cols.extend([c for c in required_feature_cols if c in df.columns])

    bad_rows = 0
    if "is_usable_for_model" in df.columns and req_cols:
        usable = df["is_usable_for_model"] == 1
        must_have = df[req_cols].notna().all(axis=1) & (~df["is_warmup"].astype(bool))
        bad_rows = int((usable & ~must_have).sum())

    return {"passed": bad_rows == 0, "bad_rows": bad_rows, "checked_columns": req_cols}


def check_missing_profile(df: pd.DataFrame) -> Dict:
    miss = df.isna().mean().sort_values(ascending=False)
    top20 = {str(k): float(v) for k, v in miss.head(20).to_dict().items()}
    allowed_all_nan = {"target_vol_1"}
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    bad_all_nan_cols = [c for c in all_nan_cols if c not in allowed_all_nan]
    return {
        "passed": len(bad_all_nan_cols) == 0,
        "top20_missing_ratio": top20,
        "all_nan_cols": all_nan_cols,
        "bad_all_nan_cols": bad_all_nan_cols,
    }


def check_label_distribution(df: pd.DataFrame, horizons: Optional[List[int]] = None) -> Dict:
    if horizons is None:
        horizons = list(TARGET_HORIZONS)

    result: Dict[str, Dict] = {}
    for h in horizons:
        item: Dict[str, Dict] = {}
        for split_name, sub in df.groupby("split", dropna=False):
            split_key = str(split_name)
            item[split_key] = {}
            for col in [
                f"target_ret_{h}",
                f"target_dir_{h}",
                f"target_excess_ret_{h}",
                f"target_excess_dir_{h}",
                f"target_band_dir_{h}",
                f"target_band_keep_{h}",
                f"target_vol_{h}",
            ]:
                if col not in sub.columns:
                    continue
                if col.endswith(f"_dir_{h}") or col == f"target_band_keep_{h}":
                    s = pd.to_numeric(sub[col], errors="coerce").dropna()
                    item[split_key][col] = {"count": int(len(s)), "positive_rate": float(s.mean()) if len(s) else None}
                else:
                    item[split_key][col] = _safe_quantiles(sub[col])
        result[str(h)] = item
    return {"passed": True, "by_horizon_and_split": result}


def build_asset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_id, sub in df.groupby("asset_id", sort=False):
        row = {
            "asset_id": str(asset_id),
            "n_rows": int(len(sub)),
            "date_min": str(sub["timestamp"].min().date()),
            "date_max": str(sub["timestamp"].max().date()),
        }
        if "split" in sub.columns:
            for k, v in sub["split"].value_counts().to_dict().items():
                row[f"split_{k}"] = int(v)
        if "is_warmup" in sub.columns:
            row["warmup_true"] = int(sub["is_warmup"].sum())
        if "is_usable_for_model" in sub.columns:
            row["usable_true"] = int(sub["is_usable_for_model"].sum())
        for h in TARGET_HORIZONS:
            for col in [f"target_dir_{h}", f"target_excess_dir_{h}", f"target_band_dir_{h}"]:
                if col in sub.columns:
                    s = pd.to_numeric(sub[col], errors="coerce").dropna()
                    row[f"{col}_positive_rate"] = float(s.mean()) if len(s) else np.nan
            keep_col = f"target_band_keep_{h}"
            if keep_col in sub.columns:
                s = pd.to_numeric(sub[keep_col], errors="coerce").dropna()
                row[f"{keep_col}_mean"] = float(s.mean()) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 总控
# ============================================================
def validate_panel(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    required_feature_cols: Optional[List[str]] = None,
) -> Dict:
    df = _to_time(df, "timestamp")
    groups = _group_columns(list(df.columns))

    checks = {
        "required_columns": check_required_columns(df),
        "no_benchmark_asset": check_no_benchmark_asset(df),
        "primary_key": check_primary_key(df),
        "timestamp_parsing": check_timestamp_parsing(df),
        "time_monotonicity": check_time_monotonicity(df),
        "basic_price_sanity": check_basic_price_sanity(df),
        "split_boundaries": check_split_boundaries(df),
        "warmup_consistency": check_warmup_consistency(df),
        "target_internal_gaps": check_target_internal_gaps(df),
        "usable_rows": check_usable_rows(df, label_col=label_col, required_feature_cols=required_feature_cols),
        "missing_profile": check_missing_profile(df),
        "label_distribution": check_label_distribution(df),
    }

    report: Dict[str, object] = {
        "panel_shape": {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "n_assets": int(df["asset_id"].nunique()) if "asset_id" in df.columns else 0,
            "date_min": str(df["timestamp"].min().date()) if len(df) else "",
            "date_max": str(df["timestamp"].max().date()) if len(df) else "",
        },
        "feature_groups": {k: len(v) for k, v in groups.items()},
        "label_col": label_col,
        "required_feature_cols": required_feature_cols or [],
        "checks": checks,
    }
    passed_all = all(v.get("passed", False) for v in checks.values() if isinstance(v, dict) and "passed" in v)
    report["passed_all"] = bool(passed_all)
    return report


def print_validation_report(report: Dict) -> None:
    panel_shape = report["panel_shape"]
    checks = report["checks"]

    print("=" * 72)
    print("  主面板验证报告")
    print("=" * 72)
    print(f"  总体是否通过: {report['passed_all']}")
    print(f"  行数: {panel_shape['n_rows']:,}")
    print(f"  列数: {panel_shape['n_cols']}")
    print(f"  资产数: {panel_shape['n_assets']}")
    print(f"  日期: {panel_shape['date_min']} ~ {panel_shape['date_max']}")
    print(f"  label_col: {report['label_col']}")
    print(f"  required_feature_cols: {report['required_feature_cols']}")

    print("\n  特征组计数:")
    for k, v in report["feature_groups"].items():
        print(f"    {k:<12} {v}")

    print("\n  关键检查:")
    for name, item in checks.items():
        if isinstance(item, dict) and "passed" in item:
            print(f"    {name:<24} {item['passed']}")

    mp = checks["missing_profile"]
    print("\n  缺失率前 15 列:")
    for i, (k, v) in enumerate(mp["top20_missing_ratio"].items()):
        if i >= 15:
            break
        print(f"    {k:<28} {v:.2%}")

    if mp["bad_all_nan_cols"]:
        print("\n  不允许的全空列:")
        for c in mp["bad_all_nan_cols"]:
            print(f"    {c}")

    ld = checks["label_distribution"]["by_horizon_and_split"]
    print("\n  标签分布摘要:")
    for h, block in ld.items():
        print(f"    Horizon={h}")
        for split_name, detail in block.items():
            cols_to_show = [
                f"target_dir_{h}",
                f"target_excess_dir_{h}",
                f"target_band_dir_{h}",
                f"target_band_keep_{h}",
            ]
            for col in cols_to_show:
                if col in detail and detail[col].get("count", 0):
                    pr = detail[col]["positive_rate"]
                    cnt = detail[col]["count"]
                    print(f"      {split_name:<6} {col:<22} positive_rate={pr:.4f}  n={cnt}")
    print()


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final stock-daily main panel.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--report_json", type=str, default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--asset_summary_csv", type=str, default=str(DEFAULT_ASSET_SUMMARY_CSV))
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--required_feature_cols", type=str, nargs="*", default=[])
    args = parser.parse_args()

    input_path = Path(args.input)
    report_json = Path(args.report_json)
    asset_summary_csv = Path(args.asset_summary_csv)
    df = _load_any(input_path)

    required_feature_cols = []
    for item in args.required_feature_cols:
        for col in str(item).split(","):
            col = col.strip()
            if col:
                required_feature_cols.append(col)
    if not required_feature_cols:
        required_feature_cols = [c for c in get_default_dc_ready_cols() if c in df.columns]

    report = validate_panel(df, label_col=args.label_col, required_feature_cols=required_feature_cols)
    print_validation_report(report)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    asset_summary = build_asset_summary(_to_time(df, "timestamp"))
    asset_summary.to_csv(asset_summary_csv, index=False, encoding="utf-8-sig")

    print(f"  ✅ 已保存 JSON 报告: {report_json}")
    print(f"  ✅ 已保存资产摘要:   {asset_summary_csv}")


if __name__ == "__main__":
    main()

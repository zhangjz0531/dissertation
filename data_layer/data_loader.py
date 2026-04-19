from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data_layer.config import (
    CACHE_DIR,
    DAILY_END,
    DAILY_START,
    RAW_OHLCV_DAILY_PATH,
    STOCK_BENCHMARK,
    STOCK_UNIVERSE,
)

warnings.filterwarnings("ignore")

STANDARD_COLS = ["asset_id", "timestamp", "open", "high", "low", "close", "volume"]
CACHE_TTL_SECONDS = 24 * 3600


# ============================================================
# 缓存工具
# ============================================================
def _make_cache_key(source: str, asset: str, interval: str, start: str, end: str) -> str:
    safe_asset = asset.replace("/", "_").replace("^", "").replace("=", "_")
    return f"{source}__{safe_asset}__{interval}__{start}__{end}"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def _load_cache(key: str, max_age_seconds: int = CACHE_TTL_SECONDS) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > max_age_seconds:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(key), index=False)
    except Exception as e:
        print(f"[cache] 保存失败: {e}")


# ============================================================
# 标准化
# ============================================================
def _standardize_ohlcv(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Date": "timestamp", "Datetime": "timestamp", "date": "timestamp",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        else:
            raise ValueError(f"{asset_id}: 找不到 timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(f"{asset_id}: timestamp 解析失败")

    if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    df["asset_id"] = str(asset_id).upper().strip()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = [c for c in STANDARD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{asset_id}: 缺失标准列 {missing}")

    return (
        df[STANDARD_COLS]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# 基类
# ============================================================
class BaseDataLoader(ABC):
    source_name: str = "base"

    @abstractmethod
    def _fetch_one(self, asset: str, interval: str, start: str, end: str) -> pd.DataFrame:
        ...

    def load(self, asset: str, interval: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
        key = _make_cache_key(self.source_name, asset, interval, start, end)
        if use_cache:
            cached = _load_cache(key)
            if cached is not None and not cached.empty:
                return cached

        raw = self._fetch_one(asset, interval, start, end)
        if raw is None or raw.empty:
            print(f"[{self.source_name}] ⚠️  {asset} 无数据")
            return pd.DataFrame(columns=STANDARD_COLS)

        std = _standardize_ohlcv(raw, asset_id=asset)
        if use_cache:
            _save_cache(key, std)
        return std

    def load_many(self, assets: List[str], interval: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
        frames = []
        for asset in assets:
            df = self.load(asset, interval, start, end, use_cache=use_cache)
            if not df.empty:
                frames.append(df)
            else:
                print(f"[{self.source_name}] ⚠️  {asset} 被跳过")
        if not frames:
            return pd.DataFrame(columns=STANDARD_COLS)
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["asset_id", "timestamp"])
            .reset_index(drop=True)
        )


class YFinanceLoader(BaseDataLoader):
    source_name = "yfinance"

    def _fetch_one(self, asset: str, interval: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            asset,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
            actions=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df


# ============================================================
# 数据质量检查
# ============================================================
def quality_check(df: pd.DataFrame, asset_id: str, interval: str) -> Dict:
    if df.empty:
        return {"asset": asset_id, "status": "EMPTY", "rows": 0, "warnings": ["empty_frame"]}

    report = {
        "asset": asset_id,
        "interval": interval,
        "status": "OK",
        "rows": int(len(df)),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "warnings": [],
    }

    n_dup = int(df["timestamp"].duplicated().sum())
    if n_dup > 0:
        report["warnings"].append(f"duplicate_ts:{n_dup}")

    n_nonpos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if n_nonpos > 0:
        report["warnings"].append(f"nonpositive_price:{n_nonpos}")

    bad_hl = int((df["high"] < df["low"]).sum())
    if bad_hl > 0:
        report["warnings"].append(f"high_lt_low:{bad_hl}")

    n_zero_vol = int((df["volume"] == 0).sum())
    if n_zero_vol > 0:
        report["warnings"].append(f"zero_volume:{n_zero_vol}")

    n_nan = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    if n_nan > 0:
        report["warnings"].append(f"nan_rows:{n_nan}")

    ret = df["close"].pct_change()
    n_jump = int((ret.abs() > 0.30).sum())
    if n_jump > 0:
        report["warnings"].append(f"jump_gt_30pct:{n_jump}")

    if report["warnings"]:
        report["status"] = "WARN"
    return report


def _print_group_report(df: pd.DataFrame, assets: List[str], interval: str) -> None:
    for asset in assets:
        sub = df[df["asset_id"] == str(asset).upper().strip()]
        rep = quality_check(sub, asset, interval)
        warns = ", ".join(rep["warnings"]) if rep["warnings"] else "-"
        print(
            f"  {asset:12s} {rep['status']:4s}  rows={rep['rows']:6d}  "
            f"{rep.get('date_min', '-')[:10]} ~ {rep.get('date_max', '-')[:10]}  {warns}"
        )


# ============================================================
# 主线 API：只构建股票日线主实验原始面板
# ============================================================
def build_raw_daily_stock_panel(use_cache: bool = True) -> Path:
    print("=" * 72)
    print("  股票日线主实验（股票池 + 基准 SPY）")
    print("=" * 72)
    loader = YFinanceLoader()
    assets = STOCK_UNIVERSE + [STOCK_BENCHMARK]
    daily_df = loader.load_many(assets, "1d", DAILY_START, DAILY_END, use_cache=use_cache)

    print(f"总行数 {len(daily_df):,}, 资产数 {daily_df['asset_id'].nunique()}")
    _print_group_report(daily_df, assets, "1d")

    RAW_OHLCV_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(RAW_OHLCV_DAILY_PATH, index=False)
    print(f"✅ 已保存: {RAW_OHLCV_DAILY_PATH}")
    print("   说明: SPY 会保留在原始面板里，用于后续构造 benchmark_ret / excess target。")
    return RAW_OHLCV_DAILY_PATH


def build_raw_panels(use_cache: bool = True, skip_crypto: bool = True) -> Dict[str, Path]:
    """兼容旧调用名；现在只返回股票日线主实验。"""
    path = build_raw_daily_stock_panel(use_cache=use_cache)
    return {"daily_stock": path}


if __name__ == "__main__":
    import sys

    no_cache = "--no-cache" in sys.argv
    build_raw_daily_stock_panel(use_cache=not no_cache)

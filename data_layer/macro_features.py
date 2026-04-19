# data_layer/macro_features.py
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from data_layer.config import (
    DAILY_START, DAILY_END, CRYPTO_HOURLY_START, CRYPTO_HOURLY_END,
    VIX_TICKER, UST10Y_TICKER, HYG_TICKER, DXY_TICKER,
    PROCESSED_DIR, BURN_IN_DAYS,
)
from data_layer.data_loader import YFinanceLoader

warnings.filterwarnings("ignore")


# ============================================================
#  底层：拉一个 ticker 的日频 close 序列
# ============================================================
def _fetch_close(ticker: str, start: str, end: str, col_name: str) -> pd.DataFrame:
    loader = YFinanceLoader()
    df = loader.load(ticker, "1d", start, end, use_cache=True)
    if df.empty:
        print(f"  ⚠️  {ticker} 无数据（输出该列将全为 NaN）")
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), col_name: pd.Series(dtype=float)})
    out = df[["timestamp", "close"]].rename(columns={"close": col_name})
    return out


def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    return (s - m) / (sd + 1e-12)


# ============================================================
#  股票主面板的日频宏观代理
# ============================================================
def build_stock_macro(start: str = DAILY_START, end: str = DAILY_END) -> pd.DataFrame:
    """VIX（风险偏好）+ UST10Y（利率）+ HYG（信用）。"""
    print(f"  拉取 {VIX_TICKER}, {UST10Y_TICKER}, {HYG_TICKER} ...")
    vix = _fetch_close(VIX_TICKER, start, end, "vix_level")
    tnx = _fetch_close(UST10Y_TICKER, start, end, "ust10y")
    hyg = _fetch_close(HYG_TICKER, start, end, "hyg_close")

    df = vix.merge(tnx, on="timestamp", how="outer").merge(hyg, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 日历日补齐：VIX 开盘日和 TNX/HYG 基本一致，但偶有节假日差异，ffill 合理
    df[["vix_level", "ust10y", "hyg_close"]] = df[["vix_level", "ust10y", "hyg_close"]].ffill()

    # ---- VIX 衍生 ----
    df["vix_change_5d"] = df["vix_level"].pct_change(5)
    df["vix_z_60"] = _rolling_z(df["vix_level"], 60)
    df["sentiment_score"] = -np.tanh(df["vix_z_60"] / 2.0)  # -1=恐惧, +1=贪婪

    # ---- 利率衍生 ----
    df["ust10y_change_5d"] = df["ust10y"].pct_change(5)
    rate_z = _rolling_z(df["ust10y_change_5d"], 60)

    # ---- 信用压力（HYG 从过去 60 天高点的相对回撤）----
    hyg_peak_60 = df["hyg_close"].rolling(60, min_periods=60).max()
    df["credit_stress"] = (hyg_peak_60 - df["hyg_close"]) / (hyg_peak_60.abs() + 1e-12)
    df["hyg_change_5d"] = df["hyg_close"].pct_change(5)
    stress_z = _rolling_z(df["credit_stress"], 60)

    # ---- 综合宏观压力 ----
    df["macro_pressure"] = np.tanh(0.5 * rate_z + 0.5 * stress_z)

    keep = [
        "timestamp",
        "vix_level", "vix_change_5d", "vix_z_60", "sentiment_score",
        "ust10y", "ust10y_change_5d",
        "credit_stress", "hyg_change_5d",
        "macro_pressure",
    ]
    return df[keep].replace([np.inf, -np.inf], np.nan)


# ============================================================
#  加密对照面板的日频宏观代理
# ============================================================
def build_crypto_macro(start: str = CRYPTO_HOURLY_START, end: str = CRYPTO_HOURLY_END) -> pd.DataFrame:
    """DXY（美元强弱）+ VIX（全球风险偏好）。"""
    print(f"  拉取 {DXY_TICKER}, {VIX_TICKER} ...")
    dxy = _fetch_close(DXY_TICKER, start, end, "dxy_level")
    vix = _fetch_close(VIX_TICKER, start, end, "vix_level")

    df = dxy.merge(vix, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    df[["dxy_level", "vix_level"]] = df[["dxy_level", "vix_level"]].ffill()

    df["dxy_change_5d"] = df["dxy_level"].pct_change(5)
    df["dxy_z_60"] = _rolling_z(df["dxy_level"], 60)

    df["vix_change_5d"] = df["vix_level"].pct_change(5)
    df["vix_z_60"] = _rolling_z(df["vix_level"], 60)

    return df[[
        "timestamp",
        "dxy_level", "dxy_change_5d", "dxy_z_60",
        "vix_level", "vix_change_5d", "vix_z_60",
    ]].replace([np.inf, -np.inf], np.nan)


# ============================================================
#  工具：把宏观特征 asof 合并到资产面板（下游 panel_builder 会调用）
# ============================================================
def attach_macro_asof(asset_panel: pd.DataFrame, macro_panel: pd.DataFrame) -> pd.DataFrame:
    """用 merge_asof backward 把宽表宏观特征对齐到资产面板。
    小时线资产面板也能用：merge_asof 会把每个小时对齐到"当时最近的"日频宏观值。"""
    if asset_panel.empty or macro_panel.empty:
        return asset_panel.copy()

    a = asset_panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True).copy()
    m = macro_panel.sort_values("timestamp").reset_index(drop=True).copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    m["timestamp"] = pd.to_datetime(m["timestamp"])

    parts = []
    for aid, g in a.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(g, m, on="timestamp", direction="backward")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


# ============================================================
#  Sanity check
# ============================================================
def _sanity_check(df: pd.DataFrame, cols: list, label: str) -> None:
    body = df.iloc[BURN_IN_DAYS:]
    print(f"  Sanity check @ {label} (burn-in 之后):")
    head_nan = df.iloc[:BURN_IN_DAYS][cols].isna().sum().sum()
    head_cells = BURN_IN_DAYS * len(cols)
    tail_nan = body[cols].isna().sum().sum()
    tail_cells = max(1, len(body)) * len(cols)
    print(f"    前 {BURN_IN_DAYS} 行 NaN 占比 {head_nan}/{head_cells} = {head_nan/head_cells*100:.1f}%")
    print(f"    剩余行 NaN 占比 {tail_nan}/{tail_cells} = {tail_nan/tail_cells*100:.2f}%")
    for c in cols:
        s = body[c].dropna()
        if len(s):
            print(f"    {c:22s} [{s.min():+.4f}, {s.max():+.4f}]  mean={s.mean():+.4f}")


# ============================================================
#  一键构建
# ============================================================
def build_macro_panels() -> dict:
    results = {}

    print("=" * 72)
    print("  股票面板日频宏观代理（VIX + UST10Y + HYG）")
    print("=" * 72)
    stock_macro = build_stock_macro()
    print(f"  输出: {len(stock_macro):,} 行, {len(stock_macro.columns)} 列")
    print(f"  日期: {stock_macro['timestamp'].min().date()} ~ {stock_macro['timestamp'].max().date()}")
    _sanity_check(stock_macro, ["vix_level", "sentiment_score", "ust10y", "credit_stress", "macro_pressure"], "stock_macro")
    path = PROCESSED_DIR / "macro_stock.parquet"
    stock_macro.to_parquet(path, index=False)
    print(f"  ✅ 已保存: {path}\n")
    results["stock"] = path

    print("=" * 72)
    print("  加密对照面板日频宏观代理（DXY + VIX）")
    print("=" * 72)
    crypto_macro = build_crypto_macro()
    if crypto_macro.empty or crypto_macro["dxy_level"].isna().all():
        print("  ⚠️  DXY 未拿到数据。可能原因：yfinance 对 'DX-Y.NYB' 偶发访问受限。")
        print("     建议 fallback：去 config.py 改 DXY_TICKER = 'UUP'（做多美元 ETF，作为代理）后重跑。")
    else:
        print(f"  输出: {len(crypto_macro):,} 行, {len(crypto_macro.columns)} 列")
        print(f"  日期: {crypto_macro['timestamp'].min().date()} ~ {crypto_macro['timestamp'].max().date()}")
        _sanity_check(crypto_macro, ["dxy_level", "dxy_z_60", "vix_level", "vix_z_60"], "crypto_macro")
        path = PROCESSED_DIR / "macro_crypto.parquet"
        crypto_macro.to_parquet(path, index=False)
        print(f"  ✅ 已保存: {path}")
        results["crypto"] = path

    return results


if __name__ == "__main__":
    build_macro_panels()
# data_layer/technical_features.py
from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd

from data_layer.config import (
    RETURN_LAGS, VOL_WINDOWS, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_WINDOW, MA_WINDOWS, VOLUME_Z_WINDOW,
    RAW_OHLCV_DAILY_PATH, RAW_OHLCV_HOURLY_STOCK_PATH, RAW_OHLCV_HOURLY_CRYPTO_PATH,
    PROCESSED_DIR, BURN_IN_DAYS,
)

warnings.filterwarnings("ignore")

# 布林带参数
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2.0


# ============================================================
#  原子指标
# ============================================================
def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder RSI（alpha = 1/window 的 EWM 平滑）。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """平均真实波幅。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


# ============================================================
#  单资产计算
# ============================================================
def _compute_for_group(g: pd.DataFrame) -> pd.DataFrame:
    """对单个 asset_id 的 OHLCV 子表计算所有技术特征。
    假定 g 已经按 timestamp 升序排列。保留 NaN，不做填充。"""
    g = g.copy()

    close = pd.to_numeric(g["close"], errors="coerce")
    high = pd.to_numeric(g["high"], errors="coerce")
    low = pd.to_numeric(g["low"], errors="coerce")
    volume = pd.to_numeric(g["volume"], errors="coerce")

    # ---- 1. 收益率（多 horizon）----
    for lag in RETURN_LAGS:
        g[f"ret_{lag}"] = close.pct_change(lag)
    g["logret_1"] = np.log(close.replace(0, np.nan)).diff()

    # ---- 2. 波动率 ----
    ret_1 = g["ret_1"]
    for w in VOL_WINDOWS:
        g[f"vol_{w}"] = ret_1.rolling(w, min_periods=w).std()

    # ---- 3. RSI ----
    g[f"rsi_{RSI_WINDOW}"] = _rsi(close, RSI_WINDOW)

    # ---- 4. MACD（只保留 hist 相关，丢弃中间量）----
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    g["macd_hist"] = macd - macd_sig
    g["macd_hist_pct"] = g["macd_hist"] / (close.abs() + 1e-12)

    # ---- 5. 均线偏离率 ----
    for w in MA_WINDOWS:
        sma = close.rolling(w, min_periods=w).mean()
        g[f"ma_ratio_{w}"] = close / (sma + 1e-12) - 1.0

    # ---- 6. ATR（只保留比例）----
    atr = _atr(high, low, close, ATR_WINDOW)
    g[f"atr_pct_{ATR_WINDOW}"] = atr / (close.abs() + 1e-12)
    g["hl_range"] = (high - low) / (close.abs() + 1e-12)

    # ---- 7. 布林带 ----
    ma_bb = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    std_bb = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
    upper = ma_bb + BOLLINGER_NUM_STD * std_bb
    lower = ma_bb - BOLLINGER_NUM_STD * std_bb
    g["bb_pctb"] = (close - lower) / (upper - lower + 1e-12)
    g["bb_bw"] = (upper - lower) / (ma_bb.abs() + 1e-12)

    # ---- 8. 成交量 z-score ----
    vm = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    vs = volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std()
    g[f"volume_z_{VOLUME_Z_WINDOW}"] = (volume - vm) / (vs + 1e-12)

    dollar_volume = close * volume
    dvm = dollar_volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).mean()
    dvs = dollar_volume.rolling(VOLUME_Z_WINDOW, min_periods=VOLUME_Z_WINDOW).std()
    g[f"dollar_vol_z_{VOLUME_Z_WINDOW}"] = (dollar_volume - dvm) / (dvs + 1e-12)

    return g.replace([np.inf, -np.inf], np.nan)


# ============================================================
#  主入口
# ============================================================
def compute_technical_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """对 STANDARD_COLS 长表计算技术特征，输入输出都是长表。"""
    if ohlcv_df.empty:
        return ohlcv_df.copy()

    df = ohlcv_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    # 显式循环 groupby，避开 pandas 2.x apply 的 FutureWarning 干扰
    parts: List[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        parts.append(_compute_for_group(g))
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return out


def list_technical_feature_cols(df: pd.DataFrame) -> List[str]:
    """返回除 OHLCV 和元数据外的所有技术特征列名，给下游参考用。"""
    exclude = {"asset_id", "timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]


def _quick_sanity_check(df: pd.DataFrame, asset_sample: str) -> None:
    """抽一只资产检查前 burn-in 行的 NaN 分布 + burn-in 之后的特征值范围。"""
    sub = df[df["asset_id"] == asset_sample].reset_index(drop=True)
    feat_cols = list_technical_feature_cols(df)
    if sub.empty:
        return

    # burn-in 之前的行（前 60 行）NaN 应该很多
    head_nan = sub.iloc[:BURN_IN_DAYS][feat_cols].isna().sum().sum()
    head_cells = BURN_IN_DAYS * len(feat_cols)
    # burn-in 之后应该几乎没有 NaN
    tail_nan = sub.iloc[BURN_IN_DAYS:][feat_cols].isna().sum().sum()
    tail_cells = (len(sub) - BURN_IN_DAYS) * len(feat_cols)

    print(f"  Sanity check @ {asset_sample}:")
    print(f"    前 {BURN_IN_DAYS} 行 NaN 占比 {head_nan}/{head_cells} = {head_nan/head_cells*100:.1f}%  (预期较高)")
    print(f"    剩余行 NaN 占比 {tail_nan}/{tail_cells} = {tail_nan/tail_cells*100:.2f}%  (预期 ~ 0%)")

    # 几个关键特征值范围
    body = sub.iloc[BURN_IN_DAYS:]
    def _rng(c):
        s = body[c].dropna()
        return f"[{s.min():.4f}, {s.max():.4f}]  mean={s.mean():.4f}" if len(s) else "n/a"

    for c in [f"rsi_{RSI_WINDOW}", "bb_pctb", f"volume_z_{VOLUME_Z_WINDOW}", "macd_hist_pct"]:
        if c in body.columns:
            print(f"    {c:20s} {_rng(c)}")


def build_technical_panels() -> dict:
    results = {}
    tasks = [
        ("股票日线", RAW_OHLCV_DAILY_PATH, "tech_daily_stock.parquet"),
        ("股票小时线", RAW_OHLCV_HOURLY_STOCK_PATH, "tech_hourly_stock.parquet"),
        ("加密小时线", RAW_OHLCV_HOURLY_CRYPTO_PATH, "tech_hourly_crypto.parquet"),
    ]

    for name, raw_path, out_name in tasks:
        print("=" * 72)
        print(f"  {name}")
        print("=" * 72)
        if not raw_path.exists():
            print(f"  ⚠️  跳过（未找到 {raw_path.name}，请先运行 data_loader）\n")
            continue

        raw_df = pd.read_parquet(raw_path)
        print(f"  输入:  {len(raw_df):,} 行, {raw_df['asset_id'].nunique()} 资产, {len(raw_df.columns)} 列")

        feat_df = compute_technical_features(raw_df)
        feat_cols = list_technical_feature_cols(feat_df)
        print(f"  输出:  {len(feat_df):,} 行, {len(feat_df.columns)} 列 (含 {len(feat_cols)} 个技术特征)")

        out_path = PROCESSED_DIR / out_name
        feat_df.to_parquet(out_path, index=False)

        # 抽一只资产做 sanity check
        sample_asset = feat_df["asset_id"].iloc[0]
        _quick_sanity_check(feat_df, sample_asset)

        print(f"  ✅ 已保存: {out_path}\n")
        results[name] = out_path

    return results


if __name__ == "__main__":
    build_technical_panels()
# data_layer/dc_engine.py
from __future__ import annotations

import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from data_layer.config import (
    theta_to_bp,
    DC_THETAS_DAILY, DC_THETAS_HOURLY_STOCK, DC_THETAS_HOURLY_CRYPTO,
    PROCESSED_DIR, DIAGNOSTIC_DIR,
    RAW_OHLCV_DAILY_PATH, RAW_OHLCV_HOURLY_STOCK_PATH, RAW_OHLCV_HOURLY_CRYPTO_PATH,
)

warnings.filterwarnings("ignore")

# 滚动事件密度窗口（bar 数）
DC_DENSITY_WINDOW = 20


# ============================================================
#  核心扫描器：单资产单 θ，一次遍历产出所有结果
# ============================================================
def _scan_dc(
        prices: np.ndarray,
        timestamps: np.ndarray,
        theta: float,
        asset_id: str,
) -> Dict:
    """对单条价格序列做 DC 扫描。
    返回 dict:
        trend, event, tmv, osv, age  -> 5 个 point-in-time 数组
        events                       -> 事件列表（list of dict）

    Point-in-time 特征的含义：
        trend:  当前趋势方向 (+1 上 / -1 下 / 0 首事件前)
        event:  本 bar 是否有 DC 事件 (+1/-1/0)
        tmv:    从上一个对向极值点的有符号 % 移动
        osv:    从上一个 DC 事件确认价的有符号 % 移动
        age:    距上一个 DC 事件的 bar 数
    首事件发生前，tmv/osv/age 全为 NaN。
    """
    n = len(prices)
    trend = np.zeros(n, dtype=np.int8)
    event = np.zeros(n, dtype=np.int8)
    tmv = np.full(n, np.nan, dtype=np.float32)
    osv = np.full(n, np.nan, dtype=np.float32)
    age = np.full(n, np.nan, dtype=np.float32)
    events: List[dict] = []

    if n == 0:
        return {"trend": trend, "event": event, "tmv": tmv, "osv": osv, "age": age, "events": events}

    # 状态变量
    cur_trend = 0  # 首事件前 = 0
    p_ext_current = prices[0]  # 当前趋势内的运行极值
    cur_ext_idx = 0
    p_ext_previous = prices[0]  # 触发当前趋势的上一个对向极值
    prev_ext_idx = 0
    last_event_price = np.nan
    last_event_idx = -1

    # Bootstrap 追踪器
    p_max_so_far = prices[0]
    p_min_so_far = prices[0]
    idx_max_so_far = 0
    idx_min_so_far = 0

    for i in range(n):
        p = prices[i]
        if not np.isfinite(p):
            # 价格缺失：趋势延续，其他特征保持 NaN
            if i > 0:
                trend[i] = trend[i - 1]
            continue

        event_at_bar = 0

        if cur_trend == 0:
            # ---- Bootstrap：同时追 max 和 min ----
            if p > p_max_so_far:
                p_max_so_far, idx_max_so_far = p, i
            if p < p_min_so_far:
                p_min_so_far, idx_min_so_far = p, i

            if p_max_so_far > 0 and p <= p_max_so_far * (1.0 - theta):
                # 首事件：下转
                cur_trend = -1
                event_at_bar = -1
                p_ext_previous = p_max_so_far
                prev_ext_idx = idx_max_so_far
                p_ext_current = p
                cur_ext_idx = i
                last_event_price = p
                last_event_idx = i
            elif p_min_so_far > 0 and p >= p_min_so_far * (1.0 + theta):
                # 首事件：上转
                cur_trend = +1
                event_at_bar = +1
                p_ext_previous = p_min_so_far
                prev_ext_idx = idx_min_so_far
                p_ext_current = p
                cur_ext_idx = i
                last_event_price = p
                last_event_idx = i
        else:
            # ---- 正常状态 ----
            if cur_trend == +1:
                if p > p_ext_current:
                    p_ext_current, cur_ext_idx = p, i
                if p <= p_ext_current * (1.0 - theta):
                    # 下转事件
                    event_at_bar = -1
                    p_ext_previous = p_ext_current
                    prev_ext_idx = cur_ext_idx
                    cur_trend = -1
                    p_ext_current = p
                    cur_ext_idx = i
                    last_event_price = p
                    last_event_idx = i
            else:
                if p < p_ext_current:
                    p_ext_current, cur_ext_idx = p, i
                if p >= p_ext_current * (1.0 + theta):
                    # 上转事件
                    event_at_bar = +1
                    p_ext_previous = p_ext_current
                    prev_ext_idx = cur_ext_idx
                    cur_trend = +1
                    p_ext_current = p
                    cur_ext_idx = i
                    last_event_price = p
                    last_event_idx = i

        trend[i] = cur_trend
        event[i] = event_at_bar

        if event_at_bar != 0:
            events.append({
                "asset_id": asset_id,
                "theta": theta,
                "event_idx": i,
                "event_time": timestamps[i],
                "event_type": int(event_at_bar),
                "event_price": float(p),
                "prev_ext_idx": int(prev_ext_idx),
                "prev_ext_time": timestamps[prev_ext_idx],
                "prev_ext_price": float(p_ext_previous),
                "dc_duration_bars": int(i - prev_ext_idx),
                "dc_tmv_raw": float((p - p_ext_previous) / p_ext_previous),
            })

        # Point-in-time 特征（首事件后才计算）
        if cur_trend != 0 and p_ext_previous > 0 and last_event_idx >= 0:
            tmv[i] = (p - p_ext_previous) / p_ext_previous
            if last_event_price > 0:
                osv[i] = (p - last_event_price) / last_event_price
            age[i] = float(i - last_event_idx)

    return {
        "trend": trend, "event": event,
        "tmv": tmv, "osv": osv, "age": age,
        "events": events,
    }


# ============================================================
#  包装：对所有资产和所有 θ 计算 DC 特征
# ============================================================
def compute_dc_features(
        ohlcv_df: pd.DataFrame,
        thetas: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """输入 STANDARD_COLS 长表，输出：
        - feature_df: 原表 + 每个 θ 的 6 个 DC 特征
        - events_df: 全部事件列表
    """
    if ohlcv_df.empty:
        return ohlcv_df.copy(), pd.DataFrame()

    all_events = []
    feature_parts = []

    for asset_id, g in ohlcv_df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        prices = pd.to_numeric(g["close"], errors="coerce").to_numpy(dtype=np.float64)
        timestamps = g["timestamp"].to_numpy()

        for theta in thetas:
            res = _scan_dc(prices, timestamps, theta, asset_id)
            tag = theta_to_bp(theta)

            g[f"dc_event_{tag}"] = res["event"]
            g[f"dc_trend_{tag}"] = res["trend"]
            g[f"dc_tmv_{tag}"] = res["tmv"]
            g[f"dc_osv_{tag}"] = res["osv"]
            g[f"dc_age_{tag}"] = res["age"]

            event_abs = np.abs(res["event"]).astype(np.float32)
            density = (
                pd.Series(event_abs)
                .rolling(DC_DENSITY_WINDOW, min_periods=1)
                .sum()
                .to_numpy(dtype=np.float32)
            )
            g[f"dc_density_{DC_DENSITY_WINDOW}_{tag}"] = density

            all_events.extend(res["events"])

        feature_parts.append(g)

    feature_df = pd.concat(feature_parts, ignore_index=True)
    feature_df = feature_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    events_df = pd.DataFrame.from_records(all_events)
    if not events_df.empty:
        events_df["theta_bp"] = events_df["theta"].apply(theta_to_bp)
        events_df = events_df.sort_values(["asset_id", "theta", "event_idx"]).reset_index(drop=True)

    return feature_df, events_df


# ============================================================
#  θ 敏感性报告
# ============================================================
def theta_sensitivity_report(
        ohlcv_df: pd.DataFrame,
        events_df: pd.DataFrame,
) -> pd.DataFrame:
    """对每个 (asset, θ) 统计事件密度与幅度分布。论文里直接用。"""
    if events_df.empty:
        return pd.DataFrame()

    obs = ohlcv_df.groupby("asset_id")["timestamp"].count().rename("n_bars")

    rows = []
    for (asset_id, theta), grp in events_df.groupby(["asset_id", "theta"]):
        n_events = len(grp)
        n_up = int((grp["event_type"] == +1).sum())
        n_dn = int((grp["event_type"] == -1).sum())
        avg_dur = float(grp["dc_duration_bars"].mean())
        avg_abs_tmv = float(grp["dc_tmv_raw"].abs().mean())
        n_bars = int(obs.get(asset_id, 0))

        rows.append({
            "asset_id": asset_id,
            "theta": theta,
            "theta_bp": theta_to_bp(theta),
            "n_bars": n_bars,
            "n_events": n_events,
            "n_up": n_up,
            "n_down": n_dn,
            "events_per_1000_bars": round(n_events / n_bars * 1000, 2) if n_bars else 0.0,
            "avg_dc_duration_bars": round(avg_dur, 2),
            "avg_abs_tmv_pct": round(avg_abs_tmv * 100, 3),
        })

    return pd.DataFrame(rows).sort_values(["asset_id", "theta"]).reset_index(drop=True)


# ============================================================
#  Sanity check：一致性校验
# ============================================================
def _consistency_check(feat_df: pd.DataFrame, events_df: pd.DataFrame, thetas: List[float]) -> None:
    """验证 feature_df 中的 dc_event 非零数量 == events_df 中的事件数。"""
    print("  一致性校验 (feature_df 事件标记 vs events_df 事件记录):")
    for theta in thetas:
        tag = theta_to_bp(theta)
        col = f"dc_event_{tag}"
        for asset_id, sub in feat_df.groupby("asset_id"):
            n_marks = int((sub[col] != 0).sum())
            n_records = int(
                ((events_df["asset_id"] == asset_id) & (events_df["theta"] == theta)).sum()
            )
            status = "✓" if n_marks == n_records else "✗ MISMATCH"
            print(f"    {asset_id:12s} θ={tag}  marks={n_marks:5d}  records={n_records:5d}  {status}")


# ============================================================
#  一键构建
# ============================================================
def build_dc_panels() -> None:
    tasks = [
        ("股票日线", RAW_OHLCV_DAILY_PATH, DC_THETAS_DAILY,
         "dc_daily_stock.parquet", "dc_events_daily_stock.parquet", "dc_sensitivity_daily_stock.csv"),
        ("股票小时线", RAW_OHLCV_HOURLY_STOCK_PATH, DC_THETAS_HOURLY_STOCK,
         "dc_hourly_stock.parquet", "dc_events_hourly_stock.parquet", "dc_sensitivity_hourly_stock.csv"),
        ("加密小时线", RAW_OHLCV_HOURLY_CRYPTO_PATH, DC_THETAS_HOURLY_CRYPTO,
         "dc_hourly_crypto.parquet", "dc_events_hourly_crypto.parquet", "dc_sensitivity_hourly_crypto.csv"),
    ]

    for name, raw_path, thetas, feat_name, events_name, sens_name in tasks:
        print("=" * 72)
        print(f"  {name}   θ = {thetas}")
        print("=" * 72)
        if not raw_path.exists():
            print(f"  ⚠️  跳过（未找到 {raw_path.name}）\n")
            continue

        raw_df = pd.read_parquet(raw_path)
        print(f"  输入: {len(raw_df):,} 行, {raw_df['asset_id'].nunique()} 资产")

        feat_df, events_df = compute_dc_features(raw_df, thetas)

        feat_path = PROCESSED_DIR / feat_name
        events_path = PROCESSED_DIR / events_name
        feat_df.to_parquet(feat_path, index=False)
        events_df.to_parquet(events_path, index=False)
        print(f"  ✅ 特征: {feat_path}  ({len([c for c in feat_df.columns if c.startswith('dc_')])} 个 DC 特征列)")
        print(f"  ✅ 事件: {events_path}  ({len(events_df):,} 个事件)")

        _consistency_check(feat_df, events_df, thetas)

        print("\n  θ 敏感性摘要:")
        summary = theta_sensitivity_report(raw_df, events_df)
        print(summary.to_string(index=False))

        sens_path = DIAGNOSTIC_DIR / sens_name
        summary.to_csv(sens_path, index=False)
        print(f"\n  ✅ 敏感性报告: {sens_path}\n")


if __name__ == "__main__":
    build_dc_panels()
from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------------
# Project-root import bootstrap
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import (
    RAW_STOCK_PANEL_PATH,
    MODEL_READY_PANEL_PATH,
    BENCHMARK_STATE_PATH,
    FEATURE_STATS_PATH,
    DATA_ENG_SUMMARY_PATH,
    ensure_all_core_dirs,
)

# -----------------------------
# Flexible analyst imports
# Works for either *_revised.py or plain *.py
# -----------------------------
try:
    from analyst_fundamental import FundamentalAnalyst
except Exception:
    from analyst_fundamental import FundamentalAnalyst

try:
    from analyst_news import NewsAnalyst
except Exception:
    from analyst_news import NewsAnalyst

try:
    from analyst_sentiment import SentimentAnalyst
except Exception:
    from analyst_sentiment import SentimentAnalyst

try:
    from analyst_technical import TechnicalAnalyst
except Exception:
    from analyst_technical import TechnicalAnalyst


DEFAULT_STOCKS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "UNH", "V"]
DEFAULT_MACRO = "SPY"
DEFAULT_START = "2010-01-01"
DEFAULT_END = datetime.datetime.now().strftime("%Y-%m-%d")
DEFAULT_TRAIN_SPLIT_DATE = "2022-01-01"
DEFAULT_TARGET_HORIZON = 5
DEFAULT_DC_THRESHOLD = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description="Build raw stock panel for dissertation.")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--macro", type=str, default=DEFAULT_MACRO)
    parser.add_argument("--target_horizon", type=int, default=DEFAULT_TARGET_HORIZON)
    parser.add_argument("--dc_threshold", type=float, default=DEFAULT_DC_THRESHOLD)
    parser.add_argument("--train_split_date", type=str, default=DEFAULT_TRAIN_SPLIT_DATE)
    parser.add_argument(
        "--stocks",
        nargs="*",
        default=DEFAULT_STOCKS,
        help="Stock universe, e.g. AAPL MSFT NVDA ..."
    )
    return parser.parse_args()


class DataEngineeringAgent:
    def __init__(self, dc_threshold: float, target_horizon: int, train_split_date: str):
        self.name = "Data Engineering Agent"
        self.dc_threshold = float(dc_threshold)
        self.target_horizon = int(target_horizon)
        self.train_split_date = str(train_split_date)

    def extract_dc_features(self, df: pd.DataFrame, theta: float | None = None) -> pd.DataFrame:
        theta = theta or self.dc_threshold
        g = df.sort_values("date").copy()
        prices = pd.to_numeric(g["close"], errors="coerce").astype(float).values
        n = len(prices)

        if n == 0:
            return g

        dc_trend = np.zeros(n, dtype=np.int8)
        dc_event = np.zeros(n, dtype=np.int8)
        dc_run_length = np.zeros(n, dtype=np.float32)
        dc_dist_extreme = np.zeros(n, dtype=np.float32)
        dc_tmv = np.zeros(n, dtype=np.float32)

        uptrend = True
        ph = prices[0]
        pl = prices[0]
        ph_idx = 0
        pl_idx = 0
        dc_trend[0] = 1

        for i in range(1, n):
            p = prices[i]
            if np.isnan(p):
                p = prices[i - 1]

            if uptrend:
                if p >= ph:
                    ph = p
                    ph_idx = i
                elif p <= ph * (1 - theta):
                    uptrend = False
                    pl = p
                    pl_idx = i
                    dc_event[i] = -1
            else:
                if p <= pl:
                    pl = p
                    pl_idx = i
                elif p >= pl * (1 + theta):
                    uptrend = True
                    ph = p
                    ph_idx = i
                    dc_event[i] = 1

            dc_trend[i] = 1 if uptrend else -1
            if uptrend:
                dc_run_length[i] = i - ph_idx
                dc_dist_extreme[i] = (p - ph) / (abs(ph) + 1e-6)
                dc_tmv[i] = (p - pl) / (abs(pl) + 1e-6)
            else:
                dc_run_length[i] = i - pl_idx
                dc_dist_extreme[i] = (p - pl) / (abs(pl) + 1e-6)
                dc_tmv[i] = (ph - p) / (abs(ph) + 1e-6)

        g["dc_event"] = dc_event.astype(np.float32)
        g["dc_trend"] = dc_trend.astype(np.float32)
        g["dc_run_length"] = dc_run_length.astype(np.float32)
        g["dc_dist_extreme"] = dc_dist_extreme.astype(np.float32)
        g["dc_tmv"] = dc_tmv.astype(np.float32)
        g["dc_event_density_20"] = (
            pd.Series(np.abs(dc_event)).rolling(20, min_periods=1).sum().values.astype(np.float32)
        )
        return g

    @staticmethod
    def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        g = df.sort_values("date").copy()
        g["future_return_1d"] = g["close"].shift(-1) / (g["close"] + 1e-6) - 1.0
        g[f"future_return_{horizon}d"] = g["close"].shift(-horizon) / (g["close"] + 1e-6) - 1.0
        g["target_up_1d"] = (g["future_return_1d"] > 0).astype(int)
        g[f"target_up_{horizon}d"] = (g[f"future_return_{horizon}d"] > 0).astype(int)
        return g

    @staticmethod
    def cross_sectional_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        def _zscore(group: pd.DataFrame) -> pd.DataFrame:
            out = group.copy()
            for col in cols:
                x = pd.to_numeric(out[col], errors="coerce")
                std = x.std()
                out[f"{col}_cs_z"] = ((x - x.mean()) / (std + 1e-6)).fillna(0.0)
            return out

        return df.groupby("date", group_keys=False).apply(_zscore)

    @staticmethod
    def _safe_numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return df

    def run(self, start: str, end: str, stocks: List[str], macro: str) -> None:
        ensure_all_core_dirs()

        print("==================================================")
        print("  🏢 Analyst Team + Data Engineering Agent 开始处理股票数据")
        print("==================================================\n")

        fund_agent = FundamentalAnalyst(filing_lag_days=45)
        fund_df = fund_agent.run(start, end, stocks, macro)
        if fund_df.empty:
            raise RuntimeError("基础数据为空，无法继续。")

        tech_agent = TechnicalAnalyst()
        tech_df = tech_agent.run(fund_df)

        print(f"[{self.name}] 🧠 正在为每只股票提取 DC 特征...")
        tech_df = tech_df.groupby("stock", group_keys=False).apply(
            lambda g: self.extract_dc_features(g, self.dc_threshold)
        )
        tech_df = tech_df.reset_index(drop=True)

        sent_agent = SentimentAnalyst()
        sent_df = sent_agent.run(start, end)

        news_agent = NewsAnalyst()
        news_df = news_agent.run(start, end)

        print(f"[{self.name}] ⚙️ 正在合并技术面、基本面、情绪与宏观代理特征...")
        master_df = tech_df.merge(sent_df, on="date", how="left").merge(news_df, on="date", how="left")
        master_df = self._safe_numeric_clean(master_df)
        master_df = master_df.sort_values(["stock", "date"]).reset_index(drop=True)

        numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
        fill_cols = list(numeric_cols)
        master_df[fill_cols] = master_df.groupby("stock")[fill_cols].ffill()
        master_df[fill_cols] = master_df[fill_cols].fillna(0.0)

        benchmark = macro
        benchmark_cols = [
            "date",
            "return_1d",
            "return_5d",
            "return_21d",
            "volatility_20d",
            "dc_event",
            "dc_trend",
            "dc_run_length",
            "dc_tmv",
            "sentiment_score",
            "macro_news_pressure",
            "interest_rate",
            "credit_stress",
        ]
        benchmark_df = master_df[master_df["stock"] == benchmark][benchmark_cols].copy()
        benchmark_df = benchmark_df.rename(columns={c: f"mkt_{c}" for c in benchmark_cols if c != "date"})

        panel_df = master_df[master_df["stock"] != benchmark].copy()
        panel_df = panel_df.merge(benchmark_df, on="date", how="left")

        cs_cols = [
            "return_21d",
            "rsi_14",
            "macd_hist_pct",
            "net_margin",
            "operating_margin",
            "revenue_growth_qoq",
            "debt_to_equity",
        ]
        cs_cols = [c for c in cs_cols if c in panel_df.columns]
        panel_df = self.cross_sectional_features(panel_df, cs_cols)

        print(f"[{self.name}] 🎯 正在生成预测标签...")
        panel_df = panel_df.groupby("stock", group_keys=False).apply(
            lambda g: self.add_targets(g, self.target_horizon)
        )
        panel_df = panel_df.reset_index(drop=True)

        continuous_features = [
            "open", "high", "low", "close", "volume",
            "return_1d", "log_return_1d", "return_5d", "return_21d",
            "volatility_20d",
            "rsi_14",
            "macd", "macd_signal", "macd_hist", "macd_hist_pct",
            "ma10_ratio", "ma20_ratio", "ma60_ratio",
            "atr_14", "atr_pct",
            "hl_spread",
            "volume_z_20",
            "net_margin", "operating_margin", "revenue_growth_qoq", "debt_to_equity", "asset_turnover",
            "vix_level", "vix_change_5d", "vix_z_60", "sentiment_score",
            "interest_rate", "rate_change_5d", "credit_stress", "hyg_return_5d", "macro_news_pressure",
            "dc_run_length", "dc_dist_extreme", "dc_tmv", "dc_event_density_20",
            "mkt_return_1d", "mkt_return_5d", "mkt_return_21d", "mkt_volatility_20d",
            "mkt_dc_run_length", "mkt_dc_tmv",
        ]
        continuous_features += [f"{c}_cs_z" for c in cs_cols]
        continuous_features = [c for c in continuous_features if c in panel_df.columns]

        discrete_features = [
            "has_fundamental_data",
            "dc_event",
            "dc_trend",
            "mkt_dc_event",
            "mkt_dc_trend",
        ]
        discrete_features = [c for c in discrete_features if c in panel_df.columns]

        train_split = pd.to_datetime(self.train_split_date)
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        train_mask = panel_df["date"] < train_split

        train_ref = panel_df.loc[train_mask, continuous_features].copy()
        means = train_ref.mean().fillna(0.0)
        stds = train_ref.std().replace(0, 1.0).fillna(1.0)

        model_df = panel_df.copy()
        model_df[continuous_features] = (model_df[continuous_features] - means) / stds
        model_df[continuous_features] = (
            model_df[continuous_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

        np.savez(
            FEATURE_STATS_PATH,
            mean=means.values.astype(np.float32),
            std=stds.values.astype(np.float32),
            continuous_features=np.array(continuous_features, dtype=object),
            discrete_features=np.array(discrete_features, dtype=object),
        )

        save_panel_df = panel_df.copy()
        save_model_df = model_df.copy()
        save_benchmark_df = benchmark_df.copy()

        save_panel_df["date"] = pd.to_datetime(save_panel_df["date"]).dt.strftime("%Y-%m-%d")
        save_model_df["date"] = pd.to_datetime(save_model_df["date"]).dt.strftime("%Y-%m-%d")
        save_benchmark_df["date"] = pd.to_datetime(save_benchmark_df["date"]).dt.strftime("%Y-%m-%d")

        RAW_STOCK_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_READY_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

        save_panel_df.to_csv(RAW_STOCK_PANEL_PATH, index=False)
        save_model_df.to_csv(MODEL_READY_PANEL_PATH, index=False)
        save_benchmark_df.to_csv(BENCHMARK_STATE_PATH, index=False)

        summary = {
            "raw_output": str(RAW_STOCK_PANEL_PATH),
            "model_ready_output": str(MODEL_READY_PANEL_PATH),
            "benchmark_output": str(BENCHMARK_STATE_PATH),
            "feature_stats_output": str(FEATURE_STATS_PATH),
            "start": start,
            "end": end,
            "stocks": stocks,
            "macro": macro,
            "target_horizon": self.target_horizon,
            "dc_threshold": self.dc_threshold,
            "train_split_date": self.train_split_date,
            "panel_rows": int(len(panel_df)),
            "panel_stocks": int(panel_df["stock"].nunique()),
            "date_min": str(panel_df["date"].min().date()) if len(panel_df) else None,
            "date_max": str(panel_df["date"].max().date()) if len(panel_df) else None,
            "continuous_feature_count": int(len(continuous_features)),
            "discrete_feature_count": int(len(discrete_features)),
            "continuous_features": continuous_features,
            "discrete_features": discrete_features,
        }
        with open(DATA_ENG_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[{self.name}] ✅ 原始面板已保存: {RAW_STOCK_PANEL_PATH}")
        print(f"[{self.name}] ✅ 模型参考面板已保存: {MODEL_READY_PANEL_PATH}")
        print(f"[{self.name}] ✅ 基准市场状态已保存: {BENCHMARK_STATE_PATH}")
        print(f"[{self.name}] ✅ 标准化统计已保存: {FEATURE_STATS_PATH}")
        print(f"[{self.name}] ✅ 摘要已保存: {DATA_ENG_SUMMARY_PATH}")
        print("\n🎉 数据工程阶段完成。下一步请运行 prepare_experiment_datasets.py")


def main():
    args = parse_args()

    agent = DataEngineeringAgent(
        dc_threshold=args.dc_threshold,
        target_horizon=args.target_horizon,
        train_split_date=args.train_split_date,
    )
    agent.run(
        start=args.start,
        end=args.end,
        stocks=list(args.stocks),
        macro=args.macro,
    )


if __name__ == "__main__":
    main()
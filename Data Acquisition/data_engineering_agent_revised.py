import datetime
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analyst_fundamental_revised import FundamentalAnalyst
from analyst_news_revised import NewsAnalyst
from analyst_sentiment_revised import SentimentAnalyst
from analyst_technical_revised import TechnicalAnalyst


# -----------------------------
# Optional project_config import
# -----------------------------
DEFAULT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "UNH", "XOM", "PG"]
DEFAULT_MACRO = "SPY"
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_MODEL_DIR = Path("./Model Runs")
DEFAULT_TRAIN_SPLIT_DATE = "2022-01-01"
DEFAULT_TARGET_HORIZON = 5
DEFAULT_DC_THRESHOLD = 0.02

try:
    from project_config import DATA_DIR as CFG_DATA_DIR  # type: ignore
except Exception:
    CFG_DATA_DIR = DEFAULT_DATA_DIR

try:
    from project_config import MODEL_DIR as CFG_MODEL_DIR  # type: ignore
except Exception:
    CFG_MODEL_DIR = DEFAULT_MODEL_DIR

try:
    from project_config import STOCK_UNIVERSE as CFG_STOCKS  # type: ignore
except Exception:
    try:
        from project_config import ESG_STOCKS as CFG_STOCKS  # type: ignore
    except Exception:
        CFG_STOCKS = DEFAULT_STOCKS

try:
    from project_config import MACRO_SYMBOL as CFG_MACRO  # type: ignore
except Exception:
    CFG_MACRO = DEFAULT_MACRO

try:
    from project_config import TRAIN_SPLIT_DATE as CFG_TRAIN_SPLIT_DATE  # type: ignore
except Exception:
    CFG_TRAIN_SPLIT_DATE = DEFAULT_TRAIN_SPLIT_DATE

try:
    from project_config import TARGET_HORIZON_DAYS as CFG_TARGET_HORIZON  # type: ignore
except Exception:
    CFG_TARGET_HORIZON = DEFAULT_TARGET_HORIZON

try:
    from project_config import DC_THRESHOLD as CFG_DC_THRESHOLD  # type: ignore
except Exception:
    CFG_DC_THRESHOLD = DEFAULT_DC_THRESHOLD

try:
    from project_config import feature_stats_path as cfg_feature_stats_path  # type: ignore
except Exception:
    cfg_feature_stats_path = None

DATA_DIR = Path(CFG_DATA_DIR)
MODEL_DIR = Path(CFG_MODEL_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "START": "2010-01-01",
    "END": datetime.datetime.now().strftime("%Y-%m-%d"),
    "STOCKS": list(CFG_STOCKS),
    "MACRO": CFG_MACRO,
    "TRAIN_SPLIT_DATE": str(CFG_TRAIN_SPLIT_DATE),
    "TARGET_HORIZON": int(CFG_TARGET_HORIZON),
    "DC_THRESHOLD": float(CFG_DC_THRESHOLD),
}


def feature_stats_output_path() -> str:
    if cfg_feature_stats_path is not None:
        try:
            return str(cfg_feature_stats_path())
        except Exception:
            pass
    return str(MODEL_DIR / "feature_stats_stock.npz")


class DataEngineeringAgent:
    def __init__(self, dc_threshold: float = CONFIG["DC_THRESHOLD"], target_horizon: int = CONFIG["TARGET_HORIZON"]):
        self.name = "Data Engineering Agent"
        self.dc_threshold = dc_threshold
        self.target_horizon = target_horizon

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
        g["dc_event_density_20"] = pd.Series(np.abs(dc_event)).rolling(20, min_periods=1).sum().values.astype(np.float32)
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

    def run(self) -> None:
        print("==================================================")
        print("  🏢 Analyst Team + Data Engineering Agent 开始处理股票数据")
        print("==================================================\n")

        fund_agent = FundamentalAnalyst(filing_lag_days=45)
        fund_df = fund_agent.run(CONFIG["START"], CONFIG["END"], CONFIG["STOCKS"], CONFIG["MACRO"])
        if fund_df.empty:
            raise RuntimeError("基础数据为空，无法继续。")

        tech_agent = TechnicalAnalyst()
        tech_df = tech_agent.run(fund_df)

        print(f"[{self.name}] 🧠 正在为每只股票提取 DC 特征...")
        tech_df = tech_df.groupby("stock", group_keys=False).apply(lambda g: self.extract_dc_features(g, self.dc_threshold))
        tech_df = tech_df.reset_index(drop=True)

        sent_agent = SentimentAnalyst()
        sent_df = sent_agent.run(CONFIG["START"], CONFIG["END"])

        news_agent = NewsAnalyst()
        news_df = news_agent.run(CONFIG["START"], CONFIG["END"])

        print(f"[{self.name}] ⚙️ 正在合并技术面、基本面、情绪与宏观代理特征...")
        master_df = tech_df.merge(sent_df, on="date", how="left").merge(news_df, on="date", how="left")
        master_df = self._safe_numeric_clean(master_df)
        master_df = master_df.sort_values(["stock", "date"]).reset_index(drop=True)

        numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_fill_zero = {"stock"}
        fill_cols = [c for c in numeric_cols if c not in exclude_fill_zero]
        master_df[fill_cols] = master_df.groupby("stock")[fill_cols].ffill()
        master_df[fill_cols] = master_df[fill_cols].fillna(0.0)

        benchmark = CONFIG["MACRO"]
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
        panel_df = panel_df.groupby("stock", group_keys=False).apply(lambda g: self.add_targets(g, self.target_horizon))
        panel_df = panel_df.reset_index(drop=True)

        continuous_features = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_1d",
            "log_return_1d",
            "return_5d",
            "return_21d",
            "volatility_20d",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_hist_pct",
            "ma10_ratio",
            "ma20_ratio",
            "ma60_ratio",
            "atr_14",
            "atr_pct",
            "hl_spread",
            "volume_z_20",
            "net_margin",
            "operating_margin",
            "revenue_growth_qoq",
            "debt_to_equity",
            "asset_turnover",
            "vix_level",
            "vix_change_5d",
            "vix_z_60",
            "sentiment_score",
            "interest_rate",
            "rate_change_5d",
            "credit_stress",
            "hyg_return_5d",
            "macro_news_pressure",
            "dc_run_length",
            "dc_dist_extreme",
            "dc_tmv",
            "dc_event_density_20",
            "mkt_return_1d",
            "mkt_return_5d",
            "mkt_return_21d",
            "mkt_volatility_20d",
            "mkt_dc_run_length",
            "mkt_dc_tmv",
            "mkt_sentiment_score",
            "mkt_macro_news_pressure",
            "mkt_interest_rate",
            "mkt_credit_stress",
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

        train_split = pd.to_datetime(CONFIG["TRAIN_SPLIT_DATE"])
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        train_mask = panel_df["date"] < train_split

        train_ref = panel_df.loc[train_mask, continuous_features].copy()
        means = train_ref.mean().fillna(0.0)
        stds = train_ref.std().replace(0, 1.0).fillna(1.0)

        model_df = panel_df.copy()
        model_df[continuous_features] = (model_df[continuous_features] - means) / stds
        model_df[continuous_features] = model_df[continuous_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        stats_path = feature_stats_output_path()
        np.savez(
            stats_path,
            mean=means.values.astype(np.float32),
            std=stds.values.astype(np.float32),
            continuous_features=np.array(continuous_features, dtype=object),
            discrete_features=np.array(discrete_features, dtype=object),
        )

        raw_output = DATA_DIR / "stock_panel_raw.csv"
        model_output = DATA_DIR / "stock_panel_model_ready.csv"
        benchmark_output = DATA_DIR / "benchmark_state.csv"

        save_panel_df = panel_df.copy()
        save_model_df = model_df.copy()
        save_benchmark_df = benchmark_df.copy()

        save_panel_df["date"] = save_panel_df["date"].dt.strftime("%Y-%m-%d")
        save_model_df["date"] = save_model_df["date"].dt.strftime("%Y-%m-%d")
        save_benchmark_df["date"] = pd.to_datetime(save_benchmark_df["date"]).dt.strftime("%Y-%m-%d")

        save_panel_df.to_csv(raw_output, index=False)
        save_model_df.to_csv(model_output, index=False)
        save_benchmark_df.to_csv(benchmark_output, index=False)

        print(f"[{self.name}] ✅ 原始面板已保存: {raw_output}")
        print(f"[{self.name}] ✅ 模型输入面板已保存: {model_output}")
        print(f"[{self.name}] ✅ 基准市场状态已保存: {benchmark_output}")
        print(f"[{self.name}] ✅ 标准化统计已保存: {stats_path}")
        print("\n🎉 数据工程阶段完成，可直接给 Transformer / LSTM 使用。")


if __name__ == "__main__":
    DataEngineeringAgent().run()

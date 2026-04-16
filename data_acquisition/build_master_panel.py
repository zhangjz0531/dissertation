from __future__ import annotations
import argparse, datetime, json, sys, warnings
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from project_paths import RAW_STOCK_PANEL_PATH, ensure_all_core_dirs
try:
    from project_paths import LEGACY_MODEL_READY_PANEL_PATH, LEGACY_BENCHMARK_STATE_PATH, LEGACY_FEATURE_STATS_PATH, LEGACY_DATA_ENG_SUMMARY_PATH
except Exception:
    try:
        from project_paths import MODEL_READY_PANEL_PATH as LEGACY_MODEL_READY_PANEL_PATH, BENCHMARK_STATE_PATH as LEGACY_BENCHMARK_STATE_PATH, FEATURE_STATS_PATH as LEGACY_FEATURE_STATS_PATH, DATA_ENG_SUMMARY_PATH as LEGACY_DATA_ENG_SUMMARY_PATH
    except Exception:
        LEGACY_MODEL_READY_PANEL_PATH = LEGACY_BENCHMARK_STATE_PATH = LEGACY_FEATURE_STATS_PATH = LEGACY_DATA_ENG_SUMMARY_PATH = None

from fundamental_feature_builder import FundamentalFeatureBuilder
from technical_feature_builder import TechnicalFeatureBuilder
from vix_sentiment_proxy_builder import VIXSentimentProxyBuilder
from macro_proxy_builder import MacroProxyBuilder

DEFAULT_STOCKS = ["AAPL","AMZN","GOOGL","JNJ","JPM","META","MSFT","NVDA","UNH","V"]
DEFAULT_MACRO = "SPY"
DEFAULT_START = "2010-01-01"
DEFAULT_END = datetime.datetime.now().strftime("%Y-%m-%d")
DEFAULT_TARGET_HORIZON = 5
DEFAULT_DC_THRESHOLD = 0.02

def parse_args():
    p = argparse.ArgumentParser(description="Build canonical raw feature panel for dissertation.")
    p.add_argument("--start", type=str, default=DEFAULT_START)
    p.add_argument("--end", type=str, default=DEFAULT_END)
    p.add_argument("--macro", type=str, default=DEFAULT_MACRO)
    p.add_argument("--target_horizon", type=int, default=DEFAULT_TARGET_HORIZON)
    p.add_argument("--dc_threshold", type=float, default=DEFAULT_DC_THRESHOLD)
    p.add_argument("--write_legacy_artifacts", type=int, default=1)
    p.add_argument("--stocks", nargs="*", default=DEFAULT_STOCKS)
    return p.parse_args()

class FeaturePanelBuilder:
    def __init__(self, dc_threshold: float, target_horizon: int):
        self.name = "Feature Panel Builder"
        self.dc_threshold = float(dc_threshold)
        self.target_horizon = int(target_horizon)

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
        ph = pl = prices[0]
        ph_idx = pl_idx = 0
        dc_trend[0] = 1
        for i in range(1, n):
            p = prices[i]
            if np.isnan(p):
                p = prices[i-1]
            if uptrend:
                if p >= ph:
                    ph, ph_idx = p, i
                elif p <= ph * (1 - theta):
                    uptrend = False
                    pl, pl_idx = p, i
                    dc_event[i] = -1
            else:
                if p <= pl:
                    pl, pl_idx = p, i
                elif p >= pl * (1 + theta):
                    uptrend = True
                    ph, ph_idx = p, i
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
        def _z(group: pd.DataFrame) -> pd.DataFrame:
            out = group.copy()
            for col in cols:
                x = pd.to_numeric(out[col], errors="coerce")
                std = x.std()
                out[f"{col}_cs_z"] = ((x - x.mean()) / (std + 1e-6)).fillna(0.0)
            return out
        return df.groupby("date", group_keys=False).apply(_z)

    @staticmethod
    def _safe_numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = df.copy()
        if "date" in tmp.columns:
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        tmp.to_csv(path, index=False)

    def run(self, start: str, end: str, stocks: List[str], macro: str, write_legacy_artifacts: bool = True) -> None:
        ensure_all_core_dirs()
        print("==================================================")
        print("  Data acquisition / feature panel build started")
        print("==================================================\n")
        base_df = FundamentalFeatureBuilder(filing_lag_days=45).run(start, end, stocks, macro)
        if base_df.empty:
            raise RuntimeError("基础数据为空，无法继续。")
        tech_df = TechnicalFeatureBuilder().run(base_df)
        print(f"[{self.name}] 正在为每只股票提取 DC 特征...")
        tech_df = tech_df.groupby("stock", group_keys=False).apply(lambda g: self.extract_dc_features(g, self.dc_threshold)).reset_index(drop=True)
        sentiment_df = VIXSentimentProxyBuilder().run(start, end)
        macro_df = MacroProxyBuilder().run(start, end)

        print(f"[{self.name}] 正在合并技术面、基本面、情绪与宏观代理特征...")
        master_df = tech_df.merge(sentiment_df, on="date", how="left").merge(macro_df, on="date", how="left")
        master_df = self._safe_numeric_clean(master_df).sort_values(["stock","date"]).reset_index(drop=True)
        numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            master_df[numeric_cols] = master_df.groupby("stock")[numeric_cols].ffill()
            master_df[numeric_cols] = master_df[numeric_cols].fillna(0.0)

        benchmark_cols = ["date","return_1d","return_5d","return_21d","volatility_20d","dc_event","dc_trend","dc_run_length","dc_tmv","sentiment_score","macro_news_pressure","interest_rate","credit_stress"]
        benchmark_df = master_df[master_df["stock"] == macro][benchmark_cols].copy()
        benchmark_df = benchmark_df.rename(columns={c: f"mkt_{c}" for c in benchmark_cols if c != "date"})

        panel_df = master_df[master_df["stock"] != macro].copy().merge(benchmark_df, on="date", how="left")

        cs_cols = ["return_21d","rsi_14","macd_hist_pct","net_margin","operating_margin","revenue_growth_qoq","debt_to_equity"]
        cs_cols = [c for c in cs_cols if c in panel_df.columns]
        panel_df = self.cross_sectional_features(panel_df, cs_cols)

        print(f"[{self.name}] 正在生成预测标签...")
        panel_df = panel_df.groupby("stock", group_keys=False).apply(lambda g: self.add_targets(g, self.target_horizon)).reset_index(drop=True)
        panel_df = panel_df.sort_values(["stock","date"]).reset_index(drop=True)

        print(f"[{self.name}] 保存 canonical raw panel -> {RAW_STOCK_PANEL_PATH}")
        self._save_csv(panel_df, RAW_STOCK_PANEL_PATH)

        if write_legacy_artifacts:
            if LEGACY_BENCHMARK_STATE_PATH is not None:
                self._save_csv(benchmark_df, LEGACY_BENCHMARK_STATE_PATH)
            if LEGACY_MODEL_READY_PANEL_PATH is not None:
                self._save_csv(panel_df, LEGACY_MODEL_READY_PANEL_PATH)
            if LEGACY_FEATURE_STATS_PATH is not None:
                numeric = panel_df.select_dtypes(include=[np.number]).columns.tolist()
                stats_cols = [c for c in numeric if not c.startswith("target_up_") and not c.startswith("future_return_")]
                means = panel_df[stats_cols].mean().fillna(0.0).values.astype(np.float32) if stats_cols else np.array([], dtype=np.float32)
                stds = panel_df[stats_cols].std().replace(0, 1.0).fillna(1.0).values.astype(np.float32) if stats_cols else np.array([], dtype=np.float32)
                LEGACY_FEATURE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez(LEGACY_FEATURE_STATS_PATH, feature_names=np.array(stats_cols, dtype=object), mean=means, std=stds, note=np.array(["Deprecated compatibility artifact. Do not use for train-time scaling."], dtype=object))
            if LEGACY_DATA_ENG_SUMMARY_PATH is not None:
                summary = {
                    "canonical_output": str(RAW_STOCK_PANEL_PATH),
                    "rows": int(len(panel_df)),
                    "stocks": int(panel_df["stock"].nunique()),
                    "date_min": str(panel_df["date"].min().date()) if len(panel_df) else None,
                    "date_max": str(panel_df["date"].max().date()) if len(panel_df) else None,
                    "dc_threshold": float(self.dc_threshold),
                    "target_horizon": int(self.target_horizon),
                    "note": "Canonical output is RAW_STOCK_PANEL_PATH. Legacy artifacts are for backward compatibility only.",
                }
                LEGACY_DATA_ENG_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(LEGACY_DATA_ENG_SUMMARY_PATH, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n✅ canonical raw panel build complete.")
        print(f"Saved canonical panel to: {RAW_STOCK_PANEL_PATH}")

DataEngineeringAgent = FeaturePanelBuilder

def main():
    args = parse_args()
    FeaturePanelBuilder(dc_threshold=args.dc_threshold, target_horizon=args.target_horizon).run(
        start=args.start, end=args.end, stocks=args.stocks, macro=args.macro, write_legacy_artifacts=bool(int(args.write_legacy_artifacts))
    )

if __name__ == "__main__":
    main()

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

class MacroProxyBuilder:
    def __init__(self):
        self.name = "Macro Proxy Builder"

    @staticmethod
    def _download_close_series(ticker: str, start: str, end: str, col_name: str) -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", col_name])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(columns={"Date":"date","Close":col_name,"Adj Close":f"{col_name}_adj"})
        if f"{col_name}_adj" in df.columns:
            df[col_name] = df[f"{col_name}_adj"].fillna(df[col_name])
            df.drop(columns=[f"{col_name}_adj"], inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", col_name]].copy()

    def run(self, start: str, end: str) -> pd.DataFrame:
        print(f"[{self.name}] 正在构造宏观/新闻冲击代理特征 (TNX, HYG)...")
        tnx = self._download_close_series("^TNX", start, end, "interest_rate")
        hyg = self._download_close_series("HYG", start, end, "hyg_close")
        df = pd.merge(tnx, hyg, on="date", how="outer").sort_values("date")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        if df.empty:
            return df
        df["rate_change_5d"] = df["interest_rate"].pct_change(5)
        rolling_hyg_max = df["hyg_close"].rolling(60, min_periods=20).max()
        df["credit_stress"] = (rolling_hyg_max - df["hyg_close"]) / (rolling_hyg_max.abs() + 1e-6)
        df["hyg_return_5d"] = df["hyg_close"].pct_change(5)
        rate_z = (df["rate_change_5d"] - df["rate_change_5d"].rolling(60, min_periods=20).mean()) / (df["rate_change_5d"].rolling(60, min_periods=20).std() + 1e-6)
        stress_z = (df["credit_stress"] - df["credit_stress"].rolling(60, min_periods=20).mean()) / (df["credit_stress"].rolling(60, min_periods=20).std() + 1e-6)
        df["macro_news_pressure"] = np.tanh(0.5 * rate_z.fillna(0.0) + 0.5 * stress_z.fillna(0.0))
        return df[["date","interest_rate","rate_change_5d","credit_stress","hyg_return_5d","macro_news_pressure"]].copy()

NewsAnalyst = MacroProxyBuilder

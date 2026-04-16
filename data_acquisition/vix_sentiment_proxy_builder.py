from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

class VIXSentimentProxyBuilder:
    def __init__(self):
        self.name = "VIX Sentiment Proxy Builder"

    def run(self, start: str, end: str) -> pd.DataFrame:
        print(f"[{self.name}] 正在构造情绪代理特征 (VIX)...")
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
        if vix is None or vix.empty:
            return pd.DataFrame(columns=["date","vix_level","vix_change_5d","vix_z_60","sentiment_score"])
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        vix = vix.reset_index().rename(columns={"Date":"date","Close":"vix_level","Adj Close":"vix_adj"})
        if "vix_adj" in vix.columns:
            vix["vix_level"] = vix["vix_adj"].fillna(vix["vix_level"])
            vix.drop(columns=["vix_adj"], inplace=True)
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        vix["vix_change_5d"] = vix["vix_level"].pct_change(5)
        vix_mean = vix["vix_level"].rolling(60, min_periods=20).mean()
        vix_std = vix["vix_level"].rolling(60, min_periods=20).std() + 1e-6
        vix["vix_z_60"] = (vix["vix_level"] - vix_mean) / vix_std
        vix["sentiment_score"] = -np.tanh(vix["vix_z_60"].fillna(0.0) / 2.0)
        return vix[["date","vix_level","vix_change_5d","vix_z_60","sentiment_score"]].copy()

SentimentAnalyst = VIXSentimentProxyBuilder

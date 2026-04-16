from __future__ import annotations
import warnings
from functools import reduce
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

class FundamentalFeatureBuilder:
    def __init__(self, filing_lag_days: int = 45):
        self.name = "Fundamental Feature Builder"
        self.filing_lag_days = int(filing_lag_days)
        self.fundamental_cols = ["net_margin","operating_margin","revenue_growth_qoq","debt_to_equity","asset_turnover","has_fundamental_data"]

    @staticmethod
    def _standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date","open","high","low","close","volume"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        keep_cols = [c for c in ["date","open","high","low","close","adj_close","volume"] if c in df.columns]
        return df[keep_cols].copy()

    @staticmethod
    def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def _download_price_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        df = self._standardize_price_df(df)
        if df.empty:
            return df
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"].fillna(df["close"])
            df = df.drop(columns=["adj_close"])
        df["stock"] = ticker
        return df

    def _load_quarterly_fundamentals(self, ticker: str) -> pd.DataFrame:
        tk = yf.Ticker(ticker)
        frames = []
        try:
            inc = tk.quarterly_financials.T.copy()
            if not inc.empty:
                inc.index = pd.to_datetime(inc.index).tz_localize(None)
                frames.append(inc.sort_index())
        except Exception:
            pass
        try:
            bal = tk.quarterly_balance_sheet.T.copy()
            if not bal.empty:
                bal.index = pd.to_datetime(bal.index).tz_localize(None)
                frames.append(bal.sort_index())
        except Exception:
            pass
        if not frames:
            return pd.DataFrame(columns=["effective_date"] + self.fundamental_cols)
        fund = reduce(lambda left, right: left.join(right, how="outer"), frames).sort_index()
        fund.index.name = "statement_date"
        fund = fund.reset_index()

        revenue = self._pick_first_existing(fund, ["Total Revenue","Operating Revenue"])
        net_income = self._pick_first_existing(fund, ["Net Income","Net Income Common Stockholders","Net Income Including Noncontrolling Interests"])
        operating_income = self._pick_first_existing(fund, ["Operating Income"])
        total_debt = self._pick_first_existing(fund, ["Total Debt","Long Term Debt","Long Term Debt And Capital Lease Obligation"])
        total_assets = self._pick_first_existing(fund, ["Total Assets"])
        total_equity = self._pick_first_existing(fund, ["Stockholders Equity","Total Equity Gross Minority Interest","Common Stock Equity"])

        out = pd.DataFrame()
        out["statement_date"] = pd.to_datetime(fund["statement_date"]).dt.tz_localize(None)
        out["net_margin"] = net_income / (revenue.abs() + 1e-6)
        out["operating_margin"] = operating_income / (revenue.abs() + 1e-6)
        out["revenue_growth_qoq"] = revenue.pct_change(1)
        out["debt_to_equity"] = total_debt / (total_equity.abs() + 1e-6)
        out["asset_turnover"] = revenue / (total_assets.abs() + 1e-6)
        out["has_fundamental_data"] = 1.0
        out["effective_date"] = out["statement_date"] + pd.Timedelta(days=self.filing_lag_days)
        out = out[["effective_date"] + self.fundamental_cols].sort_values("effective_date")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def _merge_price_and_fundamental(self, price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
        if price_df.empty:
            return price_df
        price_df = price_df.sort_values("date").copy()
        if fund_df.empty:
            for col in self.fundamental_cols:
                price_df[col] = 0.0 if col == "has_fundamental_data" else np.nan
            return price_df
        merged = pd.merge_asof(price_df, fund_df.sort_values("effective_date"), left_on="date", right_on="effective_date", direction="backward")
        if "effective_date" in merged.columns:
            merged.drop(columns=["effective_date"], inplace=True)
        for col in self.fundamental_cols:
            if col not in merged.columns:
                merged[col] = 0.0 if col == "has_fundamental_data" else np.nan
        merged[self.fundamental_cols] = merged[self.fundamental_cols].ffill()
        merged["has_fundamental_data"] = merged["has_fundamental_data"].fillna(0.0)
        return merged

    def run(self, start: str, end: str, stocks: List[str], macro: str) -> pd.DataFrame:
        print(f"[{self.name}] 正在抓取价格并对齐季度财务特征...")
        all_data = []
        for ticker in stocks:
            try:
                price_df = self._download_price_history(ticker, start, end)
                if price_df.empty:
                    print(f"[{self.name}] ⚠️ {ticker} 无价格数据，已跳过。")
                    continue
                fund_df = self._load_quarterly_fundamentals(ticker)
                all_data.append(self._merge_price_and_fundamental(price_df, fund_df))
            except Exception as exc:
                print(f"[{self.name}] ⚠️ {ticker} 处理失败: {exc}")
        try:
            benchmark_df = self._download_price_history(macro, start, end)
            if not benchmark_df.empty:
                for col in self.fundamental_cols:
                    benchmark_df[col] = 0.0 if col == "has_fundamental_data" else np.nan
                all_data.append(benchmark_df)
        except Exception as exc:
            print(f"[{self.name}] ⚠️ 宏观基准 {macro} 下载失败: {exc}")
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True).sort_values(["stock","date"]).reset_index(drop=True)

FundamentalAnalyst = FundamentalFeatureBuilder

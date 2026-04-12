import yfinance as yf
import pandas as pd


class NewsAnalyst:
    def __init__(self):
        self.name = "News Analyst"

    def run(self, start, end):
        print(f"[{self.name}] 📰 正在监测全球宏观经济指标 (TNX, HYG)...")
        tnx = yf.download("^TNX", start=start, end=end)
        if isinstance(tnx.columns, pd.MultiIndex): tnx.columns = [col[0] for col in tnx.columns]
        tnx = tnx.reset_index().rename(columns={'Date': 'date', 'Close': 'interest_rate'})
        tnx['date'] = pd.to_datetime(tnx['date']).dt.tz_localize(None)

        hyg = yf.download("HYG", start=start, end=end)
        if isinstance(hyg.columns, pd.MultiIndex): hyg.columns = [col[0] for col in hyg.columns]
        hyg = hyg.reset_index().rename(columns={'Date': 'date', 'Close': 'hyg_price'})
        hyg['date'] = pd.to_datetime(hyg['date']).dt.tz_localize(None)
        hyg['credit_stress'] = (hyg['hyg_price'].rolling(60).max() - hyg['hyg_price']) / hyg['hyg_price'].rolling(
            60).max()

        df_news = pd.merge(tnx[['date', 'interest_rate']], hyg[['date', 'credit_stress']], on='date', how='outer')
        df_news.sort_values('date', inplace=True)
        df_news.ffill(inplace=True)
        df_news.bfill(inplace=True)
        df_news['date'] = df_news['date'].dt.strftime('%Y-%m-%d')
        return df_news
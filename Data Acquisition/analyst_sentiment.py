import yfinance as yf
import pandas as pd


class SentimentAnalyst:
    def __init__(self):
        self.name = "Sentiment Analyst"

    def run(self, start, end):
        print(f"[{self.name}] 🎭 正在分析社交与公众恐慌情绪 (VIX)...")
        vix = yf.download("^VIX", start=start, end=end)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = [col[0] for col in vix.columns]
        vix = vix.reset_index()
        vix.rename(columns={'Date': 'date', 'Close': 'vix_value'}, inplace=True)
        vix['date'] = pd.to_datetime(vix['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        vix['vix_value'] = vix['vix_value'].clip(10, 40)
        vix['sentiment_score'] = 1.0 - 2.0 * ((vix['vix_value'] - 10) / 30.0)
        return vix[['date', 'sentiment_score']].copy()
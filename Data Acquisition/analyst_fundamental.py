import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class FundamentalAnalyst:
    def __init__(self):
        self.name = "Fundamental Analyst"

    def run(self, start, end, stocks, macro):
        print(f"[{self.name}] 👨‍💼 正在评估公司财务和基本面指标...")
        all_data = []

        # 1. 微观股票池
        for ticker in stocks:
            df = yf.download(ticker, start=start, end=end)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df = df.reset_index()
            df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                               'Volume': 'volume'}, inplace=True)
            df['stock'] = ticker
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            try:
                q_fin = yf.Ticker(ticker).quarterly_financials.T
                if "Net Income" in q_fin.columns and "Total Revenue" in q_fin.columns:
                    q_fin['net_margin'] = q_fin['Net Income'] / (q_fin['Total Revenue'] + 1e-6)
                    fund_df = q_fin[['net_margin']].copy()
                    fund_df.index = pd.to_datetime(fund_df.index).tz_localize(None)
                    fund_df = fund_df.reset_index().rename(columns={'index': 'date'})
                    df = pd.merge(df, fund_df, on='date', how='left')
                    df['net_margin'] = df['net_margin'].ffill().fillna(0.0)
                else:
                    df['net_margin'] = 0.10
            except:
                df['net_margin'] = 0.10
            all_data.append(df)

        # 2. 宏观大盘
        spy = yf.download(macro, start=start, end=end)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = [col[0] for col in spy.columns]
        spy = spy.reset_index()
        spy.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                            'Volume': 'volume'}, inplace=True)
        spy['stock'] = 'SPY'
        spy['date'] = pd.to_datetime(spy['date']).dt.tz_localize(None)
        spy['net_margin'] = 0.0
        all_data.append(spy)

        final_df = pd.concat(all_data, ignore_index=True)
        final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
        return final_df
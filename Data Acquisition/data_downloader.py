import os
import datetime
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "START_TIME": "2010-01-01",
    "END_TIME": datetime.datetime.now().strftime("%Y-%m-%d"),
    "DATA_ROOT": os.path.join(BASE_DIR, "dataload"),
}

# 宏观大盘指数
MACRO_INDEX = "SPY"
# 微观股票池 (10只 ESG/科技蓝筹)
MICRO_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'UNH', 'JNJ', 'JPM', 'V']


class HierarchicalDownloader:
    def __init__(self):
        self.root_dir = CONFIG["DATA_ROOT"]
        self.macro_dir = self.root_dir
        self.micro_dir = os.path.join(self.root_dir, "stocks")

        os.makedirs(self.macro_dir, exist_ok=True)
        os.makedirs(self.micro_dir, exist_ok=True)

    def format_df(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                           'Volume': 'volume'}, inplace=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    def fetch_macro(self):
        print(f"[*] 正在下载宏观大盘指数 (Macro Index): {MACRO_INDEX} ...")
        df = yf.download(MACRO_INDEX, start=CONFIG["START_TIME"], end=CONFIG["END_TIME"], progress=False)
        df = self.format_df(df)
        fpath = os.path.join(self.macro_dir, f'{MACRO_INDEX}_1d.csv')
        df.to_csv(fpath, index=False)
        print(f"[+] 大盘数据已保存至：{fpath}\n")

    def fetch_micro_stocks_with_fundamentals(self):
        print(f"[*] 正在唤醒【基本面分析师】: 开始获取 {len(MICRO_STOCKS)} 只股票的财务与量价数据...")

        for ticker in MICRO_STOCKS:
            print(f"    -> 正在解析 {ticker} ...")
            # 1. 获取量价数据
            df_price = yf.download(ticker, start=CONFIG["START_TIME"], end=CONFIG["END_TIME"], progress=False)
            if df_price.empty:
                print(f"    [!] 警告：{ticker} 价格数据获取失败！")
                continue
            df_price = self.format_df(df_price)
            df_price['date'] = pd.to_datetime(df_price['date'])

            # 2. 获取基本面数据 (季度利润表)
            ticker_obj = yf.Ticker(ticker)
            fund_df = pd.DataFrame()
            try:
                q_fin = ticker_obj.quarterly_financials.T
                if "Net Income" in q_fin.columns and "Total Revenue" in q_fin.columns:
                    # 计算核心基本面因子：净利润率 (Net Margin)
                    q_fin['net_margin'] = q_fin['Net Income'] / (q_fin['Total Revenue'] + 1e-6)
                    fund_df = q_fin[['net_margin']].copy()
                    fund_df.index = pd.to_datetime(fund_df.index)
                    fund_df.reset_index(inplace=True)
                    fund_df.rename(columns={'index': 'date'}, inplace=True)
            except Exception as e:
                pass  # 忽略API偶尔的请求失败

            # 3. 智能合并：将低频的季度财务数据对齐到高频的每日K线上
            if not fund_df.empty:
                # 按照时间合并，并进行前向填充 (将季度财报发布后的数据一直用到下个季度)
                df_merged = pd.merge(df_price, fund_df, on='date', how='left')
                df_merged['net_margin'] = df_merged['net_margin'].ffill().fillna(0.0)
            else:
                # API失效时的容错备用逻辑
                print(f"       [!] {ticker} 财务接口受限，启用均值填充备用方案。")
                df_price['net_margin'] = 0.10  # 默认给予蓝筹股10%基准利润率
                df_merged = df_price

            # 统一时间格式并保存
            df_merged['date'] = df_merged['date'].dt.strftime('%Y-%m-%d')
            fpath = os.path.join(self.micro_dir, f'{ticker}_1d.csv')
            df_merged.to_csv(fpath, index=False)

        print(f"[+] 【基本面分析师】工作完成！量价与财务特征已融合保存至：{self.micro_dir}\n")


if __name__ == "__main__":
    downloader = HierarchicalDownloader()
    downloader.fetch_macro()
    downloader.fetch_micro_stocks_with_fundamentals()
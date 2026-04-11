import os
import datetime
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "START_TIME": "2010-01-01",
    "END_TIME": datetime.datetime.now().strftime("%Y-%m-%d"),
    "DATA_ROOT": os.path.join(BASE_DIR, "dataload"),
}

# 🧠 宏观大盘指数 (供 TD3 择时使用)
MACRO_INDEX = "SPY"

# 🧠 微观股票池 (供 MUSA 选股使用) - 这里选取标普500中权重最大的10只蓝筹/科技股
MICRO_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'UNH', 'JNJ', 'JPM', 'V']


class HierarchicalDownloader:
    def __init__(self):
        self.root_dir = CONFIG["DATA_ROOT"]
        self.macro_dir = self.root_dir  # 大盘数据直接放 dataload 根目录
        self.micro_dir = os.path.join(self.root_dir, "stocks")  # 股票池放 dataload/stocks 子目录

        os.makedirs(self.macro_dir, exist_ok=True)
        os.makedirs(self.micro_dir, exist_ok=True)

    def format_df(self, df):
        """统一的数据格式化清洗函数"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.reset_index()
        df.rename(columns={
            'Date': 'start_time', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }, inplace=True)
        df['start_time'] = df['start_time'].dt.strftime('%Y-%m-%d 00:00:00+00:00')
        return df[['start_time', 'open', 'high', 'low', 'close', 'volume']]

    def fetch_macro_index(self):
        print(f"[*] 正在下载宏观大盘指数 (Macro Index): {MACRO_INDEX} ...")
        df = yf.download(MACRO_INDEX, start=CONFIG["START_TIME"], end=CONFIG["END_TIME"])
        df = self.format_df(df)

        fpath = os.path.join(self.macro_dir, f'{MACRO_INDEX}_1d.csv')
        df.to_csv(fpath, index=False)
        print(f"[+] 大盘数据已保存至：{fpath}\n")

    def fetch_micro_stocks(self):
        print(f"[*] 正在下载微观股票池 (Micro Stocks): 共 {len(MICRO_STOCKS)} 只股票 ...")
        for ticker in MICRO_STOCKS:
            print(f"    -> 下载 {ticker} ...")
            df = yf.download(ticker, start=CONFIG["START_TIME"], end=CONFIG["END_TIME"], progress=False)

            if df.empty:
                print(f"    [!] 警告：{ticker} 数据下载失败或为空！")
                continue

            df = self.format_df(df)
            fpath = os.path.join(self.micro_dir, f'{ticker}_1d.csv')
            df.to_csv(fpath, index=False)

        print(f"[+] 所有微观股票数据已保存至：{self.micro_dir}\n")


if __name__ == "__main__":
    print("==================================================")
    print("    宏观-微观分层架构 (Hierarchical) 数据下载器")
    print("==================================================\n")

    agent = HierarchicalDownloader()

    # 1. 下载宏观大盘数据
    agent.fetch_macro_index()

    # 2. 下载微观股票池数据
    agent.fetch_micro_stocks()

    print("==================================================")
    print("✅ 双核数据下载完成！准备进入特征提取环节。")
    print("==================================================")
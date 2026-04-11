import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_vix_sentiment():
    print("[+] 正在从 Yahoo Finance 获取传统市场恐慌指数 (VIX)...")

    # 获取与主流数据集匹配的时间段
    vix = yf.download("^VIX", start="2018-01-01", end="2026-04-11")

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [col[0] for col in vix.columns]

    vix = vix.reset_index()
    vix.rename(columns={'Date': 'date', 'Close': 'vix_value'}, inplace=True)
    vix['date'] = vix['date'].dt.strftime('%Y-%m-%d')

    vix['vix_value'] = vix['vix_value'].clip(10, 40)
    vix['sentiment_score'] = 1.0 - 2.0 * ((vix['vix_value'] - 10) / 30.0)

    vix.rename(columns={'vix_value': 'fng_value'}, inplace=True)
    df = vix[['date', 'fng_value', 'sentiment_score']].copy()

    # 动态绝对路径寻址：获取项目根目录并创建 dataload 文件夹
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataload_dir = os.path.join(BASE_DIR, "dataload")
    os.makedirs(dataload_dir, exist_ok=True)

    # 保存到 dataload 文件夹
    save_path = os.path.join(dataload_dir, "FNG_Sentiment_1D.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ 宏观情绪数据：成功获取 {len(df)} 天的 VIX 数据，已保存至 {save_path}")

if __name__ == "__main__":
    fetch_vix_sentiment()
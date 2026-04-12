import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime


def fetch_news_macro_agent():
    print("[+] 正在唤醒【新闻与宏观分析师】智能体...")
    start_time = "2010-01-01"
    end_time = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. 宏观新闻信号：10年期美债收益率 (代表加息/降息大环境)
    print("    -> 正在获取全球宏观利率与通胀信号 (^TNX)...")
    tnx = yf.download("^TNX", start=start_time, end=end_time)
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = [col[0] for col in tnx.columns]
    tnx = tnx.reset_index()
    tnx.rename(columns={'Date': 'date', 'Close': 'interest_rate'}, inplace=True)
    tnx['date'] = tnx['date'].dt.strftime('%Y-%m-%d')

    # 2. 市场恐慌/信用违约新闻信号：高收益债ETF (HYG)
    print("    -> 正在获取市场信用违约与衰退恐慌信号 (HYG)...")
    hyg = yf.download("HYG", start=start_time, end=end_time)
    if isinstance(hyg.columns, pd.MultiIndex):
        hyg.columns = [col[0] for col in hyg.columns]
    hyg = hyg.reset_index()
    hyg.rename(columns={'Date': 'date', 'Close': 'hyg_price'}, inplace=True)
    hyg['date'] = hyg['date'].dt.strftime('%Y-%m-%d')

    # 【核心逻辑】：将 HYG 价格转化为恐慌指数 (Credit Stress)
    # 计算公式：(过去60天最高价 - 今日价格) / 过去60天最高价
    hyg['credit_stress'] = (hyg['hyg_price'].rolling(60).max() - hyg['hyg_price']) / hyg['hyg_price'].rolling(60).max()
    hyg['credit_stress'] = hyg['credit_stress'].fillna(0)

    # 合并两大数据
    df_macro = pd.merge(tnx[['date', 'interest_rate']], hyg[['date', 'credit_stress']], on='date', how='outer')
    df_macro.sort_values('date', inplace=True)
    df_macro.ffill(inplace=True)  # 前向填充节假日缺失值
    df_macro.bfill(inplace=True)

    # 保存到 dataload 文件夹
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataload_dir = os.path.join(BASE_DIR, "dataload")
    save_path = os.path.join(dataload_dir, "News_Macro_Signals.csv")
    df_macro.to_csv(save_path, index=False)

    print(f"\n✅ 【新闻与宏观分析师】情报搜集完毕！")
    print(f"✅ 绝密宏观信号已归档至: {save_path}")


if __name__ == "__main__":
    fetch_news_macro_agent()
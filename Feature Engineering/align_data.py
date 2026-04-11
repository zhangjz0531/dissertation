import pandas as pd
import numpy as np
import os


def align_price_and_sentiment_smart():
    print("[+] 开始智能对齐 ESG K线数据与 VIX 情绪数据...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataload_dir = os.path.join(BASE_DIR, "dataload")

    # 🚨 1. 读取正名后的 ESGU_1d 数据
    esg_path = os.path.join(dataload_dir, 'ESGU_1d.csv')
    fng_path = os.path.join(dataload_dir, 'FNG_Sentiment_1D.csv')

    if not os.path.exists(esg_path) or not os.path.exists(fng_path):
        print(f"[!] 找不到原始数据文件，请先运行 Data Acquisition 里的下载脚本！")
        return

    # 🚨 将所有 df_btc 替换为 df_esg
    df_esg = pd.read_csv(esg_path)
    df_fng = pd.read_csv(fng_path)

    # 2. 统一时间格式
    if 'start_time' in df_esg.columns:
        df_esg['date'] = pd.to_datetime(df_esg['start_time']).dt.strftime('%Y-%m-%d')
    else:
        df_esg['date'] = pd.to_datetime(df_esg['Date']).dt.strftime('%Y-%m-%d')

    df_fng['date'] = df_fng['date'].astype(str)

    # 3. 🧠 智能侦测
    missing_dates = df_esg[~df_esg['date'].isin(df_fng['date'])]['date'].tolist()
    if missing_dates:
        print(f"[*] 自动检测到情绪数据缺失了 {len(missing_dates)} 天，准备进行填充。")
    else:
        print("[*] 时间轴完美匹配，无缺失数据。")

    # 4. 以资产为基准进行左连接
    df_merged = pd.merge(df_esg, df_fng[['date', 'fng_value', 'sentiment_score']], on='date', how='left')

    # 5. 执行前向填充
    print("[*] 正在执行前向填充 (Forward Fill)...")
    df_merged['fng_value'] = df_merged['fng_value'].ffill().fillna(50)
    df_merged['sentiment_score'] = df_merged['sentiment_score'].ffill().fillna(0.0)

    # 6. 保存文件 (输出到 dataload)
    cols = ['date', 'start_time', 'open', 'high', 'low', 'close', 'volume', 'fng_value', 'sentiment_score']
    available_cols = [c for c in cols if c in df_merged.columns]
    df_final = df_merged[available_cols]

    # 🚨 正名为 ESGU_1d_with_Sentiment.csv
    output_file = os.path.join(dataload_dir, 'ESGU_1d_with_Sentiment.csv')
    df_final.to_csv(output_file, index=False)
    print(f"✅ 智能对齐完成！总行数: {len(df_final)}，已保存为: {output_file}")


if __name__ == "__main__":
    align_price_and_sentiment_smart()
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 模块 1: 高阶DC特征提取
# =====================================================================
def extract_advanced_dc_features(df, threshold, symbol_name):
    """提取学术标准定向变化(DC)特征"""
    if df.empty or 'close' not in df.columns:
        raise ValueError(f"{symbol_name} 数据为空或缺少close列！")

    df = df.copy()
    df['start_time'] = pd.to_datetime(df['start_time']).dt.tz_localize(None)
    prices = df['close'].values
    n = len(prices)

    dc_trends = np.zeros(n)
    dc_events = np.zeros(n)
    dc_extremes = np.zeros(n)
    dc_T = np.zeros(n)
    dc_TMV = np.zeros(n)

    current_mode = 1 if prices[1] > prices[0] else 0
    extreme_price, reference_price = prices[1], prices[0]
    reference_idx, extreme_idx = 0, 1
    dc_extremes[0:2] = [prices[0], prices[1]]

    for i in range(2, n):
        p = prices[i]
        if current_mode == 1:  # 上涨趋势中
            if p > extreme_price:
                extreme_price, extreme_idx = p, i
            elif p <= extreme_price * (1 - threshold):  # 触发下行DC事件
                dc_events[i], current_mode = 1, 0
                reference_price, reference_idx = extreme_price, extreme_idx
                extreme_price, extreme_idx = p, i
        else:  # 下跌趋势中
            if p < extreme_price:
                extreme_price, extreme_idx = p, i
            elif p >= extreme_price * (1 + threshold):  # 触发上行DC事件
                dc_events[i], current_mode = 1, 1
                reference_price, reference_idx = extreme_price, extreme_idx
                extreme_price, extreme_idx = p, i

        dc_trends[i], dc_extremes[i] = current_mode, extreme_price
        dc_T[i] = i - reference_idx
        ref_p = reference_price if reference_price != 0 else 1e-8
        dc_TMV[i] = abs(p - ref_p) / ref_p

    prefix = f"{symbol_name}_"
    df[f'{prefix}dc_trend'] = dc_trends
    df[f'{prefix}dc_event'] = dc_events
    df[f'{prefix}dc_drawdown'] = (df['close'] - dc_extremes) / (dc_extremes + 1e-8)
    df[f'{prefix}dc_T'] = dc_T
    df[f'{prefix}dc_TMV'] = dc_TMV
    df[f'{prefix}dc_R'] = np.where(df[f'{prefix}dc_T'] == 0, 0, df[f'{prefix}dc_TMV'] / (df[f'{prefix}dc_T'] + 1e-8))

    # 补充基础量化指标
    df[f'{prefix}volume'] = np.log1p(df['volume']).fillna(0)

    # 核心修复：在这里先为每个资产计算收益率 (用于后续生成标签)
    df[f'{prefix}ret'] = df['close'].pct_change().fillna(0)

    cols_to_keep = ['start_time', 'close'] + [col for col in df.columns if col.startswith(prefix)]
    return df[cols_to_keep].copy()


# =====================================================================
# 模块 2: 多资产对齐与标签生成
# =====================================================================
def align_and_fuse_features(df_dict):
    """时序对齐并生成 Transformer 训练标签"""
    merged_df = None
    for symbol, df in df_dict.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='start_time', how='inner')

    merged_df.sort_values('start_time', inplace=True, ignore_index=True)

    # 【核心修复】生成 Transformer 训练所需的标签列
    # 我们预测的是未来一期的收益，所以不需要 shift，因为训练脚本 prepare_data 会处理索引
    merged_df['cash_ret'] = 0.0  # 现金收益恒定为0

    merged_df.ffill(inplace=True);
    merged_df.fillna(0, inplace=True)
    return merged_df


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("    DC高阶特征处理器 (Transformer 兼容版)")
    print("=" * 60)

    DATA_DIR = "../download_data/crypto_data/"
    btc_path = os.path.join(DATA_DIR, "BTCUSDT_15m.csv")
    eth_path = os.path.join(DATA_DIR, "ETHUSDT_15m.csv")

    if not os.path.exists(btc_path):
        print("[ERROR] 找不到原始数据，请先运行 data_downloader.py！")
        exit()

    print("[+] 提取 BTC/ETH DC 特征...")
    df_btc = pd.read_csv(btc_path)
    df_eth = pd.read_csv(eth_path)

    # 针对不同资产设置不同的 DC 阈值
    df_btc_dc = extract_advanced_dc_features(df_btc, threshold=0.003, symbol_name='BTC')
    df_eth_dc = extract_advanced_dc_features(df_eth, threshold=0.0035, symbol_name='ETH')

    # 保存独立特征供 RL 环境使用
    df_btc_dc.to_csv(os.path.join(DATA_DIR, "BTC_DC_features.csv"), index=False)
    df_eth_dc.to_csv(os.path.join(DATA_DIR, "ETH_DC_features.csv"), index=False)

    print("[+] 执行对齐并生成训练标签...")
    aligned_df = align_and_fuse_features({"BTC": df_btc_dc, "ETH": df_eth_dc})

    # 最终检查列名
    required_labels = ['BTC_ret', 'ETH_ret', 'cash_ret']
    for label in required_labels:
        if label not in aligned_df.columns:
            print(f"[!] 警告：缺失标签列 {label}")

    # 保存融合表供 Transformer 使用
    output_path = "../dc_features/"
    os.makedirs(output_path, exist_ok=True)
    aligned_df.to_csv(os.path.join(output_path, "merged_dc_features.csv"), index=False)

    print(f"[+] 处理完成！样本总数：{len(aligned_df)}")
    print(f"[+] 融合表路径：{output_path}merged_dc_features.csv")
    print("=" * 60)
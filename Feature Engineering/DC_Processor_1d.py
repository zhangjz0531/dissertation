import pandas as pd
import numpy as np
import os


def extract_dc_features_1d(input_file, output_file, theta=0.03):
    """
    1D 级别的方向性变化 (Directional Change) 特征提取器
    :param theta: 趋势反转阈值，日线级别推荐 0.03 (3%) 到 0.05 (5%)
    """
    print(f"[+] 开始读取数据: {input_file}")
    df = pd.read_csv(input_file)

    if 'close' not in df.columns:
        print("[!] 错误：数据中没有 'close' 列！")
        return

    prices = df['close'].values
    n = len(prices)

    # 准备存储 DC 特征的数组
    dc_trend = np.zeros(n)  # 趋势状态：1 (Up), -1 (Down)
    dc_event = np.zeros(n)  # 是否发生反转事件：1 (发生), 0 (未发生)
    dc_drawdown = np.zeros(n)  # 回撤幅度 (Overshoot)
    dc_TMV = np.zeros(n)  # 整体价格变动幅度 (Total Price Movement)
    dc_T = np.zeros(n)  # 距离上一个极值点的时间 (Time)
    dc_R = np.zeros(n)  # 当前步收益率 (Return)

    # 初始化 DC 算法变量
    mode = 1  # 初始假设为向上趋势
    extreme_price = prices[0]
    extreme_idx = 0

    print(f"[*] 正在使用阈值 Theta = {theta * 100}% 计算方向性变化...")

    for i in range(1, n):
        p = prices[i]

        # 1. 状态判断与事件触发
        if mode == 1:  # 当前是向上趋势 (Upward)
            if p > extreme_price:
                # 创新高，更新极值点
                extreme_price = p
                extreme_idx = i
            elif p <= extreme_price * (1 - theta):
                # 跌破极值点超过阈值，触发向下反转事件 (Downward Event)
                dc_event[i] = 1
                mode = -1
                extreme_price = p
                extreme_idx = i
        else:  # 当前是向下趋势 (Downward)
            if p < extreme_price:
                # 创新低，更新极值点
                extreme_price = p
                extreme_idx = i
            elif p >= extreme_price * (1 + theta):
                # 反弹突破极值点超过阈值，触发向上反转事件 (Upward Event)
                dc_event[i] = 1
                mode = 1
                extreme_price = p
                extreme_idx = i

        # 2. 实时计算连续特征
        dc_trend[i] = mode
        dc_T[i] = i - extreme_idx
        dc_TMV[i] = (p - extreme_price) / extreme_price
        dc_drawdown[i] = (p - extreme_price) / extreme_price  # 简化的回撤度量
        dc_R[i] = (p - prices[i - 1]) / prices[i - 1] if prices[i - 1] != 0 else 0

    # 3. 将计算好的特征塞回 DataFrame
    df['dc_trend'] = dc_trend
    df['dc_event'] = dc_event
    df['dc_drawdown'] = dc_drawdown
    df['dc_T'] = dc_T
    df['dc_TMV'] = dc_TMV
    df['dc_R'] = dc_R

    # 为了兼容之前的 Transformer 代码，确保时间列名叫 'date'
    if 'start_time' in df.columns and 'date' not in df.columns:
        df.rename(columns={'start_time': 'date'}, inplace=True)

    # 计算预测标签 (明天的收益率) - Transformer 预训练必须要用
    df['target_return'] = df['close'].pct_change().shift(-1).fillna(0)

    # 提取我们真正需要的核心列 (抛弃冗余的开高低量)
    # 确保之前挂载的情绪分 sentiment_score 被保留！
    final_cols = ['date', 'close', 'target_return', 'sentiment_score',
                  'dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R']

    available_cols = [c for c in final_cols if c in df.columns]
    df_final = df[available_cols]

    # 4. 保存文件
    df_final.to_csv(output_file, index=False)
    print(f"✅ 成功生成 1D DC 特征，总数据量: {len(df_final)} 条。")
    print(f"✅ 文件已保存为: {output_file}")

    # 打印前几行预览
    print("\n[🔍 最终多模态特征预览 (前3行)]:")
    print(df_final.head(3).to_string())


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataload_dir = os.path.join(BASE_DIR, "dataload")

    # 🚨 1. 输入文件：读取刚才对齐的 ESG 文件
    input_csv = os.path.join(dataload_dir, "ESGU_1d_with_Sentiment.csv")

    if not os.path.exists(input_csv):
        print(f"[!] 找不到文件！Python 正在这个路径下寻找：{input_csv}")
        exit()

    # 🚨 2. 输出文件夹：抛弃 crypto_data，建立光荣的 esg_data 文件夹！
    output_dir = os.path.join(BASE_DIR, "download_data", "esg_data")
    os.makedirs(output_dir, exist_ok=True)

    # 🚨 3. 终极数据表正名为 ESG_1D_Final.csv
    output_csv = os.path.join(output_dir, "ESG_1D_Final.csv")

    # 执行提取，大盘日线级别阈值可以设小一点，比如 2% (0.02)
    extract_dc_features_1d(input_csv, output_csv, theta=0.02)
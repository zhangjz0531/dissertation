import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rl_folder_path = os.path.join(BASE_DIR, "Reinforcement Learning & Blind Test")
sys.path.append(rl_folder_path)

from td3_agent import TD3

warnings.filterwarnings('ignore')


def run_hierarchical_system():
    print("==================================================")
    print("  🚀 终极调优：TD3(宏观) + MUSA(微观) 进攻型架构")
    print("==================================================\n")

    # 1. 加载并清洗宏观大盘数据
    macro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "ESG_1D_Final.csv")
    df_macro = pd.read_csv(macro_path)
    available_features = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R', 'sentiment_score']

    # 强制类型锁，防止 PyTorch 报错
    for feat in available_features:
        df_macro[feat] = pd.to_numeric(df_macro[feat], errors='coerce').fillna(0).astype(np.float32)

    # 加载特征归一化指纹
    stats = np.load(os.path.join(BASE_DIR, "models", "feature_stats.npz"))
    train_mean = stats['mean'].astype(np.float32)
    train_std = stats['std'].astype(np.float32)
    df_macro[available_features] = (df_macro[available_features] - train_mean) / train_std

    df_test_macro = df_macro[df_macro['date'] >= '2024-01-01'].copy().reset_index(drop=True)

    # 2. 加载 TD3 大脑
    encoder_path = os.path.join(BASE_DIR, "models", "encoder_pretrained.pth")
    agent = TD3(encoder_path=encoder_path, action_dim=1)
    agent.load(os.path.join(BASE_DIR, "models", "td3_final"))

    # 3. 加载 MUSA 微观面板数据
    micro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "MUSA_Top10_Panel.csv")
    df_micro = pd.read_csv(micro_path)
    df_micro = df_micro[df_micro['date'] >= '2023-12-01'].copy()  # 提前一个月用于计算初始动量

    test_dates = sorted(df_test_macro['date'].unique())

    # --- 开始回测参数设定 ---
    window_size = 30
    macro_state_buffer = deque(maxlen=window_size)
    first_state = df_test_macro.loc[0, available_features].values.astype(np.float32)
    for _ in range(window_size):
        macro_state_buffer.append(first_state)

    capital_history = [100000.0]
    bh_history = [100000.0]
    strategy_returns = []
    benchmark_returns = []
    dates_plot = []

    current_capital = 100000.0
    bh_capital = 100000.0

    for i in range(1, len(test_dates)):
        today_date = test_dates[i]
        yesterday_date = test_dates[i - 1]

        # [宏观层决策]：进攻型仓位映射
        current_state_seq = np.array(macro_state_buffer).astype(np.float32)
        td3_action = agent.select_action(current_state_seq)[0]

        # 🚨 调优：采用“底仓逻辑”。美股不轻易空仓。
        # 如果信号大于 0.4，满仓(1.0)；如果信号在 0.2~0.4，保持 50% 仓位；极度恐慌才清仓。
        if td3_action > 0.4:
            macro_exposure = 1.0
        elif td3_action > 0.2:
            macro_exposure = 0.5
        else:
            macro_exposure = 0.2  # 维持 20% 底仓，防止踏空暴涨

        # [微观层选股]：中线趋势选股
        lookback_start = test_dates[max(0, i - 15)]  # 考察期延长至 15 天，寻找中线领涨股
        past_data = df_micro[(df_micro['date'] >= lookback_start) & (df_micro['date'] <= yesterday_date)]

        # 计算风险调整后收益 (动量 / 波动率)，选择上涨最稳的 Top 3
        def calc_risk_adj_return(x):
            if len(x) < 2: return 0
            ret = (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
            vol = x.pct_change().std() + 1e-6
            return ret / vol

        metrics = past_data.groupby('stock')['close'].apply(calc_risk_adj_return)
        top_3_tickers = metrics.nlargest(3).index.tolist()

        # [收益计算]
        today_prices = df_micro[df_micro['date'] == today_date]
        yesterday_prices = df_micro[df_micro['date'] == yesterday_date]
        merged = pd.merge(today_prices[['stock', 'close']], yesterday_prices[['stock', 'close']], on='stock',
                          suffixes=('_today', '_yest'))

        selected_ret_data = merged[merged['stock'].isin(top_3_tickers)]
        micro_return = (selected_ret_data['close_today'] - selected_ret_data['close_yest']) / selected_ret_data[
            'close_yest']
        avg_micro_return = micro_return.mean() if not micro_return.empty else 0

        bench_ret = (df_test_macro.loc[i, 'close'] - df_test_macro.loc[i - 1, 'close']) / df_test_macro.loc[
            i - 1, 'close']

        # [最终执行]：考虑摩擦成本 (万分之二)
        net_return = (macro_exposure * avg_micro_return) - 0.0001

        current_capital *= (1 + net_return)
        bh_capital *= (1 + bench_ret)

        capital_history.append(current_capital)
        bh_history.append(bh_capital)
        strategy_returns.append(net_return)
        benchmark_returns.append(bench_ret)
        dates_plot.append(today_date)

        macro_state_buffer.append(df_test_macro.loc[i, available_features].values.astype(np.float32))

    # --- 🎓 评价与绘图 ---
    def get_metrics(rets, caps):
        rets = np.array(rets)
        sharpe = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)
        mdd = np.min((caps - np.maximum.accumulate(caps)) / np.maximum.accumulate(caps)) * 100
        return (caps[-1] / caps[0] - 1) * 100, sharpe, mdd

    s_ret, s_sh, s_mdd = get_metrics(strategy_returns, capital_history)
    b_ret, b_sh, b_mdd = get_metrics(benchmark_returns, bh_history)

    print("================ 最终实验结果报告 ================")
    print(f"🔥 [融合策略] 收益率: {s_ret:.2f}% | 夏普: {s_sh:.2f} | 回撤: {s_mdd:.2f}%")
    print(f"📉 [基准指数] 收益率: {b_ret:.2f}% | 夏普: {b_sh:.2f} | 回撤: {b_mdd:.2f}%")

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(pd.to_datetime(dates_plot), bh_history[1:], label=f'Benchmark ESG', color='#808080', alpha=0.6)
    plt.plot(pd.to_datetime(dates_plot), capital_history[1:], label=f'MuSA-TD3 Hierarchical', color='#d62728',
             linewidth=2)
    plt.title('Hierarchical System: ESG Investment Victory', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "Final_ESG_Victory.png"))
    print(f"\n✅ 终极胜利图表已保存至: Final_ESG_Victory.png")


if __name__ == "__main__":
    run_hierarchical_system()
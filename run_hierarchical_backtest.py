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
    print("  🏆 终极毕业版：TD3宏观择时 + MUSA中期龙头捕获")
    print("==================================================\n")

    # 1. 加载并清洗宏观大盘数据
    macro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "SPY_1D_Final.csv")
    df_macro = pd.read_csv(macro_path)
    available_features = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R', 'sentiment_score']

    for feat in available_features:
        df_macro[feat] = pd.to_numeric(df_macro[feat], errors='coerce').fillna(0).astype(np.float32)

    stats = np.load(os.path.join(BASE_DIR, "models", "feature_stats.npz"))
    train_mean = stats['mean'].astype(np.float32)
    train_std = stats['std'].astype(np.float32)
    df_macro[available_features] = (df_macro[available_features] - train_mean) / train_std

    df_test_macro = df_macro[df_macro['date'] >= '2024-01-01'].copy().reset_index(drop=True)

    # 2. 加载 TD3 风控大脑
    encoder_path = os.path.join(BASE_DIR, "models", "encoder_pretrained.pth")
    agent = TD3(encoder_path=encoder_path, action_dim=1)
    agent.load(os.path.join(BASE_DIR, "models", "td3_final"))

    # 3. 加载 MUSA 微观面板数据 (提取涵盖2023年的数据用于计算60天历史动量)
    micro_path = os.path.join(BASE_DIR, "download_data", "esg_data", "MUSA_Top10_Panel.csv")
    df_micro = pd.read_csv(micro_path)
    df_micro['date'] = pd.to_datetime(df_micro['date'])

    test_dates = sorted(df_test_macro['date'].unique())

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
    last_top_3 = []

    for i in range(1, len(test_dates)):
        today_date_str = test_dates[i]
        yesterday_date_str = test_dates[i - 1]
        today_dt = pd.to_datetime(today_date_str)
        yesterday_dt = pd.to_datetime(yesterday_date_str)

        # --- [宏观层]: TD3 大盘风控 ---
        current_state_seq = np.array(macro_state_buffer).astype(np.float32)
        td3_action = agent.select_action(current_state_seq)[0]

        # 映射真实仓位：极度恐慌时保持20%底仓，乐观时满仓
        macro_exposure = 0.2 + (td3_action * 0.8)

        # --- [微观层]: MUSA 60天中线龙头捕获 (解决短线追涨杀跌问题) ---
        lookback_dt = today_dt - pd.Timedelta(days=90)  # 获取过去自然日90天（约60个交易日）的数据

        past_data = df_micro[(df_micro['date'] >= lookback_dt) & (df_micro['date'] <= yesterday_dt)]

        def calc_momentum_factor(x):
            if len(x) < 20: return 0.0  # 数据不足则忽略
            ret = (x.iloc[-1] - x.iloc[0]) / x.iloc[0]  # 中期绝对涨幅
            vol = x.pct_change().std() + 1e-6  # 波动率惩罚
            return ret / vol  # 夏普型动量因子

        if not past_data.empty:
            metrics = past_data.groupby('stock')['close'].apply(calc_momentum_factor)
            top_3_tickers = metrics.nlargest(3).index.tolist()
        else:
            top_3_tickers = last_top_3

        # --- [收益评估]: 结算今日真实涨跌 ---
        today_prices = df_micro[df_micro['date'] == today_dt]
        yesterday_prices = df_micro[df_micro['date'] == yesterday_dt]
        merged = pd.merge(today_prices[['stock', 'close']], yesterday_prices[['stock', 'close']], on='stock',
                          suffixes=('_today', '_yest'))

        selected_ret_data = merged[merged['stock'].isin(top_3_tickers)]
        if not selected_ret_data.empty:
            micro_return = (selected_ret_data['close_today'] - selected_ret_data['close_yest']) / selected_ret_data[
                'close_yest']
            avg_micro_return = micro_return.mean()
        else:
            avg_micro_return = 0.0

        bench_ret = (df_test_macro.loc[i, 'close'] - df_test_macro.loc[i - 1, 'close']) / df_test_macro.loc[
            i - 1, 'close']

        # --- [摩擦成本]: 仅对调仓部分收取万分之二的手续费 ---
        turnover_penalty = 0.0
        change_count = len(set(top_3_tickers) - set(last_top_3)) if last_top_3 else 3
        if change_count > 0:
            turnover_penalty = (change_count / 3.0) * 0.0002

        net_return = (macro_exposure * avg_micro_return) - turnover_penalty

        current_capital *= (1 + net_return)
        bh_capital *= (1 + bench_ret)

        capital_history.append(current_capital)
        bh_history.append(bh_capital)
        strategy_returns.append(net_return)
        benchmark_returns.append(bench_ret)
        dates_plot.append(today_date_str)

        macro_state_buffer.append(df_test_macro.loc[i, available_features].values.astype(np.float32))
        last_top_3 = top_3_tickers

    # --- 🎓 多维学术评价 ---
    def get_metrics(rets, caps):
        rets = np.array(rets)
        sharpe = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)
        mdd = np.min((caps - np.maximum.accumulate(caps)) / np.maximum.accumulate(caps)) * 100
        return (caps[-1] / caps[0] - 1) * 100, sharpe, mdd

    s_ret, s_sh, s_mdd = get_metrics(strategy_returns, capital_history)
    b_ret, b_sh, b_mdd = get_metrics(benchmark_returns, bh_history)

    print("================ 核心实验结果 (2024-2026) ================")
    print(f"🔥 [TD3+MUSA 融合架构] 收益率: {s_ret:.2f}% | 夏普: {s_sh:.2f} | 回撤: {s_mdd:.2f}%")
    print(f"📉 [标普500基准 SPY]   收益率: {b_ret:.2f}% | 夏普: {b_sh:.2f} | 回撤: {b_mdd:.2f}%")

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(pd.to_datetime(dates_plot), bh_history[1:], label=f'Benchmark (SPY) | Ret: {b_ret:.1f}%', color='#808080',
             alpha=0.7)
    plt.plot(pd.to_datetime(dates_plot), capital_history[1:], label=f'TD3+MUSA (Mid-Term Alpha) | Ret: {s_ret:.1f}%',
             color='#d62728', linewidth=2.5)
    plt.title('Out-of-Sample Backtest: Hierarchical AI vs S&P 500', fontsize=15, pad=15)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)

    save_path = os.path.join(BASE_DIR, "Dissertation_Final_Chart.png")
    plt.savefig(save_path)
    print(f"\n✅ 完美的学术级对比图表已保存至: {save_path}")


if __name__ == "__main__":
    run_hierarchical_system()
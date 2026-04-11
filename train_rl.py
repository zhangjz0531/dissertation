import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from collections import deque
import warnings
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rl_folder_path = os.path.join(BASE_DIR, "Reinforcement Learning & Blind Test")
sys.path.append(rl_folder_path)

from td3_agent import TD3, ReplayBuffer
from tradeEnv import StockPortfolioEnv

warnings.filterwarnings('ignore')

class MockConfig:
    def __init__(self):
        self.market_name = 'ESG'
        self.mode = 'train'
        self.benchmark_algo = 'equal_weight'
        self.only_long_algo_lst = ['equal_weight']
        self.lambda_1 = 1.0
        self.dailyRetun_lookback = 10
        self.mkt_rf = {'ESG': 0.0}
        self.annual_steps = 252
        self.otherRef_indicator_lst = []
        self.trained_best_model_type = 'max_capital'
        self.reward_scaling = 1.0
        self.cost_penalty = 0.0
        self.senti_k = 1.0
        self.enable_cov_features = False

    def __getattr__(self, item): return [] if 'lst' in item else False

def run_episode(env, agent, replay_buffer, window_size, is_training, episode=0, max_episodes=10):
    obs = env.reset()
    done = False
    steps = 0
    episode_reward = 0
    batch_size = 256

    state_buffer = deque(maxlen=window_size)
    for _ in range(window_size):
        state_buffer.append(obs)
    current_state_seq = np.array(state_buffer)

    noise_std = 0.0
    if is_training:
        initial_noise = 0.2
        if episode < max_episodes - 2:
            noise_std = max(0.01, initial_noise * (1 - episode / (max_episodes * 0.8)))

    start_time = time.time()

    while not done:
        # 🚨 核心修改：状态迷雾注入 (State Noise Injection)
        # 在训练时，给模型看到的特征加上 2% 的高斯噪音，防死记硬背
        if is_training:
            input_state_seq = current_state_seq + np.random.normal(0, 0.02, size=current_state_seq.shape)
        else:
            input_state_seq = current_state_seq

        if is_training and episode < 1:
            action = np.random.uniform(0, 1, size=(1,))
        else:
            action = agent.select_action(input_state_seq)
            if noise_std > 0:
                action = (action + np.random.normal(0, noise_std, size=action.shape)).clip(0, 1)

        next_obs, reward, done, info = env.step(action)

        state_buffer.append(next_obs)
        next_state_seq = np.array(state_buffer)

        if is_training:
            replay_buffer.add(current_state_seq, action, reward, next_state_seq, float(done))
            if replay_buffer.size > batch_size and steps % 100 == 0:
                for _ in range(20):
                    agent.train(replay_buffer, batch_size)

        current_state_seq = next_state_seq
        episode_reward += reward
        steps += 1

    end_time = time.time()
    final_asset = float(env.cur_capital)
    profit_pct = ((final_asset - 100000.0) / 100000.0) * 100

    mode_str = "TRAIN" if is_training else "TEST (BLIND)"
    print(f"✅ [{mode_str}] 结束 | 耗时: {(end_time - start_time):.1f}s | 步数: {steps}")
    print(f"   最终资产: {final_asset:.2f} U | 盈亏率: {profit_pct:+.2f}% | 累计 Reward: {episode_reward:.4f}\n")
    return profit_pct

def plot_and_evaluate(test_env, test_df):
    print("\n================ 正在生成量化评估报告与图表 ================")

    dates = pd.to_datetime(test_df['date'].values)
    strategy_equity = np.array(test_env.asset_memory)
    strategy_returns = np.array(test_env.portfolio_return_memory)

    prices = test_df['close'].values
    initial_price = prices[0]
    benchmark_equity = (prices / initial_price) * 100000.0

    benchmark_returns = np.diff(prices) / prices[:-1]
    benchmark_returns = np.insert(benchmark_returns, 0, 0.0)

    min_len = min(len(dates), len(strategy_equity), len(benchmark_equity))
    dates = dates[:min_len]
    strategy_equity = strategy_equity[:min_len]
    benchmark_equity = benchmark_equity[:min_len]
    strategy_returns = strategy_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    def calc_sharpe(returns):
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        return (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    def calc_mdd(equity_curve):
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return np.min(drawdowns) * 100

    str_sharpe = calc_sharpe(strategy_returns)
    str_mdd = calc_mdd(strategy_equity)

    bm_sharpe = calc_sharpe(benchmark_returns)
    bm_mdd = calc_mdd(benchmark_equity)

    print(f"📊 [TD3 策略] 夏普比率: {str_sharpe:.2f} | 最大回撤: {str_mdd:.2f}%")
    print(f"📊 [基准持有] 夏普比率: {bm_sharpe:.2f} | 最大回撤: {bm_mdd:.2f}%")

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(dates, benchmark_equity, label='Benchmark (Buy & Hold SPY)', color='#808080', alpha=0.8, linewidth=1.5)
    plt.plot(dates, strategy_equity, label='Anti-Overfit TD3 Strategy', color='#1f77b4', linewidth=2.5)

    plt.title('Out-of-Sample Performance (2024-2026): Regularized TD3 vs Benchmark', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (USD)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

    metrics_text = (
        f"--- Anti-Overfit TD3 Strategy ---\n"
        f"Total Return: {((strategy_equity[-1] / 100000) - 1) * 100:.1f}%\n"
        f"Sharpe Ratio: {str_sharpe:.2f}\n"
        f"Max Drawdown: {str_mdd:.2f}%\n\n"
        f"--- Benchmark (SPY) ---\n"
        f"Total Return: {((benchmark_equity[-1] / 100000) - 1) * 100:.1f}%\n"
        f"Sharpe Ratio: {bm_sharpe:.2f}\n"
        f"Max Drawdown: {bm_mdd:.2f}%"
    )

    plt.gca().text(0.02, 0.75, metrics_text, transform=plt.gca().transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "OOS_Performance.png")
    plt.savefig(save_path)
    print(f"\n✅ 高清图表已成功保存至: {save_path}")

def main():
    print("==================================================")
    print("  MuSA + TD3 严谨学术回测 (抗过拟合版)")
    print("==================================================\n")

    data_path = os.path.join(BASE_DIR, "download_data", "esg_data", "SPY_1D_Final.csv")
    df = pd.read_csv(data_path)

    available_features = ['dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R', 'sentiment_score']
    df.fillna(0, inplace=True)

    stats_path = os.path.join(BASE_DIR, "models", "feature_stats.npz")
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        train_mean = stats['mean']
        train_std = stats['std']
        df[available_features] = (df[available_features] - train_mean) / train_std
    else:
        print("[!] ⚠️ 警告：未找到 feature_stats.npz！")

    split_date = '2024-01-01'
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    print(f"[*] 训练集 (In-Sample): 2010 到 2023, 共 {len(train_df)} 天")
    print(f"[*] 测试集 (Out-of-Sample): 2024 到 2026, 共 {len(test_df)} 天\n")

    train_env = StockPortfolioEnv(
        config=MockConfig(), rawdata=train_df, mode='train', stock_num=1,
        action_dim=1, tech_indicator_lst=available_features,
        initial_asset=100000.0, transaction_cost=0.0001, slippage=0.0001
    )

    test_env = StockPortfolioEnv(
        config=MockConfig(), rawdata=test_df, mode='test', stock_num=1,
        action_dim=1, tech_indicator_lst=available_features,
        initial_asset=100000.0, transaction_cost=0.0001, slippage=0.0001
    )

    encoder_path = os.path.join(BASE_DIR, "models", "encoder_pretrained.pth")
    agent = TD3(encoder_path=encoder_path, action_dim=1)

    window_size = 30
    replay_buffer = ReplayBuffer(state_dim=(window_size, len(available_features)), action_dim=1)

    # 🚨 核心修改：将训练轮数减半至 10 轮
    max_episodes = 10
    print("================ 阶段 1：在历史数据中训练 ================")
    for ep in range(max_episodes):
        print(f"[*] 训练 Episode {ep + 1}/{max_episodes}...")
        run_episode(train_env, agent, replay_buffer, window_size, is_training=True, episode=ep,
                    max_episodes=max_episodes)

    print("================ 阶段 2：在未知的未来数据中盲测 ================")
    print("[*] AI 现在面临 2024-2026 年的市场，完全关闭噪音，纯靠实力交易！")

    start_price = test_df['close'].iloc[0]
    end_price = test_df['close'].iloc[-1]
    bh_return = ((end_price - start_price) / start_price) * 100
    print(f"[*] 基准挑战：同期 SPY 持有不动收益率为 {bh_return:+.2f}%\n")

    run_episode(test_env, agent, replay_buffer, window_size, is_training=False)

    plot_and_evaluate(test_env, test_df)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    agent.save(os.path.join(BASE_DIR, "models", "td3_final"))
    print("🎉 实验圆满结束！")

if __name__ == "__main__":
    main()
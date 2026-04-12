import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from collections import deque
import warnings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

from td3_agent import TD3, ReplayBuffer
from tradeEnv import StockPortfolioEnv

warnings.filterwarnings('ignore')

DATA_PATH = r"D:\python\dissertation\Data Acquisition\download_data\esg_data\SPY_Macro_State.csv"
ENCODER_PATH = r"D:\python\dissertation\models\encoder_pretrained.pth"
MODELS_DIR = r"D:\python\dissertation\models"

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

def run_episode(env, agent, replay_buffer, window_size, is_training, episode=0, max_episodes=25):
    obs = env.reset()
    done = False
    steps = 0
    batch_size = 256

    state_buffer = deque(maxlen=window_size)
    for _ in range(window_size): state_buffer.append(obs)
    current_state_seq = np.array(state_buffer)

    # 🚨 优化探索噪音：增加初始探索率，并在前 80% 的回合中平滑衰减
    noise_std = 0.0
    if is_training:
        initial_noise = 0.3 # 更大的初始胆量
        if episode < max_episodes * 0.8:
            noise_std = max(0.01, initial_noise * (1 - episode / (max_episodes * 0.8)))

    start_time = time.time()

    while not done:
        input_state_seq = current_state_seq + np.random.normal(0, 0.02, size=current_state_seq.shape) if is_training else current_state_seq

        if is_training and episode < 3: # 前 3 轮完全随机探索，积累经验
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
                # 增加训练频次，让大脑更聪明
                for _ in range(30): agent.train(replay_buffer, batch_size)

        current_state_seq = next_state_seq
        steps += 1

    final_asset = float(env.cur_capital)
    profit_pct = ((final_asset - 100000.0) / 100000.0) * 100
    years = steps / 252.0
    cagr = ((final_asset / 100000.0) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    mode_str = "TRAIN" if is_training else "TEST (BLIND)"
    print(f"✅ [{mode_str}] 结束 | 耗时: {(time.time() - start_time):.1f}s | 时长: {years:.1f} 年")
    print(f"   最终资产: {final_asset:.2f} U | 累计盈亏: {profit_pct:+.2f}% | 🌟 年化收益(CAGR): {cagr:+.2f}%\n")
    return profit_pct

def main():
    print("==================================================")
    print("  👨‍💼 深度培训【风险管理与投资组合经理】(TD3 Agent)")
    print("==================================================\n")

    if not os.path.exists(DATA_PATH):
        print(f"[!] 找不到数据底座 {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    available_features = [
        'dc_trend', 'dc_event', 'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R',
        'RSI', 'MACD_Pct', 'sentiment_score', 'interest_rate', 'credit_stress'
    ]
    df.fillna(0, inplace=True)

    split_date = '2024-01-01'
    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    train_env = StockPortfolioEnv(config=MockConfig(), rawdata=train_df, mode='train', stock_num=1, action_dim=1, tech_indicator_lst=available_features, initial_asset=100000.0, transaction_cost=0.0001, slippage=0.0001)
    test_env = StockPortfolioEnv(config=MockConfig(), rawdata=test_df, mode='test', stock_num=1, action_dim=1, tech_indicator_lst=available_features, initial_asset=100000.0, transaction_cost=0.0001, slippage=0.0001)

    agent = TD3(encoder_path=ENCODER_PATH, action_dim=1)
    replay_buffer = ReplayBuffer(state_dim=(30, len(available_features)), action_dim=1)

    # 🚨 将训练回合数提升至 25 轮，彻底消除“创伤后遗症”
    max_episodes = 25
    print("================ 阶段 1：风控团队深度历史模拟演练 ================")
    for ep in range(max_episodes):
        print(f"[*] 演练 Episode {ep + 1}/{max_episodes}...")
        run_episode(train_env, agent, replay_buffer, 30, is_training=True, episode=ep, max_episodes=max_episodes)

    print("================ 阶段 2：投资组合经理盲测执行 ================")
    run_episode(test_env, agent, replay_buffer, 30, is_training=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save(os.path.join(MODELS_DIR, "td3_final"))
    print("\n🎉 深度培训完成！极其强健的 TD3 大脑已存入统一模型库。")

if __name__ == "__main__":
    main()
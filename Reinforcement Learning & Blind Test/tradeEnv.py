import numpy as np
import warnings

warnings.filterwarnings("ignore")


class StockPortfolioEnv:
    """
    极速版单资产交易环境 + 死区过滤器 (过滤无效摩擦成本)
    """

    def __init__(self, config, rawdata, mode='train', stock_num=1,
                 action_dim=1, tech_indicator_lst=[], max_shares=1,
                 initial_asset=100000.0, transaction_cost=0.001, slippage=0.0005):

        self.mode = mode
        self.stock_num = stock_num
        self.action_dim = action_dim
        self.initial_capital = initial_asset
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        df_sorted = rawdata.sort_values(by='date').reset_index(drop=True)
        self.prices_array = df_sorted['close'].values
        self.obs_array = df_sorted[tech_indicator_lst].values
        self.max_step = len(self.prices_array) - 1

        self.current_step = 0
        self.cur_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        self.asset_memory = [self.initial_capital]
        self.weights = np.zeros(self.action_dim)
        self.portfolio_return_memory = [0]

    def reset(self):
        self.current_step = 0
        self.cur_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.asset_memory = [self.initial_capital]
        self.weights = np.zeros(self.action_dim)
        self.portfolio_return_memory = [0]
        return self._get_obs()

    def _get_obs(self):
        return self.obs_array[self.current_step]

    def step(self, actions):
        last_price = self.prices_array[self.current_step]

        self.current_step += 1
        done = self.current_step >= self.max_step

        if done:
            return self._get_obs(), 0, done, {}

        current_price = self.prices_array[self.current_step]
        price_returns = (current_price - last_price) / last_price

        # ========================================================
        # 🛡️ 核心学术升级：信心死区过滤器 (Confidence Deadzone)
        # ========================================================
        # ========================================================
        # 🛡️ 核心学术升级：缩小信心死区 (让 AI 反应更敏捷)
        # ========================================================
        raw_action = actions[0]

        # 将阈值从 0.7/0.3 缩小到 0.55/0.45。只要轻微看多就上车，轻微看空就下车。
        if raw_action > 0.55:
            target_weights = np.array([1.0])
        elif raw_action < 0.45:
            target_weights = np.array([0.0])
        else:
            target_weights = self.weights.copy()  # 中间 10% 作为缓冲防抖

        turnover = np.abs(target_weights - self.weights).sum()
        friction_cost_rate = turnover * (self.transaction_cost + self.slippage)

        portfolio_return = np.sum(self.weights * price_returns) - friction_cost_rate
        self.cur_capital = self.cur_capital * (1 + portfolio_return)

        if self.cur_capital > self.peak_capital:
            self.peak_capital = self.cur_capital

        self.asset_memory.append(self.cur_capital)
        self.portfolio_return_memory.append(portfolio_return)
        self.weights = target_weights

        # ========================================================
        # 🧠 纯净版抗风险奖励 (移除持续扣分，保留不对称惩罚)
        # ========================================================
        reward = portfolio_return * 100

        # 只保留这一条：亏损的痛苦是赚钱的 1.5 倍（这已经足够让它规避风险了）
        if reward < 0:
            reward *= 1.5

            # 🚨 删除了那个每天扣 -0.5 的回撤惩罚！让累计 Reward 回归正常！

        if self.cur_capital < self.initial_capital * 0.4:
            reward -= 50  # 稍微降低破产惩罚
            done = True

        return self._get_obs(), reward, done, {}
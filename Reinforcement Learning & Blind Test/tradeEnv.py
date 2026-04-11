import numpy as np
import warnings

warnings.filterwarnings("ignore")


class StockPortfolioEnv:
    """
    连续控制版交易环境 (Continuous Control Edition)
    专为 TD3 等连续动作空间强化学习设计，恢复平滑梯度
    """

    def __init__(self, config, rawdata, mode='train', stock_num=1,
                 action_dim=1, tech_indicator_lst=[], max_shares=1,
                 initial_asset=100000.0, transaction_cost=0.0001, slippage=0.0001):

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
        # ⚔️ 连续动作映射 (彻底解决梯度断裂问题)
        # ========================================================
        raw_action = actions[0]  # TD3 输出的原始连续动作 [0, 1]

        # 考虑到美股长牛属性，我们给仓位加一个下限 (底仓 20%)，防止踏空
        # 映射公式：使得输出 0 时持有 20%，输出 1 时持有 100%
        final_weight = 0.2 + (raw_action * 0.8)
        target_weights = np.array([final_weight])

        # 摩擦成本计算
        turnover = np.abs(target_weights - self.weights).sum()
        friction_cost_rate = turnover * (self.transaction_cost + self.slippage)

        # 更新资产
        portfolio_return = np.sum(self.weights * price_returns) - friction_cost_rate
        self.cur_capital = self.cur_capital * (1 + portfolio_return)

        if self.cur_capital > self.peak_capital:
            self.peak_capital = self.cur_capital

        self.asset_memory.append(self.cur_capital)
        self.portfolio_return_memory.append(portfolio_return)
        self.weights = target_weights

        # ========================================================
        # 🧠 平滑奖励机制 (鼓励绝对收益)
        # ========================================================
        reward = portfolio_return * 100

        if self.cur_capital < self.initial_capital * 0.5:
            reward -= 50
            done = True

        return self._get_obs(), reward, done, {}
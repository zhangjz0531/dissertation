import os
import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from analyst_fundamental import FundamentalAnalyst
from analyst_technical import TechnicalAnalyst
from analyst_sentiment import SentimentAnalyst
from analyst_news import NewsAnalyst

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_HUB = os.path.join(BASE_DIR, "Agent_Data_Hub")
OUTPUT_DIR = os.path.join(BASE_DIR, "download_data", "esg_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_HUB, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CONFIG = {
    "START": "2010-01-01",
    "END": datetime.datetime.now().strftime("%Y-%m-%d"),
    "STOCKS": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'UNH', 'JNJ', 'JPM', 'V'],
    "MACRO": "SPY"
}


class DataEngineeringAgent:
    def __init__(self):
        self.name = "Data Engineering Agent"

    def extract_internal_dc_features(self, df, theta=0.02):
        print(f"    -> 🧠 正在进行内生 DC 物理特征提取 (阈值: {theta * 100}%)...")
        prices = df['close'].values
        n = len(prices)

        dc_trend = np.zeros(n)
        dc_event = np.zeros(n)
        dc_drawdown = np.zeros(n)
        dc_T = np.zeros(n)
        dc_TMV = np.zeros(n)
        dc_R = np.zeros(n)

        up_trend = True
        ph = prices[0]
        pl = prices[0]
        ph_idx = 0
        pl_idx = 0

        for i in range(1, n):
            p = prices[i]
            if up_trend:
                if p > ph:
                    ph = p
                    ph_idx = i
                elif p <= ph * (1 - theta):
                    up_trend = False
                    pl = p
                    pl_idx = i
                    dc_event[i] = -1  # 完美保留 -1 向下突变
            else:
                if p < pl:
                    pl = p
                    pl_idx = i
                elif p >= pl * (1 + theta):
                    up_trend = True
                    ph = p
                    ph_idx = i
                    dc_event[i] = 1  # 完美保留 1 向上突变

            dc_trend[i] = 1 if up_trend else -1
            if up_trend:
                dc_drawdown[i] = (p - ph) / ph
                dc_T[i] = i - ph_idx
                dc_TMV[i] = (p - pl) / pl if pl != 0 else 0
                dc_R[i] = (p - ph) / ph
            else:
                dc_drawdown[i] = (p - pl) / pl
                dc_T[i] = i - pl_idx
                dc_TMV[i] = (ph - p) / ph if ph != 0 else 0
                dc_R[i] = (p - pl) / pl

        df['dc_trend'] = dc_trend.astype(np.float32)
        df['dc_event'] = dc_event.astype(np.float32)
        df['dc_drawdown'] = dc_drawdown.astype(np.float32)
        df['dc_T'] = dc_T.astype(np.float32)
        df['dc_TMV'] = dc_TMV.astype(np.float32)
        df['dc_R'] = dc_R.astype(np.float32)
        return df

    def run(self):
        print("==================================================")
        print("  🏢 Multi-Agent System: 首席数据官开始调度分析师团队")
        print("==================================================\n")

        fund_agent = FundamentalAnalyst()
        fund_df = fund_agent.run(CONFIG["START"], CONFIG["END"], CONFIG["STOCKS"], CONFIG["MACRO"])

        tech_agent = TechnicalAnalyst()
        tech_df = tech_agent.run(fund_df)

        sent_agent = SentimentAnalyst()
        sent_df = sent_agent.run(CONFIG["START"], CONFIG["END"])

        news_agent = NewsAnalyst()
        news_df = news_agent.run(CONFIG["START"], CONFIG["END"])

        print(f"\n[{self.name}] ⚙️ 所有情报收集完毕，开始在内存中融合与清洗数据...")

        master_df = pd.merge(tech_df, sent_df, on='date', how='left')
        master_df = pd.merge(master_df, news_df, on='date', how='left')
        master_df.ffill(inplace=True)
        master_df.fillna(0, inplace=True)
        master_df['MACD_Pct'] = master_df['MACD'] / (master_df['close'] + 1e-6)

        # ==========================================
        # 🟢 1. 微观面板数据处理 (MUSA 选股)
        # ==========================================
        print(f"[{self.name}] 📊 正在生成微观股票池 (Cross-Sectional Z-Score)...")
        micro_df = master_df[master_df['stock'] != 'SPY'].copy()

        def cross_sectional_zscore(group):
            cols_to_norm = ['RSI', 'MACD_Pct', 'net_margin']
            for col in cols_to_norm:
                std = group[col].std()
                group[f'{col}_Z'] = (group[col] - group[col].mean()) / std if std > 1e-6 else 0.0
            return group

        micro_df = micro_df.groupby('date', group_keys=False).apply(cross_sectional_zscore)
        micro_df.sort_values(by=['stock', 'date'], inplace=True)
        micro_df.to_csv(os.path.join(OUTPUT_DIR, "MUSA_Top10_Panel.csv"), index=False)
        print(f"    -> ✅ 微观面板已保存！")

        # ==========================================
        # 🔵 2. 宏观状态处理 (特征隔离清洗)
        # ==========================================
        print(f"[{self.name}] 🧠 正在提取方向性变化特征并生成宏观状态...")
        macro_df = master_df[master_df['stock'] == 'SPY'].copy()
        macro_df.sort_values('date', inplace=True)

        macro_df_with_dc = self.extract_internal_dc_features(macro_df, theta=0.02)

        # 🚨 核心修改：明确拆分“离散特征”与“连续特征”
        discrete_features = ['dc_trend', 'dc_event']  # 绝对不碰归一化！
        continuous_features = [
            'dc_drawdown', 'dc_T', 'dc_TMV', 'dc_R',
            'RSI', 'MACD_Pct', 'sentiment_score', 'interest_rate', 'credit_stress'
        ]
        macro_features = discrete_features + continuous_features

        for feat in macro_features:
            macro_df_with_dc[feat] = pd.to_numeric(macro_df_with_dc[feat], errors='coerce').fillna(0).astype(np.float32)

        # 🚨 仅对 9 个连续特征计算均值和标准差
        train_macro = macro_df_with_dc[macro_df_with_dc['date'] < '2024-01-01']
        feature_means = train_macro[continuous_features].mean().values
        feature_stds = train_macro[continuous_features].std().values + 1e-6

        np.savez(os.path.join(MODELS_DIR, "feature_stats.npz"), mean=feature_means, std=feature_stds,
                 features=continuous_features)

        # 🚨 仅对连续特征应用 Z-Score，保留 dc_trend 和 dc_event 的纯净 [-1, 0, 1] 状态
        macro_df_with_dc[continuous_features] = (macro_df_with_dc[continuous_features] - feature_means) / feature_stds

        macro_df_with_dc.to_csv(os.path.join(OUTPUT_DIR, "SPY_Macro_State.csv"), index=False)
        print(f"    -> ✅ 宏观状态矩阵已保存！完美隔离并保留了稀疏事件触发器！")


if __name__ == "__main__":
    DataEngineeringAgent().run()
    print("\n🎉 数据工程全部杀青！")
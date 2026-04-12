import pandas as pd


class TechnicalAnalyst:
    def __init__(self):
        self.name = "Technical Analyst"

    def run(self, df_base):
        print(f"[{self.name}] 📈 正在计算 MACD, RSI 识别价格走势...")
        df = df_base.copy()

        def calc_tech_indicators(group):
            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            group['RSI'] = 100 - (100 / (1 + rs))

            exp1 = group['close'].ewm(span=12, adjust=False).mean()
            exp2 = group['close'].ewm(span=26, adjust=False).mean()
            group['MACD'] = exp1 - exp2
            group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
            return group

        df = df.groupby('stock', group_keys=False).apply(calc_tech_indicators)
        df.fillna(0, inplace=True)
        return df
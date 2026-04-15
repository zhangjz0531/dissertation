import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class TechnicalAnalyst:
    def __init__(self):
        self.name = "Technical Analyst"

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window, min_periods=window).mean()

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _calc_tech_indicators(self, group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce")

        g["return_1d"] = close.pct_change(1)
        g["log_return_1d"] = np.log(close.replace(0, np.nan)).diff()
        g["return_5d"] = close.pct_change(5)
        g["return_21d"] = close.pct_change(21)
        g["volatility_20d"] = g["return_1d"].rolling(20, min_periods=20).std()

        g["rsi_14"] = self._rsi(close, 14)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]
        g["macd_hist_pct"] = g["macd_hist"] / (close.abs() + 1e-6)

        sma10 = close.rolling(10, min_periods=10).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        sma60 = close.rolling(60, min_periods=60).mean()
        g["ma10_ratio"] = close / (sma10 + 1e-6) - 1.0
        g["ma20_ratio"] = close / (sma20 + 1e-6) - 1.0
        g["ma60_ratio"] = close / (sma60 + 1e-6) - 1.0

        g["atr_14"] = self._atr(high, low, close, 14)
        g["atr_pct"] = g["atr_14"] / (close.abs() + 1e-6)
        g["hl_spread"] = (high - low) / (close.abs() + 1e-6)

        vol_mean20 = volume.rolling(20, min_periods=20).mean()
        vol_std20 = volume.rolling(20, min_periods=20).std() + 1e-6
        g["volume_z_20"] = (volume - vol_mean20) / vol_std20

        fill_map = {
            "return_1d": 0.0,
            "log_return_1d": 0.0,
            "return_5d": 0.0,
            "return_21d": 0.0,
            "volatility_20d": 0.0,
            "rsi_14": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_hist": 0.0,
            "macd_hist_pct": 0.0,
            "ma10_ratio": 0.0,
            "ma20_ratio": 0.0,
            "ma60_ratio": 0.0,
            "atr_14": 0.0,
            "atr_pct": 0.0,
            "hl_spread": 0.0,
            "volume_z_20": 0.0,
        }
        for col, value in fill_map.items():
            g[col] = g[col].replace([np.inf, -np.inf], np.nan).fillna(value)
        return g

    def run(self, df_base: pd.DataFrame) -> pd.DataFrame:
        print(f"[{self.name}] 📈 正在计算技术指标...")
        if df_base is None or df_base.empty:
            return pd.DataFrame()
        df = df_base.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("stock", group_keys=False).apply(self._calc_tech_indicators)
        return df.reset_index(drop=True)

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

# ======================================================================
#  项目路径
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data_layer" / "outputs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PANEL_DIR = DATA_DIR / "panel"
DIAGNOSTIC_DIR = DATA_DIR / "diagnostics"
CACHE_DIR = DATA_DIR / "cache"

for d in (RAW_DIR, PROCESSED_DIR, PANEL_DIR, DIAGNOSTIC_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ======================================================================
#  主实验范围：股票日线
# ======================================================================
STOCK_UNIVERSE: List[str] = [
    "AAPL",
    "NVDA",
    "META",
    "JPM",
    "JNJ",
    "XOM",
]
STOCK_BENCHMARK: str = "SPY"

# ======================================================================
#  附录 / 兼容常量（主线默认不使用，但保留以免旧文件 import 失效）
# ======================================================================
STOCK_HOURLY_UNIVERSE: List[str] = ["AAPL", "NVDA", "JPM"]
CRYPTO_UNIVERSE: List[str] = ["BTC/USDT", "ETH/USDT"]
CRYPTO_EXCHANGE: str = "binance"

# ======================================================================
#  日期范围
# ======================================================================
_TODAY = datetime.datetime.now().strftime("%Y-%m-%d")

DAILY_START: str = "2015-01-01"
DAILY_END: str = _TODAY

HOURLY_START: str = (datetime.datetime.now() - datetime.timedelta(days=720)).strftime("%Y-%m-%d")
HOURLY_END: str = _TODAY
CRYPTO_HOURLY_START: str = "2021-01-01"
CRYPTO_HOURLY_END: str = _TODAY

# ======================================================================
#  DC 参数
# ======================================================================
DC_THETAS_DAILY: List[float] = [0.01, 0.02, 0.05]
DC_THETAS_HOURLY_STOCK: List[float] = [0.003, 0.005, 0.01]
DC_THETAS_HOURLY_CRYPTO: List[float] = [0.005, 0.01, 0.02]

# ======================================================================
#  技术指标窗口
# ======================================================================
RETURN_LAGS: List[int] = [1, 5, 20, 60]
VOL_WINDOWS: List[int] = [20, 60]
RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
ATR_WINDOW: int = 14
MA_WINDOWS: List[int] = [10, 20, 60]
VOLUME_Z_WINDOW: int = 20

BURN_IN_DAYS: int = max(max(MA_WINDOWS), max(VOL_WINDOWS), MACD_SLOW + MACD_SIGNAL)

# ======================================================================
#  标签配置
# ======================================================================
TARGET_HORIZONS: List[int] = [1, 5]
LABEL_MAIN_HORIZON: int = 5
DEFAULT_LABEL_COL: str = f"target_dir_{LABEL_MAIN_HORIZON}"

# 输出三套标签：absolute / excess / band
LABEL_FAMILIES: List[str] = ["dir", "excess_dir", "band_dir"]
BAND_BPS_H1: float = 10.0
BAND_BPS_H5: float = 25.0
DROP_BENCHMARK_FROM_MAIN_PANEL: bool = True

# ======================================================================
#  切分规则
# ======================================================================
DAILY_TRAIN_END: str = "2022-12-31"
DAILY_VAL_END: str = "2023-12-31"

WALK_FORWARD_ENABLED: bool = False
WALK_FORWARD_TRAIN_MONTHS: int = 24
WALK_FORWARD_VAL_MONTHS: int = 3
WALK_FORWARD_TEST_MONTHS: int = 3

# ======================================================================
#  宏观代理
# ======================================================================
VIX_TICKER: str = "^VIX"
UST10Y_TICKER: str = "^TNX"
HYG_TICKER: str = "HYG"
DXY_TICKER: str = "DX-Y.NYB"

# ======================================================================
#  已弃用但保留的兼容常量
# ======================================================================
FUNDAMENTAL_LAG_DAYS: int = 45
FUNDAMENTAL_COLS: List[str] = [
    "net_margin", "op_margin", "rev_growth_qoq",
    "d2e", "asset_turnover", "has_fundamental",
]
ESG_STATIC_COLS: List[str] = [
    "esg_total", "esg_env", "esg_soc", "esg_gov", "esg_controversy_level",
]
ESG_GROUP_METHOD: str = "median"

# ======================================================================
#  输出路径
# ======================================================================
RAW_OHLCV_DAILY_PATH = RAW_DIR / "ohlcv_daily_stock.parquet"
RAW_OHLCV_HOURLY_STOCK_PATH = RAW_DIR / "ohlcv_hourly_stock.parquet"
RAW_OHLCV_HOURLY_CRYPTO_PATH = RAW_DIR / "ohlcv_hourly_crypto.parquet"

TECH_DAILY_STOCK_PATH = PROCESSED_DIR / "tech_daily_stock.parquet"
DC_DAILY_STOCK_PATH = PROCESSED_DIR / "dc_daily_stock.parquet"
DC_EVENTS_DAILY_STOCK_PATH = PROCESSED_DIR / "dc_events_daily_stock.parquet"
MACRO_STOCK_PATH = PROCESSED_DIR / "macro_stock.parquet"
TARGET_PANEL_PATH = PROCESSED_DIR / "panel_with_targets.parquet"

SPLIT_PANEL_PATH = PANEL_DIR / "panel_with_targets_and_split.parquet"
MAIN_PANEL_PATH = PANEL_DIR / "panel_daily_stock_main.parquet"
VALIDATION_REPORT_JSON_PATH = PANEL_DIR / "panel_daily_stock_main.validation_report.json"
VALIDATION_ASSET_SUMMARY_CSV_PATH = PANEL_DIR / "panel_daily_stock_main.validation_by_asset.csv"

# 旧名字兼容
PANEL_DAILY_STOCK_PATH = MAIN_PANEL_PATH
PANEL_HOURLY_STOCK_PATH = PANEL_DIR / "panel_hourly_stock.parquet"
PANEL_HOURLY_CRYPTO_PATH = PANEL_DIR / "panel_hourly_crypto.parquet"
TECHNICAL_FEATURES_PATH = TECH_DAILY_STOCK_PATH
DC_EVENTS_PATH = DC_EVENTS_DAILY_STOCK_PATH
MACRO_FEATURES_PATH = MACRO_STOCK_PATH
FUNDAMENTAL_FEATURES_PATH = PROCESSED_DIR / "fundamental_features.parquet"
ESG_STATIC_PATH = PROCESSED_DIR / "esg_static.parquet"
VALIDATION_REPORT_PATH = DIAGNOSTIC_DIR / "validation_report.html"
DC_SENSITIVITY_REPORT_PATH = DIAGNOSTIC_DIR / "dc_sensitivity.html"

# ======================================================================
#  杂项
# ======================================================================
RANDOM_SEED: int = 42
YF_CACHE_ENABLED: bool = True


# ======================================================================
#  辅助函数
# ======================================================================
def theta_to_bp(theta: float) -> str:
    return f"t{int(round(theta * 10000)):03d}"


def get_dc_feature_names(theta: float) -> List[str]:
    tag = theta_to_bp(theta)
    return [
        f"dc_event_{tag}",
        f"dc_trend_{tag}",
        f"dc_tmv_{tag}",
        f"dc_osv_{tag}",
        f"dc_age_{tag}",
    ]


def get_default_dc_ready_cols() -> List[str]:
    """默认用 dc_age_* 作为 DC readiness 检查列。"""
    return [f"dc_age_{theta_to_bp(t)}" for t in DC_THETAS_DAILY]


def summarize_config() -> Dict:
    return {
        "project_root": str(PROJECT_ROOT),
        "scope": "stock_daily_main_only",
        "stock_universe": STOCK_UNIVERSE,
        "stock_benchmark": STOCK_BENCHMARK,
        "daily_date_range": [DAILY_START, DAILY_END],
        "dc_thetas_daily": DC_THETAS_DAILY,
        "target_horizons": TARGET_HORIZONS,
        "label_main_horizon": LABEL_MAIN_HORIZON,
        "default_label_col": DEFAULT_LABEL_COL,
        "label_families": LABEL_FAMILIES,
        "band_bps": {"h1": BAND_BPS_H1, "h5": BAND_BPS_H5},
        "drop_benchmark_from_main_panel": DROP_BENCHMARK_FROM_MAIN_PANEL,
        "burn_in_days": BURN_IN_DAYS,
        "split": {
            "train_end": DAILY_TRAIN_END,
            "val_end": DAILY_VAL_END,
            "walk_forward_enabled": WALK_FORWARD_ENABLED,
        },
        "macro_tickers": {
            "vix": VIX_TICKER,
            "ust10y": UST10Y_TICKER,
            "hyg": HYG_TICKER,
        },
        "dc_feature_example_t100": get_dc_feature_names(0.01),
        "default_dc_ready_cols": get_default_dc_ready_cols(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summarize_config(), indent=2, default=str, ensure_ascii=False))

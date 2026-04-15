from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# Core directories
# -----------------------------
DATA_ACQ_DIR = PROJECT_ROOT / "data_acquisition"
CLEANED_DATA_DIR = DATA_ACQ_DIR / "cleaned_datasets"
INTERMEDIATE_DATA_DIR = DATA_ACQ_DIR / "intermediate"

DOWNLOAD_DATA_DIR = PROJECT_ROOT / "download_data"
ESG_DATA_DIR = DOWNLOAD_DATA_DIR / "esg_data"

MODEL_RUNS_DIR = PROJECT_ROOT / "Model Runs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# -----------------------------
# Common data paths
# -----------------------------
RAW_STOCK_PANEL_PATH = ESG_DATA_DIR / "stock_panel_raw.csv"

MODEL_READY_PANEL_PATH = INTERMEDIATE_DATA_DIR / "stock_panel_model_ready.csv"
BENCHMARK_STATE_PATH = INTERMEDIATE_DATA_DIR / "benchmark_state.csv"
FEATURE_STATS_PATH = INTERMEDIATE_DATA_DIR / "feature_stats_stock.npz"
DATA_ENG_SUMMARY_PATH = INTERMEDIATE_DATA_DIR / "data_engineering_summary.json"

MAIN_H1_DATA = CLEANED_DATA_DIR / "main_experiment_h1.csv"
MAIN_H5_DATA = CLEANED_DATA_DIR / "main_experiment_h5.csv"
EXT_H1_DATA = CLEANED_DATA_DIR / "extension_fundamental_2025plus_h1.csv"
EXT_H5_DATA = CLEANED_DATA_DIR / "extension_fundamental_2025plus_h5.csv"
CLEAN_BASE_DATA = CLEANED_DATA_DIR / "clean_base_after_warmup.csv"
FEATURE_CONFIG_PATH = CLEANED_DATA_DIR / "feature_config.json"
DATASET_SUMMARY_PATH = CLEANED_DATA_DIR / "dataset_summary.json"

# -----------------------------
# Structured Model Runs dirs
# -----------------------------
EXPERIMENTS_DIR = MODEL_RUNS_DIR / "experiments"
EVALUATION_DIR = MODEL_RUNS_DIR / "evaluation"
EXECUTION_DIR = MODEL_RUNS_DIR / "execution"
ABLATIONS_DIR = MODEL_RUNS_DIR / "ablations"
ROBUSTNESS_DIR = MODEL_RUNS_DIR / "robustness"
REPORTS_DIR = MODEL_RUNS_DIR / "reports"
LIVE_DIR = MODEL_RUNS_DIR / "live"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_all_core_dirs() -> None:
    for p in [
        DATA_ACQ_DIR,
        CLEANED_DATA_DIR,
        INTERMEDIATE_DATA_DIR,
        DOWNLOAD_DATA_DIR,
        ESG_DATA_DIR,
        MODEL_RUNS_DIR,
        EXPERIMENTS_DIR,
        EVALUATION_DIR,
        EXECUTION_DIR,
        ABLATIONS_DIR,
        ROBUSTNESS_DIR,
        REPORTS_DIR,
        LIVE_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
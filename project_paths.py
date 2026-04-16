from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# =========================================================
# Core project directories
# =========================================================
DATA_ACQUISITION_DIR = PROJECT_ROOT / "data_acquisition"
DATA_ACQ_DIR = DATA_ACQUISITION_DIR  # backward-compatible alias

CLEANED_DATA_DIR = DATA_ACQUISITION_DIR / "cleaned_datasets"
INTERMEDIATE_DATA_DIR = DATA_ACQUISITION_DIR / "intermediate"

DOWNLOAD_DATA_DIR = PROJECT_ROOT / "download_data"
ESG_DATA_DIR = DOWNLOAD_DATA_DIR / "esg_data"

MODEL_RUNS_DIR = PROJECT_ROOT / "Model Runs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# =========================================================
# Raw / active data products
# =========================================================
RAW_STOCK_PANEL_PATH = ESG_DATA_DIR / "stock_panel_raw.csv"

# Main experiment datasets
MAIN_H1_DATA = CLEANED_DATA_DIR / "main_experiment_h1.csv"
MAIN_H5_DATA = CLEANED_DATA_DIR / "main_experiment_h5.csv"

# Extension datasets
EXT_H1_DATA = CLEANED_DATA_DIR / "extension_fundamental_2025plus_h1.csv"
EXT_H5_DATA = CLEANED_DATA_DIR / "extension_fundamental_2025plus_h5.csv"

# Clean base / metadata
CLEAN_BASE_DATA = CLEANED_DATA_DIR / "clean_base_after_warmup.csv"
FEATURE_CONFIG_PATH = CLEANED_DATA_DIR / "feature_config.json"
DATASET_SUMMARY_PATH = CLEANED_DATA_DIR / "dataset_summary.json"

# =========================================================
# Legacy / transitional intermediate artifacts
# NOTE:
# These are kept temporarily for backward compatibility with
# older scripts. They should NOT be treated as the canonical
# outputs of the new main pipeline.
# =========================================================
LEGACY_MODEL_READY_PANEL_PATH = INTERMEDIATE_DATA_DIR / "stock_panel_model_ready.csv"
LEGACY_BENCHMARK_STATE_PATH = INTERMEDIATE_DATA_DIR / "benchmark_state.csv"
LEGACY_FEATURE_STATS_PATH = INTERMEDIATE_DATA_DIR / "feature_stats_stock.npz"
LEGACY_DATA_ENG_SUMMARY_PATH = INTERMEDIATE_DATA_DIR / "data_engineering_summary.json"

# Backward-compatible aliases
MODEL_READY_PANEL_PATH = LEGACY_MODEL_READY_PANEL_PATH
BENCHMARK_STATE_PATH = LEGACY_BENCHMARK_STATE_PATH
FEATURE_STATS_PATH = LEGACY_FEATURE_STATS_PATH
DATA_ENG_SUMMARY_PATH = LEGACY_DATA_ENG_SUMMARY_PATH

# =========================================================
# Structured Model Runs dirs (active)
# =========================================================
EXPERIMENTS_DIR = MODEL_RUNS_DIR / "experiments"
EVALUATION_DIR = MODEL_RUNS_DIR / "evaluation"
EXECUTION_DIR = MODEL_RUNS_DIR / "execution"
ROBUSTNESS_DIR = MODEL_RUNS_DIR / "robustness"
REPORTING_DIR = MODEL_RUNS_DIR / "reporting"
LIVE_DIR = MODEL_RUNS_DIR / "live"

# Optional backward-compatible alias
REPORTS_DIR = REPORTING_DIR

# =========================================================
# Archive / cleanup dirs
# =========================================================
ARCHIVE_DIR = PROJECT_ROOT / "archive"
ARCHIVE_EXECUTION_DIR = ARCHIVE_DIR / "execution_legacy"
ARCHIVE_RL_DIR = ARCHIVE_DIR / "rl_legacy"
ARCHIVE_MANUAL_TUNING_DIR = ARCHIVE_DIR / "manual_tuning"
ARCHIVE_STICKY_STATIC_DIR = ARCHIVE_DIR / "sticky_static_branch"

# =========================================================
# Helper functions
# =========================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_all_core_dirs() -> None:
    for p in [
        DATA_ACQUISITION_DIR,
        CLEANED_DATA_DIR,
        INTERMEDIATE_DATA_DIR,
        DOWNLOAD_DATA_DIR,
        ESG_DATA_DIR,
        MODEL_RUNS_DIR,
        EXPERIMENTS_DIR,
        EVALUATION_DIR,
        EXECUTION_DIR,
        ROBUSTNESS_DIR,
        REPORTING_DIR,
        LIVE_DIR,
        ARCHIVE_DIR,
        ARCHIVE_EXECUTION_DIR,
        ARCHIVE_RL_DIR,
        ARCHIVE_MANUAL_TUNING_DIR,
        ARCHIVE_STICKY_STATIC_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
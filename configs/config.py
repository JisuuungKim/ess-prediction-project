from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "archive"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/model_outputs"
DEFAULT_FEATURE_CACHE_DIR = PROJECT_ROOT / "outputs/feature_cache"

DEFAULT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

DEFAULT_ALPHA_GRID = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
]
DEFAULT_L1_RATIO_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

FEATURE_BLOCKS = {
    "summary": [
        "mean_chargetime",
        "temp_rise_100",
        "baseline_QD",
        "baseline_QC",
        "mean_Tavg",
    ],
    "charging": [
        "mean_c_rate",
        "max_c_rate",
        "policy_steps",
    ],
    "fade": [],
    "delta_q": [
        "delta_q_std",
        "delta_q_highV_mean",
        "delta_q_max",
    ],
}

BLOCK_MAX_FEATURES = {
    "summary": 4,
    "charging": 3,
    "fade": 0,
    "delta_q": 3,
}

HIGH_CORR_THRESHOLD = 0.95
VIF_ALERT_THRESHOLD = 10.0

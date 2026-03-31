from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data/raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/model_outputs"
DEFAULT_FEATURE_CACHE_DIR = PROJECT_ROOT / "outputs/feature_cache"

DEFAULT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

DEFAULT_ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]
DEFAULT_L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

FEATURE_BLOCKS = {
    "summary": [
        "mean_QD",
        "std_QD",
        "qd_drop_100",
        "qd_cv_100",
        "mean_IR",
        "std_IR",
        "ir_rise_100",
        "mean_Tavg",
        "mean_Tmax",
        "mean_Tmin",
        "temp_rise_100",
        "mean_chargetime",
        "chargetime_rise_100",
    ],
    "charging": [
        "max_c_rate",
        "mean_c_rate",
        "switch_soc_pct",
        "policy_steps",
    ],
    "fade": [
        "knee_cycle",
        "baseline_fade_rate",
        "post_knee_fade_rate",
        "fade_acceleration_ratio",
    ],
    "delta_q": [
        "delta_q_mean",
        "delta_q_std",
        "delta_q_min",
        "delta_q_max",
        "delta_q_range",
        "delta_q_abs_area",
        "delta_q_signed_area",
        "delta_q_lowV_mean",
        "delta_q_midV_mean",
        "delta_q_highV_mean",
    ],
}

BLOCK_MAX_FEATURES = {
    "summary": 6,
    "charging": 4,
    "fade": 4,
    "delta_q": 5,
}

HIGH_CORR_THRESHOLD = 0.95
VIF_ALERT_THRESHOLD = 10.0

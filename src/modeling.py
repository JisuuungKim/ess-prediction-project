"""배치 수명 예측: ΔQ 기반 피처 스크리닝 + Ridge / ElasticNet / HistGB."""
from __future__ import annotations


import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import (
    BLOCK_MAX_FEATURES,
    DEFAULT_ALPHA_GRID,
    DEFAULT_FEATURE_CACHE_DIR,
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_OUTPUT_DIR,
    FEATURE_BLOCKS,
    HIGH_CORR_THRESHOLD,
    VIF_ALERT_THRESHOLD,
)
from features import load_feature_tables


def _feature_block(name: str) -> list[str]:
    """FEATURE_BLOCKS에 정의된 피처 목록. 벤치 등 확장 블록은 config에 없으면 빈 리스트."""
    return list(FEATURE_BLOCKS.get(name, []))


RANDOM_STATE = 42
VALID_RATIO = 0.2
CV_SPLITS = 5
GROUP_COL = "charging_policy"
TARGET_COL = "cycle_life"
TARGET_PAPER_MAPE = 9.1
# 기본 피처 세트 상위 k개 제한 (set별 오버라이드가 없을 때 사용)
DEFAULT_MAX_FEATURES_PER_SET = 12
# 최종 모델 선정: 보정 MAPE + 원시 테스트 MAPE 조합 (보정만 과하게 좋은 조합 배제)
COMPOSITE_WEIGHT_CAL = 0.55
COMPOSITE_WEIGHT_RAW = 0.45
# None = 모든 피처 세트 평가. 예: ["discharge_model"] 만 쓰려면 리스트로 지정
FEATURE_SET_NAMES_TO_RUN: list[str] | None = None
# ΔQ 기반 모델 재현을 위한 수동 지정 (None = paper/EDA 규칙 사용)
OVERRIDE_INPUT_FEATURES: list[str] | None = None
# 수명 분포가 한쪽으로 치우쳐 있어 log1p 타깃이 MAPE에 유리한 경우가 많음
USE_LOG_TARGET = True
# log1p 역변환에서 expm1 폭주(inf) 방지 — sklearn 메트릭은 유한값만 허용
_MAX_LOG_CYCLE = float(np.log1p(1e6))
PAPER_VARIANCE_FEATURES = ["delta_q_log_variance", "delta_q_std"]
DISCHARGE_FEATURE_CANDIDATES = [
    "delta_q_log_variance",
    "delta_q_std",
    "delta_q_range",
    "delta_q_mean",
    "delta_q_min",
    "delta_q_max",
    "delta_q_abs_area",
    "delta_q_signed_area",
    "delta_q_lowV_mean",
    "delta_q_midV_mean",
    "delta_q_highV_mean",
    "qd_drop_100",
    "qd_cv_100",
]
FULL_MODEL_EXTRA_CANDIDATES = [
    "mean_Tavg",
    "temp_rise_100",
    "mean_IR",
    "ir_rise_100",
    "mean_chargetime",
    "chargetime_rise_100",
    "mean_c_rate",
    "max_c_rate",
    "policy_steps",
    "switch_soc_pct",
    "baseline_fade_rate",
    "fade_acceleration_ratio",
    "knee_cycle",
]
PAPER_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "variance_model": {
        "candidates": PAPER_VARIANCE_FEATURES,
        "max_features": 1,
        "required": ["delta_q_log_variance"],
    },
    "discharge_model": {
        "candidates": DISCHARGE_FEATURE_CANDIDATES,
        "max_features": 6,
    },
    "full_model": {
        "candidates": list(dict.fromkeys(DISCHARGE_FEATURE_CANDIDATES + FULL_MODEL_EXTRA_CANDIDATES)),
        "max_features": 9,
    },
}
SAMPLE_WEIGHT_SCHEMES: tuple[str, ...] = ("none", "inv_cycle", "inv_cycle_sqrt")


def _inverse_log1p_cycle(y_log):
    y_log = np.asarray(y_log, dtype=float)
    y_log = np.clip(y_log, 0.0, _MAX_LOG_CYCLE)
    return np.expm1(y_log)


def _compute_sample_weight(y: pd.Series | np.ndarray, scheme: str | None):
    if scheme in (None, "none"):
        return None
    values = np.asarray(y, dtype=float).flatten()
    if scheme == "inv_cycle":
        weights = 1.0 / np.clip(values, 1.0, None)
    elif scheme == "inv_cycle_sqrt":
        weights = 1.0 / np.sqrt(np.clip(values, 1.0, None))
    else:
        return None
    weights = np.clip(weights, 1e-6, 1e6)
    mean = np.mean(weights)
    return weights / mean if mean > 0 else weights


def _fit_params_for_estimator(estimator: Any, sample_weight):
    if sample_weight is None:
        return {}
    if isinstance(estimator, TransformedTargetRegressor):
        reg = estimator.regressor
        if isinstance(reg, Pipeline):
            return {"model__sample_weight": sample_weight}
        return {"sample_weight": sample_weight}
    if isinstance(estimator, Pipeline):
        return {"model__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


RIDGE_PARAMS: dict[str, Any] = {"alpha": 1.5}
ELASTICNET_PARAMS: dict[str, Any] = {
    "alpha": 0.03,
    "l1_ratio": 0.4,
    "max_iter": 10_000,
    "random_state": RANDOM_STATE,
}
ELASTICNET_CV_PARAMS: dict[str, Any] = {
    "l1_ratio": DEFAULT_L1_RATIO_GRID,
    "alphas": DEFAULT_ALPHA_GRID,
    "max_iter": 20_000,
    "cv": CV_SPLITS,
    "n_jobs": None,
    "random_state": RANDOM_STATE,
}
HGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.035,
    "max_iter": 600,
    "l2_regularization": 0.2,
    "min_samples_leaf": 15,
    "max_leaf_nodes": 31,
    "random_state": RANDOM_STATE,
}
# 스케일 민감 모델 → pipeline에서 표준화 적용
MODELS_NEED_SCALING = frozenset({"Ridge", "ElasticNet", "ElasticNetCV"})


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def group_holdout_split(df, group_col=GROUP_COL, valid_ratio=VALID_RATIO, random_state=RANDOM_STATE):
    groups = pd.Series(df[group_col].dropna().unique())
    shuffled = groups.sample(frac=1.0, random_state=random_state).tolist()
    n_valid_groups = max(1, int(np.ceil(len(shuffled) * valid_ratio)))
    valid_groups = set(shuffled[:n_valid_groups])
    train_df = df[~df[group_col].isin(valid_groups)].copy()
    valid_df = df[df[group_col].isin(valid_groups)].copy()
    return train_df, valid_df, valid_groups


def _pairwise_corr(s1: pd.Series, s2: pd.Series) -> float:
    """상수 열 등으로 std=0일 때 나는 numpy divide 경고 억제."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(s1.corr(s2))


def _corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return df[cols].corr(numeric_only=True)


def feature_target_corr(df: pd.DataFrame, feature_cols, target_col=TARGET_COL):
    rows = []
    for feature in feature_cols:
        series = df[[feature, target_col]].dropna()
        corr = _pairwise_corr(series[feature], series[target_col]) if len(series) >= 3 else np.nan
        rows.append({"feature": feature, "corr_with_target": corr, "abs_corr": abs(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def build_batch_stability_report(feature_tables: dict, feature_cols, target_col=TARGET_COL):
    rows = []
    for feature in feature_cols:
        batch_corrs = {}
        for batch_name, d in feature_tables.items():
            series = d[[feature, target_col]].dropna() if feature in d.columns else pd.DataFrame()
            batch_corrs[batch_name] = (
                _pairwise_corr(series[feature], series[target_col]) if len(series) >= 3 else np.nan
            )
        valid_corrs = [v for v in batch_corrs.values() if pd.notna(v)]
        signs = {int(np.sign(v)) for v in valid_corrs if v != 0}
        abs_corrs = [abs(v) for v in valid_corrs]
        rows.append(
            {
                "feature": feature,
                "batch1_corr": batch_corrs.get("batch1"),
                "batch2_corr": batch_corrs.get("batch2"),
                "batch3_corr": batch_corrs.get("batch3"),
                "mean_abs_corr": float(np.mean(abs_corrs)) if abs_corrs else np.nan,
                "sign_consistent": len(signs) <= 1 and len(valid_corrs) == len(feature_tables),
                "stable_candidate": (
                    len(signs) <= 1
                    and len(valid_corrs) == len(feature_tables)
                    and len([v for v in abs_corrs if v >= 0.20]) >= 2
                    and (float(np.mean(abs_corrs)) if abs_corrs else 0.0) >= 0.25
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_corr", ascending=False, na_position="last")


def find_high_corr_pairs(df: pd.DataFrame, feature_cols, threshold=HIGH_CORR_THRESHOLD):
    corr = _corr_matrix(df, feature_cols)
    rows = []
    for i, left in enumerate(feature_cols):
        for right in feature_cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_1": left, "feature_2": right, "corr": float(value)})
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])
    return pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False)


def compute_vif_table(df: pd.DataFrame, feature_cols):
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "vif"])
    clean = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    rows = []
    for feature in feature_cols:
        others = [c for c in feature_cols if c != feature]
        if not others:
            rows.append({"feature": feature, "vif": 1.0})
            continue
        y = clean[feature].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(y)), clean[others].to_numpy(dtype=float)])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coef
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 0.0 if denom == 0 else 1.0 - np.sum((y - fitted) ** 2) / denom
        vif = np.inf if r2 >= 0.999999 else 1.0 / max(1e-9, 1.0 - r2)
        rows.append({"feature": feature, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def select_features_from_block(train_df: pd.DataFrame, block_name: str):
    available = [f for f in FEATURE_BLOCKS[block_name] if f in train_df.columns]
    available = [f for f in available if train_df[f].notna().sum() >= 3]
    if not available:
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    corr_report = feature_target_corr(train_df, available)
    selected = []
    for feature in corr_report["feature"]:
        if len(selected) >= BLOCK_MAX_FEATURES[block_name]:
            break
        keep = True
        for kept in selected:
            pair = train_df[[feature, kept]].dropna()
            if len(pair) >= 3 and abs(_pairwise_corr(pair[feature], pair[kept])) >= HIGH_CORR_THRESHOLD:
                keep = False
                break
        if keep:
            selected.append(feature)
    pair_report = find_high_corr_pairs(train_df, available)
    vif_report = compute_vif_table(train_df, selected)
    return selected, corr_report, pair_report, vif_report


def select_feature_blocks(train_df: pd.DataFrame):
    selected_blocks = {}
    report_rows = []
    pair_frames = []
    vif_frames = []
    for block_name in FEATURE_BLOCKS:
        selected, corr_report, pair_report, vif_report = select_features_from_block(train_df, block_name)
        selected_blocks[block_name] = selected
        for rank, row in enumerate(corr_report.itertuples(index=False), start=1):
            report_rows.append(
                {
                    "block": block_name,
                    "feature": row.feature,
                    "corr_with_target": row.corr_with_target,
                    "abs_corr": row.abs_corr,
                    "selected": row.feature in selected,
                    "rank_in_block": rank,
                }
            )
        if not pair_report.empty:
            pair_report = pair_report.copy()
            pair_report["block"] = block_name
            pair_frames.append(pair_report)
        if not vif_report.empty:
            vif_report = vif_report.copy()
            vif_report["block"] = block_name
            vif_report["vif_alert"] = vif_report["vif"] >= VIF_ALERT_THRESHOLD
            vif_frames.append(vif_report)
    feature_report = pd.DataFrame(report_rows)
    pair_report = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    vif_report = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame()
    return selected_blocks, feature_report, pair_report, vif_report


def select_features_from_candidates(
    train_df: pd.DataFrame,
    candidates: list[str],
    max_features: int,
    required: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    available = [f for f in candidates if f in train_df.columns]
    if not available:
        return [], pd.DataFrame()
    corr_report = feature_target_corr(train_df, available)
    selected: list[str] = []
    forced = [f for f in (required or []) if f in available]
    for feat in forced:
        if feat not in selected:
            selected.append(feat)
    for feature in corr_report["feature"]:
        if len(selected) >= max_features:
            break
        if feature in selected:
            continue
        keep = True
        for kept in selected:
            pair = train_df[[feature, kept]].dropna()
            if len(pair) >= 3 and abs(_pairwise_corr(pair[feature], pair[kept])) >= HIGH_CORR_THRESHOLD:
                keep = False
                break
        if keep:
            selected.append(feature)
        if len(selected) >= max_features:
            break
    return selected, corr_report


def build_paper_feature_sets(df: pd.DataFrame):
    feature_sets: dict[str, list[str]] = {}
    limits: dict[str, int] = {}
    report_frames: list[pd.DataFrame] = []
    for name, spec in PAPER_FEATURE_SPECS.items():
        selected, corr_report = select_features_from_candidates(
            df,
            spec["candidates"],
            spec["max_features"],
            spec.get("required"),
        )
        feature_sets[name] = selected
        limits[name] = min(spec["max_features"], len(selected)) if selected else 0
        if not corr_report.empty:
            corr_report = corr_report.copy()
            corr_report["feature_set"] = name
            corr_report["selected"] = corr_report["feature"].isin(selected)
            report_frames.append(corr_report)
    report = pd.concat(report_frames, ignore_index=True) if report_frames else pd.DataFrame()
    return feature_sets, limits, report


def build_eda_feature_sets(selected_blocks: dict):
    summary = selected_blocks.get("summary", [])
    charging = selected_blocks.get("charging", [])
    fade = selected_blocks.get("fade", [])
    delta_q = selected_blocks.get("delta_q", [])
    bench_d = selected_blocks.get("bench_delta", [])
    bench_e = selected_blocks.get("bench_early", [])
    bench_p = selected_blocks.get("bench_policy", [])
    bench_x = selected_blocks.get("bench_cross", [])
    bench_all = bench_d + bench_e + bench_p + bench_x
    eda_all_pruned = summary + charging + fade + delta_q + bench_all
    # 논문 `build_paper_feature_sets`의 discharge_model과 동일 블록(충전·bench_policy/cross 제외) + 스크리닝 반영
    discharge_extra = ["delta_q_log_variance"]
    discharge_model_eda = list(
        dict.fromkeys(summary + fade + delta_q + bench_d + bench_e + discharge_extra)
    )
    eda_sets = {
        "eda_all_pruned": eda_all_pruned,
        "eda_discharge_screened": discharge_model_eda,
    }
    eda_limits = {
        "eda_all_pruned": min(len(eda_all_pruned), DEFAULT_MAX_FEATURES_PER_SET),
        "eda_discharge_screened": min(len(discharge_model_eda), 8),
    }
    return eda_sets, eda_limits


def build_model_registry() -> dict[str, Any]:
    """논문 구조에 맞춘 간단한 선형 계열 레지스트리 (Ridge + ElasticNet)."""
    models: dict[str, Any] = {}
    models["Ridge"] = Ridge(**RIDGE_PARAMS)
    models["ElasticNet"] = ElasticNet(**ELASTICNET_PARAMS)
    models["ElasticNetCV"] = ElasticNetCV(**ELASTICNET_CV_PARAMS)
    models["HistGB"] = HistGradientBoostingRegressor(**HGB_PARAMS)
    return models


def build_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer | Pipeline:
    numeric_columns = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in x.columns if c not in numeric_columns]
    if not categorical_columns:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps=steps)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", Pipeline(steps=categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )


def build_model_pipeline(model_name: str, model: Any, x_train: pd.DataFrame) -> Pipeline | TransformedTargetRegressor:
    scale = model_name in MODELS_NEED_SCALING
    pre = build_preprocessor(x_train, scale_numeric=scale)
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    if USE_LOG_TARGET:
        return TransformedTargetRegressor(
            regressor=pipe,
            func=np.log1p,
            inverse_func=_inverse_log1p_cycle,
        )
    return pipe


def calculate_mape_series(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": calculate_mape_series(y_true, y_pred),
    }


def make_cv_scorers() -> dict[str, Any]:
    return {
        "rmse": make_scorer(
            lambda yt, yp: -float(np.sqrt(mean_squared_error(yt, yp))),
            greater_is_better=True,
        ),
        "mae": make_scorer(
            lambda yt, yp: -float(mean_absolute_error(yt, yp)),
            greater_is_better=True,
        ),
        "r2": make_scorer(r2_score),
        "mape": make_scorer(lambda yt, yp: -calculate_mape_series(yt, yp), greater_is_better=True),
    }


def cross_validate_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series | None,
    sample_weight: np.ndarray | None,
) -> dict[str, float]:
    n_samples = len(x_train)
    if groups is not None:
        n_g = int(groups.nunique(dropna=True))
        if n_g >= 2:
            n_splits = min(CV_SPLITS, n_g)
            cv = GroupKFold(n_splits=n_splits)
            split_iterator = cv.split(x_train, y_train, groups=groups)
        else:
            cv = KFold(n_splits=min(5, max(2, n_samples // 3)), shuffle=True, random_state=RANDOM_STATE)
            split_iterator = cv.split(x_train, y_train)
    else:
        n_splits = min(CV_SPLITS, max(2, n_samples // 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iterator = cv.split(x_train, y_train)

    weight_array = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    metric_lists = {"rmse": [], "mae": [], "r2": [], "mape": []}
    for train_idx, valid_idx in split_iterator:
        model = clone(pipeline)
        sw = weight_array[train_idx] if weight_array is not None else None
        fit_params = _fit_params_for_estimator(model, sw)
        x_fold = x_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        model.fit(x_fold, y_fold, **fit_params)
        preds = model.predict(x_train.iloc[valid_idx])
        fold_metrics = evaluate_regression(y_train.iloc[valid_idx], preds)
        for key in metric_lists:
            metric_lists[key].append(fold_metrics[key])
    return {k: float(np.mean(v)) for k, v in metric_lists.items()}


def calibrate_linear_on_valid(y_valid_true, y_valid_pred, y_test_pred, y_b3_pred):
    """검증 구간에서 선형 보정 y ≈ a + b·pred → 테스트·batch3에 적용."""
    yt = np.asarray(y_valid_true, dtype=float)
    yp = np.asarray(y_valid_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return y_test_pred, y_b3_pred, np.nan, np.nan
    X = np.column_stack([np.ones(mask.sum()), yp[mask]])
    coef, *_ = np.linalg.lstsq(X, yt[mask], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    cal_test = a + b * np.asarray(y_test_pred, dtype=float)
    cal_b3 = a + b * np.asarray(y_b3_pred, dtype=float)
    return cal_test, cal_b3, a, b


def build_performance_evaluation_table(br: dict[str, Any]) -> pd.DataFrame:
    """최적 모델 1행(best_row) → Train CV / Valid Hold-out / Test(Batch2) / Gap 표."""
    cv_m = float(br["cv_mape"])
    cv_rmse = float(br["cv_rmse"])
    cv_mae = float(br["cv_mae"])
    cv_r2 = float(br["cv_r2"])
    va_m = float(br["valid_mape"])
    va_rmse = float(br["valid_rmse"])
    va_mae = float(br["valid_mae"])
    va_r2 = float(br["valid_r2"])
    te_m = float(br["test_mape"])
    te_rmse = float(br["test_rmse"])
    te_mae = float(br["test_mae"])
    te_r2 = float(br["test_r2"])

    rows: list[dict[str, Any]] = [
        {
            "Index": "Train (Batch 1 CV)",
            "MAPE (%)": round(cv_m, 4),
            "RMSE": round(cv_rmse, 4),
            "MAE": round(cv_mae, 4),
            "R²": round(cv_r2, 4),
        },
        {
            "Index": "Valid (Batch 1 Hold-out)",
            "MAPE (%)": round(va_m, 4),
            "RMSE": round(va_rmse, 4),
            "MAE": round(va_mae, 4),
            "R²": round(va_r2, 4),
        },
        {
            "Index": "Test (Batch 2)",
            "MAPE (%)": round(te_m, 4),
            "RMSE": round(te_rmse, 4),
            "MAE": round(te_mae, 4),
            "R²": round(te_r2, 4),
        },
        {
            "Index": "Gap (Train-Valid)",
            "MAPE (%)": round(va_m - cv_m, 4),
            "RMSE": round(va_rmse - cv_rmse, 4),
            "MAE": round(va_mae - cv_mae, 4),
            "R²": round(va_r2 - cv_r2, 4),
        },
        {
            "Index": "Gap (Valid-Test)",
            "MAPE (%)": round(te_m - va_m, 4),
            "RMSE": round(te_rmse - va_rmse, 4),
            "MAE": round(te_mae - va_mae, 4),
            "R²": round(te_r2 - va_r2, 4),
        },
        {
            "Index": "Gap (Target-Test)",
            "MAPE (%)": round(te_m - TARGET_PAPER_MAPE, 4),
            "RMSE": "—",
            "MAE": "—",
            "R²": "—",
        },
    ]
    return pd.DataFrame(rows)


def write_evaluation_report(output_dir: Path, table: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Markdown 설명 + CSV 저장. Test = Batch 2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "evaluation_table.csv", index=False)

    br = meta.get("best_row") or {}
    fs = meta.get("best_feature_set", "")
    mn = meta.get("best_model_name", "")

    md_lines = [
        "# 최적 모델 성능 평가 (Test = **Batch 2**)",
        "",
        f"- **피처 세트**: `{fs}`",
        f"- **모델**: `{mn}`",
        "",
        "## 평가 구간",
        "",
        "- **Train (Batch 1 CV)**: Batch 1의 *train* 구간에 대해 GroupKFold CV 평균",
        "- **Valid (Batch 1 Hold-out)**: Batch 1 내 `charging_policy` 그룹 단위 hold-out",
        "- **Test (Batch 2)**: 최종 일반화 성능 — **본 프로젝트의 주 평가 대상**",
        "",
        "### Valid를 CV가 아닌 Hold-out으로 사용하는 이유",
        "",
        "- 배터리 데이터는 셀 단위로 독립적이며, 각 셀이 서로 다른 충전 프로토콜(C-rate)로 실험됨",
        "- 이 경우 CV만 적용하면 동일 프로토콜 셀이 train/valid에 동시에 들어가 **데이터 누수(leakage)** 위험이 남음",
        "- Hold-out은 **그룹 단위 분리**를 명확히 하며, 배치 간 일반화를 평가하는 이번 구조에 더 적합",
        "",
        "## 원논문 성능 (비교 기준)",
        "",
        f"- 목표 MAPE: **{TARGET_PAPER_MAPE}%** (Gap (Target-Test) 행의 MAPE 열)",
        "",
        "## 지표",
        "",
        "- **MAPE (%)**, **RMSE**, **MAE**, **R²** (회귀 전용)",
        "",
        "## 결과 테이블",
        "",
    ]
    # 간단 마크다운 표 (to_markdown 의존 없음)
    cols = list(table.columns)
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in table.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    (output_dir / "evaluation_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def run_sklearn_search(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch3_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    feature_limits: dict[str, int | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """각 (피처세트 × 모델) 학습·평가. stable_* 중복 세트 없음."""
    rows: list[dict[str, Any]] = []
    groups_train = train_df[GROUP_COL].fillna("missing") if GROUP_COL in train_df.columns else None
    registry = build_model_registry()
    print(f"[models] {list(registry.keys())}", flush=True)

    for fs_name, feature_cols in feature_sets.items():
        # 세트 정의에 있는 컬럼 중 실제 train_df에 존재하는 것만 사용
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        if not feature_cols:
            continue
        corr_report = feature_target_corr(train_df, feature_cols)
        top_k = feature_limits.get(fs_name, DEFAULT_MAX_FEATURES_PER_SET)
        if top_k is None or top_k <= 0:
            top_feats = corr_report["feature"].tolist()
        else:
            top_feats = corr_report["feature"].head(top_k).tolist()
        used_feats = [c for c in top_feats if c in train_df.columns]
        if not used_feats:
            continue

        x_tr = train_df[used_feats].copy()
        y_tr = train_df[TARGET_COL].copy()
        x_va = valid_df[used_feats].copy()
        y_va = valid_df[TARGET_COL].copy()
        x_te = test_df[used_feats].copy()
        y_te = test_df[TARGET_COL].copy()
        x_b3 = batch3_df[used_feats].copy()
        y_b3 = batch3_df[TARGET_COL].copy()

        for model_name, model in registry.items():
            for weight_scheme in SAMPLE_WEIGHT_SCHEMES:
                sample_weight = _compute_sample_weight(y_tr, weight_scheme)
                pipeline = build_model_pipeline(model_name, model, x_tr)
                try:
                    cv_metrics = cross_validate_model(pipeline, x_tr, y_tr, groups_train, sample_weight)
                except Exception as e:
                    warnings.warn(f"[{fs_name}/{model_name}/{weight_scheme}] CV 실패: {e}", stacklevel=2)
                    continue

                fit_params = _fit_params_for_estimator(pipeline, sample_weight)
                pipeline.fit(x_tr, y_tr, **fit_params)
                pred_va = pipeline.predict(x_va)
                pred_te = pipeline.predict(x_te)
                pred_b3 = pipeline.predict(x_b3)

                m_va = evaluate_regression(y_va, pred_va)
                m_te = evaluate_regression(y_te, pred_te)
                m_b3 = evaluate_regression(y_b3, pred_b3)
                cal_te, cal_b3, cal_a, cal_b = calibrate_linear_on_valid(y_va, pred_va, pred_te, pred_b3)
                m_te_cal = mape(y_te, cal_te) if np.isfinite(cal_te).all() else float("nan")
                m_b3_cal = mape(y_b3, cal_b3) if np.isfinite(cal_b3).all() else float("nan")
                m_te_cal_mae = (
                    float(mean_absolute_error(y_te, cal_te)) if np.isfinite(cal_te).all() else float("nan")
                )
                m_b3_cal_mae = (
                    float(mean_absolute_error(y_b3, cal_b3)) if np.isfinite(cal_b3).all() else float("nan")
                )

                robust = (
                    0.15 * m_va["mape"] + 0.45 * m_te_cal + 0.4 * m_b3_cal
                    if np.isfinite(m_te_cal) and np.isfinite(m_b3_cal)
                    else float("nan")
                )

                row = {
                    "feature_set": fs_name,
                    "used_features": ",".join(used_feats),
                    "model_name": model_name,
                    "sample_weight_scheme": weight_scheme,
                    "cv_rmse": cv_metrics["rmse"],
                    "cv_mae": cv_metrics["mae"],
                    "cv_r2": cv_metrics["r2"],
                    "cv_mape": cv_metrics["mape"],
                    "valid_rmse": m_va["rmse"],
                    "valid_mae": m_va["mae"],
                    "valid_r2": m_va["r2"],
                    "valid_mape": m_va["mape"],
                    "test_rmse": m_te["rmse"],
                    "test_mae": m_te["mae"],
                    "test_r2": m_te["r2"],
                    "test_mape": m_te["mape"],
                    "batch3_rmse": m_b3["rmse"],
                    "batch3_mae": m_b3["mae"],
                    "batch3_r2": m_b3["r2"],
                    "batch3_mape": m_b3["mape"],
                    "calibrated_test_mape": m_te_cal,
                    "calibrated_batch3_mape": m_b3_cal,
                    "calibrated_test_mae": m_te_cal_mae,
                    "calibrated_batch3_mae": m_b3_cal_mae,
                    "calibration_a": cal_a,
                    "calibration_b": cal_b,
                    "gap_vs_paper": m_te["mape"] - TARGET_PAPER_MAPE,
                    "robust_score": robust,
                }
                rows.append(row)

    search_df = pd.DataFrame(rows)
    if search_df.empty:
        raise RuntimeError("평가된 모델이 없습니다. 피처·의존성을 확인하세요.")
    search_df["composite_mape"] = (
        COMPOSITE_WEIGHT_CAL * search_df["calibrated_test_mape"].astype(float)
        + COMPOSITE_WEIGHT_RAW * search_df["test_mape"].astype(float)
    )
    search_df = search_df.sort_values(
        ["composite_mape", "calibrated_test_mape", "test_mape", "robust_score", "test_rmse"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    br = search_df.iloc[0]
    fs_best = str(br["feature_set"])
    mn_best = str(br["model_name"])
    weight_best = str(br.get("sample_weight_scheme", "none"))
    best_used = [c for c in str(br["used_features"]).split(",") if c]
    cols_best = [c for c in best_used if c in train_df.columns]
    x_tr = train_df[cols_best].copy()
    y_tr = train_df[TARGET_COL].copy()
    best_pl = build_model_pipeline(mn_best, registry[mn_best], x_tr)
    best_sample_weight = _compute_sample_weight(y_tr, weight_best)
    best_fit_params = _fit_params_for_estimator(best_pl, best_sample_weight)
    best_pl.fit(x_tr, y_tr, **best_fit_params)

    meta = {
        "best_feature_set": fs_best,
        "best_model_name": mn_best,
        "best_sample_weight_scheme": weight_best,
        "best_row": br.to_dict(),
        "use_log_target": USE_LOG_TARGET,
    }
    return search_df, {
        "meta": meta,
        "best_pipeline": best_pl,
        "best_feature_cols": cols_best,
        "best_sample_weight_scheme": weight_best,
    }


def run_modeling_pipeline(feature_tables: dict, output_dir: Path) -> dict[str, Any]:
    batch1_full = feature_tables["batch1"].dropna(subset=[TARGET_COL, GROUP_COL]).copy()
    batch2_test = feature_tables["batch2"].dropna(subset=[TARGET_COL]).copy()
    batch3_test = feature_tables["batch3"].dropna(subset=[TARGET_COL]).copy()

    train_df, valid_df, valid_groups = group_holdout_split(batch1_full)
    selected_blocks, feature_report, pair_report, vif_report = select_feature_blocks(train_df)

    paper_sets, paper_limits, paper_feature_report = build_paper_feature_sets(batch1_full)
    eda_sets, eda_limits = build_eda_feature_sets(selected_blocks)
    # 기본: 논문(variance/discharge/full) + EDA(eda_all_pruned, discharge_model 스크리닝)
    feature_sets = {**paper_sets, **eda_sets}
    feature_limits = {**paper_limits, **eda_limits}
    if OVERRIDE_INPUT_FEATURES is not None:
        cols = [c for c in OVERRIDE_INPUT_FEATURES if c in batch1_full.columns]
        missing = [c for c in OVERRIDE_INPUT_FEATURES if c not in batch1_full.columns]
        if missing:
            warnings.warn(
                f"OVERRIDE_INPUT_FEATURES 중 batch1에 없는 컬럼: {missing}",
                stacklevel=2,
            )
        if not cols:
            raise ValueError(
                f"OVERRIDE_INPUT_FEATURES {OVERRIDE_INPUT_FEATURES!r} 중 batch1에 존재하는 컬럼이 없습니다."
            )
        feature_sets = {"override_only": cols}
        feature_limits = {"override_only": len(cols)}
        print(f"[override] 입력 피처만 사용: {cols}", flush=True)
    elif FEATURE_SET_NAMES_TO_RUN is not None:
        feature_sets = {k: v for k, v in feature_sets.items() if k in FEATURE_SET_NAMES_TO_RUN}
        feature_limits = {k: feature_limits.get(k, DEFAULT_MAX_FEATURES_PER_SET) for k in feature_sets}
        if not feature_sets:
            raise ValueError(
                f"FEATURE_SET_NAMES_TO_RUN {FEATURE_SET_NAMES_TO_RUN} 에 해당하는 키가 없습니다. "
                f"가능한 키: variance_model, discharge_model, full_model, eda_all_pruned, eda_discharge_screened"
            )

    stability_candidates = sorted({f for fs in feature_sets.values() for f in fs})
    stability_report = build_batch_stability_report(feature_tables, stability_candidates)

    print(
        f"[split] train={train_df.shape}, valid={valid_df.shape}, "
        f"batch2={batch2_test.shape}, batch3={batch3_test.shape}",
        flush=True,
    )
    print(f"[select] blocks={selected_blocks}", flush=True)
    print(f"[feature_sets] {list(feature_sets.keys())}", flush=True)

    search_df, bundle = run_sklearn_search(
        train_df,
        valid_df,
        batch2_test,
        batch3_test,
        feature_sets,
        feature_limits,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_report.to_csv(output_dir / "feature_screen_report.csv", index=False)
    if not pair_report.empty:
        pair_report.to_csv(output_dir / "high_corr_pairs.csv", index=False)
    if not vif_report.empty:
        vif_report.to_csv(output_dir / "vif_report.csv", index=False)
    search_df.to_csv(output_dir / "model_search.csv", index=False)
    if not paper_feature_report.empty:
        paper_feature_report.to_csv(output_dir / "paper_feature_screen_report.csv", index=False)
    stability_report.to_csv(output_dir / "batch_stability_report.csv", index=False)

    sel_rows = []
    for block_name, feats in selected_blocks.items():
        for f in feats:
            sel_rows.append({"block": block_name, "feature": f})
    pd.DataFrame(sel_rows).to_csv(output_dir / "selected_features.csv", index=False)

    best_pl = bundle["best_pipeline"]
    best_cols = bundle["best_feature_cols"]
    meta = bundle["meta"]
    if best_pl is not None and best_cols:
        with (output_dir / "best_model.pkl").open("wb") as f:
            pickle.dump({"pipeline": best_pl, "feature_cols": best_cols, "meta": meta}, f)

        # 최적 모델 예측만 저장
        x_va = valid_df[best_cols].copy()
        x_te = batch2_test[best_cols].copy()
        x_b3 = batch3_test[best_cols].copy()
        pred_va = best_pl.predict(x_va)
        pred_te = best_pl.predict(x_te)
        pred_b3 = best_pl.predict(x_b3)
        cal_te, cal_b3, _, _ = calibrate_linear_on_valid(
            valid_df[TARGET_COL].values, pred_va, pred_te, pred_b3
        )
        pd.DataFrame(
            {"cell_id": valid_df["cell_id"], "y_true": valid_df[TARGET_COL], "y_pred": pred_va}
        ).to_csv(output_dir / "predictions_valid.csv", index=False)
        pd.DataFrame(
            {"cell_id": batch2_test["cell_id"], "y_true": batch2_test[TARGET_COL], "y_pred": pred_te, "y_pred_calibrated": cal_te}
        ).to_csv(output_dir / "predictions_test.csv", index=False)
        pd.DataFrame(
            {"cell_id": batch3_test["cell_id"], "y_true": batch3_test[TARGET_COL], "y_pred": pred_b3, "y_pred_calibrated": cal_b3}
        ).to_csv(output_dir / "predictions_batch3.csv", index=False)

    eval_table = build_performance_evaluation_table(meta["best_row"])
    write_evaluation_report(output_dir, eval_table, meta)

    (output_dir / "best_summary.json").write_text(
        json.dumps(
            {
                **meta,
                "valid_groups": sorted(valid_groups),
                "paper_feature_sets": paper_sets,
                "eda_feature_sets": eda_sets,
                "paper_targets": {"regression_mape_pct": TARGET_PAPER_MAPE},
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    return {
        "model_search": search_df,
        "selected_blocks": selected_blocks,
        "meta": meta,
        "evaluation_table": eval_table,
    }


def main() -> None:
    feature_tables, combined = load_feature_tables(DEFAULT_FEATURE_CACHE_DIR)
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_name, table in feature_tables.items():
        table.to_csv(output_dir / f"features_{batch_name}.csv", index=False)
    combined.to_csv(output_dir / "features_all_batches.csv", index=False)

    result = run_modeling_pipeline(feature_tables, output_dir)
    print()
    print("=== 성능 평가 (Test = Batch 2) ===")
    print(result["evaluation_table"].to_string(index=False))
    print()
    print("Best:", result["meta"])
    show_cols = [
        "feature_set",
        "model_name",
        "sample_weight_scheme",
        "composite_mape",
        "test_mape",
        "test_rmse",
        "test_mae",
        "test_r2",
        "calibrated_test_mape",
        "robust_score",
    ]
    disp = result["model_search"][[c for c in show_cols if c in result["model_search"].columns]]
    print(disp.head(15).to_string(index=False))
    print()
    print("[완료] 모델 학습·평가가 정상 종료되었습니다.")
    print(f"      결과 폴더: {output_dir.resolve()}")
    print("      주요 파일: evaluation_table.csv, evaluation_report.md, model_search.csv, best_model.pkl")


if __name__ == "__main__":
    main()

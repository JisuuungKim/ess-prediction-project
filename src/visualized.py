#
# 제작 목적: 시각화
# 제작 날짜: 2026-03-31
# 제작자: AI 3기 3반 박진
#

from __future__ import annotations

import json
import os
from pathlib import Path
import re

PLOT_CACHE_DIR = Path("outputs/.cache")
MATPLOTLIB_CACHE_DIR = PLOT_CACHE_DIR / "matplotlib"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODEL_OUTPUT_DIR = Path("outputs/model_outputs")
BEST_SUMMARY_PATH = MODEL_OUTPUT_DIR / "best_summary.json"
MODEL_SEARCH_PATH = MODEL_OUTPUT_DIR / "model_search.csv"
LEGACY_PREDICTION_RESULT_PATH = MODEL_OUTPUT_DIR / "model_predictions.csv"
FIGURE_OUTPUT_DIR = Path("outputs/images/visualizing")

ACTUAL_COLUMN_CANDIDATES = ("actual", "y_true", "target", "cycle_life", "label")
PREDICTED_COLUMN_CANDIDATES = ("predicted", "y_pred", "prediction", "pred", "y_hat")
CELL_ID_COLUMN_CANDIDATES = ("cell_id", "id")
MODEL_COLUMN_CANDIDATES = ("model_name", "model", "algorithm", "estimator")
FEATURE_SET_COLUMN_CANDIDATES = ("feature_set", "feature_group", "feature_bundle")
SPLIT_COLUMN_CANDIDATES = ("split", "dataset", "data_split", "partition")
CALIBRATION_COLUMN_CANDIDATES = ("is_calibrated", "calibrated", "use_calibration")
EXPERIMENT_COLUMN_CANDIDATES = ("experiment_label", "experiment_name", "run_name")

KNOWN_SPLITS = {
    "train": "train",
    "valid": "valid",
    "validation": "valid",
    "val": "valid",
    "test": "test",
    "batch1": "batch1",
    "batch2": "batch2",
    "batch3": "batch3",
    "holdout": "holdout",
    "external": "external",
}

FIGSIZE = (8, 6)
LEADERBOARD_FIGSIZE = (14, 10)
SUMMARY_FIGSIZE = (15, 5)
SCATTER_ALPHA = 0.75
HIST_BINS = 30
IMAGE_DPI = 150
TOP_N_LEADERBOARD = 10


def find_matching_column(columns, candidates):
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def slugify(value: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return safe_name.strip("_") or "item"


def normalize_split_name(value: str | None) -> str:
    if value is None:
        return "unknown"
    return KNOWN_SPLITS.get(str(value).strip().lower(), str(value).strip().lower() or "unknown")


def normalize_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "calibrated"}


def build_experiment_label(model_name: str | None, feature_set: str | None) -> str:
    labels: list[str] = []
    for value in (model_name, feature_set):
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in labels:
            labels.append(text)
    return " | ".join(labels) if labels else "best_model"


def load_best_summary(path: Path = BEST_SUMMARY_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def infer_prediction_file_metadata(path: Path) -> dict:
    suffix = path.stem.removeprefix("predictions_")
    is_calibrated = suffix.endswith("_calibrated")
    base_name = suffix[: -len("_calibrated")] if is_calibrated else suffix
    tokens = [token for token in base_name.split("_") if token]

    split = "unknown"
    model_token = ""
    matched_idx = None
    for idx, token in enumerate(tokens):
        if token.lower() in KNOWN_SPLITS:
            matched_idx = idx
            split = normalize_split_name(token)
            break

    if matched_idx is None:
        split = normalize_split_name(base_name)
    else:
        model_tokens = tokens[:matched_idx] + tokens[matched_idx + 1 :]
        model_token = "_".join(model_tokens)

    return {
        "split": split,
        "model_token": model_token,
        "is_calibrated": is_calibrated,
    }


def normalize_prediction_frame(
    prediction_df: pd.DataFrame,
    *,
    source_name: str,
    default_model_name: str | None = None,
    default_feature_set: str | None = None,
    default_split: str | None = None,
    default_is_calibrated: bool = False,
) -> pd.DataFrame:
    actual_col = find_matching_column(prediction_df.columns, ACTUAL_COLUMN_CANDIDATES)
    predicted_col = find_matching_column(prediction_df.columns, PREDICTED_COLUMN_CANDIDATES)
    if actual_col is None or predicted_col is None:
        raise KeyError(
            f"{source_name} 파일에서 실제값/예측값 컬럼을 찾지 못했습니다. "
            f"지원 컬럼: actual={ACTUAL_COLUMN_CANDIDATES}, predicted={PREDICTED_COLUMN_CANDIDATES}"
        )

    cell_id_col = find_matching_column(prediction_df.columns, CELL_ID_COLUMN_CANDIDATES)
    model_col = find_matching_column(prediction_df.columns, MODEL_COLUMN_CANDIDATES)
    feature_set_col = find_matching_column(prediction_df.columns, FEATURE_SET_COLUMN_CANDIDATES)
    split_col = find_matching_column(prediction_df.columns, SPLIT_COLUMN_CANDIDATES)
    calibration_col = find_matching_column(prediction_df.columns, CALIBRATION_COLUMN_CANDIDATES)
    experiment_col = find_matching_column(prediction_df.columns, EXPERIMENT_COLUMN_CANDIDATES)

    normalized = pd.DataFrame(index=prediction_df.index)
    if cell_id_col is not None:
        normalized["cell_id"] = prediction_df[cell_id_col]
    else:
        normalized["cell_id"] = prediction_df.index

    normalized["actual"] = pd.to_numeric(prediction_df[actual_col], errors="coerce")
    normalized["predicted"] = pd.to_numeric(prediction_df[predicted_col], errors="coerce")

    if model_col is not None:
        normalized["model_name"] = prediction_df[model_col].astype(str)
    else:
        normalized["model_name"] = default_model_name or default_feature_set or "best_model"

    if feature_set_col is not None:
        normalized["feature_set"] = prediction_df[feature_set_col].astype(str)
    else:
        normalized["feature_set"] = default_feature_set or ""

    if split_col is not None:
        normalized["split"] = prediction_df[split_col].map(normalize_split_name)
    else:
        normalized["split"] = normalize_split_name(default_split)

    if calibration_col is not None:
        normalized["is_calibrated"] = prediction_df[calibration_col].map(normalize_bool)
    else:
        normalized["is_calibrated"] = default_is_calibrated

    if experiment_col is not None:
        normalized["experiment_label"] = prediction_df[experiment_col].astype(str)
    else:
        normalized["experiment_label"] = [
            build_experiment_label(model_name, feature_set)
            for model_name, feature_set in zip(normalized["model_name"], normalized["feature_set"])
        ]

    normalized["prediction_variant"] = normalized["is_calibrated"].map(
        lambda value: "calibrated" if value else "raw"
    )
    normalized["residual"] = normalized["actual"] - normalized["predicted"]
    normalized["absolute_error"] = normalized["residual"].abs()
    normalized["absolute_percentage_error"] = (
        normalized["absolute_error"] / normalized["actual"].abs().replace(0, pd.NA) * 100.0
    )
    normalized["source_file"] = source_name
    normalized = normalized.dropna(subset=["actual", "predicted"]).reset_index(drop=True)
    return normalized


def load_prediction_results(
    prediction_result_path: Path | None = None,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    summary_path: Path = BEST_SUMMARY_PATH,
) -> pd.DataFrame:
    summary = load_best_summary(summary_path)
    best_metrics = summary.get("best_metrics", {})
    default_feature_set = summary.get("best_feature_set") or best_metrics.get("feature_set") or ""
    default_model_name = (
        best_metrics.get("model_name")
        or summary.get("best_model_name")
        or default_feature_set
        or "best_model"
    )

    if prediction_result_path is not None:
        frame = pd.read_csv(prediction_result_path)
        return normalize_prediction_frame(
            frame,
            source_name=prediction_result_path.name,
            default_model_name=default_model_name,
            default_feature_set=default_feature_set,
        )

    normalized_frames: list[pd.DataFrame] = []

    if LEGACY_PREDICTION_RESULT_PATH.exists():
        legacy_df = pd.read_csv(LEGACY_PREDICTION_RESULT_PATH)
        normalized_frames.append(
            normalize_prediction_frame(
                legacy_df,
                source_name=LEGACY_PREDICTION_RESULT_PATH.name,
                default_model_name=default_model_name,
                default_feature_set=default_feature_set,
            )
        )

    prediction_files = sorted(model_output_dir.glob("predictions_*.csv"))
    for csv_path in prediction_files:
        metadata = infer_prediction_file_metadata(csv_path)
        prediction_df = pd.read_csv(csv_path)
        normalized_frames.append(
            normalize_prediction_frame(
                prediction_df,
                source_name=csv_path.name,
                default_model_name=metadata["model_token"] or default_model_name,
                default_feature_set=default_feature_set,
                default_split=metadata["split"],
                default_is_calibrated=metadata["is_calibrated"],
            )
        )

    if not normalized_frames:
        raise FileNotFoundError(
            f"시각화용 예측 산출물을 찾지 못했습니다. "
            f"확인 경로: {model_output_dir / 'predictions_*.csv'} 또는 {LEGACY_PREDICTION_RESULT_PATH}"
        )

    combined = pd.concat(normalized_frames, ignore_index=True)
    combined = combined.sort_values(
        ["experiment_label", "split", "is_calibrated", "cell_id"],
        ignore_index=True,
    )
    return combined


def normalize_model_search_results(path: Path = MODEL_SEARCH_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    model_search = pd.read_csv(path)
    experiment_col = find_matching_column(model_search.columns, EXPERIMENT_COLUMN_CANDIDATES)
    model_col = find_matching_column(model_search.columns, MODEL_COLUMN_CANDIDATES)
    feature_set_col = find_matching_column(model_search.columns, FEATURE_SET_COLUMN_CANDIDATES)

    if experiment_col is not None:
        model_search["experiment_label"] = model_search[experiment_col].astype(str)
    else:
        model_names = model_search[model_col].astype(str) if model_col is not None else [""] * len(model_search)
        feature_sets = (
            model_search[feature_set_col].astype(str) if feature_set_col is not None else [""] * len(model_search)
        )
        model_search["experiment_label"] = [
            build_experiment_label(model_name, feature_set)
            for model_name, feature_set in zip(model_names, feature_sets)
        ]

    return model_search


def summarize_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["experiment_label", "model_name", "feature_set", "split", "prediction_variant"]
    for keys, group in prediction_df.groupby(group_cols, dropna=False):
        experiment_label, model_name, feature_set, split, prediction_variant = keys
        ape = group["absolute_percentage_error"].dropna()
        rows.append(
            {
                "experiment_label": experiment_label,
                "model_name": model_name,
                "feature_set": feature_set,
                "split": split,
                "prediction_variant": prediction_variant,
                "n_samples": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float((group["residual"] ** 2).mean() ** 0.5),
                "mape": float(ape.mean()) if not ape.empty else pd.NA,
                "bias": float(group["residual"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["experiment_label", "split", "prediction_variant"],
        ignore_index=True,
    )


def plot_actual_vs_predicted(
    model_df: pd.DataFrame,
    *,
    experiment_label: str,
    split_name: str,
    prediction_variant: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / (
        f"{slugify(experiment_label)}_{slugify(split_name)}_{slugify(prediction_variant)}_actual_vs_predicted.png"
    )

    plt.figure(figsize=FIGSIZE)
    plt.scatter(model_df["actual"], model_df["predicted"], alpha=SCATTER_ALPHA)
    min_value = min(model_df["actual"].min(), model_df["predicted"].min())
    max_value = max(model_df["actual"].max(), model_df["predicted"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1.2)
    plt.title(f"{experiment_label} | {split_name} ({prediction_variant})")
    plt.xlabel("Actual cycle life")
    plt.ylabel("Predicted cycle life")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def plot_residuals(
    model_df: pd.DataFrame,
    *,
    experiment_label: str,
    split_name: str,
    prediction_variant: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / (
        f"{slugify(experiment_label)}_{slugify(split_name)}_{slugify(prediction_variant)}_residual_plot.png"
    )

    plt.figure(figsize=FIGSIZE)
    plt.scatter(model_df["predicted"], model_df["residual"], alpha=SCATTER_ALPHA)
    plt.axhline(y=0, linestyle="--", color="black", linewidth=1.0)
    plt.title(f"{experiment_label} | {split_name} ({prediction_variant}) Residuals")
    plt.xlabel("Predicted cycle life")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def plot_error_distribution(
    model_df: pd.DataFrame,
    *,
    experiment_label: str,
    split_name: str,
    prediction_variant: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / (
        f"{slugify(experiment_label)}_{slugify(split_name)}_{slugify(prediction_variant)}_error_distribution.png"
    )

    plt.figure(figsize=FIGSIZE)
    plt.hist(model_df["residual"], bins=HIST_BINS, alpha=SCATTER_ALPHA)
    plt.axvline(x=0, linestyle="--", color="black", linewidth=1.0)
    plt.title(f"{experiment_label} | {split_name} ({prediction_variant}) Error Distribution")
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def plot_prediction_metric_summary(summary_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    metrics = ["mae", "rmse", "mape"]
    labels = [
        f"{row.experiment_label}\n{row.split} | {row.prediction_variant}"
        for row in summary_df.itertuples(index=False)
    ]
    x_positions = list(range(len(summary_df)))
    figure, axes = plt.subplots(1, len(metrics), figsize=SUMMARY_FIGSIZE)

    for axis, metric in zip(axes, metrics):
        values = pd.to_numeric(summary_df[metric], errors="coerce").fillna(0.0)
        axis.bar(x_positions, values)
        axis.set_title(metric.upper())
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=55, ha="right")
        axis.set_ylabel(metric.upper())

    figure.tight_layout()
    output_path = output_dir / "prediction_metric_summary.png"
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def plot_model_leaderboard(model_search_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if model_search_df.empty:
        return None

    candidate_metrics = [
        "robust_score",
        "calibrated_test_mape",
        "calibrated_batch3_mape",
        "valid_mape",
    ]
    available_metrics = [metric for metric in candidate_metrics if metric in model_search_df.columns]
    if not available_metrics:
        return None

    n_plots = len(available_metrics)
    n_rows = 2 if n_plots > 2 else 1
    n_cols = 2 if n_plots > 1 else 1
    figure, axes = plt.subplots(n_rows, n_cols, figsize=LEADERBOARD_FIGSIZE)
    axes = pd.Series(axes.flatten() if hasattr(axes, "flatten") else [axes])

    for axis, metric in zip(axes, available_metrics):
        ranked = model_search_df.sort_values(metric, ascending=True).head(TOP_N_LEADERBOARD)
        labels = list(ranked["experiment_label"])
        values = list(ranked[metric])
        positions = list(range(len(ranked)))
        axis.barh(positions, values)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_title(f"Top {len(ranked)} by {metric}")
        axis.set_xlabel(metric)

    for axis in axes[len(available_metrics) :]:
        axis.axis("off")

    figure.tight_layout()
    output_path = output_dir / "model_leaderboard.png"
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def create_all_visualizations(
    prediction_df: pd.DataFrame,
    model_search_df: pd.DataFrame,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    for stale_file in output_dir.glob("*_train_*.png"):
        stale_file.unlink(missing_ok=True)
    plot_df = prediction_df.loc[prediction_df["split"] != "train"].copy()
    if plot_df.empty:
        plot_df = prediction_df.copy()

    for keys, model_df in plot_df.groupby(
        ["experiment_label", "split", "prediction_variant"],
        dropna=False,
    ):
        experiment_label, split_name, prediction_variant = keys
        generated_files.append(
            plot_actual_vs_predicted(
                model_df,
                experiment_label=experiment_label,
                split_name=split_name,
                prediction_variant=prediction_variant,
                output_dir=output_dir,
            )
        )
        generated_files.append(
            plot_residuals(
                model_df,
                experiment_label=experiment_label,
                split_name=split_name,
                prediction_variant=prediction_variant,
                output_dir=output_dir,
            )
        )
        generated_files.append(
            plot_error_distribution(
                model_df,
                experiment_label=experiment_label,
                split_name=split_name,
                prediction_variant=prediction_variant,
                output_dir=output_dir,
            )
        )

    prediction_summary = summarize_predictions(plot_df)
    prediction_df.to_csv(output_dir / "prediction_records_normalized.csv", index=False)
    prediction_summary.to_csv(output_dir / "prediction_metrics_by_split.csv", index=False)

    summary_figure = plot_prediction_metric_summary(prediction_summary, output_dir)
    if summary_figure is not None:
        generated_files.append(summary_figure)

    if not model_search_df.empty:
        model_search_df.to_csv(output_dir / "model_search_normalized.csv", index=False)
        leaderboard_path = plot_model_leaderboard(model_search_df, output_dir)
        if leaderboard_path is not None:
            generated_files.append(leaderboard_path)

    manifest = {
        "n_prediction_rows": int(len(prediction_df)),
        "n_plot_rows": int(len(plot_df)),
        "n_generated_figures": int(len(generated_files)),
        "generated_files": [str(path) for path in generated_files],
        "experiments": sorted(plot_df["experiment_label"].dropna().unique().tolist()),
        "splits": sorted(plot_df["split"].dropna().unique().tolist()),
        "prediction_variants": sorted(plot_df["prediction_variant"].dropna().unique().tolist()),
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    return generated_files


def run_visualization_pipeline(
    prediction_result_path: Path | None = None,
    model_output_dir: Path = MODEL_OUTPUT_DIR,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> list[Path]:
    prediction_df = load_prediction_results(
        prediction_result_path=prediction_result_path,
        model_output_dir=model_output_dir,
    )
    model_search_df = normalize_model_search_results(model_output_dir / "model_search.csv")
    return create_all_visualizations(
        prediction_df=prediction_df,
        model_search_df=model_search_df,
        output_dir=output_dir,
    )


def main() -> None:
    generated_files = run_visualization_pipeline(output_dir=FIGURE_OUTPUT_DIR)
    print(f"[visualized] generated {len(generated_files)} files")
    for path in generated_files:
        print(f"[visualized] {path}")


if __name__ == "__main__":
    main()

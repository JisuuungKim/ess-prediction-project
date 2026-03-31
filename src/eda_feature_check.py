from __future__ import annotations

import json
import math
import os
from pathlib import Path

PLOT_CACHE_DIR = Path("outputs/.cache")
MATPLOTLIB_CACHE_DIR = PLOT_CACHE_DIR / "matplotlib"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import DEFAULT_FEATURE_CACHE_DIR, DEFAULT_OUTPUT_DIR
from features import load_feature_tables


FEATURE_CHECK_IMAGE_DIR = Path("outputs/images/eda_feature_check")
FEATURE_CHECK_TABLE_DIR = Path("outputs/table/eda_feature_check")
BEST_SUMMARY_PATH = Path(DEFAULT_OUTPUT_DIR) / "best_summary.json"
SELECTED_FEATURES_PATH = Path(DEFAULT_OUTPUT_DIR) / "selected_features.csv"

TARGET_COLUMN = "cycle_life"
BATCH_COLUMN = "batch"
CELL_ID_COLUMN = "cell_id"
NON_FEATURE_COLUMNS = {CELL_ID_COLUMN, TARGET_COLUMN, BATCH_COLUMN, "charging_policy"}

BATCH_COLORS = {
    "batch1": "#1f77b4",
    "batch2": "#ff7f0e",
    "batch3": "#2ca02c",
}

FIGSIZE = (8, 6)
GRID_FIGSIZE = (16, 10)
IMAGE_DPI = 150
TARGET_HIST_BINS = 20


def load_best_summary(path: Path = BEST_SUMMARY_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_requested_feature_set(
    requested_feature_set: str | None,
    available_columns: set[str],
    best_summary: dict,
) -> tuple[str, list[str], str]:
    all_feature_sets = best_summary.get("all_feature_sets", {})
    best_feature_set = best_summary.get("best_feature_set")

    if requested_feature_set:
        if requested_feature_set in all_feature_sets:
            features = all_feature_sets[requested_feature_set]
            source = "best_summary.all_feature_sets"
            label = requested_feature_set
        else:
            features = [feature.strip() for feature in requested_feature_set.split(",") if feature.strip()]
            source = "manual_input"
            label = "custom_feature_list"
    elif best_feature_set and best_feature_set in all_feature_sets:
        features = all_feature_sets[best_feature_set]
        source = "best_summary.best_feature_set"
        label = best_feature_set
    elif SELECTED_FEATURES_PATH.exists():
        selected_df = pd.read_csv(SELECTED_FEATURES_PATH)
        features = selected_df["feature"].dropna().astype(str).tolist()
        source = "selected_features.csv"
        label = "selected_features"
    else:
        features = sorted(available_columns - NON_FEATURE_COLUMNS)
        source = "all_numeric_feature_columns"
        label = "all_available_features"

    filtered = [feature for feature in features if feature in available_columns]
    if not filtered:
        raise ValueError("EDA 체크에 사용할 feature를 찾지 못했습니다.")

    return label, filtered, source


def build_feature_report(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        series = feature_df[feature]
        overall = feature_df[[feature, TARGET_COLUMN]].dropna()
        batch_corrs = {}
        for batch_name, batch_df in feature_df.groupby(BATCH_COLUMN):
            pair = batch_df[[feature, TARGET_COLUMN]].dropna()
            batch_corrs[f"{batch_name}_corr"] = pair[feature].corr(pair[TARGET_COLUMN]) if len(pair) >= 3 else np.nan

        rows.append(
            {
                "feature": feature,
                "missing_rate": float(series.isna().mean()),
                "n_valid": int(series.notna().sum()),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "std": float(series.std()) if series.notna().any() else np.nan,
                "min": float(series.min()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
                "overall_corr_with_cycle_life": (
                    overall[feature].corr(overall[TARGET_COLUMN]) if len(overall) >= 3 else np.nan
                ),
                **batch_corrs,
            }
        )

    report = pd.DataFrame(rows)
    report["abs_overall_corr"] = report["overall_corr_with_cycle_life"].abs()
    return report.sort_values("abs_overall_corr", ascending=False, ignore_index=True)


def plot_target_distribution(feature_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "target_distribution_by_batch.png"
    plt.figure(figsize=FIGSIZE)
    for batch_name, batch_df in feature_df.groupby(BATCH_COLUMN):
        color = BATCH_COLORS.get(batch_name)
        plt.hist(
            batch_df[TARGET_COLUMN].dropna(),
            bins=TARGET_HIST_BINS,
            alpha=0.45,
            label=batch_name,
            color=color,
        )
    plt.title("Cycle Life Distribution by Batch")
    plt.xlabel("Cycle Life")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def plot_feature_correlation_bar(feature_report: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "selected_feature_correlation_rank.png"
    plt.figure(figsize=(10, max(4, len(feature_report) * 0.5)))
    ordered = feature_report.sort_values("abs_overall_corr", ascending=True)
    plt.barh(ordered["feature"], ordered["overall_corr_with_cycle_life"])
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Selected Feature Correlation with Cycle Life")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def plot_correlation_heatmap(feature_df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> Path:
    output_path = output_dir / "selected_feature_heatmap.png"
    corr_columns = feature_cols + [TARGET_COLUMN]
    corr = feature_df[corr_columns].corr(numeric_only=True)

    figure, axis = plt.subplots(figsize=(max(8, len(corr_columns) * 0.7), max(6, len(corr_columns) * 0.7)))
    image = axis.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(corr_columns)))
    axis.set_yticks(range(len(corr_columns)))
    axis.set_xticklabels(corr_columns, rotation=60, ha="right")
    axis.set_yticklabels(corr_columns)
    axis.set_title("Correlation Heatmap for Selected Features")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def build_feature_axes(n_features: int):
    n_cols = min(3, max(1, n_features))
    n_rows = math.ceil(n_features / n_cols)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(GRID_FIGSIZE[0], max(6, n_rows * 4)))
    axes = np.atleast_1d(axes).flatten()
    return figure, axes


def plot_feature_scatter_grid(feature_df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> Path:
    output_path = output_dir / "selected_feature_scatter_grid.png"
    figure, axes = build_feature_axes(len(feature_cols))

    for axis, feature in zip(axes, feature_cols):
        for batch_name, batch_df in feature_df.groupby(BATCH_COLUMN):
            color = BATCH_COLORS.get(batch_name)
            axis.scatter(
                batch_df[feature],
                batch_df[TARGET_COLUMN],
                alpha=0.65,
                s=25,
                label=batch_name,
                color=color,
            )
        pair = feature_df[[feature, TARGET_COLUMN]].dropna()
        corr_value = pair[feature].corr(pair[TARGET_COLUMN]) if len(pair) >= 3 else np.nan
        axis.set_title(f"{feature}\nr={corr_value:.3f}" if pd.notna(corr_value) else feature)
        axis.set_xlabel(feature)
        axis.set_ylabel(TARGET_COLUMN)

    for axis in axes[len(feature_cols) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def plot_feature_boxplot_grid(feature_df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> Path:
    output_path = output_dir / "selected_feature_boxplot_grid.png"
    figure, axes = build_feature_axes(len(feature_cols))
    batch_order = list(dict.fromkeys(feature_df[BATCH_COLUMN].dropna().tolist()))

    for axis, feature in zip(axes, feature_cols):
        values = [
            feature_df.loc[feature_df[BATCH_COLUMN] == batch_name, feature].dropna().to_numpy()
            for batch_name in batch_order
        ]
        axis.boxplot(values, labels=batch_order, patch_artist=True)
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=30)

    for axis in axes[len(feature_cols) :]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def run_eda_feature_check_pipeline(
    feature_cache_dir: Path = DEFAULT_FEATURE_CACHE_DIR,
    model_output_dir: Path = DEFAULT_OUTPUT_DIR,
    requested_feature_set: str | None = None,
    image_output_dir: Path = FEATURE_CHECK_IMAGE_DIR,
    table_output_dir: Path = FEATURE_CHECK_TABLE_DIR,
) -> dict:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)

    feature_tables, combined = load_feature_tables(feature_cache_dir)
    best_summary = load_best_summary(Path(model_output_dir) / "best_summary.json")

    feature_set_label, feature_cols, feature_source = parse_requested_feature_set(
        requested_feature_set=requested_feature_set,
        available_columns=set(combined.columns),
        best_summary=best_summary,
    )

    selected_columns = [CELL_ID_COLUMN, BATCH_COLUMN, TARGET_COLUMN, *feature_cols]
    selected_df = combined[selected_columns].copy()
    selected_df.to_csv(table_output_dir / "selected_feature_dataset.csv", index=False)

    feature_report = build_feature_report(selected_df, feature_cols)
    feature_report.to_csv(table_output_dir / "selected_feature_report.csv", index=False)

    batch_summary = (
        selected_df.groupby(BATCH_COLUMN)[TARGET_COLUMN]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    batch_summary.to_csv(table_output_dir / "batch_target_summary.csv", index=False)

    generated_files = [
        plot_target_distribution(selected_df, image_output_dir),
        plot_feature_correlation_bar(feature_report, image_output_dir),
        plot_correlation_heatmap(selected_df, feature_cols, image_output_dir),
        plot_feature_scatter_grid(selected_df, feature_cols, image_output_dir),
        plot_feature_boxplot_grid(selected_df, feature_cols, image_output_dir),
    ]

    manifest = {
        "feature_set_label": feature_set_label,
        "feature_source": feature_source,
        "requested_feature_set": requested_feature_set,
        "selected_features": feature_cols,
        "n_selected_features": len(feature_cols),
        "n_rows": int(len(selected_df)),
        "available_batches": sorted(feature_tables.keys()),
        "generated_files": [str(path) for path in generated_files],
    }
    (table_output_dir / "eda_feature_check_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    return {
        "feature_set_label": feature_set_label,
        "feature_cols": feature_cols,
        "feature_source": feature_source,
        "selected_df": selected_df,
        "feature_report": feature_report,
        "batch_summary": batch_summary,
        "generated_files": generated_files,
    }


def main(requested_feature_set: str | None = None) -> None:
    result = run_eda_feature_check_pipeline(requested_feature_set=requested_feature_set)
    print(f"[eda_feature_check] feature_set={result['feature_set_label']}")
    print(f"[eda_feature_check] source={result['feature_source']}")
    print(f"[eda_feature_check] n_features={len(result['feature_cols'])}")
    for feature in result["feature_cols"]:
        print(f"[eda_feature_check] feature: {feature}")
    for path in result["generated_files"]:
        print(f"[eda_feature_check] {path}")


if __name__ == "__main__":
    main()

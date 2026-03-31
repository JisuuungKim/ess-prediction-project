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
from modeling import (
    build_batch_stability_report,
    build_eda_feature_sets,
    build_paper_feature_sets,
    select_feature_blocks,
)


FEATURE_CHECK_IMAGE_DIR = Path("outputs/images/eda_feature_check")
FEATURE_CHECK_TABLE_DIR = Path("outputs/table/eda_feature_check")
BEST_SUMMARY_PATH = Path(DEFAULT_OUTPUT_DIR) / "best_summary.json"
SELECTED_FEATURES_PATH = Path(DEFAULT_OUTPUT_DIR) / "selected_features.csv"
LAST_SELECTION_PATH = FEATURE_CHECK_TABLE_DIR / "last_feature_selection.json"

TARGET_COLUMN = "cycle_life"
BATCH_COLUMN = "batch"
CELL_ID_COLUMN = "cell_id"
LIFE_BAND_COLUMN = "cycle_life_band"
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


def build_feature_set_catalog(feature_tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    batch1 = feature_tables.get("batch1")
    if batch1 is None or batch1.empty:
        return {}

    train_df = batch1.dropna(subset=[TARGET_COLUMN]).copy()
    selected_blocks, *_ = select_feature_blocks(train_df)
    paper_feature_sets = build_paper_feature_sets(batch1)
    eda_feature_sets = build_eda_feature_sets(selected_blocks)
    feature_sets = {**paper_feature_sets, **eda_feature_sets}

    stability_candidates = sorted({feature for features in feature_sets.values() for feature in features})
    if stability_candidates:
        stability_report = build_batch_stability_report(feature_tables, stability_candidates)
        stable_features = set(stability_report.loc[stability_report["stable_candidate"], "feature"].tolist())
        strict_features = set(
            stability_report.loc[
                (stability_report["batch1_corr"].abs() >= 0.30)
                & (stability_report["batch2_corr"].abs() >= 0.30)
                & (stability_report["batch3_corr"].abs() >= 0.30)
                & (stability_report["mean_abs_corr"] >= 0.35),
                "feature",
            ].tolist()
        )

        stable_feature_sets = {}
        strict_feature_sets = {}
        for name, features in feature_sets.items():
            stable_only = [feature for feature in features if feature in stable_features]
            strict_only = [feature for feature in features if feature in strict_features]
            if stable_only:
                stable_feature_sets[f"{name}_stable"] = stable_only
            if strict_only:
                strict_feature_sets[f"{name}_stable_strict"] = strict_only

        feature_sets.update(stable_feature_sets)
        feature_sets.update(strict_feature_sets)

    available_columns = set(pd.concat(feature_tables.values(), ignore_index=True).columns)
    catalog = {}
    for name, features in feature_sets.items():
        filtered = [feature for feature in features if feature in available_columns]
        if filtered:
            catalog[name] = filtered
    return catalog


def get_available_feature_set_names(
    feature_cache_dir: Path = DEFAULT_FEATURE_CACHE_DIR,
    model_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    names: set[str] = set()
    best_summary = load_best_summary(Path(model_output_dir) / "best_summary.json")
    names.update(best_summary.get("all_feature_sets", {}).keys())

    model_search_path = Path(model_output_dir) / "model_search.csv"
    if model_search_path.exists():
        model_search = pd.read_csv(model_search_path)
        if "feature_set" in model_search.columns:
            names.update(model_search["feature_set"].dropna().astype(str).tolist())

    try:
        feature_tables, _ = load_feature_tables(feature_cache_dir)
        names.update(build_feature_set_catalog(feature_tables).keys())
    except FileNotFoundError:
        pass

    return sorted(name for name in names if name)


def load_last_feature_selection(path: Path = LAST_SELECTION_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_last_feature_selection(selection: dict, path: Path = LAST_SELECTION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False))


def parse_requested_feature_set(
    requested_feature_set: str | None,
    available_columns: set[str],
    best_summary: dict,
    feature_catalog: dict[str, list[str]],
    last_selection: dict | None = None,
) -> tuple[str, list[str], str]:
    all_feature_sets = best_summary.get("all_feature_sets", {})
    best_feature_set = best_summary.get("best_feature_set")

    if requested_feature_set:
        if requested_feature_set in feature_catalog:
            features = feature_catalog[requested_feature_set]
            source = "feature_catalog"
            label = requested_feature_set
        elif requested_feature_set in all_feature_sets:
            features = all_feature_sets[requested_feature_set]
            source = "best_summary.all_feature_sets"
            label = requested_feature_set
        else:
            features = [feature.strip() for feature in requested_feature_set.split(",") if feature.strip()]
            source = "manual_input"
            label = "custom_feature_list"
    elif last_selection and last_selection.get("features"):
        features = last_selection["features"]
        source = "last_selected_features"
        label = last_selection.get("label") or "last_selected_features"
    elif best_feature_set and best_feature_set in feature_catalog:
        features = feature_catalog[best_feature_set]
        source = "feature_catalog.best_feature_set"
        label = best_feature_set
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
        available_names = sorted(feature_catalog.keys())
        if requested_feature_set:
            raise ValueError(
                "요청한 feature set을 현재 feature cache 기준으로 구성할 수 없습니다. "
                f"requested={requested_feature_set}, available={available_names}"
            )
        raise ValueError("EDA 체크에 사용할 feature를 찾지 못했습니다.")

    return label, filtered, source


def add_cycle_life_bands(feature_df: pd.DataFrame) -> pd.DataFrame:
    banded = feature_df.copy()
    valid_target = banded[TARGET_COLUMN].dropna()
    if valid_target.nunique() < 2:
        banded[LIFE_BAND_COLUMN] = "all"
        return banded

    quantile_count = min(4, valid_target.nunique())
    labels = ["low", "mid_low", "mid_high", "high"][:quantile_count]

    try:
        banded[LIFE_BAND_COLUMN] = pd.qcut(
            banded[TARGET_COLUMN],
            q=quantile_count,
            labels=labels,
            duplicates="drop",
        )
    except ValueError:
        banded[LIFE_BAND_COLUMN] = "all"

    banded[LIFE_BAND_COLUMN] = banded[LIFE_BAND_COLUMN].astype(str).replace("nan", "all")
    return banded


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
    output_path = output_dir / "selected_feature_cycle_scatter_grid.png"
    figure, axes = build_feature_axes(len(feature_cols))

    for axis, feature in zip(axes, feature_cols):
        for batch_name, batch_df in feature_df.groupby(BATCH_COLUMN):
            color = BATCH_COLORS.get(batch_name)
            axis.scatter(
                batch_df[TARGET_COLUMN],
                batch_df[feature],
                alpha=0.65,
                s=25,
                label=batch_name,
                color=color,
            )
        pair = feature_df[[feature, TARGET_COLUMN]].dropna()
        corr_value = pair[feature].corr(pair[TARGET_COLUMN]) if len(pair) >= 3 else np.nan
        if not pair.empty:
            sorted_pair = pair.sort_values(TARGET_COLUMN)
            window = max(5, len(sorted_pair) // 8)
            smooth = (
                sorted_pair[feature]
                .rolling(window=window, min_periods=max(3, window // 2))
                .median()
            )
            axis.plot(sorted_pair[TARGET_COLUMN], smooth, color="black", linewidth=1.4)
        axis.set_title(f"{feature}\nr={corr_value:.3f}" if pd.notna(corr_value) else feature)
        axis.set_xlabel(TARGET_COLUMN)
        axis.set_ylabel(feature)

    for axis in axes[len(feature_cols) :]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(figure)
    return output_path


def plot_cycle_band_boxplot_grid(feature_df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> Path:
    output_path = output_dir / "selected_feature_cycle_band_boxplot_grid.png"
    figure, axes = build_feature_axes(len(feature_cols))
    band_order = ["low", "mid_low", "mid_high", "high", "all"]
    band_order = [band for band in band_order if band in set(feature_df[LIFE_BAND_COLUMN])]

    for axis, feature in zip(axes, feature_cols):
        values = [
            feature_df.loc[feature_df[LIFE_BAND_COLUMN] == band_name, feature].dropna().to_numpy()
            for band_name in band_order
        ]
        axis.boxplot(values, labels=band_order, patch_artist=True)
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
    for stale_name in (
        "selected_feature_scatter_grid.png",
        "selected_feature_boxplot_grid.png",
    ):
        (image_output_dir / stale_name).unlink(missing_ok=True)

    feature_tables, combined = load_feature_tables(feature_cache_dir)
    best_summary = load_best_summary(Path(model_output_dir) / "best_summary.json")
    feature_catalog = build_feature_set_catalog(feature_tables)
    last_selection = load_last_feature_selection()

    feature_set_label, feature_cols, feature_source = parse_requested_feature_set(
        requested_feature_set=requested_feature_set,
        available_columns=set(combined.columns),
        best_summary=best_summary,
        feature_catalog=feature_catalog,
        last_selection=last_selection,
    )

    selected_columns = [CELL_ID_COLUMN, BATCH_COLUMN, TARGET_COLUMN, *feature_cols]
    selected_df = add_cycle_life_bands(combined[selected_columns].copy())
    selected_df.to_csv(table_output_dir / "selected_feature_dataset.csv", index=False)

    feature_report = build_feature_report(selected_df, feature_cols)
    feature_report.to_csv(table_output_dir / "selected_feature_report.csv", index=False)

    batch_summary = (
        selected_df.groupby(BATCH_COLUMN)[TARGET_COLUMN]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    batch_summary.to_csv(table_output_dir / "batch_target_summary.csv", index=False)

    cycle_band_summary = (
        selected_df.groupby(LIFE_BAND_COLUMN)[TARGET_COLUMN]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    cycle_band_summary.to_csv(table_output_dir / "cycle_life_band_summary.csv", index=False)

    generated_files = [
        plot_target_distribution(selected_df, image_output_dir),
        plot_feature_correlation_bar(feature_report, image_output_dir),
        plot_correlation_heatmap(selected_df, feature_cols, image_output_dir),
        plot_feature_scatter_grid(selected_df, feature_cols, image_output_dir),
        plot_cycle_band_boxplot_grid(selected_df, feature_cols, image_output_dir),
    ]

    save_last_feature_selection(
        {
            "label": feature_set_label,
            "requested_feature_set": requested_feature_set,
            "features": feature_cols,
        }
    )

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
        "cycle_band_summary": cycle_band_summary,
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

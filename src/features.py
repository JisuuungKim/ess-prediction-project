from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import DEFAULT_DATA_DIR, DEFAULT_FEATURE_CACHE_DIR, DEFAULT_FILES
from data_loader import get_cell_cycles, load_batches, to_float_array


EARLY_CYCLE_LIMIT = 100
FINAL_FEATURE_COLUMNS = [
    "mean_chargetime",
    "temp_rise_100",
    "baseline_QD",
    "baseline_QC",
    "mean_c_rate",
    "max_c_rate",
    "policy_steps",
    "mean_Tavg",
    "delta_q_std",
    "delta_q_highV_mean",
    "delta_q_max",
]


def find_knee_point(cycles, qd, min_cycle=80, window=25, acceleration_factor=2.0):
    cycles = to_float_array(cycles)
    qd = to_float_array(qd)
    if len(cycles) < max(window, min_cycle):
        return np.nan, np.nan, np.nan

    smooth = (
        pd.Series(qd)
        .rolling(window=window, center=True, min_periods=max(5, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy()
    )
    slope = np.gradient(smooth, cycles)
    early_mask = (cycles >= 10) & (cycles <= min(100, cycles.max()))
    baseline = np.nanmedian(slope[early_mask]) if early_mask.any() else np.nanmedian(slope[:window])
    if np.isnan(baseline) or baseline >= 0:
        return np.nan, baseline, np.nan

    accelerated = np.where((cycles >= min_cycle) & (slope <= acceleration_factor * baseline))[0]
    if len(accelerated) == 0:
        return np.nan, baseline, np.nan

    knee_idx = accelerated[0]
    post = np.nanmedian(slope[knee_idx : min(len(slope), knee_idx + window)])
    return int(cycles[knee_idx]), float(baseline), float(post)


def build_knee_summary(frame: pd.DataFrame, max_cycle=EARLY_CYCLE_LIMIT):
    rows = []
    for cell_id, sub in frame.groupby("cell_id"):
        sub = sub[sub["cycle"] <= max_cycle].sort_values("cycle")
        if sub.empty:
            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle_life": np.nan,
                    "knee_cycle": np.nan,
                    "baseline_fade_rate": np.nan,
                    "post_knee_fade_rate": np.nan,
                    "fade_acceleration_ratio": np.nan,
                }
            )
            continue
        knee, base, post = find_knee_point(sub["cycle"].to_numpy(), sub["QD"].to_numpy())
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_life": float(sub["cycle_life"].iloc[0]),
                "knee_cycle": knee,
                "baseline_fade_rate": base,
                "post_knee_fade_rate": post,
                "fade_acceleration_ratio": abs(post / base)
                if pd.notna(post) and pd.notna(base) and base != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def get_delta_q_profile(cell, cycle_a=10, cycle_b=100):
    cycles = get_cell_cycles(cell)
    if len(cycles) < cycle_b:
        return pd.DataFrame()

    first = cycles[cycle_a - 1]
    last = cycles[cycle_b - 1]
    if "Qdlin" not in first or "Qdlin" not in last:
        return pd.DataFrame()

    q_a = to_float_array(first["Qdlin"])
    q_b = to_float_array(last["Qdlin"])
    voltage = np.linspace(2.0, 3.6, min(len(q_a), len(q_b)))
    delta_q = q_b[: len(voltage)] - q_a[: len(voltage)]
    mask = np.isfinite(voltage) & np.isfinite(delta_q)
    return pd.DataFrame({"voltage": voltage[mask], "delta_q": delta_q[mask]})


def extract_delta_q_features(profile: pd.DataFrame):
    if profile.empty:
        return {}

    voltage = profile["voltage"].to_numpy()
    delta_q = profile["delta_q"].to_numpy()

    def band_mean(low, high=None):
        mask = voltage >= low
        if high is not None:
            mask &= voltage < high
        return float(np.nanmean(delta_q[mask])) if mask.any() else np.nan

    return {
        "delta_q_mean": float(np.nanmean(delta_q)),
        "delta_q_std": float(np.nanstd(delta_q)),
        "delta_q_log_variance": float(np.log(np.nanvar(delta_q) + 1e-12)),
        "delta_q_min": float(np.nanmin(delta_q)),
        "delta_q_max": float(np.nanmax(delta_q)),
        "delta_q_range": float(np.nanmax(delta_q) - np.nanmin(delta_q)),
        "delta_q_abs_area": float(np.trapz(np.abs(delta_q), voltage)),
        "delta_q_signed_area": float(np.trapz(delta_q, voltage)),
        "delta_q_lowV_mean": band_mean(2.0, 2.7),
        "delta_q_midV_mean": band_mean(2.7, 3.1),
        "delta_q_highV_mean": band_mean(3.1, None),
    }


def build_delta_q_table(batch, cycle_life_df: pd.DataFrame):
    rows = []
    for cell_id in cycle_life_df["cell_id"]:
        profile = get_delta_q_profile(batch[int(cell_id)])
        if profile.empty:
            continue
        row = cycle_life_df.loc[cycle_life_df["cell_id"] == cell_id].iloc[0]
        features = extract_delta_q_features(profile)
        features.update({"cell_id": int(cell_id), "cycle_life": float(row["cycle_life"])})
        rows.append(features)
    return pd.DataFrame(rows)


def build_early_summary_features(df: pd.DataFrame, max_cycle=EARLY_CYCLE_LIMIT):
    early = df[df["cycle"] <= max_cycle].copy().sort_values(["cell_id", "cycle"])
    grouped = early.groupby("cell_id")

    summary = grouped.agg(
        cycle_life=("cycle_life", "first"),
        baseline_QD=("QD", "median"),
        baseline_QC=("QC", "median"),
        mean_QD=("QD", "mean"),
        std_QD=("QD", "std"),
        mean_IR=("IR", "mean"),
        std_IR=("IR", "std"),
        mean_Tavg=("Tavg", "mean"),
        mean_Tmax=("Tmax", "mean"),
        mean_Tmin=("Tmin", "mean"),
        mean_chargetime=("chargetime", "mean"),
    ).reset_index()

    deltas = []
    for cell_id, sub in grouped:
        sub = sub.sort_values("cycle")
        deltas.append(
            {
                "cell_id": cell_id,
                "qd_drop_100": float(sub["QD"].iloc[0] - sub["QD"].iloc[-1]) if len(sub) > 1 else np.nan,
                "ir_rise_100": float(sub["IR"].iloc[-1] - sub["IR"].iloc[0]) if len(sub) > 1 else np.nan,
                "temp_rise_100": float(sub["Tavg"].iloc[-1] - sub["Tavg"].iloc[0]) if len(sub) > 1 else np.nan,
                "chargetime_rise_100": float(sub["chargetime"].iloc[-1] - sub["chargetime"].iloc[0]) if len(sub) > 1 else np.nan,
            }
        )
    deltas = pd.DataFrame(deltas)

    summary = summary.merge(deltas, on="cell_id", how="left")
    summary["qd_cv_100"] = summary["std_QD"] / summary["mean_QD"].abs().replace(0, np.nan)
    return summary


def add_nonlinear_features(feature_table: pd.DataFrame):
    d = feature_table.copy()
    return d.replace([np.inf, -np.inf], np.nan)


def build_feature_table_for_batch(bundle: dict, batch_name: str):
    early = build_early_summary_features(bundle["df"])
    knee_summary = build_knee_summary(bundle["df_clean"], max_cycle=EARLY_CYCLE_LIMIT)
    delta_q = build_delta_q_table(bundle["batch"], bundle["cycle_life_df"])

    feature_table = (
        early.merge(
            bundle["cycle_life_df"][
                ["cell_id", "charging_policy", "max_c_rate", "mean_c_rate", "policy_steps"]
            ],
            on="cell_id",
            how="left",
        )
        .merge(
            knee_summary[
                ["cell_id", "knee_cycle", "baseline_fade_rate", "post_knee_fade_rate", "fade_acceleration_ratio"]
            ],
            on="cell_id",
            how="left",
        )
        .merge(delta_q.drop(columns=["cycle_life"], errors="ignore"), on="cell_id", how="left")
    )
    feature_table = add_nonlinear_features(feature_table)
    feature_table["batch"] = batch_name

    keep_cols = ["cell_id", "cycle_life", "charging_policy", "batch", *FINAL_FEATURE_COLUMNS]
    keep_cols = [col for col in keep_cols if col in feature_table.columns]
    return feature_table[keep_cols].copy()


def build_feature_tables(batches: dict, verbose=True):
    tables = {}
    for batch_name, bundle in batches.items():
        if verbose:
            print(f"[features] {batch_name}: building feature table", flush=True)
        bundle["knee_summary"] = build_knee_summary(bundle["df_clean"], max_cycle=EARLY_CYCLE_LIMIT)
        bundle["delta_q"] = build_delta_q_table(bundle["batch"], bundle["cycle_life_df"])
        tables[batch_name] = build_feature_table_for_batch(bundle, batch_name)
        if verbose:
            print(f"[features] {batch_name}: shape={tables[batch_name].shape}", flush=True)
    combined = pd.concat(tables.values(), ignore_index=True)
    if verbose:
        print(f"[features] combined: shape={combined.shape}", flush=True)
    return tables, combined


def save_feature_tables(feature_tables: dict, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for batch_name, table in feature_tables.items():
        table.to_csv(output_path / f"features_{batch_name}.csv", index=False)
        summary_rows.append(
            {
                "batch": batch_name,
                "n_rows": int(len(table)),
                "n_cols": int(len(table.columns)),
            }
        )

    combined = pd.concat(feature_tables.values(), ignore_index=True)
    combined.to_csv(output_path / "features_all_batches.csv", index=False)
    metadata = {
        "batches": sorted(feature_tables.keys()),
        "summary": summary_rows,
    }
    (output_path / "feature_cache_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    return combined


def load_feature_tables(input_dir, required_batches=None):
    input_path = Path(input_dir)
    batch_names = required_batches or ["batch1", "batch2", "batch3"]

    feature_tables = {}
    for batch_name in batch_names:
        csv_path = input_path / f"features_{batch_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing cached feature table: {csv_path}")
        feature_tables[batch_name] = pd.read_csv(csv_path)

    combined_path = input_path / "features_all_batches.csv"
    combined = pd.read_csv(combined_path) if combined_path.exists() else pd.concat(feature_tables.values(), ignore_index=True)
    return feature_tables, combined


def build_and_save_feature_cache(data_dir=DEFAULT_DATA_DIR, feature_cache_dir=DEFAULT_FEATURE_CACHE_DIR):
    feature_cache_dir = Path(feature_cache_dir)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    batches = load_batches(data_dir, DEFAULT_FILES)
    feature_tables, combined = build_feature_tables(batches)
    save_feature_tables(feature_tables, feature_cache_dir)

    print(f"[features] saved cache to {feature_cache_dir}")
    print(f"[features] combined shape={combined.shape}")
    return feature_tables, combined


def main():
    build_and_save_feature_cache()

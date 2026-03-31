"""Data loading utilities."""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import joblib

import mat73
import numpy as np
import pandas as pd
from scipy import io as sio


def load_mat(path: str | Path):
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return mat73.loadmat(path)
    except Exception:
        return sio.loadmat(path, simplify_cells=True)
    finally:
        logging.disable(previous_disable)


def normalize_batch(batch):
    if isinstance(batch, dict):
        keys = list(batch.keys())
        return [{key: batch[key][i] for key in keys} for i in range(len(batch[keys[0]]))]
    return batch


def to_list_of_dicts(value):
    if isinstance(value, dict):
        keys = list(value.keys())
        return [{key: value[key][i] for key in keys} for i in range(len(value[keys[0]]))]
    return value


def to_float_array(values):
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    return arr


def get_cell_cycles(cell):
    return to_list_of_dicts(cell["cycles"])


def extract_summary(batch):
    records = []
    for cell_id, cell in enumerate(batch):
        summary = cell["summary"]
        if isinstance(summary, dict):
            qd = np.asarray(summary["QDischarge"])
            qc = np.asarray(summary["QCharge"])
            ir = np.asarray(summary["IR"])
            tmax = np.asarray(summary["Tmax"])
            tavg = np.asarray(summary["Tavg"])
            tmin = np.asarray(summary["Tmin"])
            chargetime = np.asarray(summary["chargetime"])
        else:
            qd = summary["QDischarge"]
            qc = summary["QCharge"]
            ir = summary["IR"]
            tmax = summary["Tmax"]
            tavg = summary["Tavg"]
            tmin = summary["Tmin"]
            chargetime = summary["chargetime"]

        cycle_life = cell["cycle_life"]
        if pd.isna(cycle_life):
            continue

        policy = str(cell.get("policy_readable") or cell.get("policy") or "unknown")
        for cycle_idx in range(len(qd)):
            records.append(
                {
                    "cell_id": cell_id,
                    "cycle": cycle_idx + 1,
                    "cycle_life": int(cycle_life),
                    "charging_policy": policy,
                    "QD": qd[cycle_idx],
                    "QC": qc[cycle_idx],
                    "IR": ir[cycle_idx],
                    "Tmax": tmax[cycle_idx],
                    "Tavg": tavg[cycle_idx],
                    "Tmin": tmin[cycle_idx],
                    "chargetime": chargetime[cycle_idx],
                }
            )
    return pd.DataFrame(records)


def parse_policy_features(policy: str):
    policy = str(policy)
    rates = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)C", policy)]
    switches = [float(x) for x in re.findall(r"\((\d+)%\)", policy)]
    return {
        "max_c_rate": max(rates) if rates else np.nan,
        "mean_c_rate": float(np.mean(rates)) if rates else np.nan,
        "switch_soc_pct": switches[0] if switches else np.nan,
        "policy_steps": len(rates),
    }


def enrich_cycle_life_table(cycle_life_df: pd.DataFrame):
    parsed = cycle_life_df["charging_policy"].apply(parse_policy_features).apply(pd.Series)
    return pd.concat([cycle_life_df.reset_index(drop=True), parsed], axis=1)


def load_batches(data_dir, files, cache_dir="cache", verbose=True):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)  # 캐시 폴더 자동 생성
    bundles = {}
    for batch_name, filename in files.items():
        # 1. 캐시 파일 경로 설정
        cache_path = cache_dir / f"{filename}.joblib"
        if verbose:
            print(f"[process] {batch_name}: {filename}", flush=True)
        # 2. 캐시가 존재하면 바로 로드
        if cache_path.exists():
            if verbose:
                print(f"  -> Loading from cache: {cache_path.name}", flush=True)
            bundles[batch_name] = joblib.load(cache_path)
            continue
        # 3. 캐시가 없으면 원래대로 로드 및 처리
        mat = load_mat(data_dir / filename)
        batch = normalize_batch(mat["batch"])
        df = extract_summary(batch)
        cycle_life_df = enrich_cycle_life_table(
            df.drop_duplicates("cell_id")[["cell_id", "cycle_life", "charging_policy"]]
            .sort_values("cycle_life")
            .reset_index(drop=True)
        )
        nominal_qd = df[df["cycle"] <= 5].groupby("cell_id")["QD"].median().median()
        df_clean = df[df["QD"].between(nominal_qd * 0.8, nominal_qd * 1.2)].copy()
        # 결과 묶기
        result_bundle = {
            "batch": batch,
            "df": df,
            "df_clean": df_clean,
            "cycle_life_df": cycle_life_df,
            "nominal_qd": nominal_qd,
        }
        # 4. 처리가 끝난 결과를 캐시로 저장
        joblib.dump(result_bundle, cache_path)
        if verbose:
            print(f"  -> Cached to: {cache_path.name}", flush=True)
        bundles[batch_name] = result_bundle
        if verbose:
            print(
                f"[loaded] {batch_name}: rows={len(df):,}, cells={cycle_life_df['cell_id'].nunique():,}",
                flush=True,
            )
    return bundles
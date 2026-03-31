#
# 제작 목적: 피처 엔지니어링
# 제작 날짜: 2026-03-31
# 제작자: AI 3기 3반 박진
#


# 라이브러리
from __future__ import annotations

from pathlib import Path

import pandas as pd


# 하이퍼 파라미터
RAW_DATA_PATH = Path("data/raw/source_data.csv")
PROCESSED_FEATURE_PATH = Path("outputs/feature_engineering/model_input.csv")
TARGET_COLUMN = "target"


# 데이터 호출
def load_source_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"원천 데이터 파일이 없습니다: {path}. "
            "실제 원본 데이터 파일 경로로 수정해 주세요."
        )

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"지원하지 않는 파일 형식입니다: {path.suffix}. "
        "현재는 csv 또는 parquet 파일만 기본 지원합니다."
    )


# 데이터 전처리
def clean_source_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    return cleaned_df


# 피처 생성
def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    return feature_df


# 모델링용 데이터 정리
def validate_model_input(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise KeyError(
            f"타깃 컬럼 '{target_column}'이 데이터에 없습니다. "
            "실제 타깃 컬럼명으로 수정해 주세요."
        )

    return df.copy()


def save_feature_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_FEATURE_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_feature_engineering_pipeline(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_FEATURE_PATH,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    raw_df = load_source_data(input_path)
    cleaned_df = clean_source_data(raw_df)
    feature_df = create_model_features(cleaned_df)
    model_input_df = validate_model_input(feature_df, target_column=target_column)
    save_feature_data(model_input_df, path=output_path)
    return model_input_df


def main() -> None:
    run_feature_engineering_pipeline()


if __name__ == "__main__":
    main()

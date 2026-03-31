from __future__ import annotations

import argparse
from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-based EDA checks.")
    parser.add_argument(
        "--feature-set",
        dest="feature_set",
        default=None,
        help="best_summary.json의 feature set 이름 또는 쉼표로 구분한 custom feature list",
    )
    parser.add_argument(
        "--list-feature-sets",
        action="store_true",
        help="선택 가능한 feature set 이름만 출력하고 종료",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    module = importlib.import_module("eda_feature_check")
    if args.list_feature_sets:
        if not hasattr(module, "get_available_feature_set_names"):
            raise AttributeError("src/eda_feature_check.py 안에 get_available_feature_set_names() 함수를 만들어 주세요.")
        for name in module.get_available_feature_set_names():
            print(name)
        raise SystemExit(0)
    if not hasattr(module, "main"):
        raise AttributeError("src/eda_feature_check.py 안에 main() 함수를 만들어 주세요.")
    module.main(requested_feature_set=args.feature_set)

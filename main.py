from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


TASKS = {
    "1": ("feature_engineering", "Feature cache 생성"),
    "2": ("modeling", "모델링 실행"),
    "3": ("visualized", "모델 결과 시각화"),
    "4": ("eda_feature_check", "최종 feature EDA 체크"),
    "5": ("pipeline", "feature_engineering -> modeling -> visualized"),
    "6": ("pipeline_with_eda", "feature_engineering -> modeling -> visualized -> eda_feature_check"),
}


def run_module(module_name: str, *, feature_set: str | None = None) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"src/{module_name}.py 안에 main() 함수를 만들어 주세요.")

    print(f"[RUN] {module_name}.main()")
    if module_name == "eda_feature_check":
        module.main(requested_feature_set=feature_set)
    else:
        module.main()


def run_task(task_key: str, *, feature_set: str | None = None) -> None:
    if task_key == "pipeline":
        for module_name in ("feature_engineering", "modeling", "visualized"):
            run_module(module_name)
        return

    if task_key == "pipeline_with_eda":
        for module_name in ("feature_engineering", "modeling", "visualized"):
            run_module(module_name)
        run_module("eda_feature_check", feature_set=feature_set)
        return

    run_module(task_key, feature_set=feature_set)


def print_menu() -> None:
    print("실행할 작업을 선택해 주세요.")
    for key, (_, description) in TASKS.items():
        print(f"  {key}. {description}")
    print("  q. 종료")


def prompt_task_selection() -> tuple[str | None, str | None]:
    while True:
        print_menu()
        selected = input("선택: ").strip().lower()
        if selected == "q":
            return None, None
        if selected not in TASKS:
            print("올바른 번호를 입력해 주세요.\n")
            continue

        module_name = TASKS[selected][0]
        feature_set = None
        if module_name in {"eda_feature_check", "pipeline_with_eda"}:
            feature_set = input(
                "보고 싶은 feature set 이름을 입력하세요. "
                "엔터만 누르면 best_summary의 best_feature_set을 사용합니다: "
            ).strip()
            feature_set = feature_set or None
        return module_name, feature_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESS prediction project task runner")
    parser.add_argument(
        "--task",
        choices=[
            "feature_engineering",
            "modeling",
            "visualized",
            "eda_feature_check",
            "pipeline",
            "pipeline_with_eda",
        ],
        default=None,
        help="실행할 작업 이름",
    )
    parser.add_argument(
        "--feature-set",
        default=None,
        help="EDA feature check에서 사용할 feature set 이름 또는 comma-separated feature list",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task:
        run_task(args.task, feature_set=args.feature_set)
        return

    while True:
        selected_task, feature_set = prompt_task_selection()
        if selected_task is None:
            print("프로그램을 종료합니다.")
            return

        run_task(selected_task, feature_set=feature_set)
        print()


if __name__ == "__main__":
    main()

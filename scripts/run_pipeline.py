from __future__ import annotations

from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_SEQUENCE = [
    "feature_engineering",
    "modeling",
    "visualized"
]


def run_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"src/{module_name}.py 안에 main() 함수를 만들어 주세요.")
    print(f"[RUN] {module_name}.main()")
    module.main()


if __name__ == "__main__":
    for module_name in MODULE_SEQUENCE:
        run_module(module_name)

from __future__ import annotations

from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    module = importlib.import_module("feature")
    if not hasattr(module, "main"):
        raise AttributeError("src/feature.py 안에 main() 함수를 만들어 주세요.")
    module.main()

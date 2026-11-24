"""The core package must run unmodified in Pyodide: stdlib + numpy only (PRD §5.3, D2)."""

import ast
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "hpfold"
ALLOWED_THIRD_PARTY = {"numpy"}
FORBIDDEN_STDLIB = {"multiprocessing", "subprocess", "socket", "threading", "concurrent"}


def _imported_top_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_package_exists():
    assert (CORE / "__init__.py").is_file()


def test_core_imports_only_stdlib_and_numpy():
    stdlib = set(sys.stdlib_module_names)
    for py in CORE.rglob("*.py"):
        for mod in _imported_top_modules(py):
            if mod == "hpfold":
                continue
            assert mod in stdlib or mod in ALLOWED_THIRD_PARTY, f"{py.name} imports {mod}"
            assert mod not in FORBIDDEN_STDLIB, f"{py.name} imports {mod} (not Pyodide-safe)"

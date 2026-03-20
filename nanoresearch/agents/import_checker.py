"""AST-based cross-file import consistency checker."""
from __future__ import annotations

import ast
from pathlib import Path


class ImportChecker:
    """Check cross-file import consistency using AST parsing.

    Scans all .py files in a directory and detects:
    1. ``from X import Y`` where Y doesn't exist in X
    2. ``import X; X.func()`` where func doesn't exist in X
    """

    def __init__(self, code_dir: Path):
        self.code_dir = code_dir
        self.module_exports: dict[str, set[str]] = {}
        self._parse_all_modules()

    def _parse_all_modules(self) -> None:
        """Extract every exported name from each module via ast.parse."""
        for py_file in self.code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            module_name = py_file.stem
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue

            exports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.add(target.id)
            self.module_exports[module_name] = exports

    def check_imports(self) -> list[dict]:
        """Check all files for import mismatches. Returns list of mismatch dicts.

        Each dict contains:
            importer: filename that has the bad import
            module: target module name
            missing_name: name that doesn't exist in target module
            available: sorted list of names in target module
            line: (optional) line number of the import statement
        """
        local_modules = set(self.module_exports.keys())
        issues: list[dict] = []

        for py_file in self.code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                issues.append({
                    "file": str(py_file.relative_to(self.code_dir)),
                    "type": "syntax_error",
                    "message": "File has syntax errors",
                })
                continue

            # Collect `import X` aliases that refer to local modules
            import_aliases: dict[str, str] = {}  # alias -> real module name
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        real = alias.name
                        # Strip 'src.' prefix
                        if real.startswith("src."):
                            real = real[4:]
                        if real in local_modules:
                            import_aliases[alias.asname or alias.name.split(".")[-1]] = real

            for node in ast.walk(tree):
                # Pattern 1: from X import Y
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod.startswith("src."):
                        mod = mod[4:]
                    if mod not in self.module_exports:
                        continue
                    for alias in (node.names or []):
                        name = alias.name
                        if name == "*":
                            continue
                        if name not in self.module_exports[mod]:
                            issues.append({
                                "importer": py_file.name,
                                "module": mod,
                                "missing_name": name,
                                "available": sorted(self.module_exports[mod]),
                                "line": node.lineno,
                            })

            # Pattern 2: import X; X.func()
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if (isinstance(node.value, ast.Name)
                            and node.value.id in import_aliases):
                        real_mod = import_aliases[node.value.id]
                        attr = node.attr
                        if attr.startswith("_"):
                            continue
                        if attr not in self.module_exports.get(real_mod, set()):
                            issues.append({
                                "importer": py_file.name,
                                "module": real_mod,
                                "missing_name": attr,
                                "available": sorted(self.module_exports.get(real_mod, set())),
                                "usage_pattern": f"import {real_mod}; {real_mod}.{attr}()",
                                "line": node.lineno,
                            })

        # Deduplicate (same importer+module+missing_name)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[dict] = []
        for issue in issues:
            if "missing_name" not in issue:
                deduped.append(issue)
                continue
            key = (issue.get("importer", ""), issue.get("module", ""), issue["missing_name"])
            if key not in seen:
                seen.add(key)
                deduped.append(issue)
        return deduped

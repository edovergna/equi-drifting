"""Automated docstring insertion tool for the repository."""

import ast
import pathlib
import re
from typing import Optional

PLACEHOLDER = "TODO: Add docstring"


def get_docstring_for_node(node: ast.AST) -> str:
    """Generate a placeholder docstring for the given AST node."""
    if isinstance(node, ast.ClassDef):
        return f'"""{PLACEHOLDER} for class {node.name}."""'
    elif isinstance(node, ast.FunctionDef):
        return f'"""{PLACEHOLDER} for function {node.name}."""'
    elif isinstance(node, ast.AsyncFunctionDef):
        return f'"""{PLACEHOLDER} for async function {node.name}."""'
    else:
        return f'"""{PLACEHOLDER}."""'


def get_module_docstring() -> str:
    """Generate a placeholder module docstring."""
    return '"""Module docstring - TODO: Add description."""'


def process_file(filepath: pathlib.Path) -> tuple[str, int]:
    """
    Add missing docstrings to a Python file.
    
    Returns:
        (modified_source, count_of_docstrings_added)
    """
    source = filepath.read_text(encoding="utf-8")
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"SKIP (syntax error): {filepath}")
        return source, 0
    
    lines = source.split("\n")
    insertions = []  # List of (line_index, indent, docstring_text)
    
    # Check module docstring
    if ast.get_docstring(tree, clean=False) is None:
        # Find insertion point (after encoding, shebang, and blank lines)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 and stripped.startswith("#!"):
                insert_idx = i + 1
                continue
            if re.match(r"^#.*coding[:=]", stripped):
                insert_idx = i + 1
                continue
            if stripped == "" or stripped.startswith("#"):
                insert_idx = i + 1
                continue
            break
        
        insertions.append((insert_idx, 0, get_module_docstring()))
    
    # Check all functions and classes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is None:
                if not node.body:
                    continue
                
                first_stmt = node.body[0]
                # Skip if first statement is already a docstring
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, (ast.Constant, ast.Str)):
                    continue
                
                target_line = first_stmt.lineno - 1  # 0-indexed
                indent = get_indent(lines[target_line]) if target_line < len(lines) else "    "
                docstring = get_docstring_for_node(node)
                insertions.append((target_line, 0, indent + docstring))
    
    if not insertions:
        return source, 0
    
    # Sort in reverse order to preserve line indices
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    for line_idx, _, docstring_text in insertions:
        lines.insert(line_idx, docstring_text)
    
    return "\n".join(lines), len(insertions)


def get_indent(line: str) -> str:
    """Extract the indentation from a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else "    "


def main():
    root = pathlib.Path(".").resolve()
    ignore_dirs = {"__pycache__", ".venv", "venv", "env", ".git", "wandb", ".pytest_cache"}
    
    py_files = [p for p in root.rglob("*.py") if not any(part in ignore_dirs for part in p.parts)]
    py_files.sort()
    
    total_added = 0
    files_modified = 0
    
    for filepath in py_files:
        modified_source, count = process_file(filepath)
        if count > 0:
            filepath.write_text(modified_source, encoding="utf-8")
            files_modified += 1
            total_added += count
            print(f"✓ {filepath.relative_to(root)}: added {count} docstrings")
    
    print(f"\nSummary: Modified {files_modified} files, added {total_added} docstrings total")


if __name__ == "__main__":
    main()

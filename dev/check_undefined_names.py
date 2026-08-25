#!/usr/bin/env python3
"""Static scan for names used but never bound, per function.

Exists because this repo's real test gate runs inside the Docker image, and the
dev VM has neither pytest, numpy nor kwconf -- so an edit that references an
out-of-scope variable is not caught until a 90-second image build fails. Two
such NameErrors reached that gate in one day (`recipe` in _dump_policy_json,
and an earlier `_flat_epoch` deleted with the block around it).

AST-only: no imports, no dependencies, runs anywhere python3 does.

    python3 dev/check_undefined_names.py kwcoco_detector_kit tests
"""
import ast
import builtins
import sys
from pathlib import Path

_BUILTINS = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__spec__", "self", "cls",
}


def _module_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(_bound_by(t))
        elif isinstance(node, ast.AnnAssign):      # X: T = ... module constants
            names.update(_bound_by(node.target))
        elif isinstance(node, (ast.If, ast.Try)):
            # conditional imports / try-except-import blocks
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        names.update(_bound_by(t))
                elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
    return names


def _bound_by(target):
    out = set()
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _bindings_of(fn):
    """Names bound anywhere in ``fn``'s own body (not nested function bodies)."""
    names = set()
    # Every arg in the subtree, including nested defs and lambdas. Deliberately
    # over-permissive: a name that is an argument of a SIBLING nested function
    # will be treated as in scope, so this can miss a real error. That is the
    # right trade -- a checker with false positives gets ignored, and the class
    # of bug it exists to catch (referencing an enclosing function's local from
    # a function that never received it) still surfaces.
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            names.add(node.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for x in node.names:
                names.add((x.asname or x.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign):
            names.update(_bound_by(node.target))
        elif isinstance(node, ast.comprehension):
            names.update(_bound_by(node.target))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _walk_scopes(fn, enclosing, findings, seen):
    """Check ``fn`` against ``enclosing``, then recurse into nested defs.

    Closures are the reason this is recursive rather than flat: a nested
    function legitimately reads names bound by the function around it, so the
    parent's bindings must be in scope for the child.
    """
    scope = set(enclosing) | _bindings_of(fn)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in scope and node.id not in _BUILTINS:
                key = (fn.name, node.id)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        (f"{fn.name}(): undefined {node.id!r}", node.lineno))
    for child in fn.body:
        for node in ast.walk(child):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node is not fn:
                    _walk_scopes(node, scope, findings, seen)


def check_file(path):
    try:
        tree = ast.parse(Path(path).read_text())
    except SyntaxError as ex:
        return [(f"SYNTAX: {ex}", getattr(ex, "lineno", 0))]
    outer = _module_names(tree) | _BUILTINS
    findings, seen = [], set()
    # Class bodies bind methods and class attributes into their own scope.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            outer = outer | {n.name for n in node.body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for n in node.body:
                if isinstance(n, ast.Assign):
                    outer = outer | _bound_by(n.targets[0])
                elif isinstance(n, ast.AnnAssign):
                    outer = outer | _bound_by(n.target)
    for node in tree.body:
        for sub in ([node] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else [n for n in getattr(node, "body", [])
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]):
            _walk_scopes(sub, outer, findings, seen)
    return findings


def main(argv):
    roots = argv[1:] or ["kwcoco_detector_kit"]
    total = 0
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            for msg, line in check_file(path):
                print(f"{path}:{line}: {msg}")
                total += 1
    print(f"\n{total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main(sys.argv))


def _selftest():
    """`python3 dev/check_undefined_names.py --selftest`

    A checker nobody has watched fail is not evidence of anything.
    """
    sample = Path(__file__).parent / "testdata" / "undefined_names_regress.py"
    found = {name for name, _ in check_file(sample)}
    assert any("undefined 'recipe'" in f for f in found), \
        f"failed to catch the real bug: {found}"
    assert not any("closure_ok" in f or "lambda_ok" in f for f in found), \
        f"flagged a legitimate closure/lambda: {found}"
    print("selftest OK: catches the out-of-scope read, ignores closures/lambdas")
    return 0

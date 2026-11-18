"""Check that a pyCERR script only uses names that exist in the installed pyCERR.

    python check_pycerr_names.py my_script.py [more.py ...]
    python check_pycerr_names.py - < reply.md        # also finds the ```python block in a reply

For every `cerr.*` module, function and class the script imports or calls, it checks that
the name exists in the installed pyCERR and is not deprecated. Names from optional extras
that are not installed (Qt viewer, ANTs, ...) are reported as "not verifiable", not as errors.
Exit status: 0 = all names found, 1 = invented or deprecated names, 2 = no pyCERR code / syntax error.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    """Return the largest ```python block in a reply (or the reply itself if it parses as code)."""
    blocks = [b for b in _FENCE.findall(text or "") if b.strip()]
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text or "", re.S)   # fence never closed (cut off)
    if m and m.group(1).strip():
        return m.group(1)
    if text and ("import " in text or "planC" in text):
        try:
            ast.parse(text)
            return text
        except SyntaxError:
            return None
    return None


@dataclass
class CheckReport:
    syntax_ok: bool = True
    syntax_error: str = ""
    cerr_refs: set = field(default_factory=set)          # 'cerr.dvh.getDVH', ...
    invented: list = field(default_factory=list)          # names that do not exist
    deprecated: list = field(default_factory=list)
    unknown: list = field(default_factory=list)           # optional deps missing
    uses_cerr: bool = False

    @property
    def ok(self) -> bool:
        return self.syntax_ok and self.uses_cerr and not self.invented and not self.deprecated

    def summary(self) -> str:
        if not self.syntax_ok:
            return f"Syntax error: {self.syntax_error}"
        if not self.uses_cerr:
            return "Does not use pyCERR"
        parts = [f"{len(self.cerr_refs)} pyCERR name(s) checked"]
        if self.invented:
            parts.append("NOT FOUND in pyCERR: " + ", ".join(self.invented))
        if self.deprecated:
            parts.append("deprecated: " + ", ".join(self.deprecated))
        if self.unknown:
            parts.append("not verifiable (optional package missing): " + ", ".join(self.unknown))
        return "; ".join(parts)


def _import(mod: str):
    """Module, or 'missing' if it isn't part of pyCERR, or None if an optional dep is absent."""
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as e:
        name = getattr(e, "name", "") or ""
        if name == mod or (name.startswith("cerr") and mod.startswith(name)):
            return "missing"
        return None
    except Exception:
        return None


def _resolve(mod_name: str, attr: str):
    mod = _import(mod_name)
    if mod is None:
        return None
    if mod == "missing":
        return "missing-module"
    try:
        if hasattr(mod, attr):
            return True
    except Exception:                 # lazy attribute that needs an optional dep
        return None
    sub = _import(f"{mod_name}.{attr}")
    if sub is None:
        return None
    return sub != "missing"


def _deprecated(mod_name: str, attr: str) -> bool:
    try:
        obj = getattr(importlib.import_module(mod_name), attr, None)
        if obj is None or not (inspect.isfunction(obj) or inspect.isclass(obj)):
            return False
        return "DeprecationWarning" in inspect.getsource(obj)
    except Exception:
        return False


def check_script(code: str | None) -> CheckReport:
    r = CheckReport()
    if not code:
        r.syntax_ok, r.syntax_error = False, "no code found"
        return r
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        r.syntax_ok, r.syntax_error = False, f"line {e.lineno}: {e.msg}"
        return r

    alias, modules = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for nm in node.names:
                if nm.name.split(".")[0] == "cerr":
                    modules.add(nm.name)
                    alias[nm.asname or nm.name] = nm.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "cerr":
            modules.add(node.module)
            for nm in node.names:
                alias[nm.asname or nm.name] = f"{node.module}.{nm.name}"
    r.uses_cerr = bool(modules)

    targets = set()
    for node in ast.walk(tree):                    # alias.attr, e.g. pc.loadDcmDir, dvh.Dx
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in alias:
            targets.add((alias[node.value.id], node.attr))
    for full in alias.values():                    # from cerr.x import y
        if full.count(".") >= 1:
            targets.add(tuple(full.rsplit(".", 1)))

    for mod in sorted(modules):
        if _import(mod) == "missing":
            r.invented.append(mod)
    for mod, attr in sorted(targets):
        full = f"{mod}.{attr}"
        if mod in r.invented:
            continue
        res = _resolve(mod, attr)
        if res is None:
            r.unknown.append(full)
            continue
        if res == "missing-module" or res is False:
            # `from cerr import plan_container` style: the name may itself be a module
            r.invented.append(full)
            continue
        r.cerr_refs.add(full)
        if _deprecated(mod, attr):
            r.deprecated.append(full)
    r.invented = sorted(set(r.invented))
    return r


def api_names(code: str | None) -> set:
    """pyCERR names a script uses (for comparing a model answer to a reference)."""
    return check_script(code).cerr_refs if code else set()


def main(argv=None) -> int:
    import sys
    paths = (argv if argv is not None else sys.argv[1:]) or ["-"]
    worst = 0
    for p in paths:
        text = sys.stdin.read() if p == "-" else open(p, encoding="utf-8").read()
        code = extract_code(text) if (p == "-" or not p.endswith(".py")) else text
        r = check_script(code)
        print(f"{p}: {'OK' if r.ok else 'PROBLEM'} - {r.summary()}")
        status = 0 if r.ok else (2 if not (r.syntax_ok and r.uses_cerr) else 1)
        worst = max(worst, status)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())

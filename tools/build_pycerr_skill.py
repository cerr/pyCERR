"""Refresh and check the pyCERR agent skill in ``skills/pycerr``.

The skill is a folder in the Agent Skills layout (``SKILL.md`` plus ``references/``
and ``scripts/``) that tells an LLM agent how to use pyCERR correctly.

Hand-written, edited in place:
    skills/pycerr/SKILL.md
    skills/pycerr/references/viewer_desktop.md
    skills/pycerr/scripts/check_pycerr_names.py
Copied from ``docs/`` by this script (do not edit the copies):
    skills/pycerr/references/quickstart.rst, end_to_end.rst, user_guide/*.rst

Usage (from the repository root)::

    python tools/build_pycerr_skill.py              # refresh the copied docs
    python tools/build_pycerr_skill.py --check      # ... and check every pyCERR name the
                                                    #     skill's code uses against the
                                                    #     installed pyCERR (exit 1 if any is missing)
    python tools/build_pycerr_skill.py --zip DIR    # also write DIR/pycerr.zip for upload

A private build can add a folder of example scripts (``--examples``). It is written to
``--out``, which must be outside the repository so the examples are never committed::

    python tools/build_pycerr_skill.py --examples /private/examples --out /private/skills/pycerr
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "skills" / "pycerr"
DOCS = REPO / "docs"

STEP = """1. **Start from an example.** `examples/INDEX.md` lists example scripts by task, each
   checked against the pyCERR API. Find the closest one and adapt it.
"""
TABLE = "| `examples/INDEX.md` | task-indexed example scripts |\n"


def refresh_docs(skill: Path) -> int:
    ref = skill / "references"
    (ref / "user_guide").mkdir(parents=True, exist_ok=True)
    for old in (ref / "user_guide").glob("*.rst"):
        old.unlink()
    shutil.copy(DOCS / "quickstart.rst", ref / "quickstart.rst")
    shutil.copy(DOCS / "tutorials" / "end_to_end.rst", ref / "end_to_end.rst")
    n = 2
    for f in sorted((DOCS / "user_guide").glob("*.rst")):
        shutil.copy(f, ref / "user_guide" / f.name)
        n += 1
    return n


_RST_BLOCK = re.compile(r"^\.\. code-block:: python\s*\n((?:\n|[ \t]+.*\n)+)", re.M)
_MD_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def code_of(path: Path) -> str:
    """All Python code blocks of a reference, joined (later blocks use earlier imports)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".rst":
        blocks = []
        for b in _RST_BLOCK.findall(text):
            lines = b.splitlines()
            ind = min((len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()), default=0)
            blocks.append("\n".join(ln[ind:] for ln in lines))
    else:
        blocks = _MD_BLOCK.findall(text)
    return "\n\n".join(blocks)


def check(skill: Path) -> int:
    """Run the skill's own name checker on the code in SKILL.md, the references and examples."""
    files = [skill / "SKILL.md", *sorted((skill / "references").rglob("*.md")),
             *sorted((skill / "references").rglob("*.rst"))]
    tmp = Path(tempfile.mkdtemp())
    targets = []
    for f in files:
        code = code_of(f)
        if "cerr" not in code:
            continue
        t = tmp / (f.stem + ".py")
        t.write_text(code, encoding="utf-8")
        targets.append(t)
    targets += sorted((skill / "examples").glob("*.py")) if (skill / "examples").is_dir() else []
    r = subprocess.run([sys.executable, str(skill / "scripts" / "check_pycerr_names.py"), *map(str, targets)],
                       capture_output=True, text=True)
    bad = [ln for ln in r.stdout.splitlines() if "NOT FOUND" in ln or "deprecated" in ln]
    print(f"checked the code of {len(targets)} files against the installed pyCERR: "
          + ("all names found" if not bad else f"{len(bad)} with problems"))
    for ln in bad:
        print("  " + ln)
    return 1 if bad else 0


def header(path: Path) -> dict:
    meta, key = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        m = re.match(r"#\s*(id|category|prompt):\s*(.*)", line)
        if m:
            key = m.group(1)
            meta[key] = m.group(2).strip()
        elif key == "prompt" and line.startswith("#  "):
            meta["prompt"] += " " + line[1:].strip()
        else:
            key = None
    return meta


def add_examples(skill: Path, src: Path) -> int:
    ex = skill / "examples"
    ex.mkdir(exist_ok=True)
    rows = {}
    for f in sorted(src.glob("*.py")):
        if re.search("napari", f.read_text(encoding="utf-8"), re.I):
            continue                       # the skill uses the desktop viewer
        meta = header(f)
        if not meta.get("prompt"):
            continue
        shutil.copy(f, ex / f.name)
        rows.setdefault(meta.get("category", "other"), []).append((f.name, meta["prompt"]))
    lines = ["# Example index", "", "Paths such as `/path/to/patient` are placeholders.", ""]
    for cat in sorted(rows):
        lines += [f"## {cat}", "", "| File | Task |", "|---|---|"]
        lines += [f"| `{n}` | {p.replace('|', '/')} |" for n, p in rows[cat]]
        lines.append("")
    (ex / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    md = skill / "SKILL.md"
    s = md.read_text(encoding="utf-8")
    s = s.replace("<!-- examples:step -->\n1. **Read", STEP + "2. **Read").replace(
        "2. **Check the names", "3. **Check the names").replace("3. Say what", "4. Say what")
    s = s.replace("<!-- examples:table -->\n", TABLE)
    md.write_text(s, encoding="utf-8")
    return sum(len(v) for v in rows.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check the skill's pyCERR names")
    ap.add_argument("--zip", metavar="DIR", help="write DIR/pycerr.zip")
    ap.add_argument("--examples", metavar="DIR", help="private build: add these example scripts")
    ap.add_argument("--out", metavar="DIR", help="private build: where to write the skill (outside the repo)")
    a = ap.parse_args()

    n = refresh_docs(SKILL)
    print(f"refreshed {n} reference pages from docs/ in {SKILL.relative_to(REPO)}")
    skill = SKILL
    if a.examples:
        if not a.out:
            sys.exit("--examples needs --out (a folder outside the repository)")
        out = Path(a.out).resolve()
        if out == REPO or REPO in out.parents:
            sys.exit(f"refusing to write examples inside the repository ({out}): they would be committed")
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(SKILL, out, ignore=shutil.ignore_patterns("__pycache__"))
        k = add_examples(out, Path(a.examples))
        print(f"private build with {k} examples: {out}")
        skill = out
    status = check(skill) if a.check else 0
    if a.zip:
        z = Path(a.zip).resolve() / "pycerr.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(skill.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, Path("pycerr") / p.relative_to(skill))
        print(f"zip: {z}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

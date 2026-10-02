"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# ARCH: verification — keeps ARCHITECTURE.md and the code linked (ARCHITECTURE.md §12)
#
# Checks that ARCHITECTURE.md and the code still point at each other:
#   1. every `file.py::Symbol` (or `file.ipynb`) reference in the document exists — the file, and
#      the class / function / method / module-level name (resolved with `ast`, nothing imported);
#   2. every id of the code map (§13) is an anchor `<a id="...">` in the document, and has at least
#      one `# ARCH: <id>` comment in one of the files the code map lists for it;
#   3. every `# ARCH: <id>` comment in the code (all .py / .ipynb files except tf_reference/) is an
#      id of the code map;
#   4. every in-document link `](#id)` points at an existing anchor, and no anchor is defined twice.
# Standard library only. Run from anywhere:  python parity/check_architecture.py  (exit 1 on error)

import ast
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = "ARCHITECTURE.md"
CODE_MAP_HEADING = "## 13. Code map"
REF = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|ipynb))(?:::([A-Za-z_][A-Za-z0-9_.]*))?`")
ANCHOR = re.compile(r'<a id="([a-z0-9][a-z0-9-]*)"></a>')
LINK = re.compile(r"\]\(#([a-z0-9][a-z0-9-]*)\)")
CODE_ANCHOR_PY = re.compile(r"#\s*ARCH:\s*([a-z0-9][a-z0-9-]*)")
CODE_ANCHOR_NB = re.compile(r"ARCH:\s*([a-z0-9][a-z0-9-]*)")


def repo_files():
    """Tracked and untracked-but-not-ignored files, relative to the repo root."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [f for f in out.splitlines() if os.path.exists(os.path.join(ROOT, f))]
    except (OSError, subprocess.CalledProcessError):
        return [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) for f in fs]


def resolve(path, symbol):
    """True if `symbol` (dotted: Class.method, function, NAME) is defined in the Python file."""
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    scope = tree.body
    for i, part in enumerate(symbol.split(".")):
        found = None
        for node in scope:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part:
                found = node
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in node.targets):
                found = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == part:
                found = node
        if found is None:
            return False
        scope = getattr(found, "body", [])
    return True


def code_map(doc_lines):
    """{id: [code references]} from the code-map table."""
    rows, inside = {}, False
    for line in doc_lines:
        if line.startswith("## "):
            inside = line.startswith(CODE_MAP_HEADING)
            continue
        m = re.match(r"\|\s*`([a-z0-9][a-z0-9-]*)`\s*\|(.*)\|\s*(.*)\|\s*$", line) if inside else None
        if m:
            rows[m.group(1)] = [(f, s) for f, s in REF.findall(m.group(3))]
    return rows


def main():
    errors = []
    doc_text = open(os.path.join(ROOT, DOC), encoding="utf-8").read()
    doc_lines = doc_text.splitlines()

    # 1. code references
    refs = REF.findall(doc_text)
    for path, symbol in sorted(set(refs)):
        if not os.path.exists(os.path.join(ROOT, path)):
            errors.append(f"{DOC}: `{path}` does not exist")
        elif symbol and (not path.endswith(".py") or not resolve(path, symbol)):
            errors.append(f"{DOC}: `{path}::{symbol}` is not defined in {path}")

    # 4. anchors and in-document links
    anchors = ANCHOR.findall(doc_text)
    for a in sorted({a for a in anchors if anchors.count(a) > 1}):
        errors.append(f"{DOC}: anchor `{a}` is defined {anchors.count(a)} times")
    for target in sorted(set(LINK.findall(doc_text)) - set(anchors)):
        errors.append(f"{DOC}: link to #{target}, which is not an anchor")

    # code anchors
    code_anchors = {}  # id -> set of files
    for f in repo_files():
        if f.startswith("tf_reference/") or not f.endswith((".py", ".ipynb")):
            continue
        text = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
        pattern = CODE_ANCHOR_NB if f.endswith(".ipynb") else CODE_ANCHOR_PY
        for aid in pattern.findall(text):
            code_anchors.setdefault(aid, set()).add(f)

    # 2. code map ids
    cmap = code_map(doc_lines)
    if not cmap:
        errors.append(f"{DOC}: no code-map table found under '{CODE_MAP_HEADING}'")
    for aid, entries in cmap.items():
        if aid not in anchors:
            errors.append(f"code map id `{aid}` has no <a id=\"{aid}\"> in {DOC}")
        if not entries:
            errors.append(f"code map id `{aid}` lists no code location")
        listed_files = {f for f, _ in entries}
        if not (code_anchors.get(aid, set()) & listed_files):
            errors.append(f"code map id `{aid}`: no `# ARCH: {aid}` comment in {sorted(listed_files)}")

    # 3. code anchors must be registered
    for aid, files in sorted(code_anchors.items()):
        if aid not in cmap:
            errors.append(f"`ARCH: {aid}` in {sorted(files)} is not in the code map of {DOC}")

    n_code = sum(len(v) for v in code_anchors.values())
    print(f"{DOC}: {len(set(refs))} code references, {len(anchors)} anchors, {len(cmap)} code-map ids; "
          f"code: {n_code} (file, id) anchor pairs")
    if errors:
        print(f"FAILED — {len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        sys.exit(1)
    print("OK — document and code point at each other.")


if __name__ == "__main__":
    main()

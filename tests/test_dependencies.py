"""
Dependency completeness: what the documented install actually provides.

THE GAP THIS CLOSES. tests/test_clean_checkout.py copies the repository to a
temporary directory and runs it with the DEVELOPER's interpreter, which proves the
file layout is portable and nothing about installation. `instructor` shipped as an
undeclared, module-scope import of src/unsw_ai.py: a student who followed the
documented pip command could open the model card and then hit ModuleNotFoundError
on Words, Replay and Shock.

The static checks here run in the default suite - they are fast and offline:

    python -m pytest tests/test_dependencies.py -q

The real fresh-virtualenv install lives in THIS file too (not in
tests/test_clean_checkout.py, which only copies the directory), is marked
`slow_install`, and downloads packages:

    python -m pytest tests/test_dependencies.py -q -m slow_install
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_STDLIB = set(getattr(sys, "stdlib_module_names", ()))

#: import name -> distribution name, where they differ.
_DIST_BY_MODULE = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
}

#: Imported inside a try/except ImportError with a working fallback, so they are
#: genuinely optional and must NOT be required.
_OPTIONAL_IMPORTS = {"tiktoken"}


def _declared_distributions() -> set[str]:
    import re as _re
    names = set()
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _re.match(r"^([A-Za-z0-9_.\-]+)", line)
        if m:
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level import names bound at MODULE scope (not inside a function)."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:                       # module scope only
        targets = [node] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
        if isinstance(node, ast.Try):            # try: import x / except ImportError
            targets = [n for b in node.body if isinstance(b, (ast.Import, ast.ImportFrom))
                       for n in [b]]
        for imp in targets:
            if isinstance(imp, ast.Import):
                found.update(a.name.split(".")[0] for a in imp.names)
            elif imp.level == 0 and imp.module:  # absolute only
                found.add(imp.module.split(".")[0])
    return found


@pytest.mark.parametrize("where", ["src", "tests"])
def test_every_third_party_import_is_declared_in_requirements(where):
    local = {p.stem for p in (REPO / "src").glob("*.py")} | {
        p.stem for p in (REPO / "tests").glob("*.py")}
    declared = _declared_distributions()
    missing = {}
    for path in sorted((REPO / where).glob("*.py")):
        for mod in _module_scope_imports(path):
            if mod in _STDLIB or mod in local or mod in _OPTIONAL_IMPORTS:
                continue
            dist = _DIST_BY_MODULE.get(mod, mod).lower().replace("_", "-")
            if dist not in declared:
                missing.setdefault(dist, []).append(path.name)
    assert not missing, (
        f"undeclared third-party imports: {missing}. Every module-scope import must "
        f"appear in requirements.txt - a student installs from that file and nothing "
        f"else, so an undeclared import is a ModuleNotFoundError on their first run.")


def test_the_pinned_constraints_cover_every_requirement():
    import re as _re
    pinned = set()
    for line in (REPO / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and "==" in line:
            pinned.add(line.split("==")[0].strip().lower().replace("_", "-"))
    undeclared = _declared_distributions() - pinned
    assert not undeclared, (
        f"requirements.txt declares {sorted(undeclared)} with no pinned version in "
        f"constraints.txt - the documented install cannot be reproduced")


@pytest.mark.slow_install
def test_a_genuinely_fresh_virtualenv_can_import_every_stage(tmp_path):
    """
    THE REAL INSTALLATION TEST, opt-in because it downloads packages:

        python -m pytest tests/test_dependencies.py -m slow_install -q

    Builds an empty virtual environment, installs ONLY the declared requirements
    under the pinned constraints, and imports each stage module. The fast static
    test above is the everyday guard; this is what proves the pins actually work.
    """
    import venv
    if os.environ.get("OFFLINE") or os.environ.get("NO_NETWORK"):
        pytest.skip("needs network to install packages")
    env_dir = tmp_path / "venv"
    venv.create(env_dir, with_pip=True)
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = subprocess.run(
        [str(py), "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt"),
         "-c", str(REPO / "constraints.txt")],
        capture_output=True, text=True, timeout=3600)
    assert install.returncode == 0, install.stdout[-3000:] + install.stderr[-3000:]
    probe = subprocess.run(
        [str(py), "-c",
         "import sys; sys.path.insert(0, 'src'); "
         "import config, courseapi, unsw_ai, text_features, decision_replay, "
         "scenarios, cycle_model, model_card; print('ok')"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert probe.returncode == 0, probe.stdout[-3000:] + probe.stderr[-3000:]

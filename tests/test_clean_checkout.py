"""
Clean-install check: the environment and the SUPPLIED pipeline run from a fresh checkout.

What this does NOT prove: that the assignment is complete. The untouched starter passes
this suite - deliberately, since refusing to run unwritten prompts is one of the
behaviours it checks. The completed-assignment check is tests/test_submission.py, run
with -m submission.

Copies the repository to a temporary directory WITHOUT any generated artefacts, then runs
the entry points a team hits in their first hour. Nothing here needs an API key.

This exists because the repository was once only runnable from its original location: config
pointed at sibling directories on the author's machine, and the raw inputs were never copied
in. Every team would have hit that on day one.

Slow (it rebuilds the panel), so it is marked and skipped by default:

    python -m pytest tests/test_clean_checkout.py -q -m clean_install
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Everything a fresh clone would NOT contain. The frozen panel, frozen tiers and the model
# card are deliberately absent from this list: they are tracked artefacts that ship with the
# repository, and a checkout missing them is broken.
GENERATED = [
    "data/processed/panel.parquet",
    "data/processed/tiers.json",
    "data/processed/macro_asof.parquet",
    "data/processed/construct_scores.parquet",
    "data/processed/construct_scores_dev.parquet",
    "data/processed/construct_scores_validation.parquet",
    "data/processed/llm_raw",
    "data/processed/replay_raw",
    "data/processed/shock_raw",
    "outputs/words_audit.json",
    "outputs/words_validation.json",
    "outputs/replay.json",
    "outputs/shock.json",
]


@pytest.fixture(scope="module")
def clean() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="rc-clean-"))
    dst = tmp / "starter-repo"
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "desktop.ini", ".env"))
    for rel in GENERATED:
        p = dst / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    yield dst
    shutil.rmtree(tmp, ignore_errors=True)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("OPENAI_API_KEY", None)      # prove no step here needs one
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=900)


pytestmark = pytest.mark.clean_install


def test_raw_inputs_are_vendored(clean: Path):
    for name in ("rba_minutes_parsed.parquet", "meeting_calendar.csv",
                 "market_data.parquet"):
        assert (clean / "data" / "raw" / name).exists(), f"{name} not in the checkout"


def test_frozen_artefacts_ship(clean: Path):
    for rel in ("data/processed/panel_frozen.parquet",
                "data/processed/tiers_frozen.json",
                "outputs/model_card"):
        assert (clean / rel).exists(), f"{rel} must be tracked, not generated"


def test_model_card_opens_without_a_panel_or_a_key(clean: Path):
    """A team's first action is to interrogate the supplied model. It must just work."""
    r = _run(clean, "src/cycle_model.py", "--show")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "importance" in (r.stdout + r.stderr).lower()


def test_panel_builds_from_a_clean_checkout(clean: Path):
    r = _run(clean, "src/data_panel.py")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert (clean / "data/processed/panel.parquet").exists()
    assert "panel:" in r.stdout


def test_words_dry_run_needs_no_key(clean: Path):
    r = _run(clean, "src/text_features.py", "--dry-run")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "stopping before any API call" in r.stdout


def test_unwritten_prompts_fail_loudly_rather_than_silently(clean: Path):
    """
    A team that runs Words before writing its rubrics must get a clear instruction.

    STARTER ONLY. In the worked solution the prompts ARE written, so calling `run()` on a
    clean checkout with the caches stripped and no API key starts the full 1,055-call retry
    path instead of testing anything - minutes of backoff, then a failure that means nothing.
    The guard below skips rather than pretends.
    """
    import text_features as tf
    if tf._prompts_written():
        pytest.skip("prompts are written in this repository - this is a starter-only check")
    r = _run(clean, "-c",
             "import sys; sys.path.insert(0, 'src'); import text_features as t; t.run()")
    assert r.returncode != 0
    assert "NotImplementedError" in r.stderr and "SYSTEM_PROMPT" in r.stderr


def test_the_contract_suite_passes_on_a_clean_checkout(clean: Path):
    r = _run(clean, "-m", "pytest", "tests/test_contracts.py", "tests/test_causal_tests.py",
             "-q")
    assert r.returncode == 0, r.stdout[-4000:]

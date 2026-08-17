"""Run the pipeline. Start here to understand the flow.

    python run_pipeline.py --check          verify setup, no API calls
    python run_pipeline.py --all            everything, in dependency order
    python run_pipeline.py --core           stages 1-5 only
    python run_pipeline.py --calibration    historical event calibration (section 8.4 Rule 0)
    python run_pipeline.py --scenarios      scenario generation and selection
    python run_pipeline.py --stakeholders   persona modelling
    python run_pipeline.py --dashboard      build the dashboard
    python run_pipeline.py --stage 3        one numbered stage

Phases run in the order listed. Later phases read what earlier ones wrote, so --all is the
only invocation guaranteed to reproduce every artefact from nothing.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

import config

STAGES = [
    ("1", "market data", "src.stage1_market_data"),
    ("2", "documents", "src.stage2_documents"),
    ("3", "risk-voice extraction", "src.stage3_riskvoice"),
    ("4", "regime model", "src.stage4_regime_model"),
    ("5", "uncertainty", "src.stage5_uncertainty"),
]

# Phase name -> (label, modules). Order matters: each depends on the ones above it.
PHASES = {
    "core":         ("stages 1-5", [m for _, _, m in STAGES]),
    "calibration":  ("historical event calibration", ["src.historical_events"]),
    "scenarios":    ("geopolitical scenarios", ["src.scenarios"]),
    "stakeholders": ("persona modelling", ["src.personas"]),
    "dashboard":    ("dashboard", ["src.dashboard"]),
}


def _git(*args) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=15)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception:                                       # noqa: BLE001
        return 127, ""


def secret_scan() -> list[str]:
    """Is a key about to be committed? Checking .gitignore alone is not enough."""
    problems = []
    if not os.path.exists(".env"):
        return problems

    rc, _ = _git("rev-parse", "--is-inside-work-tree")
    if rc != 0:
        problems.append("Not a git repository yet, so .env exposure cannot be checked.")
        return problems

    rc, _ = _git("check-ignore", "-q", ".env")
    if rc != 0:
        problems.append("SECURITY: .env is NOT ignored by git. Add it to .gitignore now.")

    rc, out = _git("ls-files", "--error-unmatch", ".env")
    if rc == 0:
        problems.append("SECURITY: .env is TRACKED by git. Run: git rm --cached .env")

    rc, out = _git("diff", "--cached", "--name-only")
    if rc == 0 and ".env" in out.split():
        problems.append("SECURITY: .env is STAGED for commit. Run: git restore --staged .env")

    rc, out = _git("log", "--all", "--oneline", "--", ".env")
    if rc == 0 and out.strip():
        problems.append("SECURITY: .env appears in git HISTORY. Rotate the key, then purge it.")

    rc, out = _git("grep", "-I", "-l", "-E", r"sk-[A-Za-z0-9_-]{20,}", "--", ".")
    if rc == 0 and out.strip():
        problems.append(f"SECURITY: an API-key-shaped string appears in tracked files:\n"
                        f"      {out.strip().splitlines()[0]}")
    return problems


def check() -> int:
    """Verify setup without spending anything."""
    problems = []

    n = len(list(config.RBA_MINUTES_DIR.glob("*.html"))) if config.RBA_MINUTES_DIR.exists() else 0
    print(f"  RBA minutes found: {n}")
    if n < 200:
        problems.append(f"Expected ~211 minutes, found {n}. Is data/raw/rba-minutes/ populated?")

    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    else:
        print("  API key: found")

    problems += secret_scan()
    if os.path.exists(".env") and not problems:
        print("  .env is ignored, untracked, unstaged and absent from history: yes")

    print(f"  model: {config.MODEL}   temperature: {config.SAMPLING_TEMPERATURE}   "
          f"seed: {config.SEED}")
    print(f"  regimes: {config.N_REGIMES}   parallel calls: {config.N_PARALLEL_CALLS}")
    print(f"  publication lag: {config.PUBLICATION_LAG_DAYS} days")
    if config.PUBLICATION_LAG_DAYS == 0:
        print("    NOTE: 0 means scores are attached to the meeting date. That is look-ahead "
              "for any predictive claim - see the note in stage4_regime_model.align().")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  ! " + p)
        return 1
    print("\nSetup OK.")
    return 0


def run_modules(mods: list[str]) -> None:
    import importlib
    for mod in mods:
        print(f"--- {mod}")
        m = importlib.import_module(mod)
        fn = getattr(m, "run", None) or getattr(m, "build", None)
        if fn is None:
            raise SystemExit(f"{mod} has neither run() nor build()")
        fn()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stage", type=str, default=None)
    for name in PHASES:
        ap.add_argument(f"--{name}", action="store_true")
    a = ap.parse_args()

    if a.check:
        return check()

    if a.stage is not None:
        valid = [num for num, _, _ in STAGES]
        if a.stage not in valid:
            print(f"Unknown stage {a.stage!r}. Valid stages: {', '.join(valid)}", file=sys.stderr)
            return 2
        num, name, mod = next(s for s in STAGES if s[0] == a.stage)
        print(f"--- stage {num}: {name}")
        run_modules([mod])
        return 0

    selected = [n for n in PHASES if getattr(a, n)]
    if a.all or not selected:
        if not selected and not a.all:
            print("Nothing selected. Use --all, --check, a phase flag, or --stage N.\n"
                  f"Phases: {', '.join('--' + p for p in PHASES)}", file=sys.stderr)
            return 2
        selected = list(PHASES)

    for name in PHASES:                       # preserve dependency order
        if name in selected:
            label, mods = PHASES[name]
            print(f"\n=== {name}: {label}")
            run_modules(mods)
    return 0


if __name__ == "__main__":
    sys.exit(main())

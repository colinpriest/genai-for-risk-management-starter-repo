"""Run the pipeline in order. Start here to understand the flow.

    python run_pipeline.py --check     verify setup, no API calls
    python run_pipeline.py             run everything
    python run_pipeline.py --stage 3   run one stage
"""
from __future__ import annotations
import argparse, os, sys
import config

STAGES = [
    ("1", "market data", "src.stage1_market_data"),
    ("2", "documents", "src.stage2_documents"),
    ("3", "sentiment", "src.stage3_sentiment"),
    ("4", "regime model", "src.stage4_regime_model"),
    ("5", "uncertainty", "src.stage5_uncertainty"),
]


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

    if os.path.exists(".env"):
        gi = open(".gitignore").read() if os.path.exists(".gitignore") else ""
        if ".env" not in gi:
            problems.append("SECURITY: .env exists but is not in .gitignore.")
        else:
            print("  .env is git-ignored: yes")

    print(f"  model: {config.MODEL}   temperature: {config.TEMPERATURE}   seed: {config.SEED}")
    print(f"  regimes: {config.N_REGIMES}   parallel calls: {config.N_PARALLEL_CALLS}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  ! " + p)
        return 1
    print("\nSetup OK.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", type=str, default=None)
    a = ap.parse_args()

    if a.check:
        return check()

    import importlib
    for num, name, mod in STAGES:
        if a.stage and a.stage != num:
            continue
        print(f"--- stage {num}: {name}")
        importlib.import_module(mod).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

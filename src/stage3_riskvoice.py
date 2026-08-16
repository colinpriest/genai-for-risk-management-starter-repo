"""
Stage 3 — Risk-voice extraction from RBA minutes

OWNER: <put your name here>

=============================================================================================
WHAT IS PRE-BUILT AND WHAT IS YOURS
=============================================================================================
PRE-BUILT (do not rewrite - it is plumbing, not judgement):
    - parallel calling, retries, raw response saving
    - the Pydantic schema shape and the five field names
    - averaging across calls, spread measurement, output format

YOURS (this is what is marked):
    - THE PROMPTS. Both SYSTEM_PROMPT and the five FIELD_DESCRIPTIONS below are blank.
      Writing them is the core task of this stage.
    - Deciding which part of each document to send (see stage2 retrieval)
    - Judging whether your scores discriminate, and iterating until they do

=============================================================================================
THE FIVE CONSTRUCTS - AND WHY THESE FIVE
=============================================================================================
You are NOT scoring whether the RBA is hawkish or dovish. Stance tells you which way rates
move; it says nothing about how uncertain the world is, and this model predicts VOLATILITY.

Score how much uncertainty and risk the Board is signalling:

    1. financial_conditions_concern   markets, credit, funding costs, housing finance
    2. downside_risk_emphasis         are the risks discussed skewed down or up?
    3. global_risk_salience           how much of the discussion is offshore
    4. vigilance                      what the Board says it will WATCH or MONITOR
    5. uncertainty_language           how heavily the Board hedges

A sixth field, policy_stance, is included as a DELIBERATE CONTRAST. It is the obvious thing
to score and it is the wrong thing. Compare its behaviour against the five.

=============================================================================================
WRITES data/processed/riskvoice_scores.parquet
       data/processed/llm_raw/<date>.json
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

import config

load_dotenv()
client = OpenAI()

# =============================================================================================
# YOUR WORK STARTS HERE
# =============================================================================================
# The model sees ONLY these descriptions. They are the whole prompt for each score.
# Be specific about what counts as high and what counts as low, and about the scale.
#
# Things worth thinking about before you write them:
#   - The RBA writes in restrained, formulaic prose. A small shift is a big signal.
#   - If every document scores near the middle you have not discriminated. Check the spread.
#   - A description that works for the Federal Reserve may not work here.
#   - Look at some actual minutes before writing these. Do not guess.

SYSTEM_PROMPT = """
TODO: write the system prompt.

Who is the model? What is it reading? What is it trying to judge, and why?
What should it calibrate against?
"""

FIELD_DESCRIPTIONS = {
    "financial_conditions_concern": "TODO: 0 to 1. What does high mean? What does low mean?",
    "downside_risk_emphasis":       "TODO: -1 to +1. Which end is which?",
    "global_risk_salience":         "TODO: 0 to 1.",
    "vigilance":                    "TODO: 0 to 1.",
    "uncertainty_language":         "TODO: 0 to 1.",
    "policy_stance":                "TODO: -1 to +1. The contrast field.",
}

# Which part of the document to send. 'text_retrieved' is the passages your stage 2 retrieval
# selected; 'text_full' is everything. Retrieval is required by the brief, but you decide what
# your retrieval query is - see stage2.
TEXT_COLUMN = "text_retrieved"

# =============================================================================================
# PRE-BUILT BELOW THIS LINE
# =============================================================================================

FIELDS = list(FIELD_DESCRIPTIONS)

RiskVoice = create_model(
    "RiskVoice",
    **{name: (float, Field(description=desc)) for name, desc in FIELD_DESCRIPTIONS.items()},
    __base__=BaseModel,
)


def _one_call(text: str, seed: int) -> dict:
    try:
        r = client.beta.chat.completions.parse(
            model=config.MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"RBA minutes:\n\n{text}"}],
            response_format=RiskVoice,
            temperature=config.SAMPLING_TEMPERATURE,   # MUST be > 0, see config.py
            seed=seed,
        )
        d = r.choices[0].message.parsed.model_dump()
        d["_seed"] = seed
        d["_tok_in"] = r.usage.prompt_tokens
        d["_tok_out"] = r.usage.completion_tokens
        return d
    except Exception as e:                                   # noqa: BLE001
        return {"_seed": seed, "_error": str(e)[:200]}


def score_doc(text: str, n: int | None = None) -> list[dict]:
    """N separate calls, run in parallel. Different seed each, so they can differ."""
    n = n or config.N_PARALLEL_CALLS
    with ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in
                [ex.submit(_one_call, text, config.SEED + i) for i in range(n)]]


def score_field(text: str, field: str = "financial_conditions_concern", n: int = 5) -> float:
    """Averaged single-number scorer. The explainability harness calls this.

    Averaging matters: one call has a sampling sd of roughly 0.05-0.10, which is the same
    size as the perturbation effects you are trying to detect. See TRAPS.md.
    """
    ok = [c[field] for c in score_doc(text, n) if "_error" not in c]
    return float(np.mean(ok)) if ok else float("nan")


def check_prompts_written() -> None:
    if "TODO" in SYSTEM_PROMPT or any("TODO" in d for d in FIELD_DESCRIPTIONS.values()):
        raise SystemExit(
            "\nSTOP: the prompts are still TODO placeholders.\n"
            "Writing them is the marked part of this stage. Open src/stage3_riskvoice.py.\n")


def run() -> pd.DataFrame:
    check_prompts_written()
    docs = pd.read_parquet(config.DATA_PROCESSED / "documents.parquet")
    rows, t0, ti, to = [], time.time(), 0, 0

    def work(rec):
        return rec, score_doc(rec[TEXT_COLUMN])

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(work, r) for _, r in docs.iterrows()]
        done = 0
        for f in as_completed(futs):
            rec, calls = f.result()
            date = pd.Timestamp(rec["meeting_date"]).strftime("%Y-%m-%d")
            json.dump(calls, open(config.LLM_RAW / f"{date}.json", "w"), indent=1)
            ok = [c for c in calls if "_error" not in c]
            ti += sum(c.get("_tok_in", 0) for c in ok)
            to += sum(c.get("_tok_out", 0) for c in ok)
            row = {"meeting_date": pd.Timestamp(rec["meeting_date"]), "n_calls_valid": len(ok)}
            for fld in FIELDS:
                v = [c[fld] for c in ok]
                row[fld] = float(np.mean(v)) if v else np.nan
                row[f"{fld}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            rows.append(row)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(docs)}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)
    df.to_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet", index=False)

    print(f"\n  {len(df)} meetings in {time.time()-t0:.0f}s   "
          f"est cost ${ti/1e6*0.15 + to/1e6*0.60:.2f}")
    print(f"  {'construct':32s} {'mean':>7s} {'spread':>8s} {'within-doc sd':>14s}  discriminating?")
    for fld in FIELDS:
        spread = df[fld].std()
        flag = "yes" if spread > 0.10 else "NO - scores are bunched, revise the description"
        print(f"  {fld:32s} {df[fld].mean():+7.3f} {spread:8.3f} "
              f"{df[f'{fld}_sd'].mean():14.3f}  {flag}")
    return df


if __name__ == "__main__":
    run()

"""
Stakeholder modelling — section 8.5

PRE-BUILT: the situation builder (which pulls REAL output from your own model), the reaction
           schema, and the runner.
YOURS:     the persona descriptions. They are blank. Writing them is the marked work.

WRITES data/processed/stakeholders.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

import config

load_dotenv()

# Created on first use, not at import. Importing this module must not require an API key -
# tests/test_contracts.py imports it to read the construct names.
_client = None


def client_():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# =============================================================================================
# YOUR WORK STARTS HERE
# =============================================================================================
# Write a persona for each. The model is told to answer entirely in character, so everything it
# knows about this person comes from what you write.
#
# GROUND YOUR FIGURES IN PUBLISHED DATA AND CITE THE SOURCE. Suitable sources:
#   ABS                              household income, housing finance commitments, loan sizes
#   RBA Financial Stability Review   debt-to-income, share on variable rates, savings buffers
#   RBA Chart Pack                   housing and household finance
#   APRA                             bank dividends and profitability
#
# A persona built on invented numbers is a character, not a model. You do not need precision -
# you need figures a reader would accept as representative, with a source attached.
#
# Include: age and situation, financial position with real magnitudes, level of financial
# sophistication, what they actually care about, and what they do NOT care about.

PERSONAS = {
    "young mortgage holder": """
TODO: write this persona.

A recent first home buyer with a large loan relative to income, on a variable rate, with a thin
savings buffer. What are the actual numbers, and where did you get them?
""",

    "bank shareholder": """
TODO: write this persona.

An Australian RETAIL bank shareholder - typically a self-funded retiree or an SMSF - holding major
bank shares for franked dividend income. NOT an institutional fund manager: a professional would
simply hedge, which removes the conflict this exercise depends on.

What is their position, and what do they actually care about?
""",
}

# =============================================================================================
# PRE-BUILT BELOW THIS LINE
# =============================================================================================


class Reaction(BaseModel):
    headline_reaction: str = Field(description="one sentence, in this person's own voice")
    what_they_want_to_know: list[str] = Field(description="up to 3 questions they would ask")
    action_they_would_take: str
    good_or_bad_for_them: str = Field(description="good, bad, or mixed")
    why: str
    what_they_would_object_to: str = Field(
        description="what in this analysis they would push back on")


def check_personas_written() -> None:
    if any("TODO" in p for p in PERSONAS.values()):
        raise SystemExit(
            "\nSTOP: the personas are still TODO placeholders.\n"
            "Writing them is the marked part of this section. Open src/personas.py.\n")


def build_situations(scenarios: list[dict] | None = None) -> dict:
    """Situations built from YOUR model's real output, not from hypotheticals."""
    reg = pd.read_parquet(config.DATA_PROCESSED / "regimes.parquet")
    summ = json.load(open(config.DATA_PROCESSED / "regime_summary.json"))
    risk = summ["risk"]
    cur = reg.iloc[-1]
    now = config.REGIME_NAMES[int(cur["regime"])]
    worst = config.REGIME_NAMES[config.N_REGIMES - 1]

    sits = {
        "current regime call": (
            f"A risk model reports the Australian share market is currently in its '{now}' "
            f"volatility regime, with {cur['regime_confidence']:.0%} confidence. Historically "
            f"that regime has had annualised volatility of {risk[now]['ann_vol']:.0%} and a daily "
            f"expected shortfall of {risk[now]['ES_95_daily']:.1%}. What is your reaction?"),
        "crisis regime warning": (
            f"The model now reports a materially raised probability of entering its '{worst}' "
            f"regime over the next quarter. Historically that regime has had annualised volatility "
            f"of {risk[worst]['ann_vol']:.0%}, and its worst single day was "
            f"{risk[worst]['worst_day']:.1%}. The model does NOT forecast interest rates - it "
            "models market volatility. What is your reaction?"),
    }
    if scenarios:
        s = scenarios[0]
        sits[f"scenario: {s['name'][:40]}"] = (
            f"An analyst modelled this scenario: {s['description']} It would reach Australia "
            f"mainly through {s['transmission_route'].replace('_', ' ')}. Scaled from what "
            f"{s.get('analogue_used', 'a comparable past event')} actually did, volatility would be "
            f"about {s.get('scaled_vol_multiple', 1.5):.1f} times current levels. What is your "
            "reaction?")
    return sits


def ask(persona_name: str, persona: str, situation: str) -> dict:
    r = client_().beta.chat.completions.parse(
        model=config.MODEL,
        messages=[{"role": "system",
                   "content": persona + "\n\nAnswer entirely in character. Do not hedge like an "
                                        "analyst. Say what this person would actually say."},
                  {"role": "user", "content": situation}],
        response_format=Reaction, temperature=config.SAMPLING_TEMPERATURE, seed=config.SEED)
    d = r.choices[0].message.parsed.model_dump()
    d["persona"] = persona_name
    return d


def run(scenarios: list[dict] | None = None) -> dict:
    check_personas_written()
    if scenarios is None:
        p = config.DATA_PROCESSED / "scenarios.json"
        scenarios = json.load(open(p)).get("selected") if p.exists() else None

    sits = build_situations(scenarios)
    out = []
    for sname, sit in sits.items():
        for pname, persona in PERSONAS.items():
            r = ask(pname, persona, sit)
            r["situation"] = sname
            out.append(r)
            print(f"  [{sname[:26]:28s}] {pname:22s} -> {r['good_or_bad_for_them']:6s} "
                  f"| {r['headline_reaction'][:64]}")

    json.dump({"personas": PERSONAS, "situations": sits, "reactions": out},
              open(config.DATA_PROCESSED / "stakeholders.json", "w"), indent=2)

    print("\n  Now interpret: where do they conflict, where do they unexpectedly agree,")
    print("  and what does that mean for how you report a single set of numbers?")
    return {"reactions": out}


if __name__ == "__main__":
    run()

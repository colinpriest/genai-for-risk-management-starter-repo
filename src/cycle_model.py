"""
CYCLE - separate causation from association in the supplied policy-cycle model.

YOU DO NOT FIT A MODEL IN THIS STAGE. It is supplied, tuned and exported: see model_card.py
for the classifier, the held-out feature importances, the partial dependences and the
predicted regime probabilities for all 211 meetings.

WHAT YOU DO

  1. IN THE CHATGPT INTERFACE, attach the importance table and the partial-dependence plots
     from outputs/model_card/ and get the model to propose a causal mechanism for each of
     the top features. Push back. Ask it which direction the causation runs and what would
     have to be true for its story to hold. Save the transcript - it is a submitted artefact.

     Use the chat interface, not the API, for this. You need to paste in charts, you need to
     argue back, and there is no batch to process.

  2. TRANSCRIBE each proposal into a CausalClaim below - including the ones you expect to
     fail. A refuted claim is evidence. A missing one is not.

  3. RUN THE TESTS and report what survived, what was refuted, and what could not be tested.
     All three outcomes earn marks.

THE POINT
    The model is not a causal model - and, once the label embargo is applied, not much of a
    predictor either: it scores 0.570 against a 0.656 current-decision baseline. Neither
    fact stops an LLM producing a fluent economic mechanism for any feature you show it,
    including one that is noise, and it will give you no signal about which it just did.
    Telling the difference is the skill, and it is independent of whether the model works.

    At least one of the top features in this model cannot possibly be causal. Find it.

WRITES outputs/cycle.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config          # noqa: E402
import model_card      # noqa: E402
from causal_tests import CausalClaim, report, test_claim  # noqa: E402

# ###########################################################################################
# YOUR WORK STARTS HERE
# ###########################################################################################
#
# Transcribe the mechanisms your ChatGPT session proposed. One CausalClaim per feature.
#
#   feature      must be a column in panel.parquet - see model_card.importances()
#   mechanism    the causal story the LLM gave you, in one sentence, in ITS words not yours
#   claimed      what the LLM implied: "causal" | "reverse" | "confounded" | "proxy".
#                This records the LLM's claim, which IS usually causal - that is the
#                thing being tested. Your own verdict uses the stricter vocabulary below.
#   confounders  what you will condition on to test it. THIS IS YOUR JUDGEMENT and it is
#                what makes the conditioning test informative. For the RBA the obvious
#                candidates are the things the Board is reacting to.
#
# Cover AT LEAST the top five features by importance, and at least three claims must be ones
# you intend to test seriously rather than strawmen.

CLAIMS: list[CausalClaim] = [
    # CausalClaim(
    #     feature="slope_cash3y",
    #     mechanism="TODO: what did the LLM say the mechanism was?",
    #     claimed="causal",
    #     confounders=["trimmed_mean_yoy", "unemployment_rate"],
    # ),
    # TODO: at least five claims.
]

# After running the tests, record your verdict per feature. This is the assessed output:
# the tests produce evidence, you produce the judgement.
#
# NONE OF THESE LABELS CLAIMS CAUSATION, AND THE VOCABULARY IS DELIBERATE. The strongest
# thing four correlational diagnostics on 211 overlapping observations can say is that a
# mechanism has not been ruled out. An earlier version of this list offered "causal" as a
# permitted verdict, which invited students to assert exactly what the brief tells them the
# tests cannot establish.
#
#   "not_ruled_out"  the mechanism survives every test and the direction is consistent with
#                    it. This is the strongest available verdict. It is not "causal".
#   "reverse"        the evidence points the other way: policy, or its anticipation, moves
#                    the feature
#   "confounded"     the association is largely explained by something else
#   "untestable"     no test here can speak to it - a legitimate answer, if argued
#
# WHAT THESE TESTS CANNOT DO, and what your report should say alongside any verdict:
#   - they treat a three-state ORDINAL outcome as continuous
#   - they report point estimates with NO uncertainty, on a sample where the 182-day target
#     windows of neighbouring meetings overlap almost entirely, so the effective number of
#     independent observations is far below 211
#   - the direction test compares two IN-SAMPLE R-squared values fitted on different
#     variables, which is a heuristic comparison and not a test statistic
#   - none of them can see an interaction, so a feature that matters only in combination
#     comes back "untestable" rather than "important"
#
# Treat the output as four heuristic diagnostics that between them make some stories much
# harder to hold, not as an identification strategy.
VERDICTS: dict[str, str] = {
    # "slope_cash3y": "reverse",
}

# ###########################################################################################
# SUPPLIED BELOW THIS LINE
# ###########################################################################################


def _panel() -> pd.DataFrame:
    p = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    return p.set_index("meeting_date").sort_index()


def _file_fp(path) -> str:
    import hashlib
    import pathlib
    p = pathlib.Path(path)
    return (hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "absent")


def config_hash() -> str:
    """
    Identifies THIS Cycle configuration: the transcribed claims, the recorded verdicts,
    and the model-card evidence they were tested against. Stored in cycle.json and
    compared by the submission check - editing a claim or a verdict after the run leaves
    a stale artefact that no longer passes as current.
    """
    import hashlib
    payload = json.dumps({
        "claims": [{"feature": c.feature, "mechanism": c.mechanism,
                    "claimed": c.claimed, "confounders": list(c.confounders),
                    "note": getattr(c, "note", "")} for c in CLAIMS],
        "verdicts": VERDICTS,
        "panel": _file_fp(config.DATA_PROCESSED / "panel.parquet"),
        "model_card_performance": _file_fp(config.MODEL_CARD / "performance.json"),
        "model_card_importances": _file_fp(config.MODEL_CARD / "importances.parquet"),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def show_model() -> None:
    """What the supplied model looks like. Start here."""
    model_card.summary()
    print("\n  partial dependences available for: "
          + ", ".join(sorted(model_card.partial_dependences())))
    print("  regime probabilities: outputs/model_card/regime_probabilities.parquet")


def run() -> dict:
    if not CLAIMS:
        raise NotImplementedError(
            "CLAIMS is empty. Run your ChatGPT session first, transcribe at least five "
            "causal proposals, then re-run. See the brief, Cycle stage.")

    config.stage_begin("cycle", config_hash())
    print("=" * 78)
    show_model()
    print("\n" + "=" * 78)
    print(f"  TESTING {len(CLAIMS)} CAUSAL CLAIMS")
    print("=" * 78)

    panel = _panel()
    results = []
    for claim in CLAIMS:
        r = test_claim(panel, claim)
        report(r)
        r["verdict"] = VERDICTS.get(claim.feature, "NOT RECORDED")
        results.append(r)

    missing = [c.feature for c in CLAIMS if c.feature not in VERDICTS]
    if missing:
        print(f"\n  WARNING: no verdict recorded for {missing}. The tests are evidence; "
              f"the verdict is the assessed output.")

    # Coverage is checked by FAMILY: rv_5, rv_21 and rv_63 are one measure at three
    # windows, and a claim about realised volatility covers all three. Requiring a separate
    # causal story per window would reward padding rather than thinking.
    def _family(f: str) -> str:
        return re.sub(r"_[0-9]+$", "", f)

    top5 = list(model_card.importances().head(5)["feature"])
    covered = {_family(c.feature) for c in CLAIMS}
    uncovered = sorted({f for f in top5 if _family(f) not in covered})
    if uncovered:
        print(f"  WARNING: top-5 features with no claim: {uncovered}")
    else:
        print(f"  all top-5 features are covered by a claim "
              f"(by family: {sorted({_family(f) for f in top5})})")

    out = {"claims": results,
           "verdicts": VERDICTS,
           "top5_features": top5,
           "top5_uncovered": uncovered,
           "config_hash": config_hash()}
    config.atomic_write_text(config.OUTPUTS / "cycle.json",
                             json.dumps(out, indent=2, default=float))
    config.stage_complete("cycle", config_hash())
    return out


if __name__ == "__main__":
    if "--show" in sys.argv:
        show_model()
    else:
        run()

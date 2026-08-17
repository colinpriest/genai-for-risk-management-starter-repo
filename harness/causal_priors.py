"""Causal priors elicited from the LLM. SUPPLIED - DO NOT MODIFY.

WHAT THIS IS FOR
    You have a fitted model that says how each feature moves the volatility regime. That
    tells you what the DATA did. It does not tell you whether the relationship is causal,
    which direction the causation runs, or whether both variables are being moved by
    something else entirely.

    This module asks the LLM for its PRIOR - what it expects before seeing your results -
    and returns it in a structured form you can put next to your fitted coefficients.

    The point is the DISAGREEMENTS. Three cases are worth a paragraph each in your report:

      1. The model finds a strong effect where the prior says "no causal relationship".
         Usually confounding, sometimes a genuine discovery. Which one is your judgement.
      2. The prior says "increasing" and your coefficient is negative. Either the prior is
         wrong, or your construct does not measure what its name suggests.
      3. The prior says "output causes input" - reverse causation. For RBA text and market
         volatility this is the live worry: the Board writes about turmoil BECAUSE the market
         is turbulent, so a model that "predicts" volatility from that language may be
         reading the market's own history back to you.

WHAT THIS IS NOT
    An LLM's prior is not evidence. It is a compressed summary of what has been written
    about these variables, and it inherits whatever is conventional, including the
    conventional mistakes. Treat it as a colleague's opinion offered before seeing the data:
    useful for deciding where to look, worthless as proof.

    Do NOT use it to choose features. Choosing features by their elicited prior and then
    reporting the fit as evidence for the prior is circular.

USAGE
    from harness.causal_priors import elicit_priors, compare_with_fit

    priors = elicit_priors(features, target="21-day realised volatility of the ASX 200")
    print(compare_with_fit(priors, model_artifact))
"""
from __future__ import annotations

import json
import time
from typing import Literal

from pydantic import BaseModel, Field

import config

# --- the response schema ---------------------------------------------------------------
# Ordinal 0-2 rather than a continuous score: an LLM asked for 0.73 will produce 0.73, and
# the extra precision is invented. Three levels is what this elicitation can actually
# support.
Shape = Literal["increasing", "decreasing", "u_shape", "none"]
Causal = Literal["input_causes_output", "output_causes_input",
                 "shared_external_cause", "no_causal"]


class FeaturePrior(BaseModel):
    feature: str = Field(description="The exact feature name given to you.")

    shape: Shape = Field(
        description="How you expect the OUTPUT to change as the INPUT rises. "
                    "'increasing' = output rises with input. 'decreasing' = output falls. "
                    "'u_shape' = extreme values in either direction raise the output while "
                    "middling values lower it. 'none' = no systematic relationship.")
    shape_strength: int = Field(
        ge=0, le=2,
        description="How strong you expect that relationship to be. "
                    "0 = negligible or you are unsure. 1 = moderate, detectable in a large "
                    "sample. 2 = strong, would be obvious in a scatterplot.")

    causal_direction: Causal = Field(
        description="Your belief about the CAUSAL structure. "
                    "'input_causes_output' = changes in the input produce changes in the "
                    "output. 'output_causes_input' = the reverse; the output moves first and "
                    "the input responds to it. 'shared_external_cause' = neither causes the "
                    "other, both respond to some third factor. 'no_causal' = no causal link "
                    "in either direction, any association would be coincidental.")
    causal_strength: int = Field(
        ge=0, le=2,
        description="Confidence in that causal claim. 0 = weak or speculative. "
                    "1 = moderate, defensible. 2 = strong, well established.")

    reasoning: str = Field(
        description="Two sentences. Name the mechanism you have in mind, and say what would "
                    "change your view.")


class PriorSet(BaseModel):
    priors: list[FeaturePrior]


# --- elicitation -------------------------------------------------------------------------
_client = None


def _client_():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        _client = OpenAI()
    return _client


PROMPT = """You are advising a risk modelling team BEFORE they show you any results.

They are modelling this OUTPUT: {target}

They have these INPUTS, each measured from the minutes of Reserve Bank of Australia monetary
policy meetings unless the name says otherwise:

{feature_block}

For each input, give your PRIOR expectation - what you believe before seeing their data.

Two separate judgements per input, and keep them separate:

  1. SHAPE: how you expect the output to move as the input rises.
  2. CAUSAL DIRECTION: whether the input drives the output, the output drives the input,
     both are driven by something else, or there is no causal link at all.

These are different questions and they often have different answers. A strong association
with reverse causation is common here: the Reserve Bank writes about market turmoil because
markets are turbulent, so language and volatility can move together with the causation
running from the market to the text rather than the other way.

Be willing to say 'no_causal' and 0. A prior that finds a strong causal story for every
input is not a prior, it is a reflex."""


def elicit_priors(features: list[str], target: str,
                  feature_notes: dict[str, str] | None = None,
                  n_repeats: int = 3) -> dict:
    """Ask for the model's prior on each feature, REPEATEDLY.

    n_repeats matters. A single elicitation at temperature 1.0 is one draw from a
    distribution, and reporting it as "the model's prior" hides how much it wobbles. Running
    it several times and reporting the modal answer plus the disagreement rate tells you
    whether the prior is a belief or a coin toss.
    """
    notes = feature_notes or {}
    feature_block = "\n".join(
        f"- {f}" + (f": {notes[f]}" if f in notes else "") for f in features)
    prompt = PROMPT.format(target=target, feature_block=feature_block)

    runs, t0 = [], time.time()
    for i in range(n_repeats):
        try:
            r = _client_().beta.chat.completions.parse(
                model=config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=PriorSet,
                temperature=config.SAMPLING_TEMPERATURE,
                seed=config.SEED + i,
            )
            runs.append([p.model_dump() for p in r.choices[0].message.parsed.priors])
        except Exception as e:                               # noqa: BLE001
            runs.append({"_error": str(e)[:200]})

    ok = [r for r in runs if isinstance(r, list)]
    out = {"target": target, "features": list(features), "n_repeats": n_repeats,
           "n_ok": len(ok), "elapsed_s": round(time.time() - t0, 1), "runs": runs}

    # Modal answer per feature, plus how often the runs agreed.
    summary = {}
    for f in features:
        got = [p for run in ok for p in run if p.get("feature") == f]
        if not got:
            summary[f] = {"error": "not returned by any run"}
            continue

        def _modal(key):
            vals = [p[key] for p in got if key in p]
            if not vals:
                return None, 0.0
            top = max(set(vals), key=vals.count)
            return top, vals.count(top) / len(vals)

        shape, shape_agree = _modal("shape")
        causal, causal_agree = _modal("causal_direction")
        summary[f] = {
            "shape": shape,
            "shape_agreement": round(shape_agree, 2),
            "shape_strength": round(sum(p["shape_strength"] for p in got) / len(got), 2),
            "causal_direction": causal,
            "causal_agreement": round(causal_agree, 2),
            "causal_strength": round(sum(p["causal_strength"] for p in got) / len(got), 2),
            "reasoning_example": got[0].get("reasoning", ""),
            "n_responses": len(got),
        }
    out["summary"] = summary
    out["_how_to_read"] = (
        "An agreement below about 0.67 means the runs disagreed with each other, and the "
        "modal answer should not be quoted as 'the model's prior' - report the disagreement "
        "instead. Compare `shape` against the SIGN of your fitted coefficient and "
        "`causal_direction` against what your design can actually support.")
    return out


# --- comparison against the fit ----------------------------------------------------------
def compare_with_fit(priors: dict, artifact: dict, regime_index: int | None = None) -> dict:
    """Put the elicited prior next to the fitted coefficient and flag the disagreements.

    regime_index defaults to the most turbulent regime, because that is the state whose
    coefficients the report interprets.
    """
    regimes = artifact["regimes"]
    k = artifact["n_regimes"]
    r_i = (k - 1) if regime_index is None else regime_index
    beta = regimes[r_i]["beta"]

    rows, flags = {}, []
    for f, pr in priors.get("summary", {}).items():
        if "error" in pr or f not in beta:
            continue
        b = float(beta[f])
        fitted = "increasing" if b > 0 else "decreasing" if b < 0 else "none"
        agrees = (pr["shape"] == fitted) or pr["shape"] in ("u_shape", "none")

        rows[f] = {
            "prior_shape": pr["shape"],
            "prior_shape_strength": pr["shape_strength"],
            "prior_causal_direction": pr["causal_direction"],
            "prior_causal_strength": pr["causal_strength"],
            "prior_agreement_across_runs": pr["shape_agreement"],
            "fitted_coefficient": b,
            "fitted_direction": fitted,
            "sign_matches_prior": bool(agrees),
        }

        if not agrees and pr["shape_strength"] >= 1:
            flags.append(
                f"{f}: the prior expected '{pr['shape']}' with strength "
                f"{pr['shape_strength']}, the fit is {fitted} (beta {b:+.3f}). Either the "
                f"prior is wrong or the construct does not measure what its name suggests.")
        if pr["causal_direction"] == "output_causes_input" and pr["causal_strength"] >= 1:
            flags.append(
                f"{f}: the prior says REVERSE causation - volatility drives this feature, not "
                f"the other way. Any predictive claim you make from it needs the publication "
                f"lag argument, and probably a lead-lag check as well.")
        if pr["causal_direction"] == "shared_external_cause" and abs(b) > 0:
            flags.append(
                f"{f}: the prior says both this feature and volatility respond to a third "
                f"factor. Its coefficient is then an association, not an effect, and the "
                f"omitted factor belongs in your limitations.")
        if pr["causal_direction"] == "no_causal" and abs(b) > 0:
            flags.append(
                f"{f}: the prior sees no causal link, yet the model gives it a coefficient of "
                f"{b:+.3f}. Worth explaining before you rely on it.")

    return {"regime": artifact["regime_names"][r_i], "comparison": rows, "flags": flags,
            "_how_to_read": (
                "Flags are prompts for a paragraph, not errors. The prior is not evidence - "
                "it is a structured way of noticing where your model is claiming something "
                "the literature would not expect, so that you argue for it explicitly.")}


if __name__ == "__main__":
    import sys
    print(json.dumps(elicit_priors(sys.argv[2:], target=sys.argv[1]), indent=2)[:4000])

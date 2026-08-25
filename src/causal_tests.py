"""
Tests for separating causation from association. SUPPLIED - do not modify.

WHAT THIS IS FOR
    The supplied model tells you which features predict the policy cycle. It cannot tell you
    why. A feature can dominate the importances for at least four reasons:

        CAUSAL        the feature moves policy
        REVERSE       policy (or the anticipation of it) moves the feature
        CONFOUNDED    something else moves both
        PROXY         the feature stands in for a variable doing one of the above

    An LLM asked to explain the model will produce a fluent mechanism for any of the four,
    and will not signal which it thinks it is giving you. These four tests are how you find
    out. None of them PROVES causation - nothing available here could. They rule things out.

HOW YOU USE THIS
    1. In the ChatGPT interface, with the importance table and partial-dependence plots
       attached, get the model to propose causal mechanisms for the top features.
    2. Transcribe each proposal into a CausalClaim.
    3. Run test_claim() and see whether the data supports it.
    4. Report what survived, what was refuted, and what could not be tested.

    All three outcomes earn marks. An untestable claim, labelled as untestable, is a correct
    answer. An untested claim reported as a mechanism is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import config

TARGET = "y_cycle"


@dataclass
class CausalClaim:
    """
    One causal proposal, as the LLM stated it. Transcribe faithfully - including the ones
    you expect to fail, because a refuted claim is evidence and a missing one is not.
    """
    feature: str
    mechanism: str                          # the causal story, in one sentence
    claimed: str = "causal"                 # causal | reverse | confounded | proxy
    confounders: list[str] = field(default_factory=list)   # for the conditioning test
    source: str = "llm"                     # llm | ours
    note: str = ""


# ---------------------------------------------------------------------------

def lead_lag(panel: pd.DataFrame, feature: str, target: str = TARGET,
             max_lag: int = 6) -> dict:
    """
    Does the feature LEAD the policy cycle, or LAG it?

    At lag L the feature at meeting t is correlated with the target at meeting t+L, so a
    POSITIVE peak lag means the feature moves FIRST and a negative peak means it moves after.
    (An earlier version had these labels the wrong way round; `tests/test_causal_tests.py`
    pins the convention with a synthetic series that leads by exactly two.)

    A causal driver should peak at a positive lag. A feature peaking at zero or a negative
    lag is moving with or after policy, which is the signature of response or anticipation
    rather than causation.

    Caveat you must respect: `y_cycle` is itself forward-looking - it describes the 182 days
    AFTER each meeting. So a feature that genuinely anticipates policy will already correlate
    at lag 0. Read the SHAPE of the profile, not just the peak.
    """
    d = panel[[feature, target]].dropna()
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = d[target].shift(-lag)
        both = d[feature].notna() & shifted.notna()
        if both.sum() > 30:
            out[lag] = float(np.corrcoef(d[feature][both], shifted[both])[0, 1])
    peak = max(out, key=lambda k: abs(out[k])) if out else None
    return {"correlation_by_lag": {k: round(v, 4) for k, v in out.items()},
            "peak_lag": peak, "peak_correlation": round(out[peak], 4) if peak is not None else None,
            "reads_as": ("leads policy" if peak is not None and peak > 0 else
                         "moves with policy" if peak == 0 else
                         "lags policy" if peak is not None else "indeterminate")}


def reverse_regression(panel: pd.DataFrame, feature: str) -> dict:
    """
    Run the arrow both ways.

    FORWARD:  does the feature at t predict the policy cycle from t?
    BACKWARD: does the rate change already taken at t explain the feature at t?

    If backward is much stronger, the feature is largely a record of policy rather than a
    cause of it. These are not symmetric quantities and this is not a formal test - it is a
    direction-of-fit check and must be described as one.
    """
    f = panel[[feature, TARGET]].dropna()
    Xf = f[[feature]].to_numpy(float)
    r2_fwd = r2_score(f[TARGET], LinearRegression().fit(Xf, f[TARGET]).predict(Xf))

    b = panel[["decision", feature]].dropna()
    Xb = pd.get_dummies(b["decision"], prefix="d").to_numpy(float)
    r2_back = r2_score(b[feature], LinearRegression().fit(Xb, b[feature]).predict(Xb))

    ratio = (r2_back / r2_fwd) if r2_fwd > 1e-6 else None
    return {"r2_forward_feature_predicts_cycle": round(float(r2_fwd), 4),
            "r2_backward_decision_explains_feature": round(float(r2_back), 4),
            "backward_over_forward": round(ratio, 2) if ratio else None,
            "reads_as": ("mostly a record of policy" if ratio and ratio > 2
                         else "mostly forward-looking" if ratio and ratio < 0.5
                         else "both directions comparable")}


def conditional_association(panel: pd.DataFrame, feature: str,
                            confounders: list[str]) -> dict:
    """
    Does the association survive controlling for a candidate common cause?

    Regresses both the feature and the target on the confounders, then correlates the
    residuals. If the association collapses, the confounders were driving both and the
    original correlation was not evidence of a direct link.

    Choosing the confounders is YOUR judgement and it is what makes this test informative.
    For the RBA the obvious candidates are the things the Board is reacting to - underlying
    inflation and the unemployment rate.
    """
    cols = [feature, TARGET] + [c for c in confounders if c in panel.columns]
    d = panel[cols].dropna()
    if len(d) < 40 or len(cols) < 3:
        return {"error": "insufficient data or no valid confounders"}
    Z = d[[c for c in cols[2:]]].to_numpy(float)
    raw = float(np.corrcoef(d[feature], d[TARGET])[0, 1])
    rf = d[feature] - LinearRegression().fit(Z, d[feature]).predict(Z)
    rt = d[TARGET] - LinearRegression().fit(Z, d[TARGET]).predict(Z)
    partial = float(np.corrcoef(rf, rt)[0, 1])
    shrink = 1 - abs(partial) / abs(raw) if abs(raw) > 1e-6 else np.nan
    return {"raw_correlation": round(raw, 4),
            "partial_correlation": round(partial, 4),
            "conditioned_on": cols[2:],
            "shrinkage": round(float(shrink), 3),
            "reads_as": ("association largely explained by the confounders"
                         if shrink > 0.5 else
                         "association survives conditioning" if shrink < 0.2 else
                         "association partly explained")}


def subperiod_stability(panel: pd.DataFrame, feature: str) -> dict:
    """
    A causal mechanism should hold across periods. A spurious one often will not.

    Weak evidence on its own - regimes genuinely change, and a real mechanism can be
    suspended (the 2020-21 period had a policy floor, so rate-sensitive mechanisms could not
    operate). Use it to qualify a claim, not to settle one.
    """
    periods = {"2006-2013": (None, "2013-12-31"),
               "2014-2019": ("2014-01-01", "2019-12-31"),
               "2020-2026": ("2020-01-01", None)}
    out = {}
    for name, (a, b) in periods.items():
        d = panel.loc[a:b, [feature, TARGET]].dropna()
        if len(d) > 25:
            out[name] = round(float(np.corrcoef(d[feature], d[TARGET])[0, 1]), 4)
    signs = {np.sign(v) for v in out.values()}
    return {"correlation_by_period": out,
            "sign_stable": len(signs) <= 1,
            "reads_as": ("stable across periods" if len(signs) <= 1
                         else "SIGN FLIPS between periods")}


# ---------------------------------------------------------------------------

def test_claim(panel: pd.DataFrame, claim: CausalClaim) -> dict:
    """Run all four tests on one claim and return the evidence, unjudged."""
    if claim.feature not in panel.columns:
        return {"feature": claim.feature, "error": "not a panel column"}
    return {
        "feature": claim.feature,
        "claimed": claim.claimed,
        "mechanism": claim.mechanism,
        "lead_lag": lead_lag(panel, claim.feature),
        "reverse_regression": reverse_regression(panel, claim.feature),
        "conditional": conditional_association(panel, claim.feature, claim.confounders),
        "subperiod": subperiod_stability(panel, claim.feature),
    }


def report(result: dict) -> None:
    """Print one claim's evidence in the form the brief asks you to report it."""
    if "error" in result:
        print(f"  {result['feature']}: {result['error']}")
        return
    print(f"\n  {result['feature']}  (claimed: {result['claimed']})")
    print(f"    mechanism: {result['mechanism'][:88]}")
    ll, rr = result["lead_lag"], result["reverse_regression"]
    print(f"    lead/lag        peak at {ll['peak_lag']:+d} meetings, "
          f"r={ll['peak_correlation']:+.3f}  -> {ll['reads_as']}")
    print(f"    direction       fwd R2={rr['r2_forward_feature_predicts_cycle']:.4f}  "
          f"back R2={rr['r2_backward_decision_explains_feature']:.4f}  "
          f"-> {rr['reads_as']}")
    c = result["conditional"]
    if "error" not in c:
        print(f"    conditioning    r {c['raw_correlation']:+.3f} -> "
              f"{c['partial_correlation']:+.3f} (shrink {c['shrinkage']:.0%})  "
              f"-> {c['reads_as']}")
    else:
        print(f"    conditioning    {c['error']} - name confounders in the claim")
    sp = result["subperiod"]
    print(f"    stability       {sp['correlation_by_period']}  -> {sp['reads_as']}")

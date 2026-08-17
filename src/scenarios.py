"""
Geopolitical scenarios — SHARED WORK

LEAD: <put a name here>

PURPOSE
    Generate a candidate pool of scenarios anchored in the external source library,
    have the model rank them for relevance to Australia, select three with DIFFERENT
    transmission routes, and push each through the pipeline.

READS
    data/raw/geopolitical/ (GPR index), plus the linked external sources
    data/processed/regimes.parquet  (the response function)

WRITES
    data/processed/scenarios.json  - selected AND rejected, with the model's reasons
"""
from __future__ import annotations
import config

# The five transmission routes from section 8.4 Rule 3 of the brief.
# Your three selected scenarios must use three DIFFERENT routes.
TRANSMISSION_ROUTES = [
    "commodity_prices",
    "supply_chain_inflation",
    "currency_and_capital_flows",
    "confidence_and_demand",
    "labour_market",
]


def generate_pool():
    """Generate candidates anchored in the external sources. NOT from RBA documents."""
    raise NotImplementedError


def rank_relevance(pool):
    """Ask the model to rank relevance to Australia. Your job is to CHECK this ranking."""
    raise NotImplementedError


def pool_diversity(pool) -> float:
    """Mean pairwise embedding distance across the pool. Report it whatever it says."""
    raise NotImplementedError


def scenario_to_features(scenario: dict) -> dict:
    """YOURS. Map a scenario onto the model's exogenous features.

    This is the judgement step, and it is what Rule 6 is really asking for. A scenario is a
    story; your model takes numbers. You must say what this story does to each construct.

    Return a dict of {feature_name: value_on_its_own_scale}, e.g.

        {"financial_conditions_concern": 0.75, "global_risk_salience": 0.80, ...}

    Ground it. The defensible way is to look up what the constructs ACTUALLY did during the
    closest historical analogue (see src/historical_events.py), rather than picking numbers
    that feel severe. Say in your report which analogue you used and why.

    Every feature in stage4.ALL_FEATURES must appear, or predict_regime_under() will tell you
    which are missing.
    """
    raise NotImplementedError


# =============================================================================================
# PRE-BUILT BELOW THIS LINE - this is the part that runs your scenario through the real model
# =============================================================================================

def predict_regime_under(feature_values: dict, horizon_days: int = 63) -> dict:
    """Push a scenario through the FITTED regime model and return predicted regime risk.

    This is what makes section 8.4 Rule 6 true. Multiplying a historical volatility figure by
    a severity factor you chose is not running a scenario through your model - it is asserting
    the answer. Here the fitted coefficients do the work.

    How it works:
      1. Refit the primary specification (same endog, same features as stage 4).
      2. Substitute YOUR scenario's feature values into the mean equation, giving a predicted
         log realised volatility for each regime.
      3. Propagate today's filtered regime distribution forward `horizon_days` using the
         fitted transition matrix.
      4. Report the predicted volatility against the same model at average feature values.

    The scenario acts on the CONDITIONAL MEAN within each regime, not on the regime
    probabilities themselves, so the returned regime distribution is the transition matrix
    propagated forward and is the same for every scenario. Do not present it as a
    scenario-specific probability - see the note further down.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from src.stage4_regime_model import ALL_FEATURES, align, fit

    market = pd.read_parquet(config.DATA_PROCESSED / "market_data.parquet").set_index("date")
    scores = pd.read_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet")
    df = align(market, scores).dropna(subset=["ret", "rv_21"] + ALL_FEATURES)

    missing = [f for f in ALL_FEATURES if f not in feature_values]
    if missing:
        raise ValueError(f"scenario_to_features() did not set: {missing}")

    endog = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
             else (df["ret"] * 100).values)
    X = df[ALL_FEATURES].values
    res = fit(endog, X, config.N_REGIMES, "scenario transmission")

    k = config.N_REGIMES
    pmap = dict(zip(res.model.param_names, np.asarray(res.params)))
    x_star = np.array([feature_values[f] for f in ALL_FEATURES], dtype=float)

    # Predicted mean per regime under the scenario's feature values.
    #
    # statsmodels names exogenous coefficients POSITIONALLY - x1, x2, ... in the column order
    # of the exog matrix - not by feature name. Looking them up by feature name silently
    # returns zero and makes every scenario produce the same answer, which looks plausible.
    # ALL_FEATURES order is the exog column order, so x{i+1} is ALL_FEATURES[i].
    mu = []
    for r in range(k):
        c = pmap.get(f"const[{r}]", pmap.get("const", 0.0))
        beta = np.array([pmap.get(f"x{i+1}[{r}]", pmap.get(f"x{i+1}", 0.0))
                         for i in range(len(ALL_FEATURES))])
        mu.append(float(c + beta @ x_star))
    mu = np.array(mu)
    sigma = np.sqrt([pmap.get(f"sigma2[{r}]", 1.0) for r in range(k)])

    # Today's regime distribution, propagated forward by the transition matrix.
    P = np.asarray(res.regime_transition).reshape(k, k)
    if not np.allclose(P.sum(axis=0), 1.0):
        P = P.T
    p_now = np.asarray(res.filtered_marginal_probabilities)[-1]
    steps = max(1, horizon_days // 21)
    p_h = p_now.copy()
    for _ in range(steps):
        p_h = P @ p_h
    p_h = p_h / p_h.sum()

    # WHAT NOT TO DO HERE. An earlier version derived a scenario-implied volatility from the
    # regime means, then reweighted the regimes by how close each mean was to that value. That
    # is circular - the weights come from the thing they are supposed to explain - and it
    # produced P(worst regime) of 99-100% for every scenario, which is not a finding, it is an
    # artefact. If a scenario tool reports near-certainty for everything, suspect the maths.
    #
    # What the model can honestly say: the regime distribution comes from the transition
    # matrix, and the scenario shifts the CONDITIONAL MEAN within each regime. So report the
    # predicted volatility under the scenario against the same quantity at baseline features,
    # and leave the regime distribution as the transition matrix gives it.
    y_star = float(mu @ p_h)
    mu_base = np.array([pmap.get(f"const[{r}]", pmap.get("const", 0.0))
                        + np.array([pmap.get(f"x{i+1}[{r}]", 0.0)
                                    for i in range(len(ALL_FEATURES))]) @ X.mean(axis=0)
                        for r in range(k)])
    y_base = float(mu_base @ p_h)

    ann_vol = float(np.exp(y_star)) if config.REGIME_ENDOG == "log_rv" else float("nan")
    ann_vol_base = float(np.exp(y_base)) if config.REGIME_ENDOG == "log_rv" else float("nan")
    post = p_h
    names = [config.REGIME_NAMES[i] for i in range(k)]
    base_worst = float((df.index.size and
                        (pd.read_parquet(config.DATA_PROCESSED / "regimes.parquet")["regime"]
                         == k - 1).mean()))
    return {
        "horizon_days": horizon_days,
        "feature_values": feature_values,
        "predicted_ann_vol": ann_vol,
        "baseline_ann_vol": ann_vol_base,
        "vol_multiple_vs_baseline": (float(ann_vol / ann_vol_base)
                                     if ann_vol_base else float("nan")),
        "regime_probabilities_from_transition_matrix": dict(
            zip(names, [float(v) for v in post])),
        "p_worst_regime": float(post[-1]),
        "p_worst_unconditional": base_worst,
        "note": ("Volatility predicted from the fitted model under your scenario's feature "
                 "values, against the same model at average features. The regime distribution "
                 "is the transition matrix propagated forward and does NOT depend on the "
                 "scenario - the scenario acts on the conditional mean, not on the regime "
                 "probabilities. Do not present it as a scenario-specific probability."),
    }


# HINTS
#   - Rule 1: scenarios come from the external library, never from RBA minutes.
#     RBA minutes are the RESPONSE function, not the scenario source.
#   - Rule 3 is a hard requirement. Three supply-shock scenarios are one scenario.
#   - Save the REJECTED candidates too. They are evidence of the search.

if __name__ == "__main__":
    pool = generate_pool()
    print(f"pool size: {len(pool)}  diversity: {pool_diversity(pool):.3f}")

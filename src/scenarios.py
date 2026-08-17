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
    data/processed/scenarios_final.json  - selected AND rejected, with the model's reasons,
                                           pool diversity, ranking stability and transmission
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
      1. Load the model stage 4 fitted and ordered (`model_artifact.json`). No refitting -
         a second fit can land on a different optimum from the one your report describes.
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
    from src.stage4_regime_model import (ALL_FEATURES, align, load_model_artifact,
                                         predicted_log_vol, propagate)

    # Load the model stage 4 already fitted and ordered. DO NOT refit here. An earlier
    # version called fit() again and then indexed the result as though raw state k-1 meant
    # "stressed". Raw Markov state labels are arbitrary between fits, so that is wrong most
    # of the time, and a second fit can land on a different optimum from the one your report
    # describes. See build_model_artifact() in stage 4.
    art = load_model_artifact()
    k = art["n_regimes"]

    market = pd.read_parquet(config.DATA_PROCESSED / "market_data.parquet").set_index("date")
    scores = pd.read_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet")
    df = align(market, scores).dropna(subset=["ret", "rv_21"] + ALL_FEATURES)

    missing = [f for f in ALL_FEATURES if f not in feature_values]
    if missing:
        raise ValueError(f"scenario_to_features() did not set: {missing}")

    X = df[ALL_FEATURES].values

    # Conditional mean per ORDERED regime, under the scenario's feature values and at the
    # sample average, from the saved coefficients.
    mu = predicted_log_vol(art, feature_values)
    mu_base = predicted_log_vol(art, dict(zip(ALL_FEATURES, X.mean(axis=0))))

    # Today's regime distribution, propagated forward by the DAILY transition matrix.
    # 63 trading days means 63 applications of it - see stage4.propagate().
    P = np.array(art["transition_daily_col_from_row_to"])
    p_h = propagate(P, np.array(art["filtered_last"]), horizon_days)

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
    y_base = float(mu_base @ p_h)

    ann_vol = float(np.exp(y_star)) if config.REGIME_ENDOG == "log_rv" else float("nan")
    ann_vol_base = float(np.exp(y_base)) if config.REGIME_ENDOG == "log_rv" else float("nan")
    names = art["regime_names"]
    base_worst = float((pd.read_parquet(config.DATA_PROCESSED / "regimes.parquet")["regime"]
                        == k - 1).mean())

    return {
        "horizon_days": horizon_days,
        "transition_steps_applied": horizon_days,
        "feature_values": feature_values,

        # ---- MODEL-IMPLIED, and conditional on your scenario -------------------------
        "predicted_ann_vol": ann_vol,
        "baseline_ann_vol": ann_vol_base,
        "vol_multiple_vs_baseline": (float(ann_vol / ann_vol_base)
                                     if ann_vol_base else float("nan")),

        # ---- MODEL-IMPLIED, but NOT conditional on your scenario ---------------------
        # These come from the transition matrix alone. They are identical for every
        # scenario you run. Reporting them as "the model gives this scenario an X%
        # chance of stress" is false, and it is the most common way this module is
        # misused - see evidence_types below.
        "regime_probabilities_from_transition_matrix": dict(
            zip(names, [float(v) for v in p_h])),
        "p_worst_regime_unconditional_forecast": float(p_h[-1]),
        "p_worst_long_run_base_rate": base_worst,

        "evidence_types": {
            "predicted_ann_vol": "MODEL-IMPLIED, CONDITIONAL on the scenario's features",
            "regime_probabilities_from_transition_matrix":
                "MODEL-IMPLIED, NOT conditional on the scenario - same for every scenario",
            "p_worst_long_run_base_rate": "EMPIRICAL, observed share of days in the sample",
        },
        "note": (f"Volatility is predicted from the fitted model under your scenario's "
                 f"feature values, against the same model at average features. The regime "
                 f"distribution is today's filtered state propagated {horizon_days} trading "
                 f"days through the DAILY transition matrix; it does NOT depend on the "
                 f"scenario, because the scenario acts on the conditional mean rather than "
                 f"on the regime probabilities. Do not describe it as a scenario-specific "
                 f"probability. If you want scenario-conditional regime probabilities you "
                 f"have to build a model whose transitions depend on the features, and say "
                 f"that you did."),
    }


def select_three(ranked: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """YOURS. Take the three highest-ranked scenarios that use three DIFFERENT routes.

    Return (selected, rejected, conflict_record). The conflict record must say what the naive
    top three would have been and how much cumulative relevance route diversity cost, because
    section 8.4 Rule 6 asks you to report that trade-off rather than hide it.
    """
    raise NotImplementedError


def ranking_stability(pool: list[dict], n: int = 3) -> dict:
    """YOURS. Re-rank the SAME pool several times and report how stable the order is.

    The pool is fixed. Only the ranking call is repeated - different seeds, and at least one
    reworded prompt. Regenerating the pool each time measures generation variance, which is a
    different question and does not tell you whether your ranking is trustworthy.

    Report at least: top-three overlap across runs, and whether the route set is stable.
    """
    raise NotImplementedError


def run() -> dict:
    """Orchestrate the whole scenario phase. `run_pipeline.py --scenarios` calls this.

    generate -> rank (separately) -> check ranking stability -> select three with different
    routes -> map each to features -> push through the fitted model -> save everything.

    WRITES data/processed/scenarios_final.json
    """
    import json

    pool = generate_pool()
    print(f"  pool of {len(pool)}")

    ranked = rank_relevance(pool)
    div = pool_diversity(pool)
    stab = ranking_stability(pool)
    print(f"  pool diversity {div}")
    print(f"  ranking stability {stab}")

    chosen, rejected, conflict = select_three(ranked)
    routes = [s.get("transmission_route") for s in chosen]
    if len(set(routes)) != len(routes):
        raise SystemExit(
            f"\nSTOP: your three scenarios use routes {routes}.\n"
            "Rule 3 is a hard requirement - three scenarios on two routes test two things.\n")

    for s in chosen:
        s["model_transmission"] = predict_regime_under(scenario_to_features(s))

    out = {"pool": pool, "ranked": ranked, "selected": chosen, "rejected": rejected,
           "route_conflict": conflict, "pool_diversity": div, "ranking_stability": stab}
    json.dump(out, open(config.DATA_PROCESSED / "scenarios_final.json", "w"), indent=2)
    print(f"  wrote scenarios_final.json: {len(chosen)} selected, {len(rejected)} rejected")
    return out


# HINTS
#   - Rule 1: scenarios come from the external library, never from RBA minutes.
#     RBA minutes are the RESPONSE function, not the scenario source.
#   - Rule 3 is a hard requirement. Three supply-shock scenarios are one scenario.
#   - Save the REJECTED candidates too. They are evidence of the search.

if __name__ == "__main__":
    run()

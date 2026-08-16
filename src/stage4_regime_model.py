"""
Stage 4 — Regime model

OWNER: <put your name here>

PRE-BUILT: the fitting, regime ordering, per-regime risk metrics, and the specification
comparison. YOURS: which text features to include, and interpreting what you find.

THE SPECIFICATION POINT - read this before you change anything
    MarkovRegression puts exogenous variables in the MEAN equation.

    If your dependent variable is RETURNS, the text features are being asked to predict the
    DIRECTION of tomorrow's return. They cannot, and they should not be able to. Text will
    look useless and you will conclude, wrongly, that RBA communication says nothing about
    volatility.

    If your dependent variable is LOG REALISED VOLATILITY, the mean equation IS the
    volatility level, and the text features can explain it. On our data this is the
    difference between text making AIC 46 points WORSE and 1,000 points BETTER.

    config.REGIME_ENDOG controls this. Both are provided so you can see the difference.

WRITES data/processed/regimes.parquet, data/processed/regime_summary.json
"""
from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config

warnings.filterwarnings("ignore")

# YOUR CHOICE: which risk-voice features enter the model, and why.
TEXT_FEATURES = ["financial_conditions_concern", "downside_risk_emphasis",
                 "global_risk_salience", "vigilance", "uncertainty_language"]


def align(market: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Scores are per meeting; market data is daily. Forward-fill from the meeting date.

    This assumes a reading holds until the next meeting. An announcement-day-only
    specification is a defensible alternative - if you use it, say so.
    """
    df = market.copy()
    s = scores.set_index("meeting_date")
    for c in [c for c in s.columns if not c.endswith("_sd") and c != "n_calls_valid"]:
        df[c] = s[c].reindex(df.index, method="ffill")
    return df


def fit(y, exog, label):
    mod = sm.tsa.MarkovRegression(y, k_regimes=config.N_REGIMES, trend="c",
                                  exog=exog, switching_variance=True)
    res = mod.fit(em_iter=20, search_reps=8, maxiter=200, disp=False)
    print(f"  {label:46s} aic={res.aic:10.1f} conv={res.mle_retvals.get('converged')}")
    return res


def per_regime_risk(ret: pd.Series, regime: pd.Series) -> dict:
    out = {}
    for r in sorted(regime.dropna().unique()):
        x = ret[regime == r].dropna()
        if len(x) < 20:
            continue
        v95 = float(np.percentile(x, 5))
        out[config.REGIME_NAMES[int(r)]] = {
            "n_days": int(len(x)), "share": float(len(x) / len(ret.dropna())),
            "ann_vol": float(x.std() * np.sqrt(252)),
            "VaR_95_daily": v95, "ES_95_daily": float(x[x <= v95].mean()),
            "worst_day": float(x.min()),
        }
    return out


def run() -> pd.DataFrame:
    market = pd.read_parquet(config.DATA_PROCESSED / "market_data.parquet").set_index("date")
    scores = pd.read_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet")
    df = align(market, scores).dropna(subset=["ret", "rv_21"] + TEXT_FEATURES)

    endog = np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv" else (df["ret"] * 100).values
    X = df[TEXT_FEATURES].values

    base = fit(endog, None, f"baseline ({config.REGIME_ENDOG}, no text)")
    withx = fit(endog, X, f"with risk-voice text ({len(TEXT_FEATURES)} features)")

    pmap = dict(zip(base.model.param_names, np.asarray(base.params)))
    sig = np.sqrt([pmap[f"sigma2[{i}]"] for i in range(config.N_REGIMES)])
    order = np.argsort(sig)
    remap = {int(o): i for i, o in enumerate(order)}

    probs = pd.DataFrame(base.smoothed_marginal_probabilities, index=df.index)
    probs.columns = [remap[c] for c in probs.columns]
    probs = probs[sorted(probs.columns)]
    probs.columns = [f"p_{config.REGIME_NAMES[c]}" for c in probs.columns]

    out = df.join(probs)
    out["regime"] = probs.values.argmax(axis=1)
    out["regime_confidence"] = probs.values.max(axis=1)

    risk = per_regime_risk(df["ret"], out["regime"])
    summary = {
        "endog": config.REGIME_ENDOG,
        "text_features": TEXT_FEATURES,
        "aic_no_text": float(base.aic), "aic_with_text": float(withx.aic),
        "text_improves_aic": bool(withx.aic < base.aic),
        "aic_gain_from_text": float(base.aic - withx.aic),
        "per_regime_risk": risk,
        "expected_duration_days": {config.REGIME_NAMES[remap[i]]: float(base.expected_durations[i])
                                   for i in range(config.N_REGIMES)},
    }
    json.dump(summary, open(config.DATA_PROCESSED / "regime_summary.json", "w"), indent=2)
    out.reset_index().to_parquet(config.DATA_PROCESSED / "regimes.parquet", index=False)

    print()
    for k, v in risk.items():
        print(f"    {k:10s} {v['share']:5.1%} of days   vol {v['ann_vol']:6.1%}   "
              f"ES95 {v['ES_95_daily']:+.2%}")
    print(f"  text improves AIC: {summary['text_improves_aic']} "
          f"(gain {summary['aic_gain_from_text']:+.1f})")
    return out


if __name__ == "__main__":
    run()

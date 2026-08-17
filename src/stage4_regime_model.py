"""
Stage 4 — Regime model

OWNER: <put your name here>

PRE-BUILT: fitting with fallbacks, regime ordering, per-regime risk metrics, and the
           3-vs-4 regime and text-vs-no-text comparisons.
YOURS:     which text features to include, which dependent variable, and interpretation.

TWO SPECIFICATION POINTS - read both before changing anything
-------------------------------------------------------------------------------------------
1. THE DEPENDENT VARIABLE

   MarkovRegression puts exogenous variables in the MEAN equation.

   endog = returns   -> your text features are asked to predict the DIRECTION of tomorrow's
                        return. They cannot, and should not be able to. Text looks useless.
   endog = log_rv    -> the mean equation IS the volatility level, so text can explain it.

   On our data that is the difference between text making AIC ~46 points WORSE and several
   hundred points BETTER. Same text, same model, same data.

2. HOW MANY REGIMES

   The reference implementation uses 3. Australian data prefers 4.

   This module fits BOTH, deliberately:
     - the 3-regime fit is your LIKE-FOR-LIKE comparison against the reference
     - the 4-regime fit is your own finding about Australian data

   Do not silently switch to 4 and then compare regime shares against the reference's three.
   They are not comparable. Report the 3-regime numbers when comparing, and the 4-regime
   numbers when describing Australia.

WRITES data/processed/regimes.parquet          (primary fit, config.N_REGIMES)
       data/processed/regimes_3.parquet        (3-regime fit, for the reference comparison)
       data/processed/regime_summary.json
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config

warnings.filterwarnings("ignore")

# YOUR CHOICE: which risk-voice constructs enter the model, and why.
TEXT_FEATURES = ["financial_conditions_concern", "downside_risk_emphasis",
                 "global_risk_salience", "vigilance", "uncertainty_language"]

NAMES_3 = {0: "low", 1: "medium", 2: "high"}


def align(market: pd.DataFrame, scores: pd.DataFrame,
          lag_days: int | None = None) -> pd.DataFrame:
    """Scores are per meeting; market data is daily. Forward-fill from the PUBLICATION date.

    LOOK-AHEAD WARNING. The minutes of a meeting are not published on the meeting day - the
    RBA releases them about a fortnight later. Attaching a score to the meeting date gives
    your model information nobody could have had at the time, and any predictive claim built
    on it is invalid.

    config.PUBLICATION_LAG_DAYS shifts each score to the date it could first have been read.
    Set it to 0 only if you are making a RETROSPECTIVE (explanatory) claim, and say so.

    This still assumes a reading holds until the next meeting. An announcement-day-only
    specification is defensible too - if you use it, say so in your report.
    """
    lag = config.PUBLICATION_LAG_DAYS if lag_days is None else lag_days
    df = market.copy()
    s = scores.copy()
    s["available_from"] = pd.to_datetime(s["meeting_date"]) + pd.Timedelta(days=lag)
    s = s.set_index("available_from").sort_index()
    for c in [c for c in s.columns
              if not c.endswith("_sd") and c not in ("n_calls_valid", "meeting_date")]:
        df[c] = s[c].reindex(df.index, method="ffill")
    return df


def fit(y, exog, k, label):
    """Fit with fallbacks. The 4-regime model on log volatility is numerically fragile:
    random start searches can raise 'Could not untransform parameters'."""
    mod = sm.tsa.MarkovRegression(y, k_regimes=k, trend="c", exog=exog,
                                  switching_variance=True)
    for i, kw in enumerate([dict(em_iter=20, search_reps=8, maxiter=200),
                            dict(em_iter=20, search_reps=0, maxiter=200),
                            dict(em_iter=50, search_reps=0, maxiter=500),
                            dict(em_iter=10, search_reps=0, maxiter=100)]):
        try:
            res = mod.fit(disp=False, **kw)
            print(f"  {label:48s} k={k} aic={res.aic:9.1f} "
                  f"conv={res.mle_retvals.get('converged')}"
                  f"{'' if i == 0 else f'  (fallback {i})'}")
            return res
        except Exception:                                    # noqa: BLE001, S112
            continue
    raise RuntimeError(f"{label}: all fitting attempts failed")


def order_and_label(res, df, k, names):
    """Sort regimes calmest -> most turbulent using EMPIRICAL realised volatility.

    Do not sort on a fitted parameter. With endog = log_rv, sigma2 is the variance OF LOG
    VOLATILITY - how erratic volatility is, not how high - and sorting on it mislabels the
    regimes. Empirical volatility per assigned regime is unambiguous for either endog.
    """
    raw = pd.DataFrame(res.smoothed_marginal_probabilities, index=df.index)
    lab = raw.values.argmax(axis=1)
    emp = {c: df["ret"][lab == c].std() for c in range(k)}
    order = sorted(emp, key=lambda c: emp[c])
    remap = {int(o): i for i, o in enumerate(order)}
    probs = raw.rename(columns=remap)
    probs = probs[sorted(probs.columns)]
    probs.columns = [f"p_{names[c]}" for c in probs.columns]
    out = df.join(probs)
    out["regime"] = probs.values.argmax(axis=1)
    out["regime_confidence"] = probs.values.max(axis=1)
    return out, remap


def per_regime_risk(ret: pd.Series, regime: pd.Series, names: dict) -> dict:
    out = {}
    for r in sorted(regime.dropna().unique()):
        x = ret[regime == r].dropna()
        if len(x) < 20:
            continue
        v95 = float(np.percentile(x, 5))
        out[names[int(r)]] = {
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

    print(f"  dependent variable: {config.REGIME_ENDOG}")
    print(f"  publication lag   : {config.PUBLICATION_LAG_DAYS} days "
          f"({'real-time' if config.PUBLICATION_LAG_DAYS else 'RETROSPECTIVE - look-ahead'})")
    print("  regimes.parquet = WITH text; regimes_base.parquet = without")
    base4 = fit(endog, None, 4, "4 regimes, no text")
    text4 = fit(endog, X, 4, f"4 regimes + {len(TEXT_FEATURES)} text features")
    base3 = fit(endog, None, 3, "3 regimes, no text  (reference comparison)")
    text3 = fit(endog, X, 3, f"3 regimes + {len(TEXT_FEATURES)} text features")

    # The PRIMARY saved output is the model WITH the text features. Saving the no-text fit
    # here would mean the dashboard, the explainability work and the scenarios all ran on a
    # model that never saw your constructs - they would be decorative. Both are written so
    # you can compare them, and comparing them is worth reporting.
    out4, _ = order_and_label(text4, df, 4, config.REGIME_NAMES)
    out3, _ = order_and_label(text3, df, 3, NAMES_3)
    out4_base, _ = order_and_label(base4, df, 4, config.REGIME_NAMES)
    out3_base, _ = order_and_label(base3, df, 3, NAMES_3)

    risk4 = per_regime_risk(df["ret"], out4["regime"], config.REGIME_NAMES)
    risk3 = per_regime_risk(df["ret"], out3["regime"], NAMES_3)

    summary = {
        "endog": config.REGIME_ENDOG,
        "text_features": TEXT_FEATURES,
        "four_regimes": {"aic_no_text": float(base4.aic), "aic_with_text": float(text4.aic),
                         "text_gain": float(base4.aic - text4.aic), "risk": risk4},
        "three_regimes_for_reference_comparison": {
            "aic_no_text": float(base3.aic), "aic_with_text": float(text3.aic),
            "text_gain": float(base3.aic - text3.aic), "risk": risk3},
        "four_beats_three_on_aic": bool(base4.aic < base3.aic),
        "aic_gain_from_fourth_regime": float(base3.aic - base4.aic),
    }
    json.dump(summary, open(config.DATA_PROCESSED / "regime_summary.json", "w"), indent=2)
    out4.reset_index().to_parquet(config.DATA_PROCESSED / "regimes.parquet", index=False)
    out3.reset_index().to_parquet(config.DATA_PROCESSED / "regimes_3.parquet", index=False)
    out4_base.reset_index().to_parquet(
        config.DATA_PROCESSED / "regimes_base.parquet", index=False)
    out3_base.reset_index().to_parquet(
        config.DATA_PROCESSED / "regimes_3_base.parquet", index=False)

    print("\n  FOUR REGIMES (your Australian finding)")
    for k, v in risk4.items():
        print(f"    {k:9s} {v['share']:5.1%} of days   vol {v['ann_vol']:6.1%}   "
              f"ES95 {v['ES_95_daily']:+.2%}")
    print("  THREE REGIMES (like-for-like against the reference)")
    for k, v in risk3.items():
        print(f"    {k:9s} {v['share']:5.1%} of days   vol {v['ann_vol']:6.1%}   "
              f"ES95 {v['ES_95_daily']:+.2%}")
    print(f"\n  fourth regime worth it: {summary['four_beats_three_on_aic']} "
          f"(AIC gain {summary['aic_gain_from_fourth_regime']:+.1f})")
    print(f"  text improves 4-regime AIC by {summary['four_regimes']['text_gain']:+.1f}")

    print("\n  WHICH PERIODS EACH REGIME PICKS UP")
    yrs = pd.Series(out4.index.year, index=out4.index)
    for r in sorted(out4["regime"].dropna().unique()):
        top = yrs[out4["regime"] == r].value_counts().head(4).index.tolist()
        print(f"    {config.REGIME_NAMES[int(r)]:10s} most often: "
              f"{', '.join(str(y) for y in sorted(top))}")

    if any(v.startswith("regime_") for v in config.REGIME_NAMES.values()):
        print("\n  NAME YOUR REGIMES. They are still neutral placeholders.")
        print("  Using the figures above - share of days, volatility, ES, and the periods each")
        print("  regime covers - decide what each state actually IS, then rename them in")
        print("  config.py. The naming argument is marked; the exact numbers are not.")
        print("  Ask specifically: does the fourth regime separate two economically different")
        print("  states, or has it just split one state in half?")
    return out4


if __name__ == "__main__":
    run()

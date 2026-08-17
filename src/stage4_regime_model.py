"""
Stage 4 — Regime model

OWNER: <put your name here>

PRE-BUILT: fitting with fallbacks, regime ordering, per-regime risk metrics, and the
           the text-vs-no-text comparison, coefficients, transitions and out-of-sample.
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

   THREE, the same as the reference implementation - see the note in config.py.

   A fourth regime fits Australian data better on AIC, but it competes with the text: extra
   regimes absorb volatility variation your constructs would otherwise have to explain, so
   the measured contribution of your prompts shrinks. Three regimes also converges reliably
   where four does not, and keeps regime shares comparable with the reference.

WRITES data/processed/regimes.parquet       (WITH text - the primary model)
       data/processed/regimes_base.parquet  (without text, for comparison)
       data/processed/regime_summary.json
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import config

# Deliberately NOT warnings.filterwarnings("ignore"). Convergence and Hessian warnings are
# exactly the information you need when a regime model misbehaves. fit() captures them per
# attempt and reports them. Only the noisiest statsmodels repetitions are silenced.
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# YOUR CHOICE: which risk-voice constructs enter the model, and why.
TEXT_FEATURES = ["financial_conditions_concern", "downside_risk_emphasis",
                 "global_risk_salience", "vigilance", "uncertainty_language"]

# YOUR CHOICE: which macro / cross-asset features enter the model, and why.
# These come from stage1 (see config.MACRO_TICKERS). Leave the list empty to fit a text-only
# model. Anything you list here must exist as a column in market_data.parquet.
#
# WARNING WORTH READING BEFORE YOU FILL THIS IN. Realised volatility features (rv_21, vix,
# aud_vol_21) are extremely strong predictors of the volatility regime - far stronger than any
# text construct. Including them will improve fit and will also swamp the text. That is a
# legitimate model and a legitimate finding, but report the text's contribution WITH and
# WITHOUT them, or you cannot say what the RBA language added.
MACRO_FEATURES: list[str] = []

ALL_FEATURES = TEXT_FEATURES + MACRO_FEATURES

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


STARTS = [dict(em_iter=20, search_reps=8, maxiter=200),
          dict(em_iter=20, search_reps=0, maxiter=200),
          dict(em_iter=50, search_reps=0, maxiter=500),
          dict(em_iter=10, search_reps=0, maxiter=100)]


def fit(y, exog, k, label, require_converged: bool = True):
    """Fit from several starts and return the BEST CONVERGED result.

    Two things this deliberately does not do, because both produce numbers that look fine and
    are not:

    1. It does not return the first attempt that failed to raise. This model on log
       volatility is multimodal - different starts land on optima thousands of AIC apart - so
       "the first one that did not crash" is not a model, it is a coin toss. All starts are
       run and the highest log-likelihood CONVERGED fit wins.

    2. It does not hide warnings. Convergence failures and Hessian problems are reported. A
       non-converged fit has an AIC, and that AIC means nothing; reporting it as evidence is
       the mistake this guards against.

    require_converged=False returns the best non-converged fit with a loud warning, for the
    cases where nothing converges and you would rather see something than nothing. If you use
    it, say so in your report and do not draw conclusions from the AIC.
    """
    mod = sm.tsa.MarkovRegression(y, k_regimes=k, trend="c", exog=exog,
                                  switching_variance=True)
    converged, failed, errors = [], [], []
    for i, kw in enumerate(STARTS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = mod.fit(disp=False, **kw)
            except Exception as e:                           # noqa: BLE001
                errors.append(f"start {i}: {str(e)[:80]}")
                continue
        msgs = {str(w.message)[:60] for w in caught}
        (converged if res.mle_retvals.get("converged") else failed).append((res, i, msgs))

    pool = converged or ([] if require_converged else failed)
    if not pool:
        detail = "; ".join(errors) if errors else f"{len(failed)} starts ran but none converged"
        raise RuntimeError(f"{label}: no converged fit ({detail})")

    res, i, msgs = max(pool, key=lambda t: t[0].llf)
    flag = "" if converged else "  *** NOT CONVERGED - AIC IS NOT EVIDENCE ***"
    print(f"  {label:48s} k={k} aic={res.aic:9.1f} conv={bool(converged)}"
          f"  [start {i}, {len(converged)}/{len(STARTS)} converged]{flag}")
    if len(converged) > 1:
        spread = max(t[0].aic for t in converged) - min(t[0].aic for t in converged)
        if spread > 10:
            print(f"      AIC spread across converged starts: {spread:.0f}. The optimum is not "
                  f"well identified - report this as model uncertainty.")
    for m in sorted(msgs)[:2]:
        print(f"      warning: {m}")
    return res


def order_and_label(res, df, k, names):
    """Sort regimes calmest -> most turbulent using EMPIRICAL realised volatility.

    Do not sort on a fitted parameter. With endog = log_rv, sigma2 is the variance OF LOG
    VOLATILITY - how erratic volatility is, not how high - and sorting on it mislabels the
    regimes. Empirical volatility per assigned regime is unambiguous for either endog.
    """
    raw = pd.DataFrame(res.smoothed_marginal_probabilities, index=df.index)
    filt = pd.DataFrame(res.filtered_marginal_probabilities, index=df.index)
    lab = raw.values.argmax(axis=1)
    emp = {c: df["ret"][lab == c].std() for c in range(k)}
    order = sorted(emp, key=lambda c: emp[c])
    remap = {int(o): i for i, o in enumerate(order)}
    probs = raw.rename(columns=remap)
    probs = probs[sorted(probs.columns)]
    probs.columns = [f"p_{names[c]}" for c in probs.columns]

    # SMOOTHED probabilities use the WHOLE sample, including days after the one being
    # labelled. They are the right tool for describing history and the wrong tool for
    # anything presented as a real-time call - on any given day the model did not yet know
    # what came next. FILTERED probabilities use only information up to that day.
    #
    # Both are saved. Use p_* (smoothed) for retrospective description; use pf_* (filtered)
    # for anything you describe as a current or operational reading, and say which you used.
    fprobs = filt.rename(columns=remap)
    fprobs = fprobs[sorted(fprobs.columns)]
    fprobs.columns = [f"pf_{names[c]}" for c in fprobs.columns]

    out = df.join(probs).join(fprobs)
    out["regime"] = probs.values.argmax(axis=1)
    out["regime_confidence"] = probs.values.max(axis=1)
    out["regime_filtered"] = fprobs.values.argmax(axis=1)
    out["regime_filtered_confidence"] = fprobs.values.max(axis=1)
    return out, remap


def per_regime_risk(ret: pd.Series, regime: pd.Series, names: dict) -> dict:
    """Risk metrics per regime. DEFINITIONS, because these terms are used loosely elsewhere:

      horizon        ONE TRADING DAY. Nothing here is scaled to a quarter or a year except
                     ann_vol, which is daily sd x sqrt(252).
      sign           Returns, NOT losses. A bad day is NEGATIVE. VaR_95_daily = -0.031 means
                     "on the worst 5% of days in this regime, the return was -3.1% or worse".
                     If you prefer positive losses in your report, flip the sign once, say so,
                     and do it everywhere.
      VaR_95_daily   The 5th percentile of daily returns in this regime. Empirical, not
                     parametric - no normality assumed.
      ES_95_daily    Mean return CONDITIONAL on being at or below VaR_95_daily. Always <= VaR.
      estimator      Empirical, in-sample, unconditional within regime.

    DO NOT scale these to a quarter with sqrt(63). Regime switching, serial dependence and fat
    tails all break that rule, and it understates quarterly tail risk. If you need a quarterly
    figure, simulate or bootstrap quarterly paths - see the scenarios module.
    """
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
            "_definitions": "1-day horizon; returns not losses (negative = bad); empirical "
                            "5th percentile; ES = mean return at or below VaR",
        }
    return out


def coefficients(res, features: list[str], k: int) -> dict:
    """Which feature moves what, and by how much. AIC alone cannot tell you this.

    statsmodels names exogenous coefficients POSITIONALLY (x1, x2, ...) in exog column order,
    so they are mapped back to feature names here. With endog = log_rv, a coefficient is the
    change in LOG volatility per unit of the feature: exp(beta) - 1 is the proportional change
    in volatility for a full 0->1 move of a construct.
    """
    pmap = dict(zip(res.model.param_names, np.asarray(res.params)))
    try:
        se = dict(zip(res.model.param_names, np.asarray(res.bse)))
    except Exception:                                        # noqa: BLE001
        se = {}
    out = {}
    for r in range(k):
        reg = {"const": pmap.get(f"const[{r}]", pmap.get("const")),
               "sigma2": pmap.get(f"sigma2[{r}]")}
        for i, f in enumerate(features):
            b = pmap.get(f"x{i+1}[{r}]", pmap.get(f"x{i+1}"))
            if b is None:
                continue
            reg[f] = {"coef": float(b),
                      "se": float(se.get(f"x{i+1}[{r}]", float("nan"))),
                      "pct_change_in_vol_per_unit":
                          float(np.exp(b) - 1) if config.REGIME_ENDOG == "log_rv" else None}
        out[f"regime_{r}"] = reg
    return out


def rolling_origin_eval(df: pd.DataFrame, features: list[str], k: int,
                        n_folds: int = 4, min_train: float = 0.5) -> dict:
    """Blocked rolling-origin out-of-sample evaluation.

    Everything else in this module is IN-SAMPLE. AIC rewards fit on the data you fitted to,
    and with a flexible regime model that is a low bar. This is the only number here that
    says whether the features help on data the model has not seen.

    Expanding window: train on the first 50%, score the next block, extend, repeat. No
    shuffling and no random splits - the data is a time series and a random split leaks the
    future into the past.

    Scored on MEAN LOG SCORE (higher is better) of the one-step-ahead predictive density,
    which is comparable across specifications with the SAME dependent variable. It is NOT
    comparable across different dependent variables - that is the AIC trap in reverse.
    """
    endog_all = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
                 else (df["ret"] * 100).values)
    X_all = df[features].values if features else None
    n = len(endog_all)
    start = int(n * min_train)
    edges = np.linspace(start, n, n_folds + 1).astype(int)

    scores = {"with_features": [], "without_features": []}
    for i in range(n_folds):
        tr_end, te_end = edges[i], edges[i + 1]
        if te_end - tr_end < 40:
            continue
        for label, X in (("with_features", X_all), ("without_features", None)):
            try:
                r = fit(endog_all[:tr_end], None if X is None else X[:tr_end], k,
                        f"OOS fold {i+1} {label}", require_converged=False)
                pmap = dict(zip(r.model.param_names, np.asarray(r.params)))
                mu = np.array([pmap.get(f"const[{j}]", 0.0) for j in range(k)])
                sig = np.sqrt([max(pmap.get(f"sigma2[{j}]", 1.0), 1e-9) for j in range(k)])
                if X is not None:
                    beta = np.array([[pmap.get(f"x{c+1}[{j}]", 0.0)
                                      for c in range(X.shape[1])] for j in range(k)])
                    mu_t = mu[None, :] + X[tr_end:te_end] @ beta.T
                else:
                    mu_t = np.tile(mu, (te_end - tr_end, 1))
                p = np.asarray(r.filtered_marginal_probabilities)[-1]
                y = endog_all[tr_end:te_end][:, None]
                dens = np.exp(-0.5 * ((y - mu_t) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
                mix = np.maximum((dens * p).sum(axis=1), 1e-300)
                scores[label].append(float(np.log(mix).mean()))
            except Exception as e:                           # noqa: BLE001
                print(f"      OOS fold {i+1} {label} failed: {str(e)[:60]}")

    out = {k2: (float(np.mean(v)) if v else None) for k2, v in scores.items()}
    if out["with_features"] is not None and out["without_features"] is not None:
        out["gain_from_features"] = out["with_features"] - out["without_features"]
        out["features_help_out_of_sample"] = bool(out["gain_from_features"] > 0)
    out["n_folds_scored"] = len(scores["with_features"])
    out["metric"] = ("mean one-step-ahead predictive log score, higher is better; comparable "
                     "only within one dependent variable")
    return out


def transition_matrix(res, k: int) -> dict:
    """Transition probabilities and expected durations, in days."""
    P = np.asarray(res.regime_transition).reshape(k, k)
    if not np.allclose(P.sum(axis=0), 1.0):
        P = P.T
    stay = np.diag(P)
    return {"matrix_col_from_row_to": P.tolist(),
            "expected_duration_days": [float(1.0 / max(1e-9, 1 - s)) for s in stay]}


def run() -> pd.DataFrame:
    market = pd.read_parquet(config.DATA_PROCESSED / "market_data.parquet").set_index("date")
    scores = pd.read_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet")
    missing = [c for c in MACRO_FEATURES if c not in market.columns]
    if missing:
        avail = sorted(c for c in market.columns if c != "close")
        raise SystemExit(
            f"\nMACRO_FEATURES names columns stage1 did not produce: {missing}\n"
            f"Add the series to config.MACRO_TICKERS and re-run stage 1, or remove them here.\n"
            f"Available: {avail}\n")
    aligned = align(market, scores)
    df = aligned.dropna(subset=["ret", "rv_21"] + ALL_FEATURES)
    lost = len(aligned.dropna(subset=["ret", "rv_21"])) - len(df)
    if lost:
        # A feature with a short history costs you rows SILENTLY. Some macro series start
        # years after the ASX data, and dropna() removes every day they are missing.
        per = {c: int(aligned[c].isna().sum()) for c in ALL_FEATURES if aligned[c].isna().any()}
        print(f"  WARNING: {lost} rows dropped for missing features "
              f"({lost/max(len(aligned), 1):.0%} of the sample)")
        for c, n in sorted(per.items(), key=lambda x: -x[1])[:4]:
            print(f"    {c}: {n} missing")

    endog = np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv" else (df["ret"] * 100).values
    X = df[ALL_FEATURES].values

    print(f"  dependent variable: {config.REGIME_ENDOG}")
    print(f"  features          : {len(TEXT_FEATURES)} text + {len(MACRO_FEATURES)} macro"
          f"{' (text only)' if not MACRO_FEATURES else ''}")
    print(f"  publication lag   : {config.PUBLICATION_LAG_DAYS} days "
          f"({'real-time' if config.PUBLICATION_LAG_DAYS else 'RETROSPECTIVE - look-ahead'})")
    print("  regimes.parquet = WITH text; regimes_base.parquet = without")
    k = config.N_REGIMES
    base = fit(endog, None, k, f"{k} regimes, no text")
    text = fit(endog, X, k, f"{k} regimes + {len(ALL_FEATURES)} features")

    # The PRIMARY saved output is the model WITH the text features. Saving the no-text fit
    # here would mean the dashboard, the explainability work and the scenarios all ran on a
    # model that never saw your constructs - they would be decorative. Both are written so
    # you can compare them, and comparing them is worth reporting.
    out, _ = order_and_label(text, df, k, config.REGIME_NAMES)
    out_base, _ = order_and_label(base, df, k, config.REGIME_NAMES)

    risk = per_regime_risk(df["ret"], out["regime"], config.REGIME_NAMES)

    summary = {
        "endog": config.REGIME_ENDOG,
        "text_features": TEXT_FEATURES,
        "macro_features": MACRO_FEATURES,
        "publication_lag_days": config.PUBLICATION_LAG_DAYS,
        "n_regimes": k,
        "coefficients": coefficients(text, ALL_FEATURES, k),
        "transitions": transition_matrix(text, k),
        "out_of_sample": rolling_origin_eval(df, ALL_FEATURES, k),
        "AIC_COMPARISON_RULE": (
            "AIC is comparable only WITHIN one dependent variable. Returns and log realised "
            "volatility are different response scales, so their likelihoods are not on a "
            "common footing and the raw numbers must not be ranked against each other. "
            "Compare text vs no-text within a response; compare responses only on an "
            "out-of-sample score computed on a common target."),
        "aic_no_text": float(base.aic),
        "aic_with_text": float(text.aic),
        "text_gain": float(base.aic - text.aic),
        "risk": risk,
        "regime_count_note": (
            "Three regimes, matching the reference implementation. Regime shares are not "
            "comparable across different regime counts, so holding the count equal keeps the "
            "comparison in section 4 of the brief like-for-like."),
    }
    json.dump(summary, open(config.DATA_PROCESSED / "regime_summary.json", "w"), indent=2)
    out.reset_index().to_parquet(config.DATA_PROCESSED / "regimes.parquet", index=False)
    out_base.reset_index().to_parquet(
        config.DATA_PROCESSED / "regimes_base.parquet", index=False)

    print(f"\n  {k} REGIMES (same count as the reference, so shares are comparable)")
    for name, v in risk.items():
        print(f"    {name:9s} {v['share']:5.1%} of days   vol {v['ann_vol']:6.1%}   "
              f"ES95 {v['ES_95_daily']:+.2%}")
    print(f"\n  text improves AIC by {summary['text_gain']:+.1f} "
          f"({summary['aic_no_text']:.1f} -> {summary['aic_with_text']:.1f})")

    print("\n  WHICH PERIODS EACH REGIME PICKS UP")
    yrs = pd.Series(out.index.year, index=out.index)
    for r in sorted(out["regime"].dropna().unique()):
        top = yrs[out["regime"] == r].value_counts().head(4).index.tolist()
        print(f"    {config.REGIME_NAMES[int(r)]:10s} most often: "
              f"{', '.join(str(y) for y in sorted(top))}")

    if any(v.startswith("regime_") for v in config.REGIME_NAMES.values()):
        print("\n  NAME YOUR REGIMES. They are still neutral placeholders.")
        print("  Using the figures above - share of days, volatility, ES, and the periods each")
        print("  regime covers - decide what each state actually IS, then rename them in")
        print("  config.py. The naming argument is marked; the exact numbers are not.")
        print("  Ask specifically: is each state economically distinct, or is one of them just")
        print("  the tail of another?")
    return out


if __name__ == "__main__":
    run()

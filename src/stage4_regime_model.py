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

   THREE, the same as the reference implementation - see the note in config.py for the full
   reasoning. In short: regime shares are not comparable across different regime counts, so
   holding it equal to the reference keeps the section 4 comparison valid; the 3-regime model
   converges 4/4 across optimiser starts where the 4-regime model manages 2/4; and three
   states are interpretable without inventing a name for a fourth.

   The text's measured AIC contribution happens to be larger at three regimes than at four.
   That is a sensitivity result to disclose, NOT a reason to prefer three - choosing a
   specification because it flatters your own features is the practice this course teaches
   you to catch in other people's work.

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

# YOUR CHOICE, set in config.py so that this module stays identical everywhere.
# See the notes on config.MACRO_FEATURES before you change it.
MACRO_FEATURES: list[str] = list(config.MACRO_FEATURES)

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


def _standardise(X):
    """z-score the exog columns. Returns (standardised, mean, sd).

    WHY THIS IS NOT OPTIONAL. The text constructs live on 0-1 with a standard deviation around
    0.1. VIX ranges from 9 to 83 with a standard deviation near 9 - roughly ninety times
    larger. Handing both to the optimiser unscaled makes the likelihood surface badly
    conditioned in exactly the directions it has to search, and statsmodels fails outright
    with "Could not untransform parameters" rather than returning something wrong.

    That is the good case. The dangerous one is a fit that half-works: with
    require_converged=False a badly scaled model returns parameters, an AIC, and a set of
    regime probabilities that all look ordinary and are not.

    Standardising is a reparameterisation of the mean equation, so the likelihood, the AIC and
    the fitted regimes are unchanged - only the units of the coefficients move. They are moved
    back by unscaled_params() before anything is reported, so nothing downstream has to know
    this happened.
    """
    X = np.asarray(X, dtype=float)
    m = X.mean(axis=0)
    s = X.std(axis=0, ddof=0)
    s = np.where(s < 1e-12, 1.0, s)        # a constant column would divide by zero
    return (X - m) / s, m, s


def _params_by_regime(res, k: int, n_feat: int):
    """(const, beta, sigma) per regime, in whatever units the model was fitted in.

    statsmodels names exog coefficients POSITIONALLY - x1, x2, ... in exog column order - so
    they are read back by position. Reading them by feature name silently returns zero.
    """
    pmap = dict(zip(res.model.param_names, np.asarray(res.params)))
    const = np.array([pmap.get(f"const[{j}]", pmap.get("const", 0.0)) for j in range(k)])
    sigma = np.sqrt([max(pmap.get(f"sigma2[{j}]", 1.0), 1e-9) for j in range(k)])
    beta = np.array([[pmap.get(f"x{c+1}[{j}]", pmap.get(f"x{c+1}", 0.0))
                      for c in range(n_feat)] for j in range(k)])
    return const, beta, sigma


def unscaled_params(res, k: int, n_feat: int):
    """(const, beta, sigma) expressed in the ORIGINAL feature units.

    Undoes the standardisation applied in fit(), so a coefficient means what the report says
    it means: the change in log volatility per one unit of the raw feature. For a 0-1
    construct that is a full 0 -> 1 move; for VIX it is one index point.

        x_std = (x - m) / s
        beta_orig  = beta_std / s
        const_orig = const_std - sum(beta_orig * m)

    With these, the conditional mean is const_orig + X_raw @ beta_orig.T, so callers work in
    raw units throughout and never have to carry a scaler around.
    """
    const, beta, sigma = _params_by_regime(res, k, n_feat)
    scaler = getattr(res, "_exog_scaler", None)
    if scaler is None or n_feat == 0:
        return const, beta, sigma
    m, s = scaler
    beta_o = beta / s[None, :]
    const_o = const - (beta_o * m[None, :]).sum(axis=1)
    return const_o, beta_o, sigma


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
    scaler = None
    if exog is not None:
        exog, _m, _s = _standardise(exog)
        scaler = (_m, _s)

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
    res._exog_scaler = scaler          # unscaled_params() undoes the standardisation
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
    # Reported in ORIGINAL feature units. The model is fitted on standardised exog for
    # numerical stability (see _standardise), so a raw coefficient off res.params is in
    # standard-deviation units and would make a 0-1 construct look ~10x weaker than it is.
    const, beta, sigma = unscaled_params(res, k, len(features))
    scaler = getattr(res, "_exog_scaler", None)
    sd_scale = scaler[1] if scaler is not None else np.ones(len(features))

    for r in range(k):
        reg = {"const": float(const[r]), "sigma2": float(sigma[r] ** 2)}
        for i, f in enumerate(features):
            b = float(beta[r][i])
            # Standard errors come off the fit, so they are in standardised units too and
            # need the same division by the feature's sd to match the coefficient.
            raw_se = se.get(f"x{i+1}[{r}]", float("nan"))
            reg[f] = {"coef": b,
                      "se": float(raw_se / sd_scale[i]) if np.isfinite(raw_se) else float("nan"),
                      "pct_change_in_vol_per_unit":
                          float(np.exp(b) - 1) if config.REGIME_ENDOG == "log_rv" else None}
        out[f"regime_{r}"] = reg
    return out


# Trading days dropped between the end of training and the start of scoring. The dependent
# variable is a 21-day trailing window, so without a gap the first test observation shares 20
# of its 21 returns with the last training observation and is not out of sample in any useful
# sense.
FOLD_GAP_DAYS = 21


def _gaussian_logpdf(y, mu, sigma):
    """Log density of a normal. Computed in logs throughout - exponentiating first and taking
    the log afterwards underflows to -inf on the tail observations that matter most."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2


def hamilton_one_step_scores(y, mu_t, sigma, P, p0):
    """Recursive one-step-ahead predictive log scores from the Hamilton filter.

    THE BUG THIS REPLACES. The previous version took the filtered state at the END OF
    TRAINING and used that same fixed distribution to score every observation in the test
    block. It never advanced the state through the transition matrix and never updated it
    after seeing an observation. That is not a one-step-ahead forecast - it is one static
    forecast reused for months, and it gets worse the further into the block you go, in a way
    that is invisible in the averaged number.

    The actual recursion, which is three lines and has been standard since Hamilton (1989):

        predict   xi_{t|t-1} = P @ xi_{t-1|t-1}
        score     log f(y_t) = log SUM_j xi_{t|t-1}[j] * N(y_t ; mu_t[j], sigma[j])
        update    xi_{t|t}[j] proportional to xi_{t|t-1}[j] * N(y_t ; mu_t[j], sigma[j])

    Parameters stay FIXED at their training values - only the state is updated, using
    observations as they arrive. That is what a filter does in real time and it is legitimate
    out-of-sample evaluation; re-estimating parameters on test data would not be.

    y      (T,)    observations in the test block
    mu_t   (T, k)  conditional mean per regime at each t
    sigma  (k,)    per-regime standard deviation
    P      (k, k)  column-stochastic transition matrix
    p0     (k,)    filtered state entering the block

    Returns (T,) log scores. Higher is better.
    """
    y = np.asarray(y, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    T, k = mu_t.shape
    if len(y) != T:
        raise ValueError(f"y has {len(y)} rows, mu_t has {T}")
    p = np.asarray(p0, dtype=float)
    p = p / p.sum()

    out = np.empty(T)
    for t in range(T):
        pred = P @ p
        pred = np.maximum(pred, 1e-300)
        ll = _gaussian_logpdf(y[t], mu_t[t], sigma)
        # log-sum-exp: the mixture density underflows for outlying observations otherwise,
        # and those are exactly the days that separate a good model from a bad one.
        m = float(ll.max())
        w = pred * np.exp(ll - m)
        s = float(w.sum())
        out[t] = np.log(s) + m
        p = w / s
    return out


def rolling_origin_eval(df: pd.DataFrame, features: list[str], k: int,
                        n_folds: int = 4, min_train: float = 0.5) -> dict:
    """Blocked rolling-origin out-of-sample evaluation, with recursive filtering.

    Everything else in this module is IN-SAMPLE. AIC rewards fit on the data you fitted to,
    and with a flexible regime model that is a low bar. This is the only number here that
    says whether the features help on data the model has not seen.

    Expanding window: train on the first 50%, skip FOLD_GAP_DAYS, score the next block,
    extend, repeat. No shuffling and no random splits - the data is a time series and a
    random split leaks the future into the past.

    Scored on MEAN LOG SCORE (higher is better) of the RECURSIVE one-step-ahead predictive
    density - see hamilton_one_step_scores(). Comparable across specifications with the SAME
    dependent variable, and NOT comparable across different dependent variables.

    Three things this reports that the previous version did not, and each of them can reverse
    the conclusion:
      - per-fold scores, because a positive mean can hide two negative folds
      - a persistence baseline, because the target is 95% overlapping and easy to predict
      - how many folds were REJECTED for not converging, instead of scoring them anyway
    """
    endog_all = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
                 else (df["ret"] * 100).values)
    # THREE feature arms, not two. Comparing all-features against intercept-only and calling
    # the difference "the text gain" credits the constructs with whatever the macro features
    # did - and on this data the macro features do most of it. The conditional comparison
    # (text+macro vs macro-only) is the one that answers "what did the language add?".
    text_cols = [f for f in features if f in TEXT_FEATURES]
    macro_cols = [f for f in features if f not in TEXT_FEATURES]
    ARMS = {
        "text_and_macro": features or None,
        "macro_only": macro_cols or None,
        "intercept_only": None,
    }
    n = len(endog_all)
    start = int(n * min_train)
    edges = np.linspace(start, n, n_folds + 1).astype(int)

    folds = {name: [] for name in ARMS}
    folds["persistence"] = []
    for i in range(n_folds):
        tr_end, te_end = edges[i], edges[i + 1]
        te_start = tr_end + FOLD_GAP_DAYS
        if te_end - te_start < 40:
            print(f"      OOS fold {i+1}: test block too short after the {FOLD_GAP_DAYS}-day "
                  f"gap, skipped")
            continue

        # PERSISTENCE BASELINE. y_t is a 21-day TRAILING window, so y_{t+1} shares 20 of its
        # 21 returns with y_t. "Tomorrow looks like today" is therefore a very strong
        # forecast, and a model that cannot beat it has demonstrated nothing. Reporting the
        # regime model's score without this number next to it makes an easy target look hard.
        prev = endog_all[te_start - 1:te_end - 1]
        actual = endog_all[te_start:te_end]
        resid_sd = float(np.std(np.diff(endog_all[:tr_end]), ddof=1))
        folds["persistence"].append(float(np.mean(
            _gaussian_logpdf(actual, prev, max(resid_sd, 1e-9)))))

        for label, cols in ARMS.items():
            X = df[cols].values if cols else None
            try:
                # require_converged=True. A non-converged fit has parameters, produces a
                # score, and that score is not evidence. The previous version explicitly
                # allowed them, so a fold could be won by an optimiser failure.
                r = fit(endog_all[:tr_end], None if X is None else X[:tr_end], k,
                        f"OOS fold {i+1} {label}", require_converged=True)
                n_feat = 0 if X is None else X.shape[1]
                mu, beta, sig = unscaled_params(r, k, n_feat)
                if X is not None:
                    mu_t = mu[None, :] + X[te_start:te_end] @ beta.T
                else:
                    mu_t = np.tile(mu, (te_end - te_start, 1))

                P = np.asarray(r.regime_transition).reshape(k, k)
                if not np.allclose(P.sum(axis=0), 1.0):
                    P = P.T
                # State at the END of training, carried across the gap, then filtered
                # forward through the test block one observation at a time.
                p0 = np.asarray(r.filtered_marginal_probabilities)[-1]
                p0 = propagate(P, p0, FOLD_GAP_DAYS)
                s = hamilton_one_step_scores(endog_all[te_start:te_end], mu_t, sig, P, p0)
                folds[label].append(float(np.mean(s)))
            except Exception as e:                           # noqa: BLE001
                print(f"      OOS fold {i+1} {label} rejected: {str(e)[:70]}")
                folds[label].append(None)

    def _mean(v):
        good = [x for x in v if x is not None]
        return float(np.mean(good)) if good else None

    def _paired_gain(a: str, b: str):
        """Difference computed only on folds where BOTH arms converged.

        Averaging each arm over a different set of folds and subtracting compares two
        different samples, which is how a gain appears or vanishes for reasons that have
        nothing to do with the features.
        """
        pairs = [(x, y) for x, y in zip(folds[a], folds[b])
                 if x is not None and y is not None]
        if not pairs:
            return None, 0
        return float(np.mean([x - y for x, y in pairs])), len(pairs)

    out = {
        "per_fold": {k2: v for k2, v in folds.items()},
        "text_and_macro": _mean(folds["text_and_macro"]),
        "macro_only": _mean(folds["macro_only"]),
        "intercept_only": _mean(folds["intercept_only"]),
        "persistence_baseline": _mean(folds["persistence"]),
        "features_in_each_arm": {"text_and_macro": features,
                                 "macro_only": macro_cols,
                                 "intercept_only": []},
        "n_folds_attempted": n_folds,
        "n_folds_scored": len([x for x in folds["text_and_macro"] if x is not None]),
        "n_folds_rejected_not_converged": len([x for x in folds["text_and_macro"]
                                               if x is None]),
        "train_test_gap_days": FOLD_GAP_DAYS,
    }

    g, npair = _paired_gain("text_and_macro", "macro_only")
    out["text_gain_conditional_on_macro"] = g
    out["n_folds_paired_conditional"] = npair
    if g is not None:
        out["text_helps_conditional"] = bool(g > 0)

    g, npair = _paired_gain("text_and_macro", "intercept_only")
    out["all_features_gain_vs_intercept"] = g
    out["n_folds_paired_marginal"] = npair

    g, _ = _paired_gain("text_and_macro", "persistence")
    out["gain_over_persistence"] = g
    if g is not None:
        out["beats_persistence"] = bool(g > 0)

    out["metric"] = ("mean RECURSIVE one-step-ahead predictive log score, higher is better. "
                     "Comparable only within one dependent variable.")
    out["_how_to_read"] = (
        "QUOTE text_gain_conditional_on_macro as the text's out-of-sample contribution. "
        "all_features_gain_vs_intercept includes whatever the macro features did and must not "
        "be described as a text gain. Report per_fold, not only the mean - four folds covering "
        "different market conditions can average positive while two are negative. And report "
        "gain_over_persistence: with a trailing 21-day target, beating an intercept-only "
        "regime model is easy and beating persistence is not.")
    return out


def nested_comparison(df: pd.DataFrame, endog, k: int) -> dict:
    """Four nested fits, so the text contribution is not confounded with the macro one.

    THE BUG THIS REPLACES. The previous version fitted exactly two models - intercept-only,
    and text-plus-macro - and labelled the whole AIC difference `text_gain`. If you put
    anything in MACRO_FEATURES, that number silently credits your text constructs with
    whatever the macro series contributed. Since realised-volatility macro features are far
    stronger predictors of a volatility regime than any language measure, the attribution can
    be almost entirely wrong while looking spectacular.

    The four fits:

        intercept_only     no exogenous features at all
        text_only          the five risk-voice constructs
        macro_only         your macro / cross-asset choices
        text_and_macro     both - this is the PRIMARY model, saved to regimes.parquet

    and the two things you can then say:

        MARGINAL text contribution     AIC(intercept_only) - AIC(text_only)
            what the text buys you when it is the only thing in the model

        CONDITIONAL text contribution  AIC(macro_only)     - AIC(text_and_macro)
            what the text buys you ON TOP OF the macro features - the number to report if
            you have any macro features, and the one section 8.1 of the brief asks for

    The gap between those two is the answer to "did the RBA language add anything the market
    data did not already tell me?" A construct with a large marginal contribution and a
    negligible conditional one is tracking volatility that VIX already told you about. That
    is a legitimate and reportable finding - it is not a failure.

    AIC is comparable here because all four fits share one dependent variable and one sample.
    """
    X_text = df[TEXT_FEATURES].values if TEXT_FEATURES else None
    X_macro = df[MACRO_FEATURES].values if MACRO_FEATURES else None
    X_all = df[ALL_FEATURES].values if ALL_FEATURES else None

    specs = {"intercept_only": None, "text_only": X_text,
             "macro_only": X_macro, "text_and_macro": X_all}

    fits, aics = {}, {}
    for name, X in specs.items():
        n_feat = 0 if X is None else X.shape[1]
        fits[name] = fit(endog, X, k, f"{name} ({n_feat} features)")
        aics[name] = float(fits[name].aic)

    out = {
        "aic": aics,
        "text_gain_marginal": aics["intercept_only"] - aics["text_only"],
        "text_gain_conditional": aics["macro_only"] - aics["text_and_macro"],
        "macro_gain_marginal": aics["intercept_only"] - aics["macro_only"],
        "n_text_features": len(TEXT_FEATURES),
        "n_macro_features": len(MACRO_FEATURES),
        "_fits": fits,
    }

    if not MACRO_FEATURES:
        out["WARNING_no_macro_features"] = (
            "MACRO_FEATURES is empty, so macro_only is the same fit as intercept_only and "
            "text_and_macro is the same fit as text_only. The conditional and marginal text "
            "contributions are therefore identical, and you have NOT shown that the text adds "
            "anything beyond market data - you have not given the model any market data to "
            "compete with. Section 8.1 of the brief requires your macro choices to enter the "
            "model. Do not write 'market features dominate' or 'the text survives controls' "
            "on the basis of this run.")

    out["_how_to_report"] = (
        "Quote text_gain_conditional as the contribution of your constructs whenever "
        "MACRO_FEATURES is non-empty. text_gain_marginal overstates it by however much the "
        "macro features were doing.")
    return out


def forward_vol_eval(df: pd.DataFrame, features: list[str], k: int,
                     horizon: int = 21, min_train: float = 0.5) -> dict:
    """Can the model predict volatility it has NOT already seen?

    WHY THIS EXISTS ALONGSIDE rolling_origin_eval. The model's dependent variable is TRAILING
    21-day realised volatility. At date t, 20 of the 21 returns making up y_t are also in
    y_{t-1}, so the series is enormously overlapping and nearly all of y_t is already
    observed. A one-step-ahead score on it is a legitimate check that the model is
    well-specified, but it is a weak test of forecasting: persistence alone does very well,
    and a feature can look predictive purely by tracking what has already happened.

    So this evaluates against FORWARD realised volatility - the volatility of the `horizon`
    days AFTER t, which is genuinely unknown at t.

    NON-OVERLAPPING. Test dates are taken every `horizon` days so that no two targets share a
    return. Overlapping forward windows make the effective sample far smaller than the number
    of rows suggests, and inflate any apparent skill.

    Reported as R^2 against a persistence baseline, not as a log score: the quantity being
    predicted is not the model's own endog, so its likelihood is not the right yardstick.
    """
    ret = df["ret"].values
    n = len(ret)
    fwd = np.full(n, np.nan)
    for t in range(n - horizon):
        w = ret[t + 1:t + 1 + horizon]
        sd = np.std(w, ddof=1)
        if sd > 0:
            fwd[t] = np.log(sd * np.sqrt(252))

    endog_all = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
                 else (df["ret"] * 100).values)
    X_all = df[features].values if features else None
    tr_end = int(n * min_train)

    try:
        r = fit(endog_all[:tr_end], None if X_all is None else X_all[:tr_end], k,
                f"forward-vol eval ({len(features)} features)", require_converged=True)
    except RuntimeError as e:
        return {"error": str(e), "n_test": 0}

    mu, beta, sig = unscaled_params(r, k, 0 if X_all is None else X_all.shape[1])
    P = np.asarray(r.regime_transition).reshape(k, k)
    if not np.allclose(P.sum(axis=0), 1.0):
        P = P.T

    mu_t_all = (mu[None, :] + X_all @ beta.T) if X_all is not None else np.tile(mu, (n, 1))

    # Filter forward through the test region, taking a prediction every `horizon` days.
    p = np.asarray(r.filtered_marginal_probabilities)[-1]
    preds, actuals, persist = [], [], []
    for t in range(tr_end, n - horizon):
        p = P @ p
        ll = _gaussian_logpdf(endog_all[t], mu_t_all[t], sig)
        m = ll.max()
        w = p * np.exp(ll - m)
        p = w / w.sum()
        if (t - tr_end) % horizon == 0 and np.isfinite(fwd[t]):
            p_ahead = propagate(P, p, horizon)
            preds.append(float(mu_t_all[t] @ p_ahead))
            actuals.append(float(fwd[t]))
            persist.append(float(endog_all[t]))

    if len(actuals) < 10:
        return {"error": "too few non-overlapping test points", "n_test": len(actuals)}

    a = np.array(actuals)
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2_model = 1.0 - float(((a - np.array(preds)) ** 2).sum()) / ss_tot
    r2_persist = 1.0 - float(((a - np.array(persist)) ** 2).sum()) / ss_tot

    return {
        "target": f"log realised volatility over the NEXT {horizon} trading days",
        "n_test_non_overlapping": len(actuals),
        "r2_model": r2_model,
        "r2_persistence": r2_persist,
        "model_beats_persistence": bool(r2_model > r2_persist),
        "rmse_model": float(np.sqrt(((a - np.array(preds)) ** 2).mean())),
        "rmse_persistence": float(np.sqrt(((a - np.array(persist)) ** 2).mean())),
        "_note": ("Non-overlapping forward windows, so n_test is the true independent sample "
                  "size. A negative R^2 means the model is worse than predicting the mean; "
                  "that is a real result and should be reported, not hidden."),
    }


def ordered_index(remap: dict[int, int]) -> list[int]:
    """Inverse of `remap`: ordered_index[i] is the RAW statsmodels state at ordered slot i.

    `order_and_label` returns remap as {raw_state: ordered_slot}. Almost everything
    downstream needs the other direction, because it iterates over ordered regimes 0..k-1
    (calm, unsettled, stressed) and needs to know which raw state each one was.
    """
    return [raw for raw, _ in sorted(remap.items(), key=lambda kv: kv[1])]


def transition_matrix(res, k: int, remap: dict[int, int]) -> dict:
    """DAILY transition probabilities and expected durations, in days.

    THE ORDERING BUG THIS GUARDS AGAINST. statsmodels numbers its states arbitrarily - the
    label it calls state 0 is whichever component the optimiser happened to put first, and it
    changes between fits. `order_and_label` sorts them calmest -> most turbulent and returns
    `remap`, but the transition matrix comes straight off the fitted result and is still in
    RAW order. Saving it unremapped puts the matrix in a different state order from the
    regime names, shares and risk metrics sitting beside it in the same JSON file.

    That failure is silent and it looks completely plausible: you get a valid stochastic
    matrix with sensible-looking persistence, and every number computed from it is attached
    to the wrong regime. The check that catches it is at the bottom of run() - the stationary
    distribution of a correctly ordered matrix must be close to the empirical regime shares.

    ONE DAY PER STEP. These are daily probabilities, fitted on daily data. To get a horizon
    of h trading days you need P to the power h - see propagate() - not h/21 steps of some
    monthly matrix, which does not exist.
    """
    P = np.asarray(res.regime_transition).reshape(k, k)
    if not np.allclose(P.sum(axis=0), 1.0):
        P = P.T
    inv = ordered_index(remap)
    P = P[np.ix_(inv, inv)]
    stay = np.diag(P)
    return {"matrix_col_from_row_to": P.tolist(),
            "step_days": 1,
            "state_order": "ordered calmest -> most turbulent, matching config.REGIME_NAMES",
            "expected_duration_days": [float(1.0 / max(1e-9, 1 - s)) for s in stay]}


def propagate(P: np.ndarray, p0: np.ndarray, horizon_days: int) -> np.ndarray:
    """Propagate a regime distribution forward `horizon_days` TRADING DAYS.

    P is a DAILY column-stochastic matrix, so a one-quarter horizon is 63 applications of it.
    An earlier version of this function advanced the chain `horizon_days // 21` times, on the
    assumption that the matrix was monthly. It is not, and there is no monthly matrix
    anywhere in the model. That mistake made a 63-day horizon into a 3-day one and reported
    quarter-ahead stress of about 0.2% where the correct figure was 15%, which reverses the
    conclusion a reader would draw.
    """
    P = np.asarray(P, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    if not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
        raise ValueError("propagate() expects a COLUMN-stochastic matrix (columns sum to 1)")
    if horizon_days < 0:
        raise ValueError("horizon_days must be >= 0")
    p = np.linalg.matrix_power(P, int(horizon_days)) @ (p0 / p0.sum())
    return p / p.sum()


def stationary(P: np.ndarray) -> np.ndarray:
    """Long-run regime distribution: the eigenvector of P for eigenvalue 1."""
    w, v = np.linalg.eig(np.asarray(P, dtype=float))
    s = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    return s / s.sum()


def _prompt_fingerprint() -> str | None:
    """The stage-3 prompt version behind the scores this model was fitted on.

    Read from the manifest stage 3 writes. Returns None if stage 3 has not run, which the
    submission test treats as a failure - an untraceable score is not a reportable one.
    """
    import glob as _glob
    hits = sorted(_glob.glob(str(config.LLM_RAW / "*" / "MANIFEST.json")))
    if not hits:
        return None
    try:
        return json.load(open(hits[-1])).get("fingerprint")
    except Exception:                                        # noqa: BLE001
        return None


def build_model_artifact(res, out: pd.DataFrame, k: int, features: list[str],
                         remap: dict[int, int]) -> dict:
    """The single source of truth for the fitted model, in ORDERED regime space.

    WHY THIS EXISTS. Every module that needs the model used to refit it independently -
    scenarios, explainability, the stakeholder module - and none of them applied the
    volatility ordering, because ordering lived inside stage 4. They then indexed the result
    as though state k-1 meant "stressed". Raw Markov state labels are arbitrary between fits,
    so that assumption is wrong roughly (k-1)/k of the time, and it fails silently: you get a
    confident probability attached to the wrong regime.

    Refitting was also wasteful and non-deterministic - a second fit can land on a different
    optimum from the one the report describes.

    So stage 4 fits ONCE, orders once, and writes everything downstream needs here.
    Everything else loads it. If you find yourself calling sm.tsa.MarkovRegression outside
    this module or stage 5's bootstrap, stop and use this instead.

    Indices throughout are ORDERED: 0 is calmest, k-1 is most turbulent, matching
    config.REGIME_NAMES, the regime shares, and the p_*/pf_* columns in regimes.parquet.
    """
    inv = ordered_index(remap)
    # Coefficients in ORIGINAL feature units, so scenarios can pass raw construct values.
    const, beta, sigma = unscaled_params(res, k, len(features))

    regimes = []
    for slot, raw in enumerate(inv):
        regimes.append({
            "ordered_index": slot,
            "name": config.REGIME_NAMES[slot],
            "raw_state": int(raw),
            "const": float(const[raw]),
            "sigma2": float(sigma[raw] ** 2),
            "beta": {f: float(beta[raw][i]) for i, f in enumerate(features)},
        })

    P = np.asarray(res.regime_transition).reshape(k, k)
    if not np.allclose(P.sum(axis=0), 1.0):
        P = P.T
    P = P[np.ix_(inv, inv)]

    pf_cols = [f"pf_{config.REGIME_NAMES[i]}" for i in range(k)]
    p_last = out[pf_cols].iloc[-1].values.astype(float)

    return {
        "endog": config.REGIME_ENDOG,
        "n_regimes": k,
        "features": list(features),
        "regime_names": [config.REGIME_NAMES[i] for i in range(k)],
        # PROVENANCE. "3 regimes on log_rv" does not identify a model - two runs with
        # different features, a different publication lag or a different prompt version
        # are both that, and are not the same model. Anyone reading a number from this
        # artefact must be able to say which run produced it.
        "endog": config.REGIME_ENDOG,
        "text_features": list(TEXT_FEATURES),
        "macro_features": list(MACRO_FEATURES),
        "publication_lag_days": config.PUBLICATION_LAG_DAYS,
        "macro_lag_days": config.MACRO_LAG_DAYS,
        "prompt_fingerprint": _prompt_fingerprint(),
        "corpus_window": [str(config.CORPUS_START), str(config.CORPUS_END or "latest")],
        "sample_dates": [str(out.index.min().date()), str(out.index.max().date())],
        "n_observations": int(len(out)),
        "model": config.MODEL,
        "seed": config.SEED,
        "raw_to_ordered": {str(r): int(o) for r, o in remap.items()},
        "ordered_to_raw": [int(r) for r in inv],
        "transition_daily_col_from_row_to": P.tolist(),
        "regimes": regimes,
        "filtered_last": (p_last / p_last.sum()).tolist(),
        "filtered_last_date": str(out.index[-1].date()),
        "aic": float(res.aic),
        "converged": bool(res.mle_retvals.get("converged", False)),
        "_note": ("Ordered regime space: index 0 is calmest. Transition matrix is DAILY and "
                  "column-stochastic (columns sum to 1); use stage4.propagate() to advance "
                  "it. Do not refit the model to obtain these - load this file."),
    }


def load_model_artifact() -> dict:
    """Load the stage 4 artifact. Raises a useful error if stage 4 has not been run."""
    p = config.DATA_PROCESSED / "model_artifact.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run stage 4 first - scenarios, explainability and the "
            f"stakeholder module all read the fitted model from it rather than refitting.")
    return json.load(open(p))


def predicted_log_vol(artifact: dict, feature_values: dict) -> np.ndarray:
    """Conditional mean of the dependent variable per ORDERED regime, at given features.

    Returns one value per regime, in ordered index order.
    """
    missing = [f for f in artifact["features"] if f not in feature_values]
    if missing:
        raise ValueError(f"feature values missing for: {missing}")
    return np.array([r["const"] + sum(r["beta"][f] * float(feature_values[f])
                                      for f in artifact["features"])
                     for r in artifact["regimes"]], dtype=float)


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
    print(f"  regimes.parquet = text + macro; regimes_base.parquet = "
          f"{'macro only' if MACRO_FEATURES else 'intercept only'} (differs by text alone)")
    k = config.N_REGIMES
    nested = nested_comparison(df, endog, k)
    text = nested["_fits"]["text_and_macro"]
    # The "without text" comparator must differ from the reported model ONLY in the text.
    # Using intercept-only strips the macro controls too, so regimes_base.parquet would
    # differ for two reasons at once and any comparison against it would be confounded.
    base_name = "macro_only" if MACRO_FEATURES else "intercept_only"
    base = nested["_fits"][base_name]

    # The PRIMARY saved output is the model WITH the text features. Saving the no-text fit
    # here would mean the dashboard, the explainability work and the scenarios all ran on a
    # model that never saw your constructs - they would be decorative. Both are written so
    # you can compare them, and comparing them is worth reporting.
    out, remap = order_and_label(text, df, k, config.REGIME_NAMES)
    out_base, _ = order_and_label(base, df, k, config.REGIME_NAMES)

    risk = per_regime_risk(df["ret"], out["regime"], config.REGIME_NAMES)

    artifact = build_model_artifact(text, out, k, ALL_FEATURES, remap)
    json.dump(artifact, open(config.DATA_PROCESSED / "model_artifact.json", "w"), indent=2)

    summary = {
        "endog": config.REGIME_ENDOG,
        "text_features": TEXT_FEATURES,
        "macro_features": MACRO_FEATURES,
        "publication_lag_days": config.PUBLICATION_LAG_DAYS,
        "n_regimes": k,
        "coefficients": coefficients(text, ALL_FEATURES, k),
        "transitions": transition_matrix(text, k, remap),
        "out_of_sample": rolling_origin_eval(df, ALL_FEATURES, k),
        "forward_vol_out_of_sample": forward_vol_eval(df, ALL_FEATURES, k),
        "AIC_COMPARISON_RULE": (
            "AIC is comparable only WITHIN one dependent variable. Returns and log realised "
            "volatility are different response scales, so their likelihoods are not on a "
            "common footing and the raw numbers must not be ranked against each other. "
            "Compare text vs no-text within a response; compare responses only on an "
            "out-of-sample score computed on a common target."),
        "nested_comparison": {kk: vv for kk, vv in nested.items() if kk != "_fits"},
        # Named for WHAT THEY ARE, not "with/without". aic_no_text was ambiguous: it meant
        # intercept-only in an earlier version and macro-only now, and nothing in the name
        # said which.
        "aic_baseline_comparator": float(base.aic),
        "baseline_comparator_is": base_name,
        "aic_text_and_macro": float(text.aic),
        "aic_text_gain_conditional": float(base.aic - text.aic),
        "aic_text_gain_conditional_note": (
            f"AIC improvement from adding the text constructs to a model that already "
            f"contains {'the macro features' if MACRO_FEATURES else 'nothing but an intercept'}. "
            f"See nested_comparison for the full four-model decomposition."),
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

    # ORDERING CHECK. If the transition matrix is in a different state order from the regime
    # labels, its long-run distribution will not match the observed regime shares. This is
    # the only cheap test that catches a mis-ordered matrix, and it catches it every time:
    # a reversed 3-state matrix put 44% of days in "stressed" against an observed 21%.
    P_ord = np.array(summary["transitions"]["matrix_col_from_row_to"])
    stat = stationary(P_ord)
    emp = np.array([risk[config.REGIME_NAMES[i]]["share"] for i in range(k)])
    drift = float(np.abs(stat - emp).sum())
    print(f"\n  ordering check: stationary distribution vs observed shares, L1 = {drift:.3f}")
    for i in range(k):
        print(f"    {config.REGIME_NAMES[i]:9s} long-run {stat[i]:5.1%}   observed {emp[i]:5.1%}")
    if drift > 0.10:
        print("    *** WARNING: these should agree closely. A large gap almost always means "
              "the transition matrix is not in the same regime order as the labels. ***")
    print("    expected duration, days: " + "  ".join(
        f"{config.REGIME_NAMES[i]} {summary['transitions']['expected_duration_days'][i]:.0f}"
        for i in range(k)))
    nc = summary["nested_comparison"]
    print("\n  NESTED MODEL COMPARISON (AIC, lower is better)")
    for name in ("intercept_only", "text_only", "macro_only", "text_and_macro"):
        print(f"    {name:16s} {nc['aic'][name]:10.1f}")
    print(f"    text contribution, MARGINAL    {nc['text_gain_marginal']:+9.1f}"
          f"   (vs intercept only)")
    print(f"    text contribution, CONDITIONAL {nc['text_gain_conditional']:+9.1f}"
          f"   (on top of macro - REPORT THIS ONE)")
    print(f"    macro contribution, marginal   {nc['macro_gain_marginal']:+9.1f}")
    if "WARNING_no_macro_features" in nc:
        print("    *** MACRO_FEATURES is empty: marginal and conditional are the same fit. "
              "You have not controlled for market data. ***")

    oos = summary["out_of_sample"]
    print(f"\n  OUT OF SAMPLE (recursive one-step-ahead log score, {oos['n_folds_scored']}"
          f"/{oos['n_folds_attempted']} folds scored, "
          f"{oos['n_folds_rejected_not_converged']} rejected)")
    for name in ("with_features", "without_features", "persistence_baseline"):
        v = oos.get(name)
        print(f"    {name:22s} {v:+8.4f}" if v is not None else f"    {name:22s}   n/a")
    if oos.get("gain_over_persistence") is not None:
        verdict = "BEATS" if oos["beats_persistence"] else "DOES NOT BEAT"
        print(f"    -> {verdict} persistence by {oos['gain_over_persistence']:+.4f}")
    print(f"    per fold, with features: "
          f"{[None if v is None else round(v, 3) for v in oos['per_fold']['with_features']]}")

    fv = summary["forward_vol_out_of_sample"]
    if "error" not in fv:
        print(f"\n  FORWARD 21-DAY VOLATILITY ({fv['n_test_non_overlapping']} "
              f"non-overlapping test points)")
        print(f"    R^2 model {fv['r2_model']:+.3f}   R^2 persistence "
              f"{fv['r2_persistence']:+.3f}   "
              f"{'model wins' if fv['model_beats_persistence'] else 'PERSISTENCE WINS'}")

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

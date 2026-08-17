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

   On our data that is the difference between the text measurably hurting held-out
   prediction and measurably helping it. Same text, same model, same data.

2. HOW MANY REGIMES

   THREE, the same as the reference implementation - see the note in config.py for the full
   reasoning. In short: regime shares are not comparable across different regime counts, so
   holding it equal to the reference keeps the section 4 comparison valid; the 3-regime model
   converges 4/4 across optimiser starts where the 4-regime model manages 2/4; and three
   states are interpretable without inventing a name for a fourth.

   The text's measured out-of-sample contribution happens to be larger at three regimes
   than at four. That is a sensitivity result to disclose, NOT a reason to prefer three -
   choosing a specification because it flatters your own features is the practice this
   course teaches you to catch in other people's work.

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


# DETERMINISTIC multi-start. Two things were wrong with the obvious approach.
#
# First, statsmodels' `search_reps` asks for random starting values drawn from a global RNG
# we do not control, so two runs of the SAME specification can land on different optima and
# report different numbers. Nothing here is reproducible if that is left on.
#
# Second, a handful of starts is not enough. This likelihood is strongly multimodal once
# exogenous regressors enter: the log-likelihood spread across converged starts within ONE
# specification runs to several hundred points. A few draws from that surface will not
# reliably find the best mode, so what gets compared is two arbitrary local optima.
#
# EM_STARTS varies the EM burn-in and iteration budget deterministically; the perturbed
# starts jitter the best EM solution using a generator we own and seed, so the search is
# broad AND reproduces exactly.
EM_STARTS = [dict(em_iter=n, search_reps=0, maxiter=m)
             for n, m in [(5, 100), (10, 200), (20, 200), (35, 400), (50, 500), (80, 800)]]
N_PERTURBED_STARTS = 6          # seeded perturbations of the best EM solution
PERTURB_SCALE = 0.25            # relative sd of the perturbation

# REDUCED budget for the out-of-sample fold fits and the ablation refits. Those run dozens of
# times and the full budget makes stage 4 take over an hour. The fold fits are shorter
# samples and start more easily; the cost of the smaller budget is a slightly noisier held-out
# score, which is visible in per_fold rather than hidden.
OOS_EM_STARTS = EM_STARTS[:3]
OOS_N_PERTURBED = 3

# Leave-one-out ablation refits every feature on every fold, so it is the most expensive
# thing in the module: n_features x ABLATION_FOLDS fits. Two folds is enough to show whether
# a feature's contribution holds up in more than one period without tripling the runtime.
ABLATION_FOLDS = 2


def _standardise(X):
    """z-score the exog columns. Returns (standardised, mean, sd).

    WHY THIS IS NOT OPTIONAL. The text constructs live on 0-1 with a standard deviation around
    0.1. VIX ranges from 9 to 83 with a standard deviation near 9 - roughly ninety times
    larger. Handing both to the optimiser unscaled makes the likelihood surface badly
    conditioned in exactly the directions it has to search, and statsmodels fails outright
    with "Could not untransform parameters" rather than returning something wrong.

    That is the good case. The dangerous one is a fit that half-works: with
    require_converged=False a badly scaled model returns parameters and a set of regime
    probabilities that all look ordinary and are not.

    Standardising is a reparameterisation of the mean equation, so the likelihood and the
    fitted regimes are unchanged - only the units of the coefficients move. They are moved
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


def fit(y, exog, k, label, require_converged: bool = True,
        em_starts: list | None = None, n_perturbed: int | None = None):
    """Fit from several starts and return the BEST CONVERGED result.

    Two things this deliberately does not do, because both produce numbers that look fine and
    are not:

    1. It does not return the first attempt that failed to raise. This model on log
       volatility is multimodal - different starts land on optima hundreds of log-likelihood
       points apart - so "the first one that did not crash" is not a model, it is a coin
       toss. All starts are run and the highest log-likelihood CONVERGED fit wins.

    2. It does not hide warnings. Convergence failures and Hessian problems are reported. A
       non-converged fit still produces coefficients and regime probabilities, and they mean
       nothing; using them is the mistake this guards against.

    require_converged=False returns the best non-converged fit with a loud warning, for the
    cases where nothing converges and you would rather see something than nothing. If you use
    it, say so in your report and do not draw conclusions from that fit.
    """
    scaler = None
    if exog is not None:
        exog, _m, _s = _standardise(exog)
        scaler = (_m, _s)

    mod = sm.tsa.MarkovRegression(y, k_regimes=k, trend="c", exog=exog,
                                  switching_variance=True)
    converged, failed, errors = [], [], []

    def _try(kw, start_params, tag):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = mod.fit(disp=False, start_params=start_params, **kw)
            except Exception as e:                           # noqa: BLE001
                errors.append(f"{tag}: {str(e)[:70]}")
                return
        msgs = {str(w.message)[:60] for w in caught}
        (converged if res.mle_retvals.get("converged") else failed).append((res, tag, msgs))

    # Pass 1: deterministic EM budgets, no random search. The budget can be reduced by
    # callers that fit dozens of models (ablation), at the cost of a wider spread - which
    # those callers report rather than hide.
    em_starts = EM_STARTS if em_starts is None else em_starts
    n_perturbed = N_PERTURBED_STARTS if n_perturbed is None else n_perturbed
    for i, kw in enumerate(em_starts):
        _try(kw, None, f"em{i}")

    # Pass 2: seeded perturbations of the best solution so far. Our RNG, our seed, so the
    # search is wide and still reproduces exactly.
    best = max(converged or failed, key=lambda t: t[0].llf, default=None)
    if best is not None:
        rng = np.random.default_rng(config.SEED)
        base_params = np.asarray(best[0].params, dtype=float)
        kw = dict(em_iter=10, search_reps=0, maxiter=400)
        for j in range(n_perturbed):
            noise = 1.0 + PERTURB_SCALE * rng.standard_normal(base_params.shape)
            _try(kw, base_params * noise, f"perturb{j}")

    pool = converged or ([] if require_converged else failed)
    if not pool:
        detail = "; ".join(errors[:3]) if errors else f"{len(failed)} starts ran, none converged"
        raise RuntimeError(f"{label}: no converged fit ({detail})")

    res, tag, msgs = max(pool, key=lambda t: t[0].llf)
    n_starts = len(em_starts) + n_perturbed
    flag = "" if converged else "  *** NOT CONVERGED - DO NOT USE THIS FIT AS EVIDENCE ***"
    # Spread across converged starts, in log-likelihood. This is a SEARCH diagnostic: it says
    # how rugged the surface was, not how good the model is. Nothing in the pipeline compares
    # models on it - comparison happens out of sample, in rolling_origin_eval().
    spread = None
    if len(converged) > 1:
        spread = max(t[0].llf for t in converged) - min(t[0].llf for t in converged)
    print(f"  {label:44s} k={k} llf={res.llf:9.1f} conv={bool(converged)}"
          f"  [{tag}, {len(converged)}/{n_starts} converged"
          + (f", llf spread {spread:.0f}" if spread is not None else "") + f"]{flag}")
    for m in sorted(msgs)[:2]:
        print(f"      warning: {m}")

    res._exog_scaler = scaler          # unscaled_params() undoes the standardisation
    res._llf_spread = spread           # how rugged the surface was at this specification
    res._n_converged = len(converged)
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
    """Which feature moves what, and by how much.

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
                        n_folds: int = 4, min_train: float = 0.5,
                        arms: dict | None = None, require_all_arms: bool = True,
                        fit_kwargs: dict | None = None) -> dict:
    """Blocked rolling-origin out-of-sample evaluation, with recursive filtering.

    THE ONLY MODEL-COMPARISON METRIC IN THIS PIPELINE. Everything else that fits a model -
    nested_fits(), ablation_oos(), permutation_importance() - either supplies coefficients or
    reports a difference in the score computed here. Nothing is compared on in-sample fit,
    because a flexible regime model can always fit the data it was given better.

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

    `arms` overrides the default three-arm comparison with any {label: columns} mapping, so
    the same folds, the same gap and the same pairing logic can be reused for leave-one-out
    ablation - see ablation_oos(). With many arms, set require_all_arms=False so one bad arm
    does not destroy the fold for every other arm; pairing is still done per comparison.
    """
    endog_all = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
                 else (df["ret"] * 100).values)
    # THREE feature arms, not two. Comparing all-features against intercept-only and calling
    # the difference "the text gain" credits the constructs with whatever the macro features
    # did - and on this data the macro features do most of it. The conditional comparison
    # (text+macro vs macro-only) is the one that answers "what did the language add?".
    text_cols = [f for f in features if f in TEXT_FEATURES]
    macro_cols = [f for f in features if f not in TEXT_FEATURES]
    ARMS = arms if arms is not None else {
        "text_and_macro": features or None,
        "macro_only": macro_cols or None,
        "text_only": text_cols or None,
        "intercept_only": None,
    }
    fk = dict(em_starts=OOS_EM_STARTS, n_perturbed=OOS_N_PERTURBED)
    fk.update(fit_kwargs or {})
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

        # A fold is only usable if EVERY arm fits on it. Dropping one arm and keeping the
        # others would compare arms averaged over different folds - a different sample, not a
        # different feature set. Fit all arms first, then keep the fold only if all succeeded.
        fold_scores = {}
        for label, cols in ARMS.items():
            X = df[cols].values if cols else None
            try:
                # require_converged=True. A non-converged fit has parameters, produces a
                # score, and that score is not evidence. The previous version explicitly
                # allowed them, so a fold could be won by an optimiser failure.
                r = fit(endog_all[:tr_end], None if X is None else X[:tr_end], k,
                        f"OOS fold {i+1} {label}", require_converged=True, **fk)
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
                fold_scores[label] = float(np.mean(s))
            except Exception as e:                           # noqa: BLE001
                print(f"      OOS fold {i+1} {label} rejected: {str(e)[:70]}")
                fold_scores[label] = None

        complete = all(v is not None for v in fold_scores.values())
        if not complete and require_all_arms:
            missing = [k2 for k2, v in fold_scores.items() if v is None]
            print(f"      OOS fold {i+1} DROPPED for all arms - {missing} did not converge, "
                  f"and an unpaired fold cannot be compared")
        keep = complete or not require_all_arms
        for label in ARMS:
            folds[label].append(fold_scores[label] if keep else None)

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
        "mean_score": {name: _mean(folds[name]) for name in ARMS},
        "persistence_baseline": _mean(folds["persistence"]),
        "features_in_each_arm": {name: (cols or []) for name, cols in ARMS.items()},
        "n_folds_attempted": n_folds,
        "n_folds_scored": len([x for x in folds["text_and_macro"] if x is not None]),
        "n_folds_rejected_not_converged": len([x for x in folds["text_and_macro"]
                                               if x is None]),
        "train_test_gap_days": FOLD_GAP_DAYS,
    }

    # ALL comparisons, in ONE metric, each paired over the folds where both arms converged.
    # The whole nested decomposition lives here - there is no separate in-sample criterion to
    # reconcile it against, because there is no second metric.
    gains, pairs = {}, {}
    for label, (a, b) in {
            "text_gain_conditional_on_macro": ("text_and_macro", "macro_only"),
            "text_gain_marginal":             ("text_only", "intercept_only"),
            "macro_gain_marginal":            ("macro_only", "intercept_only"),
            "all_features_gain_vs_intercept": ("text_and_macro", "intercept_only"),
            "gain_over_persistence":          ("text_and_macro", "persistence"),
    }.items():
        if a not in folds or b not in folds:
            continue
        g, npair = _paired_gain(a, b)
        gains[label] = g
        pairs[label] = npair
    out["gains"] = gains
    out["n_folds_paired"] = pairs
    if gains.get("text_gain_conditional_on_macro") is not None:
        out["text_helps_conditional"] = bool(gains["text_gain_conditional_on_macro"] > 0)
    if gains.get("gain_over_persistence") is not None:
        out["beats_persistence"] = bool(gains["gain_over_persistence"] > 0)

    out["metric"] = ("mean RECURSIVE one-step-ahead predictive log score on held-out data, "
                     "higher is better. This is the ONLY model-comparison metric used "
                     "anywhere in this pipeline - the nested comparison, the leave-one-out "
                     "ablation and the permutation importance are all differences in it, so "
                     "they can be read against each other directly. Comparable only within "
                     "one dependent variable.")
    out["_how_to_read"] = (
        "QUOTE gains.text_gain_conditional_on_macro as the text's contribution. "
        "all_features_gain_vs_intercept includes whatever the macro features did and must not "
        "be described as a text gain, and text_gain_marginal is what the text is worth with "
        "nothing to compete against. Report per_fold, not only the mean - four folds covering "
        "different market conditions can average positive while two are negative. And report "
        "gain_over_persistence: with a trailing 21-day target, beating an intercept-only "
        "regime model is easy and beating persistence is not.")
    return out


def permutation_importance(df: pd.DataFrame, features: list[str], k: int,
                           n_repeats: int = 10, n_folds: int | None = None,
                           min_train: float = 0.5, seed: int | None = None) -> dict:
    """Shuffle each feature in the HELD-OUT block and measure how much prediction degrades.

    THE OTHER HALF OF ATTRIBUTION. ablation_oos() refits without the feature, which answers
    "how much worse is the best model I can build without it?". Permutation keeps the FITTED
    coefficients and destroys only the feature's alignment with time, which answers "how much
    of this model's performance depends on that feature carrying real information?".

    They differ, and the difference is the finding. A feature cheap to ablate but expensive
    to permute was redundant with something else at fit time yet is doing work in the fitted
    model. A feature expensive to ablate but cheap to permute is carrying almost no signal -
    the model is using its MEAN, not its variation.

    SCORED OUT OF SAMPLE, ON THE SAME FOLDS, IN THE SAME UNITS as ablation_oos() and
    rolling_origin_eval(). An in-sample version would answer a different question from the
    ablation it is supposed to be compared against, and the comparison is the point.

    Cheap: one fit per fold, not one per feature. The model is fitted once on each training
    window, then the test block's columns are shuffled and rescored with the coefficients
    held fixed. Shuffling is repeated because one shuffle is one draw - the spread across
    repeats says whether a small drop is real.
    """
    n_folds = ABLATION_FOLDS if n_folds is None else n_folds
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    endog_all = (np.log(df["rv_21"].values) if config.REGIME_ENDOG == "log_rv"
                 else (df["ret"] * 100).values)
    X_all = df[features].to_numpy(dtype=float)

    n = len(endog_all)
    edges = np.linspace(int(n * min_train), n, n_folds + 1).astype(int)
    per_fold_base, per_fold_drops = [], {f: [] for f in features}

    for i in range(n_folds):
        tr_end, te_end = edges[i], edges[i + 1]
        te_start = tr_end + FOLD_GAP_DAYS
        if te_end - te_start < 40:
            continue
        try:
            r = fit(endog_all[:tr_end], X_all[:tr_end], k, f"permutation fold {i+1}",
                    require_converged=True, em_starts=OOS_EM_STARTS,
                    n_perturbed=OOS_N_PERTURBED)
        except Exception as e:                               # noqa: BLE001
            print(f"      permutation fold {i+1} rejected: {str(e)[:70]}")
            continue

        mu, beta, sig = unscaled_params(r, k, X_all.shape[1])
        P = np.asarray(r.regime_transition).reshape(k, k)
        if not np.allclose(P.sum(axis=0), 1.0):
            P = P.T
        p0 = propagate(P, np.asarray(r.filtered_marginal_probabilities)[-1], FOLD_GAP_DAYS)
        y_te = endog_all[te_start:te_end]
        X_te = X_all[te_start:te_end]

        def _score(Xm: np.ndarray) -> float:
            mu_t = mu[None, :] + Xm @ beta.T
            return float(np.mean(hamilton_one_step_scores(y_te, mu_t, sig, P, p0)))

        base = _score(X_te)
        per_fold_base.append(base)
        for j, f in enumerate(features):
            for _ in range(n_repeats):
                Xp = X_te.copy()
                rng.shuffle(Xp[:, j])
                per_fold_drops[f].append(base - _score(Xp))

    if not per_fold_base:
        return {"error": "no fold produced a converged fit", "features": {}}

    out = {"baseline_score": float(np.mean(per_fold_base)),
           "n_repeats": n_repeats, "n_folds_scored": len(per_fold_base),
           "metric": ("mean held-out one-step-ahead predictive log score lost when the "
                      "feature is shuffled. Same units as ablation_oos.ablation_cost."),
           "features": {}}
    for f in features:
        drops = np.asarray(per_fold_drops[f], dtype=float)
        sd = float(drops.std(ddof=1)) if len(drops) > 1 else None
        out["features"][f] = {
            "mean_drop": float(drops.mean()),
            "sd_drop": sd,
            "min_drop": float(drops.min()),
            "max_drop": float(drops.max()),
            # A drop smaller than its own spread across shuffles is not distinguishable
            # from the noise the shuffling itself introduces.
            "distinguishable_from_shuffle_noise":
                bool(drops.mean() > 2 * sd) if sd else None,
        }
    out["_how_to_read"] = (
        "mean_drop is held-out predictive log score lost when this feature's values are "
        "scrambled in time. Compare it directly against ablation_oos.ablation_cost - same "
        "metric, same folds: a feature cheap to ablate but expensive to permute is redundant "
        "at fit time yet load-bearing in the fitted model, and one expensive to ablate but "
        "cheap to permute is being used for its mean rather than its variation.")
    return out


def nested_fits(df: pd.DataFrame, endog, k: int) -> dict:
    """Fit the four nested specifications IN SAMPLE, and return the fitted models.

    THIS FUNCTION PRODUCES COEFFICIENTS, NOT EVIDENCE. It exists because the pipeline needs
    fitted parameters to save: regimes.parquet comes from `text_and_macro`, regimes_base.parquet
    from the comparator without text, and the report's marginal effects come from their
    coefficients. Whether the text HELPS is not decided here - it is decided out of sample, in
    rolling_origin_eval(), on data none of these fits has seen.

    No in-sample information criterion is computed anywhere in this module. When held-out
    performance can be measured directly - and it can, on every comparison this assignment
    asks for - there is nothing for an approximation to it to add.

    The four specifications:

        intercept_only     no exogenous features at all
        text_only          the five risk-voice constructs
        macro_only         your macro / cross-asset choices
        text_and_macro     both - this is the PRIMARY model, saved to regimes.parquet

    Keeping all four matters even though the numbers come from elsewhere, because the
    comparison the brief asks for is CONDITIONAL: what the text adds on top of the macro
    features, not what it adds to nothing. Realised-volatility macro features are far stronger
    predictors of a volatility regime than any language measure, so a text contribution
    measured against an empty model can look spectacular while being almost entirely the macro
    features' work.
    """
    X_text = df[TEXT_FEATURES].values if TEXT_FEATURES else None
    X_macro = df[MACRO_FEATURES].values if MACRO_FEATURES else None
    X_all = df[ALL_FEATURES].values if ALL_FEATURES else None

    specs = {"intercept_only": None, "text_only": X_text,
             "macro_only": X_macro, "text_and_macro": X_all}

    fits = {}
    for name, X in specs.items():
        n_feat = 0 if X is None else X.shape[1]
        fits[name] = fit(endog, X, k, f"{name} ({n_feat} features)")

    out = {
        "n_text_features": len(TEXT_FEATURES),
        "n_macro_features": len(MACRO_FEATURES),
        # SEARCH diagnostic only. How far apart the converged starts landed, in log-likelihood.
        # A large number here says the surface is rugged and the fitted coefficients are one
        # of several defensible answers - which belongs in your limitations. It is NOT a model
        # comparison and nothing should be ranked by it.
        "llf_spread_across_starts": {n: getattr(f, "_llf_spread", None)
                                     for n, f in fits.items()},
        "n_converged": {n: getattr(f, "_n_converged", None) for n, f in fits.items()},
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

    ragged = {n: v for n, v in out["llf_spread_across_starts"].items()
              if v is not None and v > 50}
    if ragged:
        out["NOTE_rugged_likelihood"] = (
            f"{sorted(ragged)}: converged starts landed more than 50 log-likelihood points "
            f"apart, so these coefficients are the best of several local optima rather than "
            f"a unique answer. This does not invalidate the out-of-sample comparison, which "
            f"is scored on held-out data, but it does belong in your limitations section.")

    out["_how_to_report"] = (
        "These are coefficients. For whether the text CONTRIBUTES, quote "
        "out_of_sample.gains.text_gain_conditional_on_macro - a held-out predictive log "
        "score difference. Do not compute or report an in-sample information criterion.")
    return out


def ablation_oos(df: pd.DataFrame, features: list[str], k: int,
                 n_folds: int | None = None) -> dict:
    """Leave-one-out ablation, scored OUT OF SAMPLE in the same units as everything else.

    For each feature: refit without it and measure how much held-out predictive log score is
    lost, paired fold by fold against the full model. Positive cost = the model got worse
    without that feature.

    WHY THE SAME UNITS AS PERMUTATION IMPORTANCE. The two measures answer different questions
    and the brief asks students to compare them - which is only possible if they are on one
    scale. Both are now differences in mean held-out one-step-ahead log score.

      ablation    refits without the feature: how much worse is the best model I can build
                  WITHOUT it? Cheap to ablate = something else could do its job.
      permutation keeps the fitted coefficients and destroys only the feature's alignment
                  with time: how much of THIS model's performance depends on it carrying
                  real information? Cheap to permute = the model is using its mean, not its
                  variation.

    A feature cheap to ablate but expensive to permute was redundant at fit time yet is
    load-bearing in the model you actually have. The reverse pattern means it is acting as an
    intercept shift. Both are reportable findings.

    COST. n_features x n_folds refits, the most expensive thing in this module. ABLATION_FOLDS
    defaults to 2.
    """
    n_folds = ABLATION_FOLDS if n_folds is None else n_folds
    arms = {"full": list(features)}
    for f in features:
        arms[f"without_{f}"] = [c for c in features if c != f] or None

    print(f"    leave-one-out ablation: {len(features)} features x {n_folds} folds "
          f"= {len(features) * n_folds} refits")
    # require_all_arms=False: with a dozen arms, insisting that every one converges on every
    # fold would throw away folds wholesale. Pairing is still exact - each feature's cost is
    # averaged only over folds where BOTH that arm and the full model scored.
    ev = rolling_origin_eval(df, features, k, n_folds=n_folds, arms=arms,
                             require_all_arms=False)

    per_fold = ev["per_fold"]
    full = per_fold["full"]
    out = {"metric": ev["metric"], "n_folds": n_folds, "features": {}}
    for f in features:
        pairs = [(a, b) for a, b in zip(full, per_fold[f"without_{f}"])
                 if a is not None and b is not None]
        if not pairs:
            out["features"][f] = {"ablation_cost": None,
                                  "note": "no fold where both models converged"}
            continue
        diffs = [a - b for a, b in pairs]
        out["features"][f] = {
            "ablation_cost": float(np.mean(diffs)),
            "per_fold_cost": [float(d) for d in diffs],
            "n_folds_paired": len(diffs),
            # Two folds that disagree in SIGN mean the feature helped in one period and hurt
            # in another. That is a finding about regime dependence, not a number to average.
            "consistent_sign": bool(all(d > 0 for d in diffs) or all(d < 0 for d in diffs)),
        }
    out["_how_to_read"] = (
        "ablation_cost is held-out log score LOST by removing the feature: positive means "
        "the model needs it, negative means it was actively hurting out-of-sample "
        "performance. Directly comparable with permutation_importance.mean_drop, which is in "
        "the same units - read the two together and explain any feature where they disagree.")
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
    nested = nested_fits(df, endog, k)
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
        # THE EVIDENCE. One metric, held-out, used for every comparison in this pipeline:
        # the four nested specifications, the leave-one-out ablation and the permutation
        # importance are all differences in mean one-step-ahead predictive log score on data
        # the model has not seen, paired fold by fold.
        #
        # There is deliberately no in-sample information criterion here. Reporting one
        # alongside this would leave the reader arbitrating between a measured quantity and
        # an approximation to it, and the approximation assumes a well-behaved interior
        # optimum that this likelihood does not have.
        "out_of_sample": rolling_origin_eval(df, ALL_FEATURES, k),
        "PRIMARY_EVIDENCE": (
            "out_of_sample.gains.text_gain_conditional_on_macro - the held-out predictive log "
            "score the text constructs add on top of the macro features. Report it with "
            "out_of_sample.per_fold and out_of_sample.gains.gain_over_persistence beside it."),
        "ablation_out_of_sample": ablation_oos(df, ALL_FEATURES, k),
        "permutation_importance": permutation_importance(df, ALL_FEATURES, k),
        "forward_vol_out_of_sample": forward_vol_eval(df, ALL_FEATURES, k),
        "METRIC_COMPARISON_RULE": (
            "The held-out log score is comparable only WITHIN one dependent variable. Returns "
            "and log realised volatility are different response scales, so their predictive "
            "densities are not on a common footing and must not be ranked against each other. "
            "Compare text vs no-text within a response; compare responses only on a score "
            "computed for a common target."),
        "nested_fits": {kk: vv for kk, vv in nested.items() if kk != "_fits"},
        "baseline_comparator_is": base_name,
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
    if "NOTE_rugged_likelihood" in nc:
        print(f"    note: {nc['NOTE_rugged_likelihood'][:96]}...")
    if "WARNING_no_macro_features" in nc:
        print("    *** MACRO_FEATURES is empty: marginal and conditional are the same fit. "
              "You have not controlled for market data. ***")

    oos = summary["out_of_sample"]
    print(f"\n  NESTED COMPARISON, OUT OF SAMPLE (recursive one-step-ahead log score, "
          f"higher is better, {oos['n_folds_scored']}/{oos['n_folds_attempted']} folds "
          f"scored, {oos['n_folds_rejected_not_converged']} rejected)")
    for name in ("text_and_macro", "macro_only", "text_only", "intercept_only"):
        v = oos["mean_score"].get(name)
        print(f"    {name:22s} {v:+8.4f}" if v is not None else f"    {name:22s}   n/a")
    pb = oos.get("persistence_baseline")
    print(f"    {'persistence_baseline':22s} {pb:+8.4f}" if pb is not None
          else f"    {'persistence_baseline':22s}   n/a")

    gains, npair = oos["gains"], oos["n_folds_paired"]
    g = gains.get("text_gain_conditional_on_macro")
    if g is not None:
        print(f"    -> text, CONDITIONAL on macro: {g:+.4f} "
              f"(paired over {npair['text_gain_conditional_on_macro']} folds) <- REPORT THIS")
    if gains.get("text_gain_marginal") is not None:
        print(f"    -> text, MARGINAL (vs intercept only): "
              f"{gains['text_gain_marginal']:+.4f}")
    if gains.get("macro_gain_marginal") is not None:
        print(f"    -> macro, marginal           : {gains['macro_gain_marginal']:+.4f}")
    if gains.get("all_features_gain_vs_intercept") is not None:
        print(f"    -> all features vs intercept : "
              f"{gains['all_features_gain_vs_intercept']:+.4f}  (NOT a text gain)")
    if gains.get("gain_over_persistence") is not None:
        verdict = "BEATS" if oos["beats_persistence"] else "DOES NOT BEAT"
        print(f"    -> {verdict} persistence by {gains['gain_over_persistence']:+.4f}")
    print(f"    per fold, text+macro: "
          f"{[None if v is None else round(v, 3) for v in oos['per_fold']['text_and_macro']]}")

    # ATTRIBUTION, in the same units as everything above, so the two measures can be read
    # against each other rather than converted.
    abl = summary["ablation_out_of_sample"]["features"]
    perm = summary["permutation_importance"].get("features", {})
    print("\n  PER-FEATURE ATTRIBUTION (held-out log score, same metric as above)")
    print(f"    {'feature':28s} {'ablation':>9s} {'permute':>9s}  reading")
    for f in ALL_FEATURES:
        a = abl.get(f, {}).get("ablation_cost")
        pm = perm.get(f, {}).get("mean_drop")
        if a is None or pm is None:
            print(f"    {f:28s} {'n/a':>9s} {'n/a':>9s}")
            continue
        if a < 0.005 and pm >= 0.005:
            note = "redundant at fit time, load-bearing in this model"
        elif a >= 0.005 and pm < 0.005:
            note = "used for its mean, not its variation"
        elif a < 0.005 and pm < 0.005:
            note = "carrying little"
        else:
            note = "contributes on both measures"
        print(f"    {f:28s} {a:+9.4f} {pm:+9.4f}  {note}")

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

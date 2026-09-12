"""
Rolling-origin evaluation and baselines. PRE-BUILT - do not rewrite.

EVERY NUMBER IN YOUR REPORT MUST COME THROUGH HERE.

Train on meetings 0..t-1, predict meeting t, step forward. A random split or k-fold on a
time series leaks the future into the past, and on this data the leak is large: policy is
strongly autocorrelated, so a randomly-chosen test meeting usually sits between two
training meetings that between them almost give the answer away.

The scaler is refitted inside every window. Fitting it once on the whole sample leaks the
future's mean and variance into every training set - a small leak, but the kind that
quietly inflates every figure in a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def default_classifier():
    """Multinomial logit, balanced class weights. Replace it if you justify the choice."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        # `multi_class="multinomial"` was removed from the signature in scikit-learn 1.7
        # and warns from 1.5. Multinomial is the default for a multiclass target, so
        # dropping the argument changes nothing except the warning.
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))


def build_design(panel: pd.DataFrame, tiers: dict, upto: list[str],
                 verbose: bool = False) -> pd.DataFrame:
    """
    Feature matrix for a cumulative set of tiers. RAW - no imputation here.

    NOTHING IN THIS FUNCTION LOOKS AT THE DATA. An earlier version computed medians and
    dropped columns by missingness and variance across the WHOLE panel, then handed the
    result to rolling-origin evaluation - which leaks the future into every training window
    exactly as full-sample scaling would.

    The feature set is now fixed by the tier lists, and imputation happens inside the model
    pipeline where it is refitted on each training window. The only column dropped is one
    that is entirely absent from the panel, which is a structural fact about the build
    rather than a decision taken from the values.
    """
    cols: list[str] = []
    for t in upto:
        cols += [c for c in tiers.get(t, []) if c in panel.columns]
    X = panel[cols].copy()
    empty = [c for c in X.columns if X[c].notna().sum() == 0]
    if empty:
        if verbose:
            print(f"      dropped (no values at all): {sorted(empty)}")
        X = X.drop(columns=empty)
    return X


def rolling_origin(X: pd.DataFrame, y: pd.Series, clf=None,
                   min_train: int | None = None,
                   available_on: pd.Series | None = None,
                   min_embargoed_train: int = 40,
                   allow_unresolved_labels: bool = False) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead prediction, with a LABEL-AVAILABILITY EMBARGO.

    `available_on` gives, per row, the date that row's LABEL became knowable. At a prediction
    date t the training set is every earlier row whose label had already resolved by t.

    WHY THIS IS NOT OPTIONAL. `y_cycle` at meeting M describes the 182 days after M. Training
    on every preceding row - the obvious implementation, and what this function used to do -
    means that when predicting meeting t the model has already been shown the outcomes of the
    previous six months of meetings, which nobody could have observed at t. Because policy
    moves in runs, those are precisely the labels that give away t's answer.

    On this data the embargo costs the supplied model 23 accuracy points and takes it from
    beating the current-decision baseline to losing to it. That is not a bug in the embargo.
    It is what the model is actually worth.

    OMITTING `available_on` IS NOW AN ERROR, not a default. It used to be the default, and a
    supplied helper called it that way - so a team could follow the instruction to "use
    rolling_origin()" and still produce exactly the leakage the assignment penalises, with
    nothing in the output saying so. Reproducing the leaky number deliberately is still
    supported, and now has to be asked for:

        rolling_origin(X, y, available_on=None, allow_unresolved_labels=True)

    Every row of the returned frame carries `label_embargo`, so a table of results cannot
    lose track of which protocol produced it.
    """
    if available_on is None and not allow_unresolved_labels:
        raise ValueError(
            "rolling_origin() needs `available_on`: the date each row's LABEL became "
            "knowable. Without it, training at meeting t includes labels that had not "
            "resolved at t - the leak this assignment marks teams down for. Use "
            "panel['label_available_on'] (see model_card.py for the supplied call). To "
            "reproduce the leaky comparison ON PURPOSE, pass "
            "allow_unresolved_labels=True as well, and report it as the leaky arm.")
    min_train = min_train or config.MIN_TRAIN_MEETINGS
    clf = clf or default_classifier()
    ok = y.notna()
    X, y = X[ok], y[ok]
    avail = None if available_on is None else pd.to_datetime(available_on).reindex(y.index)
    rows = []
    for i in range(min_train, len(y)):
        t = y.index[i]
        if avail is None:
            train_idx = y.index[:i]
        else:
            past = avail.iloc[:i]
            train_idx = past.index[past <= t]
            if len(train_idx) < min_embargoed_train:
                continue
        ytr = y.loc[train_idx]
        if ytr.nunique() < 2:
            continue
        m = clone(clf).fit(X.loc[train_idx], ytr)
        proba = m.predict_proba(X.iloc[[i]])[0]
        rows.append({"date": t, "y_true": y.iloc[i],
                     "y_pred": m.classes_[proba.argmax()],
                     "n_train": len(train_idx),
                     # stamped per row so a results table can never lose track of which
                     # protocol produced it
                     "label_embargo": avail is not None,
                     **{f"p_{int(c)}": p for c, p in zip(m.classes_, proba)}})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def score(preds: pd.DataFrame, classes: list[int]) -> dict:
    if preds.empty:
        return {}
    y, yh = preds["y_true"], preds["y_pred"]
    pcols = [f"p_{c}" for c in classes if f"p_{c}" in preds.columns]
    P = preds[pcols].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    labels = [int(c.split("_")[1]) for c in pcols]
    try:
        ll = float(log_loss(y, P, labels=labels))
    except ValueError:
        ll = float("nan")
    return {"n": int(len(preds)),
            "accuracy": float(accuracy_score(y, yh)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yh)),
            "log_loss": ll,
            "confusion": confusion_matrix(y, yh, labels=classes).tolist()}


def baselines(panel: pd.DataFrame, y: pd.Series,
              min_train: int | None = None,
              on_dates: pd.Index | None = None) -> dict:
    """
    The numbers any model must be reported against.

    `on_dates` MUST be the model's own prediction dates. Under the label-availability
    embargo the model predicts a shorter, later span than the naive index, and scoring the
    baselines on a different span compares two different questions. Pass
    `preds.index` from `rolling_origin()`.

    'majority' matters most on y_decision, where it is close to 74% and a model can look
    respectable while being worse than doing nothing. 'current_decision' matters most on
    y_cycle, where the label is partly a restatement of the decision just taken.
    """
    min_train = min_train or config.MIN_TRAIN_MEETINGS
    yy = y.dropna()
    idx = yy.index[min_train:] if on_dates is None else pd.Index(on_dates)
    idx = idx.intersection(yy.index)
    truth = yy.loc[idx]
    is_cycle = set(truth.unique()) <= {0, 1, 2}
    out = {}

    maj = yy.iloc[:min_train].mode().iloc[0]
    out["majority"] = {"accuracy": float((truth == maj).mean()),
                       "balanced_accuracy": float(balanced_accuracy_score(
                           truth, np.full(len(truth), maj)))}

    if "last_change_sign" in panel.columns:
        pers = panel.loc[idx, "last_change_sign"]
        if is_cycle:
            pers = pers.map({-1: 0, 0: 1, 1: 2})
        out["persistence"] = {"accuracy": float((truth == pers).mean()),
                              "balanced_accuracy": float(
                                  balanced_accuracy_score(truth, pers))}

    # THE BASELINE THAT MATTERS MOST on y_cycle. Policy is persistent, so simply carrying
    # the decision just taken forward as the cycle direction is a strong and completely
    # trivial rule. Any model claiming to forecast the cycle must beat THIS, not the
    # majority-class rate.
    if is_cycle and "decision" in panel.columns:
        cd = panel.loc[idx, "decision"].map({-1: 0, 0: 1, 1: 2})
        out["current_decision"] = {"accuracy": float((truth == cd).mean()),
                                   "balanced_accuracy": float(
                                       balanced_accuracy_score(truth, cd))}

    if "bab_spread" in panel.columns:
        sp = panel.loc[idx, "bab_spread"]
        mk = pd.Series(np.select([sp < -0.05, sp > 0.05], [-1, 1], 0), index=idx)
        if is_cycle:
            mk = mk.map({-1: 0, 0: 1, 1: 2})
        out["market_implied"] = {"accuracy": float((truth == mk).mean()),
                                 "balanced_accuracy": float(
                                     balanced_accuracy_score(truth, mk))}
    return out


TIER_ORDER = ["persistence", "macro", "market", "text"]


def nested_evaluation(panel: pd.DataFrame, tiers: dict, target: str,
                      classes: list[int], clf=None,
                      tier_order: list[str] | None = None,
                      quiet: bool = False,
                      available_on: pd.Series | None = None,
                      allow_unresolved_labels: bool = False) -> dict:
    """
    Fit the cumulative tier sequence and report each step.

    This is the instrument the whole assignment is built on: it separates what a tier adds
    CONDITIONAL on the tiers before it from what it looks worth on its own.

    `available_on` is passed straight to `rolling_origin()` and is required for the same
    reason it is required there: a tier ladder built on unresolved labels compares tiers
    on how well each one reads the future.
    """
    tier_order = tier_order or TIER_ORDER
    y = panel[target]
    avail = available_on
    if avail is None and not allow_unresolved_labels:
        col = f"{target}_available_on"
        if col not in panel.columns:
            raise ValueError(
                f"nested_evaluation() needs label-availability dates: pass "
                f"available_on=, or build the panel so it carries {col!r}.")
        avail = pd.to_datetime(panel[col])
    res = {"baselines": baselines(panel, y), "tiers": {}}
    prev = None
    for i in range(1, len(tier_order) + 1):
        upto = tier_order[:i]
        if not any(tiers.get(t) for t in upto):
            continue
        name = "+".join(upto)
        X = build_design(panel, tiers, upto)
        if X.empty:
            continue
        s = score(rolling_origin(X, y, clf, available_on=avail,
                                 allow_unresolved_labels=allow_unresolved_labels),
                  classes)
        if not s:
            continue
        s["n_features"] = int(X.shape[1])
        res["tiers"][name] = s
        if not quiet:
            d = "" if prev is None else f"  (delta {s['accuracy']-prev:+.3f})"
            print(f"    {name:34s} n={s['n']:3d} acc={s['accuracy']:.3f} "
                  f"bal={s['balanced_accuracy']:.3f} ll={s['log_loss']:.3f}{d}")
        prev = s["accuracy"]

    a, b = "persistence+macro+market", "persistence+macro+market+text"
    if a in res["tiers"] and b in res["tiers"]:
        res["text_conditional"] = {
            "d_accuracy": res["tiers"][b]["accuracy"] - res["tiers"][a]["accuracy"],
            "d_balanced_accuracy": (res["tiers"][b]["balanced_accuracy"]
                                    - res["tiers"][a]["balanced_accuracy"]),
            "d_log_loss": res["tiers"][b]["log_loss"] - res["tiers"][a]["log_loss"]}
    return res

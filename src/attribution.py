"""
Attribution and data-staleness helpers. SUPPLIED - do not modify.

Two things the Cycle stage asks for that are easy to get subtly wrong:

    permutation_importance_oos()  computed on HELD-OUT meetings, never on the training
                                  sample. Importance measured on data the model has
                                  memorised tells you how hard the model leaned on a
                                  feature, not what the feature was worth.

    staleness_cost()              refits with the slow-publishing series removed, so you
                                  can quantify what months-old data is costing you.

READ THE NOTE ON NEGATIVE IMPORTANCE in permutation_importance_oos(). It is not a bug and
you are expected to interpret it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config
import evaluation as ev


def permutation_importance_oos(panel: pd.DataFrame, tiers: dict, target: str,
                               upto: list[str], clf=None, n_repeats: int = 30,
                               scoring: str = "balanced_accuracy",
                               available_on: pd.Series | None = None) -> dict:
    """
    Permutation importance on held-out meetings, refit once on the head.

    THE EMBARGO APPLIES HERE TOO. This helper used to fit on the first
    MIN_TRAIN_MEETINGS labelled rows and evaluate on everything after them, with no
    reference to when those labels became knowable. On the supplied panel that put four
    rows in the training set whose outcomes had not resolved by the first evaluation
    date - the same leak `rolling_origin()` now refuses to commit, in a helper advertised
    as an honest out-of-sample attribution. The training set is now trimmed to the rows
    whose labels had resolved BEFORE the first evaluation date.

    NEGATIVE IMPORTANCE IS A REAL RESULT, NOT AN ERROR. It means the model scored BETTER
    when that feature was shuffled - the feature was actively misleading it. On a small
    sample with collinear predictors this is common and it is worth reporting: a feature
    with negative importance is a candidate for removal, and saying so is part of the Cycle stage.

    The interval this reports is Monte-Carlo precision over `n_repeats` shuffles of THIS
    sample, exactly as in the model card - not uncertainty about another sample.

    Returns per-feature and per-tier importance. The per-tier figure is the SUM over that
    tier's features, so a tier of seventeen weakly-harmful variables can total to a large
    negative number while no single feature looks dramatic.
    """
    X = ev.build_design(panel, tiers, upto)
    y = panel[target]
    ok = y.notna()
    X, y = X[ok], y[ok]
    cut = config.MIN_TRAIN_MEETINGS
    if len(y) <= cut + 10:
        raise ValueError("not enough held-out meetings for permutation importance")

    avail = available_on
    if avail is None:
        col = f"{target}_available_on"
        if col not in panel.columns:
            raise ValueError(
                f"permutation_importance_oos() needs label-availability dates: pass "
                f"available_on=, or build the panel so it carries {col!r}. Without them "
                f"the fit includes labels that had not resolved when the held-out "
                f"meetings began, which is the leak this assignment deducts for.")
        avail = panel[col]
    avail = pd.to_datetime(avail).reindex(y.index)
    first_eval = y.index[cut]
    train_idx = avail.iloc[:cut].index[avail.iloc[:cut] <= first_eval]
    dropped = cut - len(train_idx)
    if len(train_idx) < 40:
        raise ValueError(
            f"only {len(train_idx)} training rows have labels resolved by "
            f"{first_eval.date()}; widen the head or move the split")
    if dropped:
        print(f"  embargo: {dropped} of {cut} head rows dropped - their labels had not "
              f"resolved by {first_eval.date()}")

    model = (clf or ev.default_classifier()).fit(X.loc[train_idx], y.loc[train_idx])
    r = permutation_importance(model, X.iloc[cut:], y.iloc[cut:], n_repeats=n_repeats,
                               random_state=config.REGIME_SEED, scoring=scoring)

    imp = (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                         "sd": r.importances_std})
           .sort_values("importance", ascending=False))
    tier_of = {c: t for t in upto for c in tiers.get(t, []) if c in X.columns}
    imp["tier"] = imp["feature"].map(tier_of)
    by_tier = imp.groupby("tier")["importance"].sum().sort_values(ascending=False)

    print(f"  by tier: {by_tier.round(4).to_dict()}")
    print("  strongest: " + ", ".join(f"{r.feature}={r.importance:.3f}"
                                      for r in imp.head(6).itertuples()))
    weak = imp[imp["importance"] <= 0]
    print(f"  {len(weak)} of {len(imp)} features have zero or NEGATIVE importance")
    if len(weak):
        print("  weakest:   " + ", ".join(f"{r.feature}={r.importance:.3f}"
                                          for r in imp.tail(4).itertuples()))
    return {"by_tier": by_tier.round(4).to_dict(),
            "features": imp.round(4).to_dict("records"),
            "n_zero_or_negative": int(len(weak))}


def staleness_table(panel: pd.DataFrame, tiers: dict) -> pd.DataFrame:
    """Median age, at the meeting, of the newest published figure for each macro series."""
    rows = []
    for c in tiers.get("macro", []):
        col = f"{c}__age_days"
        if col in panel.columns:
            rows.append({"series": c,
                         "median_age_days": float(panel[col].median()),
                         "max_age_days": float(panel[col].max())})
    return pd.DataFrame(rows).sort_values("median_age_days", ascending=False)


def staleness_cost(panel: pd.DataFrame, tiers: dict, target: str, classes: list[int],
                   upto: list[str], slow_threshold_days: int = 60, clf=None,
                   available_on: pd.Series | None = None) -> dict:
    """
    What the publication lag costs, measured by removing the slowest-published series.

    Quarterly series are two to four months old at a typical meeting. Refitting without
    them tells you how much of your model's performance rests on figures that were already
    stale when the Board met.

    BOTH arms run under the label-availability embargo. They used to run without it - this
    helper called `rolling_origin()` with no `available_on`, so a team that followed the
    instruction to use the supplied evaluator still produced two leaky numbers and a
    difference between them. The difference was the least wrong part: both arms leaked in
    the same direction, which is exactly the kind of error that survives a sanity check.
    """
    ages = staleness_table(panel, tiers)
    slow = ages.loc[ages["median_age_days"] >= slow_threshold_days, "series"].tolist()
    print(f"  {len(slow)} of {len(ages)} macro series are >={slow_threshold_days}d stale")

    y = panel[target]
    avail = available_on
    if avail is None:
        col = f"{target}_available_on"
        if col not in panel.columns:
            raise ValueError(
                f"staleness_cost() needs label-availability dates: pass available_on=, "
                f"or build the panel so it carries {col!r}. Without them both arms train "
                f"on labels that had not resolved yet.")
        avail = pd.to_datetime(panel[col])
    full = ev.score(ev.rolling_origin(ev.build_design(panel, tiers, upto), y, clf,
                                      available_on=avail), classes)
    t2 = {**tiers, "macro": [c for c in tiers.get("macro", []) if c not in slow]}
    fast = ev.score(ev.rolling_origin(ev.build_design(panel, t2, upto), y, clf,
                                      available_on=avail), classes)

    print(f"  all macro        acc={full['accuracy']:.3f} bal={full['balanced_accuracy']:.3f}")
    print(f"  fast macro only  acc={fast['accuracy']:.3f} bal={fast['balanced_accuracy']:.3f}"
          f"   (delta {fast['accuracy']-full['accuracy']:+.3f})")
    return {"slow_series": slow, "ages": ages.round(1).to_dict("records"),
            "with_slow": full, "without_slow": fast,
            "d_accuracy": fast["accuracy"] - full["accuracy"],
            "d_balanced_accuracy": (fast["balanced_accuracy"]
                                    - full["balanced_accuracy"])}

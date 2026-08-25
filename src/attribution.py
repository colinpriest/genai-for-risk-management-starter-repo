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
                               scoring: str = "balanced_accuracy") -> dict:
    """
    Permutation importance on held-out meetings, refit once on the head.

    NEGATIVE IMPORTANCE IS A REAL RESULT, NOT AN ERROR. It means the model scored BETTER
    when that feature was shuffled - the feature was actively misleading it. On a small
    sample with collinear predictors this is common and it is worth reporting: a feature
    with negative importance is a candidate for removal, and saying so is part of the Cycle stage.

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

    model = (clf or ev.default_classifier()).fit(X.iloc[:cut], y.iloc[:cut])
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
                   upto: list[str], slow_threshold_days: int = 60, clf=None) -> dict:
    """
    What the publication lag costs, measured by removing the slowest-published series.

    Quarterly series are two to four months old at a typical meeting. Refitting without
    them tells you how much of your model's performance rests on figures that were already
    stale when the Board met.
    """
    ages = staleness_table(panel, tiers)
    slow = ages.loc[ages["median_age_days"] >= slow_threshold_days, "series"].tolist()
    print(f"  {len(slow)} of {len(ages)} macro series are >={slow_threshold_days}d stale")

    y = panel[target]
    full = ev.score(ev.rolling_origin(ev.build_design(panel, tiers, upto), y, clf), classes)
    t2 = {**tiers, "macro": [c for c in tiers.get("macro", []) if c not in slow]}
    fast = ev.score(ev.rolling_origin(ev.build_design(panel, t2, upto), y, clf), classes)

    print(f"  all macro        acc={full['accuracy']:.3f} bal={full['balanced_accuracy']:.3f}")
    print(f"  fast macro only  acc={fast['accuracy']:.3f} bal={fast['balanced_accuracy']:.3f}"
          f"   (delta {fast['accuracy']-full['accuracy']:+.3f})")
    return {"slow_series": slow, "ages": ages.round(1).to_dict("records"),
            "with_slow": full, "without_slow": fast,
            "d_accuracy": fast["accuracy"] - full["accuracy"],
            "d_balanced_accuracy": (fast["balanced_accuracy"]
                                    - full["balanced_accuracy"])}

"""
The supplied policy-cycle model, and everything you need to interrogate it. SUPPLIED.

YOU DO NOT FIT THIS MODEL. It is trained, tuned and exported for you, because fitting it is
procedural and the marks are for what you do with it afterwards.

WHAT YOU GET

    load_model()        the fitted classifier - ONE fit on the meetings whose labels had
                        resolved by TRAIN_END, NOT a rolling-origin object
    importances()       permutation importance on HELD-OUT meetings, per feature and tier
    partial_dependence()  how the predicted state changes as one feature moves
    regime_probabilities()  easing / stable / hardening probability for all 211 meetings
    performance()       out-of-sample accuracy against all four baselines

WHAT THE MODEL IS
    A gradient-boosted classifier predicting `y_cycle` - the direction of the cash rate over
    the 182 days after each meeting - from the persistence, macro, market and text tiers.

    TWO ACCURACIES, AND ONLY ONE OF THEM IS HONEST. Under rolling-origin evaluation WITHOUT
    the label-availability embargo it reaches about 0.805; WITH the embargo - training only
    on labels that had actually resolved at each prediction date - it reaches 0.570. The
    second is the real one. The baseline that matters is not the 0.266 majority rate but
    `current_decision` at 0.656, so the honest number LOSES to the trivial rule by nine
    points.

    It is not a good model, and it is not a causal model. Telling the difference is your job in
    the Cycle stage. A feature can dominate the importances for at least four different
    reasons: it causes the outcome, the outcome causes it, something else causes both, or it
    is a proxy for a variable that does one of those. The model cannot distinguish them and
    neither can an LLM asked to explain the model. Only argument from mechanism and evidence
    can.

BUILDING IT
    `python src/model_card.py` refits and re-exports everything. You should not need to -
    the artefacts are committed - but it is here so nothing is a black box.

WRITES outputs/model_card/*
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence, permutation_importance

import config
import evaluation as ev

# The train/holdout boundary. Everything the card exports - the fitted model, its
# importances, its partial dependences - refers to this one split, so the object a
# student interrogates is the object whose importances they are reading.
TRAIN_END = "2018-12-31"

# The Shock stage shocks a model with no text tier: a scenario has no minutes.
SHOCK_TIERS = ["persistence", "macro", "market"]

CARD = config.MODEL_CARD
TIERS = ["persistence", "macro", "market", "text"]
CLASSES = [0, 1, 2]
STATE = config.CYCLE_STATES          # {0: easing, 1: stable, 2: hardening}
N_TOP_PDP = 8


def _panel_and_tiers() -> tuple[pd.DataFrame, dict]:
    """
    The FROZEN instructor panel, not your own.

    Cycle must interrogate the same model for every team. Your own panel carries your own
    constructs, so a model built from it would differ team to team and the stage would not
    be comparable - or markable. `panel_frozen.parquet` and `tiers_frozen.json` are shipped
    with the repository and are what this module reads.
    """
    src = config.FROZEN_PANEL if config.FROZEN_PANEL.exists() else (
        config.DATA_PROCESSED / "panel.parquet")
    p = pd.read_parquet(src)
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    p = p.set_index("meeting_date").sort_index()
    tsrc = config.FROZEN_TIERS if config.FROZEN_TIERS.exists() else (
        config.DATA_PROCESSED / "tiers.json")
    tiers = json.loads(tsrc.read_text())
    return p, tiers


def _classifier():
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.06,
        random_state=config.REGIME_SEED)


# ---------------------------------------------------------------------------
# Loading the exported artefacts - this is what you will actually call
# ---------------------------------------------------------------------------

def load_model():
    """The fitted classifier, plus the FULL design matrix - all 211 meetings.

    NOT "the matrix it was trained on": the fit uses a subset (the rows with a
    resolved label, under the rolling-origin embargo), and the exported frame is
    the whole panel so that scoring and inspection have every row available.
    Confusing the two overstates how much data the model actually saw.
    """
    if not (CARD / "model.joblib").exists():
        raise FileNotFoundError(
            f"model card not built - run `python src/model_card.py` first")
    return joblib.load(CARD / "model.joblib"), pd.read_parquet(CARD / "design.parquet")


def importances() -> pd.DataFrame:
    """
    Permutation importance, measured on held-out meetings.

    NEGATIVE IMPORTANCE IS A REAL RESULT. It means the model scored better when that feature
    was shuffled - the feature was actively misleading it. On the 73 held-out meetings, with
    collinear predictors, this is common; a feature with negative importance is a candidate
    for removal, not a bug to be hidden.

    The interval is MONTE-CARLO precision over the permutation repeats (`mc_lo`/`mc_hi`),
    not a confidence interval for another sample.
    """
    return pd.read_parquet(CARD / "importances.parquet")


def partial_dependences() -> dict[str, pd.DataFrame]:
    """
    For each of the top features: how the predicted probability of each state changes as
    that feature moves across its range, holding everything else at its observed values.

    A partial dependence is a STATEMENT ABOUT THE MODEL, not about the economy. If the curve
    for `slope_cash3y` slopes upward into 'hardening', that tells you the model has learned
    an association. It tells you nothing about whether moving the yield curve would move RBA
    policy.
    """
    out = {}
    for f in (CARD / "pdp").glob("*.parquet"):
        out[f.stem] = pd.read_parquet(f)
    return out


def regime_probabilities() -> pd.DataFrame:
    """Predicted easing / stable / hardening probability for every meeting."""
    df = pd.read_parquet(CARD / "regime_probabilities.parquet")
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df.set_index("meeting_date")


def performance() -> dict:
    return json.loads((CARD / "performance.json").read_text())


def summary() -> None:
    """`python -c "import model_card; model_card.summary()"` - what the model looks like."""
    perf = performance()
    m, l = perf["model_embargoed"], perf["model_no_embargo"]
    print(f"  out-of-sample, LABEL-EMBARGOED: acc {m['accuracy']:.3f}, "
          f"balanced {m['balanced_accuracy']:.3f}, n={m['n']}")
    print(f"  the same protocol WITHOUT the embargo: acc {l['accuracy']:.3f} "
          f"(+{perf['embargo_cost_accuracy']:.3f})")
    print("    that gap is training on outcomes nobody could have observed yet - "
          "see rolling_origin()")
    for k, v in perf["baselines"].items():
        flag = "   <-- NOT BEATEN" if v["accuracy"] >= m["accuracy"] else ""
        print(f"    baseline {k:18s} acc {v['accuracy']:.3f}{flag}")
    if not perf.get("beats_current_decision", False):
        print("")
        print("  THE SUPPLIED MODEL DOES NOT BEAT THE CURRENT-DECISION BASELINE.")
        print("  That is the finding, not a bug. Read the Cycle stage of the brief.")
    imp = importances()
    print("")
    print(f"  top features (permutation importance on the meetings after "
          f"{perf['train_end']}):")
    for r in imp.head(8).itertuples():
        sig = "" if r.mc_lo > 0 else "  (interval spans zero)"
        print(f"    {r.feature:28s} {r.importance:+.4f}  ({r.tier}){sig}")
    neg = imp[imp["importance"] < 0]
    print("")
    print(f"  {len(neg)} of {len(imp)} features have NEGATIVE importance; "
          f"{perf['n_features_significant']} have a Monte-Carlo interval clear of zero")
    print("  That interval is the precision of 30 permutation shuffles on THIS held-out "
          "sample.")
    print("  It is not evidence that a feature would matter in another sample - more "
          "shuffles would")
    print("  narrow it without adding information. Do not call it significance.")
    print(f"  by tier: {imp.groupby('tier')['importance'].sum().round(4).to_dict()}")


# ---------------------------------------------------------------------------
# Building the card
# ---------------------------------------------------------------------------

def _plot_pdp(feature: str, tbl: pd.DataFrame) -> None:
    """
    One PNG per partial dependence. The brief asks students to attach these; shipping only
    Parquet tables and then asking for plots made every team write the same throwaway
    matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {"easing": "#2f7ab5", "stable": "#9aa0a6", "hardening": "#c1443c"}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for state, g in tbl.groupby("state"):
        ax.plot(g["value"], g["avg_probability"], label=state,
                color=colour.get(state, None), lw=1.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("average predicted probability")
    ax.set_title(f"Partial dependence: {feature}\n"
                 f"in-sample, from the exported model - not a causal effect",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CARD / "pdp_plots" / f"{feature}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    """
    Build the card. Every interrogation artefact refers to ONE fitted model.

    THE SPLIT. `TRAIN_END` divides the sample. The exported model is fitted on every row
    whose LABEL had resolved by `TRAIN_END`; the permutation importances are measured on the
    meetings after it. An earlier version fitted importances on the first 80 rows, PDPs on
    the full sample, and reported rolling-origin accuracy from a third fit - three different
    models presented as one, so a feature could be "important" in an object whose partial
    dependence you were also reading.

    THE EMBARGO. Both accuracy figures are reported, deliberately. `y_cycle` at meeting M
    describes the 182 days after M, so at a prediction date only labels older than 182 days
    had resolved. Training on everything earlier - the obvious implementation - shows the
    model six months of outcomes nobody had yet. The gap between the two numbers is the
    single most important thing in this stage.
    """
    if CARD.exists():
        # Rebuild atomically. Leaving the old directory in place stranded 22 PDP files from
        # a run when N_TOP_PDP was larger, and a stale artefact is worse than a missing one.
        shutil.rmtree(CARD)
    (CARD / "pdp").mkdir(parents=True)
    (CARD / "pdp_plots").mkdir(parents=True)

    panel, tiers = _panel_and_tiers()
    X = ev.build_design(panel, tiers, TIERS)
    y = panel["y_cycle"]
    ok = y.notna()
    Xf, yf = X[ok], y[ok]
    avail = pd.to_datetime(panel["y_cycle_available_on"]).reindex(yf.index)

    # ---- performance, with and without the embargo -----------------------------------------
    print("  rolling-origin evaluation, WITHOUT the label embargo (the leaky number)")
    # DELIBERATE, and now explicit: this arm exists to show what the leak is worth. The
    # flag is what makes it a teaching comparison rather than an accident.
    leaky = ev.rolling_origin(Xf, yf, _classifier(), available_on=None,
                              allow_unresolved_labels=True)
    s_leaky = ev.score(leaky, CLASSES)
    print(f"    acc {s_leaky['accuracy']:.3f}  bal {s_leaky['balanced_accuracy']:.3f}  "
          f"n={s_leaky['n']}")

    print("  rolling-origin evaluation, WITH the label embargo (the honest number)")
    preds = ev.rolling_origin(Xf, yf, _classifier(), available_on=avail)
    s_emb = ev.score(preds, CLASSES)
    base = ev.baselines(panel, y, on_dates=preds.index)
    print(f"    acc {s_emb['accuracy']:.3f}  bal {s_emb['balanced_accuracy']:.3f}  "
          f"n={s_emb['n']}  from {preds.index.min().date()}")
    for k, v in base.items():
        flag = "   <-- the model does not beat this" if (
            v["accuracy"] >= s_emb["accuracy"]) else ""
        print(f"      baseline {k:18s} acc {v['accuracy']:.3f}{flag}")

    perf = {"model_embargoed": s_emb,
            "model_no_embargo": s_leaky,
            "embargo_cost_accuracy": round(s_leaky["accuracy"] - s_emb["accuracy"], 4),
            "baselines": base,
            "baseline_dates": [str(d.date()) for d in (preds.index.min(),
                                                       preds.index.max())],
            "beats_current_decision": bool(
                s_emb["accuracy"] > base.get("current_decision", {}).get("accuracy", 0)),
            "n_features": int(X.shape[1]), "n_meetings_labelled": int(len(Xf))}

    # ---- one model, one training set, for every interrogation artefact ---------------------
    train_idx = avail.index[avail <= pd.Timestamp(TRAIN_END)]
    hold_idx = yf.index[yf.index > pd.Timestamp(TRAIN_END)]
    print(f"  exported model: fitted on {len(train_idx)} meetings whose label had resolved "
          f"by {TRAIN_END}")
    print(f"  held-out for interrogation: {len(hold_idx)} meetings after {TRAIN_END}")
    model = _classifier().fit(Xf.loc[train_idx], yf.loc[train_idx])
    joblib.dump(model, CARD / "model.joblib")
    X.to_parquet(CARD / "design.parquet")
    perf["train_end"] = TRAIN_END
    perf["n_train"] = int(len(train_idx))
    perf["n_holdout"] = int(len(hold_idx))
    perf["holdout"] = ev.score(
        pd.DataFrame({"y_true": yf.loc[hold_idx],
                      "y_pred": model.predict(Xf.loc[hold_idx]),
                      **{f"p_{int(c)}": model.predict_proba(Xf.loc[hold_idx])[:, j]
                         for j, c in enumerate(model.classes_)}}), CLASSES)
    print(f"    holdout acc {perf['holdout']['accuracy']:.3f}")

    print("  permutation importance, measured on the held-out meetings")
    # WHAT THE INTERVAL BELOW IS, AND IS NOT.
    #
    # `permutation_importance` shuffles ONE COLUMN of THIS held-out sample, against THIS
    # fitted model, n_repeats times. `importances_std` is the spread across those shuffles
    # (scikit-learn documents it exactly that way), so mean +/- 1.96*sd/sqrt(n_repeats) is
    # the Monte Carlo precision of the shuffling: how well 30 shuffles pin down the number
    # you would get from infinitely many shuffles of this same data.
    #
    # It is NOT a confidence interval for the importance in the population. It does not
    # account for a different historical sample, for the overlapping policy cycles that
    # make these 73 meetings far less independent than they look, or for refitting the
    # model. Doing MORE shuffles narrows it without adding any evidence - which is the
    # giveaway that it cannot be measuring sampling uncertainty.
    #
    # The columns are named `mc_lo`/`mc_hi` for that reason. A feature whose interval
    # clears zero has a shuffling effect this sample can resolve; it has NOT been shown to
    # matter in general, and nothing in the report may claim it has.
    n_repeats = 30
    r = permutation_importance(model, Xf.loc[hold_idx], yf.loc[hold_idx],
                               n_repeats=n_repeats,
                               random_state=config.REGIME_SEED,
                               scoring="balanced_accuracy")
    tier_of = {c: t for t in TIERS for c in tiers.get(t, []) if c in X.columns}
    imp = (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                         "sd": r.importances_std})
           .assign(tier=lambda d: d["feature"].map(tier_of),
                   interval_kind="monte_carlo_over_permutation_repeats",
                   n_repeats=n_repeats,
                   mc_lo=lambda d: d["importance"] - 1.96 * d["sd"] / np.sqrt(n_repeats),
                   mc_hi=lambda d: d["importance"] + 1.96 * d["sd"] / np.sqrt(n_repeats))
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    imp.to_parquet(CARD / "importances.parquet", index=False)
    n_sig = int((imp["mc_lo"] > 0).sum())
    print(f"    {(imp['importance'] < 0).sum()} of {len(imp)} features negative; "
          f"only {n_sig} have a Monte-Carlo interval clear of zero")
    print("    (that interval is permutation precision on THIS sample, not evidence "
          "about a new one)")
    perf["n_features_negative"] = int((imp["importance"] < 0).sum())
    perf["n_features_mc_interval_clear_of_zero"] = n_sig
    # kept under the old name so existing readers do not break; both mean the
    # Monte-Carlo permutation interval, never statistical significance
    perf["n_features_significant"] = n_sig

    print(f"  partial dependences for the top {N_TOP_PDP} features, from the SAME model")
    for f in imp.head(N_TOP_PDP)["feature"]:
        res = partial_dependence(model, Xf.loc[train_idx], [f], kind="average",
                                 grid_resolution=25)
        grid = res["grid_values"][0]
        avg = np.asarray(res["average"])
        rows = [pd.DataFrame({"value": grid,
                              "state": STATE[int(model.classes_[i])],
                              "avg_probability": avg[i]})
                for i in range(avg.shape[0])]
        tbl = pd.concat(rows)
        tbl.to_parquet(CARD / "pdp" / f"{f}.parquet", index=False)
        _plot_pdp(f, tbl)
    print(f"    {N_TOP_PDP} tables and {N_TOP_PDP} plots written")

    print("  regime probabilities for every meeting")
    proba = model.predict_proba(X)
    rp = pd.DataFrame(proba, columns=[STATE[int(c)] for c in model.classes_],
                      index=X.index).reset_index()
    rp.columns = ["meeting_date"] + [c for c in rp.columns if c != "meeting_date"]
    rp["predicted"] = [STATE[int(c)] for c in model.classes_[proba.argmax(axis=1)]]
    rp["actual"] = [STATE[int(v)] if pd.notna(v) else None for v in panel["y_cycle"]]
    rp["in_training_set"] = rp["meeting_date"].isin(train_idx)
    rp["label_resolved"] = panel["y_cycle"].notna().to_numpy()
    rp.to_parquet(CARD / "regime_probabilities.parquet", index=False)
    print(f"    {len(rp)} rows: {int(rp['in_training_set'].sum())} in the training set "
          f"(in-sample), {int((~rp['in_training_set']).sum())} outside it; "
          f"{int(rp['label_resolved'].sum())} have a resolved label")
    perf["regime_probability_populations"] = {
        "rows_described": int(len(rp)),
        "in_training_set_in_sample": int(rp["in_training_set"].sum()),
        "outside_training_set": int((~rp["in_training_set"]).sum()),
        "labels_resolved": int(rp["label_resolved"].sum()),
        "rolling_origin_predictions": int(s_emb["n"])}

    # ---- the frozen model SHOCK loads ------------------------------------------------------
    # Construct-free, and the SAME family as this card. Shock used to refit a multinomial
    # logit at runtime while the brief explained its behaviour in terms of tree extrapolation,
    # so the documentation described a model nobody was running.
    print("  exporting the construct-free Shock model (same family, no text tier)")
    Xs = ev.build_design(panel, tiers, SHOCK_TIERS)
    Xs_f = Xs[ok]
    shock_model = _classifier().fit(Xs_f.loc[train_idx], yf.loc[train_idx])
    joblib.dump(shock_model, CARD / "shock_model.joblib")
    Xs.to_parquet(CARD / "shock_design.parquet")
    # The rows the model was ACTUALLY FITTED ON, exported separately. Support has to be
    # judged against these: `shock_design.parquet` holds all 211 meetings, so using it as
    # the reference made the 2026 base row a member of its own comparison set and its
    # nearest-neighbour distance identically zero. The question "does this shock leave the
    # data the model learned from" was being asked of the wrong population.
    Xs_f.loc[train_idx].to_parquet(CARD / "shock_train_design.parquet")
    print(f"    {Xs.shape[1]} features, no text tier, trained on the same {len(train_idx)} "
          f"meetings")
    perf["shock_model"] = {"family": type(shock_model).__name__,
                           "n_training_rows_exported": int(len(train_idx)),
                           "n_features": int(Xs.shape[1]),
                           "tiers": SHOCK_TIERS,
                           "n_train": int(len(train_idx))}

    # ---- SUPPORT CALIBRATION -----------------------------------------------------------
    # How unusual is an ORDINARY unseen meeting? Without this the support test has no scale
    # and defaults to a wrong one.
    #
    # THE MISCALIBRATION THIS REPLACES. Support used to be judged two ways, and both flagged
    # essentially all real data:
    #
    #   MARGINALLY, against the training min/max. All 82 held-out meetings breach it - a
    #   median of 6 variables each, and not one meeting is clean. A rule that fails 100% of
    #   ordinary data is not a support test.
    #
    #   JOINTLY, against the MEDIAN nearest-neighbour distance among training rows. That
    #   number is 2.77, and it is small because consecutive RBA meetings are near-duplicates
    #   - it measures how far apart adjacent months are, not how spread the data is. Ordinary
    #   held-out meetings sit at 8.75, which is 3.2x it, so they too were "out of support".
    #
    # The right reference is the distribution of held-out-to-training distances: a row is
    # outside support when it is more unusual than real unseen meetings ever are.
    print("  calibrating support against the held-out meetings")
    from scipy.spatial import cKDTree
    hold_rows = Xs.index.difference(train_idx)
    scols = [c for c in Xs_f.columns if Xs_f.loc[train_idx, c].notna().sum() > 10]
    Atr = Xs_f.loc[train_idx, scols].astype(float)
    mu, sdv = Atr.mean(), Atr.std().replace(0, 1)
    Ztr = ((Atr - mu) / sdv).fillna(0.0).to_numpy()
    kdt = cKDTree(Ztr)
    Zho = ((Xs.loc[hold_rows, scols].astype(float) - mu) / sdv).fillna(0.0).to_numpy()
    d_ho = kdt.query(Zho, k=1)[0]
    breaches = [int(sum(1 for c in scols
                        if not (Atr[c].min() <= Xs.loc[r, c] <= Atr[c].max())))
                for r in hold_rows]
    calib = {
        "n_training": int(len(train_idx)),
        "n_heldout": int(len(hold_rows)),
        "columns": scols,
        "joint_heldout_p50": round(float(np.percentile(d_ho, 50)), 3),
        "joint_heldout_p95": round(float(np.percentile(d_ho, 95)), 3),
        "joint_heldout_p99": round(float(np.percentile(d_ho, 99)), 3),
        "joint_heldout_max": round(float(d_ho.max()), 3),
        "marginal_breaches_p50": int(np.percentile(breaches, 50)),
        "marginal_breaches_p95": int(np.percentile(breaches, 95)),
        "marginal_breaches_p99": int(np.percentile(breaches, 99)),
        "marginal_breaches_max": int(max(breaches)),
        "note": ("thresholds are the 99th percentile of how unusual a REAL unseen meeting "
                 "is, measured against the rows the model was fitted on"),
    }
    (CARD / "support_calibration.json").write_text(json.dumps(calib, indent=2))
    print(f"    ordinary unseen meetings: joint distance median "
          f"{calib['joint_heldout_p50']}, p99 {calib['joint_heldout_p99']}; "
          f"{calib['marginal_breaches_p50']} variables typically outside the training range")
    perf["support_calibration"] = calib

    (CARD / "performance.json").write_text(json.dumps(perf, indent=2, default=float))
    print(f"\n  model card written to {CARD}")
    return perf


if __name__ == "__main__":
    run()
    print()
    summary()

"""
Meeting calendar, the targets, and the tiered modelling panel. PRE-BUILT.

===========================================================================================
THE PREDICTION ORIGIN: THE END OF MEETING T
===========================================================================================
Every feature in this panel is something a forecaster standing in the room at the close of
meeting T could have known, and every target describes what happens afterwards. That single
choice fixes four things that were previously incoherent:

  KNOWN at the origin              NOT known at the origin
  ------------------------------   --------------------------------------------------
  the decision taken at T          the minutes of meeting T (published ~14 days later)
  macro releases published by T    any macro release after T
  market prices at the close of T  any market move after T
  the minutes of meeting T-1       what the Board does at T+1
  (public >= 15 days before T)

WHY THE TEXT TIER IS LAGGED ONE MEETING
    Minutes for meeting T publish about 14 days after it, so at the close of T they do not
    exist. The text tier therefore carries the constructs from meeting T-1, whose minutes
    were public before T. The shortest meeting gap in the corpus is 15 days, so this holds
    for every meeting without exception - `_assert_text_available()` checks it.

WHY THE TARGET STARTS AFTER THE DECISION TAKES EFFECT
    An RBA decision announced at meeting T takes effect the following business day, so the
    cash rate ON the meeting date is still the OLD rate. Measuring the forward change from
    that stale value puts meeting T's own decision inside its own target - which is why
    100% of hikes used to be labelled "hardening" and a trivial current-decision lookup
    scored 61% on a target the model was credited with predicting at 82%.

    The forward window now runs from the POST-DECISION rate. The decision at T is a
    feature; it is no longer also the answer.

WHY SOME TARGETS ARE NaN
    A 182-day horizon that has not finished cannot be labelled. Any meeting whose window
    extends past the last available cash rate returns NaN rather than being scored on a
    partial horizon.

THE TARGETS
    y_cycle     the net direction of the cash rate over the 182 days AFTER meeting T's
                decision takes effect: easing (0) / stable (1) / hardening (2).
                It is a NET endpoint-to-endpoint measure, not a path: a cut reversed by a
                later hike reads as stable. Adjacent meetings share most of their window,
                so the labels are heavily autocorrelated and there are far fewer
                independent episodes than there are rows.
    y_decision  the decision at the NEXT meeting: cut (-1) / hold (0) / hike (+1).

WRITES data/processed/panel.parquet, data/processed/tiers.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import data_macro
import data_rates

MARKET_FEATURES = ["rv_5", "rv_21", "rv_63", "parkinson_21", "downside_21",
                   "aud_vol_21", "aud_ret", "vix"]
RATE_FEATURES = ["cash_rate", "bab_spread", "slope_cash3y", "slope_3y10y",
                 "bond_3y", "bond_10y"]
PERSISTENCE_FEATURES = ["trailing_change_182d", "meetings_since_change",
                        "last_change_sign", "decision"]


def meeting_calendar() -> pd.DataFrame:
    """One row per RBA monetary policy meeting, from the supplied calendar."""
    dates = pd.read_csv(config.MEETING_CALENDAR, parse_dates=["meeting_date"])
    # REQUIRED-INPUT CONTRACT: the calendar is vendored, so a short, duplicated or
    # gap-ridden calendar is a corrupt input, not a configuration.
    problems = []
    if "meeting_date" not in dates.columns or dates["meeting_date"].isna().any():
        problems.append("meeting_date column missing or carries unparseable dates")
    else:
        md = dates["meeting_date"]
        if md.duplicated().any():
            problems.append("duplicated meeting dates")
        if len(md) != config.CORPUS_MEETINGS:
            problems.append(f"{len(md)} meetings - the frozen corpus has "
                            f"{config.CORPUS_MEETINGS}")
        if md.min() > pd.Timestamp(config.CORPUS_START) + pd.Timedelta(days=60):
            problems.append(f"first meeting {md.min().date()} is far past the corpus "
                            f"start - early rows are missing")
        if md.max() != pd.Timestamp(config.CORPUS_LAST_MEETING):
            problems.append(f"last meeting {md.max().date()} is not the frozen corpus "
                            f"end {config.CORPUS_LAST_MEETING} - the calendar is "
                            f"truncated or extended")
        gaps = md.sort_values().diff().dt.days.dropna()
        if len(gaps) and gaps.max() > 120:
            problems.append(f"a {int(gaps.max())}-day gap between meetings - rows are "
                            f"missing")
    if problems:
        raise RuntimeError(f"meeting calendar failed its input contract: {problems}. "
                           f"Restore data/raw and re-run.")
    cal = pd.DataFrame(index=pd.DatetimeIndex(sorted(dates["meeting_date"]),
                                              name="meeting_date"))
    cal["minutes_published"] = cal.index + pd.Timedelta(
        days=config.MINUTES_PUBLICATION_LAG_DAYS)
    cal["meeting_no"] = np.arange(len(cal))
    cal["next_meeting"] = cal.index.to_series().shift(-1).to_numpy()
    cal["days_to_next"] = (cal["next_meeting"] - cal.index).dt.days
    return cal


def attach_decisions(cal: pd.DataFrame, decisions: pd.DataFrame,
                     tolerance_days: int = 2) -> pd.DataFrame:
    """
    Join A2's announced changes onto the calendar.

    A2 lists ONLY the meetings that moved, so every meeting it does not name is a hold.
    That default is applied here, once, explicitly - a merge that fills holds by accident
    is indistinguishable from one that fills them because the join failed.
    """
    dec = decisions.copy()
    if not isinstance(dec.index, pd.DatetimeIndex):
        dec = dec.set_index("announced_date")
    dec = dec.sort_index().loc[cal.index.min() - pd.Timedelta(days=7):]

    left = pd.DataFrame({"meeting_date": cal.index}).sort_values("meeting_date")
    right = dec.reset_index().sort_values("announced_date")
    matched = pd.merge_asof(left, right, left_on="meeting_date",
                            right_on="announced_date", direction="forward",
                            tolerance=pd.Timedelta(days=tolerance_days))

    out = cal.copy()
    out["change_pct"] = matched["change_pct"].to_numpy()
    out["announced_date"] = matched["announced_date"].to_numpy()

    in_window = dec.loc[cal.index.min():cal.index.max() + pd.Timedelta(days=tolerance_days)]
    unmatched = set(in_window.index) - set(pd.Series(out["announced_date"]).dropna())
    if unmatched:
        print(f"  WARNING: {len(unmatched)} announced changes matched no meeting")

    out["change_pct"] = out["change_pct"].fillna(0.0)     # the explicit hold default
    out["decision"] = np.sign(out["change_pct"]).astype(int)
    return out


# An RBA decision announced at meeting T takes effect the FOLLOWING business day, so the
# cash rate stamped on the meeting date is still the pre-decision rate. Reading the
# post-decision rate a few days later is what keeps meeting T's own decision out of its own
# forward window. Seven days is comfortably after the change and comfortably before the next
# meeting, whose minimum gap in this corpus is 15 days.
DECISION_EFFECTIVE_DAYS = 7


def _rate_at(cash: pd.Series, when) -> float:
    """Cash rate as at a date, or NaN if the series does not reach back that far."""
    i = cash.index.searchsorted(when, "right") - 1
    return np.nan if i < 0 else float(cash.iloc[i])


def _cum_change(cash: pd.Series, start, end, require_complete: bool = True) -> float:
    """
    Change in the cash rate between two dates.

    require_complete guards against RIGHT CENSORING. A 182-day horizon that has not
    finished cannot be labelled, and silently reading the last available rate would score a
    partial horizon as though it were whole - which previously labelled three 2026 meetings
    on as little as two months of a six-month window.
    """
    if require_complete and end > cash.index.max():
        return np.nan
    a, b = _rate_at(cash, start), _rate_at(cash, end)
    return np.nan if (np.isnan(a) or np.isnan(b)) else b - a


def add_targets(m: pd.DataFrame, rates_daily: pd.DataFrame,
                window_days: int | None = None,
                threshold: float | None = None) -> pd.DataFrame:
    """
    Both targets, plus backward-looking cycle features.

    window_days and threshold are exposed so the target can be swept. The defaults are
    config.CYCLE_WINDOW_DAYS and config.CYCLE_THRESHOLD_PCT.
    """
    window_days = window_days or config.CYCLE_WINDOW_DAYS
    threshold = config.CYCLE_THRESHOLD_PCT if threshold is None else threshold

    out = m.copy()
    cash = rates_daily["cash_rate"].dropna()

    out["y_decision"] = out["decision"].shift(-1)
    out["y_decision_change"] = out["change_pct"].shift(-1)

    # The forward window starts from the POST-DECISION rate, so meeting T's own decision is
    # a feature and not also part of its answer. It ends 182 days after the meeting, and is
    # NaN whenever that horizon has not finished.
    eff = pd.Timedelta(days=DECISION_EFFECTIVE_DAYS)
    out["rate_after_decision"] = [_rate_at(cash, d + eff) for d in out.index]
    out["forward_change"] = [
        _cum_change(cash, d + eff, d + pd.Timedelta(days=window_days))
        for d in out.index]
    out["y_cycle"] = np.select(
        [out["forward_change"] <= -threshold, out["forward_change"] >= threshold],
        [0, 2], default=1).astype(float)
    out.loc[out["forward_change"].isna(), "y_cycle"] = np.nan

    # ---- WHEN EACH LABEL BECAME KNOWABLE ---------------------------------------------------
    # This is not decoration. `y_cycle` at meeting M summarises the 182 days AFTER M, so it
    # cannot be used as a TRAINING label until 182 days after M have elapsed. Training on
    # every preceding row - which is the obvious thing to do, and what an earlier version of
    # `rolling_origin` did - hands the model the last six months of outcomes that nobody
    # could have observed yet. On this data that single mistake was worth 23 accuracy points.
    #
    # `y_decision` is the decision at the NEXT meeting, so it resolves on that meeting's date.
    out["y_cycle_available_on"] = out.index + pd.Timedelta(days=window_days)
    nxt = pd.Series(out.index, index=out.index).shift(-1)
    out["y_decision_available_on"] = nxt

    # Backward-looking, therefore legal as features at the close of meeting T. Policy is
    # strongly autocorrelated, so these are the strongest naive predictors and the baseline
    # the model has to beat. Trailing change is measured to the post-decision rate, so it
    # includes today's move - which the forecaster knows.
    out["trailing_change_182d"] = [
        _cum_change(cash, d - pd.Timedelta(days=182), d + eff, require_complete=False)
        for d in out.index]
    out["meetings_since_change"] = out.groupby((out["decision"] != 0).cumsum()).cumcount()
    out["last_change_sign"] = out["decision"].replace(0, np.nan).ffill().fillna(0).astype(int)
    return out


# Rolling-window features legitimately start blank while their window warms up. Each
# feature's first usable value must appear within its window plus a grace period; a
# feature whose values only begin years later is a truncated column, not a warm-up.
_MARKET_WARMUP_DAYS = {"rv_5": 5, "rv_21": 21, "rv_63": 63, "parkinson_21": 21,
                       "downside_21": 21, "aud_vol_21": 21, "aud_ret": 1, "vix": 0}


def _check_market(market: pd.DataFrame) -> None:
    """
    REQUIRED-INPUT CONTRACT for the vendored market file, PER FEATURE. A frame whose
    index spanned the corpus once passed while one required column held only its final
    twenty observations - dataframe-level coverage says nothing about a column.
    """
    problems = [c for c in MARKET_FEATURES if c not in market.columns]
    if len(market) < 4000:
        problems.append(f"only {len(market)} daily rows")
    if not market.index.is_unique:
        problems.append("duplicated dates")
    if len(market):
        if market.index.min() > (pd.Timestamp(config.CORPUS_START)
                                 + pd.Timedelta(days=7)):
            problems.append(f"market data starts {market.index.min().date()}, after "
                            f"the corpus start")
        gap = market.index.to_series().diff().dt.days.max()
        if gap and gap > 10:
            problems.append(f"a {int(gap)}-day hole inside the market series")
        start = pd.Timestamp(config.CORPUS_START)
        end = pd.Timestamp(config.CORPUS_LAST_MEETING)
        for c in MARKET_FEATURES:
            if c not in market.columns:
                continue
            col = market[c].dropna()
            if col.empty:
                problems.append(f"{c} is empty")
                continue
            if not np.isfinite(col.to_numpy()).all():
                problems.append(f"{c} contains non-finite values")
            warmup = _MARKET_WARMUP_DAYS.get(c, 0)
            if col.index.min() > start + pd.Timedelta(days=2 * warmup + 30):
                problems.append(f"{c} first usable value {col.index.min().date()} is "
                                f"far past its warm-up window")
            if col.index.max() < end - pd.Timedelta(days=7):
                problems.append(f"{c} ends {col.index.max().date()}, before the last "
                                f"meeting {config.CORPUS_LAST_MEETING}")
            in_corpus = market.loc[start:end, c]
            if len(in_corpus) and in_corpus.isna().mean() > 0.20:
                problems.append(f"{c} is {in_corpus.isna().mean():.0%} missing inside "
                                f"the corpus window")
    if problems:
        raise RuntimeError(f"market data failed its input contract: {problems}. "
                           f"Restore data/raw and re-run.")


def _asof_daily(daily: pd.DataFrame, cols: list[str],
                dates: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    """
    Value of each daily series as at the last trading day ON OR BEFORE each meeting.

    Backward, no tolerance cap: a meeting on a public holiday picks up the previous
    session rather than going missing. Market prices are known same-day, so no publication
    lag applies here - unlike the macro block.
    """
    have = [c for c in cols if c in daily.columns]
    if set(cols) - set(have):
        print(f"  WARNING: {label} missing {sorted(set(cols) - set(have))}")
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates).sort_values()})
    out = left.copy()
    # PER COLUMN, not one shared as-of row. The daily frames have per-series gaps, so a
    # single backward join lands every column on the same date and picks up whatever
    # happened to be missing there - which silently cost `bab_spread` and the bond series
    # about 25 observations each once the pre-decision join was added.
    for c in have:
        s = daily[c].dropna()
        if s.empty:
            out[c] = np.nan
            continue
        right = s.reset_index()
        right.columns = ["date", c]
        m = pd.merge_asof(left, right.sort_values("date"), on="date", direction="backward")
        out[c] = m[c].to_numpy()
    return out.set_index("date")


def load_text() -> pd.DataFrame | None:
    """
    The seven constructs from the Words stage, if they have been produced yet.

    Returns None before the Words stage is run, so the panel builds on a fresh checkout.
    """
    if not config.CONSTRUCT_SCORES.exists():
        print("  (no construct scores yet - text tier will be empty until Words is run)")
        return None
    # THE construct-score contract, shared with Replay and Shock, at the point of
    # consumption. The panel may build from a partial (development-only) table, so
    # reportable=False here; the reportable entry points demand the full meeting set.
    t = config.validate_construct_scores(reportable=False)
    t["meeting_date"] = pd.to_datetime(t["meeting_date"])
    return t.set_index("meeting_date").sort_index()


def _assert_text_available(panel: pd.DataFrame) -> None:
    """
    Every row's text must come from minutes that were PUBLIC before that meeting.

    Enforced rather than trusted. The shortest meeting gap in this corpus is 15 days and
    the publication lag is 14, so a one-meeting lag always clears - but if the calendar or
    the lag ever changes, this fails loudly instead of leaking quietly.
    """
    gaps = panel.index.to_series().diff().dt.days
    too_short = gaps[gaps <= config.MINUTES_PUBLICATION_LAG_DAYS].dropna()
    if len(too_short):
        raise AssertionError(
            f"{len(too_short)} meeting(s) fall within {config.MINUTES_PUBLICATION_LAG_DAYS} "
            f"days of the previous one, so the prior minutes were not yet public: "
            f"{[str(d.date()) for d in too_short.index]}")


def run() -> pd.DataFrame:
    dec, rates = data_rates.run()
    cal = meeting_calendar()
    print(f"  meetings: {len(cal)}, {cal.index.min().date()} -> {cal.index.max().date()}")
    per_year = cal.index.to_series().groupby(cal.index.year).size()
    print(f"  meetings per year: {per_year.to_dict()}")

    m = add_targets(attach_decisions(cal, dec), rates)
    dates = m.index

    macro = data_macro.run(dates)
    macro_cols = [c for c in macro.columns if not c.endswith("__age_days")]

    market = pd.read_parquet(config.MARKET_DATA)
    market["date"] = pd.to_datetime(market["date"])
    market = market.set_index("date").sort_index()
    _check_market(market)

    panel = (m
             .join(_asof_daily(rates, RATE_FEATURES, dates, "rates"), how="left")
             .join(_asof_daily(market, MARKET_FEATURES, dates, "market"), how="left")
             .join(macro, how="left"))

    # ---- PRE-DECISION MARKET STATE, for REPLAY ---------------------------------------------
    # The joins above take the last close ON OR BEFORE the meeting - which on a meeting day
    # is the meeting-day close. The RBA announces at 2:30pm Sydney and the ASX closes at
    # 4pm, with FX and bonds trading through, so that close already contains the decision.
    #
    # That is correct for CYCLE, whose estimand is explicitly "predict at the end of meeting
    # T, knowing T's decision". It is wrong for REPLAY, which is trying to predict the
    # decision: handing it the market's reaction to the announcement is handing it the
    # answer. So every market and rate feature is also computed as at the PREVIOUS BUSINESS
    # DAY and stored with a `__pre` suffix. `decision_replay.load_as_at()` swaps these in.
    prev_day = dates - pd.tseries.offsets.BDay(1)
    pre = (_asof_daily(rates, RATE_FEATURES, prev_day, "rates (pre-decision)")
           .join(_asof_daily(market, MARKET_FEATURES, prev_day, "market (pre-decision)")))
    pre.index = dates
    pre.columns = [f"{c}__pre" for c in pre.columns]
    panel = panel.join(pre, how="left")

    text = load_text()
    if text is not None and len(text) < 200:
        # Words scores development and validation in separate runs over disjoint
        # meetings, so the text tier is incomplete until BOTH have been run.
        print(f"  NOTE: construct scores cover only {len(text)} meetings. Run"
              f" `python src/text_features.py --validate` to complete the text tier;"
              f" until then the held-out meetings carry no construct values.")
    text_cols: list[str] = []
    if text is not None:
        raw = [c for c in config.TEXT_FEATURES if c in text.columns]
        # LAG ONE MEETING. Minutes for meeting T publish ~14 days after it, so at the close
        # of T they do not exist. Each row therefore carries the constructs from the
        # PREVIOUS meeting, whose minutes were public before T.
        lagged = text[raw].reindex(panel.index).shift(1)
        lagged.columns = raw
        panel = panel.join(lagged, how="left")
        text_cols = raw
        _assert_text_available(panel)
        print(f"  text tier lagged one meeting: "
              f"{int(panel[text_cols].notna().all(axis=1).sum())}/{len(panel)} rows complete "
              f"(the first meeting has no prior minutes)")

    tiers = {"persistence": PERSISTENCE_FEATURES, "macro": macro_cols,
             "market": MARKET_FEATURES + RATE_FEATURES, "text": text_cols}

    print(f"\n  panel: {len(panel)} meetings x {panel.shape[1]} columns")
    for t, cols in tiers.items():
        have = [c for c in cols if c in panel.columns]
        print(f"    tier {t:12s} {len(have):2d} features, "
              f"{panel[have].notna().all(axis=1).sum() if have else 0}/{len(panel)} complete")

    for name, col, labels in (("y_cycle", "y_cycle", config.CYCLE_STATES),
                              ("y_decision", "y_decision", config.DECISION_STATES)):
        d = panel[col].dropna().value_counts().sort_index()
        share = {labels[int(k)]: int(v) for k, v in d.items()}
        print(f"  {name:12s} {share}  majority {d.max()/d.sum():.1%}")

    panel.reset_index().to_parquet(config.DATA_PROCESSED / "panel.parquet", index=False)
    (config.DATA_PROCESSED / "tiers.json").write_text(json.dumps(tiers, indent=1))
    # PANEL PROVENANCE: which construct-score table this panel's text tier was built
    # from, by content hash. The submission suite compares it against the score file on
    # disk, so editing the scores after the panel was built - even consistently across
    # the partials and the combined table - leaves a panel that visibly rests on a
    # table that no longer exists.
    import hashlib
    from datetime import datetime, timezone
    config.atomic_write_text(
        config.DATA_PROCESSED / "panel.provenance.json",
        json.dumps({
            "construct_scores_sha256":
                (hashlib.sha256(config.CONSTRUCT_SCORES.read_bytes()).hexdigest()
                 if config.CONSTRUCT_SCORES.exists() else None),
            "n_meetings": len(panel),
            "written_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    return panel


if __name__ == "__main__":
    run()

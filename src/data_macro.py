"""
Inflation and macro structured data, publication timing enforced. PRE-BUILT.

THE POINT OF THIS MODULE IS THE PUBLICATION LAG, NOT THE DATA.

Every series here is stamped with the period it DESCRIBES, not the date it became known.
The June-quarter CPI describes 30 June but is not published until late July. A model that
reads the June figure at a 7 July meeting is using information nobody had, and it will
look far better than it deserves to.

So each series gets a publication date (period end + a lag), and the meeting panel is
built with a backward as-of join on THAT date. The lag is deliberately conservative:
where the real lag varies, the longer value is used, because over-lagging costs a little
accuracy while under-lagging invents it.

WRITES data/processed/macro_asof.parquet   (one row per RBA meeting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data_rates import _read_rba_csv

# series id -> (output name, which lag in config.PUBLICATION_LAGS)
SERIES = {
    # G1 Consumer price inflation (quarterly, ABS)
    "g1-data.csv": {
        "GCPIAGYP":      ("cpi_yoy", "cpi"),
        "GCPIOCPMTMYP":  ("trimmed_mean_yoy", "cpi"),
        "GCPIOCPMWMYP":  ("weighted_median_yoy", "cpi"),
        "GCPIAGSAQP":    ("cpi_qoq", "cpi"),
        "GCPIOCPMTMQP":  ("trimmed_mean_qoq", "cpi"),
    },
    # G3 Inflation expectations. GCONEXP (consumers) and GBONYLD (bond-implied) exist but
    # carry only 18 and 46 observations respectively - too sparse to survive the as-of join,
    # so the two well-populated survey measures are used instead.
    "g3-data.csv": {
        "GMAREXPY":   ("infexp_economists_1y", "inflation_expectations"),
        "GUNIEXPY":   ("infexp_union_1y", "inflation_expectations"),
        "GBUSEXP":    ("infexp_business_3m", "inflation_expectations"),
    },
    # H5 Labour force (monthly, ABS). The RBA has a dual mandate, so the unemployment rate
    # belongs in any reaction function.
    "h5-data.csv": {
        "GLFSURSA":   ("unemployment_rate", "labour"),
        "GLFSPRSA":   ("participation_rate", "labour"),
        "GLFSEPTSA":  ("employment", "labour"),
    },
    # H1 GDP and income (quarterly, ABS). Terms of trade matters more in Australia than in
    # most economies - it is the channel a commodity price shock reaches domestic income by.
    "h1-data.csv": {
        "GGDPCVGDPY": ("gdp_growth_yoy", "gdp"),
        "GGDPCVGDP":  ("real_gdp", "gdp"),
        "GOPITT":     ("terms_of_trade", "gdp"),
    },
    # H3 Monthly activity indicators - the timeliest read the Board has on demand.
    "h3-data.csv": {
        "GICWMICS":   ("consumer_sentiment", "activity"),
        "GICNBC":     ("business_conditions", "activity"),
        "GISPSDA":    ("dwelling_approvals", "activity"),
    },
}

# Series that are LEVELS rather than rates: a year-on-year change is more useful to the
# model and removes the trend that would otherwise dominate.
AS_GROWTH = {"employment", "real_gdp"}


# -------------------------------------------------------------------------------------------
# Geopolitical risk (Caldara & Iacoviello). Monthly, newspaper-based.
# -------------------------------------------------------------------------------------------
# This is in the panel because the SHOCK stage needs it. A Taiwan blockade or a Hormuz
# closure is a geopolitical-risk event before it is anything else, and without a GPR column
# the only proxies a team can shock are the downstream ones - the AUD, the VIX, the terms of
# trade - which forces them to guess the size of the second-order move while the first-order
# move has nowhere to go.
#
# It also gives the shock calibration something to anchor on: the GPR index has observed
# peaks for the Gulf War, 9/11 and the invasion of Ukraine, so "how many standard deviations
# is a Taiwan blockade" becomes a question with a historical reference rather than a guess.
GPR_SERIES = {"GPR": "gpr_index", "GPRT": "gpr_threats", "GPRA": "gpr_acts"}


def load_gpr() -> pd.DataFrame:
    """The GPR index in the same long format as the RBA series, with a publication lag."""
    if not config.GPR_FILE.exists():
        print(f"  WARNING: {config.GPR_FILE.name} not found - GPR columns will be absent")
        return pd.DataFrame(columns=["period_end", "series", "value", "published_on"])
    d = pd.read_excel(config.GPR_FILE)
    d["month"] = pd.to_datetime(d["month"])
    rows = []
    for sid, name in GPR_SERIES.items():
        if sid not in d.columns:
            print(f"  WARNING: GPR series {sid} absent - skipped")
            continue
        s = d[["month", sid]].dropna()
        # Stamp the period end as the last day of the month it describes.
        period_end = s["month"] + pd.offsets.MonthEnd(0)
        rows.append(pd.DataFrame({
            "period_end": period_end,
            "series": name,
            "value": s[sid].to_numpy(),
            "published_on": period_end + pd.Timedelta(days=config.PUBLICATION_LAGS["gpr"]),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["period_end", "series", "value", "published_on"])


def load_series() -> pd.DataFrame:
    """
    Returns a long frame: period_end, series, value, published_on.

    Series IDs that are absent from a table are reported and skipped rather than silently
    dropped - RBA renames them from time to time and a quiet disappearance is exactly the
    failure this pipeline should not have.
    """
    rows = []
    for fname, mapping in SERIES.items():
        try:
            tbl = _read_rba_csv(fname)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: {fname} unreadable ({e}) - skipped")
            continue
        missing = [sid for sid in mapping if sid not in tbl.columns]
        if missing:
            print(f"  WARNING: {fname} missing series {missing}")
        for sid, (name, lag_key) in mapping.items():
            if sid not in tbl.columns:
                continue
            s = tbl[sid].dropna()
            if s.empty:
                print(f"  WARNING: {name} ({sid}) is entirely empty - skipped")
                continue
            rows.append(pd.DataFrame({
                "period_end": s.index,
                "series": name,
                "value": s.to_numpy(),
                "published_on": s.index + pd.Timedelta(days=config.PUBLICATION_LAGS[lag_key]),
            }))
    if not rows:
        raise RuntimeError("no macro series loaded")
    gpr = load_gpr()
    if len(gpr):
        rows.append(gpr)
    long = pd.concat(rows, ignore_index=True)

    # REQUIRED-INPUT CONTRACT. The warnings above are diagnostics; this is the gate.
    # Every declared series is vendored and required - a missing one means a corrupt or
    # renamed input, and a panel quietly built without it would flow into every
    # downstream stage looking complete.
    expected = ({name for m in SERIES.values() for name, _ in m.values()}
                | set(GPR_SERIES.values()))
    problems = sorted(expected - set(long["series"]))
    vals = long["value"].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        bad = long.loc[~np.isfinite(long["value"].astype(float)), "series"].unique()
        problems.append(f"non-finite values in {sorted(bad)[:4]}")
    thin = long.groupby("series").size()
    thin = thin[thin < 20]
    if len(thin):
        problems.append(f"series with under 20 observations: {sorted(thin.index)[:4]}")
    # CHRONOLOGICAL COVERAGE per series, not just volume: twenty observations dated
    # 1900-1904 once passed. Quarterly GDP with its 65-day lag can trail the last
    # meeting by ~5 months, so the end tolerance is generous but bounded. Two vendored
    # series have KNOWN, documented gaps the as-of join already handles - they are
    # exempted BY NAME, and the exemption is BOUNDED at the documented endpoint (with a
    # quarter's tolerance), so a differently-truncated fake in either series still
    # fails, as does a new gap in any other series:
    #   consumer_sentiment   the H3 series begins 2010-01; must start by ~2010-03
    #   infexp_union_1y      discontinued after 2023-09; must still reach ~2023-06
    KNOWN_LIMITED = {"consumer_sentiment": ("starts_by", pd.Timestamp("2010-03-31")),
                     "infexp_union_1y": ("ends_from", pd.Timestamp("2023-06-30"))}
    span = long.groupby("series")["period_end"].agg(["min", "max"])
    for name, row in span.iterrows():
        kind, bound = KNOWN_LIMITED.get(name, (None, None))
        if kind == "starts_by":
            if row["min"] > bound:
                problems.append(f"{name} starts {row['min'].date()}, past its "
                                f"documented {bound.date()} onset")
        elif row["min"] > pd.Timestamp(config.CORPUS_START):
            problems.append(f"{name} starts {row['min'].date()}, after the corpus "
                            f"start")
        if kind == "ends_from":
            if row["max"] < bound:
                problems.append(f"{name} ends {row['max'].date()}, before its "
                                f"documented {bound.date()} discontinuation window")
        elif row["max"] < (pd.Timestamp(config.CORPUS_LAST_MEETING)
                           - pd.Timedelta(days=200)):
            problems.append(f"{name} ends {row['max'].date()}, long before the last "
                            f"meeting")
    if (pd.to_datetime(long["published_on"])
            < pd.to_datetime(long["period_end"])).any():
        problems.append("a series is stamped as published BEFORE the period it "
                        "describes ended - the publication-lag stamping is broken")
    if problems:
        raise RuntimeError(
            f"macro inputs failed their contract: {problems}. The raw files are "
            f"vendored, so this is a corrupt or renamed input, not a normal condition "
            f"- restore data/raw and re-run.")

    # Convert level series to year-on-year growth before the as-of join, so the growth rate
    # inherits the correct publication date.
    for name in AS_GROWTH:
        m = long["series"] == name
        if not m.any():
            continue
        sub = long[m].sort_values("period_end").copy()
        freq_months = sub["period_end"].diff().dt.days.median() / 30.44
        periods = max(1, round(12 / freq_months))
        sub["value"] = sub["value"].pct_change(periods) * 100
        sub["series"] = f"{name}_yoy"
        long = pd.concat([long[~m], sub.dropna(subset=["value"])], ignore_index=True)

    return long.sort_values(["series", "published_on"]).reset_index(drop=True)


def as_of(long: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    For each date, the latest value of each series that had ALREADY BEEN PUBLISHED.

    merge_asof with direction='backward' on published_on is the whole leakage control.
    A forward or nearest join here would silently hand the model the future, and the
    resulting accuracy would look excellent.
    """
    target = pd.DataFrame({"date": pd.DatetimeIndex(dates).sort_values()})
    out = target.copy()
    for name, grp in long.groupby("series"):
        g = (grp[["published_on", "value", "period_end"]]
             .dropna(subset=["value"])
             .sort_values("published_on")
             .rename(columns={"value": name}))
        merged = pd.merge_asof(target, g, left_on="date", right_on="published_on",
                               direction="backward")
        out[name] = merged[name].to_numpy()
        # Staleness: how old the newest published figure was at the meeting. A quarterly
        # series is routinely 2-4 months stale at any given meeting, and that is a real
        # constraint on the model rather than a data-quality defect.
        out[f"{name}__age_days"] = (merged["date"] - merged["period_end"]).dt.days.to_numpy()
    return out.set_index("date")


def run(meeting_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    long = load_series()
    print(f"  loaded {long['series'].nunique()} macro series, "
          f"{long['period_end'].min().date()} -> {long['period_end'].max().date()}")

    if meeting_dates is None:
        from data_panel import meeting_calendar
        meeting_dates = meeting_calendar().index

    panel = as_of(long, meeting_dates)
    val_cols = [c for c in panel.columns if not c.endswith("__age_days")]
    print(f"  macro_asof: {len(panel)} meetings x {len(val_cols)} series")
    miss = panel[val_cols].isna().mean().mul(100).round(1)
    print(f"  missing % at meeting dates:\n{miss.to_string()}")
    age = panel[[c for c in panel.columns if c.endswith('__age_days')]].median().round(0)
    print(f"  median staleness (days) of the newest published figure:\n{age.to_string()}")

    panel.reset_index().to_parquet(config.DATA_PROCESSED / "macro_asof.parquet", index=False)
    return panel


if __name__ == "__main__":
    run()

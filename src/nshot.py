"""
N-shot machinery for REPLAY: choosing which past meetings to show the model, and rendering
them into a prompt block.

SUPPLIED IN FULL. You choose k and the strategy, and you write the prompts in
`decision_replay.py`. What you must not do is choose the strategy by trying all four on your
shortlisted meeting and keeping the winner - see `dev_sample()` / `holdout_sample()`.

-------------------------------------------------------------------------------------------
LEAKAGE, IN THREE PLACES
-------------------------------------------------------------------------------------------
1. SHOT DATES. Every example must predate the target meeting. `every_shot_predates()`
   raises; it is not a warning.

2. SHOT LABELS. The `regimes` strategy selects on `y_cycle`, which is a FORWARD-looking
   label: the cycle state at meeting M is defined by what the cash rate does over the
   following CYCLE_WINDOW_DAYS. A meeting two months before the target has a `y_cycle` that
   was not yet observable at the target. `_label_known_by()` drops those. Without it the
   strategy quietly selects on the future and looks unbeatable.

3. TARGET CONDITIONS. The question block must be built from the panel as it stood before the
   meeting, which is what `decision_replay.load_as_at()` returns. Pass that frame in as
   `as_at`; if you pass the raw panel you will hand the model the meeting's own lagged text
   features and its own decision.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402

# ###########################################################################################
# SUPPLIED IN FULL - DO NOT MODIFY
# ###########################################################################################

SHOT_FIELDS = [
    ("cash_rate", "cash rate", "%.2f%%"),
    ("trimmed_mean_yoy", "underlying inflation (trimmed mean, y/y)", "%.1f%%"),
    ("cpi_yoy", "headline CPI (y/y)", "%.1f%%"),
    ("unemployment_rate", "unemployment", "%.1f%%"),
    ("gdp_growth_yoy", "GDP growth (y/y)", "%.1f%%"),
    ("business_conditions", "business conditions index", "%.1f"),
    ("consumer_sentiment", "consumer sentiment", "%.1f"),
    ("slope_cash3y", "3y yield less cash rate", "%+.2f"),
    ("bab_spread", "3m bank bill spread over cash", "%+.2f"),
    ("terms_of_trade", "terms of trade index", "%.1f"),
    ("trailing_change_182d", "cash rate change over prior 6 months", "%+.2f pp"),
]

# The WORDS constructs, lagged one meeting by `data_panel`, so the value shown at meeting M
# comes from the minutes of M-1 and was public before M. This is how Words reaches Replay:
# the model sees not only what the economy was doing but how the Board had been TALKING
# about it. Drop these and the two stages are unconnected.
CONSTRUCT_SHOT_FIELDS = [
    ("policy_stance", "Board's stance at the previous meeting (0 dovish, 1 hawkish)", "%.2f"),
    ("inflation_concern", "inflation concern in the previous minutes", "%.2f"),
    ("downside_risk_emphasis", "downside-risk emphasis in the previous minutes", "%.2f"),
    ("financial_conditions_concern", "financial-conditions concern, previous minutes", "%.2f"),
    ("uncertainty_language", "hedging in the previous minutes", "%.2f"),
    ("vigilance", "stated readiness to act, previous minutes", "%.2f"),
    ("global_risk_salience", "offshore share of the previous risk discussion", "%.2f"),
]

SIMILARITY_FIELDS = ["trimmed_mean_yoy", "unemployment_rate", "gdp_growth_yoy",
                     "slope_cash3y", "business_conditions", "cash_rate"]

DECISION_WORD = {-1: "CUT", 0: "HOLD", 1: "HIKE"}

STRATEGIES = ("recent", "similar", "stratified", "regimes")


def decision_label(decision: int, change_pct: float | None) -> str:
    """
    'CUT 50bp', 'HOLD', 'HIKE 25bp'.

    The size is shown because it is a real part of the decision and the model is being asked
    to reproduce policy reasoning, not to fill in a three-way blank. About four in five
    moves in this sample are 25bp; the rest range from 15bp to 100bp, so a model that has
    only ever seen 'CUT 25bp or more' has been told the sizes do not vary when they do.
    """
    if decision == 0:
        return "HOLD"
    if change_pct is None or not np.isfinite(change_pct):
        return DECISION_WORD[decision]
    return f"{DECISION_WORD[decision]} {abs(round(change_pct * 100)):.0f}bp"


def describe_conditions(row: pd.Series, include_constructs: bool = True) -> str:
    out = []
    for col, label, fmt in SHOT_FIELDS:
        if col in row.index and pd.notna(row[col]):
            out.append(f"  {label}: {fmt % row[col]}")
    if include_constructs:
        block = [f"  {label}: {fmt % row[col]}"
                 for col, label, fmt in CONSTRUCT_SHOT_FIELDS
                 if col in row.index and pd.notna(row[col])]
        if block:
            out.append("  -- how the Board had been writing (previous meeting's minutes) --")
            out.extend(block)
    return "\n".join(out)


def format_shot(date: pd.Timestamp, row: pd.Series, include_outcome: bool = True,
                include_constructs: bool = True, age_days: int | None = None) -> str:
    head = f"MEETING: {date.date()}"
    if age_days is not None:
        head += f"   ({age_days} days before the meeting being assessed)"
    s = f"{head}\n{describe_conditions(row, include_constructs)}"
    if include_outcome:
        s += ("\n  DECISION TAKEN: "
              + decision_label(int(row["decision"]), row.get("change_pct")))
    return s


# -------------------------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------------------------

def _eligible(panel: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """Every meeting strictly before the target, with a known decision."""
    return panel.loc[:target].iloc[:-1].dropna(subset=["decision"])


def _label_known_by(pool: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """
    Restrict to meetings whose FORWARD-looking label had already resolved by the target.

    `y_cycle` at meeting M summarises the CYCLE_WINDOW_DAYS after M, so it is observable
    only from M + CYCLE_WINDOW_DAYS onwards. Selecting shots on an unresolved label is
    selecting on the future.
    """
    cutoff = target - pd.Timedelta(days=config.CYCLE_WINDOW_DAYS)
    return pool.loc[:cutoff]


def _validate_k(k: int, pool_size: int, strategy: str) -> None:
    if not isinstance(k, (int, np.integer)) or k < 3:
        raise ValueError(f"k must be an integer >= 3, got {k!r}")
    if k > pool_size:
        raise ValueError(f"k={k} but only {pool_size} eligible meetings precede the target")
    if strategy in ("stratified", "regimes") and k % 3:
        raise ValueError(
            f"strategy {strategy!r} fills three groups, so k must be a multiple of 3. "
            f"You asked for {k}; use {3 * (k // 3)} or {3 * (k // 3 + 1)}.")


def select_shots(panel: pd.DataFrame, target: pd.Timestamp, k: int = 9,
                 strategy: str = "stratified") -> pd.DataFrame:
    """
    Choose k historical meetings to show the model. Returned in chronological order, with
    an `__age_days` column so the prompt can state how far back each example sits.

    Chronological order matters: a model reading examples in time order can see the arc of a
    cycle, which is part of what you want it to learn.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    pool = _eligible(panel, target)
    _validate_k(k, len(pool), strategy)

    if strategy == "recent":
        chosen = pool.tail(k)

    elif strategy == "similar":
        cols = [c for c in SIMILARITY_FIELDS if c in pool.columns]
        # COMPLETE ROWS ONLY. Summing squared differences over whatever happens to be
        # present makes a row with three missing fields look closer than a complete row that
        # genuinely matches, so the early-sample meetings win on missingness alone.
        num = pool[cols].apply(pd.to_numeric, errors="coerce")
        complete = pool.loc[num.dropna().index]
        num = num.loc[complete.index]
        if len(complete) < k:
            raise ValueError(
                f"only {len(complete)} meetings before {target.date()} have all of {cols}; "
                f"k={k} is too large for the 'similar' strategy this early in the sample")
        mu, sd = num.mean(), num.std().replace(0, 1)
        z = (num - mu) / sd
        t = (pd.to_numeric(panel.loc[target, cols], errors="coerce") - mu) / sd
        if t.isna().any():
            raise ValueError(
                f"target {target.date()} is missing {list(t.index[t.isna()])}; "
                f"'similar' cannot be computed against a partial target")
        chosen = complete.loc[((z - t) ** 2).sum(axis=1).pow(0.5).nsmallest(k).index]

    elif strategy == "stratified":
        parts = [pool[pool["decision"] == d].tail(k // 3) for d in (-1, 0, 1)]
        chosen = pd.concat(parts)
        if len(chosen) < k:
            raise ValueError(
                f"stratified needs {k // 3} of each of cut/hold/hike before "
                f"{target.date()}; only {len(chosen)} available. Reduce k or pick a later "
                f"meeting.")

    else:  # regimes
        if "y_cycle" not in pool.columns:
            raise ValueError("regimes strategy needs y_cycle in the panel")
        resolved = _label_known_by(pool, target).dropna(subset=["y_cycle"])
        parts = [resolved[resolved["y_cycle"] == s].tail(k // 3) for s in (0, 1, 2)]
        chosen = pd.concat(parts)
        if len(chosen) < k:
            raise ValueError(
                f"regimes needs {k // 3} meetings of each cycle state whose label had "
                f"resolved by {target.date()}; only {len(chosen)} available. The horizon is "
                f"{config.CYCLE_WINDOW_DAYS} days, so the most recent "
                f"{config.CYCLE_WINDOW_DAYS // 30} months are unusable here.")

    chosen = chosen.sort_index().copy()
    chosen["__age_days"] = [(target - d).days for d in chosen.index]
    every_shot_predates(chosen, target)
    return chosen


def every_shot_predates(shots: pd.DataFrame, target: pd.Timestamp) -> None:
    """Enforced, not advisory. Raises if any shot is dated on or after the target."""
    bad = [d for d in shots.index if d >= target]
    if bad:
        raise ValueError(
            f"LEAKAGE: {len(bad)} shot(s) dated on or after the target meeting "
            f"{target.date()}: {[str(d.date()) for d in bad]}")


def build_prompt_block(panel: pd.DataFrame, target: pd.Timestamp, k: int = 9,
                       strategy: str = "stratified",
                       as_at: pd.DataFrame | None = None,
                       include_constructs: bool = True) -> tuple[str, pd.DataFrame]:
    """
    The examples block and the target block, ready to drop into your prompt.

    `as_at` is the frame the TARGET row is read from and should be
    `decision_replay.load_as_at(target)`. It defaults to `panel` only so the function can be
    unit-tested; in the Replay run it is always passed, because the raw panel row carries
    the meeting's own decision and its own lagged text features.

    Print the text once and read it before you spend credit on it - the commonest Replay
    error is a prompt whose examples are 90% holds.
    """
    shots = select_shots(panel, target, k=k, strategy=strategy)
    src = panel if as_at is None else as_at
    if target not in src.index:
        raise ValueError(f"{target.date()} is not in the as-at frame")
    target_row = src.loc[target]
    for c in ("decision", "change_pct", "y_decision", "y_cycle"):
        if c in target_row.index and pd.notna(target_row[c]):
            raise ValueError(
                f"LEAKAGE: the target row still carries {c!r}. Build the question block "
                f"from decision_replay.load_as_at(), not from the raw panel.")

    examples = "\n\n".join(
        format_shot(d, r, include_constructs=include_constructs,
                    age_days=int(r["__age_days"]))
        for d, r in shots.iterrows())
    question = format_shot(target, target_row, include_outcome=False,
                           include_constructs=include_constructs)
    text = (f"HISTORICAL EXAMPLES ({len(shots)} past meetings, "
            f"{shots.index.min().date()} to {shots.index.max().date()}, selected by "
            f"'{strategy}'):\n\n"
            f"{examples}\n\n"
            f"{'=' * 60}\n"
            f"NOW ASSESS THIS MEETING:\n\n{question}")
    return text, shots


def shot_mix(shots: pd.DataFrame) -> dict:
    """What the model is actually being shown. Check this before running."""
    c = shots["decision"].value_counts()
    return {"n": int(len(shots)),
            "cut": int(c.get(-1, 0)), "hold": int(c.get(0, 0)), "hike": int(c.get(1, 0)),
            "hold_share": round(float(c.get(0, 0) / len(shots)), 3),
            "from": str(shots.index.min().date()), "to": str(shots.index.max().date()),
            "median_age_days": int(shots["__age_days"].median()),
            "oldest_age_days": int(shots["__age_days"].max())}

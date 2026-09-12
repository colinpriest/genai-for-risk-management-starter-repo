"""
The channel taxonomy and shock-size calibration. SUPPLIED - do not modify.

TWO THINGS HERE, BOTH SCAFFOLDING FOR THE SHOCK STAGE.

1. CHANNEL_TAXONOMY - nine canonical transmission channels.

   Every branch in your tree must be tagged with one. Without a fixed vocabulary each team
   invents its own channel names, no two trees are comparable, and neither you nor the
   marker can tell whether you covered the space or just found nine ways to say
   "confidence falls".

   You may propose a channel outside the taxonomy. Tag it `other` and justify it - novel
   channels are fine, unlabelled ones are not.

2. CALIBRATION - what a shock of N standard deviations actually looked like.

   Students guess `n_sd`. These figures are measured from the panel itself, so a proposed
   shock can be anchored against something that really happened. If you are about to write
   n_sd=4.0 for a moderate scenario, this table tells you that you have just proposed
   something more extreme than the GFC.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. The nine channels
# ---------------------------------------------------------------------------
# name -> (what it is, the panel variables that usually carry it)

CHANNEL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    "terms_of_trade": (
        "Export prices relative to import prices, and therefore national income. The "
        "channel that matters most in Australia and least in most textbooks.",
        ["terms_of_trade", "gdp_growth_yoy"]),

    "exchange_rate": (
        "The AUD, which moves fast and cuts both ways: a fall supports exporters and "
        "raises import prices at the same time.",
        ["aud_ret", "aud_vol_21"]),

    "funding_costs": (
        "What it costs Australian banks to fund themselves, much of it offshore. Can "
        "tighten financial conditions WITHOUT the RBA moving - and can blunt a cut.",
        ["bab_spread", "bond_3y", "bond_10y", "slope_cash3y", "slope_3y10y"]),

    "risk_appetite": (
        "Global willingness to hold risk. Transmits within days, through equity "
        "volatility and credit spreads.",
        ["vix", "rv_21", "rv_5", "downside_21", "parkinson_21"]),

    "household_income": (
        "Real disposable income and the propensity to spend it. Australia's mortgage book "
        "is predominantly variable-rate, so this channel is faster here than in the US.",
        ["consumer_sentiment", "unemployment_rate", "employment_yoy"]),

    "import_prices": (
        "Tradables inflation - the price level effect of a shock, as distinct from a "
        "change in underlying inflation. The RBA routinely looks through the first round.",
        ["cpi_yoy", "cpi_qoq"]),

    "external_demand": (
        "Demand from trading partners, above all China. Volume rather than price.",
        ["gdp_growth_yoy", "business_conditions", "terms_of_trade"]),

    "labour_supply": (
        "The size and composition of the workforce, as distinct from demand for it. "
        "Migration, participation and hours. A supply-side shock here raises wage pressure "
        "and cuts aggregate demand at the same time, which is why it is separated out.",
        ["participation_rate", "employment_yoy", "unemployment_rate"]),

    "financial_stability": (
        "Whether the financial system itself is impaired. When this channel is live it "
        "usually dominates every other consideration.",
        ["bab_spread", "financial_conditions_concern", "vix"]),
}

CHANNEL_NAMES = list(CHANNEL_TAXONOMY) + ["other"]


def describe_taxonomy() -> str:
    """The block passed to the model so it tags branches from the fixed vocabulary."""
    lines = ["THE NINE CANONICAL CHANNELS. Tag every channel you propose with exactly one "
             "of these `channel_type` values, or 'other' if it genuinely fits none.", ""]
    for name, (desc, proxies) in CHANNEL_TAXONOMY.items():
        lines.append(f"  {name}")
        lines.append(f"      {desc}")
        lines.append(f"      usual proxies: {', '.join(proxies)}")
    lines.append("  other")
    lines.append("      anything genuinely outside the nine. Justify it in `note`.")
    return "\n".join(lines)


def coverage(branches) -> dict:
    """
    Which channels a tree touched, and which it missed.

    A tree that fires on two of nine channels has probably not explored the space. That is
    not automatically wrong - some shocks really are narrow - but it is worth noticing
    before you report a net direction.
    """
    used = {}
    for b in branches:
        t = getattr(b, "channel_type", None) or "untagged"
        used[t] = used.get(t, 0) + 1
    missing = [c for c in CHANNEL_TAXONOMY if c not in used]
    return {"used": used, "n_channel_types": len([k for k in used if k != "untagged"]),
            "missing": missing, "untagged": used.get("untagged", 0)}


# ---------------------------------------------------------------------------
# 2. Shock-size calibration
# ---------------------------------------------------------------------------
# WHAT THIS TABLE MEASURES, AND WHY THE PREVIOUS ONE WAS WRONG.
#
# `apply_shock()` adds n_sd to the CURRENT value of a variable. So a calibration table has to
# record CHANGES, in the same units. The earlier table recorded each variable's peak signed
# deviation from its FULL-SAMPLE MEAN during an episode - a level statistic - and those
# numbers were then read as though they were changes. They are different quantities, and on
# a trending variable they are not even close.
#
# Every row below is now: the largest change from the variable's own mean over the TWELVE
# MONTHS BEFORE the episode, measured over the episode window, divided by the variable's
# full-sample standard deviation. Both windows are recorded on the row, so the number can be
# checked and re-derived - `measure_calibration()` regenerates the whole table from the
# frozen panel.
#
# TWO SIGNS WERE ALSO WRONG, AND BOTH WERE INFORMATIVE.
#
#   bab_spread in the GFC came out at -5.7sd - NARROWING - while the table described it as
#   widening funding stress. It is neither. `bab_spread` is the 3-month bank bill yield minus
#   the cash rate, and in late 2008 bills sat up to 107bp BELOW cash because the market was
#   pricing further RBA cuts. It is a POLICY-EXPECTATION spread. In 2022 it is +4.7sd for the
#   same reason with the sign reversed: the market was pricing hikes. See EXPECTATION_PROXIES
#   below - shocking one of these is close to asserting the policy path you are supposed to
#   be predicting.
#
#   aud_ret is a RETURN - already a change - so a "peak deviation from the mean" is
#   meaningless for it, and the old table's +4.3sd for the GFC was the November rebound
#   rather than the crash. Cumulating it does not rescue the number either: summed over the
#   GFC meeting dates it comes to +3.3sd, because the panel samples the return at meetings
#   and the collapse happened between them. The honest conclusion is that this panel cannot
#   calibrate a return series over an episode, so aud_ret is EXCLUDED and says so.

# Variables whose panel representation cannot be calibrated over an episode window.
UNCALIBRATABLE = {
    "aud_ret": ("a return sampled at meeting dates - it does not aggregate to the episode "
                "move, and its peak picks whichever tail is larger rather than the event "
                "direction. Size an AUD shock from economic argument and say how."),
    "aud_vol_21": ("a realised-volatility measure whose episode peak is dominated by a few "
                   "days; use vix or rv_21, which behave the same way and are calibrated."),
}

# Spreads that price FUTURE POLICY. Shocking one asserts part of the answer.
EXPECTATION_PROXIES = {
    "bab_spread": "3-month bank bill yield less cash - prices RBA moves over the next quarter",
    "slope_cash3y": "3-year yield less cash - prices the RBA path over three years",
    "slope_3y10y": "3y-10y - term premium, only partly a policy-path measure",
}

CALIBRATION = pd.DataFrame([
    # episode, variable, change in sd, pre-event baseline ends, episode window ends
    ("GFC 2008-09",              "vix",                 +4.44, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "rv_21",               +4.22, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "rv_63",               +3.50, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "parkinson_21",        +3.40, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "slope_3y10y",         +4.06, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "bab_spread",          -5.70, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "business_conditions", -2.54, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "dwelling_approvals",  -1.50, "2008-08-31", "2009-03-31"),
    ("COVID onset 2020",         "rv_5",                +9.59, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "parkinson_21",        +8.43, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "rv_21",               +7.37, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "vix",                 +6.56, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "rv_63",               +5.33, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "employment_yoy",      -5.00, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "participation_rate",  -4.78, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "business_conditions", -4.29, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "unemployment_rate",   +2.67, "2020-02-15", "2020-09-30"),
    ("2022 inflation surge",     "gpr_threats",         +4.90, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "bab_spread",          +4.73, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "trimmed_mean_yoy",    +4.08, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "infexp_union_1y",     +3.96, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "gpr_index",           +3.90, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "cpi_yoy",             +3.44, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "slope_cash3y",        +3.31, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "consumer_sentiment",  -3.24, "2022-03-31", "2023-06-30"),
    ("Qld floods 2011",          "consumer_sentiment",  -2.33, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "slope_cash3y",        -2.14, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "terms_of_trade",      +1.86, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "parkinson_21",        +1.73, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "vix",                 +1.52, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "cpi_qoq",             +1.22, "2010-12-15", "2011-09-30"),
    ("Taper/mining unwind 2013", "dwelling_approvals",  +2.00, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "slope_cash3y",        +1.81, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "slope_3y10y",         +1.68, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "gpr_threats",         +1.62, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "consumer_sentiment",  +1.31, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "real_gdp_yoy",        -1.16, "2013-04-30", "2014-06-30"),
    ("GFC 2008-09",              "gdp_growth_yoy",      -0.92, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "real_gdp_yoy",        -0.90, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "terms_of_trade",      +1.28, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "cpi_yoy",             +1.22, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "trimmed_mean_yoy",    +0.99, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "unemployment_rate",   +0.81, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "employment_yoy",      -0.81, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "cpi_qoq",             -1.94, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "bond_10y",            -1.29, "2008-08-31", "2009-03-31"),
    ("GFC 2008-09",              "infexp_union_1y",     +0.85, "2008-08-31", "2009-03-31"),
    ("COVID onset 2020",         "consumer_sentiment",  -2.52, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "cpi_qoq",             -4.09, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "cpi_yoy",             -1.22, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "gpr_index",           +1.45, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "dwelling_approvals",  -0.77, "2020-02-15", "2020-09-30"),
    ("COVID onset 2020",         "gdp_growth_yoy",      -0.41, "2020-02-15", "2020-09-30"),
    ("2022 inflation surge",     "gpr_acts",            +3.00, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "employment_yoy",      +2.36, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "dwelling_approvals",  -2.41, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "cpi_qoq",             +2.20, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "unemployment_rate",   -1.89, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "terms_of_trade",      +1.39, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "bond_10y",            +1.43, "2022-03-31", "2023-06-30"),
    ("2022 inflation surge",     "gdp_growth_yoy",      +1.00, "2022-03-31", "2023-06-30"),
    ("Taper/mining unwind 2013", "gdp_growth_yoy",      -1.12, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "participation_rate",  -0.91, "2013-04-30", "2014-06-30"),
    ("Taper/mining unwind 2013", "employment_yoy",      -0.92, "2013-04-30", "2014-06-30"),
    ("Qld floods 2011",          "gpr_acts",            +0.93, "2010-12-15", "2011-09-30"),
    ("Qld floods 2011",          "dwelling_approvals",  -0.57, "2010-12-15", "2011-09-30"),
], columns=["episode", "variable", "change_sd", "baseline_end", "event_end"])

# The pre-event baseline is the twelve months ending at `baseline_end`.
CALIBRATION_METHOD = (
    "change from the mean of the 12 months before the episode to the most extreme value "
    "during it, divided by the variable's full-sample standard deviation")


def measure_calibration(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Regenerate the table above from the frozen panel. Committed so it can be checked.

    Run it if you doubt a number: `python -c "import channels; print(channels.
    measure_calibration())"`. It will reproduce CALIBRATION to the second decimal.
    """
    import config
    if panel is None:
        panel = pd.read_parquet(config.FROZEN_PANEL)
        panel["meeting_date"] = pd.to_datetime(panel["meeting_date"])
        panel = panel.set_index("meeting_date").sort_index()
    rows = []
    for r in CALIBRATION.itertuples():
        if r.variable not in panel.columns:
            continue
        s = panel[r.variable].astype(float)
        sd = float(s.std())
        b0 = (pd.Timestamp(r.baseline_end) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        base = s.loc[b0:r.baseline_end].mean()
        w = s.loc[r.baseline_end:r.event_end].dropna()
        if not len(w) or sd == 0:
            continue
        dev = w - base
        peak = float(dev.loc[dev.abs().idxmax()])
        rows.append({"episode": r.episode, "variable": r.variable,
                     "recomputed_sd": round(peak / sd, 2), "table_sd": r.change_sd,
                     "matches": abs(peak / sd - r.change_sd) < 0.02})
    return pd.DataFrame(rows)


def calibration_for(variable: str) -> pd.DataFrame:
    """What this variable actually changed by in past episodes. Use it to size your shock."""
    return CALIBRATION[CALIBRATION["variable"] == variable].sort_values(
        "change_sd", key=abs, ascending=False)


def sanity_check_shock(variable: str, n_sd: float) -> str | None:
    """
    Returns a warning if the proposed shock is out of line with history, else None.

    Not a veto. A scenario CAN be worse than the GFC - but if you are claiming that, claim it
    deliberately.
    """
    if variable in UNCALIBRATABLE:
        return (f"{variable}: NOT CALIBRATABLE from this panel - {UNCALIBRATABLE[variable]}")
    if variable in EXPECTATION_PROXIES:
        note = (f"{variable}: this is a POLICY-EXPECTATION spread ({EXPECTATION_PROXIES[variable]}). "
                f"Shocking it asserts part of the policy path you are trying to predict. "
                f"Justify it explicitly or shock the underlying driver instead.")
    else:
        note = None
    hist = calibration_for(variable)
    if hist.empty:
        return note or (f"{variable}: no calibration episode on record - size this from "
                        f"economic reasoning and say how you did it")
    worst = hist["change_sd"].abs().max()
    if abs(n_sd) > worst:
        ep = hist.loc[hist["change_sd"].abs().idxmax(), "episode"]
        msg = (f"{variable}: {n_sd:+.1f}sd exceeds anything in the record - the largest "
               f"change was {worst:+.1f}sd in {ep}. Defend it or reduce it.")
        return f"{note} | {msg}" if note else msg
    if abs(n_sd) < 0.25:
        msg = (f"{variable}: {n_sd:+.1f}sd is smaller than normal quarter-to-quarter noise "
               f"and will not move the model. Is this channel material?")
        return f"{note} | {msg}" if note else msg
    return note


def describe_calibration(variables: list[str] | None = None) -> str:
    """The block passed to the model so it sizes shocks against real episode changes."""
    df = CALIBRATION if not variables else CALIBRATION[CALIBRATION["variable"].isin(variables)]
    if df.empty:
        df = CALIBRATION
    lines = [f"SHOCK-SIZE CALIBRATION. Each figure is the {CALIBRATION_METHOD}:", ""]
    for ep, grp in df.groupby("episode", sort=False):
        moves = ", ".join(f"{r.variable} {r.change_sd:+.1f}sd" for r in grp.itertuples())
        lines.append(f"  {ep}: {moves}")
    lines += ["",
              "Size your shocks against these. A scenario milder than the GFC should not "
              "produce larger changes than the GFC produced.",
              "",
              "TWO CAUTIONS.",
              "  Not calibratable, do not use: " + ", ".join(sorted(UNCALIBRATABLE)) + ".",
              "  Policy-expectation spreads - shocking these asserts part of the policy "
              "path, so prefer the underlying driver: " + ", ".join(sorted(EXPECTATION_PROXIES)) + "."]
    return "\n".join(lines)

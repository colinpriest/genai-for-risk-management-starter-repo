"""
Historical calibration — section 8.4 Rule 0

PRE-BUILT: the event list, the volatility and regime calculations, the RBA language comparison,
           and the quote finder.
YOURS:     interpreting it, and deciding what it means for your scenario impacts.

Run this BEFORE generating any scenarios. It turns your scenario impacts from guesses into
estimates scaled from observed behaviour.

WRITES data/processed/historical_calibration.json
"""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
import config

# The date each shock became widely recognised, not the date it began.
# Add your own if you find others in the window.
EVENTS = {
    "GFC — Lehman collapse":          "2008-09-15",
    "European sovereign debt crisis": "2011-08-05",
    "China growth scare":             "2015-08-24",
    "COVID-19 market crash":          "2020-03-09",
    "Russian invasion of Ukraine":    "2022-02-24",
}

# Search terms for finding what the RBA said about each event.
EVENT_TERMS = {
    "GFC — Lehman collapse":          r"Lehman|global financial crisis|market turmoil",
    "European sovereign debt crisis": r"(European|euro area) (sovereign )?debt|Greece|Greek",
    "China growth scare":             r"Chinese (equity|share) market|China.{0,40}(slowdown|concerns)",
    "COVID-19 market crash":          r"COVID|coronavirus|pandemic",
    "Russian invasion of Ukraine":    r"Ukraine|Russia",
}

WINDOW = 63          # trading days either side (about one quarter)
CONSTRUCTS = ["financial_conditions_concern", "downside_risk_emphasis",
              "global_risk_salience", "vigilance", "uncertainty_language"]


def find_rba_quote(docs: pd.DataFrame, event: str, on_or_after: pd.Timestamp,
                   chars: int = 320) -> dict | None:
    """The first thing the RBA said about this event after it happened."""
    pat = EVENT_TERMS.get(event)
    if not pat:
        return None
    after = docs[docs.meeting_date >= on_or_after]
    for _, r in after.iterrows():
        m = re.search(pat, r.text_full, re.I)
        if m:
            lo = max(0, m.start() - chars // 2)
            return {"meeting_date": str(r.meeting_date.date()),
                    "quote": "..." + r.text_full[lo:m.start() + chars // 2] + "..."}
    return None


def run() -> dict:
    reg = pd.read_parquet(config.DATA_PROCESSED / "regimes.parquet").set_index("date")
    scores = pd.read_parquet(config.DATA_PROCESSED / "riskvoice_scores.parquet")
    docs = pd.read_parquet(config.DATA_PROCESSED / "documents.parquet")

    crisis_label = config.N_REGIMES - 1          # highest regime
    base_crisis = float((reg["regime"] == crisis_label).mean())
    out = {"unconditional_crisis_share": base_crisis, "events": {}}

    for name, d in EVENTS.items():
        d = pd.Timestamp(d)
        # STRICTLY before and strictly after. reg.loc[:d] and reg.loc[d:] both INCLUDE d
        # when it is a trading day, so the event day itself - usually the most violent one -
        # was counted in both windows and inflated the "before" volatility.
        pre = reg.loc[reg.index < d].tail(WINDOW)
        post = reg.loc[reg.index > d].head(WINDOW)
        if len(post) < 20:
            print(f"  {name}: too close to the end of the data, skipped")
            continue

        before = scores[scores.meeting_date <= d].tail(1)
        after = scores[scores.meeting_date > d].head(2)
        lang = {}
        for c in CONSTRUCTS:
            if len(before) and len(after) and c in scores.columns:
                lang[c] = {"before": float(before[c].iloc[0]),
                           "after": float(after[c].mean()),
                           "change": float(after[c].mean() - before[c].iloc[0])}

        out["events"][name] = {
            "date": str(d.date()),
            "vol_before": float(pre["ret"].std() * np.sqrt(252)),
            "vol_after": float(post["ret"].std() * np.sqrt(252)),
            "vol_multiple": float(post["ret"].std() / max(pre["ret"].std(), 1e-9)),
            "crisis_share_before": float((pre["regime"] == crisis_label).mean()),
            "crisis_share_after": float((post["regime"] == crisis_label).mean()),
            "worst_day_after": float(post["ret"].min()),
            "rba_language_change": lang,
            "rba_said": find_rba_quote(docs, name, d),
        }

    json.dump(out, open(config.DATA_PROCESSED / "historical_calibration.json", "w"), indent=2)

    print(f"  unconditional crisis-regime share: {base_crisis:.1%}\n")
    for k, v in out["events"].items():
        fc = v["rba_language_change"].get("financial_conditions_concern", {})
        print(f"  {k}")
        print(f"    volatility        x{v['vol_multiple']:.2f}  "
              f"({v['vol_before']:.0%} -> {v['vol_after']:.0%})")
        print(f"    crisis regime     {v['crisis_share_before']:.0%} -> {v['crisis_share_after']:.0%}")
        print(f"    RBA fin-conditions {fc.get('change', float('nan')):+.2f}")
        if v["rba_said"]:
            print(f"    RBA said ({v['rba_said']['meeting_date']}): "
                  f"{v['rba_said']['quote'][:150]}...")
        print()

    print("  INTERPRET THIS. Two questions to answer in your report:")
    print("   1. Which event did NOT move the Australian market, and why not?")
    print("   2. Where your construct barely moved but the RBA quote is dramatic,")
    print("      is that the RBA being restrained, or your extraction failing?")
    return out


if __name__ == "__main__":
    run()

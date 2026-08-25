"""
WORDS - seven quantitative dimensions of the Board's reaction, extracted from the minutes.

YOU SUPPLY FOUR CONSTRUCT RUBRICS AND THE SYSTEM PROMPT. EVERYTHING ELSE IS BUILT.

Three of the seven rubrics are supplied as fixed exemplars, one per scale type. They are
hash-checked and the run ABORTS if they have been edited - see `_require_exemplars()`.

-------------------------------------------------------------------------------------------
WHAT THESE MEASURES ARE FOR
-------------------------------------------------------------------------------------------
They are the AXES along which the Board's reaction to economic circumstances is described.
They are not predictors, and the Cycle stage shows they add nothing to forecasting the cycle.
Their job is to give Shock a vocabulary: a stress scenario can then be positioned against
the reaction profiles of episodes that actually happened.

So what they must be is DISCRIMINATING, RELIABLE and CORRECTLY ORIENTED. The audit below
tests all three, and the third is the one that catches real failures - a dimension can pass
every spread and stability statistic while pointing the wrong way.

-------------------------------------------------------------------------------------------
THE DECISION IS LEFT IN THE TEXT, DELIBERATELY
-------------------------------------------------------------------------------------------
The minutes state the decision taken at that meeting. That is not leakage here: the panel
carries `decision` as a numeric column anyway, both targets are forward-looking, and for a
measure of the Board's reaction the decision is part of what is being measured.

Where it WOULD leak is using a meeting's own minutes at that meeting, and `data_panel`
prevents that by lagging the whole text tier one meeting.

RUN IT TWICE, OVER DISJOINT DOCUMENTS:
    python src/text_features.py              development - validation meetings withheld
    python src/text_features.py --validate   one shot, held-out meetings only

WRITES data/processed/construct_scores.parquet
       data/processed/llm_raw/<config-hash>/<date>.json  (reproducibility
           envelopes - the parsed result plus the request settings, usage and
           request id. NOT the complete API response object)
       outputs/words_audit.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402

load_dotenv()

# ###########################################################################################
# YOUR WORK STARTS HERE
# ###########################################################################################

SYSTEM_PROMPT = """
TODO: WRITE THE SYSTEM PROMPT.

Who is the model? What is it reading? What is it judging?

Things worth deciding explicitly, because the default behaviour is poor:
  - what the midpoint means, and whether it is a legitimate answer or a hedge
  - what it should do when it is unsure (hint: its uncertainty is measured by the spread
    ACROSS calls, so hedging inside a single call destroys that measurement)
  - whether you want it to quote its evidence

Two things NOT to do, both of which were tried and failed:

  Do not claim the model has read the whole corpus. It sees one document per request. A
  persona that says otherwise does not create calibration, it just licenses invention.
  If you want cross-corpus calibration, describe the range in the rubric itself.

  Do not tell it to avoid round numbers to make the scores look spread out. That defeats
  the concentration diagnostic without improving the measurement, and the audit will still
  report the binned concentration.
"""

# -------------------------------------------------------------------------------------------
# THREE OF SEVEN ARE WRITTEN FOR YOU. FOUR ARE YOURS.
# -------------------------------------------------------------------------------------------
# The exemplars demonstrate the three SCALE TYPES and three techniques:
#
#   policy_stance          SIGNED. How to defeat a midpoint pile-up, and how to keep an axis
#                          pointing at DIRECTION when the language is about severity.
#   uncertainty_language   INTENSITY. How to score linguistic FORM rather than content.
#   global_risk_salience   PROPORTIONAL. How to score a SHARE of the discussion.
#
# Yours: inflation_concern, downside_risk_emphasis, financial_conditions_concern, vigilance.
# `downside_risk_emphasis` is the hard one - it is signed, and "risks are broadly balanced"
# is boilerplate that will concentrate the scores if you let it.

CONSTRUCTS: dict[str, str] = {

    # ---- SUPPLIED EXEMPLAR 1 of 3 - SIGNED - DO NOT ALTER --------------------------------
    "policy_stance": (
        "How hawkish is the Board's stance? SIGNED, 50 = exactly neutral. "
        "THIS AXIS IS DIRECTION, NOT FORCE. A Board cutting aggressively in a crisis is at "
        "the DOVISH extreme however decisive, urgent or grave its language. Do not let the "
        "severity of the discussion pull the score upward - severity belongs to the other "
        "constructs. Ask only: which way is policy leaning? "
        "MOST MEETINGS ARE HOLDS, AND A HOLD IS ALMOST NEVER EXACTLY NEUTRAL. The "
        "concluding 'Considerations for Monetary Policy' section nearly always leans one "
        "way - through what the Board says it would need to see, which risk it names "
        "first, and whether it calls the current setting accommodative or restrictive. "
        "Find the lean and score it. Reserve 50 for the rare document with no detectable "
        "lean. "
        "0 = maximally dovish: a large cut delivered with more foreshadowed. "
        "15 = a cut delivered, or an explicit statement that further easing is likely. "
        "30 = no cut this meeting, but a clear easing bias: conditions named that would "
        "prompt one, or policy called too restrictive. "
        "45 = hold leaning slightly dovish: downside risks named before upside ones. "
        "50 = no detectable lean. Rare. "
        "55 = hold leaning slightly hawkish: inflation risk named before activity risk. "
        "70 = no hike this meeting, but a clear tightening bias: conditions named that "
        "would prompt one, or policy called still accommodative. "
        "85 = a hike delivered, or an explicit statement that further tightening is likely. "
        "100 = maximally hawkish: a large hike delivered with more foreshadowed and policy "
        "described as needing to become restrictive."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "inflation_concern": (
        "TODO: how concerned is the Board about inflation? An INTENSITY construct - study "
        "the uncertainty_language exemplar for the pattern. Note that being BELOW target is "
        "not the same as being comfortably within it, and your rubric should separate them."
    ),

    # ---- YOURS - the hard one -----------------------------------------------------------
    "downside_risk_emphasis": (
        "TODO: are the risks the Board discusses skewed to the downside? SIGNED, like "
        "policy_stance - study that exemplar and apply the same landmark technique. "
        "State DOWNSIDE TO WHAT. And note the trap: 'upside risk to INFLATION' is not "
        "upside in the sense of activity or employment - it is adverse, and it usually "
        "implies tighter policy. Decide how you treat it and say so in the rubric."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "financial_conditions_concern": (
        "TODO: credit availability, funding costs, bank lending, housing finance, market "
        "functioning. An INTENSITY construct."
    ),

    # ---- SUPPLIED EXEMPLAR 2 of 3 - INTENSITY - DO NOT ALTER ----------------------------
    "uncertainty_language": (
        "How heavily does the Board hedge? INTENSITY, 50 = moderate. "
        "Count and weigh hedging constructions: 'uncertain', 'difficult to predict', 'a "
        "range of outcomes', 'considerable uncertainty', 'depends on', conditional forecast "
        "language, explicit scenario branching. Judge the FORM of the language, not whether "
        "the Board is right to be uncertain. "
        "0 = confident, declarative, single-path prose with no hedging at all. "
        "15 = one or two routine caveats in an otherwise assured document. "
        "25 = routine caveats only. "
        "50 = uncertainty acknowledged in the usual places. "
        "75 = uncertainty is a recurring theme and shapes the discussion. "
        "100 = the Board repeatedly says it does not know: multiple scenarios, explicit "
        "refusal to forecast, uncertainty named as the reason for the decision."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "vigilance": (
        "TODO: how strongly does the Board commit to watching and responding? An INTENSITY "
        "construct. Judge the COMMITMENT LANGUAGE, not the sentiment - a worried Board that "
        "promises nothing scores lower than a calm Board that names what it will do. Almost "
        "every document contains some monitoring boilerplate, so decide what the floor is."
    ),

    # ---- SUPPLIED EXEMPLAR 3 of 3 - PROPORTIONAL - DO NOT ALTER -------------------------
    "global_risk_salience": (
        "How much of the Board's risk discussion is OFFSHORE rather than domestic? "
        "PROPORTIONAL, 50 = evenly split. "
        "AN EVEN SPLIT IS UNCOMMON. Every set of minutes opens with an international "
        "section, so the presence of offshore material tells you nothing. The question is "
        "how much of it reaches the RISK discussion and the policy conclusion. Most "
        "documents lean domestic; find the lean rather than settling on 50. "
        "0 = wholly domestic: the international section is perfunctory and no offshore "
        "risk enters the policy discussion. "
        "15 = offshore conditions summarised as benign background, then not referred to "
        "again. "
        "25 = mostly domestic; one offshore risk named but not developed. "
        "40 = domestic risks lead, offshore risks argued rather than merely listed. "
        "50 = offshore and domestic risks get comparable weight and comparable "
        "development. Rare. "
        "60 = offshore risks lead, but the policy conclusion turns on domestic conditions. "
        "75 = offshore risk is the dominant theme in the risk discussion. "
        "85 = an offshore development is named as a reason for the policy setting. "
        "100 = the decision is framed primarily around international developments - "
        "global financial stress, a major trading partner's downturn, or a global shock."
    ),
}

STUDENT_CONSTRUCTS = ["inflation_concern", "downside_risk_emphasis",
                      "financial_conditions_concern", "vigilance"]
SUPPLIED_CONSTRUCTS = ["policy_stance", "uncertainty_language", "global_risk_salience"]

# Whether the model must quote the phrase that drove each score. Quotes are validated as
# verbatim substrings of the source document, so this is a real check rather than decoration.
REQUIRE_EVIDENCE = True

# ###########################################################################################
# SUPPLIED BELOW THIS LINE - DO NOT MODIFY
# ###########################################################################################

# THE SCALE IS FIXED AT 0-100 INTEGERS. It is not a student choice, because the three
# supplied exemplars are written with 0/15/25/.../100 landmarks and rescaling them would
# make the locked rubrics incoherent. Scores are divided by 100 on aggregation, so every
# downstream contract sees 0-1.
SCORE_MIN, SCORE_MAX = 0, 100

FIELDS = list(CONSTRUCTS)
_client: OpenAI | None = None

EXEMPLAR_HASHES = {
    "policy_stance": "c1d9f9d7e666adfa",
    "uncertainty_language": "b25c2f6439d26a3c",
    "global_risk_salience": "20dbb4c6db17f337",
}


def client_() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _require_exemplars() -> None:
    """
    ABORT if a supplied exemplar has been edited. Not a warning.

    They are marked separately and every team must score against identical locked rubrics,
    or the audits are not comparable. If an exemplar genuinely fails on your data, that is a
    staff matter - report it, do not repair it yourself.
    """
    changed = [c for c in SUPPLIED_CONSTRUCTS
               if _sha(CONSTRUCTS[c]) != EXEMPLAR_HASHES.get(c)]
    if changed:
        raise RuntimeError(
            f"Supplied exemplar rubric(s) altered: {changed}. Restore them from the "
            f"repository. If you believe an exemplar is genuinely defective, report it to "
            f"the course staff - a versioned replacement will be issued.")


def _prompts_written() -> bool:
    return ("TODO" not in SYSTEM_PROMPT
            and not any("TODO" in CONSTRUCTS[c] for c in STUDENT_CONSTRUCTS))


def config_hash() -> str:
    """
    Identifies THIS scoring configuration. Everything that can change an answer is in it.

    The cache is keyed on this. An earlier version keyed only on the meeting date, so
    revising a rubric and re-running silently returned the previous run's answers - which
    made the required iteration impossible to perform and impossible to detect.
    """
    payload = json.dumps({
        "system_prompt": SYSTEM_PROMPT,
        "constructs": CONSTRUCTS,
        "scale": [SCORE_MIN, SCORE_MAX],
        "evidence": REQUIRE_EVIDENCE,
        "model": config.MODEL,
        "temperature": config.SAMPLING_TEMPERATURE,
        "seed": config.SEED,
        "n_calls": config.N_PARALLEL_CALLS,
    }, sort_keys=True)
    return _sha(payload, 12)


def run_dir() -> "os.PathLike":
    d = config.LLM_RAW / config_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


# -------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------

def _schema_model() -> type[BaseModel]:
    fields: dict = {}
    for name, rubric in CONSTRUCTS.items():
        fields[name] = (int, Field(ge=SCORE_MIN, le=SCORE_MAX, description=rubric))
        if REQUIRE_EVIDENCE:
            fields[f"{name}_evidence"] = (
                str, Field(description=f"The exact phrase from the document that drove the "
                                       f"{name} score. Quote it verbatim; do not paraphrase."))
    return create_model("ConstructScores", **fields)


def score_once(text: str, seed: int) -> dict:
    """
    One call, with bounded exponential backoff.

    Returns a REPRODUCIBILITY ENVELOPE: the parsed result plus the request settings, token
    usage, attempt count and request id. It is deliberately not called a raw response
    archive, because it is not one - the full API response object, including the unparsed
    message content and response headers, is not retained. What is here is enough to
    reproduce the run and to audit what was asked and answered.
    """
    schema = _schema_model()
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = client_().beta.chat.completions.parse(
                model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE, seed=seed,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content":
                           "RBA minutes of the monetary policy meeting:\n\n" + text}],
                response_format=schema)
            return {
                "ok": True,
                "parsed": r.choices[0].message.parsed.model_dump(),
                "model": config.MODEL,
                "temperature": config.SAMPLING_TEMPERATURE,
                "seed": seed,
                "config_hash": config_hash(),
                "prompt_hash": _sha(SYSTEM_PROMPT + json.dumps(CONSTRUCTS, sort_keys=True)),
                "request_id": getattr(r, "id", None),
                "usage": (r.usage.model_dump() if getattr(r, "usage", None) else None),
                "attempt": attempt + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BASE_SECONDS * (2 ** attempt)
                           + random.uniform(0, 0.5))
    return {"ok": False, "error": last_err, "seed": seed,
            "config_hash": config_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat()}


def score_document(date: str, text: str) -> dict:
    """N calls for one document, cached under the configuration hash."""
    out = run_dir() / f"{date}.json"
    if out.exists():
        calls = json.loads(out.read_text())
        if sum(1 for c in calls if c.get("ok")) >= config.N_PARALLEL_CALLS:
            return {"meeting_date": date, "calls": calls, "cached": True}
    with ThreadPoolExecutor(max_workers=config.N_PARALLEL_CALLS) as ex:
        futs = [ex.submit(score_once, text, config.SEED + i)
                for i in range(config.N_PARALLEL_CALLS)]
        calls = [f.result() for f in as_completed(futs)]
    out.write_text(json.dumps(calls, indent=1))
    return {"meeting_date": date, "calls": calls, "cached": False}


def _validate_evidence(quote: str, source: str) -> bool:
    """A quote must appear verbatim in the document. Whitespace-normalised comparison."""
    if not quote or len(quote) < 12:
        return False
    norm = lambda s: " ".join(s.split()).lower()
    return norm(quote) in norm(source)


def aggregate(records: list[dict], docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mean and spread per construct, plus per-call evidence and its validation status."""
    span = float(SCORE_MAX - SCORE_MIN)
    source = dict(zip(docs["meeting_date"], docs["text_scored"]))
    rows, ev_stats = [], {"checked": 0, "verbatim": 0}
    for rec in records:
        ok = [c["parsed"] for c in rec["calls"] if c.get("ok")]
        if len(ok) < config.MIN_VALID_CALLS:
            raise RuntimeError(
                f"{rec['meeting_date']}: only {len(ok)} valid calls, "
                f"minimum is {config.MIN_VALID_CALLS}. Re-run; do not proceed on partial "
                f"data.")
        row = {"meeting_date": rec["meeting_date"], "n_calls_valid": len(ok)}
        for f in FIELDS:
            v = np.array([c[f] for c in ok if f in c], dtype=float) / span
            row[f] = v.mean()
            row[f"{f}_sd"] = v.std(ddof=1) if len(v) > 1 else 0.0
            if REQUIRE_EVIDENCE:
                quotes = [c.get(f"{f}_evidence", "") for c in ok]
                good = [q for q in quotes
                        if _validate_evidence(q, source.get(rec["meeting_date"], ""))]
                ev_stats["checked"] += len(quotes)
                ev_stats["verbatim"] += len(good)
                # The quote from the call whose score is CLOSEST TO THE AGGREGATE, not the
                # longest one. The longest quote can come from the outlier call, so the
                # evidence shown could argue for a score the row does not report.
                order = np.argsort(np.abs(v - row[f]))
                row[f"{f}_evidence"] = next(
                    (quotes[i] for i in order
                     if i < len(quotes) and quotes[i] in good), 
                    (good or quotes or [""])[0])
                row[f"{f}_evidence_verbatim"] = len(good) / max(1, len(quotes))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True), ev_stats


# -------------------------------------------------------------------------------------------
# The audit
# -------------------------------------------------------------------------------------------

def check_construct_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three properties, each with an explicit criterion.

    DISCRIMINATION - binned concentration. Scores are binned at config.BIN_WIDTH and the
        share falling in the busiest bin is reported. This is a CONCENTRATION statistic, not
        the share at an exact value, and the bin width is part of its definition. Standard
        deviation alone passes a construct that is constant for most of the corpus, because
        a minority of dispersed documents drags the sd up.

    SEPARATION - the between/within variance ratio. Between-document variance divided by
        mean within-document (call-to-call) variance. Below 1 the construct varies more
        between repeat calls on one document than between different documents, which means
        it is measuring noise.

        This is NOT reliability in the psychometric sense, and it was called that until a
        reviewer pointed out the overstatement. It is an ad hoc signal-to-noise diagnostic
        computed on five calls per document: no variance-component model, no uncertainty,
        no intraclass correlation. Read it as "does this dimension distinguish documents
        by more than it wobbles", which is a useful question, and not as a reliability
        coefficient you could quote in a methods section.

    COVERAGE - effective number of bins, from the entropy of the binned distribution. A
        construct using three effective bins out of twenty is coarse even if no single bin
        dominates.
    """
    rows = []
    for f in FIELDS:
        v = df[f].dropna()
        binned = (v / config.BIN_WIDTH).round().astype(int)
        share = binned.value_counts(normalize=True)
        conc = float(share.iloc[0])
        entropy = float(-(share * np.log(share)).sum())
        within = float((df[f"{f}_sd"] ** 2).mean())
        snr = float(v.var() / within) if within > 0 else np.inf
        rows.append({
            "construct": f,
            "sd": float(v.std()),
            "binned_concentration": conc,
            "modal_bin_centre": float(share.index[0] * config.BIN_WIDTH),
            "effective_bins": float(np.exp(entropy)),
            "mean_call_sd": float(df[f"{f}_sd"].mean()),
            "between_within_ratio": snr,
            "evidence_verbatim_rate": (float(df[f"{f}_evidence_verbatim"].mean())
                                       if f"{f}_evidence_verbatim" in df.columns else np.nan),
            "passes_spread": float(v.std()) >= config.MIN_CONSTRUCT_SPREAD,
            "passes_concentration": conc <= config.MAX_BINNED_CONCENTRATION,
            "passes_coverage": float(np.exp(entropy)) >= config.MIN_EFFECTIVE_BINS,
            "passes_separation": snr >= config.MIN_SIGNAL_TO_NOISE,
        })
    out = pd.DataFrame(rows)
    print(f"  gates: sd >= {config.MIN_CONSTRUCT_SPREAD}, concentration <= "
          f"{config.MAX_BINNED_CONCENTRATION:.0%} at bin width {config.BIN_WIDTH}, "
          f"effective bins >= {config.MIN_EFFECTIVE_BINS}, "
          f"between/within >= {config.MIN_SIGNAL_TO_NOISE}")
    for r in out.itertuples():
        fails = [n for n, ok in (("spread", r.passes_spread),
                                 ("concentration", r.passes_concentration),
                                 ("coverage", r.passes_coverage),
                                 ("separation", r.passes_separation)) if not ok]
        flag = f"   <-- FAILS {', '.join(fails)}" if fails else ""
        print(f"  {r.construct:30s} sd={r.sd:.3f} conc={r.binned_concentration:5.1%} "
              f"eff_bins={r.effective_bins:4.1f} B/W={r.between_within_ratio:5.1f} "
              f"quotes_ok={r.evidence_verbatim_rate:5.1%}{flag}")
    return out


# What each construct SHOULD do as the decision moves cut -> hold -> hike.
#   "up"    higher on hikes  (hawkishness, inflation concern)
#   "down"  higher on cuts   (downside-risk emphasis - it runs the other way, and an earlier
#           version demanded cut < hold < hike for every construct, so the one dimension
#           that is correctly inverted was reported as failing)
#   None    no directional expectation; reported but not judged
EXPECTED_ORIENTATION = {
    "policy_stance": "up",
    "inflation_concern": "up",
    "downside_risk_emphasis": "down",
    "financial_conditions_concern": None,
    "uncertainty_language": None,
    "vigilance": None,
    "global_risk_salience": None,
}


def orientation_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does each SIGNED dimension point the right way?

    Mean score grouped by the decision actually taken. `policy_stance` must be monotonic
    with a visible gap between cuts and holds. This is the check that catches a dimension
    reading the Board's forcefulness as hawkishness, and no spread or stability statistic
    can substitute for it.
    """
    panel_path = config.DATA_PROCESSED / "panel.parquet"
    if not panel_path.exists():
        print("  (no panel yet - run `python src/data_panel.py` to enable this check)")
        return pd.DataFrame()
    p = pd.read_parquet(panel_path)
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    s = df.copy()
    s["meeting_date"] = pd.to_datetime(s["meeting_date"])
    j = s.merge(p[["meeting_date", "decision"]], on="meeting_date")

    rows = []
    for f in FIELDS:
        g = j.groupby("decision")[f].mean()
        cut, hold, hike = g.get(-1, np.nan), g.get(0, np.nan), g.get(1, np.nan)
        want = EXPECTED_ORIENTATION.get(f)
        if want == "up":
            ok = bool(cut < hold < hike)
            gap = float(hold - cut)
        elif want == "down":
            ok = bool(cut > hold > hike)
            gap = float(cut - hold)
        else:
            ok, gap = None, float(abs(hike - cut))
        rows.append({"construct": f, "mean_on_cut": cut, "mean_on_hold": hold,
                     "mean_on_hike": hike, "expected": want,
                     "monotonic_as_expected": ok, "cut_hold_gap": gap})
    out = pd.DataFrame(rows)
    print("\n  ORIENTATION (mean score by the decision actually taken)")
    print("  NOTE: for policy_stance this is an IMPLEMENTATION CHECK, not external "
          "validation -")
    print("  the supplied rubric names the decision explicitly, so monotonicity confirms "
          "the rubric")
    print("  was applied, not that the construct measures something independent of it.")
    for r in out.itertuples():
        if r.expected is None:
            note = "   (no directional expectation)"
        elif r.monotonic_as_expected and r.cut_hold_gap > 0.1:
            note = "   OK"
        else:
            note = f"   <-- expected to run {r.expected} across cut/hold/hike"
        print(f"  {r.construct:30s} cut={r.mean_on_cut:.3f} hold={r.mean_on_hold:.3f} "
              f"hike={r.mean_on_hike:.3f}{note}")
    return out


# -------------------------------------------------------------------------------------------
# Episode separation
# -------------------------------------------------------------------------------------------
# Prompt iteration and prompt validation must not use the same episodes. If you tune a rubric
# until it produces the right answer on the GFC, then report that it produces the right answer
# on the GFC, you have reported your own tuning back to yourself.
#
# DEVELOPMENT episodes are what you may read, argue about and tune against.
# VALIDATION episodes are scored ONCE, after the rubrics are final, and reported as they fall.
# The supplied exemplars name no episode at all, for the same reason.

DEV_EPISODES = {
    "pre-GFC tightening":   ("2007-08-07", "2008-03-04"),
    "mining boom plateau":  ("2011-02-01", "2011-11-01"),
    "post-taper easing":    ("2013-05-07", "2013-08-06"),
}

VALIDATION_EPISODES = {
    "GFC":                  ("2008-09-02", "2009-04-07"),
    "COVID onset":          ("2020-03-03", "2020-11-03"),
    "2022 tightening":      ("2022-05-03", "2022-12-06"),
    "calm (2015-16)":       ("2015-01-01", "2016-12-31"),
}


def episode_check(df: pd.DataFrame, which: str = "validation") -> pd.DataFrame:
    """
    Mean construct scores over named episodes. `which` is 'dev' or 'validation'.

    Run 'dev' as often as you like while writing rubrics. Run 'validation' ONCE, when the
    rubrics are frozen, and report what it gives you - including the dimensions that come
    out wrong. A construct that fails here and is then re-tuned must be revalidated, and the
    report must say that it was.
    """
    eps = DEV_EPISODES if which == "dev" else VALIDATION_EPISODES
    d = df.copy()
    d["meeting_date"] = pd.to_datetime(d["meeting_date"])
    rows = []
    for name, (a, b) in eps.items():
        w = d[(d.meeting_date >= a) & (d.meeting_date <= b)]
        if not len(w):
            continue
        r = {"episode": name, "n": len(w)}
        r.update({f: float(w[f].mean()) for f in FIELDS})
        rows.append(r)
    out = pd.DataFrame(rows)
    print(f"\n  {which.upper()} EPISODES (mean score)")
    if len(out):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.round(3).to_string(index=False))
    return out


def load_documents() -> pd.DataFrame:
    docs = pd.read_parquet(config.DOCUMENTS)
    docs["meeting_date"] = pd.to_datetime(docs["meeting_date"]).dt.strftime("%Y-%m-%d")
    docs["text_scored"] = docs["text_full"]
    return docs


# -------------------------------------------------------------------------------------------
# Development and validation are SEPARATE RUNS over DISJOINT documents
# -------------------------------------------------------------------------------------------
#
#     python src/text_features.py                 development: everything EXCEPT the
#                                                 validation meetings. Run as often as you like.
#     python src/text_features.py --validate      one shot, validation meetings only, stamped
#     python src/text_features.py --validate --revalidate
#                                                 do it again after a prompt change, recorded
#                                                 as a second exposure. Declare it in the report.
#     python src/text_features.py --dry-run       no API calls
#
# WHY THIS IS TWO COMMANDS AND NOT ONE. An earlier version scored ALL 211 documents on every
# run and stamped the validation table on the first one. The validation episodes were
# therefore scored under every prompt a team ever tried; freezing the first table afterwards
# only hid that, it did not prevent it. Held out means NOT SCORED, not "scored and then not
# shown".

VALIDATION_STAMP = config.OUTPUTS / "words_validation.json"


def validation_meetings(docs: pd.DataFrame) -> set[str]:
    """The meeting dates reserved for validation. Never scored in a development run."""
    d = pd.to_datetime(docs["meeting_date"])
    keep: set[str] = set()
    for a, b in VALIDATION_EPISODES.values():
        keep |= set(docs.loc[(d >= a) & (d <= b), "meeting_date"])
    return keep


def _score_documents(docs: pd.DataFrame, label: str) -> tuple[list[dict], dict]:
    t0, records = time.time(), []
    with ThreadPoolExecutor(max_workers=config.N_DOC_WORKERS) as ex:
        futs = [ex.submit(score_document, r.meeting_date, r.text_scored)
                for r in docs.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            records.append(f.result())
            if i % 50 == 0:
                print(f"    {i}/{len(docs)} ({time.time()-t0:.0f}s)")
    fresh = [c for r in records if not r["cached"] for c in r["calls"]]
    usage = {"scope": label,
             "api_calls": len(fresh),
             "failed_calls": sum(1 for c in fresh if not c.get("ok")),
             "retried_calls": sum(1 for c in fresh if c.get("attempt", 1) > 1),
             "prompt_tokens": sum((c.get("usage") or {}).get("prompt_tokens", 0)
                                  for c in fresh),
             "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0)
                                      for c in fresh),
             "wall_seconds": round(time.time() - t0, 1),
             "documents_cached": sum(1 for r in records if r["cached"])}
    print(f"\n  {usage['api_calls']} fresh calls "
          f"({usage['failed_calls']} failed, {usage['retried_calls']} retried), "
          f"{usage['prompt_tokens']:,} in / {usage['completion_tokens']:,} out tokens, "
          f"{usage['wall_seconds']:.0f}s, {usage['documents_cached']} from cache")
    return records, usage


def _guard(dry_run: bool, docs: pd.DataFrame) -> bool:
    _require_exemplars()
    print(f"  {len(docs)} documents, median {docs.text_scored.str.len().median():,.0f} chars")
    print(f"  config hash {config_hash()}  ->  {run_dir()}")
    if dry_run:
        print("  --dry-run: stopping before any API call.")
        return False
    if not _prompts_written():
        missing = [c for c in STUDENT_CONSTRUCTS if "TODO" in CONSTRUCTS[c]]
        raise NotImplementedError(
            f"Write SYSTEM_PROMPT and your four rubrics first. Still TODO: "
            f"{missing or ['SYSTEM_PROMPT']}.")
    return True


def _write_scores(df: pd.DataFrame, which: str) -> int:
    """
    Write this run's partial and rebuild the combined table from whatever exists.

    Development and validation score DISJOINT meetings, so neither alone covers the corpus.
    The panel needs both: until validation has been run, its meetings simply have no
    construct values and `data_panel` says so rather than silently carrying NaNs into
    Replay and Shock.
    """
    target = (config.CONSTRUCT_SCORES_DEV if which == "development"
              else config.CONSTRUCT_SCORES_VAL)
    df.to_parquet(target, index=False)
    parts = [pd.read_parquet(p) for p in
             (config.CONSTRUCT_SCORES_DEV, config.CONSTRUCT_SCORES_VAL) if p.exists()]
    combined = (pd.concat(parts, ignore_index=True)
                  .drop_duplicates(subset="meeting_date", keep="last")
                  .sort_values("meeting_date").reset_index(drop=True))
    combined.to_parquet(config.CONSTRUCT_SCORES, index=False)
    return len(combined)


def run(dry_run: bool = False) -> pd.DataFrame | None:
    """
    DEVELOPMENT run. Scores every document EXCEPT the validation meetings.

    Iterate here as much as you like: the episodes you will be judged on are not in this
    sample and cannot be inspected from it.
    """
    all_docs = load_documents()
    held = validation_meetings(all_docs)
    docs = all_docs[~all_docs["meeting_date"].isin(held)].reset_index(drop=True)
    print(f"  DEVELOPMENT run - {len(held)} validation meetings withheld "
          f"({len(docs)} of {len(all_docs)} scored)")
    if not _guard(dry_run, docs):
        return None

    records, usage = _score_documents(docs, "development")
    df, ev = aggregate(records, docs)
    print(f"\n  evidence quotes verbatim: {ev['verbatim']}/{ev['checked']} "
          f"({ev['verbatim']/max(1,ev['checked']):.0%})")
    print("\n  CONSTRUCT QUALITY AUDIT (development documents only)")
    quality = check_construct_quality(df)
    orient = orientation_check(df)
    episode_check(df, "dev")

    n_total = _write_scores(df, "development")
    print(f"  construct_scores.parquet now covers {n_total} of 211 meetings")
    (config.OUTPUTS / "words_audit.json").write_text(json.dumps({
        "scope": "development",
        "config_hash": config_hash(),
        # Repository-relative: an absolute path here leaked the author's machine layout
        # into a committed artefact and is meaningless on anyone else's.
        "run_dir": str(run_dir().relative_to(config.ROOT)).replace("\\", "/"),
        "n_documents_scored": len(docs),
        "n_validation_withheld": len(held),
        "usage": usage,
        "evidence": ev,
        "quality": quality.round(4).to_dict("records"),
        "orientation": orient.round(4).to_dict("records") if len(orient) else [],
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n  when your rubrics are final: python src/text_features.py --validate")
    return df


def validate(dry_run: bool = False, revalidate: bool = False) -> pd.DataFrame | None:
    """
    VALIDATION run. Scores ONLY the held-out meetings, once, and records what produced it.

    The stamp stores the prompt text, the config hash and the exact meeting ids, so a marker
    can confirm that the rubrics which produced the validation numbers are the rubrics you
    submitted. Running it a second time requires `--revalidate` and is recorded as a second
    exposure - which is a defensible choice you must declare, not a silent one.
    """
    if VALIDATION_STAMP.exists() and not revalidate:
        rec = json.loads(VALIDATION_STAMP.read_text())
        same = rec["config_hash"] == config_hash()
        print(f"\n  VALIDATION ALREADY RUN on {rec['run_at'][:10]} under config "
              f"{rec['config_hash']}")
        print(f"  your prompts are {'UNCHANGED' if same else 'DIFFERENT'} since then")
        if not same:
            print("  re-running would be a SECOND exposure of the held-out episodes. If you "
                  "need it,")
            print("  pass --revalidate and say so in your report.")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pd.DataFrame(rec["episodes"]).round(3).to_string(index=False))
        return pd.DataFrame(rec["episodes"])

    all_docs = load_documents()
    held = validation_meetings(all_docs)
    docs = all_docs[all_docs["meeting_date"].isin(held)].reset_index(drop=True)
    print(f"  VALIDATION run - {len(docs)} held-out meetings only")
    if revalidate and VALIDATION_STAMP.exists():
        print("  --revalidate: this is a REPEAT exposure and is recorded as one")
    if not _guard(dry_run, docs):
        return None

    records, _ = _score_documents(docs, "validation")
    df, _ = aggregate(records, docs)
    n_total = _write_scores(df, "validation")
    print(f"  construct_scores.parquet now covers {n_total} of 211 meetings")
    out = episode_check(df, "validation")

    prior = []
    if VALIDATION_STAMP.exists():
        old = json.loads(VALIDATION_STAMP.read_text())
        prior = old.get("previous_exposures", []) + [{
            "run_at": old["run_at"], "config_hash": old["config_hash"]}]
    VALIDATION_STAMP.write_text(json.dumps({
        "config_hash": config_hash(),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "exposure_number": len(prior) + 1,
        "previous_exposures": prior,
        "meetings": sorted(held),
        "system_prompt": SYSTEM_PROMPT,
        "constructs": CONSTRUCTS,
        "episodes": out.round(4).to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    if prior:
        print(f"\n  RECORDED AS EXPOSURE {len(prior) + 1}. Your report must say why you "
              f"revalidated.")
    else:
        print(f"\n  frozen to {VALIDATION_STAMP.name} with the prompts that produced it")
    return out


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate(dry_run="--dry-run" in sys.argv, revalidate="--revalidate" in sys.argv)
    else:
        run(dry_run="--dry-run" in sys.argv)

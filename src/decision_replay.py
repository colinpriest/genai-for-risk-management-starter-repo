"""
REPLAY - reconstruct a historical RBA decision by n-shot prompting.

YOU SUPPLY THE PROMPT AND THE SHOT STRATEGY. EVERYTHING ELSE IS BUILT.

WHAT YOU DO

  1. Choose a meeting from draft-assignment/replay-meeting-shortlist.md and set MEETING.

  2. Write RECOMMENDATION_PROMPT. It receives k worked historical meetings - conditions in,
     decision out - and then the target meeting's conditions, and must return a
     recommendation with reasoning and transmission mechanisms.

  3. Choose the shot strategy and k, and JUSTIFY THEM. This matters more than the prompt
     wording. See nshot.py: `recent`, `similar`, `stratified`, `regimes`.

  4. Run the A/B: the same prompt with and without your Cycle causal findings included as
     guidance. Does telling the model which associations are causal improve the answer?

  5. Have the model write the public statement, then audit every claim in it.

WHAT IS SUPPLIED
    load_as_at()          the panel truncated to the target, with that meeting's own
                          decision and targets blanked. The lagged constructs are KEPT -
                          the previous minutes were public before this meeting
    feasible_strategies() which strategies can actually be built for your meeting at your k
    nshot.select_shots()  the four strategies, with three leakage checks that raise
    dev_sample()          meetings you may iterate on
    holdout_sample()      meetings you run once and report as they fall
    evaluate_all()        every strategy across a sample, concurrently and cached
    audit_statement()     claim-by-claim check against evidence built from the PANEL
    arithmetic_check()    deterministic check that the statement quotes the right new rate

HOW WORDS REACHES REPLAY
    Every shot carries the seven construct scores from the PREVIOUS meeting's minutes, so
    the model sees not only what the economy was doing but how the Board had been writing
    about it. If your Words constructs are poor, Replay inherits the problem.

THE TRAP YOU WILL HIT
    74% of meetings are holds. `recent` and `similar` will both hand the model examples that
    are almost entirely holds, and it will learn to say hold. `stratified` fixes the balance
    but shows a history that never happened. Print `shot_mix()` before you spend credit.

WRITES outputs/replay.json, outputs/replay_statement.md
       data/processed/replay_raw/*.json  (every raw call, for reproduction without a key)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field as PField

sys.path.insert(0, os.path.dirname(__file__))
import config      # noqa: E402
import nshot       # noqa: E402

load_dotenv()
_client: OpenAI | None = None


def client_() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ###########################################################################################
# YOUR WORK STARTS HERE
# ###########################################################################################

MEETING = "TODO: choose from replay-meeting-shortlist.md, e.g. 2010-11-02"

K_SHOTS = 9
SHOT_STRATEGY = "stratified"     # recent | similar | stratified | regimes

# Paste your Cycle verdicts here for the A/B in step 4. Leave empty to skip the A/B.
CAUSAL_GUIDANCE = ""
# e.g. """Our causal testing found:
#  - slope_cash3y is REVERSE causal: the bond market forecasts the RBA, it does not move it.
#    Do not treat a steep curve as a reason to tighten; treat it as other people's forecast.
#  - trimmed_mean_yoy is CAUSAL: it is half the Board's mandate.
#  - policy_stance is mostly a RECORD of the decision just taken, not a leading indicator."""

RECOMMENDATION_PROMPT = """
TODO: WRITE THE RECOMMENDATION PROMPT.

The model receives k worked historical meetings with the decision taken at each, then the
target meeting's conditions with no decision. It must recommend hike / hold / cut.

It must return:
  {"recommendation": "hike"|"hold"|"cut",
   "size_bp": int,
   "confidence": "low"|"moderate"|"high",
   "key_risks": [str],
   "reasoning": str,
   "transmission_mechanisms": [{"channel": str, "horizon": str, "mechanism": str}],
   "what_would_change_our_mind": str}

Things worth deciding explicitly:
  - what you tell it about the RBA's mandate and reaction function
  - whether you tell it the base rate of holds, or let the examples imply it
  - how you stop it simply pattern-matching the most recent example
  - how you make it commit to a size, not just a direction
"""

STATEMENT_PROMPT = """
TODO: WRITE THE STATEMENT PROMPT.

The model receives your recommendation and the evidence behind it, and must write a 250-350
word public statement in the RBA's register.

The constraint that carries the marks: it may only assert what the evidence supports. Write
that constraint so it actually binds - and then audit whether it did.
"""

# YOUR verdict on the machine's classifications, keyed by CLAIM ID.
#
# The audit assigns every claim a stable id - the first 8 hex characters of the SHA-256 of
# its normalised text - printed in the audit output and stored in replay.json. Reviews are
# keyed on that id EXACTLY: an earlier version matched 60-character substrings, so one
# malformed record could blanket several claims, be counted once per match, and satisfy
# the requirement while carrying no verdict at all.
#
#     "a1b2c3d4": {
#         "human_verdict": "forecast-or-judgement",   # one of the five audit categories
#         "reason": "the evidence has one unemployment print, not a trend",
#         "final_action": "kept the number, flagged the inference",
#         "by": "AB"},
#
# All four fields are required; human_verdict must be one of the five categories. Review
# at least MIN_CLAIM_REVIEWS distinct claims - the highest-risk classifications, typically
# anything data-supported that rests on a number and anything unverifiable a person could
# in fact verify. A DISAGREEMENT is recorded where one exists; where none does, the review
# is a DEFENDED agreement. The reportable run raises, not warns, until this holds.
CLAIM_REVIEWS: dict[str, dict] = {}


# ###########################################################################################
# SUPPLIED BELOW THIS LINE - DO NOT MODIFY
# ###########################################################################################
# How many classifications you must review yourself. Requiring a DISAGREEMENT outright,
# which an earlier version did, rewards performative contradiction: the auditing model
# is sometimes simply right. What is required is that you looked, on the claims where
# being wrong would matter most, and that an agreement is defended rather than assumed.
MIN_CLAIM_REVIEWS = 3

# Seeds per meeting in the strategy comparison. Three is what turns one lucky call into a
# majority vote; the stage hash covers it because changing it changes every reported
# accuracy.
N_SEEDS = 3


def config_hash() -> str:
    """
    Identifies THIS Replay configuration - prompts, settings and the human judgement
    tables. Stored in replay.json and compared by the submission check, so an artefact
    generated under different prompts than the ones in the repository cannot pass as
    current. Words has carried the same guarantee since its cache was rebuilt; Replay and
    Shock used to carry none, which in an assignment about GenAI meant the marker could
    not establish that the assessed prompts produced the submitted results.
    """
    payload = json.dumps({
        "recommendation_prompt": RECOMMENDATION_PROMPT,
        "statement_prompt": STATEMENT_PROMPT,
        "audit_prompt": AUDIT_PROMPT,
        # CAUSAL_GUIDANCE is inserted verbatim into the guided recommendation prompt - an
        # earlier version of this hash omitted it, so the entire guidance block could be
        # rewritten after artefact generation without the submission check noticing.
        "causal_guidance": CAUSAL_GUIDANCE,
        "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
        "seed": config.SEED, "n_seeds": N_SEEDS,
        "meeting": MEETING, "k": K_SHOTS, "strategy": SHOT_STRATEGY,
        "claim_reviews": CLAIM_REVIEWS,
        "min_claim_reviews": MIN_CLAIM_REVIEWS,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Feasibility note: the two earliest meetings have short histories. Before 2007-11-06 there
# are not three cuts on record, and the forward-looking cycle label has not resolved for the
# recent past, so `stratified` and `regimes` cannot be built there at any k. Those meetings
# support `recent` and `similar` only. `feasible_strategies()` reports this before a run
# rather than letting select_shots raise halfway through one.
SHORTLIST = ["2007-11-06", "2008-10-07", "2010-11-02", "2013-08-06", "2022-05-03"]

RAW_DIR = config.DATA_PROCESSED / "replay_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class TransmissionMechanism(BaseModel):
    channel: str
    horizon: str
    mechanism: str


class Recommendation(BaseModel):
    """
    Schema-validated, so a malformed reply is a retry rather than a silent empty dict.

    `size_bp` is cross-checked against `recommendation` after parsing - the schema alone
    permits "hold, 25bp" and "cut, 0bp", both of which are incoherent and both of which the
    model produced during development.
    """
    recommendation: Literal["hike", "hold", "cut"]
    size_bp: int = PField(ge=0, le=200)
    confidence: Literal["low", "moderate", "high"]
    key_risks: list[str]
    reasoning: str
    transmission_mechanisms: list[TransmissionMechanism]
    what_would_change_our_mind: str


def validate_recommendation(rec: dict) -> str | None:
    """Deterministic coherence check the schema cannot express. Returns a problem or None."""
    d, bp = rec.get("recommendation"), rec.get("size_bp")
    if d is None or bp is None:
        return "missing recommendation or size_bp"
    if d == "hold" and bp != 0:
        return f"'hold' with a size of {bp}bp is not a coherent recommendation"
    if d in ("hike", "cut") and bp == 0:
        return f"'{d}' with a size of 0bp is not a coherent recommendation"
    if d in ("hike", "cut") and bp % 5:
        return f"{bp}bp is not a multiple of 5 - the RBA does not move in these increments"
    return None


def _prompts_written() -> bool:
    return ("TODO" not in RECOMMENDATION_PROMPT and "TODO" not in STATEMENT_PROMPT
            and "TODO" not in MEETING)


def _panel(pre_decision: bool = True) -> pd.DataFrame:
    """
    The meeting panel. For Replay, on the PRE-DECISION market state by default.

    WHY. The panel's market and rate features are the last close on or before the meeting,
    which on a meeting day is the meeting-day close. The RBA announces at 2:30pm Sydney and
    the ASX closes at 4pm, with bonds and FX trading through, so that close already contains
    the decision. On 7 October 2008 - a 100bp cut - `bab_spread` moved 63 basis points
    between the previous close and the meeting close. Handing Replay that number is handing
    it the answer.

    `data_panel` therefore also stores every market and rate feature as at the previous
    business day, suffixed `__pre`. With `pre_decision=True` those values replace the
    same-day ones under their ordinary names, so nothing downstream has to know.

    This applies to the SHOTS as well as the target. A shot that pairs post-announcement
    conditions with the decision teaches the model to read the announcement.

    Cycle does NOT do this, and should not: its estimand is explicitly "predict at the end
    of meeting T, knowing T's decision".
    """
    p = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    p = p.set_index("meeting_date").sort_index()
    if pre_decision:
        pre = [c for c in p.columns if c.endswith("__pre")]
        if not pre:
            raise RuntimeError(
                "the panel has no __pre columns - rebuild it with `python src/data_panel.py`. "
                "Replay must not run on meeting-day closes.")
        for c in pre:
            p[c[:-5]] = p[c]
        p = p.drop(columns=pre)
    return p


def feasible_strategies(meeting: str, k: int | None = None) -> list[str]:
    """Which shot strategies can actually be built for this meeting at this k."""
    k = K_SHOTS if k is None else k
    panel = _panel()
    out = []
    for s in nshot.STRATEGIES:
        try:
            nshot.select_shots(panel, pd.Timestamp(meeting), k=k, strategy=s)
            out.append(s)
        except Exception:  # noqa: BLE001
            pass
    return out


def load_as_at(meeting_date: str) -> pd.DataFrame:
    """
    The panel as it stood immediately before a meeting.

    Two cuts:
      1. every row after the meeting is removed
      2. the meeting's own decision, rate change and both targets are blanked

    WHY THE CONSTRUCTS ARE NOT BLANKED. `data_panel` lags the whole text tier by one
    meeting, so the construct values on row M are scored from the minutes of M-1. Those
    minutes published about 14 days after M-1 and the meetings are roughly five weeks apart,
    so they were public well before M. `data_panel._assert_text_available()` fails the build
    if any meeting gap is short enough to break that.

    Blanking them - which an earlier version did - would have shown the model seven construct
    values for every historical example and none for the meeting it is being asked to judge.
    That is not caution, it is an asymmetric prompt.
    """
    p = _panel()
    d = pd.Timestamp(meeting_date)
    if d not in p.index:
        raise ValueError(f"{meeting_date} is not a meeting date")
    hist = p.loc[:d].copy()
    for c in ("decision", "change_pct", "y_cycle", "y_decision",
              "y_decision_change", "forward_change"):
        if c in hist.columns:
            hist.loc[d, c] = np.nan
    return hist


def actual_decision(meeting_date: str) -> dict:
    """What the RBA actually did. For the comparison step ONLY - not for the prompt."""
    r = _panel().loc[pd.Timestamp(meeting_date)]
    d = int(r["decision"])
    return {"decision": d, "change_pct": float(r["change_pct"]),
            "size_bp": int(round(abs(float(r["change_pct"])) * 100)),
            "word": nshot.DECISION_WORD[d],
            "label": nshot.decision_label(d, float(r["change_pct"]))}


def rate_before(meeting_date: str) -> float:
    """The cash rate going INTO the meeting. Used for the deterministic arithmetic check."""
    return float(_panel().loc[pd.Timestamp(meeting_date), "cash_rate"])


# -------------------------------------------------------------------------------------------
# Calling, cached and retried
# -------------------------------------------------------------------------------------------

def _cache_key(kind: str, system: str, user: str, seed_offset: int = 0) -> str:
    return hashlib.sha256(
        json.dumps([kind, system, user, config.MODEL, config.SAMPLING_TEMPERATURE,
                    config.SEED + seed_offset], sort_keys=True
                   ).encode("utf-8")).hexdigest()[:16]


def _cached_call(kind: str, system: str, user: str,
                 schema: type[BaseModel] | None = None, seed_offset: int = 0) -> dict:
    """
    One call, cached on the full request and retried with bounded exponential backoff.

    Every raw reply is kept under data/processed/replay_raw/ so a marker can reproduce the
    run without an API key, and so re-running does not re-bill work already done. The key
    covers both prompts, the model, the temperature and the seed, so editing a prompt
    produces a genuinely fresh call rather than the previous answer.
    """
    key = _cache_key(kind, system, user, seed_offset)
    path = RAW_DIR / f"{kind}-{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("ok"):
            return rec
    last = None
    for attempt in range(config.MAX_RETRIES):
        try:
            if schema is not None:
                r = client_().beta.chat.completions.parse(
                    model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                    seed=config.SEED + seed_offset, response_format=schema,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                payload = r.choices[0].message.parsed.model_dump()
            else:
                r = client_().chat.completions.create(
                    model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                    seed=config.SEED + seed_offset,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                payload = {"text": r.choices[0].message.content.strip()}
            rec = {"ok": True, "kind": kind, "payload": payload,
                   "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
                   "seed": config.SEED + seed_offset, "attempt": attempt + 1,
                   "prompt_sha": key, "request_id": getattr(r, "id", None),
                   "usage": (r.usage.model_dump() if getattr(r, "usage", None) else None),
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            path.write_text(json.dumps(rec, indent=1))
            return rec
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:140]}"
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, .5))
    return {"ok": False, "kind": kind, "error": last,
            "timestamp": datetime.now(timezone.utc).isoformat()}


def recommend(meeting: str, k: int | None = None, strategy: str | None = None,
              causal_guidance: str = "", seed_offset: int = 0) -> dict:
    """One n-shot recommendation. Returns the parsed result plus the shot mix."""
    k = K_SHOTS if k is None else k
    strategy = SHOT_STRATEGY if strategy is None else strategy
    target = pd.Timestamp(meeting)
    # Shots come from the full panel - past decisions are public - but the TARGET row is read
    # from the as-at frame, so the meeting's own decision cannot reach the prompt.
    block, shots = nshot.build_prompt_block(
        _panel(), target, k=k, strategy=strategy, as_at=load_as_at(meeting))
    sep = "=" * 60
    user = block if not causal_guidance else (
        f"{block}\n\n{sep}\nCAUSAL GUIDANCE FROM OUR OWN ANALYSIS:\n{causal_guidance}")
    rec = _cached_call("rec", RECOMMENDATION_PROMPT, user, schema=Recommendation,
                       seed_offset=seed_offset)
    return {"meeting": meeting, "k": k, "strategy": strategy,
            "with_causal_guidance": bool(causal_guidance),
            "shot_mix": nshot.shot_mix(shots),
            "ok": rec.get("ok", False), "error": rec.get("error"),
            "prompt_chars": len(user),
            "result": rec.get("payload", {})}


# -------------------------------------------------------------------------------------------
# Choosing a strategy without choosing it on your answer
# -------------------------------------------------------------------------------------------

def benchmark_sample(n_per_class: int = 12, min_history: int = 60) -> list[str]:
    """
    Meetings for comparing shot strategies. Stratified by ACTUAL decision.

    WHY NOT JUST THE FIVE SHORTLISTED MEETINGS. Five is far too few to separate four
    strategies - one lucky call moves a strategy by 20 percentage points and you would be
    reporting noise.

    WHY STRATIFIED. On a random sample 74% of meetings are holds, "always hold" scores 74%,
    and no strategy can be distinguished from any other. Stratifying the EVALUATION set is
    not cheating - it is how you get a readable signal on the minority classes. Report
    balanced accuracy alongside raw, and say what you did.
    """
    p = _panel().iloc[min_history:]
    out = []
    for d in (-1, 0, 1):
        sub = p[p["decision"] == d]
        idx = np.linspace(0, len(sub) - 1, min(n_per_class, len(sub))).astype(int)
        out += [str(x.date()) for x in sub.index[idx]]
    return sorted(out)


def dev_sample() -> list[str]:
    """
    Meetings you may iterate on: try strategies, k values and prompt wordings here.

    Alternate meetings of the stratified sample, so dev and holdout are balanced the same way
    and span the same period.
    """
    return benchmark_sample()[0::2]


def holdout_sample() -> list[str]:
    """
    Meetings you run ONCE, after the strategy and k are fixed, and report as they fall.

    Choosing the strategy on the same meetings you then quote the accuracy from is selection
    on the outcome: the best of four strategies looks several points better than it is. Pick
    on `dev_sample()`, report on this. If you go back and change the strategy after seeing
    these numbers, say so in the report - that is a defensible choice, but only if declared.
    """
    return benchmark_sample()[1::2]


def evaluate_all(k: int | None = None, strategies: list[str] | None = None,
                 causal_guidance: str = "", meetings: list[str] | None = None,
                 sample: str = "dev", max_workers: int = 8,
                 n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """
    Run every strategy across a PAIRED sample of meetings, repeated over several seeds.

    THREE THINGS THIS FIXES, ALL OF WHICH MADE THE OLD COMPARISON UNREADABLE.

    PAIRING. Meetings where any strategy cannot be built are dropped for ALL strategies and
    reported. Previously each strategy was scored on whatever it could manage, so the ones
    with the strictest selection rules skipped the hardest early meetings and were rewarded
    for it.

    REPETITION. The model is sampled at temperature 1, so a single call per meeting is one
    draw. With 18 meetings a strategy can move five points on noise alone. Each meeting is
    now run `n_seeds` times and the per-meeting majority vote is scored, with the unanimity
    rate reported so you can see how settled each call was.

    SIZE SEPARATELY. `size_exact` used to include holds, where the size is trivially 0, so
    a strategy predicting more holds scored better on sizing without being better at it.
    Size accuracy is now conditional on a move having been correctly called.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sklearn.metrics import balanced_accuracy_score

    k = K_SHOTS if k is None else k
    strategies = strategies or list(nshot.STRATEGIES)
    if meetings is None:
        meetings = dev_sample() if sample == "dev" else holdout_sample()
    meetings, dropped = feasible_everywhere(meetings, strategies, k)
    if dropped:
        print(f"  {len(dropped)} meeting(s) dropped so every strategy is scored on the same "
              f"set:")
        for m, bad in list(dropped.items())[:5]:
            print(f"    {m}: infeasible for {bad}")
    jobs = [(s, m, i) for s in strategies for m in meetings for i in range(n_seeds)]

    def _one(job):
        strat, m, seed_i = job
        try:
            r = recommend(m, k=k, strategy=strat, causal_guidance=causal_guidance,
                          seed_offset=seed_i)
        except Exception as e:  # noqa: BLE001
            return {"strategy": strat, "meeting": m, "seed": seed_i, "skipped": True,
                    "invalid": False, "reason": str(e)[:80]}
        if not r["ok"]:
            return {"strategy": strat, "meeting": m, "seed": seed_i, "skipped": True,
                    "invalid": False, "reason": f"api: {r['error']}"}
        problem = validate_recommendation(r["result"])
        if problem:
            # An incoherent answer is a FAILED response, not a wrong one. Scoring "hold,
            # 25bp" as an incorrect prediction credited the model with an attempt it did
            # not coherently make, and quietly penalised whichever strategy produced them.
            return {"strategy": strat, "meeting": m, "seed": seed_i, "skipped": True,
                    "invalid": True, "reason": f"incoherent: {problem[:60]}"}
        got = str(r["result"].get("recommendation", "")).lower()
        act = actual_decision(m)
        return {"strategy": strat, "meeting": m, "seed": seed_i, "skipped": False,
                "invalid": False,
                "recommended": got, "actual": act["word"].lower(),
                "correct": got == act["word"].lower(),
                "confidence": r["result"].get("confidence"),
                "size_bp": r["result"].get("size_bp"), "actual_size_bp": act["size_bp"],
                "actual_is_move": act["decision"] != 0,
                "hold_share_of_shots": r["shot_mix"]["hold_share"]}

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 60 == 0:
                print(f"    {i}/{len(jobs)}")

    df = pd.DataFrame(rows)
    if "invalid" not in df.columns:
        df["invalid"] = False
    df["invalid"] = df["invalid"].fillna(False)
    ok = df[~df["skipped"]]
    n_fail = int((df["skipped"] & ~df["invalid"]).sum())
    n_bad = int(df["invalid"].sum())
    print(f"  GenAI reliability: {n_fail}/{len(df)} calls failed after retries, "
          f"{n_bad}/{len(df)} returned an incoherent direction/size pair "
          f"({(n_fail + n_bad) / max(1, len(df)):.1%} unusable)")
    if not len(ok):
        return df

    # PAIRED COMPLETENESS. A strategy/meeting pair missing any seed is dropped for EVERY
    # strategy, so the comparison is over identical work. Comparing a strategy with three
    # seeds against one with two is comparing different estimators.
    counts = ok.groupby(["strategy", "meeting"]).size().unstack(fill_value=0)
    complete = [m for m in counts.columns if (counts[m] == n_seeds).all()]
    dropped_incomplete = [m for m in counts.columns if m not in complete]
    if dropped_incomplete:
        print(f"  {len(dropped_incomplete)} meeting(s) dropped for incomplete seed coverage "
              f"across strategies: {dropped_incomplete[:4]}")
    ok = ok[ok["meeting"].isin(complete)]
    if not len(ok):
        print("  no meeting has complete coverage across all strategies")
        return df

    def _vote(g):
        return pd.Series({
            "recommended": g["recommended"].mode().iloc[0],
            "actual": g["actual"].iloc[0],
            "actual_is_move": bool(g["actual_is_move"].iloc[0]),
            "actual_size_bp": g["actual_size_bp"].iloc[0],
            "size_bp": g["size_bp"].mode().iloc[0],
            "unanimous": g["recommended"].nunique() == 1})

    voted = (ok.groupby(["strategy", "meeting"])
               .apply(_vote, include_groups=False).reset_index())
    voted["correct"] = voted["recommended"] == voted["actual"]

    def _summary(g):
        moves = g[g["actual_is_move"] & g["correct"]]
        return pd.Series({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "balanced_accuracy": balanced_accuracy_score(g["actual"], g["recommended"]),
            "share_predicted_hold": (g["recommended"] == "hold").mean(),
            "seed_unanimous": g["unanimous"].mean(),
            "n_moves_called": len(moves),
            "size_exact_given_move": ((moves["size_bp"] == moves["actual_size_bp"]).mean()
                                      if len(moves) else float("nan"))})

    summary = voted.groupby("strategy").apply(_summary, include_groups=False).round(3)
    print(f"  sample: {sample} | {len(meetings)} paired meetings x {n_seeds} seeds, k={k}")
    print(summary.to_string())
    base = (voted[voted["strategy"] == strategies[0]]["actual"] == "hold").mean()
    print(f"  'always hold' would score {base:.3f} here")
    n = len(meetings)
    half = 1.96 * (0.25 / n) ** 0.5
    print(f"  a 95% interval on any single accuracy is about +/-{half:.3f} on {n} meetings")
    print(f"  differences smaller than that are sampling noise, not a ranking")
    return df


# -------------------------------------------------------------------------------------------
# The statement, and two independent checks on it
# -------------------------------------------------------------------------------------------

def evidence_block(meeting: str, rec: dict) -> dict:
    """
    The evidence the statement is allowed to rest on, built from the PANEL.

    Deliberately independent of the model's own reasoning. Auditing a statement against the
    reasoning that produced it asks whether the model contradicted itself, which it rarely
    does. Auditing it against the data asks whether the claims are true, which is the
    question. `rec` supplies only the decision and its size.
    """
    row = load_as_at(meeting).loc[pd.Timestamp(meeting)]
    facts = {label: (fmt % row[col])
             for col, label, fmt in nshot.SHOT_FIELDS + nshot.CONSTRUCT_SHOT_FIELDS
             if col in row.index and pd.notna(row[col])}
    return {"meeting_date": meeting,
            "cash_rate_before_decision_pct": round(rate_before(meeting), 2),
            "decision_recommended": rec.get("recommendation"),
            "size_bp": rec.get("size_bp"),
            "observed_conditions": facts}


CLAIM_CATEGORIES = ["data-supported", "model-derived", "forecast-or-judgement",
                    "contradicted", "unverifiable"]


class Claim(BaseModel):
    claim: str
    category: Literal["data-supported", "model-derived", "forecast-or-judgement",
                      "contradicted", "unverifiable"]
    evidence_key: str | None = None
    note: str = ""


class ClaimAudit(BaseModel):
    claims: list[Claim]


AUDIT_PROMPT = """
You are auditing a draft central bank statement against the evidence used to write it.

Classify EVERY factual or forward-looking claim into exactly one category:

  data-supported        traceable to a specific item in the evidence block. Give the key.
  model-derived         follows from the recommendation and its stated reasoning rather
                        than from an observation - for example a statement about what the
                        decision is intended to achieve.
  forecast-or-judgement a claim about the future or an assessment. NOT a defect: a central
                        bank statement is largely forward-looking. Flag it as such so a
                        human can judge whether it is reasonable.
  contradicted          the evidence says otherwise. This is the serious one.
  unverifiable          too vague to check, or about something the evidence does not cover
                        at all.

A blanket "unsupported" category was tried and was useless: an ordinary RBA statement is
mostly forecasts, so everything came back unsupported and the audit carried no information.
The distinction that matters is between a claim the evidence CONTRADICTS and one it merely
does not contain.

Do not check arithmetic - that is checked separately and deterministically.
"""





def claim_id(claim: str) -> str:
    """First 8 hex chars of SHA-256 over the normalised claim text - stable across runs."""
    normalised = " ".join(str(claim).split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:8]


def audit_statement(statement: str, evidence: dict,
                    reviews: dict[str, dict] | None = None,
                    require: bool = True) -> dict:
    """
    Structured claim audit against PANEL-DERIVED evidence. Returns counts plus the records.

    The evidence block is built from the data, not from the model's own reasoning: auditing
    a statement against the reasoning that produced it asks whether the model contradicted
    itself, which it rarely does.
    """
    rec = _cached_call(
        "audit", AUDIT_PROMPT,
        f"EVIDENCE BLOCK:\n{json.dumps(evidence, indent=1, default=str)}\n\n"
        f"STATEMENT:\n{statement}", schema=ClaimAudit)
    if not rec.get("ok"):
        return {"ok": False, "error": rec.get("error"), "claims": [], "counts": {}}
    claims = rec["payload"]["claims"]
    reviews = CLAIM_REVIEWS if reviews is None else reviews
    for c in claims:
        c["claim_id"] = claim_id(c["claim"])
    ids = {c["claim_id"] for c in claims}

    # STRICT, BY ID. The old matcher accepted any truthy record whose 60-character key was
    # a substring of a claim, so {"junk": True} could blanket two claims, count twice, and
    # leave both with no verdict, no reason and no reviewer. Every review must now name an
    # existing claim id and carry all four fields with a legal verdict.
    unknown = sorted(set(reviews) - ids)
    if unknown:
        raise ValueError(
            f"CLAIM_REVIEWS names claim id(s) {unknown} that are not in this audit. Ids "
            f"are printed with each claim and stored in replay.json; re-run the audit and "
            f"key your reviews on what it actually produced.")
    for cid, r in reviews.items():
        gaps = [f for f in ("human_verdict", "reason", "final_action", "by")
                if not str((r or {}).get(f, "") if isinstance(r, dict) else "").strip()]
        if gaps:
            raise ValueError(
                f"the claim review for id {cid} is missing {gaps}. Every review needs "
                f"your verdict, a reason, what you did about it, and initials.")
        if r["human_verdict"] not in CLAIM_CATEGORIES:
            raise ValueError(
                f"claim review {cid}: human_verdict {r['human_verdict']!r} must be one "
                f"of {sorted(CLAIM_CATEGORIES)}.")

    n_reviewed = n_challenged = 0
    for c in claims:
        match = reviews.get(c["claim_id"])
        if match:
            n_reviewed += 1
            c["human_verdict"] = match["human_verdict"]
            c["human_reason"] = match["reason"]
            c["final_action"] = match["final_action"]
            c["reviewed_by"] = match["by"]
            c["challenged"] = bool(match["human_verdict"] != c["category"])
            n_challenged += int(c["challenged"])
        else:
            c["human_verdict"] = None
            c["challenged"] = None
    counts = {c: sum(1 for x in claims if x["category"] == c) for c in CLAIM_CATEGORIES}
    if n_reviewed < MIN_CLAIM_REVIEWS and require:
        listing = chr(10).join(f"      {c['claim_id']}  [{c['category']:22s}] "
                               f"{c['claim'][:70]}" for c in claims)
        raise ValueError(
            f"only {n_reviewed} of the required {MIN_CLAIM_REVIEWS} distinct claims carry "
            f"a human review. Review the highest-risk classifications - typically "
            f"anything data-supported that rests on a number, and anything unverifiable "
            f"a person could in fact verify. The claims and their ids:" + chr(10)
            + listing)
    if n_challenged == 0:
        print(f"    no classification challenged: all {n_reviewed} reviewed claims agree "
              f"with the machine. A defended agreement is a legitimate review - say in "
              f"the report what each was checked against and what would have changed "
              f"your mind.")
    else:
        print(f"    {n_challenged} of {n_reviewed} reviewed classifications challenged by "
              f"the team")
    return {"ok": True, "claims": claims, "counts": counts,
            "n_claims": len(claims),
            "n_reviewed_by_human": n_reviewed,
            "n_challenged_by_human": n_challenged,
            "reviews": reviews,
            "n_contradicted": counts.get("contradicted", 0)}


def feasible_everywhere(meetings: list[str], strategies: list[str],
                        k: int) -> tuple[list[str], dict]:
    """
    The meetings on which EVERY strategy can be built, plus what was dropped.

    Comparing strategies across different meeting sets compares different questions: the
    strategies with the strictest selection rules skip the hardest early meetings and score
    higher for it. Accuracies must be paired.
    """
    panel = _panel()
    keep, dropped = [], {}
    for m in meetings:
        bad = []
        for s in strategies:
            try:
                nshot.select_shots(panel, pd.Timestamp(m), k=k, strategy=s)
            except Exception:  # noqa: BLE001
                bad.append(s)
        if bad:
            dropped[m] = bad
        else:
            keep.append(m)
    return keep, dropped


def generate_statement(rec: dict, evidence: dict) -> str:
    """
    Produce the public statement from your `STATEMENT_PROMPT`. Supplied, cached, retried.

    The README says the machinery is built, so this is built. What is yours is the PROMPT and
    the orchestration in `run()` - which of the pieces below you call, in what order, and what
    you conclude from them.
    """
    out = _cached_call(
        "stmt", STATEMENT_PROMPT,
        f"DECISION AND REASONING:{chr(10)}{json.dumps(rec, indent=1, default=str)}"
        f"{chr(10)}{chr(10)}EVIDENCE BLOCK - you may assert nothing beyond this:{chr(10)}"
        f"{json.dumps(evidence, indent=1, default=str)}")
    return out.get("payload", {}).get("text", "")


def arithmetic_check(statement: str, meeting: str, rec: dict) -> dict:
    """
    Deterministic. Does the statement quote the correct post-decision cash rate?

    Asking the auditing model to check arithmetic asks a language model to do the one thing
    it is worst at, and in testing it passed a statement asserting that a 25 basis point
    increase took 4.50 per cent to 4.50 per cent. This computes the expected rate and looks
    for it in the text.
    """
    before = rate_before(meeting)
    sign = {"hike": 1, "cut": -1, "hold": 0}.get(str(rec.get("recommendation", "")).lower())
    if sign is None:
        return {"checked": False, "reason": "no parseable recommendation"}
    after = round(before + sign * (rec.get("size_bp", 0) or 0) / 100.0, 2)
    # Only rates stated in CASH RATE context. An earlier version matched every percentage
    # in the statement, so inflation and unemployment figures landed in `rates_quoted` and
    # a hold passed on any 4.5 anywhere in the text.
    # "per cent", "percent" and "%" are the same unit. Matching only "per cent|%" gave a
    # FALSE FAIL on a statement that opened "the cash rate at 4.50 percent" - the check
    # reported that the post-decision rate was never stated when it was in the first
    # sentence. A deterministic check that is wrong is worse than no check, because the
    # whole point of it is that its verdict is not a matter of opinion.
    num = r"(\d+(?:\.\d+)?)\s*(?:per\s*cent|%)"
    ctx = r"cash rate[^.]{0,120}?" + num + r"|" + num + r"[^.]{0,60}?cash rate"
    hits = [g for m in re.finditer(ctx, statement, re.I) for g in m.groups() if g]
    quoted = sorted({float(h) for h in hits})
    ok = any(abs(q - after) < 0.005 for q in quoted)
    stale = [q for q in quoted if abs(q - before) < 0.005] if sign != 0 else []
    return {"checked": True, "rate_before": before, "expected_after": after,
            "rates_quoted": quoted, "correct_rate_stated": ok,
            "stale_rate_stated": bool(stale),
            "verdict": ("pass" if ok and not stale else
                        "FAIL - quotes the pre-decision rate as the new rate" if stale and not ok
                        else "pass with a stale rate also mentioned" if ok and stale
                        else "FAIL - the post-decision rate is never stated")}


# ###########################################################################################
# YOUR ORCHESTRATION BELOW THIS LINE - everything above is supplied framework
# ###########################################################################################

def run() -> dict:
    """
    YOURS TO ASSEMBLE. Every piece it needs is supplied; the ORDER and the CONCLUSIONS are
    the assessed part.

    A working run() calls, roughly in this order:

        feasible_strategies(MEETING)     which strategies can be built here at all
        evaluate_all(sample="dev")       choose k and the strategy
        evaluate_all(sample="holdout")   report it ONCE, as it falls
        recommend(MEETING)               with and without CAUSAL_GUIDANCE, for the A/B
        validate_recommendation(...)     reject an incoherent direction/size pair
        evidence_block(MEETING, rec)     panel-derived evidence, not the model's reasoning
        generate_statement(rec, ev)      your STATEMENT_PROMPT
        audit_statement(stmt, ev)        structured claim categories
        arithmetic_check(stmt, ...)      deterministic, not delegated to an LLM
        actual_decision(MEETING)         the comparison - LAST, never before you commit

    Write outputs/replay.json and outputs/replay_statement.md. See the brief, Replay stage.
    """
    if not _prompts_written():
        raise NotImplementedError(
            "Set MEETING and write RECOMMENDATION_PROMPT and STATEMENT_PROMPT first. "
            "See the brief, Replay stage.")
    raise NotImplementedError(
        "Assemble run() from the sequence in this docstring. Every helper it names is "
        "supplied and tested; the orchestration and the conclusions are what is assessed.")


if __name__ == "__main__":
    run()

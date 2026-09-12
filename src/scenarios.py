"""
SHOCK - geopolitical risk scenarios by tree-of-thought prompting.

YOU SUPPLY THE FOUR PROMPTS AND THE PRUNING THRESHOLD. EVERYTHING ELSE IS BUILT.

    branch    generate candidate transmission channels from the context documents
    evaluate  score each candidate for economic credibility
    prune     keep the survivors - and record why you dropped the rest
    expand    second-order consequences of what survived

BEFORE YOU RUN THIS
    1. Download the context documents into data/raw/context/. The course cannot
       redistribute them. See data/raw/context/README.md.
    2. Check them: `python src/context_docs.py`
    3. Write the four prompts below.

-------------------------------------------------------------------------------------------
WHY TREE OF THOUGHT AND NOT ONE PROMPT
-------------------------------------------------------------------------------------------
Asked "what does this shock mean for the RBA?", an LLM produces one fluent answer down a
single line of reasoning, and it will sound complete. The channels it happens not to mention
are invisible to you - you cannot audit an absence.

Tree of thought forces the alternatives into the open: generate several branches
independently, score each, prune explicitly, then expand only the survivors. What you gain
is not a better answer. It is a RECORD of the answers you rejected and why - which is what
you are marked on, and what a risk committee would actually ask for.

Every scenario in this stage has channels that push RBA policy in OPPOSITE directions. A
single-pass prompt typically surfaces one direction and stops.

WRITES outputs/shock.json, outputs/tot_trees.json
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
from functools import lru_cache
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field as PField

sys.path.insert(0, os.path.dirname(__file__))
import channels        # noqa: E402
import config          # noqa: E402
import courseapi  # noqa: E402
import context_docs    # noqa: E402
from scenario_engine import (HORIZONS, Branch, Tree,  # noqa: E402
                             apply_adjudications, opposing_direction_guard,
                             adjudicate_shocks, apply_shock, channels_at,
                             support_check,
                             cap_to_calibration,
                             compare_reaction,
                             episode_profiles, focused_corners, frozen_shock_model,
                             cascade_orphans, opposite_of, prune_sweep,
                             resolve_proxy_conflicts,
                             run_scenario, uncertainty_bounds, validate_direction_weights,
                             DIRECTION_MARGIN)

load_dotenv()
_client = None


def client_():
    """The course API client (see courseapi): same shape, UNSW proxy underneath."""
    return courseapi.client_()


# The WORKED EXAMPLE. Not assessed - it exists so you can see one complete
# branch -> evaluate -> prune cycle before committing to your own prompts.
#
# READ docs/worked-example-tree.md FIRST. It is a static record of this scenario's tree,
# costs nothing to read, and needs no prompts of your own. `--worked` runs the same
# scenario live through YOUR branch prompt, so it only works once Step 3 is done.
WORKED_SCENARIO = (
    "Sharp cut to net overseas migration",
    "The Australian government cuts the permanent migration cap and tightens student visa "
    "rules, and net overseas migration falls by roughly 60% over eighteen months. The "
    "population growth rate halves. The policy is announced with immediate effect and is "
    "expected to persist for at least three years.")

WORKED_RETRIEVAL_TERMS = ["migration", "population", "labour supply", "labour market",
                          "housing", "rents", "wages", "demographic", "skills shortage"]

# The two assessed scenarios. Text is in draft-assignment/geopolitical-scenarios.md.
SCENARIOS = {
    "A. Taiwan Strait blockade":
        "A sustained maritime quarantine of Taiwan begins. It is not a shooting war. "
        "Shipping insurance in the East China Sea becomes unobtainable, semiconductor "
        "exports from Taiwan halt, and China's industrial output contracts sharply over "
        "the following two quarters. Australia's iron ore and LNG shipments to China fall "
        "by roughly a third.",
    "B. Strait of Hormuz closure":
        "A regional conflict closes the Strait of Hormuz to commercial traffic for an "
        "extended period. Brent crude trades above USD 200. European and Asian gas prices "
        "spike. Global equity markets fall sharply and credit spreads widen.",
}

# Terms used to pull relevant passages out of your downloaded documents, per scenario.
# These are a starting point. Tune them, and say in your report what you changed and why -
# retrieval that misses the relevant sections produces channels grounded in nothing.
RETRIEVAL_TERMS = {
    "A. Taiwan Strait blockade": ["Taiwan", "semiconductor", "China", "supply chain",
                                  "shipping", "trade disruption"],
    "B. Strait of Hormuz closure": ["Hormuz", "oil price", "energy", "Middle East",
                                    "LNG", "commodity price"],
}

# THE STIPULATED FACTS, one registry per scenario. Each value is quoted VERBATIM from the
# narrative above, and `verify_scenario_facts()` raises if it is not - so a "stipulated
# fact" can never be something the scenario does not actually say.
#
# WHY IDS AND NOT A FREE-FORM FLAG. An earlier version let a channel review write
# `global_verdict: "scenario_fact"` with nothing attached, and the code accepted it
# unconditionally. That turned the exception into the main bypass: an invented claim that
# Australian wages accelerate could ride through grounding on the same verdict as a fact the
# narrative really states. A review must now name WHICH fact carries its global leg, the
# fact must exist here, and the registry itself is checked against the narrative text.
SCENARIO_FACTS: dict[str, dict[str, str]] = {
    "Sharp cut to net overseas migration": {
        "migration_cut": "net overseas migration falls by roughly 60% over eighteen months",
        "population": "The population growth rate halves",
        "persistence": "expected to persist for at least three years",
    },
    "A. Taiwan Strait blockade": {
        "insurance": "Shipping insurance in the East China Sea becomes unobtainable",
        "semiconductors": "semiconductor exports from Taiwan halt",
        "china_contraction": "China's industrial output contracts sharply over the "
                             "following two quarters",
        "shipments_fall": "Australia's iron ore and LNG shipments to China fall by "
                          "roughly a third",
    },
    "B. Strait of Hormuz closure": {
        "closure": "closes the Strait of Hormuz to commercial traffic for an extended "
                   "period",
        "oil_price": "Brent crude trades above USD 200",
        "gas_price": "European and Asian gas prices spike",
        "equities_fall": "Global equity markets fall sharply",
        "spreads_widen": "credit spreads widen",
    },
}


# ###########################################################################################
# YOUR WORK STARTS HERE - THE FOUR PROMPTS
# ###########################################################################################
#
# YOU DO NOT HAND-WRITE THE RESPONSE SHAPE. The harness appends it to your prompt at call
# time, generated from the Pydantic schema, so it cannot drift out of date. See it with:
#
#     python -c "import scenarios; scenarios.print_contracts()"
#
# An earlier version of this file printed a hand-written example that had fallen four fields
# behind the schema, while the brief told students the displayed shape was the shape to
# request. Write the INSTRUCTIONS; the contract is supplied.

BRANCH_PROMPT = """
TODO: WRITE THE BRANCH PROMPT.

The model receives a scenario description and passages from the context documents, and must
propose candidate transmission channels from the shock to Australian monetary policy.

The response shape is appended for you from `BranchOut` - the exact class the API call passes as `response_format` - do not restate it. Run
`scenarios.print_contracts()` to see exactly what the model will be asked for, including the
four fields teams most often forget to write instructions for: `source`, `source_quote`,
`proxy_movement` and `first_round_effect`.

YOUR job is to tell it HOW to fill those fields well.

The harness automatically appends the channel taxonomy, the shock-size calibration table
and the list of panel columns to your prompt, so you do not need to paste them - but you
DO need to tell the model to use them.

Things worth deciding explicitly:
  - how you stop it proposing only channels that point one way
  - how you make it ground channels in the supplied passages rather than general knowledge
  - how you make it tag every channel with a `channel_type` from the taxonomy
  - how you make it admit when a channel has NO measurable proxy, instead of inventing one

Rules the harness enforces, so your prompt had better ask for them:
  - `source` must quote a "[file p.N]" tag from the supplied passages, or the literal
    "not in sources". Citations to documents you did not download are counted as ungrounded.
  - the pathway fields must agree with each other: proxy_movement -> first_round_effect ->
    policy_implication -> direction, and the sign of n_sd must match proxy_movement.
  - `proxy` must be an exact panel column, or null. NEVER the cash rate, the decision, or a
    trailing measure of past rate changes - those ARE the policy stance, and shocking them
    assumes the conclusion. `gpr_index` is usually the most direct first-order proxy for a
    geopolitical shock.
  - channels in BOTH directions, and several distinct channel_types.
  - shock sizes sanity-checked against the calibration table.
"""

EVALUATE_PROMPT = """
TODO: WRITE THE EVALUATE PROMPT.

The model receives the pooled candidate channels and must score each for economic
credibility on 0-1.

Do NOT hand-write the JSON shape into your prompt. Append the generated contract
instead - `scenarios.print_contracts()` prints exactly what each call is told to
return, and the code appends it for you. A hand-written example drifts from the
schema and the API silently wins.

Think about what "credible" means here. A channel can be real but negligible; real but
operating over a horizon the RBA would look through; or real, large and fast. Decide which
of those you want scored highly and say so.
"""

EXPAND_PROMPT = """
TODO: WRITE THE EXPAND PROMPT.

The model receives ONE surviving first-order channel and must propose its second-order
consequences. Same JSON shape as the branch prompt.

This is where the opposing directions usually appear - a first-order tightening channel
often has a second-order easing consequence through the exchange rate or funding costs.
Write the prompt so it looks for those rather than reinforcing the first-order direction.
"""

ADVERSARIAL_PROMPT = """
TODO: WRITE THE ADVERSARIAL PROMPT.

The Shock stage requires you to have the model argue AGAINST your conclusion for each
scenario, and then to answer its strongest point in `ADVERSARIAL_RESPONSES`.

It receives your conclusion and your surviving channels. The response shape is appended from
`AdversarialOut`; write the instructions, not the schema.
"""


# Branches scoring below this are pruned. Your choice - justify it in the report.
PRUNE_THRESHOLD = 0.5
# Independent branch samples per call. More samples explore more starting points and cost
# more. Below 3 you are not really sampling.
N_BRANCH_SAMPLES = 3

# YOUR PRUNING ADJUDICATIONS. {scenario: {channel name: keep?}}
# The credibility score sets the default; this is where you overrule it. A channel you saved
# or killed against the score, with a reason, is worth more marks than one you left to the
# threshold - it is the judgement the stage is actually assessing.
# {scenario: {channel name: {"keep": bool, "reason": "...", "by": "initials"}}}
ADJUDICATIONS: dict[str, dict[str, dict]] = {}

# YOUR DECISIONS ON FLAGGED BRANCHES. {channel name: reason}
#
# Two things land here, and the run BLOCKS until they do:
#   - a channel whose stated pathway contradicts itself (proxy movement -> first-round
#     effect -> policy implication -> direction). Correct it, set its proxy to None to make
#     it unmodellable, or record here why the flag is wrong.
#   - a channel the opposing-direction guard reinstated on your behalf after you pruned.
#     Confirm it or drop it; it is a decision the code made for you.
PATHWAY_ADJUDICATIONS: dict[str, str] = {}

# What to do with a channel whose stated pathway contradicts itself.
#   None            block, and adjudicate each one by name in PATHWAY_ADJUDICATIONS
#   "unmodellable"  keep its reasoning, drop it from the shock (proxy set to None)
# Either is a decision you must state in the report. The default blocks, because doing
# nothing is not one of the options.
PATHWAY_POLICY: str | None = None



# -------------------------------------------------------------------------------------------
# YOUR JUDGEMENTS. The run BLOCKS until these are supplied.
# -------------------------------------------------------------------------------------------

# Which horizon this scenario is being assessed at. Branches carry a horizon and only those
# matching are applied, because a day-one volatility spike and a 2-4-quarter GDP contraction
# are not simultaneous inputs. The other horizons are still reported, for contrast.
#   immediate = days + weeks | short = 1-2q | medium = 2-4q + years
SHOCK_HORIZON = "short"

# WHICH STARTING STATE THE SHOCK IS APPLIED TO. {label: meeting date}
#
# A stress test starts from a stated position, and the position matters more than teams
# expect. The model's unshocked probability at the LAST observation is 0.93 on easing,
# which leaves 0.07 of probability for any shock to move - so a small shift there means
# "there was nowhere to go", not "the shock does not matter".
#
# The default pair is deliberate: one meeting where the model is genuinely undecided, and
# the latest observation. Reporting both is the answer to "what does this shock do" -
# it does something from a neutral position, and almost nothing from one where policy is
# already easing hard. Both are true and the pair is more informative than either alone.
BASE_ROWS: dict[str, str] = {
    "neutral": "2025-11-04",   # model at 0.667 stable - undecided, and in support
    "latest": "2026-06-16",    # the last meeting - already 0.93 easing
}
HEADLINE_BASE = "neutral"

# STRICT makes `Tree.validate()` blocking on the reportable path: a tree with a blocking
# defect raises instead of returning a headline. Leave it True; strict=False is for
# inspecting a broken tree, and nothing produced that way is reportable.
STRICT = True

# How far beyond the largest move on record a shock may go. 1.0 caps at the historical
# peak for that variable; raise it only for a scenario you can argue is genuinely worse
# than anything in the sample, and say so in the report.
CALIBRATION_ALLOWANCE = 1.0

# {scenario: {proxy: {"central": sd, "low": sd, "high": sd, "reason": "..."}}}
#
# One structure, three jobs. `central` resolves sign conflicts AND sets the magnitude;
# `low`/`high` are the uncertainty bounds `focused_corners()` varies. Give low/high for the
# three or four proxies you are genuinely least sure of - not the largest ones - and omit
# them elsewhere. A net effect of zero is expressible: set central to 0.0.
#
# The reason is marked. The arithmetic is not.
PROXY_JUDGEMENTS: dict[str, dict[str, dict]] = {}

# Scenario-specific co-movement, used to exclude incoherent sensitivity corners.
# [{"proxies": ["a", "b"], "reason": "..."}] - two proxies that must be sized in the same
# direction FOR THIS SCENARIO. There is no global list, because whether the AUD and the terms
# of trade move together depends on whether the shock is a commodity shock or a risk-off one.
COHERENCE: dict[str, list[dict]] = {}

# YOUR ANSWER TO THE ADVERSARIAL CRITIQUE. {scenario: {...}}
#
# The critique is generated for you; ANSWERING it is the assessed part, and an unstructured
# paragraph in the report is not auditable. Record, per scenario:
#   "strongest_point"      the critique's best argument, in your words
#   "our_response"         why it does or does not change the conclusion
#   "confidence_before"    "low" | "moderate" | "high"
#   "confidence_after"     the same scale, AFTER considering the critique
#   "what_we_changed"      a channel, a judgement, a bound - or "nothing, and here is why"
ADVERSARIAL_RESPONSES: dict[str, dict] = {}

# YOUR ASSESSMENT OF EACH MECHANISM GROUP. {scenario: {group key: {...}}}
#
# ---------------------------------------------------------------------------------------
# WHAT THIS IS. A channel asserts TWO things, and they are evidenced differently:
#
#   LEG 1, GLOBAL      scenario -> world effect.  Settled by a retrieved passage a person
#                      has read, or by a REGISTERED scenario fact.
#   LEG 2, AUSTRALIAN  world effect -> proxy -> RBA.  Only you can settle it.
#
# A channel drives a number only if BOTH legs hold. The machine checks half of leg 1 (is
# the page tag real, is the quote actually on that page, is the fact verbatim in the
# narrative); you supply the rest. See `global_leg()` and `australian_leg()` below the
# SUPPLIED line for the full vocabulary.
#
# THE VERDICT BINDS ONE BRANCH. A group is every restatement of one mechanism, and its
# members cite different passages. Your review names the CANONICAL member - the one whose
# evidence you actually read - and only that branch can move a number. The rest are barred
# as restatements automatically. One signature never vouches for several claims.
#
# ---------------------------------------------------------------------------------------
# THE WORKFLOW. Four commands, in this order.
#
#   1. DISCOVER:  python src/scenarios.py --discover
#      Runs BOTH assessed scenarios with reviews switched off, stops before any shock,
#      probability, sweep or adversarial output, prints every group key, and writes
#      outputs/channel_reviews.template.json - a skeleton with every member branch's
#      citation and quotation inline, ready to fill in.
#      (python src/scenarios.py --worked demonstrates the same workflow on the unassessed
#      migration scenario; it cannot produce the assessed keys.)
#
#   2. INSPECT each group in the template. Open the cited page and check that the quoted
#      words support the claim - a verbatim quote from a country risk-ranking table is
#      still not evidence for a terms-of-trade channel.
#
#   3. REVIEW. Fill the template IN PLACE - one entry per group key - and copy it to
#      data/processed/channel_reviews.json; put the pruning audit in
#      data/processed/pruning_reviews.json. Groups with a proxy need the full field set;
#      groups with no proxy need the five-field minimum. (Pasting the records into the
#      Python dicts below is the supported alternative; the dicts win when non-empty.)
#
#   4. RERUN: python src/scenarios.py
#      Anything you reject, every restatement, and anything whose legs do not both hold is
#      out of every number in the report - and stays out at every sweep threshold.
#
# ---------------------------------------------------------------------------------------
# THE GROUP KEY is exactly this, with single spaces around each pipe:
#
#       "<channel_type> | <direction> | <horizon> | <proxy or 'unmodellable'>"
#
# THE FIELDS. The four verdict fields must take a listed value.
#
#   "canonical"           the member branch this verdict is about. Required when the group
#                         has more than one member; the single member otherwise. With ONE
#                         record, naming it ATTESTS every other member restates it. A
#                         member that is a genuinely DISTINCT claim gets its own record:
#                         make the group's value a LIST of records - and then each record
#                         must also carry "restatements": [members that restate ITS
#                         canonical], so the partition is yours, not inferred
#   "restatements"        required when a group has SEVERAL records: the members restating
#                         this record's canonical. Together with the canonicals these must
#                         cover every member exactly once
#   "confidence"          "low" | "moderate" | "high"          (groups with a proxy)
#   "global_verdict"      "supports"         the canonical branch's VERIFIED quotation
#                                            genuinely supports the global claim
#                         "scenario_fact"    the narrative stipulates the world effect.
#                                            Requires "fact_id" (a registered id, verified
#                                            verbatim against the narrative - so the FACT
#                                            is always real) and "fact_link_reason" (your
#                                            signed sentence on why that fact supports
#                                            THIS claim - the machine cannot judge that)
#                         "does_not_support" the words are there; they do not entail it
#                         "uncertain"        you could not tell
#   "fact_id"             required with scenario_fact; e.g. "oil_price"
#   "fact_link_reason"    required with scenario_fact: why THIS fact supports THIS
#                         canonical claim. The registry check is identity, not entailment
#                         - the entailment is this sentence, signed by "by"
#   "australian_verdict"  "supports" | "does_not_support" | "uncertain"
#   "evidence_cycle"      what the Cycle causal tests say about that proxy   (proxy groups)
#   "evidence_words"      what the construct dimensions say about it         (proxy groups)
#   "evidence_replay"     what the reconstruction showed about it            (proxy groups)
#   "decision"            "accept" | "modify" | "reject"
#   "reason"              why, in your words
#   "by"                  initials
#
# Anything other than supports/scenario_fact on the global leg, or anything other than
# supports on the Australian leg, bars the canonical branch. There is no override: an
# adjudication is a judgement about credibility, and credibility does not supply evidence.
#
# ---------------------------------------------------------------------------------------
# THE SHAPE OF ONE RECORD, on the UNASSESSED migration scenario.
#
# It is deliberately written for the scenario you are NOT marked on, and the evidence
# fields are left as descriptions of what belongs there rather than findings. Both
# assessed scenarios are yours to reach: a worked record for one of them would hand you
# the canonical claim, the verdict and the cross-stage evidence that the marks are for.
#
# The KEY is emitted by the discovery step (`--discover`) and must be copied exactly.
#
# CHANNEL_REVIEWS = {
#     "Sharp cut to net overseas migration": {
#         "labour_supply | <direction> | <horizon> | <proxy>": {
#             "canonical": "<the canonical claim text, as the tree emitted it>",
#             "confidence": "high" | "moderate" | "low",
#             "global_verdict": "supports" | "does_not_support" | "scenario_fact"
#                               | "uncertain",
#             "fact_id": "<required only with scenario_fact; a key from SCENARIO_FACTS>",
#             "fact_link_reason": "<required with scenario_fact: why THAT stipulated fact "
#                                 "carries THIS claim - one sentence, signed>",
#             "australian_verdict": "supports" | "does_not_support" | "uncertain",
#             "evidence_cycle": "<what YOUR causal tests found about this proxy, with the "
#                               "verdict and the number you are resting on>",
#             "evidence_words": "<what YOUR construct scores show, with the figures>",
#             "evidence_replay": "<what YOUR reconstruction showed about how the Board "
#                                "weighs this>",
#             "decision": "accept" | "modify" | "reject",
#             "reason": "<why, in your words - this sentence is the judgement being "
#                       "assessed>",
#             "by": "<initials>"},
#     },
# }
CHANNEL_REVIEWS: dict[str, dict[str, dict]] = {}

# Set False only while you are still generating trees and do not yet know the group keys -
# step 1 of the workflow above. It must be True for a reportable run.
REQUIRE_CHANNEL_REVIEWS = True

# YOUR AUDIT OF WHAT THE MODEL THREW AWAY. {scenario: {channel name: {...}}}
#
# The same model that generates the branches scores their credibility, so left alone it
# prunes its own output and nobody looks at what it discarded - which is exactly where a
# valid, inconvenient or opposing mechanism disappears. `required_pruning_reviews()`
# computes a small audit set per scenario:
#
#   - the three highest-scoring branches the threshold rejected,
#   - at least one rejected branch in each direction (tightening and easing),
#   - for each of the nine taxonomy families with NO surviving branch, its best-scoring
#     rejected branch, if it has one.
#
# `--discover` lists the audit set in the template. For each, record:
#
#   "verdict"   "agree"      the model was right to drop it - and you say why
#               "reinstate"  it was wrong; you must ALSO add an ADJUDICATIONS entry with
#                            keep=True so the reinstatement is a recorded credibility
#                            decision, not a silent edit
#   "reason"    your words. A defended agreement is full credit; a bare tick is not a review
#   "by"        initials
PRUNING_REVIEWS: dict[str, dict[str, dict]] = {}
REQUIRE_PRUNING_REVIEWS = True

# THE HISTORICAL ANALOGUE YOU CLAIM, per scenario - named BEFORE you run the comparison.
# {scenario: episode name from episode_profiles()}. `compare_reaction()` checks the claim
# against the profile distance and tells you when the data is closer to a different
# episode; defending or changing the claim is then your call, in the report.
CLAIMED_ANALOGUES: dict[str, str] = {}

# YOUR EXPECTED BOARD REACTION, per scenario, on the seven Words dimensions - each value a
# percentile in [0, 1]. Written from the narratives and the construct rubrics BEFORE you
# run the comparison; `reaction_profiles()` validates that both scenarios and all seven
# constructs are present. This is where Words feeds Shock.
#
# EXPECTED_PROFILES = {
#     "A. Taiwan Strait blockade": {
#         "policy_stance": 0.30, "inflation_concern": 0.60, "downside_risk_emphasis": 0.85,
#         "financial_conditions_concern": 0.65, "uncertainty_language": 0.90,
#         "vigilance": 0.85, "global_risk_salience": 0.95},
#     ...
# }
EXPECTED_PROFILES: dict[str, dict[str, float]] = {}

# YOUR WEIGHTS FOR THE QUALITATIVE DIRECTION. {mechanism family: weight}
#
# Without these, `channel_direction()` falls back to the LLM's own credibility scores and
# says so - a tally of one model's self-assessment is not your reasoning. Supplying a weight
# and a reason per mechanism family makes the qualitative conclusion yours.
# Families are the `channel_type` values in `channels.CHANNEL_TAXONOMY`.
DIRECTION_WEIGHTS: dict[str, float] = {}
DIRECTION_WEIGHT_REASONS: dict[str, str] = {}

# ###########################################################################################
# SUPPLIED BELOW THIS LINE - DO NOT MODIFY
# ###########################################################################################


class ChannelOut(BaseModel):
    """
    One candidate channel, with the WHOLE pathway made explicit.

    The five pathway fields exist so the sign can be checked instead of trusted. A model will
    happily label a channel `tightening` while describing a mechanism that is plainly
    disinflationary, and nothing downstream notices - the n_sd goes into the shock with the
    wrong sign and the scenario's answer flips. `validate_pathway()` walks

        scenario -> channel -> proxy movement -> first-round effect -> policy implication

    and rejects a channel whose stated `direction` does not follow from its own description.
    """
    channel: str
    channel_type: str
    mechanism: str
    proxy: str | None = None
    proxy_movement: Literal["rises", "falls", "no clear movement"]
    first_round_effect: Literal[
        "demand up", "demand down",
        "supply capacity up", "supply capacity down",
        "inflation up", "inflation down",
        "financial stability worse", "financial stability better",
        "none"]
    policy_implication: Literal["tightening", "easing", "ambiguous"]
    direction: Literal["tightening", "easing", "ambiguous"]
    horizon: Literal["days", "weeks", "1-2q", "2-4q", "years"]
    n_sd: float = PField(ge=-4.0, le=4.0)
    source: str = PField(description="the [file p.N] citation this rests on, or the literal "
                                     "'not in sources'")
    source_quote: str = PField(
        default="",
        description="the exact words from that passage that support this channel, quoted "
                    "verbatim. Every fragment is checked against the passage in order; "
                    "an ellipsis is allowed, a paraphrase fails, and a real fragment "
                    "followed by an invented one fails. Leave empty only when source is "
                    "'not in sources'.")
    note: str = ""


class BranchOut(BaseModel):
    channels: list[ChannelOut]


class ScoreOut(BaseModel):
    channel_id: str = PField(description="the id given with the channel, echoed exactly")
    channel: str
    credibility: float = PField(ge=0.0, le=1.0)
    reason: str


class EvaluateOut(BaseModel):
    scores: list[ScoreOut]


class AdversarialOut(BaseModel):
    strongest_counterargument: str
    channels_you_underweighted: list[str]
    channel_overweighted: str
    historical_counterexample: str | None = None
    historical_claim_grounded_in: str = PField(
        default="unsupported",
        description="the [file p.N] tag supporting the historical counterexample, or the "
                    "literal 'unsupported' if the passages do not contain it")
    what_would_have_to_be_true: str
    can_construct_case: bool


# The first-round effects that argue for tightening, and those that argue for easing. Used to
# check a channel's stated direction against its own stated mechanism.
# What each first-round effect implies for policy, and which are genuinely two-sided.
#
# THE SUPPLY/DEMAND DISTINCTION IS THE POINT. An earlier version offered only "activity up"
# and "activity down", so a supply-side contraction - fewer workers, a blocked shipping lane
# - had to be recorded as "activity down", which the check read as disinflationary and then
# flagged every such channel as self-contradicting. It is not a contradiction: a supply shock
# lowers output AND raises prices, and which side the Board acts on is exactly the judgement
# this stage is about. Those channels are now "supply capacity down" and come back ambiguous
# rather than wrong.
IMPLIED_BY_EFFECT = {
    "demand up": "tightening",
    "demand down": "easing",
    "supply capacity up": "easing",        # disinflationary and expansionary
    "supply capacity down": None,          # contractionary AND inflationary - genuinely both
    "inflation up": "tightening",
    "inflation down": "easing",
    "financial stability worse": "easing",
    "financial stability better": "tightening",
    "none": None,
}





def validate_pathway(c: dict) -> str | None:
    """
    Does the channel's stated direction follow from its own stated first-round effect?

    Returns the inconsistency, or None. An inconsistent channel is not silently corrected -
    the sign is genuinely ambiguous in some cases (a terms-of-trade shock raises income and
    lifts the currency) and the right response is for a person to adjudicate it, which is
    what the run does.
    """
    implied = IMPLIED_BY_EFFECT.get(c.get("first_round_effect", "none"))
    stated = c.get("direction")
    if implied and stated not in (implied, "ambiguous"):
        return (f"'{str(c.get('channel'))[:40]}': first-round effect "
                f"'{c['first_round_effect']}' implies {implied}, but the channel is "
                f"labelled {stated}")
    if c.get("policy_implication") not in (None, "ambiguous") and \
            stated not in (c.get("policy_implication"), "ambiguous"):
        return (f"'{str(c.get('channel'))[:40]}': policy_implication "
                f"'{c['policy_implication']}' contradicts direction '{stated}'")
    if c.get("proxy") and c.get("n_sd"):
        mv, n = c["proxy_movement"], float(c["n_sd"])
        if (mv == "rises" and n < 0) or (mv == "falls" and n > 0):
            return (f"'{str(c.get('channel'))[:40]}': proxy '{c['proxy']}' is said to "
                    f"{mv} but n_sd is {n:+.2f}")
    return None


def _user_block(narrative: str, X, passages: str) -> str:
    """
    The standard user message: scenario, panel columns, channel taxonomy, shock calibration,
    then the retrieved passages.

    Assembled here rather than left to your prompt so every team gives the model the same
    reference material and the trees stay comparable. Used by BOTH the worked example and
    the assessed run - an earlier version built its own, thinner block in the assessed path,
    so the worked example the students were told to copy was not the thing being marked.
    """
    return (f"SCENARIO: {narrative}\n\n"
            f"AVAILABLE PANEL COLUMNS (a proxy must be one of these, exactly):\n"
            f"{list(X.columns)}\n\n"
            f"{channels.describe_taxonomy()}\n\n"
            f"{channels.describe_calibration()}\n\n"
            f"CONTEXT PASSAGES FROM THE SOURCE DOCUMENTS. Each is tagged with its source "
            f"file and page; cite that tag in `source`:\n{passages}")


def _prompts_written() -> bool:
    return not any("TODO" in p for p in
                   (BRANCH_PROMPT, EVALUATE_PROMPT, EXPAND_PROMPT, ADVERSARIAL_PROMPT))


RAW_DIR = config.DATA_PROCESSED / "shock_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# How many channels one evaluator call is asked to score. Chosen well inside the point at
# which a model stops early: asked for 36 scores in one response, gpt-4o-mini returned 12.
# The chunk size was NOT re-tuned when the course moved to gpt-5.4-mini - it is a
# conservative bound, and raising it to save calls would be a change to what is generated,
# which the Shock configuration hash covers deliberately.
EVALUATE_CHUNK = 12


class LLMCallError(RuntimeError):
    """
    An LLM call this stage NEEDS did not produce a valid response.

    Raised, never swallowed: a failed branch call is not "the model proposed no channels",
    a failed evaluator call is not a set of default credibilities, and a failed adversarial
    call is not an answered challenge. The run stops, the failure envelope is on disk, and
    re-running retries only the failed call - everything successful is already cached.
    """


# Errors that retrying cannot fix: bad credentials, malformed requests (including a schema
# the endpoint rejects), and plain programming errors. Matched by name so no extra imports
# are needed; anything else (rate limits, timeouts, connection drops, 5xx) is transient and
# retried with backoff.
_NON_TRANSIENT = ("AuthenticationError", "PermissionDeniedError", "BadRequestError",
                  "NotFoundError", "UnprocessableEntityError",
                  "TypeError", "KeyError", "AttributeError", "ValidationError"
                  ) + courseapi.NON_TRANSIENT_ERRORS


def _quarantine(path, why: str) -> None:
    """Move an unusable cache file aside and say exactly what to do about it."""
    bad = path.with_suffix(path.suffix + ".invalid")
    try:
        path.replace(bad)
    except OSError:
        bad = None
    print(f"    CACHE QUARANTINED: {path.name} - {why}."
          + (f" Moved to {bad.name}; " if bad else " ")
          + "re-run to make a fresh call (an unchanged prompt re-bills nothing else).")


def reusable_under_current_ceiling(rec: dict) -> bool:
    """
    THE OUTPUT-CEILING COMPATIBILITY RULE. Same rule as Words - see
    `text_features.reusable_under_current_ceiling` for the full reasoning.

    A cached SUCCESS is reusable only when the ceiling now in force is at least the one
    it was produced under (it finished inside that allowance, so more room cannot cut it
    short; less room might). A cached TRUNCATION stops being settled as soon as the
    ceiling RISES, which is what makes the brief's advice - raise
    config.MAX_OUTPUT_TOKENS and re-run - actually do something.
    """
    current = int(config.MAX_OUTPUT_TOKENS)
    v = (rec.get("request") or {}).get("max_output_tokens")
    recorded = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if rec.get("ok"):
        return recorded is not None and current >= recorded
    if "IncompleteResponseError" in str(rec.get("error", "")):
        return recorded is not None and current <= recorded
    return True


def llm_parsed(system: str, user: str, schema, seed_offset: int = 0) -> dict:
    """
    Schema-validated, cached and retried. Raises LLMCallError instead of failing open.

    The earlier version asked for a bare JSON object and swallowed a parse failure as an
    empty dict, which surfaced as "the model proposed no channels" - indistinguishable from
    the model genuinely proposing none.

    Cached payloads are RE-validated against the current schema on load - a cache written
    under an older schema, or corrupted on disk, is quarantined rather than trusted just
    because it once said "ok". Failures are persisted as envelopes too, so a marker can see
    what failed and when; a failure envelope never short-circuits a retry on the next run.
    Non-transient errors (authentication, malformed request/schema, programming errors)
    fail immediately - exponential backoff cannot fix a bad key.
    """
    # The key covers the SCHEMA ITSELF, not just its class name: a schema edit under an
    # unchanged name used to leave old envelopes valid-looking, so a cached payload could
    # bypass the validation a fresh response would have faced.
    schema_fp = hashlib.sha256(json.dumps(
        schema.model_json_schema(), sort_keys=True, default=str
    ).encode("utf-8")).hexdigest()[:12]
    key = hashlib.sha256(json.dumps(
        [system, user, config.MODEL, config.SAMPLING_TEMPERATURE,
         config.CALL_INDEX_BASE + seed_offset, schema.__name__, schema_fp], sort_keys=True
    ).encode("utf-8")).hexdigest()[:16]
    path = RAW_DIR / f"{schema.__name__}-{key}.json"
    if path.exists():
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            _quarantine(path, f"unreadable ({type(e).__name__})")
            rec = {}
        if rec.get("ok"):
            # metadata must match the request being served: a valid envelope renamed or
            # copied over another cache path must not be accepted for it
            expected = {"model": config.MODEL,
                        "temperature": config.SAMPLING_TEMPERATURE,
                        "call_index": config.CALL_INDEX_BASE + seed_offset,
                        "prompt_sha": key}
            wrong = {k: (rec.get(k), v) for k, v in expected.items()
                     if rec.get(k) != v}
            if wrong:
                _quarantine(path, f"envelope metadata does not match this request "
                                  f"({list(wrong)})")
            elif not reusable_under_current_ceiling(rec):
                print(f"    re-asking {path.name}: produced under a different output "
                      f"ceiling than the one now in force")
            else:
                try:
                    payload = schema.model_validate(rec["payload"]).model_dump()
                    config.ledger_add("shock", path)
                    return payload
                except Exception as e:  # noqa: BLE001 - any invalid cache is quarantined
                    _quarantine(path, f"payload no longer validates against "
                                      f"{schema.__name__} ({type(e).__name__})")
    last = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = client_().beta.chat.completions.parse(
                model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                call_index=config.CALL_INDEX_BASE + seed_offset,
                max_tokens=config.MAX_OUTPUT_TOKENS, response_format=schema,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            payload = r.choices[0].message.parsed.model_dump()
            path.write_text(json.dumps(
                {"ok": True, "payload": payload, "attempt": attempt + 1,
                 "model": config.MODEL,
                 "call_index": config.CALL_INDEX_BASE + seed_offset,
                 "temperature": config.SAMPLING_TEMPERATURE,
                 "request": getattr(r, "request", None),
                 "provenance": getattr(r, "provenance", None),
                 "model_served": getattr(r, "model", None),
                 "prompt_sha": key,
                 "request_id": getattr(r, "id", None),
                 "usage": getattr(r, "usage", None),
                 "timestamp": datetime.now(timezone.utc).isoformat()}, indent=1))
            config.ledger_add("shock", path)
            return payload
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:140]}"
            if type(e).__name__ in _NON_TRANSIENT:
                break
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, .5))
    path.write_text(json.dumps(
        {"ok": False, "error": last, "schema": schema.__name__,
         "model": config.MODEL,
         "call_index": config.CALL_INDEX_BASE + seed_offset,
         "prompt_sha": key,
         "timestamp": datetime.now(timezone.utc).isoformat()}, indent=1))
    raise LLMCallError(
        f"{schema.__name__} call failed ({last}). Nothing reportable can be built from a "
        f"failed call, so the run stops here; the failure envelope is {path.name}. Fix the "
        f"cause and re-run - every successful call is cached and will not re-bill.")


_TAG_RE = re.compile(r"([^\[\]]+?\.(?:pdf|html?|txt|md))(?:\s+p\.(\d+))?", re.I)


def _cited_tag(src: str) -> str:
    """
    Parse a citation into the canonical "file p.N" form used by the retrieval tags.

    Anything that does not parse returns a sentinel that can never match, so a malformed
    citation is ungrounded rather than accidentally valid.
    """
    m = _TAG_RE.search(src or "")
    if not m:
        return "\x00unparseable"
    fname, page = m.group(1).strip(), m.group(2)
    return f"{fname} p.{int(page)}" if page else fname


# The controlled vocabularies a channel review may use. The two verdicts are the two
# evidence legs, defined at length above `global_leg()` further down this file.
GLOBAL_VERDICTS = ("supports", "does_not_support", "uncertain", "scenario_fact")
AUSTRALIAN_VERDICTS = ("supports", "does_not_support", "uncertain")
CONFIDENCE_VALUES = ("low", "moderate", "high")
DECISIONS = ("accept", "modify", "reject")

GLOBAL_LEG_OK = ("quote_supported", "scenario_fact")


def mechanism_groups(branches: list["Branch"]) -> dict:
    """
    Collapse branches into the distinct mechanisms a person actually has to judge.

    A run generates 40-48 branches and most are restatements: the model says "external
    demand falls" five ways. Reviewing all of them is clerical, and reviewing none of them
    is what the previous version amounted to.

    THE PROXY IS PART OF THE KEY. Grouping on (channel_type, direction, horizon) alone was
    too coarse for a source judgement: `risk_appetite | easing | weeks` held both
    `gpr_index` and `vix`; `import_prices | tightening | 1-2q` held `cpi_qoq`, `cpi_yoy`
    and inflation expectations. One support verdict then covered several separately
    generated, separately cited factual claims. Keying on the proxy as well costs a handful
    of extra reviews and stops one signature from vouching for four different claims.

    Returns {group_key: [branches]}, ordered by the best credibility score in each group.
    """
    groups: dict[str, list] = {}
    for b in branches:
        key = (f"{b.channel_type or 'other'} | {b.direction} | {b.horizon} | "
               f"{b.proxy or 'unmodellable'}")
        groups.setdefault(key, []).append(b)
    return dict(sorted(groups.items(),
                       key=lambda kv: -max((x.score or 0) for x in kv[1])))


# Two tiers, because the evidence fields ask about a PROXY and some groups have none.
#
#   FULL  a group with a modellable branch can move a number. It needs your confidence and
#         what Cycle, Words and Replay say about that proxy.
#   MIN   a group with no proxy can only move the qualitative direction tally, and there is
#         no proxy for Cycle or Words to have said anything about. It still needs both
#         verdicts, a decision, a reason and initials - `channel_direction()` counts these
#         branches, so an unreviewed one is an unsigned vote.
REVIEW_FIELDS = ("confidence", "global_verdict", "australian_verdict", "evidence_cycle",
                 "evidence_words", "evidence_replay", "decision", "reason", "by")
REVIEW_FIELDS_MIN = ("global_verdict", "australian_verdict", "decision", "reason", "by")

# Free text is fine for the evidence and the reason. It is not fine for a verdict: any
# non-empty string used to pass for a confidence and a support verdict, so "probably?" was
# accepted as a judgement and nothing downstream could act on it.
REVIEW_VALUES = {"confidence": CONFIDENCE_VALUES,
                 "global_verdict": GLOBAL_VERDICTS,
                 "australian_verdict": AUSTRALIAN_VERDICTS,
                 "decision": DECISIONS}


def _validate_one_review(key: str, rec: dict, members: list, facts: dict | None,
                         needs_full: bool, already: list[str]) -> str:
    """Validate ONE review record against its group; return its canonical channel name."""
    wanted = REVIEW_FIELDS if needs_full else REVIEW_FIELDS_MIN
    gaps = [f for f in wanted if not str(rec.get(f, "")).strip()]
    if gaps:
        raise ValueError(
            f"the channel review for '{key}' is missing {gaps}. A group that shocks a "
            f"proxy needs a confidence, a verdict on EACH of the two evidence legs, "
            f"evidence from Cycle/Words/Replay, a decision, a reason and initials - the "
            f"rubric marks all of them. A group with no proxy needs the two verdicts, a "
            f"decision, a reason and initials.")
    for field, allowed in REVIEW_VALUES.items():
        if field not in wanted and not str(rec.get(field, "")).strip():
            continue
        if rec.get(field) not in allowed:
            raise ValueError(
                f"'{key}': {field} is {rec[field]!r}; it must be one of {list(allowed)}. "
                f"A verdict that is not one of these cannot be acted on, and free text "
                f"in this field is how 'probably?' came to count as a judgement.")

    by_name = {b.channel: b for b in members}
    name = str(rec.get("canonical", "")).strip()
    if not name:
        if len(members) == 1:
            name = members[0].channel
        else:
            raise ValueError(
                f"'{key}' has {len(members)} member branches and no 'canonical' field. "
                f"Name the ONE branch whose evidence you actually read - the verdict "
                f"binds that branch. Naming it ATTESTS that the other members restate "
                f"it; a member that is a genuinely distinct claim gets its own record, "
                f"in a list under this key. Members: {[b.channel[:60] for b in members]}")
    canon = by_name.get(name)
    if canon is None:
        raise ValueError(
            f"'{key}': canonical branch {name!r} is not a member of this group. "
            f"Members: {[b.channel[:60] for b in members]}")
    if name in already:
        raise ValueError(f"'{key}': two records name the same canonical {name!r}")
    if rec["decision"] != "reject" and not canon.keep:
        raise ValueError(
            f"'{key}': canonical branch '{name[:50]}' was pruned on its credibility "
            f"score, so an accept/modify verdict bound to it changes nothing. Pick a "
            f"kept member, or reinstate this one in ADJUDICATIONS with a reason.")
    if rec["global_verdict"] == "supports" and not canon.quote_verified:
        raise ValueError(
            f"'{key}': global_verdict is 'supports' but the canonical branch "
            f"'{name[:50]}' carries no machine-verified quotation. You cannot certify "
            f"that a quotation supports the claim when the quotation itself failed "
            f"verification - pick the member whose quote verified, use a scenario "
            f"fact id, or change the verdict.")
    if rec["global_verdict"] == "scenario_fact":
        fid = str(rec.get("fact_id", "")).strip()
        if not fid or fid not in (facts or {}):
            raise ValueError(
                f"'{key}': global_verdict is 'scenario_fact' but fact_id "
                f"{fid or '(missing)'!r} is not in SCENARIO_FACTS for this scenario "
                f"({sorted(facts or {})}). A stipulated fact is named, registered and "
                f"verified - not declared.")
        if not str(rec.get("fact_link_reason", "")).strip():
            raise ValueError(
                f"'{key}': scenario_fact cites {fid!r} but records no fact_link_reason. "
                f"The registry check proves the fact EXISTS verbatim in the narrative - "
                f"provenance, not entailment - and an unrelated claim could borrow a "
                f"valid id. Say, in a sentence, why THIS fact carries THIS canonical "
                f"claim's global leg. That sentence is the entailment judgement, and "
                f"your initials sign it.")
    return name


def _stamp_canonical(b: "Branch", rec: dict, facts: dict | None) -> None:
    b.human_confidence = rec.get("confidence", "")
    b.human_decision = rec["decision"]
    b.human_reason = rec["reason"]
    b.reviewed_by = rec["by"]
    b.global_verdict = rec["global_verdict"]
    b.australian_verdict = rec["australian_verdict"]
    if rec["global_verdict"] == "scenario_fact":
        b.scenario_fact_id = rec["fact_id"]
        b.scenario_fact_quote = (facts or {})[rec["fact_id"]]
        b.scenario_fact_link = rec["fact_link_reason"]
    if rec["decision"] == "reject":
        b.terminal_exclusion = f"rejected on review ({rec['by']})"
        b.keep, b.proxy = False, None
        b.kept_by = b.terminal_exclusion
        b.note = (b.note + f" | REJECTED ON REVIEW by {rec['by']}: "
                           f"{rec['reason']}")[:900]


def apply_channel_reviews(branches: list["Branch"], reviews: dict | None,
                          require: bool = True, scenario: str = "",
                          narrative: str = "",
                          facts: dict[str, str] | None = None) -> dict:
    """
    Apply YOUR per-mechanism review before anything is shocked, swept or reported.

    WHERE THIS SITS. The credibility prune and the expansion have already run - they must,
    because expansion needs first-order survivors and a review needs something to review.
    Everything that produces a NUMBER comes after: the grounding bar, the shocks, the
    threshold sweep, the corners. So a `reject` here removes the channel from every figure
    in the report. It is `reject`, not a footnote.

    THE VERDICT BINDS ONE BRANCH, AND RESTATEMENT-HOOD IS ATTESTED, NOT INFERRED. A group
    key is deliberately coarse - (channel_type | direction | horizon | proxy) - so two
    genuinely different causal mechanisms CAN collide under it. Equivalence is therefore a
    human decision: writing one record for a group attests, over your initials, that every
    other member restates your canonical; where a member is a distinct claim, the group's
    value is a LIST of records, one per distinct canonical, and only the members no record
    names are barred as restatements.

    SCENARIO FACTS: PROVENANCE BY MACHINE, ENTAILMENT BY A PERSON. `scenario_fact` must
    name a `fact_id` from SCENARIO_FACTS, and the registered text is verified verbatim
    against the narrative - an identity check, which is all a machine can do. It cannot
    check that the fact SUPPORTS the claim ("Australian wages accelerate" could borrow the
    valid `insurance` id), so the record must also carry `fact_link_reason`: the named
    reviewer's one-sentence answer to why this fact carries this claim. The judgement is
    the entailment; the machine only guarantees it cannot be about a fact that is not
    there.

    A "supports" verdict requires the canonical branch's OWN quotation to have passed the
    machine check: certifying a quotation that failed verification is a review of nothing.

    `reviews` is keyed on the group key from `mechanism_groups()`. Every record needs all
    of REVIEW_FIELDS (REVIEW_FIELDS_MIN for groups with no proxy), and the verdict fields
    must take a listed value; a partial or free-text record is rejected, not half-applied.
    """
    groups = mechanism_groups(branches)
    reviews = {k: (v if isinstance(v, list) else [v]) for k, v in (reviews or {}).items()}
    if any(rec.get("global_verdict") == "scenario_fact"
           for recs in reviews.values() for rec in recs):
        verify_scenario_facts(narrative, facts)

    # A group needs a review if it would otherwise SURVIVE - not merely if it is
    # modellable. `channel_direction()` tallies surviving unmodellable branches too, and
    # that tally is the qualitative conclusion the report leads with.
    def matters(members):
        return any(b.keep for b in members)

    def modellable(members):
        return any(b.keep and b.proxy and b.n_sd for b in members)

    applied, missing, not_required, restatements = 0, [], [], 0
    n_records = 0
    for key, members in groups.items():
        recs = reviews.get(key)
        if recs is None:
            (missing if matters(members) else not_required).append(key)
            continue
        canon_names: list[str] = []
        for rec in recs:
            canon_names.append(_validate_one_review(key, rec, members, facts,
                                                    modellable(members), canon_names))
        by_canon = dict(zip(canon_names, recs))
        # WHO RESTATES WHOM is part of the attestation. With one record it is unambiguous:
        # every other member restates the one canonical. With SEVERAL records it is not -
        # a member could restate either canonical - so each record must carry an explicit
        # "restatements" list, and together the lists plus the canonicals must partition
        # the group exactly: every member owned once, none twice, none left dangling.
        member_names = [b.channel for b in members]
        if len(recs) == 1:
            owner_of = {n: canon_names[0] for n in member_names if n != canon_names[0]}
        else:
            owner_of = {}
            for name, rec in by_canon.items():
                listed = rec.get("restatements")
                if not isinstance(listed, list):
                    raise ValueError(
                        f"'{key}' has {len(recs)} records, so each must carry a "
                        f"'restatements' list saying which members restate ITS canonical "
                        f"- with several canonicals the code cannot infer who restates "
                        f"whom, and refuses to guess. Record for '{name[:50]}' has none.")
                for m in listed:
                    if m not in member_names:
                        raise ValueError(f"'{key}': record for '{name[:40]}' lists "
                                         f"{m!r}, which is not a member of this group")
                    if m in canon_names:
                        raise ValueError(f"'{key}': {m!r} is itself a canonical and "
                                         f"cannot be listed as a restatement")
                    if m in owner_of:
                        raise ValueError(f"'{key}': {m!r} is claimed as a restatement by "
                                         f"two records - every member belongs to exactly "
                                         f"one canonical")
                    owner_of[m] = name
            dangling = [n for n in member_names
                        if n not in canon_names and n not in owner_of]
            if dangling:
                raise ValueError(
                    f"'{key}': members {dangling} are neither a canonical nor listed in "
                    f"any record's 'restatements'. Assign each to the canonical it "
                    f"restates, or give it its own record.")
        for b in members:
            b.review_group = key
            rec = by_canon.get(b.channel)
            if rec is not None:
                _stamp_canonical(b, rec, facts)
                continue
            owner = owner_of[b.channel]
            b.human_decision = "restatement"
            b.reviewed_by = by_canon[owner]["by"]
            b.restatement_of = owner
            b.terminal_exclusion = f"restatement (canonical: {owner[:50]})"
            if b.keep or b.proxy:
                restatements += 1
            b.keep, b.proxy = False, None
            b.kept_by = b.terminal_exclusion
            b.note = (b.note + f" | RESTATEMENT of '{owner[:50]}' - attested by "
                               f"{by_canon[owner]['by']}; kept as provenance, barred "
                               f"from every number")[:900]
        applied += 1
        n_records += len(recs)
    if missing and require:
        raise ValueError(
            f"{len(missing)} mechanism group(s) would survive into the answer and have no "
            f"entry in CHANNEL_REVIEWS:" + chr(10) + "  - "
            + (chr(10) + "  - ").join(missing) + chr(10)
            + "Review the GROUPS, not the individual branches: "
            + f"{len(missing)} decisions, not {len(branches)}. Run "
            + "`python src/scenarios.py --discover`, fill the generated template in "
            + "place, and copy it to data/processed/channel_reviews.json (the "
            + "CHANNEL_REVIEWS dict in this file is the supported alternative).")
    print(f"    channel review: {applied} group(s) reviewed with {n_records} record(s), "
          f"{len(missing)} outstanding, {len(not_required)} not required (nothing in "
          f"them survives) - {len(groups)} groups from {len(branches)} raw branches, "
          f"{restatements} attested restatement(s) barred")
    return {"n_groups": len(groups), "n_reviewed": applied, "n_records": n_records,
            "unreviewed": missing, "not_required": not_required,
            "n_branches": len(branches), "restatements_barred": restatements,
            "groups": {k: [b.channel for b in v] for k, v in groups.items()}}


# -------------------------------------------------------------------------------------------
# THE ASSESSED JUDGEMENT RECORDS MAY LIVE IN JSON FILES instead of the Python dicts above.
# -------------------------------------------------------------------------------------------
# Forty-odd review records pasted into a .py file is clerical, merge-conflict-prone group
# work, and makes malformed Python a failure mode of judgement work. The supported route:
#
#   1. `--discover` writes outputs/channel_reviews.template.json.
#   2. Copy it to data/processed/channel_reviews.json and fill each group's "review"
#      object in place (or "reviews": [...] for a group with several distinct canonicals).
#      Write pruning-audit records into data/processed/pruning_reviews.json as
#      {scenario: {channel: {verdict, reason, by}}}.
#   3. The pipeline loads and validates them exactly as it would the Python dicts.
#
# The Python dicts still work and take precedence when non-empty, so either form is a
# legitimate assessed record; the stage hash covers whichever is in effect.

def _load_json_table(filename: str) -> dict:
    path = config.DATA_PROCESSED / filename
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{path} is not valid JSON ({e}). Fix it or delete it - a "
                         f"judgement file that cannot be read must not be skipped "
                         f"silently.") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object")
    return raw


def effective_channel_reviews(scenario: str) -> dict | None:
    """CHANNEL_REVIEWS entry for `scenario`, or the JSON record file's, in that order."""
    if CHANNEL_REVIEWS.get(scenario):
        return CHANNEL_REVIEWS[scenario]
    raw = _load_json_table("channel_reviews.json").get(scenario)
    if raw is None:
        return None
    groups = raw.get("groups", raw)
    out = {}
    for key, g in groups.items():
        if not isinstance(g, dict):
            continue
        if "reviews" in g and isinstance(g["reviews"], list):
            recs = [r for r in g["reviews"] if str(r.get("decision", "")).strip()]
        elif "review" in g:
            recs = [g["review"]] if str(g["review"].get("decision", "")).strip() else []
        elif "decision" in g:
            recs = [g] if str(g.get("decision", "")).strip() else []
        else:
            recs = []
        if recs:
            out[key] = recs if len(recs) > 1 else recs[0]
    return out or None


def effective_pruning_reviews(scenario: str) -> dict | None:
    if PRUNING_REVIEWS.get(scenario):
        return PRUNING_REVIEWS[scenario]
    return _load_json_table("pruning_reviews.json").get(scenario)


@lru_cache(maxsize=64)
def _file_fingerprint(path_str: str, mtime_ns: int, size: int) -> str:
    """sha256 of a file's bytes, memoised on (path, mtime, size) so repeated hashing of
    the same unchanged input costs one read per process."""
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def file_fingerprint(path) -> str:
    """Fingerprint one input file; a missing file fingerprints as 'absent'."""
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        return "absent"
    st = p.stat()
    return _file_fingerprint(str(p), st.st_mtime_ns, st.st_size)


def _json_fingerprint(path, drop: tuple = ()) -> str:
    """
    Content fingerprint of a JSON input: canonicalised, with run incidentals (wall time,
    cache counts) dropped. Byte-hashing these files made an honest offline regeneration
    look like a changed input, because the incidentals differ on every run while the
    content that shapes the outputs does not.
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        return "absent"
    rec = json.loads(p.read_text(encoding="utf-8"))
    for k in drop:
        rec.pop(k, None)
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()[:16]


#: Committed evidence about the context documents, so a checkout WITHOUT them can still be
#: verified. The documents themselves are not redistributable and are gitignored; this file
#: is their fingerprint record and IS committed.
CONTEXT_FINGERPRINTS = config.OUTPUTS / "context_fingerprints.json"


def _live_context_fingerprints() -> dict:
    """{filename: content hash} for the SOURCE DOCUMENTS present on disk, if any.

    Uses `context_docs.source_files()` so this is exactly the set `load_documents()`
    reads - not "every supported file in the folder", which used to sweep in README.md
    and sources.json and thereby made the set impossible to reproduce from a record.
    """
    return {p.name: file_fingerprint(p)
            for p in context_docs.source_files(context_docs.CONTEXT_DIR)}


def write_context_fingerprints(docs: dict | None = None) -> dict:
    """Record one content hash per context document, and commit it.

    THE GAP THIS CLOSES. The Shock stage hash reads the bytes of every context document,
    and the submission suite loads the real files. Both are right when the documents are
    present - and neither can run at all on a Git-only checkout, because the documents are
    correctly gitignored as non-redistributable. A marker who cloned the repository was
    therefore asked to reproduce a hash from files the repository is forbidden to carry.

    The fix is to commit the fingerprints rather than the sources. `input_fingerprints()`
    prefers the files when they are present and falls back to this record when they are
    not, so the stage hash is the same either way, and the submission suite can verify the
    declaration offline. Reading the documents themselves remains a staff action, done
    from the team's own downloads.
    """
    live = _live_context_fingerprints()
    rec = {"documents": live,
           "n_documents": len(live),
           # the DECLARATION is committed, so it is recorded separately rather than
           # mixed in with the documents it describes
           "declaration_sha256": file_fingerprint(
               context_docs.CONTEXT_DIR / "sources.json"),
           "declared": sorted(context_docs._declared_sources()),
           "written_at": datetime.now(timezone.utc).isoformat()}
    config.atomic_write_text(CONTEXT_FINGERPRINTS, json.dumps(rec, indent=1))
    return rec


def _committed_context_fingerprints() -> dict:
    if not CONTEXT_FINGERPRINTS.exists():
        return {}
    try:
        return json.loads(CONTEXT_FINGERPRINTS.read_text(encoding="utf-8")).get(
            "documents", {})
    except (json.JSONDecodeError, OSError):
        return {}


def input_fingerprints() -> dict:
    """
    The DATA this stage reads, fingerprinted: the source declaration, every context
    document, and the frozen-model inputs. Part of the stage hash, so replacing a source
    PDF, editing sources.json or swapping the frozen panel after generation invalidates
    the committed artefact exactly as editing a prompt would.

    The SOURCE DOCUMENTS are fingerprinted from the files when they are present, and from
    the committed `outputs/context_fingerprints.json` when they are not - see
    `write_context_fingerprints()`. The two are the same mapping by construction, because
    both use `context_docs.source_files()`, so the stage hash is reproducible on a clone
    that cannot legally carry the sources.

    The DECLARATION (`sources.json`) is hashed as its own field. It is committed, so it is
    always available and never needs a fallback - and mixing it in with the documents was
    what stopped the fallback ever firing: the dictionary could not become empty, so the
    "no sources present" branch was unreachable.
    """
    docs = _live_context_fingerprints() or _committed_context_fingerprints()
    return {"context_documents": docs,
            "context_declaration": file_fingerprint(
                context_docs.CONTEXT_DIR / "sources.json"),
            "frozen_panel": file_fingerprint(config.DATA_PROCESSED
                                             / "panel_frozen.parquet"),
            "frozen_tiers": file_fingerprint(config.FROZEN_TIERS),
            # the Shock model artefacts and the Words audit both shape the outputs: the
            # model card supplies the frozen coefficients, and words_audit.json decides
            # which constructs the reaction-profile ranking may use
            "shock_design": file_fingerprint(config.MODEL_CARD / "shock_design.parquet"),
            "shock_train_design": file_fingerprint(config.MODEL_CARD
                                                   / "shock_train_design.parquet"),
            "model_card_performance": _json_fingerprint(config.MODEL_CARD
                                                        / "performance.json"),
            "words_audit": _json_fingerprint(config.OUTPUTS / "words_audit.json",
                                             drop=("usage",)),
            # the TEAM's construct scores are the seven columns the episode profiles
            # rank against (see _profile_panel) - editing them after generation must
            # invalidate the committed Shock artefacts
            "construct_scores": file_fingerprint(config.CONSTRUCT_SCORES)}


def config_hash() -> str:
    """
    Identifies THIS Shock configuration - the four prompts, the model settings, every
    human judgement table in effect (Python dict or JSON file alike), AND the input data
    the stage read (sources.json, the context documents, the frozen-model files). Stored
    in shock.json by `write_shock_outputs()` and compared by the submission check: an
    artefact generated under different prompts, judgements or inputs than the repository
    holds cannot pass as current.
    """
    payload = json.dumps({
        "prompts": [BRANCH_PROMPT, EVALUATE_PROMPT, EXPAND_PROMPT, ADVERSARIAL_PROMPT],
        "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
        "call_index_base": config.CALL_INDEX_BASE,
        "inputs": input_fingerprints(),
        "evaluate_chunk": EVALUATE_CHUNK,
        # the output ceiling changes the answer: a lower one truncates
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "response_schemas": {m.__name__: hashlib.sha256(json.dumps(
            m.model_json_schema(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12] for m in (BranchOut, EvaluateOut, AdversarialOut)},
        # Everything that shapes what is GENERATED, RETRIEVED, PRUNED or REPORTED. A field
        # missing here is a field that can be edited after artefact generation without
        # invalidating the submission check - which is how the check gets lied to.
        "scenarios": SCENARIOS,
        "retrieval_terms": RETRIEVAL_TERMS,
        "n_branch_samples": N_BRANCH_SAMPLES,
        "prune_threshold": PRUNE_THRESHOLD,
        "direction_margin": DIRECTION_MARGIN,
        "min_quote_chars": MIN_QUOTE_CHARS,
        "min_fragment_chars": MIN_FRAGMENT_CHARS,
        "implied_by_effect": IMPLIED_BY_EFFECT,
        "require_reviews": [REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS],
        "horizon": SHOCK_HORIZON, "strict": STRICT,
        "base_rows": BASE_ROWS, "headline": HEADLINE_BASE,
        "allowance": CALIBRATION_ALLOWANCE,
        "scenario_facts": SCENARIO_FACTS,
        "channel_reviews": {s: effective_channel_reviews(s) for s in SCENARIOS},
        "pruning_reviews": {s: effective_pruning_reviews(s) for s in SCENARIOS},
        "proxy_judgements": PROXY_JUDGEMENTS,
        "adjudications": ADJUDICATIONS,
        "pathway_adjudications": PATHWAY_ADJUDICATIONS,
        "pathway_policy": PATHWAY_POLICY,
        "direction_weights": DIRECTION_WEIGHTS,
        "direction_weight_reasons": DIRECTION_WEIGHT_REASONS,
        "coherence": COHERENCE,
        "adversarial_responses": ADVERSARIAL_RESPONSES,
        "expected_profiles": EXPECTED_PROFILES,
        "claimed_analogues": CLAIMED_ANALOGUES,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


_RUN_ID: str | None = None


def stage_run_id() -> str:
    """
    One identifier per Shock PROCESS, stamped into shock.json, every tree in
    tot_trees.json and reaction_profiles.json. The three artefacts are only coherent when
    written by the same run - a reaction_profiles.json regenerated on its own describes a
    tree that may no longer exist, and before the run_id nothing could tell.
    """
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = f"{random.getrandbits(48):012x}"
    return _RUN_ID


def write_shock_outputs(out: dict, trees: dict, wall_seconds: float) -> None:
    """The one writer for shock.json and tot_trees.json - the stage hash rides along.
    Atomic: a crash between the two writes cannot leave one current and one stale."""
    for t in trees.values():
        t["run_id"] = stage_run_id()
    config.atomic_write_text(
        config.OUTPUTS / "shock.json",
        json.dumps({"scenarios": out, "config_hash": config_hash(),
                    "run_id": stage_run_id(),
                    "wall_seconds": round(wall_seconds, 1)},
                   indent=2, default=float))
    config.atomic_write_text(
        config.OUTPUTS / "tot_trees.json",
        json.dumps(trees, indent=2, default=float))


PRUNING_REVIEW_VERDICTS = ("agree", "reinstate")


def required_pruning_reviews(tree: "Tree") -> dict[str, str]:
    """
    The rejected branches a person must look at, and why each is on the list.

    THE GAP THIS CLOSES. The model generates the branches AND scores their credibility, so
    with no human audit of the discard pile it prunes its own output unsupervised. The
    assessed work is checking the machine; the discard pile is where an inconvenient or
    opposing mechanism goes missing, and nothing else in the pipeline looks at it.

    Only threshold- and evaluator-driven rejections are audited. Branches YOU rejected in
    CHANNEL_REVIEWS or ADJUDICATIONS were human decisions already, and restatements are
    barred mechanically.
    """
    rejected = [b for b in tree.branches
                if b.keep is False and not b.terminal_exclusion
                and b.kept_by in ("", "threshold")]
    out: dict[str, str] = {}
    for b in sorted(rejected, key=lambda x: -(x.score or 0))[:3]:
        out[b.channel] = f"top-scoring rejected branch (score {round(b.score or 0, 2)})"
    for want in ("tightening", "easing"):
        if any(b.direction == want for r, b in
               ((k, x) for k in out for x in rejected if x.channel == k)):
            continue
        cands = [b for b in rejected if b.direction == want]
        if cands:
            best = max(cands, key=lambda b: b.score or 0)
            out.setdefault(best.channel, f"best rejected '{want}' branch - the direction "
                                         f"the survivors may be missing")
    surviving_families = {b.channel_type for b in tree.survivors()}
    for fam in sorted(channels.CHANNEL_NAMES):
        if fam in surviving_families:
            continue
        cands = [b for b in rejected if b.channel_type == fam]
        if cands:
            best = max(cands, key=lambda b: b.score or 0)
            out.setdefault(best.channel,
                           f"no '{fam}' channel survived; this is its best rejected one")
    return out


def apply_pruning_reviews(tree: "Tree", reviews: dict | None,
                          adjudications: dict | None, require: bool = True) -> dict:
    """Attach the audit verdicts; block a reportable run while any are outstanding."""
    reviews = reviews or {}
    needed = required_pruning_reviews(tree)
    by_branch = {b.channel: b for b in tree.branches}
    adj = {k.strip().lower()[:60] for k, v in (adjudications or {}).items()
           if isinstance(v, dict) and v.get("keep")}
    done, missing = [], []
    for channel, why in needed.items():
        rec = reviews.get(channel)
        if rec is None:
            missing.append(f"{channel}  <- {why}")
            continue
        gaps = [f for f in ("verdict", "reason", "by") if not str(rec.get(f, "")).strip()]
        if gaps:
            raise ValueError(f"the pruning review for '{channel[:50]}' is missing {gaps}")
        if rec["verdict"] not in PRUNING_REVIEW_VERDICTS:
            raise ValueError(f"'{channel[:50]}': verdict must be one of "
                             f"{list(PRUNING_REVIEW_VERDICTS)}, got {rec['verdict']!r}")
        if (rec["verdict"] == "reinstate"
                and channel.strip().lower()[:60] not in adj):
            raise ValueError(
                f"'{channel[:50]}' is marked reinstate but has no ADJUDICATIONS entry "
                f"with keep=True. A reinstatement is a credibility decision and must be "
                f"recorded as one, with its reason, where the sweep will replay it.")
        b = by_branch.get(channel)
        if b is not None:
            b.note = (b.note + f" | PRUNING AUDIT ({rec['by']}): {rec['verdict']} - "
                               f"{rec['reason']}")[:900]
        done.append(channel)
    if missing and require:
        raise ValueError(
            f"{len(missing)} rejected branch(es) need a pruning review - the model scored "
            f"its own output and threw these away, and a person has to look at the "
            f"discard pile:" + chr(10) + "  - " + (chr(10) + "  - ").join(missing)
            + chr(10) + "Record verdict/reason/by for each, in "
            + "data/processed/pruning_reviews.json (or the PRUNING_REVIEWS dict). "
            + "'agree' with a reason is full credit; 'reinstate' also needs an "
            + "ADJUDICATIONS entry.")
    extras = [k for k in reviews if k not in needed and k not in by_branch]
    if extras:
        raise ValueError(f"PRUNING_REVIEWS names branches that do not exist: {extras}")
    print(f"    pruning audit: {len(done)} of {len(needed)} required rejected branches "
          f"reviewed" + (f", {len(missing)} outstanding" if missing else ""))
    return {"required": needed, "reviewed": done, "outstanding": missing}


def assert_reviews_consistent(branches: list["Branch"],
                              adjudications: dict | None) -> None:
    """
    A branch cannot be kept by an adjudication and removed by a terminal decision.

    Terminal means rejected in CHANNEL_REVIEWS, or barred by the grounding rule. An
    adjudication is a judgement about CREDIBILITY, and credibility does not restore
    evidence: if you believe the channel, the fix is to evidence it, not to overrule the
    gate. `_apply_terminal_exclusions()` would win anyway - this raises so that the
    contradiction is visible rather than silently resolved.
    """
    if not adjudications:
        return
    by = {k.strip().lower()[:60]: v for k, v in adjudications.items()}
    for b in branches:
        rec = by.get(b.channel.strip().lower()[:60])
        if not (rec and isinstance(rec, dict) and rec.get("keep")):
            continue
        if b.human_decision == "reject":
            raise ValueError(
                f"'{b.channel[:45]}' is rejected by its channel review and kept by an "
                f"adjudication. Decide which, and remove the other.")
        if b.terminal_exclusion:
            raise ValueError(
                f"'{b.channel[:45]}' is kept by an adjudication but excluded by "
                f"{b.terminal_exclusion}. An adjudication cannot supply missing evidence - "
                f"either record the evidence in CHANNEL_REVIEWS or drop the adjudication.")


def _adjudication_for(channel: str) -> str:
    """Your recorded decision on a flagged branch, matched on the channel name."""
    key = channel.strip().lower()[:60]
    for k, v in (PATHWAY_ADJUDICATIONS or {}).items():
        if k.strip().lower()[:60] == key:
            return v
    return ""


def _norm(t: str) -> str:
    return " ".join((t or "").split()).lower()


def _typename(ann) -> str:
    """A readable name for a plain annotation, including optionals and containers."""
    import types
    import typing
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)
    if origin in (typing.Union, getattr(types, "UnionType", None)):
        return " | ".join("null" if a is type(None) else _typename(a) for a in args)
    if origin in (list, set, tuple):
        return "[" + ", ".join(_typename(a) for a in args) + ", ...]"
    if origin is dict:
        return "{" + ": ".join(_typename(a) for a in args) + "}"
    return getattr(ann, "__name__", str(ann))


def _nested_model(ann):
    """(the BaseModel this field renders as, whether it is wrapped in a list)."""
    import typing
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann, False
    if typing.get_origin(ann) is list:
        inner = typing.get_args(ann)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return inner, True
    return None, False


def required_fields(schema, _indent: int = 0, _seen: tuple = ()) -> str:
    """
    The response shape, rendered FROM the Pydantic class that is passed to the API.

    TWO FAULTS THIS FIXES, both of which structured output hid at runtime.

    The prompts used to display a hand-written JSON example, and it drifted: the branch
    example omitted `source`, `proxy_movement`, `first_round_effect` and
    `policy_implication`, all required, while the adversarial example asked for a field the
    schema does not have.

    Generating the block then fixed the fields but named the wrong class. The branch call
    passes `response_format=BranchOut`, whose shape is `{"channels": [...]}`, and printed
    the contract for `ChannelOut`, which is one element of that list. Students were being
    told to design a prompt against a contract that contradicted the response format the
    API was given. It is now rendered RECURSIVELY from the outer class, so the block in the
    prompt is the shape the API is told to return, nested models expanded in place.
    """
    pad, inner = "  " * _indent, "  " * (_indent + 1)
    seen = _seen + (schema.__name__,)
    lines = []
    if _indent == 0:
        lines.append(f"Return JSON matching this shape ({schema.__name__}):")
    lines.append(pad + "{")
    for name, f in schema.model_fields.items():
        req = "   (required)" if f.is_required() else ""
        desc = (f.description or "").strip()
        tail = req + (f"  // {desc[:110]}" if desc else "")
        model, is_list = _nested_model(f.annotation)
        if model is not None and model.__name__ not in seen:
            lines.append(f'{inner}"{name}": ' + ("[" if is_list else "") + tail)
            body = required_fields(model, _indent + 2, seen)
            lines.append(body + (", ..." if is_list else ""))
            lines.append(inner + ("]," if is_list else ","))
        else:
            lines.append(f'{inner}"{name}": {_typename(f.annotation)},{tail}')
    lines.append(pad + "}")
    return chr(10).join(lines)


CONTRACTS = (("branch and expand", "BranchOut"), ("evaluate", "EvaluateOut"),
             ("adversarial", "AdversarialOut"))


def print_contracts() -> None:
    """
    The exact contracts the API calls use.

        python -c "import scenarios; scenarios.print_contracts()"

    These are the classes passed as `response_format`, rendered from the classes
    themselves. What you paste into a prompt therefore cannot disagree with what the API is
    told to return - which is the fault this replaces.
    """
    for step, cls in CONTRACTS:
        print(f"--- {step}: response_format={cls} ---")
        print(required_fields(globals()[cls]))
        print()


# The shortest quoted fragment that counts as evidence. Below this a "quote" is a
# phrase that could appear in almost any document.
# -------------------------------------------------------------------------------------------
# WHAT COUNTS AS EVIDENCE, AND WHY THERE ARE TWO KINDS
# -------------------------------------------------------------------------------------------
# The context documents are global risk reports. Measured across all four of the marker set:
# "Taiwan" appears 7 times, "Australia" 13, "monetary policy" 6, and "iron ore" not at all,
# in roughly 600,000 characters. They are good evidence that a blockade would disrupt
# semiconductor supply chains and shipping. They contain almost nothing about how that
# reaches the RBA.
#
# So a transmission channel has TWO legs, it needs BOTH, and they are evidenced differently:
#
#   THE GLOBAL LEG   scenario -> world effect. Settled by ONE of:
#                      - a retrieved passage, cited to file and page, with a verbatim
#                        quote a named person has read and judged relevant; or
#                      - a fact the scenario narrative itself stipulates, named by its id
#                        in SCENARIO_FACTS and verified verbatim against the narrative.
#
#   THE AUSTRALIAN LEG   world effect -> Australian proxy -> RBA reaction. NOT in these
#                    documents and not machine-checkable. A named person signs it in
#                    CHANNEL_REVIEWS, with what Cycle, Words and Replay told them.
#
# The verdicts bind ONE branch per mechanism group - the CANONICAL branch the reviewer
# actually read. The other members are restatements: they stay in the record as provenance
# and are barred from every number, because one signature must not vouch for several
# separately generated, separately cited claims. See `apply_channel_reviews()`.

MIN_QUOTE_CHARS = 30
# Below this a fragment is connective tissue rather than a claim - but it is still checked,
# because dropping short fragments silently is how a fabricated clause gets in for free.
MIN_FRAGMENT_CHARS = 12


def verify_citation(cited: str, quote: str, passages: dict) -> dict:
    """
    Three separate questions, answered separately. They used to be collapsed into one.

    1. Does the citation NAME a passage that actually reached the prompt?  `tag_valid`
    2. Does the QUOTE appear verbatim in that passage?                     `quote_verified`
    3. Does that passage SUPPORT the claim the channel makes?              a person's job

    Only the first two are machine-checkable, and only the third is grounding in the sense
    the rubric means. `global_verdict` in CHANNEL_REVIEWS is where a person answers it.

    ELISIONS. Models quote with ellipses - "the supply chain... is exposed" - so the check
    splits on the ellipsis and matches the fragments. It requires EVERY material fragment,
    IN ORDER, with at least one of them MIN_QUOTE_CHARS long. The earlier version accepted
    the quote if ANY single 30-character fragment occurred anywhere in the passage, so

        [genuine 54-character fragment] ... [fabricated policy conclusion]

    verified. Half a quotation with an invented conclusion bolted on is the precise failure
    a quote check exists to catch.
    """
    tag = _cited_tag(cited)
    passage = passages.get(tag)
    if passage is None:
        return {"source_tag": tag, "tag_valid": False, "quote_verified": False,
                "fragments": 0, "fragments_matched": 0, "matched_chars": 0,
                "why": "the citation names a passage that was never retrieved"}
    body = _norm(passage)
    parts = [_norm(f) for f in re.split(r"\.{2,}|\[\s*\.{3}\s*\]|\u2026", quote or "")]
    frags = [f for f in parts if len(f) >= MIN_FRAGMENT_CHARS]
    if not frags or max(len(f) for f in frags) < MIN_QUOTE_CHARS:
        return {"source_tag": tag, "tag_valid": True, "quote_verified": False,
                "fragments": len(frags), "fragments_matched": 0, "matched_chars": 0,
                "why": f"no quoted fragment of at least {MIN_QUOTE_CHARS} characters"}
    pos, matched, missing = 0, [], []
    for f in frags:
        i = body.find(f, pos)
        if i < 0:
            missing.append(f[:45])
        else:
            matched.append(f)
            pos = i + len(f)
    ok = not missing
    return {"source_tag": tag, "tag_valid": True, "quote_verified": ok,
            "fragments": len(frags), "fragments_matched": len(matched),
            "matched_chars": sum(len(f) for f in matched),
            "why": "" if ok else
                   (f"{len(missing)} of {len(frags)} quoted fragment(s) are not in the "
                    f"cited passage, in order: {missing}")}


# What a branch needs before it may drive a shock or the qualitative direction.
#
# BOTH LEGS. Not either. A channel asserts two things and they are evidenced differently:
#
#   LEG 1, GLOBAL       scenario -> world effect. A retrieved passage can settle this. It
#                       needs a valid page tag, a verbatim quote, AND a person's verdict
#                       that the quoted words support the claim - because a WEF country
#                       risk-ranking table can be quoted verbatim under "terms of trade
#                       improvement" and a substring test will pass it. Where the SCENARIO
#                       NARRATIVE itself stipulates the world effect, record
#                       global_verdict="scenario_fact": no source is needed for a premise
#                       the assignment handed you, and pretending to find one is worse.
#
#   LEG 2, AUSTRALIAN   world effect -> Australian proxy -> RBA reaction. These documents
#                       cannot evidence it and no quote check can. A person records
#                       australian_verdict, with what Cycle, Words and Replay told them.
#
# An earlier version required one leg OR the other, described itself as two-legged, and
# reported the two counts from a single field that answered the human question first - so
# a channel with a verified quote AND a human verdict was counted only as human-supported,
# and the report said Taiwan had 0 verifiable quotes when the raw field said 1.


def verify_scenario_facts(narrative: str, facts: dict[str, str] | None) -> None:
    """
    Every registered fact must appear VERBATIM in the scenario narrative.

    This is what stops "scenario_fact" from meaning "things we would like to be stipulated".
    The registry is data, so it is checked like data - the same way a source quotation is
    checked against its passage.
    """
    body = _norm(narrative)
    bad = [fid for fid, q in (facts or {}).items() if _norm(q) not in body]
    if bad:
        raise ValueError(
            f"SCENARIO_FACTS entries {bad} are not verbatim in the scenario narrative. A "
            f"stipulated fact is a sentence the scenario actually contains - fix the quote "
            f"or delete the entry.")


def global_leg(b: "Branch") -> str:
    """
    LEG 1. Machine check AND human relevance verdict, in that order.

    In the marker trees the machine check alone passed several channels whose quoted words
    plainly do not entail them: a country risk-ranking table under "Terms of Trade
    Improvement"; one general paragraph on energy security under BOTH "Increased Australian
    Demand for Energy Exports" and "Consumer Demand Squeeze"; an IMF sentence about energy
    costs, tourism and remittances under "Increased Credit Spreads". The words are on the
    cited pages. They do not support the claims.

        quote_supported    verified quote, and a person says it supports the global claim
        scenario_fact      the narrative stipulates it - the branch carries the VERIFIED
                           fact id and quote, stamped by `apply_channel_reviews()`
        fact_unverified    a scenario_fact verdict with no verified fact id behind it.
                           BARRED - this is the bypass an earlier version allowed
        quote_irrelevant   verified quote, and a person says it does NOT support the claim
        quote_unreviewed   verified quote that nobody has read
        quote_unverified   a real retrieved passage, but the quote is not in it
        tag_invalid        cites a passage that was never retrieved
        not_in_sources     the model said so itself
    """
    if b.global_verdict == "scenario_fact":
        return "scenario_fact" if b.scenario_fact_id else "fact_unverified"
    if _norm(b.source) == "not in sources":
        return "not_in_sources"
    if not b.citation_tag_valid:
        return "tag_invalid"
    if not b.quote_verified:
        return "quote_unverified"
    if b.global_verdict == "does_not_support":
        return "quote_irrelevant"
    if b.global_verdict != "supports":
        return "quote_unreviewed"
    return "quote_supported"


def australian_leg(b: "Branch") -> str:
    """
    LEG 2. A person, or nothing.

    Across all four documents in the marker set, in roughly 600,000 characters: "Taiwan"
    appears 7 times, "Australia" 13, "monetary policy" 6, "iron ore" not at all. Requiring
    a quotation for Australian monetary transmission would be requiring teams to invent one.
    """
    return {"supports": "human_supported",
            "does_not_support": "human_rejected",
            "uncertain": "human_uncertain"}.get(
                (b.australian_verdict or "").strip(), "unreviewed")


def evidence_status(b: "Branch") -> str:
    """both | global_only | australian_only | neither. Only `both` may drive a number."""
    g_ok = global_leg(b) in GLOBAL_LEG_OK
    a_ok = australian_leg(b) == "human_supported"
    return {(True, True): "both", (True, False): "global_only",
            (False, True): "australian_only", (False, False): "neither"}[(g_ok, a_ok)]


def enforce_grounding(branches: list["Branch"]) -> dict:
    """
    Bar every channel that does not carry BOTH legs, and return counts that overlap
    honestly rather than a partition that hides the overlap.

    Nothing is deleted: a barred channel stays in the tree with its reason, because what
    the model proposed and could not support is itself a finding. The bar is TERMINAL -
    `terminal_exclusion` is set, so no sweep threshold can reinstate it.
    """
    from collections import Counter
    counts = {"both": 0, "global_only": 0, "australian_only": 0, "neither": 0,
              "barred_from_shock": 0}
    g_why, a_why = Counter(), Counter()
    for b in branches:
        b.global_leg = global_leg(b)
        b.australian_leg = australian_leg(b)
        st = evidence_status(b)
        b.evidence_status = st
        counts[st] += 1
        g_why[b.global_leg] += 1
        a_why[b.australian_leg] += 1
        if st == "both":
            continue
        # A restatement or review rejection is already terminal for a MORE specific
        # reason; overwriting it here would erase the audit trail of who barred it first.
        if not b.terminal_exclusion:
            b.terminal_exclusion = (f"grounding: {st} (global {b.global_leg}, "
                                    f"australian {b.australian_leg})")
        if b.keep or b.proxy:
            b.keep, b.proxy = False, None
            b.kept_by = b.terminal_exclusion
            b.note = (b.note + f" | BARRED FROM THE SHOCK: {st}. Global leg "
                               f"{b.global_leg}; Australian leg {b.australian_leg}. Both "
                               f"are required. It stays in the record as something the "
                               f"model proposed and the evidence did not carry.")[:900]
            counts["barred_from_shock"] += 1
    # RAW machine counts, independent of every human verdict. These are the numbers to
    # quote when the report says how much of the tree the SOURCES could evidence.
    counts["quote_verified_raw"] = sum(1 for b in branches if b.quote_verified)
    counts["tag_valid_raw"] = sum(1 for b in branches if b.citation_tag_valid)
    counts["scenario_fact"] = int(g_why.get("scenario_fact", 0))
    counts["restatements"] = sum(1 for b in branches
                                 if b.terminal_exclusion.startswith("restatement"))
    counts["canonical_claims"] = sum(
        1 for b in branches
        if b.review_group and not b.terminal_exclusion.startswith("restatement"))
    counts["n_branches"] = len(branches)
    counts["global_leg"] = dict(g_why)
    counts["australian_leg"] = dict(a_why)
    return counts


def branch(tree: Tree, user: str, depth: int = 0, parent: str | None = None,
           n_samples: int = N_BRANCH_SAMPLES, prompt: str | None = None,
           valid_tags: set[str] | None = None) -> list[Branch]:
    """
    BRANCH. Generate candidates n_samples times INDEPENDENTLY, then pool and deduplicate.

    Independent samples matter. One call asking for eight channels returns eight variations
    on whichever line of reasoning the model started down. Several separate calls at
    temperature > 0 start in different places, and the disagreement between them tells you
    how settled the question is.

    Two checks run on every candidate as it arrives:
      SOURCE GROUNDING - `source` must name a file you actually downloaded, or say
        "not in sources". A citation to a document that is not in your folder is recorded as
        ungrounded and counted in the run summary.
      PATHWAY CONSISTENCY - `validate_pathway()`; an inconsistent channel is kept, flagged,
        and must be adjudicated by a person.
    """
    seen = {b.channel.strip().lower()[:60] for b in tree.branches}
    out, ungrounded, inconsistent = [], 0, 0
    for i in range(n_samples):
        sys_prompt = (prompt or BRANCH_PROMPT) + chr(10) + chr(10) + required_fields(BranchOut)
        data = llm_parsed(sys_prompt, user, BranchOut, depth * 100 + i)
        for c in (data.get("channels") or []):
            key = str(c.get("channel", "")).strip().lower()[:60]
            if not key or key in seen:
                continue
            seen.add(key)
            note = str(c.get("note", ""))[:250]
            src = str(c.get("source", "")).strip()
            # Validate against the ACTUAL page tags that were retrieved, not just the
            # filename. Checking the filename alone let "[wef-global-risks-report-2026.pdf
            # p.999]" through - a real document and an invented page.
            # EXACT membership after parsing, not a substring test. "file.pdf p.4" is a
            # substring of "file.pdf p.41", so substring matching accepted a citation to a
            # page that was never retrieved.
            ver = verify_citation(src, c.get("source_quote", ""), valid_tags or {})
            grounded = ver["tag_valid"] and ver["quote_verified"]
            if not grounded and _norm(src) != "not in sources":
                ungrounded += 1
                note = (note + f" | CITATION NOT VERIFIED: {ver['why']}")[:500]
            bad = validate_pathway(c)
            if bad:
                inconsistent += 1
                note = (note + f" | PATHWAY INCONSISTENT: {bad}")[:600]
            ctype = str(c.get("channel_type", "other")).strip().lower()
            b = Branch(channel=str(c.get("channel", ""))[:200],
                       channel_type=ctype if ctype in channels.CHANNEL_NAMES else "other",
                       mechanism=str(c.get("mechanism", ""))[:400],
                       direction=str(c.get("direction", "ambiguous")).lower(),
                       horizon=str(c.get("horizon", "unknown")),
                       proxy=(c.get("proxy") or None),
                       n_sd=float(c.get("n_sd") or 0.0),
                       depth=depth, parent=parent,
                       source=src[:120],
                       source_tag=ver["source_tag"],
                       source_quote=str(c.get("source_quote", ""))[:400],
                       citation_tag_valid=ver["tag_valid"],
                       quote_verified=ver["quote_verified"],
                       pathway=(f"{c.get('proxy_movement')} -> "
                                f"{c.get('first_round_effect')} -> "
                                f"{c.get('policy_implication')}"),
                       pathway_ok=(bad is None),
                       pathway_note=(bad or ""),
                       adjudication=(_adjudication_for(str(c.get("channel", "")))
                                     or (f"policy: {PATHWAY_POLICY}"
                                         if bad and PATHWAY_POLICY else "")),
                       note=note)
            if bad and PATHWAY_POLICY == "unmodellable" and b.proxy:
                b.proxy = None
                b.note = (b.note + " | MADE UNMODELLABLE by PATHWAY_POLICY: its stated "
                                   "pathway does not hold together, so its reasoning is "
                                   "kept but it does not drive a number")[:700]
            out.append(b)
            tree.branches.append(b)
    tree.grounding.setdefault(f"depth{depth}", {"generated": 0, "grounded": 0,
                                                 "ungrounded": 0, "not_in_sources": 0})
    g = tree.grounding[f"depth{depth}"]
    g["generated"] += len(out)
    g["grounded"] += sum(1 for b in out if b.citation_tag_valid and b.quote_verified)
    g["not_in_sources"] += sum(1 for b in out
                               if b.source.strip().lower() == "not in sources")
    g["ungrounded"] += sum(1 for b in out
                           if not (b.citation_tag_valid and b.quote_verified)
                           and _norm(b.source) != "not in sources")
    print(f"    branch(depth={depth}): {n_samples} samples -> {len(out)} new channels"
          + (f", {ungrounded} ungrounded citations" if ungrounded else "")
          + (f", {inconsistent} pathway inconsistencies" if inconsistent else ""))
    return out


def evaluate(branches: list[Branch], scenario: str, panel_columns: list[str],
             passages: str = "") -> None:
    """
    EVALUATE. Score each branch in place, and check its proposed proxy against the panel.

    A model will cheerfully propose 'oil_price' as a proxy. It is not a panel column. A
    channel with no real proxy is marked unmodellable, not quietly given one.
    """
    # The evaluator is asked to judge credibility, so it gets the evidence: the source
    # citation, the horizon, the shock magnitude, the pathway flag and the retrieved
    # passages. An earlier version passed only mechanism/direction/proxy and then asked
    # whether the channel was well grounded and correctly sized - questions its input
    # could not answer.
    #
    # CHUNKED, because coverage is the contract. One call carrying every channel is how a
    # 36-channel second-order tree came back with 12 scores - the model simply stopped -
    # and the missing 24 then took the silent default. Small chunks keep each response
    # comfortably inside what the model will actually finish, and one follow-up call
    # re-asks for anything an individual chunk still skipped.
    # Matching is by GENERATED ID, not by name. Matching on a lowercased name prefix let
    # two long, similarly worded channels collide, silently applying one score to both;
    # the ids make the required one-to-one match checkable exactly.
    ids = {f"ch{i:03d}": b for i, b in enumerate(branches)}
    id_of = {id(b): k for k, b in ids.items()}

    def _score_chunk(chunk: list[Branch]) -> dict[str, dict]:
        payload = [{"id": id_of[id(b)], "channel": b.channel,
                    "channel_type": b.channel_type,
                    "mechanism": b.mechanism, "direction": b.direction, "proxy": b.proxy,
                    "n_sd": b.n_sd, "horizon": b.horizon, "pathway": b.pathway,
                    "source": b.source, "pathway_flag": b.pathway_note or None,
                    "note": b.note[:200]}
                   for b in chunk]
        data = llm_parsed(EVALUATE_PROMPT + chr(10) + chr(10)
                          + "Echo each channel's `id` back as `channel_id`, exactly. "
                          + required_fields(EvaluateOut),
                          f"SCENARIO: {scenario}\n\nAVAILABLE PANEL COLUMNS: "
                          f"{panel_columns}\n\n"
                          f"{channels.describe_calibration()}\n\n"
                          f"THE SOURCE PASSAGES THE CHANNELS CITE:\n{passages[:12000]}\n\n"
                          f"CHANNELS:\n{json.dumps(payload, indent=1)}", EvaluateOut)
        got, dupes, unknown = {}, [], []
        want = {id_of[id(b)] for b in chunk}
        for s in (data.get("scores") or []):
            cid = str(s.get("channel_id", "")).strip()
            if cid not in ids:
                unknown.append(cid)
            elif cid in got:
                dupes.append(cid)
            elif cid in want:
                got[cid] = s
        if dupes or unknown:
            raise LLMCallError(
                f"the evaluator's response is not a one-to-one match: duplicate ids "
                f"{dupes[:4]}, unknown ids {unknown[:4]}. One score per requested "
                f"channel is the contract; re-run to retry the call.")
        return got

    by_id: dict[str, dict] = {}
    for i in range(0, len(branches), EVALUATE_CHUNK):
        by_id.update(_score_chunk(branches[i:i + EVALUATE_CHUNK]))
    # FAIL CLOSED on coverage. An earlier version gave any unscored channel a default
    # credibility of 0.5, so an evaluator that dropped channels - or failed entirely -
    # produced a tree of confident-looking middling scores nobody had assigned.
    still = [b for b in branches if id_of[id(b)] not in by_id]
    if still:
        by_id.update(_score_chunk(still))
    missing = [b.channel for b in branches if id_of[id(b)] not in by_id]
    if missing:
        raise LLMCallError(
            f"the evaluator scored {len(branches) - len(missing)} of {len(branches)} "
            f"channels even after a follow-up call; unscored: "
            f"{[m[:50] for m in missing[:5]]}. A default credibility is not a judgement, "
            f"so the run stops - re-run to retry the evaluator calls.")
    for b in branches:
        s = by_id[id_of[id(b)]]
        score = float(s.get("credibility"))
        if not 0.0 <= score <= 1.0 or score != score:
            raise LLMCallError(
                f"evaluator credibility {score!r} for '{b.channel[:50]}' is outside [0, 1]")
        b.score = score
        if s.get("reason"):
            b.note = (b.note + " | eval: " + str(s["reason"]))[:800]
        if b.proxy and b.proxy not in panel_columns:
            b.note = (b.note + f" | PROXY '{b.proxy}' NOT IN PANEL")[:800]
            b.proxy = None


def prune(branches: list[Branch], threshold: float = PRUNE_THRESHOLD,
          require_both_directions: bool = True,
          adjudications: dict[str, dict] | None = None) -> list[Branch]:
    """
    PRUNE. Threshold, then YOUR overrides, then the opposing-direction guard.

    The same three steps the sensitivity sweep runs, from the same functions in
    `scenario_engine` - `apply_adjudications()` and `opposing_direction_guard()`. Two
    implementations of this is how the sweep came to reinstate branches teams had rejected.

    `adjudications` is {channel name: {"keep": bool, "reason": str, "by": initials}}. A bare
    boolean is rejected: the rubric marks the reason and a boolean has nowhere to put one.
    """
    for b in branches:
        b.keep = (b.score or 0) >= threshold
        b.kept_by = "threshold"
    n = apply_adjudications(branches, adjudications)
    if n:
        print(f"    {n} branch(es) overridden by the team")
    reinstated = opposing_direction_guard(branches, enabled=require_both_directions)
    for c in reinstated:
        print(f"    prune: no opposing channel survived; reinstated '{c[:45]}' for "
              f"adjudication")
    kept = [b for b in branches if b.keep]
    print(f"    prune: {len(kept)}/{len(branches)} kept "
          f"(directions {sorted({b.direction for b in kept})})")
    return kept


def expand(tree: Tree, survivors: list[Branch], scenario: str, user_base: str,
           n_samples: int = 2, valid_tags: set[str] | None = None) -> list[Branch]:
    """EXPAND. Second-order consequences of each surviving first-order channel."""
    out = []
    for b in survivors:
        user = (f"{user_base}\n\n{'=' * 60}\n"
                f"FIRST-ORDER CHANNEL ALREADY ESTABLISHED:\n"
                f"  {b.channel} - {b.mechanism} "
                f"(direction {b.direction}, horizon {b.horizon}, pathway {b.pathway})")
        out += branch(tree, user, depth=b.depth + 1, parent=b.channel,
                      n_samples=n_samples, prompt=EXPAND_PROMPT, valid_tags=valid_tags)
    return out


def adversarial(tree: Tree, scenario: str, result: dict, passages: str = "") -> dict:
    """
    Argue the other side, WITH the source evidence in hand.

    An earlier version passed only the surviving channels and the conclusion, so the model
    had to invent supporting history - and did: it asserted that central banks responded to
    2008 with tighter policy. It now receives the same retrieved passages the branch step
    saw, and it is told to mark any historical claim it cannot ground.

    The case to argue is derived from the model verdict, including the case where the model
    declined to speak at all - see `scenario_engine.opposite_of()`.
    """
    conclusion = result.get("model_verdict", "out_of_support")
    payload = [{"channel": b.channel, "direction": b.direction,
                "mechanism": b.mechanism, "pathway": b.pathway,
                "n_sd": b.n_sd, "proxy": b.proxy, "source": b.source}
               for b in tree.survivors()]
    stated = (f"the model returned '{conclusion}'"
              if conclusion != "out_of_support" else
              "the model declined to speak: the shocked economy lies outside anything in "
              "its training data")
    return llm_parsed(
        ADVERSARIAL_PROMPT + chr(10) + chr(10) + required_fields(AdversarialOut),
        f"SCENARIO: {scenario}" + chr(10) + chr(10) +
        f"OUR CONCLUSION: {stated}. The channels we kept point "
        f"'{result.get('channel_direction', {}).get('direction', 'unknown')}'." + chr(10) +
        f"THE CASE YOU MUST ARGUE: {opposite_of(conclusion)}" + chr(10) + chr(10) +
        f"THE SOURCE PASSAGES AVAILABLE. Ground every historical claim in one of these or "
        f"mark it unsupported:" + chr(10) + f"{passages[:12000]}" + chr(10) + chr(10) +
        f"OUR SURVIVING CHANNELS:" + chr(10) + f"{json.dumps(payload, indent=1)}",
        AdversarialOut)


# -------------------------------------------------------------------------------------------
# The reaction profile - this is where WORDS feeds SHOCK
# -------------------------------------------------------------------------------------------

def failed_constructs() -> set[str]:
    """
    The constructs whose audit gates FAILED, read from outputs/words_audit.json.

    `compare_reaction()` must not rank analogues on a failed instrument - a between/within
    ratio below 1 means the construct varies more between repeat calls on one document than
    between documents. The audit already knows which ones failed; this reads its verdict
    rather than asking anyone to re-declare it.
    """
    path = config.OUTPUTS / "words_audit.json"
    if not path.exists():
        print("    words_audit.json not found - no constructs excluded from the "
              "reaction-profile ranking. Run text_features first.")
        return set()
    audit = json.loads(path.read_text(encoding="utf-8"))
    out = set()
    for row in audit.get("quality", []):
        gates = [v for k, v in row.items() if k.startswith("passes_")]
        if gates and not all(bool(g) for g in gates):
            out.add(row["construct"])
    return out


def _profile_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    WORDS FEEDS SHOCK HERE, literally. The frozen panel supplies the structured history
    every team shares, but the seven construct columns the episode profiles rank
    against are YOUR validated measurements from construct_scores.parquet - not the
    instructor's. An earlier version ranked the frozen panel's own text columns, so a
    team's prompt-engineered instrument never actually reached the Shock task it was
    built for; the brief promised otherwise.
    """
    scores = config.validate_construct_scores(reportable=True)
    scores = scores.set_index(pd.to_datetime(scores["meeting_date"])).sort_index()
    missing = panel.index.difference(scores.index)
    if len(missing):
        raise RuntimeError(
            f"{len(missing)} panel meetings carry no construct scores (first: "
            f"{missing[0].date()}) - run BOTH Words passes before a reportable Shock "
            f"run")
    out = panel.copy()
    for c in config.TEXT_FEATURES:
        out[c] = scores.loc[out.index, c].to_numpy()
    return out


def reaction_profiles(panel: pd.DataFrame, expected: dict[str, dict],
                      claimed: dict[str, str] | None = None) -> dict:
    """
    Your expected reaction profile per scenario, checked against real episodes.

    `expected` is {scenario: {construct: percentile}} - YOUR judgement, on the seven
    dimensions Words produced, of how the Board would sound if this scenario happened. It is
    compared against the profiles of episodes that actually occurred, with any construct
    that FAILED its audit gates excluded from the ranking (and the all-construct mean
    reported alongside). `claimed` is CLAIMED_ANALOGUES - the analogue you named before
    running the comparison; a mismatch is flagged for you to defend or change.

    This is the whole reason Words exists in this assignment. The shock model tells you
    where the cycle goes; it cannot tell you what the Board's reaction would LOOK like,
    because a scenario has no minutes. The construct dimensions give you a vocabulary for
    that, and the historical episodes give you calibration for it.

    WRITES outputs/reaction_profiles.json, which is a required submission artefact.
    """
    problems = []
    for scen in SCENARIOS:
        exp = (expected or {}).get(scen)
        if exp is None:
            problems.append(f"{scen}: no expected profile")
            continue
        missing = [c for c in config.TEXT_FEATURES if c not in exp]
        if missing:
            problems.append(f"{scen}: missing constructs {missing}")
        bad = [c for c, v in exp.items()
               if not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0]
        if bad:
            problems.append(f"{scen}: non-numeric or out-of-[0,1] values for {bad}")
    if problems:
        raise ValueError(
            "EXPECTED_PROFILES is incomplete:" + chr(10) + "  - "
            + (chr(10) + "  - ").join(problems) + chr(10)
            + "State a percentile in [0, 1] for every one of the seven constructs, for "
            + "both scenarios, before running the comparison.")
    excluded = failed_constructs()
    out = {}
    for scen, exp in expected.items():
        print(f"\n  reaction profile - {scen}")
        claim = (claimed or {}).get(scen)
        cmp = compare_reaction(exp, panel, analogue=claim, exclude=excluded)
        out[scen] = {"expected": exp,
                     "claimed_analogue": claim,
                     "closest_episode": cmp.index[0],
                     "claim_matches": (None if not claim else bool(cmp.index[0] == claim)),
                     "mean_abs_gap": float(cmp["mean_abs_gap"].iloc[0]),
                     "excluded_constructs": sorted(excluded),
                     "gaps": cmp.round(3).to_dict("index")}
    config.atomic_write_text(
        config.OUTPUTS / "reaction_profiles.json",
        json.dumps({"episode_profiles": episode_profiles(panel).round(3).to_dict("index"),
                    "excluded_constructs": sorted(excluded),
                    "run_id": stage_run_id(),
                    "scenarios": out}, indent=2, default=float))
    return out

# -------------------------------------------------------------------------------------------
# The SUPPLIED runners. You do not write control flow anywhere in this assignment: you
# supply prompts, judgement tables and reviews above, and these runners validate and
# execute them. run() blocks until the reviews are complete; --discover produces the
# review template; --worked demonstrates the workflow on the unassessed scenario.
# -------------------------------------------------------------------------------------------

def _setup():
    """The FROZEN, CONSTRUCT-FREE shock model. Everyone shocks the same coefficients."""
    clf, X, panel = frozen_shock_model()
    return panel, clf, X


def _grow_tree(name: str, narrative: str, terms: list[str], docs: dict,
               clf, X, panel) -> tuple[dict, Tree]:
    """
    Everything UP TO the reviews and grounding - and nothing that produces a number.

    Discovery and the worked demonstration stop here, so a tree whose reviews are still
    being written never acquires probabilities, sweeps or an adversarial exchange to be
    quoted from. `_evaluate_tree()` is the paid-off half.
    """
    manifest = context_docs.manifest(docs)
    passages, tags = context_docs.relevant_passages(docs, terms, return_tags=True)
    context_docs.require_sources(manifest, tags, scenario=name)

    user = _user_block(narrative, X, passages)
    tree = Tree(scenario=name, coherence=COHERENCE.get(name, []))
    judgements = PROXY_JUDGEMENTS.get(name, {})
    horizon = SHOCK_HORIZON

    first = branch(tree, user, depth=0, valid_tags=tags)
    evaluate(first, narrative, list(X.columns), passages)
    survivors = prune(first, adjudications=ADJUDICATIONS.get(name))
    second = expand(tree, survivors, narrative, user, valid_tags=tags)
    if second:
        evaluate(second, narrative, list(X.columns), passages)
        prune(second, adjudications=ADJUDICATIONS.get(name))

    # HUMAN REVIEW AND GROUNDING, BEFORE anything is shocked. Order matters: a rejected
    # mechanism group must be out before pruning settles, and a channel the model could not
    # ground must be barred unless a person supplied the evidence for it.
    review = apply_channel_reviews(tree.branches, effective_channel_reviews(name),
                                   require=REQUIRE_CHANNEL_REVIEWS,
                                   scenario=name, narrative=narrative,
                                   facts=SCENARIO_FACTS.get(name))
    pruning_audit = apply_pruning_reviews(tree, effective_pruning_reviews(name),
                                          ADJUDICATIONS.get(name),
                                          require=REQUIRE_PRUNING_REVIEWS)
    grounding = enforce_grounding(tree.branches)
    assert_reviews_consistent(tree.branches, ADJUDICATIONS.get(name))
    print(f"    grounding: {grounding['both']} carry BOTH legs, "
          f"{grounding['global_only']} global only, "
          f"{grounding['australian_only']} Australian only, "
          f"{grounding['neither']} neither "
          f"-> {grounding['barred_from_shock']} barred from the shock")
    print(f"      raw machine check, independent of every human verdict: "
          f"{grounding['quote_verified_raw']} of {grounding['n_branches']} branches "
          f"carry a verified quotation; {grounding['scenario_fact']} rest on a fact "
          f"the scenario itself stipulates")
    print(f"      global leg: {grounding['global_leg']}")
    print(f"      australian leg: {grounding['australian_leg']}")
    res = {"source_manifest": manifest,
           "channel_review_summary": review,
           "pruning_audit": pruning_audit,
           "grounding_counts": grounding,
           "_passages": passages}
    return res, tree


def _evaluate_tree(name: str, res: dict, tree: Tree, clf, X, panel) -> dict:
    """The half that produces numbers. Runs only on a fully reviewed tree."""
    judgements = PROXY_JUDGEMENTS.get(name, {})
    horizon = SHOCK_HORIZON
    # A reportable direction is weighted by YOU, with a reason per family - the LLM's own
    # credibility scores are the fallback for exploration only.
    validate_direction_weights(DIRECTION_WEIGHTS or None,
                               DIRECTION_WEIGHT_REASONS or None, required=True)
    conflicts = resolve_proxy_conflicts(tree.branches, judgements)

    # Support status BEFORE any capping. Capping until a scenario fits inside the training
    # range is how you would hide an extrapolation rather than handle it, so the uncapped
    # status is measured first and reported alongside.
    pre_shocks, _ = adjudicate_shocks(channels_at(tree.modellable(), SHOCK_HORIZON),
                                      judgements)
    pre_row, pre_applied = apply_shock(X, panel, pre_shocks)
    uncapped_support = support_check(X, pre_row, pre_applied)

    caps = cap_to_calibration(tree.branches, allowance=CALIBRATION_ALLOWANCE)

    headline = None
    per_base = {}
    for label, when in BASE_ROWS.items():
        print(f"{chr(10)}    --- base row: {label} ({when}) ---")
        r = run_scenario(tree, clf, X, panel, strict=STRICT, horizon=horizon,
                         judgements=judgements, base_row=when,
                         weights=DIRECTION_WEIGHTS or None)
        per_base[label] = {"base_row_date": r["base_row_date"],
                           "headroom": r["headroom"],
                           "model_verdict": r["model_verdict"],
                           "baseline": r["baseline"],
                           "probabilities": r["probabilities"],
                           "shift": r["shift"],
                           "in_support": r["support"]["in_support"]}
        if label == HEADLINE_BASE:
            headline = r
    res = {**(headline if headline is not None else r), **res}
    res["per_base_row"] = per_base
    res["proxy_judgements"] = conflicts
    res["calibration_caps"] = caps
    res["support_before_capping"] = uncapped_support
    if caps and uncapped_support["in_support"] != res["support"]["in_support"]:
        print(f"    NOTE: capping changed the support verdict "
              f"(uncapped in_support={uncapped_support['in_support']}, "
              f"capped={res['support']['in_support']}). Report both.")

    print("\n    PRUNING-THRESHOLD SWEEP")
    res["prune_sweep"] = prune_sweep(
        tree, clf, X, panel, judgements=judgements, horizon=horizon,
        adjudications=ADJUDICATIONS.get(name),
        pathway_adjudications=PATHWAY_ADJUDICATIONS,
        base_row=BASE_ROWS[HEADLINE_BASE],
        weights=DIRECTION_WEIGHTS or None).to_dict("records")

    print("\n    FOCUSED SENSITIVITY - your stated uncertainty bounds")
    fc = focused_corners(tree, clf, X, panel,
                         bounds=uncertainty_bounds(judgements),
                         horizon=horizon, judgements=judgements,
                         base_row=BASE_ROWS[HEADLINE_BASE],
                         require_bounds=True)
    res["focused_corners"] = fc.to_dict("records") if len(fc) else []

    print("\n    THE SAME TREE AT THE OTHER HORIZONS")
    other = {}
    for h in HORIZONS:
        if h == horizon:
            continue
        try:
            r = run_scenario(tree, clf, X, panel, strict=False, horizon=h,
                             judgements=judgements)
            other[h] = {"model_verdict": r["model_verdict"],
                        "n_channels": r["n_channels_at_horizon"],
                        "channel_direction": r["channel_direction"]["direction"]}
        except (ValueError, RuntimeError) as e:
            # the expected operational outcome: nothing usable at this horizon. A
            # KeyError, NameError or corrupt panel is a DEFECT and propagates - an
            # earlier version recorded those as horizon "errors" too, which presented
            # a programming fault as an ordinary empty horizon.
            other[h] = {"error": str(e)[:80]}
    for h, r in other.items():
        print(f"      {h:10s} {r.get('n_channels', 0):2d} channels -> model "
              f"{r.get('model_verdict', 'n/a')}, channels say "
              f"{r.get('channel_direction', 'n/a')}")
    res["other_horizons"] = other

    res["channel_coverage"] = channels.coverage(tree.survivors())
    res["channel_reviews"] = effective_channel_reviews(name) or {}
    res["pruning_reviews"] = effective_pruning_reviews(name) or {}
    res["direction_weights"] = DIRECTION_WEIGHTS
    res["direction_weight_reasons"] = DIRECTION_WEIGHT_REASONS
    res["adversarial"] = adversarial(tree, SCENARIOS.get(name, name), res,
                                     res.get("_passages", ""))
    res["adversarial_response"] = ADVERSARIAL_RESPONSES.get(name)
    if res["adversarial_response"] is None:
        print("    NOTE: no ADVERSARIAL_RESPONSES entry for this scenario. The "
              "critique is generated for you; answering it is the assessed part.")
    res.pop("_passages", None)
    return res


def _one_tree(name: str, narrative: str, terms: list[str], docs: dict,
              clf, X, panel) -> tuple[dict, Tree]:
    """Grow, review, then evaluate - the reportable path."""
    res, tree = _grow_tree(name, narrative, terms, docs, clf, X, panel)
    res = _evaluate_tree(name, res, tree, clf, X, panel)
    return res, tree


def _review_template(name: str, tree: Tree, summary: dict, audit: dict) -> dict:
    """One scenario's skeleton for channel_reviews.template.json."""
    by_name = {b.channel: b for b in tree.branches}
    groups = {}
    for key, members in summary["groups"].items():
        ms = [by_name[m] for m in members]
        needs_full = not key.endswith("| unmodellable")
        # Every field a completed review can NEED appears in the skeleton, including the
        # conditionally required ones - fact_id and fact_link_reason travel together, and
        # an earlier template omitted the link, so a student filling it "in place" as
        # instructed hit an unexplained blocking error on their first scenario_fact.
        skeleton = {"canonical": ms[0].channel if len(ms) == 1 else "",
                    "global_verdict": "", "fact_id": "", "fact_link_reason": "",
                    "australian_verdict": "", "decision": "", "reason": "", "by": ""}
        if needs_full:
            skeleton = {"canonical": skeleton["canonical"], "confidence": "",
                        "global_verdict": "", "fact_id": "", "fact_link_reason": "",
                        "australian_verdict": "",
                        "evidence_cycle": "", "evidence_words": "", "evidence_replay": "",
                        "decision": "", "reason": "", "by": ""}
        groups[key] = {
            "review": skeleton,
            "needs_review": key in summary["unreviewed"],
            # In an unreviewed tree the grounding bar has removed EVERYTHING, so
            # b.keep is uniformly False here and useless for choosing a canonical.
            # `kept_after_pruning` recovers the pruning outcome: kept now, or kept until
            # the (review-less) grounding pass barred it.
            # depth and parent are shown so the canonical can be chosen safely: a
            # depth-1 canonical whose depth-0 parent is barred as a restatement
            # leaves an orphan the strict validator refuses. Prefer a depth-0
            # member as canonical where the group has one.
            "members": [{"channel": b.channel, "score": b.score,
                         "depth": b.depth, "parent": b.parent,
                         "kept_after_pruning": bool(
                             b.keep or b.terminal_exclusion.startswith("grounding")),
                         "proxy": b.proxy, "n_sd": b.n_sd,
                         "source": b.source, "source_quote": b.source_quote,
                         "quote_verified": b.quote_verified} for b in ms]}
    return {"scenario_facts": SCENARIO_FACTS.get(name, {}),
            "groups": groups,
            "pruning_reviews_required": audit["required"]}


def run_discover() -> dict:
    """
    STEP 1 FOR THE ASSESSED SCENARIOS: python src/scenarios.py --discover

    Runs BOTH assessed scenarios with the review requirements off and STOPS before any
    shock, probability, sweep or adversarial output - a tree whose reviews are unwritten
    has nothing quotable in it. Prints every mechanism-group key and the required pruning
    audit, and writes outputs/channel_reviews.template.json with each member branch's
    citation and quotation inline, plus the registered scenario-fact ids.

    Fill the template in place, copy it to data/processed/channel_reviews.json, write the
    pruning audit to data/processed/pruning_reviews.json, and run the reportable pipeline
    - the requirement flags are restored automatically when this function returns. The
    CHANNEL_REVIEWS / PRUNING_REVIEWS dicts remain the supported alternative.
    """
    if not _prompts_written():
        raise NotImplementedError("Write the four prompts first.")
    global REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS, CHANNEL_REVIEWS
    global PRUNING_REVIEWS
    was = (REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS,
           CHANNEL_REVIEWS, PRUNING_REVIEWS)
    # Discovery is a picture of the UNREVIEWED tree - any reviews already pasted in are
    # set aside for this run, so the template always shows every group.
    REQUIRE_CHANNEL_REVIEWS = REQUIRE_PRUNING_REVIEWS = False
    CHANNEL_REVIEWS, PRUNING_REVIEWS = {}, {}
    print(chr(10) + "  DISCOVERY RUN - review requirements off, nothing here is "
          "reportable: no shocks, no probabilities, no sweep.")
    out = {}
    try:
        docs = context_docs.load_documents()
        panel, clf, X = _setup()
        for name, narrative in SCENARIOS.items():
            print(f"{chr(10)}  {name}")
            res, tree = _grow_tree(name, narrative, RETRIEVAL_TERMS[name], docs,
                                   clf, X, panel)
            summary = res["channel_review_summary"]
            out[name] = _review_template(name, tree, summary, res["pruning_audit"])
            print(f"    {len(summary['unreviewed'])} group(s) to review:")
            for k in summary["unreviewed"]:
                print(f"      - {k}")
            print(f"    {len(res['pruning_audit']['outstanding'])} pruning audit(s) to "
                  f"write:")
            for k in res["pruning_audit"]["outstanding"]:
                print(f"      - {k}")
    finally:
        (REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS,
         CHANNEL_REVIEWS, PRUNING_REVIEWS) = was
    path = config.OUTPUTS / "channel_reviews.template.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"{chr(10)}  wrote {path.name}: "
          + ", ".join(f"{len(v['groups'])} groups for {k}" for k, v in out.items()))
    return out


def run_worked() -> dict:
    """
    THE WORKFLOW DEMONSTRATION, on a scenario that is NOT assessed.

    `python src/scenarios.py --worked` grows a tree on the migration scenario, with the
    review requirements off, and STOPS SHORT of every number - there are no probabilities,
    no sweep and no adversarial exchange in its output, because nothing in it has been
    reviewed. What it shows is the SHAPE of the work: the group keys, the member branches
    with their citations, the pruning-audit list, and what a properly recorded rejected
    branch looks like.

    It cannot produce the assessed group keys - it runs the migration scenario. For those,
    run `python src/scenarios.py --discover`.
    """
    if not _prompts_written():
        raise NotImplementedError("Write the four prompts first.")
    global REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS
    was = (REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS)
    REQUIRE_CHANNEL_REVIEWS = REQUIRE_PRUNING_REVIEWS = False
    print(chr(10) + "  Reviews are OFF for this demonstration and the tree is therefore "
          "not reportable: no shocks, no probabilities, no sweep - every channel is "
          "barred until a person signs both of its evidence legs. That is the point.")
    try:
        name, narrative = WORKED_SCENARIO
        docs = context_docs.load_documents()
        panel, clf, X = _setup()
        print(f"{chr(10)}  WORKED EXAMPLE (not assessed): {name}")
        res, tree = _grow_tree(name, narrative, WORKED_RETRIEVAL_TERMS, docs,
                               clf, X, panel)
        res.pop("_passages", None)
        (config.OUTPUTS / "worked_scenario.json").write_text(
            json.dumps({**res,
                        "template": _review_template(
                            name, tree, res["channel_review_summary"],
                            res["pruning_audit"]),
                        "tree": tree.to_dict()}, indent=2, default=float),
            encoding="utf-8")
    finally:
        REQUIRE_CHANNEL_REVIEWS, REQUIRE_PRUNING_REVIEWS = was
    return res


def _require_complete_scores() -> None:
    """
    A reportable Shock run needs the full corpus scored: the reaction profiles position
    the scenarios against episode profiles built from every meeting's constructs.
    Delegates to THE shared contract (exact authoritative meeting set, values, sd,
    n_calls_valid) - the same one Replay and the panel builder apply, so the three
    consumers cannot drift apart on what "valid scores" means.
    """
    config.validate_construct_scores(reportable=True)


def run() -> dict:
    if not _prompts_written():
        raise NotImplementedError(
            "All four tree-of-thought prompts must be written first. See the Shock stage of "
            "the brief.")
    _require_complete_scores()
    t0 = time.time()
    config.stage_begin("shock", config_hash())
    config.ledger_reset("shock")
    docs = context_docs.load_documents()
    # commit the fingerprints so a clone WITHOUT the non-redistributable sources can
    # still recompute this stage's hash and verify the declaration
    write_context_fingerprints(docs)
    panel, clf, X = _setup()
    print(f"  shock model: construct-free, {X.shape[1]} features, frozen panel "
          f"({len(panel)} meetings)")

    out, trees = {}, {}
    for name, narrative in SCENARIOS.items():
        print(f"\n  {name}")
        res, tree = _one_tree(name, narrative, RETRIEVAL_TERMS[name], docs, clf, X, panel)
        out[name] = res
        trees[name] = tree.to_dict()

    print("\n  REACTION PROFILES - the Words constructs applied to the scenarios")
    profiles = reaction_profiles(_profile_panel(panel), EXPECTED_PROFILES,
                                 claimed=CLAIMED_ANALOGUES)

    write_shock_outputs(out, trees, time.time() - t0)
    config.ledger_commit("shock")
    config.stage_complete("shock", config_hash())
    print(f"\n  wrote shock.json, tot_trees.json, reaction_profiles.json "
          f"({time.time() - t0:.0f}s)")
    return {"scenarios": out, "reaction_profiles": profiles}


if __name__ == "__main__":
    if "--worked" in sys.argv:
        run_worked()
    elif "--discover" in sys.argv:
        run_discover()
    else:
        run()

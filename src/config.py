"""Shared configuration. Every path is repository-relative: this repo is standalone."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LLM_RAW = DATA_PROCESSED / "llm_raw"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
DOCS = ROOT / "docs"

# --- Supplied source data, all inside this repository -------------------------
# RBA statistical tables (A2/F1/F2/G1/G3/H1/H3/H5) live directly in data/raw/.
DOCUMENTS = DATA_RAW / "rba_minutes_parsed.parquet"   # 211 minutes, parsed to text
MEETING_CALENDAR = DATA_RAW / "meeting_calendar.csv"  # one row per meeting date
MARKET_DATA = DATA_RAW / "market_data.parquet"        # ASX 200, VIX, AUD
GPR_FILE = DATA_RAW / "data_gpr_export.xls"           # geopolitical risk index

CONSTRUCT_SCORES = DATA_PROCESSED / "construct_scores.parquet"
# Development and validation are scored in separate runs over disjoint meetings, so each
# writes its own partial and CONSTRUCT_SCORES is rebuilt as the union of whichever exist.
CONSTRUCT_SCORES_DEV = DATA_PROCESSED / "construct_scores_dev.parquet"
CONSTRUCT_SCORES_VAL = DATA_PROCESSED / "construct_scores_validation.parquet"

# The frozen Cycle model card, shipped with the repository. Built by model_card.py from a
# FIXED instructor panel so that every team interrogates the same model in Cycle, whatever
# their own constructs later look like.
MODEL_CARD = OUTPUTS / "model_card"
FROZEN_PANEL = DATA_PROCESSED / "panel_frozen.parquet"
FROZEN_TIERS = DATA_PROCESSED / "tiers_frozen.json"

# --- Corpus window -----------------------------------------------------------
CORPUS_START = "2006-10-01"
CORPUS_END = None
# The FROZEN corpus extent, used by the input contracts: vendored data that stops short
# of the last meeting, or a calendar with a different meeting count, is a corrupt or
# truncated input, not a smaller corpus.
CORPUS_MEETINGS = 211
CORPUS_LAST_MEETING = "2026-06-16"

# --- LLM ---------------------------------------------------------------------
# The course model, served by the UNSW proxy (see courseapi.py). This is the ALIAS, not a
# dated snapshot: the proxy deploys `gpt-5.4-mini` only, so pinning a snapshot is no longer
# available to us. Two consequences the report must acknowledge rather than hide:
#   * the provider can re-point the alias mid-semester, and identical prompts would then
#     produce different behaviour;
#   * so the committed call envelopes, not the model name, are what make a run
#     reproducible. They are the authority; re-running is not.
# Do not change this mid-semester.
MODEL = "gpt-5.4-mini"
# Temperature must be > 0: the parallel calls exist to measure how much the model disagrees
# WITH ITSELF, and at temperature 0 that spread collapses to nothing.
#
# NOTE: temperature is accepted by this model ONLY while reasoning effort is unset or
# "none". Asking for both is rejected by the API, which is why no reasoning effort is
# requested anywhere in this repository.
SAMPLING_TEMPERATURE = 1.0
# CALL INDEX, not a random seed. The GPT-5 family removed `seed` and the API rejects it, so
# nothing here can pin the model's sampling. What this number still does - and the reason it
# survives - is distinguish the N parallel draws of one prompt from one another, in the
# cache key and in the audit trail. Call i is CALL_INDEX_BASE + i. Read it as "which draw",
# never as "which random state".
CALL_INDEX_BASE = 20260818
# Output ceiling per call. The largest response this assignment has ever needed was ~1,150
# tokens (a Shock branch); Words has never exceeded 400. It is sent on every call: the
# Responses API requires a budget, and an implicit default is one more thing the envelope
# could not explain.
#
# WHY 8,000. The value has moved twice and the reasoning is worth keeping, because both
# moves were evidence-driven and the first one was wrong.
#
# It was 4,000, then cut to 1,500 on the hypothesis that the proxy metered the RESERVED
# output rather than what was generated - which would have made a loose ceiling expensive.
# MEASURED 2026-09-11 AND IT DOES NOT: firing identical requests at the per-minute bucket
# with max_output_tokens of 64 and of 12,000 admitted the same volume of prompt tokens
# (~90k vs ~99k), where metering the reservation would have admitted about a quarter as
# many. The bucket counts the PROMPT, so headroom here is free.
#
# 1,500 then truncated a Shock branch mid-JSON. The largest response this stage has ever
# produced is 1,420 tokens - under gpt-4o-mini it was 1,148 - so 1,500 was inside the
# range of normal output, not above it. A truncated structured response fails validation
# outright rather than degrading quietly, which is the right behaviour and an expensive
# way to discover a tight ceiling.
#
# 8,000 is therefore chosen to be comfortably clear of anything observed. It costs nothing
# unused, and the response schemas bound what can come back in any case.
MAX_OUTPUT_TOKENS = 8000
N_PARALLEL_CALLS = 5
N_DOC_WORKERS = 8
MAX_RETRIES = 4                # bounded exponential backoff per call
RETRY_BASE_SECONDS = 2.0
MIN_VALID_CALLS = 3            # a document with fewer valid calls fails the run

# --- Seven constructs --------------------------------------------------------
TEXT_FEATURES = [
    "policy_stance",
    "inflation_concern",
    "downside_risk_emphasis",
    "financial_conditions_concern",
    "uncertainty_language",
    "vigilance",
    "global_risk_salience",
]

# Binned concentration gate. Scores are binned at BIN_WIDTH before the modal bin is found,
# so this is a concentration statistic and not the share at an exact value - the name says
# so deliberately. A construct above the threshold is not discriminating whatever its
# standard deviation says.
BIN_WIDTH = 0.05
MAX_BINNED_CONCENTRATION = 0.50
MIN_CONSTRUCT_SPREAD = 0.05
MIN_EFFECTIVE_BINS = 4.0        # entropy-equivalent bins actually used
MIN_SIGNAL_TO_NOISE = 1.0       # between-document variance / within-document variance

# --- Targets -----------------------------------------------------------------
CYCLE_WINDOW_DAYS = 182
CYCLE_THRESHOLD_PCT = 0.25
CYCLE_WINDOW_SWEEP = [91, 182, 273, 365]
CYCLE_THRESHOLD_SWEEP = [0.125, 0.25, 0.50]
CYCLE_STATES = {0: "easing", 1: "stable", 2: "hardening"}
DECISION_STATES = {-1: "cut", 0: "hold", 1: "hike"}

MINUTES_PUBLICATION_LAG_DAYS = 14
PUBLICATION_LAGS = {"cpi": 28, "gdp": 65, "labour": 16,
                    "inflation_expectations": 14, "activity": 30,
                    "gpr": 35}   # Caldara-Iacoviello GPR, monthly, ~1 month behind

# --- Models ------------------------------------------------------------------
REGIME_SEED = 20260818
MIN_TRAIN_MEETINGS = 80

for _d in (DATA_RAW, DATA_PROCESSED, LLM_RAW, OUTPUTS, FIGURES, DOCS):
    _d.mkdir(parents=True, exist_ok=True)


# --- Stage status and atomic writes -------------------------------------------
# A failed re-run must not hide behind an older artefact: each supplied runner marks its
# stage "running" at invocation and "complete" only after every output is committed, and
# the submission suite requires the LATEST attempt under the CURRENT configuration to
# have completed. Artefact writers go through atomic_write_* so a crash mid-stage cannot
# leave a half-written reportable file.

def stage_status_path(stage: str) -> Path:
    d = OUTPUTS / ".stage_status"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.json"


def _write_status(stage: str, status: str, config_hash: str) -> None:
    import json
    from datetime import datetime, timezone
    atomic_write_text(stage_status_path(stage), json.dumps(
        {"stage": stage, "status": status, "config_hash": config_hash,
         "at": datetime.now(timezone.utc).isoformat()}, indent=1))


def stage_begin(stage: str, config_hash: str) -> None:
    _write_status(stage, "running", config_hash)


def stage_complete(stage: str, config_hash: str) -> None:
    _write_status(stage, "complete", config_hash)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to a sibling temp file and os.replace() it into place - atomic on one
    filesystem, so readers never observe a partially written artefact. On any failure
    the temp file is removed and the previous artefact is untouched."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp-write")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_to_parquet(df, path: Path) -> None:
    import os
    tmp = path.with_suffix(path.suffix + ".tmp-write")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- Shared input validation --------------------------------------------------

def validate_construct_scores(df=None, reportable: bool = False):
    """
    THE one contract for construct-score tables, shared by the panel builder, Replay
    and Shock. Loads CONSTRUCT_SCORES when df is None. Always checks: the meeting_date
    column exists (by name, not a KeyError), the table is non-empty, dates are unique,
    all seven constructs AND all seven sd columns are present, scores are finite in
    [0, 1], sd values are finite and non-negative, and n_calls_valid is a finite whole
    number in [MIN_VALID_CALLS, N_PARALLEL_CALLS].
    With reportable=True additionally requires EXACTLY the authoritative meeting set
    from the calendar - a reportable Replay/Shock run must fail here, before producing
    an artefact, not later in the submission suite.
    """
    import numpy as np
    import pandas as pd
    if df is None:
        if not CONSTRUCT_SCORES.exists():
            raise RuntimeError(
                "construct_scores.parquet is missing - run the Words stage "
                "(development AND --validate) first")
        df = pd.read_parquet(CONSTRUCT_SCORES)
    problems = []
    if "meeting_date" not in df.columns:
        raise RuntimeError(
            "construct scores have no 'meeting_date' column - the table is not a "
            "construct-score file")
    if not len(df):
        raise RuntimeError(
            "the construct-score table exists but is EMPTY - a zero-row score file is "
            "corrupt, not a fresh checkout; re-run the Words stage")
    md = pd.to_datetime(df["meeting_date"])
    if md.duplicated().any():
        problems.append("duplicated meeting dates")
    missing_cols = [c for c in TEXT_FEATURES if c not in df.columns]
    if missing_cols:
        problems.append(f"missing constructs: {missing_cols}")
    missing_sd = [f"{c}_sd" for c in TEXT_FEATURES if f"{c}_sd" not in df.columns]
    if missing_sd:
        problems.append(f"missing sd columns: {missing_sd}")
    have = [c for c in TEXT_FEATURES if c in df.columns]
    if have:
        v = df[have].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            problems.append("non-finite construct scores")
        elif v.min() < 0.0 or v.max() > 1.0:
            problems.append(f"construct scores outside [0, 1] "
                            f"(min {v.min():.3f}, max {v.max():.3f})")
        sd_cols = [f"{c}_sd" for c in have if f"{c}_sd" in df.columns]
        if sd_cols:
            sd = df[sd_cols].to_numpy(dtype=float)
            if not np.isfinite(sd).all() or (sd < 0).any():
                problems.append("score sd columns must be finite and non-negative")
        if "n_calls_valid" in df.columns:
            ncv = df["n_calls_valid"].to_numpy(dtype=float)
            if not np.isfinite(ncv).all() or (ncv != np.round(ncv)).any():
                problems.append("n_calls_valid must be a finite whole number of "
                                "calls per meeting")
            elif ((ncv < MIN_VALID_CALLS) | (ncv > N_PARALLEL_CALLS)).any():
                problems.append(f"n_calls_valid outside "
                                f"[{MIN_VALID_CALLS}, {N_PARALLEL_CALLS}]")
        else:
            problems.append("n_calls_valid column missing")
    if reportable:
        cal = pd.read_csv(MEETING_CALENDAR, parse_dates=["meeting_date"])
        expected = set(pd.to_datetime(cal["meeting_date"]))
        got = set(md)
        if got != expected:
            problems.append(
                f"a reportable run needs EXACTLY the {len(expected)} authoritative "
                f"meetings scored; missing {len(expected - got)}, "
                f"unexpected {len(got - expected)} - run BOTH Words passes "
                f"(development and --validate)")
    if problems:
        raise RuntimeError(f"construct scores failed their contract: {problems}")
    return df


def validate_envelope_entry(stage: str, entry: dict, root=None) -> None:
    """
    THE one rule for a ledgered envelope, used by the submission suite and testable on
    its own. The file must exist, hash-match the ledger, and its actual success state
    must EQUAL the recorded expected state; a recorded failure is permitted only for
    Replay's bounded benchmark attrition (rec-* envelopes) - statement, audit,
    deep-replay and every other stage's calls must have succeeded.
    """
    import hashlib
    import json
    root = Path(root) if root else ROOT
    f = root / entry["path"]
    if not f.exists():
        raise RuntimeError(f"{stage}: envelope {entry['path']} referenced by the "
                           f"artefact is missing")
    data = f.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(
            f"{stage}: envelope {entry['path']} changed since the artefact was "
            f"produced - a later run overwrote it; re-run the stage")
    expected_ok = entry.get("ok", True)
    j = json.loads(data)
    if isinstance(j, dict):
        if bool(j.get("ok")) != expected_ok:
            raise RuntimeError(
                f"{stage}: {entry['path']} success state no longer matches what the "
                f"artefact consumed")
        if not expected_ok and not (stage == "replay"
                                    and f.name.startswith("rec-")):
            raise RuntimeError(
                f"{stage}: {entry['path']} is a failed call in a role that does not "
                f"tolerate failure")
    else:
        n_ok = sum(1 for c in j if c.get("ok"))
        if n_ok < MIN_VALID_CALLS:
            raise RuntimeError(f"{stage}: {entry['path']} has only {n_ok} valid calls")


# --- Envelope ledger ----------------------------------------------------------
# Which cached calls did each submitted artefact actually rest on? The supplied call
# layers record every envelope they serve or write; the runner commits the list, with a
# SHA-256 per file, next to the stage status. The submission suite then requires every
# referenced envelope to exist, hash-match, and MATCH the success/failure state the
# ledger recorded - failures are permitted only for Replay's bounded benchmark
# attrition. A same-configuration re-run that overwrote an envelope (including a
# failure envelope replaced by a success) is detected even though the old artefact and
# its configuration hash still agree.

_ENVELOPE_LEDGERS: dict = {}


def ledger_reset(stage: str) -> None:
    _ENVELOPE_LEDGERS[stage] = {}


def ledger_add(stage: str, path, ok: bool = True) -> None:
    """Record a consumed call envelope and its EXPECTED success state. Tolerated
    failures (Replay's bounded benchmark attrition) are ledgered too - a failure that
    shaped the usable-call denominator is part of what the artefact rests on, and a
    later run replacing it with a success must invalidate the old artefact."""
    _ENVELOPE_LEDGERS.setdefault(stage, {})[str(Path(path).resolve())] = ok


def ledger_commit(stage: str, name: str | None = None) -> None:
    import hashlib
    import json
    ledger = _ENVELOPE_LEDGERS.get(stage, {})
    entries = []
    for p in sorted(ledger):
        fp = Path(p)
        rel = fp.relative_to(ROOT).as_posix() if fp.is_relative_to(ROOT) else str(fp)
        entries.append({"path": rel, "ok": ledger[p],
                        "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()})
    n_failed = sum(1 for e in entries if not e["ok"])
    atomic_write_text(
        OUTPUTS / ".stage_status" / f"{name or stage}.envelopes.json",
        json.dumps({"stage": name or stage, "n": len(entries),
                    "n_failed": n_failed, "entries": entries}, indent=1))

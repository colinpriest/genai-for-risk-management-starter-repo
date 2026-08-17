"""Shared configuration. Everything that must be identical across stages lives here."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
LLM_RAW = DATA_PROCESSED / "llm_raw"
OUTPUTS = ROOT / "outputs"

RBA_MINUTES_DIR = DATA_RAW / "rba-minutes"
GPR_FILE = DATA_RAW / "geopolitical" / "data_gpr_export.xls"

# --- Corpus window -----------------------------------------------------------
# Minutes run 2006-10-03 to 2026-06-16. Narrow this if you justify it in your report.
CORPUS_START = "2006-10-01"
CORPUS_END = None                      # None = everything available

# --- Market data -------------------------------------------------------------
MARKET_TICKER = "^AXJO"                # ASX 200
MARKET_START = CORPUS_START

# --- Model -------------------------------------------------------------------
# The course provides GPT-4o-mini only. The reference implementation used GPT-4o.
# Section 4 of the brief: this difference is part of what you must discuss.
MODEL = "gpt-4o-mini"                  # the only model available on this course

# TEMPERATURE MUST BE > 0.
#   The N parallel calls exist to measure how much the model disagrees WITH ITSELF.
#   At temperature 0 the calls come back near-identical, the measured spread collapses to
#   about zero, and the "LLM uncertainty" layer reports near-perfect confidence on every
#   document no matter how ambiguous it is. The layer becomes decorative.
#   Sampling temperature plus a DIFFERENT FIXED SEED per call gives genuine variation and
#   still reproduces on a best-effort basis. NOT exactly, and not forever: the
#   provider can change model weights or routing without notice, and nothing in
#   this repository controls that. The RAW SAVED RESPONSES are the artefact that
#   cannot change underneath you - which is why they are committed.
SAMPLING_TEMPERATURE = 1.0
SEED = 20260817                        # call i uses SEED + i
N_PARALLEL_CALLS = 10                  # matches the reference implementation

# Concurrency. N_DOC_WORKERS documents are processed at once and each starts N_PARALLEL_CALLS
# calls, so without a global cap you would have N_DOC_WORKERS x N_PARALLEL_CALLS = 100 requests
# in flight. MAX_CONCURRENT_CALLS is the real limit; lower it if you hit rate limits on the
# shared course key.
N_DOC_WORKERS = 10
MAX_CONCURRENT_CALLS = 20

# --- Regime model ------------------------------------------------------------
# THREE regimes, the same as the reference implementation.
#
# The reasons, in the order they actually carry weight:
#
# 1. COMPARABILITY. Section 4 of the brief asks you to compare your results against the
#    reference program's published results. Regime shares, per-regime risk metrics and
#    expected durations are NOT comparable across different regime counts - a "high
#    volatility" state means something different when it is one of three than one of four.
#    Holding the count equal removes a confound from the one comparison the assignment is
#    built around.
#
# 2. RELIABLE CONVERGENCE. Across the optimiser starts in stage 4 the 3-regime model
#    converged on nearly all of them. The 4-regime model converged on about half, and the
#    ones that did converge landed on visibly different optima. You run this once, and a fit
#    that depends on which start it drew is not an instrument you can report from.
#
# 3. PARSIMONY AND INTERPRETABILITY. Three states map onto language a risk committee already
#    uses. A fourth state has to be characterised, named and defended, and on this data it is
#    not clear it separates two economically different states rather than splitting one.
#
# A SENSITIVITY RESULT, WHICH IS NOT A REASON FOR THE CHOICE. The measured out-of-sample
# contribution of the text features is LARGER at three regimes than at four, which is
# unsurprising: extra regimes absorb volatility variation the constructs would otherwise
# have to explain. Fit both and quote YOUR OWN numbers - the size of the gap depends on
# which features you include and on how many rows they cost you, so any figure quoted here
# would not match what you get. Report it as a robustness observation.
#
# But do NOT choose the specification because it makes your features look better. Picking the
# model that maximises the apparent contribution of the thing you are advocating is
# outcome-driven specification selection, and it is exactly the practice this course teaches
# you to audit in other people's work. The regime count is settled by 1-3 above; the
# difference in measured contribution is something you observe afterwards and disclose.
N_REGIMES = 3

# YOURS TO NAME. These are deliberately neutral placeholders - a regime cannot be named before
# it has been fitted and inspected.
#
# Regimes are ordered automatically from LOWEST to HIGHEST realised volatility, so 0 is always
# your calmest state and N-1 your most violent. That ordering is done for you and is stable.
#
# After running stage4, look at each regime's share of days, annualised volatility, Expected
# Shortfall, and which historical periods it covers. Then replace these with names a risk
# committee would understand, and use them consistently in your dashboard and report.
REGIME_NAMES = {0: "regime_0", 1: "regime_1", 2: "regime_2"}

# The dependent variable for the regime model. THIS CHOICE MATTERS MORE THAN ANY OTHER.
#   "log_rv"  - log realised volatility. Exogenous text features enter the mean equation,
#               which IS the volatility level, so text can explain volatility.
#   "returns" - daily returns. Exogenous features then predict the DIRECTION of returns,
#               which text cannot do. Text will appear useless. It is not; the spec is.
REGIME_ENDOG = "log_rv"

# --- Macro / cross-asset series ----------------------------------------------
# YOURS TO CHOOSE. name -> yfinance ticker. These are downloaded by stage1 and become
# <name>, <name>_ret and <name>_vol_21. To make one actually enter the model you must also
# name it in MACRO_FEATURES below.
#
# The three below are a starting point, not an answer. Australia is a commodity exporter with
# a floating currency - argue from that when you decide what to add or drop.
MACRO_TICKERS = {
    "aud": "AUDUSD=X",       # trade-weighted proxy; falls when risk appetite falls
    "vix": "^VIX",           # global risk appetite
    "iron": "TIO=F",         # iron ore; Australia's largest single export. NOTE: this series
                             # starts much later than the ASX data - about 1,000 trading days
                             # are missing. Putting it in MACRO_FEATURES will drop those rows
                             # from the fit. stage1 prints missingness and stage4 reports how
                             # many rows a feature costs you. Check both before committing.
}

# Days to lag EVERY macro series, to respect publication timing. Daily market prices (FX,
# VIX, futures) are known same-day and need 0. An ABS statistic is published weeks after the
# period it describes, and using it unlagged is look-ahead. If you add a released statistic
# rather than a market price, raise this and say so in your report.
MACRO_LAG_DAYS = 0

# WHICH macro columns actually ENTER THE REGIME MODEL. Downloading a series is not the same
# as using it: a ticker in MACRO_TICKERS that is not named here is decoration - it cannot
# affect any result. Section 8.1 of the brief requires your macro choices to enter the model.
#
# Names must be COLUMNS of market_data.parquet, so <name>, <name>_ret or <name>_vol_21 - for
# example "vix", "aud_vol_21", "aud_ret". Leave empty only if you intend to fit a text-only
# model and to say so.
#
# TWO WARNINGS BEFORE YOU FILL THIS IN.
#
# 1. Do NOT include rv_21 (or rv_5, rv_63, parkinson_21) when REGIME_ENDOG = "log_rv". Those
#    ARE the dependent variable, or a near-copy of it, and regressing a variable on itself
#    produces a spectacular fit that means nothing.
#
# 2. Volatility-based features such as vix and aud_vol_21 are far stronger predictors of the
#    volatility regime than any language measure, and they WILL swamp the text. That is a
#    legitimate model and a legitimate finding - but it is the reason stage 4 fits four nested
#    models and reports the text's CONDITIONAL contribution (on top of macro) separately from
#    its MARGINAL one. Quote the conditional number. A construct with a big marginal
#    contribution and a negligible conditional one is tracking volatility that VIX already
#    told you about, and saying so is worth more than hiding it.
#
# Series coverage differs: aud_vol_21 is missing about 5% of days and dropping those rows
# costs you sample. stage1 prints missingness and stage4 reports the rows each feature costs.
MACRO_FEATURES: list[str] = []

# Days between a meeting and the publication of its minutes. The RBA releases minutes about
# a fortnight after the meeting, so a score attached to the meeting date is information
# nobody had at the time. 14 keeps the model honest for a PREDICTIVE claim; 0 is only
# defensible for a RETROSPECTIVE one, and you must say which you are making.
PUBLICATION_LAG_DAYS = 14

# Between-document spread below which a construct is not discriminating. ONE definition,
# used by stage3's report and by tests/test_contracts.py, which previously disagreed (0.10
# against 0.02).
#
# SPREAD IS NOT VALIDITY. A construct can have a wide spread and still measure the wrong
# thing - you can always widen spread by writing a more extreme prompt. This threshold only
# catches the failure where every document scores the same. Judging whether the construct
# measures what you claim needs the human agreement work in section 8.2.
MIN_CONSTRUCT_SPREAD = 0.05

# --- Embeddings (local, no API cost) -----------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

for _d in (DATA_INTERIM, DATA_PROCESSED, LLM_RAW, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

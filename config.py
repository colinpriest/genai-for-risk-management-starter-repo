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
#   still reproduces exactly.
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
# Four regimes, not the reference's three. Whether the fourth regime earns its place on YOUR
# data is a finding you have to produce: stage4 fits three and four and reports AIC for both.
N_REGIMES = 4

# YOURS TO NAME. These are deliberately neutral placeholders - a regime cannot be named before
# it has been fitted and inspected.
#
# Regimes are ordered automatically from LOWEST to HIGHEST realised volatility, so 0 is always
# your calmest state and N-1 your most violent. That ordering is done for you and is stable.
#
# After running stage4, look at each regime's share of days, annualised volatility, Expected
# Shortfall, and which historical periods it covers. Then replace these with names a risk
# committee would understand, and use them consistently in your dashboard and report.
REGIME_NAMES = {0: "regime_0", 1: "regime_1", 2: "regime_2", 3: "regime_3"}

# The dependent variable for the regime model. THIS CHOICE MATTERS MORE THAN ANY OTHER.
#   "log_rv"  - log realised volatility. Exogenous text features enter the mean equation,
#               which IS the volatility level, so text can explain volatility.
#   "returns" - daily returns. Exogenous features then predict the DIRECTION of returns,
#               which text cannot do. Text will appear useless. It is not; the spec is.
REGIME_ENDOG = "log_rv"

# --- Macro / cross-asset series ----------------------------------------------
# YOURS TO CHOOSE. name -> yfinance ticker. These are downloaded by stage1 and become
# <name>, <name>_ret and <name>_vol_21. To make one actually enter the model you must also
# name it in stage4.MACRO_FEATURES.
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

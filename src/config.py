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

# --- LLM ---------------------------------------------------------------------
MODEL = "gpt-4o-mini"
# Temperature must be > 0: the parallel calls exist to measure how much the model disagrees
# WITH ITSELF, and at temperature 0 that spread collapses to nothing.
SAMPLING_TEMPERATURE = 1.0
SEED = 20260818                # call i uses SEED + i
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

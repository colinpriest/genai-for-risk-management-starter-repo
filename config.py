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
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0                      # RECORD any change; it affects reproducibility
SEED = 20260817                        # pass to the API where supported
N_PARALLEL_CALLS = 10                  # matches the reference implementation

# --- Regime model ------------------------------------------------------------
N_REGIMES = 3                          # FIXED. Do not change.

# --- Embeddings (local, no API cost) -----------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

for _d in (DATA_INTERIM, DATA_PROCESSED, LLM_RAW, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

# RBA Volatility Regime Detection — Starter Repository

RISK5110 group assignment. Adapt the US reference implementation to Australian data.

**Reference:** https://github.com/colinpriest/financial-market-volatility-regime-detection

---

## Why this looks different from the reference

The reference is a single 147 KB Python file. That works for one author and fails badly for a team
of four — every merge is a conflict and nobody owns anything.

This starter has **the same pipeline**, split into one module per stage so each stage has an owner
and a defined hand-over. The stage order, the methods and the outputs are unchanged.

| Reference | Here |
|---|---|
| `financial_regime_detection.py` (all stages) | `src/stage1..stage5` |
| `.cache/fomc_minutes_*.txt` | `data/raw/rba-minutes/*.html` |
| `dashboard.html` | `outputs/dashboard.html` |

---

## What is pre-built, and what is yours

The Australian customisations are done for you. Your effort should go into judgement and design,
not into re-deriving plumbing.

| Pre-built and working | Yours |
|---|---|
| ASX 200 + macro download, volatility features (`stage1`) | Which macro series (`config.MACRO_TICKERS`) and **which of them enter the model** (`stage4.MACRO_FEATURES`) |
| RBA HTML parsing, chunking, retrieval (`stage2`) | **The retrieval query** — which parts of a document matter |
| Parallel calling, retries, raw saving, averaging (`stage3`) | **The prompts.** All six field descriptions are blank |
| Best-converged fitting, regime ordering, risk metrics, coefficients, transitions, rolling-origin out-of-sample (`stage4`) | Which features to include; interpreting the dependent-variable comparison |
| — | **`stage5` is a stub.** All three uncertainty layers are yours: block bootstrap, specification ensemble, LLM spread. Traps 1, 4 and 6 all bite here. |
| Perturbation + faithfulness harness, patterns verified against the RBA corpus | Running it and interpreting the result |

**Read `TRAPS.md` first.** Six issues that cost hours and teach nothing, with fixes.

## Two decisions already made for you, and why

**Four regimes, not three.** The reference used three on US data. `stage4` fits three *and* four
and reports AIC for both, so whether the fourth regime earns its place on your data is something
you measure and report — not something we tell you. `REGIME_NAMES` ships as neutral placeholders
for the same reason: naming the regimes is your work, done after you have inspected them.

**Sampling temperature is 1.0, not 0.** The parallel calls exist to measure how much the model
disagrees with itself. At temperature 0 they come back identical and that measurement is
meaningless. Each call uses a different fixed seed, so runs still reproduce.

## Setup

**Python 3.12.** Other versions may work; 3.12 is what this was tested on.

**Windows (PowerShell or Git Bash):**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run_pipeline.py --check
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_pipeline.py --check
```

Then put your API key in `.env`. `--check` verifies setup and makes no API calls.

**Your API key goes in `.env`, which is git-ignored. Never commit it.**
Run `python run_pipeline.py --check` before your first commit — it will tell you if a key is
about to be exposed.

---

## Layout

```
config.py              Shared settings. Paths, model name, corpus window.
run_pipeline.py        Phases: --all --core --calibration --scenarios --stakeholders
                       --dashboard --stage N --check. Start here to see the flow.

src/
  stage1_market_data.py    ASX 200 + macro indicators        OWNER: ?
  stage2_documents.py      RBA minutes + retrieval           OWNER: ?
  stage3_riskvoice.py      10 parallel calls -> 5 constructs OWNER: ?
  stage4_regime_model.py   Markov switching, 4 regimes and 3 OWNER: ?
  stage5_uncertainty.py    3 uncertainty layers   *** STUB *** OWNER: ?
  dashboard.py             Plotly output                     OWNER: ?
  scenarios.py             Geopolitical scenarios            SHARED

harness/                 SUPPLIED - DO NOT MODIFY
  perturbation.py          Text perturbation experiments
  faithfulness.py          Does the model's stated reason drive its behaviour?

contracts/               One file per hand-over between stages
tests/                   test_contracts.py checks stage outputs match their contracts
data/raw/rba-minutes/    211 RBA minutes, Oct 2006 - Jun 2026 (supplied)
data/processed/          Stage outputs. Git-ignored except .gitkeep.
outputs/                 dashboard.html and figures
```

**Put your name in the `OWNER:` line at the top of your stage file.** It is the first thing marked.

---

## The harness is supplied — run it, do not rebuild it

`harness/` contains the perturbation and faithfulness tools for section 8.3 of the brief. You run
them and interpret the results. Rebuilding them is not worth marks; interpreting them is.

Both take **your** functions as arguments, so they work with whatever you build in stage 3:

```python
from harness.perturbation import perturbation_sweep
from harness.faithfulness import faithfulness_test

from src.stage3_riskvoice import score_field

# score_field averages several calls, which trap 5 requires - a single call is too noisy
scorer = lambda t: score_field(t, "financial_conditions_concern", n=5)

result = perturbation_sweep(score_fn=scorer, text=doc, edits=DEFAULT_EDITS)
fx     = faithfulness_test(score_fn=scorer, explain_fn=my_explainer, text=doc)
```

---

## Interface contracts

`contracts/` holds one markdown file per hand-over. Agree them in Week 2 and commit them **before**
you write the code they describe.

`tests/test_contracts.py` turns each contract into a test. A prose contract gets violated silently;
a tested one fails at the boundary and names the stage that broke it.

```bash
pytest tests/                 # development: missing outputs SKIP
pytest tests/ --submission    # before you submit: missing outputs FAIL
```

**Run `pytest tests/ --submission` before you hand in.** In development mode a missing stage
output is skipped, so an almost-empty repository reports "all tests passed". Submission mode
turns every missing required artefact into a failure, and also checks that your prompts are
written, your regimes are named, and your blind agreement labels exist.

---

## Data

**Supplied:** `data/raw/rba-minutes/` — 211 RBA minutes, 3 Oct 2006 to 16 Jun 2026.

> **Source: Reserve Bank of Australia.** Licensed **CC BY 4.0**. You must attribute the RBA in your
> report and on any chart built from this corpus — see [DATA-LICENCE.md](DATA-LICENCE.md) for the
> exact wording and for two exclusions that apply to ABS data and the Cash Rate.

**Supplied:** `data/raw/geopolitical/` — GPR index (Caldara & Iacoviello).

**You collect:** ASX 200 via yfinance, and Australian macro indicators of your choosing.

---

## Before you commit

- [ ] `.env` is not staged — `python run_pipeline.py --check` verifies this properly
- [ ] Raw API responses saved under `data/processed/llm_raw/<prompt-fingerprint>/`
- [ ] Temperature and seed recorded in `config.py`
- [ ] `pytest tests/` passes
- [ ] Your name is in the `OWNER:` line of your stage

## Before you SUBMIT

- [ ] `pytest tests/ --submission` passes
- [ ] Regimes renamed from `regime_0` … in `config.py`
- [ ] Fixed market-data copy committed under `data/snapshot/`
- [ ] Blind agreement labels committed under `data/processed/agreement/`
- [ ] You have said whether your model is retrospective or predictive
      (`config.PUBLICATION_LAG_DAYS`)

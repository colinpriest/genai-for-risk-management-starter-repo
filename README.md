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

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env            # then put your API key in .env
python run_pipeline.py --check    # verifies setup without calling the API
```

**Your API key goes in `.env`, which is git-ignored. Never commit it.**
Run `python run_pipeline.py --check` before your first commit — it will tell you if a key is
about to be exposed.

---

## Layout

```
config.py              Shared settings. Paths, model name, corpus window.
run_pipeline.py        Runs stages in order. Start here to see the flow.

src/
  stage1_market_data.py    ASX 200 + macro indicators        OWNER: ?
  stage2_documents.py      RBA minutes + retrieval           OWNER: ?
  stage3_sentiment.py      10 parallel calls -> scores       OWNER: ?
  stage4_regime_model.py   3-regime Markov switching         OWNER: ?
  stage5_uncertainty.py    3 uncertainty layers              OWNER: ?
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

`harness/` contains the perturbation and faithfulness tools for section 5.3 of the brief. You run
them and interpret the results. Rebuilding them is not worth marks; interpreting them is.

Both take **your** functions as arguments, so they work with whatever you build in stage 3:

```python
from harness.perturbation import perturbation_sweep
from harness.faithfulness import faithfulness_test

result = perturbation_sweep(score_fn=my_sentiment_scorer, text=doc, edits=DEFAULT_EDITS)
fx     = faithfulness_test(score_fn=my_sentiment_scorer, explain_fn=my_explainer, text=doc)
```

---

## Interface contracts

`contracts/` holds one markdown file per hand-over. Agree them in Week 2 and commit them **before**
you write the code they describe.

`tests/test_contracts.py` turns each contract into a test. A prose contract gets violated silently;
a tested one fails at the boundary and names the stage that broke it.

```bash
pytest tests/            # run after any stage produces output
```

---

## Data

**Supplied:** `data/raw/rba-minutes/` — 211 RBA minutes, 3 Oct 2006 to 16 Jun 2026.
Source: Reserve Bank of Australia. CC BY 4.0.

**Supplied:** `data/raw/geopolitical/` — GPR index (Caldara & Iacoviello).

**You collect:** ASX 200 via yfinance, and Australian macro indicators of your choosing.

---

## Before you commit

- [ ] `.env` is not staged
- [ ] Raw API responses saved under `data/processed/llm_raw/`
- [ ] Temperature and seed recorded in `config.py`
- [ ] `pytest tests/` passes
- [ ] Your name is in the `OWNER:` line of your stage

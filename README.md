# RISK5110 Group Assignment — starter repository

> **The question:** how should GenAI output be generated, tested, grounded, challenged and
> overruled in a consequential analytical workflow?

The case study is Australian monetary policy — can we tell whether the RBA is moving into an
easing, stable or hardening phase, and what would the Board do about a shock that has never
happened? **You are marked on what you checked, not on what the models produced.**

Read **the assignment brief on Moodle** first — then [TRAPS.md](TRAPS.md) before you
write any code.

This repository is published at
<https://github.com/colinpriest/genai-for-risk-management-starter-repo>; the ZIP on Moodle
is the same release. **Your work does not go back there** — you cannot push to it, and a
public fork publishes your team's answers. Push this into your own **private** repository
as its first commit and add the teaching accounts listed on Moodle (brief §7).

## Setup

```bash
pip install -r requirements.txt -c constraints.txt
```

```bash
cp .env.example .env
```

Two commands prove your environment is correct. Neither needs an API key.

```bash
python src/cycle_model.py --show
```

That opens the supplied model card. It should print a label-embargoed accuracy of **0.570**
against a `current_decision` baseline of **0.656** — and say, in as many words, that the model
does not beat it. That is not a broken install. **It is the Cycle stage's subject matter**, and
the same summary prints the leaky 0.805 the model scores when the embargo is removed, so you
can see exactly what the difference is made of. It reads artefacts that ship with this
repository, so it works before you have built anything.

```bash
python src/data_panel.py
```

That should print 211 meetings and both target distributions, and takes about ten seconds.
If it does, every data-plumbing problem in this assignment is already solved for you.

Then run the contract tests:

```bash
python -m pytest tests/ -q
```

**If an import fails, check the install first.** This one is fast and offline, and tells
you whether every package the code imports is actually declared:

```bash
python -m pytest tests/test_dependencies.py -q
```

There is also a slower end-to-end check that copies the repository somewhere clean and runs
it from scratch. Worth running once before you submit:

```bash
python -m pytest tests/test_clean_checkout.py -q -m clean_install
```

And a genuine install test, which builds a fresh virtual environment and installs only the
declared requirements. It downloads packages, so it is opt-in:

```bash
python -m pytest tests/test_dependencies.py -q -m slow_install
```

That suite **passes on this untouched starter by design** — it proves the environment, not
your work. The repository-and-pipeline completeness check — run it before you submit; it
verifies everything that lives in the repository, and only that (the transcript, AI-use
log, presentation and peer form are submitted separately):

```bash
python -m pytest tests/test_submission.py -q -m submission
```

Every failure names a missing piece; it needs no API key, because the committed caches and
artefacts are the submission.

## What is already built

| File | What it gives you |
|---|---|
| `src/data_rates.py` | RBA A2/F1/F2 — the decision series and the daily rates panel |
| `src/data_macro.py` | CPI, expectations, labour, GDP, activity, geopolitical risk — publication-lagged |
| `src/data_panel.py` | Meeting calendar, both targets, the tiered panel |
| `src/model_card.py` | The frozen cycle model: importances with intervals, partial dependences (tables **and** plots), regime probabilities, plus the construct-free model Shock loads |
| `src/evaluation.py` | Rolling origin **with a label-availability embargo**, scoring, four baselines |
| `src/attribution.py` | Permutation importance on held-out data, staleness-cost refit |
| `src/causal_tests.py` | Lead–lag, reverse regression, conditioning, sub-period stability |
| `src/nshot.py` | Four shot-selection strategies, with three leakage checks that raise |
| `src/context_docs.py` | Loads the Shock context documents you download; retrieves passages with page citations |
| `src/channels.py` | Nine-channel taxonomy and the measured shock calibration table |
| `src/scenario_engine.py` | `Branch`/`Tree` records, shock adjudication, support checking, sensitivity |

Rewriting these earns no marks.

## What you write

**Across the whole assignment you supply four kinds of thing: LLM prompts, model and
strategy choices, adjudications, and interpretation.** Everything else is built.

| File | Stage | What is blank |
|---|---|---|
| `src/cycle_model.py` | Cycle | The causal claims and your verdicts |
| `src/text_features.py` | Words | The system prompt and **four of the seven** construct rubrics |
| `src/decision_replay.py` | Replay | The two prompts, the meeting, k, the strategy and **`CLAIM_REVIEWS`** |
| `src/scenarios.py` | Shock | The four prompts, `PROXY_JUDGEMENTS`, `COHERENCE`, `PATHWAY_ADJUDICATIONS`, **`CHANNEL_REVIEWS`** (see below), **`DIRECTION_WEIGHTS`** and their reasons, **`ADVERSARIAL_RESPONSES`** and `EXPECTED_PROFILES` |

**You write no control flow.** Every stage's runner is supplied — `python src/decision_replay.py`
and `python src/scenarios.py` validate your prompts, tables and reviews and execute them,
refusing to write an artefact from any failed or unreviewed state. Your work is the
prompts, the choices, the reviews and the judgement they encode — the parts that are
marked — not pipeline plumbing.

Stubs raise `NotImplementedError`. `text_features.py` and `scenarios.py` additionally refuse
to run until their prompts are written, so you cannot spend API credit on placeholders.

**The order the runners expect**, because two of them have a cheap mode you should use
first:

```bash
python src/text_features.py --pilot     # 25 fixed documents - iterate on rubrics here
python src/text_features.py             # the frozen development pass, once
python src/text_features.py --validate  # the held-out episodes, once

python src/decision_replay.py --dev     # choose a strategy, then FREEZE it
python src/decision_replay.py           # the full stage, including the holdout

python src/scenarios.py --discover      # the review template for your own scenarios
python src/scenarios.py                 # the full stage, after the reviews are written
```

`--offline` works on any Words command and replays committed envelopes without calling the
service. The Replay holdout **refuses to run** if the prompts, `k` or the strategy have
changed since `--dev` froze them: re-freeze and declare the second exposure, or restore
what you froze.

Three of the seven Words rubrics are **supplied as locked exemplars**, one per scale type.
They are hash-checked and the run aborts if they are edited. If you believe one is genuinely
defective on your data, report it — a versioned replacement will be issued.

## Before Shock

`data/raw/context/` is **empty and you fill it** — several Shock sources are free to read but
not redistributable, so the course cannot ship them. Download at least three, from at least
three organisations, then:

```bash
python src/context_docs.py
```

The code does not look for particular filenames. It reads whatever supported files are in the
folder, and tags every retrieved passage with its source file and page so your citations can
be checked. See [`data/raw/context/README.md`](data/raw/context/README.md).

## The channel review, which is most of the Shock stage

A channel makes two claims and they are evidenced differently. The **global** leg — the
scenario causes some world effect — can rest on a retrieved passage. The **Australian** leg —
that world effect reaches an Australian proxy and then the Board — cannot: these are global
risk reports and they contain almost nothing about Australian monetary transmission. **Both
legs are required before a channel may move a number**, and where the scenario narrative
itself stipulates the world effect you record `global_verdict: "scenario_fact"` rather than
inventing a citation for a premise you were handed.

You review **mechanism groups**, keyed `channel_type | direction | horizon | proxy`, not
individual branches — and each verdict names the ONE **canonical** branch it binds; the
code bars the rest as restatements. Where the scenario narrative itself stipulates the
world effect, cite a **registered fact id** from `SCENARIO_FACTS` — verified verbatim
against the narrative, so the fact is always real — **and write its `fact_link_reason`**,
your signed sentence on why that fact supports this claim; the machine checks provenance,
a person signs entailment, and the run blocks without both. To find the groups
for the ASSESSED scenarios:

```bash
python src/scenarios.py --discover
```

It runs both assessed scenarios with the review requirements off, stops before any number,
prints every group key and the required pruning audit, and writes
`outputs/channel_reviews.template.json` with each member branch's citation and quotation
inline. (`--worked` demonstrates the same workflow on the unassessed migration scenario —
it cannot produce the assessed keys.) Read the pages the branches cite, then enter the
completed records — the recommended route is the JSON files (copy the template to
`data/processed/channel_reviews.json` and fill it in place; pruning audits go in
`data/processed/pruning_reviews.json`), with the `CHANNEL_REVIEWS`/`PRUNING_REVIEWS`
Python dicts as a supported alternative — and rerun. The full field list and a
complete example sit above the SUPPLIED line in `src/scenarios.py`.

## Four rules

**1. Every accuracy claim must come through the SUPPLIED protocol for that kind of claim.**
A random split or k-fold leaks the future into the past on this data, and costs **20 marks**
— but so does training on labels that had not yet resolved, or choosing a strategy on the
sample you then report.

There is more than one protocol because the stages predict different things:

- **Cycle** classifier accuracy → `evaluation.rolling_origin()`, with the label-availability
  embargo. It refuses to run without one.
- **Replay** strategy accuracy → the supplied paired benchmark `decision_replay.evaluate_all()`,
  with its per-target cut-offs. Choose on `dev_sample()`, freeze with
  `python src/decision_replay.py --dev`, then report on `holdout_sample()`. It is not a
  rolling-origin fit, and re-deriving it as one is wrong, not safer.

The model card and the Shock model are both fitted on the **129 meetings whose labels had
resolved by 2018-12-31**, not on the full sample — a partial dependence has no out-of-sample
analogue, and a scenario has no date to roll forward from, so neither is a rolling-origin
object. Using them as designed is correct. Quoting either *as evidence of predictive skill*
is what the gate is for.

**2. Never shock a policy-outcome variable.** The cash rate, the decision and the trailing
rate-change measures *are* the policy stance. A channel whose proxy is `cash_rate` asserts
its own conclusion, and the model will agree with it. `scenario_engine` refuses them.

**3. Say which starting state you shocked.** The model's unshocked probability at the last
meeting is 0.93 on easing, which leaves 0.07 for any shock to move. `BASE_ROWS` ships with two
declared starting states and every scenario is reported from both — a shock that looks
negligible from one can move the model two-thirds of the way across the space from the other.

**4. Out of support, the model says nothing at all — and "in support" is calibrated.** Support
is judged against the 129 meetings the model was fitted on, at the 99th percentile of how
unusual a REAL unseen meeting is on both a marginal and a joint test. An absolute "any variable
outside its range" rule would fail all 82 held-out meetings, so it measures nothing. When a
shocked row leaves the range the
model was fitted on, `run_scenario()` returns `model_verdict = "out_of_support"` with **no
probability and no direction** — the argmax of an extrapolated distribution is not a verdict
either. What remains sayable comes from `channel_direction()`, which tallies your own
adjudicated channels and is labelled as such wherever it appears. Nothing is clipped to make
it quotable; clipping would hide the one fact that matters.

"The model cannot speak here, and here is why" is a complete answer and is marked as one — as
is "the model moved, but only these channels could have moved it", which `responsiveness()`
tells you.

## Data

Reserve Bank of Australia statistical tables (CC BY 4.0) in `data/raw/`, 211 parsed RBA
minutes, and the Caldara–Iacoviello geopolitical risk index. Attribute RBA material as
*Source: Reserve Bank of Australia [year]*. Everything the pipeline needs is vendored here —
no external paths, no sibling directories.

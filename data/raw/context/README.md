# Shock context documents — you download these yourself

**This folder is empty on purpose.**

Several of the sources below are free to read but **not free to redistribute**. The course
cannot ship them to you, so you download them into this folder yourself.

```bash
python src/context_docs.py
```

That lists what it found and how much usable text each file yielded. Run it before you
spend any API credit.

---

## Declare what you downloaded — `sources.json`

Copy `sources.json.template` to `sources.json` and fill in one record per file:

```json
[{"file": "wef-global-risks-report-2026.pdf",
  "organisation": "World Economic Forum",
  "title": "Global Risks Report 2026",
  "url": "https://www.weforum.org/publications/global-risks-report-2026/",
  "retrieved": "2026-08-12"}]
```

**The Shock run will not start without it.** The three-organisation rule is checked against
the `organisation` field you declare, not against filenames — three reports from one
publisher would otherwise pass on their names alone.

---

## What to download

**Get at least three, from at least three different organisations.** One report's risk
taxonomy is one organisation's house view, not a survey of geopolitical risk.

Beyond that, the choice is yours — and different teams will legitimately end up with
different sets. **The code does not look for particular filenames.** It reads every
supported file in this folder and uses what it finds. Name them however you like.

| Source | Where | Good for |
|---|---|---|
| **WEF Global Risks Report** | weforum.org/publications | The standard risk taxonomy; two- and ten-year horizons |
| **Eurasia Group Top Risks** | eurasiagroup.net/issues | Named, specific, near-term political scenarios |
| **IMF World Economic Outlook** | imf.org/en/publications/weo | Transmission channels and quantified downside scenarios |
| **Lowy Institute** | lowyinstitute.org | Australia-specific exposure and regional analysis |
| **Lloyd's systemic risk scenarios** | lloyds.com | Fully worked shock scenarios with economic impact estimates |
| **Cambridge Centre for Risk Studies** | jbs.cam.ac.uk/faculty-research/centres/risk | Stress-test scenario design methodology |
| **DFAT trade statistics** | dfat.gov.au/trade/resources | Which export exposures actually matter, by value |
| **RBA Statement on Monetary Policy** | rba.gov.au (CC BY 4.0) | How the Board frames international risk, and how it reacted to past episodes. **Useful, but it does NOT count towards your three independent organisations** - see below |

Full descriptions are in
`draft-assignment/shock-context-sources.md`, alongside the brief.

**Already supplied** in `data/raw/`: the Geopolitical Risk (GPR) index workbook
(`data_gpr_export.xls`) — this one *is* redistributable.

---

## Supported formats

`.pdf` · `.html` · `.htm` · `.txt` · `.md`

**Scanned PDFs will not work.** If a file has no text layer, `context_docs.py` reports it
as yielding almost no characters and skips it. Save the page as HTML instead, or find a
text-based version.

---

## Two rules about where scenarios come from

**Not from the model's general knowledge.** A scenario an LLM invents unprompted reflects
whatever was common in its training data, which is a poor sample of the risks facing
Australia now.

> **The RBA is a permitted source but not a counted one.** You may use RBA material to
> study how the Board reacted to comparable past episodes - that is what the reaction
> profiles need. It does not count towards the three independent organisations that
> ground the scenarios, because of the reason immediately below.

**Not from RBA documents.** The scenario is the *cause* and the RBA's reaction is the
*effect*. A scenario drawn from RBA minutes is a risk the Board has already identified and
already responded to, so its effect is already inside your model and it cannot test
anything.

Use RBA documents for the opposite purpose — to see how the Board reacted to comparable
past events.

---

## Cite what you used

Your report must list which documents you downloaded, their publication dates, and which
scenarios each one informed. `context_docs.relevant_passages()` prints a per-document
passage count each run, so you can see which of your sources actually fed each scenario —
and which contributed nothing.

**Do not commit these files to your repository.** `.gitignore` already excludes them.

# Data sources, licences and attribution

This repository redistributes third-party data. Read this before you publish, present or submit
anything built from it.

---

## RBA minutes — `data/raw/rba-minutes/`

**211 HTML documents.** Minutes of the Reserve Bank of Australia monetary policy meeting,
3 October 2006 to 16 June 2026.

| | |
|---|---|
| **Source** | Reserve Bank of Australia — https://www.rba.gov.au/monetary-policy/rba-board-minutes/ |
| **Licence** | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| **Terms** | https://www.rba.gov.au/copyright/ |
| **Collected** | 16 August 2026 |

RBA material "may be reproduced, published, communicated to the public and adapted provided that
the RBA is properly attributed."

**Required attribution — use this exact form in your report and on any chart built from it:**

> Source: Reserve Bank of Australia [year]

Attribution must not imply that the RBA endorses your work.

### Two exclusions that apply to you

- **ABS-sourced data appearing in RBA tables is third-party material.** It is not covered by the
  RBA's CC BY 4.0 licence. If you pull ABS series, check the ABS terms separately.
- **The Cash Rate carries additional conditions** (§4 of the RBA notice). Use with attribution is
  permitted, but you must not imply RBA endorsement.

---

## Geopolitical Risk index — `data/raw/geopolitical/data_gpr_export.xls`

| | |
|---|---|
| **Source** | Dario Caldara and Matteo Iacoviello — https://www.matteoiacoviello.com/gpr.htm |
| **Archive** | https://doi.org/10.3886/E154781V1 |
| **Downloaded** | 16 August 2026 |

**Required citation:**

> Caldara, Dario and Matteo Iacoviello, *Measuring Geopolitical Risk*, Board of Governors of the
> Federal Reserve System, August 2017.

Freely available academic data released by Federal Reserve Board economists. The index updates
monthly — the copy here is a snapshot.

---

## Data you collect yourself

**ASX 200 and macro indicators** are not redistributed here. You download them. Check the terms of
whichever provider you use — yfinance is a wrapper around Yahoo Finance, whose terms permit
personal and academic use but not redistribution.

---

## Reference implementation

Method and architecture adapted from
https://github.com/colinpriest/financial-market-volatility-regime-detection

---

## Summary

| Asset | Redistributable? | You must |
|---|---|---|
| RBA minutes | Yes, CC BY 4.0 | Attribute: *"Source: Reserve Bank of Australia [year]"* |
| GPR index | Yes, academic release | Cite Caldara & Iacoviello (2017) |
| ABS series | **No — check separately** | Read ABS terms before use |
| Market data you download | **Usually not** | Do not commit bulk price data to a public repository |

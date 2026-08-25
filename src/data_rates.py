"""
Interest rate structured data. PRE-BUILT - do not rewrite.

Builds two things:
  1. The DECISION series - every announced change in the cash rate target (RBA table A2).
     This is the ground truth for both targets.
  2. A daily rates panel - cash rate target, OIS 1/3/6m, government bond yields - which
     supplies the market-implied expectation benchmark and the yield-curve features.

WRITES data/processed/decisions.parquet
       data/processed/rates_daily.parquet
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

import config


# RBA tables mark the real header row with 'Series ID' in the current files and 'Mnemonic'
# in the older -hist ones. Both appear in the files this stage reads.
_HEADER_MARKERS = ("Series ID", "Mnemonic")

# Date formats seen across the tables: F1/G1/H5 use 30/06/1969, F2 uses 20-May-2013.
_DATE_FORMATS = ("%d/%m/%Y", "%d-%b-%Y")


def _parse_dates(s: pd.Series) -> pd.Series:
    """Try each known RBA date format; keep whichever parses the most rows."""
    best, best_n = None, -1
    for fmt in _DATE_FORMATS:
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if d.notna().sum() > best_n:
            best, best_n = d, d.notna().sum()
    return best


def _finalise(raw: pd.DataFrame, h: int, dates_are_datetime: bool,
              numeric: bool) -> pd.DataFrame:
    df = raw.iloc[h + 1:].copy()
    df.columns = ["date"] + [str(c).strip() for c in raw.iloc[h, 1:]]
    df["date"] = (pd.to_datetime(df["date"], errors="coerce") if dates_are_datetime
                  else _parse_dates(df["date"]))
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    # A2 stores pre-1998 changes as ranges ("-0.50 to -1.00"); coercing them here would
    # turn them into NaN before load_decisions gets a chance to recover them.
    return df.apply(pd.to_numeric, errors="coerce") if numeric else df


def _find_header(col0: pd.Series, name: str) -> int:
    stripped = col0.astype(str).str.strip()
    for marker in _HEADER_MARKERS:
        hit = stripped.index[stripped == marker]
        if len(hit):
            return hit[0]
    raise ValueError(f"{name}: no header row found (tried {_HEADER_MARKERS})")


def _read_rba_csv(name: str, numeric: bool = True) -> pd.DataFrame:
    """
    RBA CSV tables carry ~10 metadata rows before the real header.

    engine='python' is needed because the metadata rows are ragged - the notes block has
    fewer fields than the data rows, and the C parser rejects the file outright rather
    than padding.
    """
    path = config.DATA_RAW / name
    # The metadata block is ragged: line 1 is a bare title with one field, the data rows
    # have many. Left to itself pandas infers the column count from line 1 and then either
    # errors or (with on_bad_lines='skip') silently discards every real row including the
    # header. Count the widest row first and force that many columns.
    with open(path, encoding="utf-8-sig", newline="") as fh:
        ncols = max(len(row) for row in csv.reader(fh))
    raw = pd.read_csv(path, header=None, encoding="utf-8-sig",
                      names=range(ncols), engine="python")
    return _finalise(raw, _find_header(raw[0], name), False, numeric)


def _read_rba_excel(name: str, numeric: bool = True) -> pd.DataFrame:
    path = config.DATA_RAW / name
    raw = pd.read_excel(path, sheet_name="Data", header=None)
    return _finalise(raw, _find_header(raw[0], name), True, numeric)


def load_decisions() -> pd.DataFrame:
    """
    RBA table A2 - every announced change in the cash rate target, back to 1990.

    A2 records only the MEETINGS THAT MOVED. Holds are absent by construction, so the
    hold observations are created later by joining this onto the meeting calendar. Getting
    that join wrong is the single easiest way to fabricate a target variable, which is why
    it is done explicitly in stage5 rather than by a merge with a default.
    """
    df = _read_rba_excel("a02hist.xlsx", numeric=False)
    out = pd.DataFrame(index=df.index)
    # Older rows record a RANGE ("-0.50 to -1.00") because the target was a corridor
    # before 1998. Those all predate the corpus, but parse them rather than crash.
    chg = df["ARBAMPCCCR"].astype(str).str.strip()
    out["change_pct"] = pd.to_numeric(chg, errors="coerce")
    ranged = out["change_pct"].isna() & chg.str.contains(" to ", na=False)
    if ranged.any():
        out.loc[ranged, "change_pct"] = (
            chg[ranged].str.split(" to ").str[0].astype(float)
        )
    new = df["ARBAMPCNCRT"].astype(str).str.strip()
    out["new_target"] = pd.to_numeric(new, errors="coerce")
    ranged_t = out["new_target"].isna() & new.str.contains(" to ", na=False)
    if ranged_t.any():
        out.loc[ranged_t, "new_target"] = (
            new[ranged_t].str.split(" to ").str[0].astype(float)
        )
    out = out.dropna(subset=["change_pct"])
    out.index.name = "announced_date"
    return out


def load_rates_daily() -> pd.DataFrame:
    """
    Daily cash rate target and money-market yields.

    F1 splits across two files: f01dhist.xls covers 1976-2010, f01d.xlsx covers 2011 on.
    Neither alone spans the Oct-2006 corpus start, so they are concatenated. The overlap
    is checked rather than assumed.
    """
    hist = _read_rba_excel("f01dhist.xls")
    curr = _read_rba_excel("f01d.xlsx")
    overlap = hist.index.intersection(curr.index)
    if len(overlap):
        print(f"  F1 overlap {len(overlap)} days - current file wins")
        hist = hist.drop(index=overlap)
    f1 = pd.concat([hist, curr]).sort_index()
    f1 = f1.apply(pd.to_numeric, errors="coerce")

    keep = {
        "FIRMMCRTD": "cash_rate",
        "FIRMMOIS1D": "ois_1m",
        "FIRMMOIS3D": "ois_3m",
        "FIRMMOIS6D": "ois_6m",
        "FIRMMBAB90D": "bab_3m",
    }
    df = f1[[c for c in keep if c in f1.columns]].rename(columns=keep)

    # Government bond yields (F2), the long end of the curve. Same two-file split as F1:
    # f02dhist.xls covers 1995-2013, f2-data.csv covers 2013 on.
    f2_parts = []
    for reader, fname in ((_read_rba_excel, "f02dhist.xls"), (_read_rba_csv, "f2-data.csv")):
        try:
            f2_parts.append(reader(fname))
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: {fname} unavailable ({e})")
    if f2_parts:
        f2 = pd.concat(f2_parts).sort_index()
        f2 = f2[~f2.index.duplicated(keep="last")]
        f2keep = {"FCMYGBAG3D": "bond_3y", "FCMYGBAG10D": "bond_10y"}
        have = {k: v for k, v in f2keep.items() if k in f2.columns}
        if have:
            df = df.join(f2[list(have)].rename(columns=have), how="left")

    df = df.loc[config.CORPUS_START:]
    df["cash_rate"] = df["cash_rate"].ffill()

    # --- market-implied expectation of the next move -------------------------
    # The spread of a short money-market rate over the current cash rate target is, near
    # enough, the market's probability-weighted expectation of the next move. This is the
    # benchmark the text has to beat. All of these are market prices, so they are known
    # same-day and carry no publication lag.
    #
    # OIS IS THE TEXTBOOK CHOICE AND IT IS UNUSABLE HERE. FENICS stopped supplying the OIS
    # series to the RBA during 2022, so FIRMMOIS*D is empty from 2023 onward - which is
    # precisely the 2022-23 tightening and the 2024-26 easing, the most informative episode
    # in the sample. A benchmark that is missing over the period you most need it is not a
    # benchmark. It is retained as a cross-check on 2006-2022 only.
    #
    # 3-month bank bills (BAB) run the full sample with 0.8% missing, so bab_spread is the
    # primary market-expectation feature. It is a noisier proxy than OIS because it carries
    # bank credit risk as well as rate expectations - and that contamination is itself worth
    # reporting, since it blows out in exactly the funding-stress episodes (2008, 2020) where
    # the model most needs a clean read.
    for h in ("1m", "3m", "6m"):
        col = f"ois_{h}"
        if col in df.columns:
            df[f"ois_spread_{h}"] = df[col] - df["cash_rate"]
    if "bab_3m" in df.columns:
        df["bab_spread"] = df["bab_3m"] - df["cash_rate"]
    if "bond_3y" in df.columns and "bond_10y" in df.columns:
        df["slope_3y10y"] = df["bond_10y"] - df["bond_3y"]
    if "bond_3y" in df.columns:
        # 3y yield less cash: the market's view of the average policy rate over 3 years.
        # Full-sample coverage, and a cleaner rate-expectation read than BAB because it
        # carries sovereign rather than bank credit risk.
        df["slope_cash3y"] = df["bond_3y"] - df["cash_rate"]

    df.index.name = "date"
    return df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    dec = load_decisions()
    rates = load_rates_daily()

    in_corpus = dec.loc[config.CORPUS_START:]
    print(f"  decisions: {len(dec)} total since {dec.index.min().date()}, "
          f"{len(in_corpus)} in corpus window")
    print(f"    hikes {int((in_corpus['change_pct'] > 0).sum())}, "
          f"cuts {int((in_corpus['change_pct'] < 0).sum())}")
    print(f"    sizes: {in_corpus['change_pct'].value_counts().sort_index().to_dict()}")
    print(f"  rates_daily: {len(rates)} rows, "
          f"{rates.index.min().date()} -> {rates.index.max().date()}")
    miss = rates.isna().mean().mul(100).round(1)
    print(f"  missing %:\n{miss.to_string()}")

    dec.reset_index().to_parquet(config.DATA_PROCESSED / "decisions.parquet", index=False)
    rates.reset_index().to_parquet(config.DATA_PROCESSED / "rates_daily.parquet", index=False)
    return dec, rates


if __name__ == "__main__":
    run()

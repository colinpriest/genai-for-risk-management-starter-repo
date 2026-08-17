"""
Agreement statistics for section 8.2 Part B — the human side of the fourth uncertainty layer.

PRE-BUILT: ICC(2,1) with a bootstrap confidence interval, the human-range definition, and the
           loader that assembles your team's label files into a rater x document matrix.
YOURS:     the labelling, the adjudication, and the interpretation.

WHY THIS FILE EXISTS. The protocol asks you to report ICC(2,1) "with a 95% interval". An
earlier version of this repository gave you the point-estimate formula in a comment and no
interval, and did not ship a package that computes one. That is a requirement you cannot meet
with what you were given, so it is built here and tested in tests/test_agreement.py.

WHAT ICC(2,1) IS. Two-way random effects, absolute agreement, single rater. It asks: if I pick
one rater at random, how well does their score for a document predict the score another random
rater would give the same document? It is the right coefficient here because your raters are a
sample of possible careful readers, not the only readers who matter, and because you care about
absolute agreement (0.7 vs 0.4 is disagreement) rather than mere correlation (a rater who is
consistently 0.3 low is still disagreeing).

Conventional reading: below 0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, above 0.9 excellent.
Report the interval, not just the point estimate - with 20 documents and 3 raters the interval
is wide, and pretending otherwise is the mistake.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config

# Columns every labels_<initials>.csv must have. See the protocol, section 3.
# `set` says which half of the design a row belongs to: "development" (you may look at AI
# scores) or "confirmatory" (you may not, until the prompt is frozen). Without it, load_labels
# cannot separate the two and every reported statistic is contaminated with the documents you
# tuned the prompt on.
VALID_SETS = ("development", "confirmatory")
EXPECTED_PER_SUBSET = 10        # protocol section 1: 10 development + 10 confirmatory
LABEL_COLUMNS = ["meeting_date", "labeller", "set", "financial_conditions_concern",
                 "downside_risk_emphasis"]
CONSTRUCTS = ["financial_conditions_concern", "downside_risk_emphasis"]


def icc21(matrix: np.ndarray) -> float:
    """ICC(2,1) for a complete documents x raters matrix.

    matrix[i, j] is rater j's score for document i. No missing values - the protocol uses a
    complete design precisely so this holds.

        ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n)

    where n documents, k raters, MSR = mean square for rows (documents), MSC = mean square for
    columns (raters), MSE = residual mean square.

    Returns a float in [-1, 1]. Negative values mean between-rater variation exceeds
    between-document variation, i.e. no reliability at all; report them as such rather than
    clipping to zero.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D (documents x raters)")
    if np.isnan(x).any():
        raise ValueError("icc21() requires a complete matrix - no missing labels. The "
                         "protocol's design is complete by construction, so a NaN here "
                         "means a labeller skipped a document.")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError(f"need at least 2 documents and 2 raters, got {n} x {k}")

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if abs(denom) < 1e-12:
        return float("nan")
    return float((msr - mse) / denom)


def icc21_ci(matrix: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
             seed: int | None = None) -> dict:
    """ICC(2,1) with a bootstrap confidence interval, resampling DOCUMENTS.

    Documents are the sampling unit, not individual judgements: your raters are fixed, and it
    is the choice of documents that would differ if you ran the study again. Resampling
    judgements independently would break the rater structure the coefficient depends on and
    give an interval that is far too narrow.

    Percentile interval. With 20 documents it will be wide. That width is the finding - it
    tells you how much of your agreement estimate is sampling noise, which is exactly what a
    reader needs to know before believing that 0.62 differs from 0.48.
    """
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    rng = np.random.default_rng(config.SEED if seed is None else seed)

    point = icc21(x)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = x[idx]
        # A resample can duplicate one document n times, leaving no between-document
        # variance and an undefined coefficient. Skip those rather than propagating NaN.
        if np.allclose(resampled.mean(axis=1), resampled.mean()):
            continue
        try:
            v = icc21(resampled)
        except ValueError:
            continue
        if np.isfinite(v):
            draws.append(v)

    if len(draws) < n_boot * 0.5:
        return {"icc": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot_valid": len(draws), "n_documents": int(n), "n_raters": int(x.shape[1]),
                "warning": "too few valid bootstrap draws for a usable interval"}

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"icc": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot_valid": len(draws), "n_documents": int(n), "n_raters": int(x.shape[1]),
            "interpretation": _band(point)}


def _band(v: float) -> str:
    if not np.isfinite(v):
        return "undefined"
    if v < 0:
        return "no reliability (between-rater variation exceeds between-document variation)"
    return ("poor" if v < 0.5 else "moderate" if v < 0.75 else "good" if v < 0.9
            else "excellent")


def human_range(matrix: np.ndarray) -> dict:
    """Define "the human range" so the phrase means something.

    The protocol asks whether the AI "sits inside the human range". That is undefined until
    you say what the range is, so here it is, explicitly: the range of RATER MEAN scores.

    Each human has a mean score across the documents. Those means differ - some readers are
    systematically more willing to call a document concerned than others. The interval from
    the lowest to the highest rater mean is the span of defensible average readings among
    people who read the documents against your definitions.

    The AI's mean sits inside that span, or it does not. Inside means the model's overall
    calibration is within the disagreement your own team already exhibits, and you cannot call
    it wrong on this evidence. Outside means it is more extreme than any human reader, which
    is a stronger and more reportable finding.

    NOTE WHAT THIS DOES NOT SAY. It is a statement about average level, not about agreement
    document by document. A model can sit dead centre of the human range on average and still
    disagree with every rater on every individual document. Report the ICC as well - the range
    check is a second, cruder question, not a substitute.
    """
    x = np.asarray(matrix, dtype=float)
    means = x.mean(axis=0)
    return {"rater_means": [float(m) for m in means],
            "low": float(means.min()), "high": float(means.max()),
            "spread": float(means.max() - means.min()),
            "definition": "range of per-rater mean scores across the labelled documents"}


def ai_vs_humans(matrix: np.ndarray, ai_scores: np.ndarray) -> dict:
    """Compare the AI against the human raters, on the same documents, in that order.

    REPORT HUMAN-HUMAN FIRST. If your raters do not agree with each other, "the AI disagrees
    with a human" is not evidence the AI is wrong - you are comparing it against an unstable
    benchmark, and the honest conclusion is that the construct is harder to judge than your
    prompt assumes. This function returns both so they cannot be separated.
    """
    x = np.asarray(matrix, dtype=float)
    ai = np.asarray(ai_scores, dtype=float).reshape(-1)
    if len(ai) != x.shape[0]:
        raise ValueError(f"ai_scores has {len(ai)} rows, matrix has {x.shape[0]} documents")

    hr = human_range(x)
    per_rater = {}
    for j in range(x.shape[1]):
        pair = np.column_stack([x[:, j], ai])
        per_rater[f"rater_{j}"] = icc21(pair)

    return {
        "human_human": icc21_ci(x),
        "human_range": hr,
        "ai_mean": float(ai.mean()),
        "ai_inside_human_range": bool(hr["low"] <= ai.mean() <= hr["high"]),
        "icc_each_human_vs_ai": per_rater,
        "icc_pooled_humans_vs_ai": icc21(np.column_stack([x.mean(axis=1), ai])),
        "_reading_order": ("Read human_human FIRST. A low human-human ICC means the "
                           "human-AI numbers below cannot be interpreted as the AI being "
                           "wrong - the benchmark itself is unstable."),
    }


def load_labels(construct: str, directory=None,
                subset: str = "confirmatory") -> pd.DataFrame:
    """Assemble labels_<initials>.csv files into a documents x raters frame.

    subset defaults to "confirmatory" BECAUSE THAT IS THE ONE YOU REPORT. The development
    documents are the ones you tuned the prompt against; computing your headline agreement
    statistic on them measures how well you fitted your own labels. Pass
    subset="development" when you are iterating, and subset=None only when you genuinely
    want both halves.

    Raises if the design is incomplete, because ICC(2,1) needs a complete matrix and a
    silently dropped document is a silently changed sample.
    """
    if subset is not None and subset not in VALID_SETS:
        raise ValueError(f"subset must be one of {VALID_SETS} or None, got {subset!r}")
    d = (config.DATA_PROCESSED / "agreement") if directory is None else directory
    files = sorted(d.glob("labels_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no labels_*.csv in {d}. Each team member commits one - see the protocol, "
            f"section 3. Note that data/processed/ is gitignored EXCEPT this directory.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{f.name} is missing required columns: {missing}. The 'set' column must say "
                f"'development' or 'confirmatory' for every row - see the protocol, section 2.")
        bad = set(df["set"].dropna().unique()) - set(VALID_SETS)
        if bad:
            raise ValueError(f"{f.name} has invalid set values {sorted(bad)}; "
                             f"expected {VALID_SETS}")
        if subset is not None:
            df = df[df["set"] == subset]
        frames.append(df[["meeting_date", "labeller", construct]])

    long = pd.concat(frames, ignore_index=True)
    if long.empty:
        raise ValueError(
            f"no rows with set == {subset!r}. Every labelling sheet needs both halves of the "
            f"design - 10 development and 10 confirmatory documents.")
    wide = long.pivot(index="meeting_date", columns="labeller", values=construct).sort_index()

    if wide.isna().any().any():
        gaps = {c: wide.index[wide[c].isna()].tolist() for c in wide.columns
                if wide[c].isna().any()}
        raise ValueError(
            f"incomplete design for '{construct}' - every rater must label every document. "
            f"Missing: { {k: v[:3] for k, v in gaps.items()} }")

    vals = wide.to_numpy(dtype=float)
    if vals.min() < 0 or vals.max() > 1:
        raise ValueError(f"'{construct}' has labels outside [0, 1]: "
                         f"min {vals.min()}, max {vals.max()}")
    if subset is not None and len(wide) != EXPECTED_PER_SUBSET:
        raise ValueError(
            f"the {subset} set has {len(wide)} documents, expected {EXPECTED_PER_SUBSET}. "
            f"The protocol specifies {EXPECTED_PER_SUBSET} development and "
            f"{EXPECTED_PER_SUBSET} confirmatory documents.")
    return wide

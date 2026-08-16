"""Perturbation experiments. SUPPLIED - DO NOT MODIFY.

Change specific wording in a document, re-score it, and measure how far the score moves.

NOTE ON THE EDIT LIST
    Every pattern below was verified to occur in the RBA minutes corpus before being
    included. A perturbation pattern that never matches produces no evidence at all, and
    does so silently. Run check_edit_coverage() on your own corpus before trusting a sweep.
"""
from __future__ import annotations
from typing import Callable, Iterable
import re

# (label, pattern, replacement)
DEFAULT_EDITS = [
    # --- signal edits: these change the monetary policy meaning ---------------
    ("inflation_soften", r"inflation (remained|was) (high|too high)", r"inflation \1 low"),
    ("inflation_harden", r"inflation remained low", "inflation remained too high"),
    ("inflation_expect_down", r"inflation was expected to increase",
     "inflation was expected to decline"),
    ("inflation_expect_up", r"inflation was expected to (decline|moderate|fall)",
     "inflation was expected to increase"),
    ("labour_loosen", r"labour market (remained|was) tight", r"labour market \1 loose"),
    ("labour_tighten", r"labour market conditions had (eased|softened)",
     "labour market conditions had tightened"),
    ("outlook_soften", r"upside risks", "downside risks"),
    ("outlook_harden", r"downside risks", "upside risks"),
    ("growth_soften", r"growth (had|has) (picked up|strengthened)", r"growth \1 slowed"),
    ("conditions_soften", r"conditions had continued to improve",
     "conditions had continued to deteriorate"),

    # --- control edits: meaning-preserving ------------------------------------
    # A well-behaved scorer should NOT move on these. If it does, it is responding to
    # surface wording rather than to economic content.
    ("control_verb", r"[Mm]embers noted", "members observed"),
    ("control_discussed", r"[Mm]embers discussed", "members considered"),
    ("control_considered", r"[Mm]embers considered", "members reviewed"),
]


def check_edit_coverage(texts: Iterable[str]) -> dict:
    """How many documents does each pattern actually match? Run this first."""
    texts = list(texts)
    out = {}
    for label, pat, _rep in DEFAULT_EDITS:
        n = sum(bool(re.search(pat, t)) for t in texts)
        out[label] = {"docs_matched": n, "share": n / len(texts) if texts else 0.0}
    return out


def apply_edit(text: str, pattern: str, replacement: str) -> tuple[str, int]:
    return re.subn(pattern, replacement, text)


def perturbation_sweep(score_fn: Callable[[str], float],
                       text: str,
                       edits: Iterable[tuple[str, str, str]] = DEFAULT_EDITS) -> list[dict]:
    """Score the original, then each perturbed version.

    score_fn: your stage 3 scorer, str -> float
    Returns one dict per edit, including edits that did not apply (n_applied == 0).
    """
    base = score_fn(text)
    out = []
    for label, pat, rep in edits:
        new_text, n = apply_edit(text, pat, rep)
        if n == 0:
            out.append({"edit": label, "n_applied": 0, "base": base,
                        "perturbed": None, "delta": None,
                        "is_control": label.startswith("control_"),
                        "note": "pattern not present in this document"})
            continue
        s = score_fn(new_text)
        out.append({"edit": label, "n_applied": n, "base": base,
                    "perturbed": s, "delta": s - base,
                    "is_control": label.startswith("control_")})
    return out


def summarise(results: list[dict]) -> dict:
    """Signal-vs-control summary. A healthy pipeline moves on signal edits, not controls."""
    applied = [r for r in results if r["n_applied"] > 0 and r["delta"] is not None]
    sig = [abs(r["delta"]) for r in applied if not r.get("is_control")]
    ctl = [abs(r["delta"]) for r in applied if r.get("is_control")]

    m_sig = sum(sig) / len(sig) if sig else None
    m_ctl = sum(ctl) / len(ctl) if ctl else None

    if m_sig is None or m_ctl is None:
        ratio = None
    elif m_ctl == 0 and m_sig > 0:
        ratio = float("inf")
    elif m_ctl == 0:
        ratio = None
    else:
        ratio = m_sig / m_ctl

    return {
        "n_edits_applied": len(applied),
        "n_signal_applied": len(sig),
        "n_control_applied": len(ctl),
        "mean_abs_delta_signal": m_sig,
        "mean_abs_delta_control": m_ctl,
        "signal_to_control_ratio": ratio,
        "note": (
            "inf = controls did not move at all, which is the ideal result. "
            "Ratio near 1 = the scorer responds as much to meaningless edits as to "
            "meaningful ones. None with both means present = nothing moved."
        ),
    }

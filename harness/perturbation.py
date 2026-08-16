"""Perturbation experiments. SUPPLIED - DO NOT MODIFY.

Change specific wording in a document, re-score it, and measure how far the score moves.
"""
from __future__ import annotations
from typing import Callable, Iterable
import re

# (label, pattern, replacement) - hawkish <-> dovish flips in RBA language
DEFAULT_EDITS = [
    ("inflation_down", r"inflation remains elevated", "inflation has moderated"),
    ("inflation_up", r"inflation has moderated", "inflation remains elevated"),
    ("labour_loosen", r"labour market remains tight", "labour market has eased"),
    ("labour_tighten", r"labour market has eased", "labour market remains tight"),
    ("growth_down", r"growth (?:is|remains) (?:solid|robust)", "growth has slowed"),
    ("outlook_soften", r"upside risks", "downside risks"),
    ("outlook_harden", r"downside risks", "upside risks"),
    # CONTROL: should not move a well-behaved sentiment score
    ("control_date", r"\bmembers noted\b", "members observed"),
    ("control_filler", r"\bthe Board\b", "the Board"),
]


def apply_edit(text: str, pattern: str, replacement: str) -> tuple[str, int]:
    new, n = re.subn(pattern, replacement, text, flags=re.I)
    return new, n


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
        ratio = None                      # nothing to compare
    elif m_ctl == 0 and m_sig > 0:
        ratio = float("inf")              # perfect separation - the ideal result
    elif m_ctl == 0:
        ratio = None                      # nothing moved at all; check your scorer
    else:
        ratio = m_sig / m_ctl

    return {
        "n_edits_applied": len(applied),
        "mean_abs_delta_signal": m_sig,
        "mean_abs_delta_control": m_ctl,
        "signal_to_control_ratio": ratio,
        "note": (
            "inf = controls did not move at all, which is the ideal result. "
            "Ratio near 1 = your scorer responds as much to meaningless edits as to "
            "meaningful ones. None with both means present = nothing moved; check the scorer."
        ),
    }

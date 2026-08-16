"""Faithfulness test. SUPPLIED - DO NOT MODIFY.

Does the reason the model GIVES actually drive the answer it PRODUCES?

Method
    1. Ask the model to explain its score, and to name the phrases that drove it.
    2. Perturb THOSE phrases. A faithful explanation means the score moves a lot.
    3. Perturb other, unnamed phrases. A faithful explanation means the score moves little.
    4. faithfulness_gap = movement on named phrases - movement on unnamed phrases.

    Gap near zero, or negative, means the stated reason does not describe the behaviour.
    That is a finding. Report it.
"""
from __future__ import annotations
from typing import Callable
import re, random

random.seed(20260817)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]


def _neutralise(sentence: str) -> str:
    """Replace a sentence with a content-free equivalent of similar length."""
    return "The Board discussed this matter at the meeting."


def faithfulness_test(score_fn: Callable[[str], float],
                      explain_fn: Callable[[str], list[str]],
                      text: str,
                      n_controls: int = 5) -> dict:
    """
    score_fn:   your stage 3 scorer, str -> float
    explain_fn: str -> list of phrases the model says drove its score
    """
    base = score_fn(text)
    named = explain_fn(text)
    sents = _sentences(text)

    named_hits, named_deltas = [], []
    for phrase in named:
        for s in sents:
            if phrase.lower()[:40] in s.lower():
                perturbed = text.replace(s, _neutralise(s))
                named_deltas.append(abs(score_fn(perturbed) - base))
                named_hits.append(s)
                break

    unnamed = [s for s in sents if s not in named_hits]
    random.shuffle(unnamed)
    control_deltas = [abs(score_fn(text.replace(s, _neutralise(s))) - base)
                      for s in unnamed[:n_controls]]

    m_named = sum(named_deltas) / len(named_deltas) if named_deltas else None
    m_ctrl = sum(control_deltas) / len(control_deltas) if control_deltas else None

    return {
        "base_score": base,
        "phrases_named_by_model": named,
        "n_named_located": len(named_deltas),
        "mean_delta_named": m_named,
        "mean_delta_unnamed": m_ctrl,
        "faithfulness_gap": (m_named - m_ctrl) if (m_named is not None and m_ctrl is not None) else None,
        "interpretation": (
            "Gap much greater than 0: the stated reason drives the score (faithful). "
            "Gap near 0 or negative: the explanation does not describe the behaviour. "
            "Report whichever you find."
        ),
    }

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
    """Replace a sentence with a content-free equivalent of SIMILAR LENGTH.

    Length matters: a long, information-dense sentence replaced by a short stub changes the
    document's size as well as its content, and the score can move for that reason alone.
    """
    filler = ("The Board discussed this matter at the meeting. "
              "Members considered the information presented. "
              "The discussion covered the points raised by staff. "
              "Members noted the material provided for the meeting. ")
    n = max(1, len(sentence))
    out = (filler * (n // len(filler) + 1))[:n]
    return out.rstrip() + "."


def _sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _gap_ci(named: list[float], ctrl: list[float]) -> list[float] | None:
    """Rough 95% interval for the gap (Welch), so a gap is not read as real when it is noise."""
    if len(named) < 2 or len(ctrl) < 2:
        return None
    mn, mc = sum(named) / len(named), sum(ctrl) / len(ctrl)
    sn, sc = _sd(named), _sd(ctrl)
    se = ((sn ** 2) / len(named) + (sc ** 2) / len(ctrl)) ** 0.5
    return [round((mn - mc) - 1.96 * se, 4), round((mn - mc) + 1.96 * se, 4)]


def _matched_controls(sents: list[str], named_hits: list[str],
                      unnamed: list[str], n: int) -> list[str]:
    """Pick unnamed sentences matched to the named ones on length and position."""
    if not named_hits or not unnamed:
        random.shuffle(unnamed)
        return unnamed[:n]
    pos = {s: i for i, s in enumerate(sents)}
    picked, pool = [], list(unnamed)
    for target in named_hits[:n]:
        tl, tp = len(target), pos.get(target, 0)
        best = min(pool, key=lambda s: abs(len(s) - tl) / max(tl, 1)
                   + abs(pos.get(s, 0) - tp) / max(len(sents), 1))
        picked.append(best)
        pool.remove(best)
        if not pool:
            break
    return picked


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

    # MATCHED controls. Deleting a sentence changes length and context as well as content,
    # so an unmatched control confounds "this sentence mattered" with "the document got
    # shorter". Each control is chosen to be the unnamed sentence closest in LENGTH and
    # POSITION to a named one, so the two sets differ in what was said and not in how much.
    unnamed = [s for s in sents if s not in named_hits]
    controls = _matched_controls(sents, named_hits, unnamed, n_controls)
    control_deltas = [abs(score_fn(text.replace(s, _neutralise(s))) - base) for s in controls]

    m_named = sum(named_deltas) / len(named_deltas) if named_deltas else None
    m_ctrl = sum(control_deltas) / len(control_deltas) if control_deltas else None

    return {
        "base_score": base,
        "phrases_named_by_model": named,
        "n_named_located": len(named_deltas),
        "n_named_not_located": len(named) - len(named_deltas),
        "mean_delta_named": m_named,
        "mean_delta_unnamed": m_ctrl,
        "sd_delta_named": _sd(named_deltas),
        "sd_delta_unnamed": _sd(control_deltas),
        "faithfulness_gap": ((m_named - m_ctrl)
                             if (m_named is not None and m_ctrl is not None) else None),
        "gap_95ci": _gap_ci(named_deltas, control_deltas),
        "controls_matched_on": "sentence length and position",
        "interpretation": (
            "Gap much greater than 0: the stated reason drives the score (faithful). "
            "Gap near 0 or negative: the explanation does not describe the behaviour. "
            "Report whichever you find."
        ),
    }

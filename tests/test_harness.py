"""The harness is supplied and students depend on its output shape.

These tests exist because a refactor of faithfulness_test() silently dropped the
`faithfulness_gap` key. Nothing failed at import; the pipeline ran for twenty minutes and
then died on a KeyError in the caller. A contract that is only checked by its consumer is
checked too late.
"""
from __future__ import annotations
import pytest

from harness.faithfulness import faithfulness_test
from harness.perturbation import perturbation_sweep, summarise, DEFAULT_EDITS

REQUIRED_FAITHFULNESS_KEYS = {
    "base_score", "phrases_named_by_model", "n_named_located",
    "mean_delta_named", "mean_delta_unnamed", "faithfulness_gap",
}
REQUIRED_SWEEP_KEYS = {"mean_abs_delta_signal", "mean_abs_delta_control", "n_edits_applied"}

TEXT = ("Members noted that inflation remained high and that the labour market remained "
        "tight. Members observed that financial conditions had tightened further over the "
        "period. The Board considered the outlook and judged that downside risks had "
        "increased. Members discussed the implications for the cash rate at length.")


def _fake_scorer(calls):
    """Deterministic scorer: length-sensitive, so perturbation actually moves it."""
    def score(text: str) -> float:
        calls.append(text)
        return min(1.0, len(text) / 1000.0)
    return score


def test_faithfulness_returns_every_documented_key():
    calls = []
    out = faithfulness_test(_fake_scorer(calls), lambda t: ["inflation remained high"], TEXT,
                            n_controls=2)
    missing = REQUIRED_FAITHFULNESS_KEYS - set(out)
    assert not missing, f"faithfulness_test dropped documented keys: {sorted(missing)}"


def test_faithfulness_gap_is_named_minus_unnamed():
    calls = []
    out = faithfulness_test(_fake_scorer(calls), lambda t: ["inflation remained high"], TEXT,
                            n_controls=2)
    if out["mean_delta_named"] is None or out["mean_delta_unnamed"] is None:
        pytest.skip("no phrases located in this fixture")
    assert out["faithfulness_gap"] == pytest.approx(
        out["mean_delta_named"] - out["mean_delta_unnamed"])


def test_neutralise_preserves_length():
    from harness.faithfulness import _neutralise
    for s in ["Short one here.", "A considerably longer sentence " * 6]:
        assert abs(len(_neutralise(s)) - len(s)) <= 2, "control edit changes document length"


def test_perturbation_sweep_shape():
    calls = []
    res = perturbation_sweep(_fake_scorer(calls), TEXT, DEFAULT_EDITS)
    s = summarise(res)
    missing = REQUIRED_SWEEP_KEYS - set(s)
    assert not missing, f"summarise() dropped documented keys: {sorted(missing)}"

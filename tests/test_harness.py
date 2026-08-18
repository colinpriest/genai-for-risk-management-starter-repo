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


# =============================================================================================
# PIPELINE WIRING
# =============================================================================================
# `run_pipeline.py --all` dispatches by looking for run() or build() on each module. A phase
# whose module exposes neither is silently unrunnable: --all appears to succeed and simply
# skips it. src/scenarios.py was in exactly that state - it had only a __main__ block.

def test_every_phase_has_an_entry_point():
    import ast, re, io, pathlib
    src = io.open("run_pipeline.py", encoding="utf-8").read()
    mods = sorted(set(re.findall(r'"(src\.[a-z0-9_]+)"', src)))
    assert mods, "no phase modules found in run_pipeline.py - has PHASES been renamed?"
    missing = []
    for m in mods:
        path = pathlib.Path(m.replace(".", "/") + ".py")
        assert path.exists(), f"run_pipeline.py names {m}, which does not exist"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if not ({"run", "build"} & fns):
            missing.append(m)
    assert not missing, (
        f"these phase modules expose neither run() nor build(), so `--all` cannot run them: "
        f"{missing}")


def test_scenario_output_filename_is_consistent():
    """One filename for the scenario output, everywhere.

    personas.py read scenarios.json while the tests and the marker used
    scenarios_final.json, so a student could produce scenarios and have the persona module
    silently find nothing.
    """
    import io, pathlib
    offenders = []
    for p in pathlib.Path("src").glob("*.py"):
        txt = io.open(p, encoding="utf-8").read()
        if '"scenarios.json"' in txt or "'scenarios.json'" in txt:
            offenders.append(p.name)
    assert not offenders, (
        f"{offenders} refer to scenarios.json; the agreed filename is scenarios_final.json")


def test_rolling_origin_eval_accepts_custom_arms():
    """rolling_origin_eval must work with ANY arm names, not just the default three.

    THE BUG THIS CATCHES. The summary block counted scored folds with
    folds["text_and_macro"], an arm name that only exists in the default set. ablation_oos()
    supplies its own arms ("full", "without_vix", ...), so the KeyError fired AFTER every
    refit had completed - a forty-minute run destroyed by a dictionary lookup on the last
    line. Fitting is slow enough that this has to be caught by a test, not by a run.

    Deliberately tiny: two regimes, two folds, one exogenous column, the smallest search
    budget that fits. This tests the plumbing, not the modelling.
    """
    import numpy as np
    import pandas as pd
    from src import stage4_regime_model as s4

    rng = np.random.default_rng(0)
    n = 400
    # Two-state volatility series with an exogenous column that tracks the state, so the
    # optimiser has something to find and converges quickly.
    state = (np.arange(n) // 100) % 2
    x = state + 0.30 * rng.standard_normal(n)
    ret = (0.004 + 0.012 * state) * rng.standard_normal(n)
    df = pd.DataFrame({"ret": ret,
                       "rv_21": np.exp(-3.4 + 0.9 * state + 0.05 * rng.standard_normal(n)),
                       "x": x})

    budget = dict(em_starts=[dict(em_iter=5, search_reps=0, maxiter=60)], n_perturbed=0)
    out = s4.rolling_origin_eval(
        df, ["x"], k=2, n_folds=2, arms={"full": ["x"], "no_x": None},
        require_all_arms=False, fit_kwargs=budget)

    assert out["reference_arm"] == "full"
    assert set(out["mean_score"]) == {"full", "no_x"}
    assert set(out["per_fold"]) >= {"full", "no_x", "persistence"}
    assert isinstance(out["n_folds_scored"], int)
    # Gains are only defined for arms that exist; the default gain labels must not appear.
    assert "text_gain_conditional_on_macro" not in out["gains"]

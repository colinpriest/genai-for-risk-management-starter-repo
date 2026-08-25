"""
Pins the lead/lag sign convention with synthetic series whose answer is known.

An earlier version of `lead_lag()` reported a feature that genuinely LED policy as
"lags policy", and every causal reading in the worked solution inherited the error. These
tests exist so that cannot recur silently.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import causal_tests as ct  # noqa: E402

N = 300
IDX = pd.date_range("2000-01-01", periods=N, freq="ME")


def _series(lead: int) -> pd.DataFrame:
    """Feature leads the target by `lead` periods (negative = feature follows)."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=N)
    t = np.roll(f, lead)
    t[:abs(lead)] = 0.0
    return pd.DataFrame({"f": f, "y_cycle": t}, index=IDX)


@pytest.mark.parametrize("lead", [1, 2, 4])
def test_leading_feature_reported_as_leading(lead):
    r = ct.lead_lag(_series(lead), "f")
    assert r["peak_lag"] == lead
    assert r["reads_as"] == "leads policy"


@pytest.mark.parametrize("lag", [1, 3])
def test_lagging_feature_reported_as_lagging(lag):
    r = ct.lead_lag(_series(-lag), "f")
    assert r["peak_lag"] == -lag
    assert r["reads_as"] == "lags policy"


def test_contemporaneous_feature_reported_as_moving_with():
    r = ct.lead_lag(_series(0), "f")
    assert r["peak_lag"] == 0
    assert r["reads_as"] == "moves with policy"
